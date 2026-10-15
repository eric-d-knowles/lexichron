"""lexichron tune: a grid on sampled years, scored, with a stand-in trainer."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                      # noqa: E402
from ngramprep.common_db import api as dbapi            # noqa: E402
from ngramprep.ngram_acquire.progress import read_progress   # noqa: E402
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths   # noqa: E402
from lexichron.stages import tune                      # noqa: E402
from lexichron.training.trainer import Vectors         # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def fake_train(corpus_file, cell, seed, workers):
    """Deterministic stand-in for gensim: vocabulary from the corpus file in
    frequency order (min_count applied), vectors from a seeded generator so
    two seeds disagree a little and two halves disagree a little more."""
    counts = {}
    with open(corpus_file, encoding="utf-8") as fh:
        for line in fh:
            for tok in line.split():
                counts[tok] = counts.get(tok, 0) + 1
    words = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= cell.min_count]
    base = np.random.default_rng(1234)                        # shared structure
    structure = {w: base.normal(size=cell.vector_size) for w in words}
    rng = np.random.default_rng(seed + 17 * cell.window)
    m = np.array([structure[w] + 0.3 * rng.normal(size=cell.vector_size) for w in words], dtype=np.float32)
    return Vectors(words=words, matrix=m, counts=np.array([counts[w] for w in words]))


def _datasets(tmp_path, monkeypatch=None):
    """Tiny similarity and analogy files, named as the image's copies are;
    with monkeypatch they become the bundled datasets via LEXICHRON_DATA_DIR."""
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    (d / "simlex999.txt").write_text("word1\tword2\tPOS\tSimLex999\nw01\tw02\tN\t8.0\nw03\tw04\tN\t2.0\nw05\tw06\tN\t5.0\nw07\tw08\tN\t1.0\n")
    (d / "wordsim353.tsv").write_text("# Word 1\tWord 2\tHuman (mean)\nw01\tw02\t9\nw03\tw04\t3\nw05\tw06\t6\n")
    (d / "questions-words.txt").write_text(": test\nw01 w02 w03 w04\nw05 w06 w07 w08\n")
    if monkeypatch is not None:
        monkeypatch.setenv("LEXICHRON_DATA_DIR", str(d))
    return d


def _corpus(stub, years):
    paths = derive_paths(str(stub), 5, "20200217", "eng")
    db = fake_rocks.open(str(paths["pivoted_db"]), mode="rw", create_if_missing=True)
    words = [f"w{i:02d}" for i in range(40)]
    rng = np.random.default_rng(0)
    with db.write_batch() as wb:
        for y in years:
            for i in range(300):
                toks = list(rng.choice(words, size=5))
                if i % 3 == 0:
                    toks[rng.integers(5)] = "<UNK>"
                wb.put(struct.pack(">I", y) + " ".join(toks).encode(), struct.pack("<QQQ", y, int(rng.integers(1, 200)), 3))
    uni = derive_paths(str(stub), 1, "20200217", "eng")
    uni["dst_db"].mkdir(parents=True, exist_ok=True)
    uni["whitelist"].write_text("\n".join(words[:30]) + "\n")
    return paths


def test_sampled_years_are_adjacent_pairs_over_the_range():
    assert tune.sample_years_in((1968, 2019), 4) == [1968, 1969, 1985, 1986, 2001, 2002, 2018, 2019]
    assert tune.sample_years_in((2000, 2001), 3) == [2000, 2001]
    assert tune.sample_years_in((1990, 1990), 2) == [1990]
    assert tune.sample_years_in((1900, 2000), 1) == [1949, 1950]


def test_tune_trains_scores_and_summarizes(rocks, tmp_path):
    stub = tmp_path / "corpora"
    paths = _corpus(stub, [1990, 1991, 2010, 2011])
    progress = tmp_path / "p.json"
    out = tune.tune_models(5, "20200217", "eng", str(stub), years=(1990, 2011), year_pairs=2,
                           vector_size=[8, 16], window=[2], min_count=[2], epochs=[1], approach=["skip-gram"],
                           weight_by=["freq"], seeds=2, split_half=True, name="t1", mode="restart",
                           workers=1, workers_per_model=1, probes_per_decile=3,
                           similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"),
                           analogy_dataset=str(tmp_path / "data" / "questions-words.txt"),
                           progress_path=str(progress), train_fn=fake_train)
    out = Path(out)
    assert out == paths["pivoted_db"].parent / "tuning" / "t1"
    # 2 cells x 4 years x (2 seeds + 2 halves) = 32 models
    models = sorted(p.name for p in (out / "models").glob("*.npz"))
    assert len(models) == 32 and "y1990_vs8_w2_mc2_e1_sg1_wbfreq_s1.npz" in models and "y2011_vs16_w2_mc2_e1_sg1_wbfreq_hb.npz" in models
    results = (out / "results.tsv").read_text().splitlines()
    assert results[0].split("\t")[:3] == ["cell", "year", "vector_size"] and len(results) == 1 + 8
    header = results[0].split("\t")
    row = dict(zip(header, results[1].split("\t")))
    assert row["year"] == "1990" and 0 < float(row["seed_overlap"]) <= 1 and 0 < float(row["split_half_overlap"]) <= 1
    assert float(row["coverage"]) > 0.5 and int(row["vocab_size"]) > 20
    assert "simlex999_rho" in header and "analogy_accuracy" in header
    pairs = (out / "pairs.tsv").read_text().splitlines()
    kinds = [l.split("\t")[3] for l in pairs[1:]]
    assert kinds.count("adjacent") == 4 and kinds.count("distant") == 2       # per cell: 2 adjacent + 1 distant
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["cells"]) == 2 and summary["best_cell"] in {c["cell"] for c in summary["cells"]}
    assert "most stable" in summary["best_rule"]
    for c in summary["cells"]:
        assert c["years"] == 4 and c["adjacent_overlap"] == c["adjacent_overlap"]
    assert (out / "summary.tsv").read_text().startswith("cell\t")
    assert json.loads((out / "search.json").read_text())["sample_years"] == [1990, 1991, 2010, 2011]
    assert not list((out / "corpus").glob("*.txt"))                              # corpus files cleaned up
    doc = read_progress(progress)
    assert doc["state"] == "done" and doc["files_total"] == 32 and doc["files_done"] == 32
    assert doc["unit_label"] == "models" and doc["counters"]["models"] == 32 and doc["counters"]["cells"] == 2
    assert doc["inputs"]["vocabulary"].endswith("whitelist.txt")

    # resume: nothing retrained, the tables rebuilt
    (out / "results.tsv").unlink()
    tune.tune_models(5, "20200217", "eng", str(stub), years=(1990, 2011), year_pairs=2,
                     vector_size=[8, 16], window=[2], min_count=[2], epochs=[1], seeds=2, name="t1",
                     workers=1, probes_per_decile=3, progress_path=str(tmp_path / "p2.json"), train_fn=fake_train,
                     similarity_dataset=str(tmp_path / "data" / "simlex999.txt"))
    doc2 = read_progress(tmp_path / "p2.json")
    assert doc2["files_skipped"] == 32 and doc2["files_done"] == 0 and (out / "results.tsv").exists()
    # growing the grid trains only the new cell
    tune.tune_models(5, "20200217", "eng", str(stub), years=(1990, 2011), year_pairs=2,
                     vector_size=[8, 16], window=[2, 5], min_count=[2], epochs=[1], seeds=2, name="t1",
                     workers=1, probes_per_decile=3, progress_path=str(tmp_path / "p3.json"), train_fn=fake_train,
                     similarity_dataset=str(tmp_path / "data" / "simlex999.txt"))
    doc3 = read_progress(tmp_path / "p3.json")
    assert doc3["files_total"] == 64 and doc3["files_skipped"] == 32 and doc3["files_done"] == 32
    assert len(json.loads((out / "summary.json").read_text())["cells"]) == 4


def test_tune_refuses_without_the_pivoted_database(rocks, tmp_path):
    with pytest.raises(FileNotFoundError):
        tune.tune_models(5, "20200217", "eng", str(tmp_path / "none"), train_fn=fake_train)


def test_tune_refuses_without_a_similarity_dataset(rocks, tmp_path, monkeypatch):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    monkeypatch.setenv("LEXICHRON_DATA_DIR", str(tmp_path / "nowhere"))
    import lexichron.training.benchmarks as bm
    monkeypatch.setattr(bm, "bundled_dataset", lambda name: None)
    with pytest.raises(FileNotFoundError, match="similarity dataset"):
        tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990, 1991], train_fn=fake_train, workers=1)


def test_bundled_datasets_come_from_the_image_directory(tmp_path, monkeypatch):
    from lexichron.training.benchmarks import bundled_dataset
    _datasets(tmp_path, monkeypatch)
    assert bundled_dataset("simlex999.txt") == str(tmp_path / "data" / "simlex999.txt")
    assert bundled_dataset("nothing.txt") is None or not bundled_dataset("nothing.txt").startswith(str(tmp_path))


def test_check_datasets_script_verifies_shape_and_records_checksums(tmp_path):
    import subprocess
    d = _datasets(tmp_path)
    script = Path(__file__).parent.parent / "scripts" / "check_datasets.py"
    r = subprocess.run([sys.executable, str(script), str(d)], capture_output=True, text=True)
    assert r.returncode == 1 and "only 4 records" in r.stderr                # the tiny files are too small
    assert (d / "SOURCES.md").read_text().count("sha256") == 3


def test_cli_runs_tune_from_a_settings_file(rocks, tmp_path, monkeypatch):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    _datasets(tmp_path, monkeypatch)                                           # the image's copies
    import lexichron.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "gensim_train", fake_train)
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {stub}\n"
                    "tune:\n  years: [1990, 1991]\n  sample_years: [1990, 1991]\n  vector_size: [8]\n  window: 2\n"
                    "  min_count: 2\n  epochs: 1\n  seeds: 2\n  split_half: false\n  name: cli\n  workers: 1\n"
                    "  probes_per_decile: 2\n")
    from lexichron.cli import main
    assert main(["tune", str(proj)]) == 0
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "cli"
    assert (out / "summary.tsv").exists() and (out / "lexichron.json").exists()     # provenance too
    rec = json.loads((out / "lexichron.json").read_text())
    assert rec["stage"] == "tune" and rec["settings"]["vector_size"] == [8]
    assert rec["inputs"]["simlex_dataset"]["path"].endswith("simlex999.txt") and len(rec["inputs"]["simlex_dataset"]["sha256"]) == 64
    assert "analogy_dataset" in rec["inputs"]

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
    # a valence lexicon over the corpus words (VADER's layout) and pole lists
    (d / "vader_lexicon.txt").write_text("".join(f"w{i:02d}\t{(i - 25) / 5:.1f}\t0.5\t[0]\n" for i in range(10, 40)))
    (d / "poles.txt").write_text("w00 w01 w02 w03 w04\nw05 w06 w07 w08 w09\n")
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
                           axis_lexicon=str(tmp_path / "data" / "vader_lexicon.txt"),
                           axis_poles=str(tmp_path / "data" / "poles.txt"),
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
    assert "simlex_rho" in header and "analogy_accuracy" in header           # a user's simlex999.txt is SimLex
    # the valence-axis yardstick: validity, coverage of the rated words, and reliability of the projections
    assert -1 <= float(row["axis_rho"]) <= 1 and float(row["axis_coverage"]) > 0.5
    assert -1 <= float(row["axis_seed"]) <= 1 and -1 <= float(row["axis_split_half"]) <= 1
    pairs = (out / "pairs.tsv").read_text().splitlines()
    kinds = [l.split("\t")[3] for l in pairs[1:]]
    # per cell: 2 adjacent pairs, 1 period pair (1990 vs 2010), and the first-vs-last pair for reference
    assert kinds.count("adjacent") == 4 and kinds.count("period") == 2 and kinds.count("distant") == 2
    assert "seed_overlap_se" in header and 0 < float(row["seed_overlap_se"]) < 0.5
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["cells"]) == 2 and summary["best_cell"] in {c["cell"] for c in summary["cells"]}
    assert "axis measurement" in summary["best_rule"]                       # overall: the axis family's pick
    assert "most stable" in summary["suggestions"]["neighbourhoods"]["rule"]
    assert "SimLex" in summary["suggestions"]["benchmarks"]["rule"]
    for c in summary["cells"]:
        assert c["years"] == 4 and c["adjacent_overlap"] == c["adjacent_overlap"]
        assert c["period_overlap"] == c["period_overlap"] and c["change_ratio"] > 0
        assert c["seed_overlap_sd"] >= 0 and c["adjacent_overlap_sd"] >= 0            # spread across years / pairs
        assert c["axis_rho"] == c["axis_rho"] and c["axis_adjacent"] == c["axis_adjacent"]
        assert c["axis_period"] == c["axis_period"] and c["axis_change_ratio"] == c["axis_change_ratio"]
    pair_header = pairs[0].split("\t")
    assert "axis" in pair_header                                                  # agreement per year pair
    assert (out / "summary.tsv").read_text().startswith("cell\t")
    search_json = json.loads((out / "search.json").read_text())
    assert search_json["sample_years"] == [1990, 1991, 2010, 2011]
    assert search_json["axis_lexicon"].endswith("vader_lexicon.txt") and search_json["axis_poles"].endswith("poles.txt")
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


def test_cells_are_scored_as_their_models_finish_and_resume_keeps_scores(rocks, tmp_path):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "t2"
    seen = []                                  # (model being trained, result rows so far, pair rows so far)

    def watching_train(corpus_file, cell, seed, workers):
        def n_rows(name):
            p = out / name
            return max(0, len(p.read_text().splitlines()) - 1) if p.exists() else 0
        seen.append((Path(corpus_file).name, cell.stem, n_rows("results.tsv"), n_rows("pairs.tsv"),
                     n_rows("summary.tsv")))
        return fake_train(corpus_file, cell, seed, workers)

    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990, 1991], vector_size=[8, 16],
                     window=[2], min_count=[2], epochs=[1], seeds=2, name="t2", mode="restart", workers=1,
                     probes_per_decile=3, similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"),
                     train_fn=watching_train)
    # the first cell's row was written before the second cell's models were trained, within the year;
    # the 1990-1991 pairs were written before 1991's second cell trained
    first_of_second_cell_1990 = next(s for s in seen if s[1].startswith("vs16") and "y1990" in s[0])
    assert first_of_second_cell_1990[2] == 1 and first_of_second_cell_1990[4] == 1     # summary follows at once
    first_of_second_cell_1991 = next(s for s in seen if s[1].startswith("vs16") and "y1991" in s[0])
    assert first_of_second_cell_1991[2] == 3 and first_of_second_cell_1991[3] == 1
    assert len((out / "pairs.tsv").read_text().splitlines()) == 3                 # header + adjacent x 2 cells
    # resume: scores are read back, not recomputed (the file is untouched); the summary is rebuilt
    before = (out / "results.tsv").stat().st_mtime_ns
    (out / "summary.json").unlink()
    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990, 1991], vector_size=[8, 16],
                     window=[2], min_count=[2], epochs=[1], seeds=2, name="t2", workers=1,
                     probes_per_decile=3, similarity_dataset=str(tmp_path / "data" / "simlex999.txt"),
                     train_fn=watching_train)
    assert (out / "results.tsv").stat().st_mtime_ns == before and (out / "summary.json").exists()
    rows = tune._read_results(out / "results.tsv", [1990, 1991])
    assert len(rows) == 4 and isinstance(rows[("vs8_w2_mc2_e1_sg1_wbfreq", 1990)]["seed_overlap"], float)
    assert tune._read_results(out / "results.tsv", [1990]) and not tune._read_results(out / "results.tsv", [2000])


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
    assert (d / "SOURCES.md").read_text().count("sha256") == 4


def test_cli_runs_tune_from_a_settings_file(rocks, tmp_path, monkeypatch):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    _datasets(tmp_path, monkeypatch)                                           # the image's copies
    import lexichron.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "gensim_train", fake_train)
    # the corpus writer process cannot see the fake store: run it inline
    from concurrent.futures import Future

    class Inline:
        def __init__(self, *a, **k): pass
        def submit(self, fn, *a, **k):
            f = Future(); f.set_result(fn(*a, **k)); return f
        def shutdown(self, *a, **k): pass
    import lexichron.stages.tune as tune_mod
    monkeypatch.setattr(tune_mod, "ProcessPoolExecutor", Inline)
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


def test_threads_per_model_grow_when_a_year_has_few_models(rocks, tmp_path):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    threads = []

    def counting_train(corpus_file, cell, seed, workers):
        threads.append(workers)
        return fake_train(corpus_file, cell, seed, workers)

    # one cell, 4 models a year over 2 years, 64 "CPUs" at 4 threads minimum: the 8 models
    # train side by side (years are not a barrier), 8 threads each
    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990, 1991], vector_size=[8], window=[2],
                     min_count=[2], epochs=[1], seeds=2, name="t3", mode="restart", workers=64, workers_per_model=4,
                     probes_per_decile=3, similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"),
                     train_fn=counting_train)
    assert threads == [8] * 8
    # four cells, 16 models a year: the slots are full, so the minimum applies
    threads.clear()
    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990], vector_size=[8, 16], window=[2, 3],
                     min_count=[2], epochs=[1], seeds=2, name="t4", mode="restart", workers=64, workers_per_model=4,
                     probes_per_decile=3, similarity_dataset=str(tmp_path / "data" / "simlex999.txt"),
                     train_fn=counting_train)
    assert threads == [4] * 16


def test_temp_dir_expands_environment_variables(rocks, tmp_path, monkeypatch):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990])
    monkeypatch.setenv("LEXICHRON_TEST_TMP", str(tmp_path / "local"))
    seen = []

    def noting_train(corpus_file, cell, seed, workers):
        seen.append(Path(corpus_file).parent)
        return fake_train(corpus_file, cell, seed, workers)

    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990], vector_size=[8], window=[2],
                     min_count=[2], epochs=[1], seeds=1, split_half=False, name="t5", mode="restart", workers=1,
                     probes_per_decile=3, similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"),
                     temp_dir="$LEXICHRON_TEST_TMP/spool", train_fn=noting_train)
    assert seen and seen[0] == tmp_path / "local" / "spool"


def test_suggestion_compares_cells_on_equal_years_only():
    base = {"coverage": 0.9, "simlex_rho": 0.3, "seed_overlap": 0.5, "split_half_overlap": 0.4,
            "adjacent_overlap": 0.4}
    cells = [dict(base, cell="old_a", years=8), dict(base, cell="old_b", years=8, seed_overlap=0.55),
             dict(base, cell="new", years=2, seed_overlap=0.7)]           # added later, early years only
    best, rule = tune._suggest(cells)
    assert best == "old_b" and "scored on 8 year(s)" in rule


def test_summary_keeps_cells_outside_a_narrowed_grid():
    rows = {("vs8_w2_mc2_e1_sg1_wbfreq", 1990): {"cell": "vs8_w2_mc2_e1_sg1_wbfreq", "year": 1990, "coverage": 0.9,
                                                 "simlex_rho": 0.3, "seed_overlap": 0.5},
            ("vs8_w2_mc2_e1_sg1_wbnone", 1990): {"cell": "vs8_w2_mc2_e1_sg1_wbnone", "year": 1990, "coverage": 0.9,
                                                 "simlex_rho": 0.3, "seed_overlap": 0.6}}
    grid = tune.Grid.from_settings({"vector_size": [8], "window": [2], "min_count": [2], "epochs": [1],
                                    "weight_by": ["freq"]})
    summary = tune.summarize_results(rows, [], grid.cells())
    assert [c["cell"] for c in summary["cells"]] == ["vs8_w2_mc2_e1_sg1_wbfreq", "vs8_w2_mc2_e1_sg1_wbnone"]
    assert summary["best_cell"] == "vs8_w2_mc2_e1_sg1_wbnone"                 # the earlier cell still competes


def test_years_train_side_by_side_in_the_pool(rocks, tmp_path, monkeypatch):
    """With a real pool, a slot freed by one year's model takes the next
    year's model: the tail of a year no longer idles the node. Run with a
    thread pool and a stand-in trainer whose models take unequal times."""
    import threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(tune, "_pool", lambda n: ThreadPoolExecutor(max_workers=n))
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991, 1992])
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "t5"
    events = []                                # (what, year, cell, time)
    lock = threading.Lock()

    def slow_train(corpus_file, cell, seed, workers):
        year = int(Path(corpus_file).name.split("_")[1].lstrip("y"))      # corpus_y1990_wbfreq_strip.txt
        with lock:
            events.append(("start", year, cell.stem, _time.monotonic()))
        # one straggler per year (the bigger cell's second-half model, queued last for its year)
        _time.sleep(0.5 if cell.vector_size == 16 and corpus_file.endswith("_b.txt") else 0.05)
        res = fake_train(corpus_file, cell, seed, workers)
        with lock:
            events.append(("end", year, cell.stem, _time.monotonic()))
        return res

    tune.tune_models(5, "20200217", "eng", str(stub), sample_years=[1990, 1991, 1992], vector_size=[8, 16],
                     window=[2], min_count=[2], epochs=[1], seeds=2, name="t5", mode="restart", workers=8,
                     workers_per_model=4, probes_per_decile=3,
                     similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"), train_fn=slow_train)
    last_end_1990 = max(t for what, y, _, t in events if what == "end" and y == 1990)
    first_start_1991 = min(t for what, y, _, t in events if what == "start" and y == 1991)
    assert first_start_1991 < last_end_1990                  # 1991 began while 1990's straggler ran
    assert len([e for e in events if e[0] == "start"]) == 2 * 3 * 4
    # never more than the two slots at once
    running = 0
    for what, *_ in sorted(events, key=lambda e: e[3]):
        running += 1 if what == "start" else -1
        assert 0 <= running <= 2
    # everything scored: 2 cells x 3 years, adjacent pairs x 2 cells, the distant pair x 2 cells
    rows = tune._read_results(out / "results.tsv", [1990, 1991, 1992])
    assert len(rows) == 6 and all(isinstance(r["seed_overlap"], float) for r in rows.values())
    pairs = tune._read_pairs(out / "pairs.tsv", [1990, 1991, 1992])
    assert sorted(pairs) == sorted([(c, a, b) for c in ("vs8_w2_mc2_e1_sg1_wbfreq", "vs16_w2_mc2_e1_sg1_wbfreq")
                                    for a, b in ((1990, 1991), (1991, 1992), (1990, 1992))])
    assert {p["kind"] for p in pairs.values()} == {"adjacent", "distant"}       # three consecutive years: one period
    summary = json.loads((out / "summary.json").read_text())
    assert {c["cell"]: c["years"] for c in summary["cells"]} == {"vs8_w2_mc2_e1_sg1_wbfreq": 3,
                                                                  "vs16_w2_mc2_e1_sg1_wbfreq": 3}
    # the corpora were cleaned up as each year finished
    assert not list((out / "corpus").glob("*.txt"))


def test_year_pairs_of_a_search():
    # four adjacent pairs: 4 adjacent, 6 period (first years crossed), and first-vs-last for reference
    pairs = tune.year_pairs_of([1968, 1969, 1985, 1986, 2001, 2002, 2018, 2019])
    assert [p for p in pairs if p[2] == "adjacent"] == [(1968, 1969, "adjacent"), (1985, 1986, "adjacent"),
                                                        (2001, 2002, "adjacent"), (2018, 2019, "adjacent")]
    assert [p[:2] for p in pairs if p[2] == "period"] == [(1968, 1985), (1968, 2001), (1968, 2018), (1985, 2001),
                                                          (1985, 2018), (2001, 2018)]
    assert (1968, 2019, "distant") in pairs
    # two adjacent years only: nothing but the adjacent pair
    assert tune.year_pairs_of([1990, 1991]) == [(1990, 1991, "adjacent")]
    # lone years are periods of their own
    assert [p[:2] for p in tune.year_pairs_of([1990, 2000, 2010]) if p[2] == "period"] == [(1990, 2000), (1990, 2010), (2000, 2010)]


def test_change_ratio_is_period_over_adjacent_dissimilarity():
    from lexichron.training.grid import Cell
    cell = Cell(vector_size=8, window=2, min_count=2, epochs=1, approach="skip-gram", weight_by="none")
    rows = {(cell.stem, y): {"cell": cell.stem, "year": y, "seed_overlap": v, "split_half_overlap": 0.5}
            for y, v in ((1990, 0.60), (1991, 0.62), (2010, 0.58), (2011, 0.60))}
    pairs = [{"cell": cell.stem, "year_a": 1990, "year_b": 1991, "kind": "adjacent", "overlap": 0.80},
             {"cell": cell.stem, "year_a": 2010, "year_b": 2011, "kind": "adjacent", "overlap": 0.70},
             {"cell": cell.stem, "year_a": 1990, "year_b": 2010, "kind": "period", "overlap": 0.50},
             {"cell": cell.stem, "year_a": 1990, "year_b": 2011, "kind": "distant", "overlap": 0.45}]
    c = tune.summarize_results(rows, pairs, [cell])["cells"][0]
    assert abs(c["adjacent_overlap"] - 0.75) < 1e-9 and abs(c["period_overlap"] - 0.5) < 1e-9
    assert abs(c["change_ratio"] - 0.5 / 0.25) < 1e-9                     # (1-0.5) / (1-0.75)
    assert abs(c["seed_overlap"] - 0.6) < 1e-9 and abs(c["seed_overlap_sd"] - 0.01632993) < 1e-6
    assert "signal_to_noise" not in c


def test_resume_scores_missing_pairs_for_every_cell_with_models(rocks, tmp_path):
    """A search scored by a version without period pairs, resumed on a grid
    narrowed to one of its cells: the resume scores the period pairs of
    every cell whose seed-0 models are on disk, the other grid's included,
    and retrains nothing."""
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991, 2010, 2011])
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "t6"
    common = dict(sample_years=[1990, 1991, 2010, 2011], window=[2], min_count=[2], epochs=[1], seeds=2,
                  name="t6", workers=1, probes_per_decile=3,
                  similarity_dataset=str(_datasets(tmp_path) / "simlex999.txt"))
    tune.tune_models(5, "20200217", "eng", str(stub), vector_size=[8, 16], mode="restart", train_fn=fake_train, **common)
    # as the old version left it: no period rows
    lines = (out / "pairs.tsv").read_text().splitlines()
    (out / "pairs.tsv").write_text("\n".join(l for l in lines if "\tperiod\t" not in l) + "\n")
    assert "period" not in (out / "pairs.tsv").read_text()
    trained = []
    tune.tune_models(5, "20200217", "eng", str(stub), vector_size=[16], mode="resume",
                     train_fn=lambda *a: trained.append(a) or fake_train(*a), **common)
    assert trained == []
    pairs = tune._read_pairs(out / "pairs.tsv", [1990, 1991, 2010, 2011])
    assert {k for k, p in pairs.items() if p["kind"] == "period"} == {
        ("vs8_w2_mc2_e1_sg1_wbfreq", 1990, 2010), ("vs16_w2_mc2_e1_sg1_wbfreq", 1990, 2010)}
    summary = json.loads((out / "summary.json").read_text())
    ratios = {c["cell"]: c["change_ratio"] for c in summary["cells"]}
    assert set(ratios) == {"vs8_w2_mc2_e1_sg1_wbfreq", "vs16_w2_mc2_e1_sg1_wbfreq"}
    assert all(r == r and r > 0 for r in ratios.values())


def test_resume_adds_axis_scores_to_rows_scored_before_them(rocks, tmp_path):
    """Rows and pairs from a run without a valence lexicon are rescored, from
    the models on disk, by a resume that has one; nothing retrains."""
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "t7"
    d = _datasets(tmp_path)
    common = dict(sample_years=[1990, 1991], vector_size=[8], window=[2], min_count=[2], epochs=[1], seeds=2,
                  name="t7", workers=1, probes_per_decile=3, similarity_dataset=str(d / "simlex999.txt"))
    tune.tune_models(5, "20200217", "eng", str(stub), mode="restart", axis_lexicon="none", train_fn=fake_train,
                     **common)                                          # 'none': even where a lexicon is bundled
    assert "axis_rho" not in (out / "results.tsv").read_text().splitlines()[0]
    trained = []
    tune.tune_models(5, "20200217", "eng", str(stub), mode="resume", axis_lexicon=str(d / "vader_lexicon.txt"),
                     axis_poles=str(d / "poles.txt"), train_fn=lambda *a: trained.append(a) or fake_train(*a), **common)
    assert trained == []
    rows = tune._read_results(out / "results.tsv", [1990, 1991])
    assert len(rows) == 2 and all(r["axis_coverage"] == r["axis_coverage"] for r in rows.values())
    pairs = tune._read_pairs(out / "pairs.tsv", [1990, 1991])
    assert all(isinstance(p.get("axis"), float) for p in pairs.values())
    assert "axis_adjacent" in json.loads((out / "summary.json").read_text())["cells"][0]
    # a third run has nothing left to do: the tables are untouched
    before = (out / "results.tsv").stat().st_mtime_ns
    tune.tune_models(5, "20200217", "eng", str(stub), mode="resume", axis_lexicon=str(d / "vader_lexicon.txt"),
                     axis_poles=str(d / "poles.txt"), train_fn=fake_train, **common)
    assert (out / "results.tsv").stat().st_mtime_ns == before


def test_bundled_lexicon_is_the_default(rocks, tmp_path, monkeypatch):
    """With the lexicon bundled (LEXICHRON_DATA_DIR) and axis_lexicon unset,
    the axis scores are computed; 'none' switches them off."""
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    out = stub / "20200217" / "eng" / "5gram_files" / "tuning" / "t8"
    _datasets(tmp_path, monkeypatch)                                  # bundles vader_lexicon.txt too
    common = dict(sample_years=[1990, 1991], vector_size=[8], window=[2], min_count=[2], epochs=[1], seeds=2,
                  name="t8", workers=1, probes_per_decile=3, axis_poles=str(tmp_path / "data" / "poles.txt"))
    tune.tune_models(5, "20200217", "eng", str(stub), mode="restart", train_fn=fake_train, **common)
    header = (out / "results.tsv").read_text().splitlines()[0]
    assert "axis_rho" in header and "axis_seed" in header
    assert json.loads((out / "search.json").read_text())["axis_lexicon"].endswith("vader_lexicon.txt")
    tune.tune_models(5, "20200217", "eng", str(stub), mode="restart", axis_lexicon="none", train_fn=fake_train,
                     **common)
    assert "axis_rho" not in (out / "results.tsv").read_text().splitlines()[0]


def test_suggestions_per_family():
    from lexichron.training.grid import Cell
    def cell(vs, e, **scores):
        c = Cell(vector_size=vs, window=5, min_count=1, epochs=e, approach="skip-gram", weight_by="none")
        return {"cell": c.stem, **c.as_dict(), "years": 8, "coverage": 0.999, **scores}
    # the search's picture: 300 dims wins the benchmarks, 200/30 the neighbourhoods, 100 dims the axis
    cells = [cell(100, 10, simlex_rho=0.356, wordsim_rho=0.64, analogy_accuracy=0.48, seed_overlap=0.567,
                  split_half_overlap=0.348, adjacent_overlap=0.404, axis_rho=0.725, axis_split_half=0.968,
                  axis_split_half_sd=0.004, axis_adjacent=0.974, axis_adjacent_sd=0.007),
             cell(100, 30, simlex_rho=0.359, wordsim_rho=0.66, analogy_accuracy=0.47, seed_overlap=0.606,
                  split_half_overlap=0.357, adjacent_overlap=0.413, axis_rho=0.725, axis_split_half=0.969,
                  axis_split_half_sd=0.005, axis_adjacent=0.977, axis_adjacent_sd=0.006),
             cell(200, 30, simlex_rho=0.395, wordsim_rho=0.67, analogy_accuracy=0.53, seed_overlap=0.60,
                  split_half_overlap=0.352, adjacent_overlap=0.422, axis_rho=0.722, axis_split_half=0.956,
                  axis_split_half_sd=0.007, axis_adjacent=0.968, axis_adjacent_sd=0.010),
             cell(300, 30, simlex_rho=0.411, wordsim_rho=0.665, analogy_accuracy=0.54, seed_overlap=0.59,
                  split_half_overlap=0.342, adjacent_overlap=0.412, axis_rho=0.721, axis_split_half=0.946,
                  axis_split_half_sd=0.009, axis_adjacent=0.960, axis_adjacent_sd=0.013)]
    picks = tune.suggest_by_family(cells)
    assert picks["benchmarks"]["cell"].startswith("vs300_") and "SimLex" in picks["benchmarks"]["rule"]
    assert picks["neighbourhoods"]["cell"].startswith("vs200_") and "most stable" in picks["neighbourhoods"]["rule"]
    # 100/30 is best on the axis score but within one SE of 100/10, which is cheaper
    assert picks["axis"]["cell"] == "vs100_w5_mc1_e10_sg1_wbnone" and "cheapest" in picks["axis"]["rule"]
    summary = tune.summarize_results({(c["cell"], 1990): dict(c, year=1990) for c in cells}, [], [])
    assert summary["best_cell"].startswith("vs100_") and "axis" in summary["best_rule"]   # overall: the axis pick
    assert set(summary["suggestions"]) == {"benchmarks", "neighbourhoods", "axis"}
    # without axis scores: the neighbourhood pick is the overall one, and there is no axis family
    bare = [{k: v for k, v in c.items() if not k.startswith("axis_")} for c in cells]
    picks = tune.suggest_by_family(bare)
    assert "axis" not in picks and picks["neighbourhoods"]["cell"].startswith("vs200_")

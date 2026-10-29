"""lexichron measure: a word's series, with its error, from a replicate ensemble."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                          # noqa: E402
from ngramprep.common_db import api as dbapi                # noqa: E402
from ngramprep.ngram_acquire.progress import read_progress  # noqa: E402
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths   # noqa: E402
from lexichron import inspection                            # noqa: E402
from lexichron.safeguards import overwrite_target           # noqa: E402
from lexichron.stages import measure as stage               # noqa: E402
from lexichron.training import measures, trainer            # noqa: E402

WORDS = ["he", "she", "man", "woman", "king", "queen", "doctor", "nurse", "car", "rare"]
POS, NEG = ["he", "man", "king"], ["she", "woman", "queen"]


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def _ensemble(tmp_path, years=(1990, 1991, 1992), corpora=2, seeds=2, unfinished=(1992,), seed=0):
    """A fake ensemble: random 8-d models, 'rare' missing from corpus 1, the
    ``unfinished`` years short of one model. Returns (stub, ensemble dir)."""
    stub = tmp_path / "corp"
    corpus = Path(derive_paths(str(stub), 5, "20200217", "eng")["pivoted_db"]).parent
    ens = corpus / "models" / "e1"
    rng = np.random.default_rng(seed)
    for y in years:
        (ens / f"y{y}").mkdir(parents=True)
        for c in range(corpora):
            for s in range(seeds):
                if y in unfinished and c == corpora - 1 and s == seeds - 1:
                    continue
                ws = [w for w in WORDS if not (w == "rare" and c == 1)]
                m = rng.normal(size=(len(ws), 8)).astype(np.float32)
                v = trainer.Vectors(words=ws, matrix=m, counts=np.arange(len(ws), 0, -1) * 10)
                trainer.save_vectors(v, str(ens / f"y{y}" / f"c{c:02d}_s{s}.npz"))
    (ens / "train.json").write_text(json.dumps({"cell_stem": "vs8_w2_mc5_e10_sg1_wbnone", "years": [years[0], years[-1]],
                                                "years_with_data": list(years), "corpus_replicates": corpora,
                                                "seeds_per_corpus": seeds}))
    return stub, ens


def _run(stub, **kw):
    args = dict(ngram_size=5, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                ensemble="e1", name="gender", words=["doctor", "nurse", "car", "rare", "missing"],
                positive=POS, negative=NEG)
    args.update(kw)
    return Path(stage.measure_series(**args))


def test_projection_matches_a_hand_computation_and_pairs_equal_lists(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path)
    out = _run(stub, progress_path=str(tmp_path / "p.json"))
    assert out == ens / "measures" / "gender"
    assert sorted(p.name for p in out.iterdir()) == ["logs", "measure.json", "series.tsv", "summary.tsv",
                                                     "y1990.json", "y1990.npz", "y1991.json", "y1991.npz"]
    # by hand, on one model: cosine with the unit axis between the poles' unit centroids
    v = trainer.load_vectors(str(ens / "y1990" / "c00_s1.npz"))
    u = v.unit()
    ax = u[[v.index(w) for w in POS]].mean(axis=0) - u[[v.index(w) for w in NEG]].mean(axis=0)
    ax /= np.linalg.norm(ax)
    want = float(u[v.index("doctor")] @ ax)
    with np.load(out / "y1990.npz", allow_pickle=True) as z:
        words = [str(w) for w in z["words"]]
        assert words == ["doctor", "nurse", "car", "rare", "missing"]
        assert z["values"].shape == (2, 2, 5) and z["values"][0, 1, 0] == pytest.approx(want, abs=1e-6)
        assert np.isnan(z["values"][1, :, 3]).all() and np.isnan(z["values"][:, :, 4]).all()
    # the same axis from pairs
    spec_pairs = measures.Spec("projection", ("doctor",), positive=("he", "man", "king"), negative=("she", "woman", "queen"))
    m = measures.measure(v, spec_pairs)
    assert m.values["doctor"] == pytest.approx(want, abs=1e-6) and m.coverage == {"positive": 3, "negative": 3}
    rec = json.loads((out / "y1990.json").read_text())
    rows = {r["word"]: r for r in rec["words"]}
    assert rows["doctor"]["n"] == 4 and rows["doctor"]["se"] > 0 and rows["doctor"]["var_corpus"] >= 0
    assert rows["rare"]["n"] == 2 and rows["rare"]["mean"] == rows["rare"]["mean"]       # a mean from the replicates that have it
    assert rows["missing"]["n"] == 0 and rows["missing"]["mean"] is None
    assert rec["words_all"] == 3 and rec["words_some"] == 1 and rec["words_none"] == 1
    assert rec["coverage"] == {"positive": [3, 3], "negative": [3, 3]} and rec["summary"]["words"] == 3
    series = (out / "series.tsv").read_text().splitlines()
    assert series[0].split("\t") == list(stage.SERIES_COLUMNS)
    assert len(series) == 1 + 2 * 5 and series[1].startswith("1990\tdoctor\t")
    doc = read_progress(tmp_path / "p.json")
    assert doc["state"] == "done" and doc["files_done"] == 3 and doc["counters"]["years_pending"] == 1
    assert "1992" in doc["message"]                                       # the unfinished year is pending
    manifest = json.loads((out / "measure.json").read_text())
    assert manifest["kind"] == "projection" and manifest["positive"] == POS and manifest["ensemble_name"] == "e1"


def test_resume_skips_measured_years_and_a_changed_spec_is_refused(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path)
    out = _run(stub)
    before = (out / "y1990.json").stat().st_mtime_ns
    _run(stub, progress_path=str(tmp_path / "p.json"))
    assert (out / "y1990.json").stat().st_mtime_ns == before
    doc = read_progress(tmp_path / "p.json")
    assert doc["files_skipped"] == 2 and doc["counters"].get("years_done", 0) == 0
    with pytest.raises(ValueError, match="another name"):
        _run(stub, words=["doctor"])
    # finishing the year's training lets a resume fill it in
    v = trainer.load_vectors(str(ens / "y1992" / "c00_s0.npz"))
    trainer.save_vectors(v, str(ens / "y1992" / "c01_s1.npz"))
    _run(stub)
    assert (out / "y1992.json").is_file()
    assert len((out / "series.tsv").read_text().splitlines()) == 1 + 3 * 5
    # restart replaces; the safeguard names the directory
    kwargs = dict(ngram_size=5, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                  ensemble="e1", name="gender", mode="restart")
    assert overwrite_target("measure", kwargs) == out
    out2 = _run(stub, words=["doctor"], mode="restart")
    assert out2 == out and json.loads((out / "measure.json").read_text())["words"] == ["doctor"]


def test_array_tasks_take_alternate_years_and_files_merge(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path, unfinished=())
    _run(stub, array_task=0, array_count=2)
    out = ens / "measures" / "gender"
    assert sorted(p.name for p in out.glob("y*.json")) == ["y1990.json", "y1992.json"]
    _run(stub, array_task=1, array_count=2)
    assert sorted(p.name for p in out.glob("y*.json")) == ["y1990.json", "y1991.json", "y1992.json"]
    assert [l.split("\t")[0] for l in (out / "summary.tsv").read_text().splitlines()[1:]] == ["1990", "1991", "1992"]
    with pytest.raises(ValueError, match="job array"):
        _run(stub, array_task=0, array_count=2, mode="restart")


def test_similarity_frequency_files_and_years(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path, unfinished=())
    (tmp_path / "words.txt").write_text("doctor\nnurse  # a comment\n\ncar, bus\n")
    (tmp_path / "pairs.txt").write_text("he\tshe\nman\twoman\nking\tqueen\n")
    out = _run(stub, name="sim", kind="similarity", words=(), words_file=str(tmp_path / "words.txt"),
               target=["he", "man"], years=(1990, 1991))
    manifest = json.loads((out / "measure.json").read_text())
    assert manifest["words"] == ["doctor", "nurse", "car", "bus"] and manifest["target"] == ["he", "man"]
    assert sorted(p.name for p in out.glob("y*.json")) == ["y1990.json", "y1991.json"]
    v = trainer.load_vectors(str(ens / "y1990" / "c00_s0.npz"))
    u = v.unit()
    cen = u[[v.index("he"), v.index("man")]].mean(axis=0)
    cen /= np.linalg.norm(cen)
    with np.load(out / "y1990.npz", allow_pickle=True) as z:
        assert z["values"][0, 0, 0] == pytest.approx(float(u[v.index("doctor")] @ cen), abs=1e-6)
    out = _run(stub, name="pairs", positive=(), negative=(), poles_file=str(tmp_path / "pairs.txt"), words=["doctor"])
    manifest = json.loads((out / "measure.json").read_text())
    assert manifest["positive"] == POS and manifest["negative"] == NEG
    out = _run(stub, name="freq", kind="frequency", words=["he", "rare"])
    rec = {r["word"]: r for r in json.loads((out / "y1990.json").read_text())["words"]}
    assert rec["he"]["mean"] == pytest.approx(100 / 550, rel=0.1) and rec["he"]["var_seed"] == 0   # counts differ by corpus only
    assert rec["rare"]["n"] == 2


def test_bad_specifications_are_refused(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path)
    with pytest.raises(ValueError, match="no words"):
        _run(stub, words=())
    with pytest.raises(ValueError, match="each pole"):
        _run(stub, positive=["he"])
    with pytest.raises(ValueError, match="target"):
        _run(stub, kind="similarity")
    with pytest.raises(ValueError, match="kind must be"):
        _run(stub, kind="drift")
    with pytest.raises(FileNotFoundError, match="no ensemble"):
        _run(stub, ensemble="nope")
    with pytest.raises(FileNotFoundError, match="words file"):
        _run(stub, words_file=str(tmp_path / "none.txt"))


def test_inspection_reads_a_measure(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path)
    out = _run(stub)
    corpus = ens.parent.parent
    assert inspection.measures(str(corpus)) == [("e1 / gender", str(out))]
    assert inspection.describe(str(out))["kind"] == "measure"
    m = inspection.read_measure(str(out))
    assert m["years_done"] == [1990, 1991] and m["years_pending"] == [1992] and m["words"][0] == "doctor"
    headings, rows = inspection.measure_table(m, "mean")
    assert headings == ["Year", "doctor", "nurse", "car", "rare", "missing"] and len(rows) == 2
    assert rows[0][0] == "1990" and rows[0][5] == "–" and rows[0][1] != "–"
    headings, rows = inspection.measure_table(m, "n")
    assert rows[0][1:] == ["4", "4", "4", "2", "–"]
    tiles = dict((l, v) for l, v, _ in inspection.measure_overview(m))
    assert tiles["Kind"] == "projection" and tiles["Words"] == "5" and tiles["Years"] == "2/3"
    title, prof = inspection.measure_profile(m)
    assert title.startswith("Error · 2 years") and prof[0][0] == "SE of mean"
    text = "\n".join(inspection.format_measure(m))
    assert "measure gender" in text and "pending" in text and "doctor" in text
    assert inspection.measure_table(inspection.read_measure(str(tmp_path / "nowhere")))[1] == []


def test_a_word_group_is_averaged_inside_each_model(rocks, tmp_path):
    stub, ens = _ensemble(tmp_path, unfinished=())
    out = _run(stub, name="groups", words=["doctor", "nurse|doctor", "car|nope"])
    v = trainer.load_vectors(str(ens / "y1990" / "c00_s0.npz"))
    spec = measures.Spec("projection", ("doctor", "nurse|doctor", "car|nope"), positive=tuple(POS), negative=tuple(NEG))
    m = measures.measure(v, spec)
    single = measures.measure(v, measures.Spec("projection", ("doctor", "nurse", "car"), positive=tuple(POS), negative=tuple(NEG))).values
    assert m.values["nurse|doctor"] == pytest.approx((single["nurse"] + single["doctor"]) / 2)
    assert m.values["car|nope"] == pytest.approx(single["car"])            # a missing member is left out
    assert spec.lexemes == ["doctor", "nurse", "car", "nope"]
    series = (out / "series.tsv").read_text().splitlines()
    assert any(l.split("\t")[1] == "nurse|doctor" for l in series[1:])
    with pytest.raises(ValueError, match="empty word or group"):
        measures.Spec("projection", ("doctor", "|"), positive=tuple(POS), negative=tuple(NEG)).validate()

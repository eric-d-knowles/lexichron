"""lexichron train: a replicate ensemble per year, with a stand-in trainer."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                      # noqa: E402
from ngramprep.common_db import api as dbapi            # noqa: E402
from ngramprep.ngram_acquire.progress import read_progress   # noqa: E402
from lexichron.stages import train                     # noqa: E402
from lexichron.training import ensemble                # noqa: E402
from lexichron.training.grid import Cell               # noqa: E402
from test_tune_stage import _corpus, _datasets, fake_train   # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def test_replicates_are_planned_year_by_corpus_by_seed_and_seeded_from_their_coordinates():
    cell = Cell(8, 2, 2, 1, "skip-gram", "none")
    tasks = ensemble.plan_replicas(cell, [1990, 1991], 2, 2, noise_seed=7)
    assert [t.stem for t in tasks] == ["y1990/c00_s0", "y1990/c00_s1", "y1990/c01_s0", "y1990/c01_s1",
                                       "y1991/c00_s0", "y1991/c00_s1", "y1991/c01_s0", "y1991/c01_s1"]
    assert tasks[0].corpus_key == (1990, 0) and tasks[2].corpus_key == (1990, 1)
    seeds = {t.seed for t in tasks}
    assert len(seeds) == 8 and all(0 <= s < 2 ** 31 for s in seeds)         # every replicate its own seed
    assert ensemble.resample_seed(7, 1990, 1) == ensemble.resample_seed(7, 1990, 1)
    assert ensemble.resample_seed(7, 1990, 1) != ensemble.resample_seed(8, 1990, 1)
    assert ensemble.training_seed(7, 1990, 1, 0) != ensemble.training_seed(7, 1990, 1, 1)
    with pytest.raises(ValueError):
        ensemble.plan_replicas(cell, [1990], 0, 2, 0)


def test_variance_components_separate_the_corpus_and_the_seed():
    rng = np.random.default_rng(0)
    C, S, W = 30, 4, 500
    truth = rng.normal(size=W)
    corpus_effect = rng.normal(scale=0.3, size=(C, 1, W))              # var 0.09
    seed_effect = rng.normal(scale=0.1, size=(C, S, W))                # var 0.01
    v = truth + corpus_effect + seed_effect
    vc = ensemble.variance_components(v)
    assert vc.words == W and vc.corpora == C and vc.seeds == S
    assert vc.var_seed == pytest.approx(0.01, rel=0.1) and vc.var_corpus == pytest.approx(0.09, rel=0.15)
    assert vc.se_mean == pytest.approx(np.sqrt(0.09 / C + 0.01 / (C * S)), rel=0.15)
    assert vc.var_words == pytest.approx(1.0, rel=0.15) and 0 < vc.noise_share < 0.01
    d = vc.as_dict()
    assert set(d) == {"axis_words", "axis_var_seed", "axis_var_corpus", "axis_var_replicate", "axis_se_mean",
                      "axis_var_words", "axis_noise_share"}
    # one factor only: the other's variance is the replicate variance
    one_seed = ensemble.variance_components(v[:, :1, :])
    assert one_seed.var_seed != one_seed.var_seed and one_seed.var_replicate == pytest.approx(0.10, rel=0.15)
    one_corpus = ensemble.variance_components(v[:1, :, :])
    assert one_corpus.var_corpus != one_corpus.var_corpus and one_corpus.var_replicate == pytest.approx(0.01, rel=0.2)
    words, cube = ensemble.projection_cube({(0, 0): {"a": 1.0, "b": 2.0}, (0, 1): {"b": 3.0, "a": 0.0, "c": 9.0}}, 1, 2)
    assert words == ["a", "b"] and cube.tolist() == [[[1.0, 2.0], [0.0, 3.0]]]


def test_train_writes_an_ensemble_per_year_with_its_noise_profile(rocks, tmp_path):
    stub = tmp_path / "corpora"
    paths = _corpus(stub, [1990, 1991, 1992])
    d = _datasets(tmp_path)
    progress = tmp_path / "p.json"
    written = []
    import lexichron.stages.train as train_mod
    real_write = train_mod.write_corpus

    def watching_write(*a, **kw):
        written.append((a[1], kw.get("resample_seed")))
        return real_write(*a, **kw)
    train_mod.write_corpus = watching_write
    try:
        out = train.train_models(5, "20200217", "eng", str(stub), years=(1989, 1993), vector_size=8, window=2,
                                 min_count=2, epochs=1, approach="skip-gram", weight_by="none",
                                 corpus_replicates=3, seeds_per_corpus=2, noise_seed=5, name="e1", mode="restart",
                                 workers=1, axis_lexicon=str(d / "vader_lexicon.txt"), axis_poles=str(d / "poles.txt"),
                                 progress_path=str(progress), train_fn=fake_train)
    finally:
        train_mod.write_corpus = real_write
    out = Path(out)
    assert out == paths["pivoted_db"].parent / "models" / "e1"
    manifest = json.loads((out / "train.json").read_text())
    assert manifest["years_with_data"] == [1990, 1991, 1992] and manifest["cell_stem"] == "vs8_w2_mc2_e1_sg1_wbnone"
    assert manifest["corpus_replicates"] == 3 and manifest["seeds_per_corpus"] == 2 and manifest["resampling"] == "poisson"
    models = sorted(str(p.relative_to(out)) for p in out.glob("y*/*.npz"))
    assert len(models) == 18 and models[0] == "y1990/c00_s0.npz" and "y1992/c02_s1.npz" in models
    # every corpus replicate was written once, with its own Poisson seed, and dropped when its models were done
    assert len(written) == 9 and len({s for _, s in written}) == 9 and all(s is not None for _, s in written)
    assert not list((out / "corpus").glob("*.txt"))
    year = json.loads((out / "y1990" / "year.json").read_text())
    assert year["models_on_disk"] == 6 and len(year["models"]) == 6 and len(year["corpora"]) == 3
    assert year["vocab_size"] > 10 and year["train_seconds"] is not None
    assert year["axis_words"] > 5 and year["axis_var_seed"] >= 0 and year["axis_var_corpus"] >= 0
    assert year["axis_se_mean"] > 0 and 0 <= year["axis_noise_share"]
    doc = read_progress(progress)
    assert doc["state"] == "done" and doc["files_total"] == 18 and doc["files_done"] == 18
    assert doc["counters"]["models"] == 18 and doc["counters"]["years"] == 3 and "3 of 3 years" in doc["message"]
    # resume: nothing retrained; a year whose record is missing gets it back from the models on disk
    (out / "y1991" / "year.json").unlink()
    trained = []
    train.train_models(5, "20200217", "eng", str(stub), years=(1989, 1993), vector_size=8, window=2, min_count=2,
                       epochs=1, approach="skip-gram", weight_by="none", corpus_replicates=3, seeds_per_corpus=2,
                       noise_seed=5, name="e1", workers=1, axis_lexicon=str(d / "vader_lexicon.txt"),
                       axis_poles=str(d / "poles.txt"), progress_path=str(tmp_path / "p2.json"),
                       train_fn=lambda *a: trained.append(a) or fake_train(*a))
    assert trained == [] and read_progress(tmp_path / "p2.json")["files_skipped"] == 18
    again = json.loads((out / "y1991" / "year.json").read_text())
    assert again["models_on_disk"] == 6 and again["axis_words"] == year["axis_words"]
    assert again["models"][0]["vocab_size"] == year["models"][0]["vocab_size"]      # read from the model itself
    assert again["train_seconds"] is None                                            # not known to this run


def test_resampling_none_is_a_seed_only_ensemble(rocks, tmp_path):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990])
    out = Path(train.train_models(5, "20200217", "eng", str(stub), years=(1990, 1990), vector_size=8, window=2,
                                  min_count=2, epochs=1, corpus_replicates=5, seeds_per_corpus=3, resampling="none",
                                  name="e2", mode="restart", workers=1, axis_lexicon="none", train_fn=fake_train))
    assert sorted(p.name for p in (out / "y1990").glob("*.npz")) == ["c00_s0.npz", "c00_s1.npz", "c00_s2.npz"]
    manifest = json.loads((out / "train.json").read_text())
    assert manifest["corpus_replicates"] == 1
    year = json.loads((out / "y1990" / "year.json").read_text())
    assert "axis_var_seed" not in year                                     # no lexicon: no noise profile
    with pytest.raises(ValueError):
        train.train_models(5, "20200217", "eng", str(stub), years=(1990, 1990), resampling="bootstrap", name="e2",
                           train_fn=fake_train)
    with pytest.raises(ValueError, match="no year"):
        train.train_models(5, "20200217", "eng", str(stub), years=(1800, 1801), name="e3", train_fn=fake_train)


def test_a_job_array_shares_the_years(rocks, tmp_path):
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991, 1992])
    common = dict(years=(1990, 1992), vector_size=8, window=2, min_count=2, epochs=1, corpus_replicates=2,
                  seeds_per_corpus=1, name="arr", workers=1, axis_lexicon="none", train_fn=fake_train)
    with pytest.raises(ValueError, match="restart"):
        train.train_models(5, "20200217", "eng", str(stub), mode="restart", array_task=0, array_count=2, **common)
    out = Path(train.train_models(5, "20200217", "eng", str(stub), array_task=0, array_count=2,
                                  progress_path=str(tmp_path / "t0.json"), **common))
    doc0 = read_progress(tmp_path / "t0.json")
    assert doc0["array"] == {"task": 0, "count": 2, "years": [1990, 1992]} and doc0["files_total"] == 4
    assert sorted(p.name for p in out.glob("y*")) == ["y1990", "y1992"] and "so far" in doc0["message"]
    train.train_models(5, "20200217", "eng", str(stub), array_task=1, array_count=2,
                       progress_path=str(tmp_path / "t1.json"), **common)
    doc1 = read_progress(tmp_path / "t1.json")
    assert doc1["files_total"] == 2 and "3 of 3 years complete" in doc1["message"] and "so far" not in doc1["message"]
    assert len(list(out.glob("y*/year.json"))) == 3
    train.train_models(5, "20200217", "eng", str(stub), array_task=4, array_count=5,
                       progress_path=str(tmp_path / "t4.json"), **common)
    assert "no year falls" in read_progress(tmp_path / "t4.json")["message"]


def test_cli_runs_train_from_a_settings_file(rocks, tmp_path, monkeypatch):
    import yaml
    from lexichron import cli
    import lexichron.training.trainer as trainer_mod
    stub = tmp_path / "corpora"
    _corpus(stub, [1990])
    _datasets(tmp_path, monkeypatch)
    monkeypatch.setattr(trainer_mod, "gensim_train", fake_train)
    import lexichron.stages.train as train_mod
    monkeypatch.setattr(train_mod, "_pool", lambda n: __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=n))
    project = tmp_path / "lexichron.yaml"
    project.write_text(yaml.safe_dump({
        "corpus": {"release": "20200217", "language": "eng", "ngram_size": 5, "db_path_stub": str(stub)},
        "train": {"years": [1990, 1990], "vector_size": 8, "window": 2, "min_count": 2, "epochs": 1,
                  "corpus_replicates": 2, "seeds_per_corpus": 2, "name": "cli", "workers": 2, "workers_per_model": 1},
    }))
    assert cli.main(["train", str(project), "--dry-run"]) == 0
    assert cli.main(["train", str(project)]) == 0
    out = stub / "20200217" / "eng" / "5gram_files" / "models" / "cli"
    assert len(list((out / "y1990").glob("*.npz"))) == 4 and (out / "lexichron.json").exists()
    runs = list((tmp_path / ".lexichron" / "runs").glob("*_train"))
    assert len(runs) == 1


def test_inspect_reads_an_ensemble(rocks, tmp_path, capsys):
    from lexichron import cli
    from lexichron.inspection import describe, ensembles, ensemble_table, read_ensemble
    stub = tmp_path / "corpora"
    _corpus(stub, [1990, 1991])
    d = _datasets(tmp_path)
    out = Path(train.train_models(5, "20200217", "eng", str(stub), years=(1990, 1991), vector_size=8, window=2,
                                  min_count=2, epochs=1, corpus_replicates=2, seeds_per_corpus=2, name="ins",
                                  mode="restart", workers=1, axis_lexicon=str(d / "vader_lexicon.txt"),
                                  axis_poles=str(d / "poles.txt"), train_fn=fake_train))
    assert ensembles(str(out.parent.parent)) == [("ins", str(out))]
    ens = read_ensemble(str(out))
    assert ens["years_done"] == [1990, 1991] and ens["models_on_disk"] == 8 and ens["models_expected"] == 8
    headings, rows = ensemble_table(ens)
    assert headings[:3] == ["Year", "Models", "Vocab"] and "SE of mean" in headings and "Replicate var" not in headings
    assert rows[0][0] == "1990" and rows[0][1] == "4/4"
    (out / "y1991" / "year.json").unlink()
    rows = ensemble_table(read_ensemble(str(out)))[1]
    assert rows[1][1] == "4/4 (training)"
    assert describe(str(out))["kind"] == "ensemble"
    assert cli.main(["inspect", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ensemble ins" in text and "1990" in text

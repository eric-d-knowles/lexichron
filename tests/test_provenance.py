"""lexichron.provenance: every database a stage writes carries lexichron.json."""
import json
import struct
import pytest

core = pytest.importorskip("ngramprep.ngram_filter.filters.core_cy")

import fake_rocks  # noqa: E402
from ngramprep.common_db import api as dbapi  # noqa: E402
from lexichron import provenance  # noqa: E402
from lexichron.stages import filter as stage  # noqa: E402
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths  # noqa: E402
from test_filter_stage import FakeLemmatizer, _raw_unigrams  # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


@pytest.fixture
def no_spacy(monkeypatch):
    import ngramprep.ngram_filter.stopwords as sw
    import ngramprep.ngram_filter.lemmatizer as lm
    monkeypatch.setattr(sw, "load_stopwords", lambda lang: ({"the", "of"}, lang))
    monkeypatch.setattr(lm, "CachedSpacyLemmatizer", lambda language=None, **kw: FakeLemmatizer())


def _fake_acquire_record(db_dir):
    """What an acquisition run leaves behind, without downloading anything."""
    doc = {"stage": "acquire", "state": "done", "db_path": str(db_dir), "files_total": 14,
           "started": "2026-10-14T03:04:35+00:00", "finished": "2026-10-14T03:11:57+00:00",
           "hostname": "cs602", "slurm_job_id": "19767911",
           "counters": {"lines": 49_924_752, "inserted": 26_292_918},
           "corpus": {"files": 14, "lines": 49_924_752, "bytes": 26.5e9, "entries": 26_292_918},
           "database": {"records": 26_292_918, "bytes": 3.3e9, "compacted": True},
           "inputs": {"source": "Google Books 20200217 eng-us 1-grams (tagged)"}}
    settings = {"ngram_size": 1, "repo_release_id": "20200217", "repo_corpus_id": "eng-us",
                "db_path_stub": "/x", "ngram_type": "tagged", "random_seed": 123}
    return provenance.write_provenance(db_dir, provenance.build_record("acquire", settings, doc))


def test_record_from_a_finished_run_and_chain_to_its_source(rocks, no_spacy, tmp_path):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    _fake_acquire_record(paths["src_db"])
    progress = tmp_path / "p.json"
    settings = dict(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                    workers=2, mode="restart", output_vocabulary_top_n=2, spool_dir=str(tmp_path),
                    progress_path=str(progress))
    stage.filter_ngrams(**settings)
    written = provenance.record_run("filter", settings, str(progress))
    assert written == paths["dst_db"] / "lexichron.json"
    rec = json.loads(written.read_text())
    assert rec["format"] == 1 and rec["stage"] == "filter" and rec["lexichron"]
    assert rec["settings"]["mode"] == "restart" and "progress_path" not in rec["settings"]
    assert rec["figures"]["scanned"] == 8 and rec["figures"]["kept"] == 5
    assert rec["database"]["records"] == 3 and rec["database"]["compacted"]
    # the source's own record is summarized and fingerprinted
    src = rec["inputs"]["source"]
    assert src["path"] == str(paths["src_db"])
    assert src["provenance"]["stage"] == "acquire" and src["provenance"]["figures"]["lines_read"] == 49_924_752
    assert src["provenance"]["sha256"] == provenance.digest(paths["src_db"] / "lexichron.json")
    assert rec["inputs"]["vocabulary"] is None                     # unigrams: rule filters only
    # readable through inspect
    from lexichron.inspection import describe
    assert describe(str(paths["dst_db"]))["provenance"]["stage"] == "filter"
    lines = provenance.format_provenance(rec)
    assert lines[0].startswith("Made by lexichron") and any(l.startswith("Source:") for l in lines)
    assert any("5 kept" in l for l in lines)


def test_multigram_record_names_the_vocabulary(rocks, no_spacy, tmp_path):
    stub = tmp_path / "corpora"
    uni = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(uni["src_db"]))
    stage.filter_ngrams(1, "20200217", "eng", str(stub), workers=2, mode="restart", output_vocabulary_top_n=2,
                        spool_dir=str(tmp_path))
    bi = derive_paths(str(stub), 2, "20200217", "eng")
    db = fake_rocks.open(str(bi["src_db"]), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"cat_NOUN run_VERB", struct.pack("<QQQ", 1990, 3, 1))
    progress = tmp_path / "p2.json"
    settings = dict(ngram_size=2, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                    workers=2, mode="restart", spool_dir=str(tmp_path), progress_path=str(progress))
    stage.filter_ngrams(**settings)
    rec = json.loads(provenance.record_run("filter", settings, str(progress)).read_text())
    v = rec["inputs"]["vocabulary"]
    assert v["path"] == str(uni["whitelist"]) and v["tokens"] == 2 and len(v["sha256"]) == 64
    assert rec["database"]["path"].endswith("2grams_pivoted.db")
    assert any("Vocabulary:" in l and "2 tokens" in l for l in provenance.format_provenance(rec))


def test_no_record_for_an_unfinished_run(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"stage": "filter", "state": "running", "db_path": str(tmp_path)}))
    assert provenance.record_run("filter", {}, str(tmp_path / "p.json")) is None
    assert provenance.read_provenance(tmp_path) is None

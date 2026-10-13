"""The filter stage end to end on the range-map engine, with the real Cython
kernel and an in-memory database."""
import struct

import pytest

core = pytest.importorskip("ngramprep.ngram_filter.filters.core_cy")

import fake_rocks  # noqa: E402
from ngramprep.common_db import api as dbapi  # noqa: E402
from ngramprep.ngram_acquire.progress import read_progress  # noqa: E402
from ngramprep.ngram_filter.config import FilterConfig, PipelineConfig  # noqa: E402
from ngramprep.ngram_filter.pipeline.orchestrator import build_processed_db, derive_paths  # noqa: E402


def _packed(*triples):
    flat = [x for t in triples for x in t]
    return struct.pack(f"<{len(flat)}Q", *flat)


def _unpack(blob):
    n = len(blob) // 8
    v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


class FakeLemmatizer:
    def lemmatize(self, word, pos="NOUN"):
        return {"women": "woman", "cats": "cat", "running": "run"}.get(word, word)


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def _raw_unigrams(path):
    db = fake_rocks.open(path, mode="rw", create_if_missing=True)
    rows = {
        b"Cats_NOUN": _packed((1990, 10, 2), (2000, 5, 1)),
        b"cats_NOUN": _packed((1990, 1, 1)),
        b"cat_NOUN": _packed((1990, 7, 3)),
        b"the_DET": _packed((1990, 100, 50)),          # stopword
        b"women_NOUN": _packed((1995, 3, 1)),
        b"running_VERB": _packed((1990, 2, 1)),
        b"a1_NUM": _packed((1990, 1, 1)),              # not alphabetic
        b"ox_NOUN": _packed((1990, 1, 1)),             # too short
        b"__processed__/1-00000-of-00001.gz": b"1",    # bookkeeping
    }
    with db.write_batch() as wb:
        for k, v in rows.items():
            wb.put(k, v)
    return db


def test_filter_stage_end_to_end(rocks, tmp_path):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    progress_path = tmp_path / "progress.json"

    out = build_processed_db(
        filter_config=FilterConfig(stop_set={"the"}, lemma_gen=FakeLemmatizer(), min_len=3,
                                   min_context_tokens=1),
        pipeline_config=PipelineConfig(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(stub), mode="restart", num_workers=2,
                                       num_initial_work_units=4, use_smart_partitioning=False,
                                       compact_after_ingest=True, output_whitelist_path="default",
                                       output_whitelist_top_n=3),
        progress_path=str(progress_path), spool_dir=str(tmp_path),
    )
    assert out == str(paths["dst_db"])
    data = fake_rocks.dump(paths["dst_db"])
    # Cats_NOUN + cats_NOUN + cat_NOUN -> "cat", summed per year
    assert _unpack(data[b"cat"]) == [(1990, 18, 6), (2000, 5, 1)]
    assert _unpack(data[b"woman"]) == [(1995, 3, 1)] and _unpack(data[b"run"]) == [(1990, 2, 1)]
    assert set(data) == {b"cat", b"woman", b"run"}        # stopword, number, short word dropped
    # whitelist written from the result, ranked by total frequency
    lines = paths["whitelist"].read_text().splitlines()
    assert [l.split("\t")[0] for l in lines] == ["cat", "woman", "run"]
    # progress document finished and describes the run in work units
    doc = read_progress(progress_path)
    assert doc["state"] == "done" and doc["stage"] == "filter" and doc["unit_label"] == "work units"
    assert doc["files_total"] == 4 and doc["counters"]["scanned"] == 8 and doc["counters"]["kept"] == 5
    # the three record counts and what each rule removed: 8 tokens in 8 unigrams;
    # 'the' stopword, 'a1' not alphabetic, 'ox' too short -> 3 n-grams dropped;
    # 5 kept records merged into 3 database keys (Cats/cats/cat -> cat)
    c = doc["counters"]
    assert c["tokens"] == 8 and c["tokens_kept"] == 5
    assert (c["tokens_stopword"], c["tokens_not_alpha"], c["tokens_too_short"], c["tokens_not_in_vocab"]) == (1, 1, 1, 0)
    assert c["ngrams"] == 8 and c["ngrams_dropped_context"] == 3
    assert 3 <= c["inserted"] <= 5                    # chunk entries: duplicates may merge in a chunk
    assert doc["database"] == {"records": 3, "bytes": doc["database"]["bytes"], "compacted": True}

    # a second run in resume mode does nothing and leaves the result alone
    build_processed_db(
        filter_config=FilterConfig(stop_set={"the"}, lemma_gen=FakeLemmatizer(), min_len=3,
                                   min_context_tokens=1),
        pipeline_config=PipelineConfig(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(stub), mode="resume", num_workers=2,
                                       use_smart_partitioning=False, output_whitelist_path=None),
        spool_dir=str(tmp_path),
    )
    assert fake_rocks.dump(paths["dst_db"]) == data


def test_flat_keyword_api_builds_same_configs(rocks, tmp_path):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    out = build_processed_db(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                             db_path_stub=str(stub), stop_set={"the"}, lemma_gen=FakeLemmatizer(),
                             min_len=3, min_context_tokens=1, num_workers=2, mode="restart",
                             use_smart_partitioning=False, output_whitelist_path=None,
                             spool_dir=str(tmp_path))
    assert set(fake_rocks.dump(out)) == {b"cat", b"woman", b"run"}
    assert not paths["whitelist"].exists()


def test_a_run_leaves_no_temporary_directories(rocks, tmp_path, monkeypatch):
    """The engine spools under $TMPDIR and keeps its partition in the
    destination; the retired pipeline's processing_tmp and .partition_cache
    must not reappear next to the databases."""
    from ngramprep.ngram_filter.config import FilterConfig, PipelineConfig
    from ngramprep.ngram_filter.pipeline.orchestrator import build_processed_db
    src = tmp_path / "corpus" / "1grams.db"
    src.parent.mkdir(parents=True)
    db = fake_rocks.open(str(src), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for i in range(300):
            wb.put(f"word{i}".encode(), _packed((1990, 1, 1)))
    db.close()
    spool = tmp_path / "spool"; spool.mkdir()
    pc = PipelineConfig(src_db=src, dst_db=src.parent / "1grams_processed.db",
                        tmp_dir=src.parent / "processing_tmp", mode="restart", num_workers=2,
                        num_initial_work_units=8, output_whitelist_path=None, compact_after_ingest=False)
    build_processed_db(filter_config=FilterConfig(min_len=1, alpha_only=False), pipeline_config=pc,
                       spool_dir=str(spool))
    # only the databases, plus the log directory when the stage set up its own
    # log file (it does not under pytest, whose capture handler is already on
    # the root logger; it does in a Slurm job)
    left = {p.name for p in src.parent.iterdir()} - {"logs"}
    assert left == {"1grams.db", "1grams_processed.db"}
    assert list(spool.iterdir()) == []

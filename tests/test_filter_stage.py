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


# ---- the checks a filter run makes of its own output --------------------------------------

def _run(stub, tmp_path, progress=None, **pipeline):
    pipeline.setdefault("mode", "restart")
    return build_processed_db(
        filter_config=FilterConfig(stop_set={"the"}, lemma_gen=FakeLemmatizer(), min_len=3, min_context_tokens=1),
        pipeline_config=PipelineConfig(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(stub), num_workers=2, num_initial_work_units=4,
                                       use_smart_partitioning=False, **pipeline),
        progress_path=str(progress) if progress else None, spool_dir=str(tmp_path))


# what _raw_unigrams holds: occurrences and volume counts of every record, and of those the filter keeps
SOURCE = {"match_total": 10 + 5 + 1 + 7 + 100 + 3 + 2 + 1 + 1, "volume_total": 2 + 1 + 1 + 3 + 50 + 1 + 1 + 1 + 1}
KEPT = (10 + 5 + 1 + 7 + 3 + 2, 2 + 1 + 1 + 3 + 1 + 1)


def test_a_filter_run_verifies_its_counts_and_that_it_read_the_whole_source(rocks, tmp_path, capsys):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    progress = tmp_path / "p.json"
    _run(stub, tmp_path, progress, source_totals=SOURCE, output_whitelist_path="default", compact_after_ingest=False)
    doc = read_progress(progress)
    v = doc["verification"]
    assert doc["state"] == "done" and v["status"] == "verified"
    assert (v["database"]["match_total"], v["database"]["volume_total"]) == KEPT
    assert (v["ledger"]["match_total"], v["ledger"]["volume_total"]) == KEPT
    assert v["source"] == {"status": "verified", "read": SOURCE, "expected": SOURCE, "disordered": 0}
    # read = kept + dropped, each counted on its own: 'the', 'a1' and 'ox' were dropped
    dropped = {"match_total": 100 + 1 + 1, "volume_total": 50 + 1 + 1}
    assert v["balance"]["status"] == "verified" and v["balance"]["dropped"] == dropped
    assert v["balance"]["emitted"] == {"match_total": KEPT[0], "volume_total": KEPT[1]} and v["balance"]["read"] == SOURCE
    assert doc["steps"] == ["ingesting", "compacting", "verifying", "whitelist"] and doc["phase"] == "whitelist"
    assert doc["checks"] == {"verify_counts": True, "source_known": True}
    out = capsys.readouterr().out
    assert "Counts verified" in out and "Source read in full" in out
    assert "Filtering accounted for: of 130 occurrences read, 28 were kept and 102 dropped (78.5%)" in out
    assert "compact_after_ingest is ignored" in out               # a complete run always compacts
    store = fake_rocks._REGISTRY[str(paths["dst_db"])]
    assert store.compactions == 1 and store.ingests >= 1          # the unigram run hands over SST files too
    assert paths["whitelist"].exists()

    # the source's own total is not known (a database acquired before counts were verified): said, not an error
    _run(stub, tmp_path, progress)
    v = read_progress(progress)["verification"]
    assert v["status"] == "verified" and v["source"]["status"] == "unknown" and v["source"]["read"] == SOURCE
    assert "Source total not known" in capsys.readouterr().out

    _run(stub, tmp_path, progress, verify_counts=False)
    assert read_progress(progress)["verification"] == {"status": "skipped"}
    assert read_progress(progress)["steps"] == ["ingesting", "compacting"]


def test_a_source_not_read_exactly_once_fails_the_run(rocks, tmp_path, capsys):
    """What a partition that skipped a key range, or gave one to two units,
    would cause: the units' total of everything they read is not the source's."""
    from ngramprep.ngram_filter.pipeline.orchestrator import FilterError
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    progress = tmp_path / "p.json"
    wrong = dict(SOURCE, match_total=SOURCE["match_total"] + 40)
    with pytest.raises(FilterError, match="counts do not match"):
        _run(stub, tmp_path, progress, source_totals=wrong, output_whitelist_path="default")
    doc = read_progress(progress)
    assert doc["state"] == "failed" and doc["verification"]["status"] == "mismatch"
    assert doc["verification"]["source"]["status"] == "mismatch"
    out = capsys.readouterr().out
    assert "THE SOURCE WAS NOT READ EXACTLY ONCE" in out and "(difference -40)" in out
    assert "Counts verified" in out                               # the destination itself is what the units emitted
    assert not paths["whitelist"].exists()                        # no vocabulary from a database that failed its check


def test_a_source_that_was_never_compacted_is_refused(rocks, tmp_path):
    from ngramprep.ngram_filter.pipeline.orchestrator import FilterError
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    src = _raw_unigrams(str(paths["src_db"]))
    src.finalize_bulk()                                           # flushed, never compacted: an unfinished acquisition
    with pytest.raises(FilterError, match="has not been compacted.*Run that stage again"):
        _run(stub, tmp_path)
    assert not paths["dst_db"].exists()
    src.compact_all()
    _run(stub, tmp_path)
    assert set(fake_rocks.dump(paths["dst_db"])) == {b"cat", b"woman", b"run"}


def test_counts_lost_inside_the_filter_fail_the_run(rocks, tmp_path, monkeypatch, capsys):
    """Binning and pivoting rewrite every kept value. A count lost there would
    pass the other checks: the source is read in full and the destination holds
    what was emitted. Kept plus dropped no longer equals read, and the run fails."""
    from ngramprep.ngram_filter.pipeline import mapper as mapper_mod
    from ngramprep.ngram_filter.pipeline.orchestrator import FilterError
    monkeypatch.setattr(mapper_mod, "bin_year_records", lambda value, bin_size: value[:24])    # loses every later year
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    progress = tmp_path / "p.json"
    with pytest.raises(FilterError, match="counts do not match"):
        build_processed_db(
            filter_config=FilterConfig(stop_set={"the"}, lemma_gen=FakeLemmatizer(), min_len=3, min_context_tokens=1,
                                       bin_size=5),
            pipeline_config=PipelineConfig(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                                           db_path_stub=str(stub), num_workers=2, num_initial_work_units=4,
                                           use_smart_partitioning=False, mode="restart", source_totals=SOURCE,
                                           output_whitelist_path="default"),
            progress_path=str(progress), spool_dir=str(tmp_path))
    v = read_progress(progress)["verification"]
    assert v["status"] == "mismatch" and v["source"]["status"] == "verified"
    assert v["balance"]["status"] == "mismatch" and v["balance"]["difference"] == {"match_total": -5, "volume_total": -1}
    out = capsys.readouterr().out
    assert "KEPT AND DROPPED DO NOT ADD UP TO WHAT WAS READ" in out and "(difference -5; volume counts -1)" in out
    assert "Counts verified" in out and not paths["whitelist"].exists()


def test_words_kept_by_request_are_in_the_vocabulary_whatever_their_rank(rocks, tmp_path, capsys):
    """always_include spares a word the discard rules, but the vocabulary was
    cut to its top N by frequency alone: a rare word passed the unigram
    rules, missed the cut, and was then dropped from every longer n-gram
    filtered against that vocabulary. The kept words now follow the top N."""
    from ngramprep.ngram_filter.filters.processor_factory import kept_forms
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    cfg = FilterConfig(stop_set={"the"}, lemma_gen=FakeLemmatizer(), min_len=3, min_context_tokens=1,
                       always_include={"women", "running", "the", "absent"})
    # what the spellings become: lemmatized like any token, and the stopword is kept
    assert kept_forms(cfg) == {b"woman", b"run", b"the", b"absent"}
    assert kept_forms(FilterConfig(lemma_gen=FakeLemmatizer())) == set()

    def run(**kw):
        build_processed_db(filter_config=cfg, pipeline_config=PipelineConfig(
            ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub), num_workers=2,
            num_initial_work_units=4, use_smart_partitioning=False, mode="restart", output_whitelist_path="default",
            **kw), spool_dir=str(tmp_path))
        return [line.split("\t") for line in paths["whitelist"].read_text().splitlines()]

    # the two most frequent are 'the' (kept from the stopword rule, 100) and 'cat' (23); 'woman' (3) and
    # 'run' (2) rank below the cut and follow it, most frequent first; 'absent' never occurs and is left out
    rows = run(output_whitelist_top_n=2)
    assert rows == [["the", "100"], ["cat", "23"], ["woman", "3"], ["run", "2"]]
    assert "the 2 most frequent and 2 kept by request below them (woman, run)" in capsys.readouterr().out
    # a cut that already includes them adds nothing, and no cut means every word
    assert [r[0] for r in run(output_whitelist_top_n=3)] == ["the", "cat", "woman", "run"]
    assert [r[0] for r in run(output_whitelist_top_n=10)] == ["the", "cat", "woman", "run"]
    assert [r[0] for r in run()] == ["the", "cat", "woman", "run"]

"""The whole acquisition, and the checks that make its counts trustworthy:
both spool formats build the same database, a complete run always compacts,
and the finished database is verified against the totals every file recorded.
Downloads are stubbed and the store is the in-memory fake."""
import gzip
import io
import json
import struct

import pytest

import fake_rocks
from ngramprep.common_db import api
from ngramprep.ngram_acquire import core as core_mod
from ngramprep.ngram_acquire import worker as worker_mod
from ngramprep.ngram_acquire.core import AcquisitionError, download_and_ingest_to_rocksdb
from ngramprep.ngram_acquire.db.build_path import build_db_path
from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX
from ngramprep.ngram_acquire.progress import read_progress
from ngramprep.ngram_acquire.verify import describe_verification, ledger_totals, scan_totals, verify_database


class _Resp:
    def __init__(self, payload):
        self.raw = io.BytesIO(payload); self.headers = {}
    def close(self): pass


def _unpack(blob):
    n = len(blob) // 8; v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


def _pack(*triplets):
    flat = [x for t in triplets for x in t]
    return struct.pack(f"<{len(flat)}Q", *flat)


# Three shards (the lines are short; the stage is told they are 4-grams so the four-token sequence is allowed). "working class" is combined, so its tagged variants
# collapse onto one key within a shard, and the same n-gram arrives from two
# shards; "a . i ." has a named target, so its case variants collapse across
# shards too. Every count is distinct enough that a wrong sum cannot hide.
FILES = {
    "2-00000-of-00003.gz": [f"a{i:03d}_NOUN x_VERB\t1990,{i + 1},1\t1991,{2 * i + 1},2" for i in range(260)]
                           + ["working_NOUN class_NOUN\t1990,500,80\t2000,600,90",
                              "working_NOUN class_VERB\t1990,20,5",
                              "A_DET ._. I_NOUN ._.\t2001,11,3"],
    "2-00001-of-00003.gz": [f"b{i:03d}_NOUN y_VERB\t1995,{3 * i + 1},1" for i in range(140)]
                           + ["working_NOUN class_NOUN\t1990,7,1\t2010,9,2",
                              "a_DET ._. i_NOUN ._.\t2001,5,2\t2002,1,1"],
    "2-00002-of-00003.gz": ["untagged pair\t1990,1000,1000",                 # filtered out: tagged only
                            "c_NOUN z_VERB\t1800,1,1"],
}
COMBINE = {"working class", "a . i . => A.I."}
EXPECTED_MATCH = (sum(i + 1 + 2 * i + 1 for i in range(260)) + 500 + 600 + 20 + 11
                  + sum(3 * i + 1 for i in range(140)) + 7 + 9 + 5 + 1 + 1)
EXPECTED_VOLUME = 260 * 3 + 80 + 90 + 5 + 3 + 140 + 1 + 2 + 2 + 1 + 1


@pytest.fixture
def store(monkeypatch, tmp_path):
    fake_rocks.reset()
    monkeypatch.setattr(api, "rs", fake_rocks)
    return fake_rocks


def acquire(monkeypatch, tmp_path, files=FILES, name="run", **kw):
    """Run the whole stage over ``files`` into its own corpus directory."""
    stub = tmp_path / name
    monkeypatch.setattr(core_mod, "discover_files",
                        lambda n, release, corpus: ("http://x/", [f"http://x/{f}" for f in files]))
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _Resp(gzip.compress("\n".join(files[url.rsplit("/", 1)[-1]]).encode())))
    kw.setdefault("combined_multigrams", COMBINE)
    kw.setdefault("ngram_type", "tagged")
    kw.setdefault("workers", 2)
    kw.setdefault("chunk_entries", 100)
    progress = stub / "progress.json"
    db_path = build_db_path(str(stub), 4, "20200217", "eng")
    try:
        download_and_ingest_to_rocksdb(ngram_size=4, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(stub), spool_dir=str(tmp_path), progress_path=str(progress), **kw)
    finally:
        acquire.progress = read_progress(progress)
    return db_path


def test_both_spool_formats_build_the_same_verified_database(monkeypatch, tmp_path, store, capsys):
    sst = acquire(monkeypatch, tmp_path, name="sst", spool_format="sst")
    doc = acquire.progress
    chunk = acquire(monkeypatch, tmp_path, name="chunk", spool_format="chunk")

    a, b = store.dump(sst), store.dump(chunk)
    assert a == b and len(a) == 260 + 140 + 3
    # the combined sequences were summed within a shard and across shards
    assert _unpack(a[b"working-class_NOUN"]) == [(1990, 527, 86), (2000, 600, 90), (2010, 9, 2)]
    assert _unpack(a[b"A.I._DET"]) == [(2001, 16, 5), (2002, 1, 1)]
    assert b"untagged pair" not in a

    for path in (sst, chunk):
        db = store.open(path)
        assert db.compactions == 1
        ledger = ledger_totals(db)
        assert (ledger["shards"], ledger["unledgered"]) == (3, 0)
        assert (ledger["match_total"], ledger["volume_total"]) == (EXPECTED_MATCH, EXPECTED_VOLUME)
    assert store.open(sst).flushes <= 1 and store.open(sst).ingests == 3       # adopted, not replayed

    v = doc["verification"]
    assert v["status"] == "verified" and doc["state"] == "done" and doc["database"]["compacted"]
    assert v["database"] == {"records": 403, "match_total": EXPECTED_MATCH, "volume_total": EXPECTED_VOLUME,
                             "disordered": 0, "malformed": 0}
    assert "Counts verified" in capsys.readouterr().out


def test_a_complete_run_always_compacts(monkeypatch, tmp_path, store, capsys):
    path = acquire(monkeypatch, tmp_path, compact_after_ingest=False)
    assert store.open(path).compactions == 1 and acquire.progress["database"]["compacted"]
    assert "compact_after_ingest is ignored" in capsys.readouterr().out


def test_a_shard_ingested_twice_fails_the_run(monkeypatch, tmp_path, store, capsys):
    """What a kill between a shard's records and its marker would cause if
    the two were not written together: the shard looks unprocessed, the next
    run takes it in again, and every count in it is doubled. The ledger
    counts the shard once; the database holds it twice."""
    path = acquire(monkeypatch, tmp_path)
    db = store.open(path)
    del db.data[PROCESSED_PREFIX + b"2-00001-of-00003.gz"]

    with pytest.raises(AcquisitionError, match="do not match"):
        acquire(monkeypatch, tmp_path)
    doc = acquire.progress
    assert doc["state"] == "failed" and doc["verification"]["status"] == "mismatch"
    second = sum(3 * i + 1 for i in range(140)) + 7 + 9 + 5 + 1
    assert doc["verification"]["database"]["match_total"] - doc["verification"]["ledger"]["match_total"] == second
    out = capsys.readouterr().out
    assert "COUNTS DO NOT MATCH" in out and f"+{second:,}" in out


def test_a_lost_record_or_a_wrong_value_is_found(monkeypatch, tmp_path, store):
    path = acquire(monkeypatch, tmp_path)
    assert verify_database(path, 3)["status"] == "verified"            # in parallel over several ranges
    db = store.open(path)

    kept = db.data.pop(b"a007_NOUN x_VERB")
    v = verify_database(path, 1)
    assert v["status"] == "mismatch" and v["ledger"]["match_total"] - v["database"]["match_total"] == 8 + 15
    assert any("difference -23" in line for line in describe_verification(v))

    db.data[b"a007_NOUN x_VERB"] = _pack((1991, 15, 2), (1990, 8, 1))     # the right counts, years out of order
    v = verify_database(path, 2)
    assert v["status"] == "mismatch" and v["database"]["disordered"] == 1
    assert v["database"]["match_total"] == v["ledger"]["match_total"]

    db.data[b"a007_NOUN x_VERB"] = kept[:-3]                              # not whole triplets
    assert scan_totals(path, 2)["malformed"] == 1
    db.data[b"a007_NOUN x_VERB"] = kept
    assert verify_database(path, 2)["status"] == "verified"


def test_unreadable_lines_fail_the_run_unless_accepted(monkeypatch, tmp_path, store, capsys):
    files = dict(FILES)
    files["2-00002-of-00003.gz"] = FILES["2-00002-of-00003.gz"] + ["d_NOUN w_VERB\t1990,5\t1991,2,2",
                                                                    "e_NOUN w_VERB\t1990,x,1"]
    with pytest.raises(AcquisitionError, match="2 lines had counts that could not be read in full"):
        acquire(monkeypatch, tmp_path, files=files)
    doc = acquire.progress
    # the work is done and the counts that were read are verified: only the lines are at issue
    assert doc["state"] == "failed" and doc["verification"]["status"] == "verified"
    assert doc["verification"]["ledger"]["problem_lines"] == 2
    path = build_db_path(str(tmp_path / "run"), 4, "20200217", "eng")
    assert _unpack(store.dump(path)[b"d_NOUN w_VERB"]) == [(1991, 2, 2)]
    assert store.open(path).compactions == 1

    acquire(monkeypatch, tmp_path, files=files, allow_unparsed_lines=True)      # nothing left to fetch: accepts it
    assert acquire.progress["state"] == "done"
    assert "2 lines had content that could not be read" in capsys.readouterr().out


def test_shards_from_before_the_ledger_cannot_be_verified(monkeypatch, tmp_path, store, capsys):
    path = acquire(monkeypatch, tmp_path)
    db = store.open(path)
    db.data[PROCESSED_PREFIX + b"2-00002-of-00003.gz"] = b'{"entries": 1, "bytes": 10, "chunks": 1}'
    acquire(monkeypatch, tmp_path)                                               # a resumed run: not an error
    v = acquire.progress["verification"]
    assert v["status"] == "unverifiable" and v["ledger"]["unledgered"] == 1 and acquire.progress["state"] == "done"
    assert "Counts not verifiable: 1 of 3 files" in capsys.readouterr().out


def test_verification_can_be_turned_off(monkeypatch, tmp_path, store, capsys):
    acquire(monkeypatch, tmp_path, verify_counts=False)
    v = acquire.progress["verification"]
    assert v["status"] == "skipped" and v["ledger"]["match_total"] == EXPECTED_MATCH and "database" not in v
    assert "Count verification skipped" in capsys.readouterr().out


def test_a_run_with_a_failed_file_neither_compacts_nor_verifies(monkeypatch, tmp_path, store):
    import requests
    stub = tmp_path / "run"
    monkeypatch.setattr(core_mod, "discover_files",
                        lambda n, release, corpus: ("http://x/", [f"http://x/{f}" for f in FILES]))
    def dl(url, session=None):
        if "00001" in url:
            raise requests.ConnectionError("down")
        return _Resp(gzip.compress("\n".join(FILES[url.rsplit("/", 1)[-1]]).encode()))
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", dl)
    with pytest.raises(AcquisitionError, match="1 of 3 files failed"):
        download_and_ingest_to_rocksdb(ngram_size=4, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(stub), spool_dir=str(tmp_path), workers=2,
                                       ngram_type="tagged", progress_path=str(stub / "progress.json"))
    doc = read_progress(stub / "progress.json")
    db = store.open(build_db_path(str(stub), 4, "20200217", "eng"))
    assert db.compactions == 0 and "verification" not in doc and not doc["database"]["compacted"]


def test_a_failed_compaction_fails_the_run(monkeypatch, tmp_path, store):
    def boom(self): raise RuntimeError("no space left")
    monkeypatch.setattr(fake_rocks.DB, "compact_all", boom)
    with pytest.raises(AcquisitionError, match="compaction failed"):
        acquire(monkeypatch, tmp_path)
    assert acquire.progress["state"] == "failed" and not acquire.progress["database"]["compacted"]


def test_unknown_spool_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="spool_format"):
        download_and_ingest_to_rocksdb(ngram_size=4, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(tmp_path), spool_format="parquet")
    assert not list(tmp_path.iterdir())


def test_the_progress_document_carries_what_the_screen_shows(monkeypatch, tmp_path, store):
    """Sizes from the listing, the plan of steps with their times, what the
    run will check, and the ledger's running totals."""
    from ngramprep.ngram_acquire.io.fetch import FileListing
    payloads = {name: gzip.compress("\n".join(lines).encode()) for name, lines in FILES.items()}
    stub = tmp_path / "run"
    monkeypatch.setattr(core_mod, "discover_files", lambda n, release, corpus: (
        "http://x/", FileListing([f"http://x/{f}" for f in FILES], {f: len(p) for f, p in payloads.items()})))
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _Resp(payloads[url.rsplit("/", 1)[-1]]))
    download_and_ingest_to_rocksdb(ngram_size=4, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                                   spool_dir=str(tmp_path), workers=2, ngram_type="tagged", combined_multigrams=COMBINE,
                                   progress_path=str(stub / "progress.json"))
    doc = read_progress(stub / "progress.json")
    assert doc["steps"] == ["ingesting", "compacting", "verifying"] and doc["phase"] == "verifying"
    assert all(doc["phase_times"][step]["finished"] for step in doc["steps"])
    assert doc["checks"] == {"verify_counts": True, "allow_unparsed_lines": False}
    assert doc["bytes_total"] == doc["bytes_done"] == sum(len(p) for p in payloads.values())
    assert doc["in_flight"] == {"lines": 0, "bytes": 0} and "failed_files" not in doc
    assert doc["counters"]["empty_files"] == 0 and doc["counters"]["lines"] == sum(len(v) for v in FILES.values())
    assert (doc["corpus"]["match_total"], doc["corpus"]["volume_total"]) == (EXPECTED_MATCH, EXPECTED_VOLUME)

    # resumed with nothing left to fetch and an archive asked for: four steps, and the ledger is the earlier run's
    monkeypatch.setattr(core_mod, "_archive_database", lambda *a, **k: None)
    download_and_ingest_to_rocksdb(ngram_size=4, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(stub),
                                   spool_dir=str(tmp_path), workers=2, ngram_type="tagged", combined_multigrams=COMBINE,
                                   progress_path=str(stub / "progress.json"), archive_path_stub=str(tmp_path / "arch"))
    doc = read_progress(stub / "progress.json")
    assert doc["steps"] == ["ingesting", "compacting", "verifying", "archiving"] and doc["state"] == "done"
    assert doc["files_skipped"] == 3 and doc["corpus"]["match_total"] == EXPECTED_MATCH and doc["bytes_total"] is None

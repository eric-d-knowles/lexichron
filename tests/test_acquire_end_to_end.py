"""process_files end to end: thread pool, stubbed downloads, fake DB."""
import gzip
import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from ngramprep.ngram_acquire import worker as worker_mod
from ngramprep.ngram_acquire.executor import process_files
from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate


class _Resp:
    def __init__(self, payload):
        self.raw = io.BytesIO(payload); self.headers = {}
    def close(self): pass


@pytest.fixture
def fake_store(monkeypatch):
    """The in-memory stand-in for rocks_shim, for the SST writer and the
    database alike."""
    import fake_rocks
    from ngramprep.common_db import api
    fake_rocks.reset()
    monkeypatch.setattr(api, "rs", fake_rocks)
    return fake_rocks


def _unpack(blob):
    n = len(blob) // 8; v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


@pytest.mark.parametrize("spool_format", ["sst", "chunk"])
def test_process_files_ingests_marks_and_sums(monkeypatch, tmp_path, fake_store, spool_format):
    files = {
        "http://x/1-00000-of-00003.gz": [f"a{i}_NOUN\t1990,{i},1" for i in range(700)],
        "http://x/1-00001-of-00003.gz": [f"b{i}_NOUN\t1990,{i},1" for i in range(300)]
                                        + ["shared_NOUN\t1990,5,1"],
        "http://x/1-00002-of-00003.gz": ["shared_NOUN\t1990,7,2", "drop me\t1990,1,1"],
    }
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _Resp(gzip.compress("\n".join(files[url]).encode())))

    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, written, batches, nbytes = process_files(
        list(files), ThreadPoolExecutor, 2, db,
        filter_pred=make_ngram_type_predicate("tagged"),
        spool_dir=str(tmp_path), chunk_entries=250, spool_format=spool_format,
    )

    assert bad == [] and len(ok) == 3
    assert written == 700 + 301 + 1                      # entries written, pre-merge
    # cross-shard duplicate summed by merge
    assert _unpack(db.data[b"shared_NOUN"]) == [(1990, 12, 3)]
    assert b"drop me" not in db.data                     # filtered out
    # every shard marked, with what it contributed
    names = ["1-00000-of-00003.gz", "1-00001-of-00003.gz", "1-00002-of-00003.gz"]
    markers = {k: json.loads(v) for k, v in db.data.items() if k.startswith(PROCESSED_PREFIX)}
    assert set(markers) == {PROCESSED_PREFIX + name.encode() for name in names}
    assert [markers[PROCESSED_PREFIX + n.encode()]["match_total"] for n in names] == [sum(range(700)), sum(range(300)) + 5, 7]
    assert [markers[PROCESSED_PREFIX + n.encode()]["volume_total"] for n in names] == [700, 301, 2]
    assert all(m["problem_lines"] == 0 and m["reordered"] == 0 for m in markers.values())
    if spool_format == "sst":
        # one ingest call per shard, carrying one data file (its parts merged) and the marker
        assert db.ingests == 3 and db.ingested_files == 6 and batches == 3
        assert db.flushes == 0
    else:
        assert batches == 3 + 2 + 1                      # 700->3 chunks, 301->2, 1->1
        assert db.flushes == 3                           # one flush per shard, after its marker
    # spool cleaned up
    assert not [p for p in tmp_path.rglob("*") if p.suffix in (".chunk", ".part", ".sst")]
    assert nbytes > 0


def test_failed_shard_is_reported_not_marked(monkeypatch, tmp_path, fake_store):
    def dl(url, session=None):
        if "00001" in url:
            import requests
            raise requests.ConnectionError("down")
        return _Resp(gzip.compress(b"ok_NOUN\t1990,1,1"))
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", dl)
    monkeypatch.setattr(worker_mod.time, "sleep", lambda s: None) if hasattr(worker_mod, "time") else None

    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, written, _, _ = process_files(
        ["http://x/1-00000-of-00002.gz", "http://x/1-00001-of-00002.gz"],
        ThreadPoolExecutor, 2, db, spool_dir=str(tmp_path),
    )
    assert len(ok) == 1 and bad == ["NETWORK_ERROR: 1-00001-of-00002.gz"]
    assert written == 1
    assert PROCESSED_PREFIX + b"1-00000-of-00002.gz" in db.data
    assert PROCESSED_PREFIX + b"1-00001-of-00002.gz" not in db.data


def test_sequence_list_is_refused_where_it_cannot_apply_before_anything_runs(tmp_path):
    """A 1-gram line holds one token, and a sequence longer than the n-grams
    never fits in one line, so such lists can never match: the entry point
    says so at once, before discovering files or creating the database (no
    network, nothing on disk)."""
    import pytest
    from ngramprep.ngram_acquire.core import download_and_ingest_to_rocksdb
    with pytest.raises(ValueError, match="1-grams"):
        download_and_ingest_to_rocksdb(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(tmp_path), combined_multigrams={"human being"})
    with pytest.raises(ValueError, match="two or more words"):
        download_and_ingest_to_rocksdb(ngram_size=2, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(tmp_path), combined_multigrams={"human-being"})
    with pytest.raises(ValueError, match="3 words, which cannot occur in a 2-gram"):
        download_and_ingest_to_rocksdb(ngram_size=2, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(tmp_path), combined_multigrams={"new york city", "new york"})
    with pytest.raises(ValueError, match="3 words, which cannot occur in a 2-gram"):
        download_and_ingest_to_rocksdb(ngram_size=2, repo_release_id="20200217", repo_corpus_id="eng",
                                       db_path_stub=str(tmp_path), combined_multigrams_case={"New York City"})
    assert not list(tmp_path.iterdir())


def test_progress_is_republished_while_files_are_still_being_read(monkeypatch, tmp_path, fake_store):
    """A wave of large files can take minutes with none finishing. The parent
    republishes the progress document meanwhile, so what the workers report
    of files in flight reaches the screen and the run does not look dead."""
    import threading
    from ngramprep.ngram_acquire import executor as executor_mod
    from ngramprep.ngram_acquire.progress import ProgressReporter, read_progress
    monkeypatch.setattr(executor_mod, "_REFRESH_EVERY_S", 0.05)
    release = threading.Event()
    spool = tmp_path / "spool"

    class Slow(_Resp):
        """The whole file is there, but the worker is held up before reading it."""
        def __init__(self, payload):
            super().__init__(payload)
            real = self.raw
            class Raw:
                def read(self, n=-1):
                    release.wait(10)
                    return real.read(n)
                def readinto(self, b):
                    release.wait(10)
                    return real.readinto(b)
                def readable(self): return True
                def tell(self): return real.tell()
                def close(self): pass
            self.raw = Raw()

    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: Slow(gzip.compress(b"a_NOUN\t1990,1,1")))
    p = tmp_path / "progress.json"
    progress = ProgressReporter(p, stage="acquire", min_interval_s=0)
    refreshes = []
    real_refresh = progress.refresh

    def refresh():
        refreshes.append(read_progress(p)["phases"])
        real_refresh()
        if len(refreshes) >= 3:
            release.set()                                   # let the worker finish
    progress.refresh = refresh

    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, *_ = process_files(["http://x/1-00000-of-00001.gz"], ThreadPoolExecutor, 1, db,
                                spool_dir=str(spool), progress=progress)
    assert len(ok) == 1 and bad == [] and len(refreshes) >= 3
    assert refreshes[-1]["parsing"] == 1                    # the worker's marker was seen while it was still reading

"""process_files end to end: thread pool, stubbed downloads, fake DB."""
import gzip
import io
import struct
from concurrent.futures import ThreadPoolExecutor

from ngramprep.ngram_acquire import worker as worker_mod
from ngramprep.ngram_acquire.executor import process_files
from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate


class _Resp:
    def __init__(self, payload):
        self.raw = io.BytesIO(payload); self.headers = {}
    def close(self): pass


class _Batch:
    def __init__(self, db): self.db = db; self.ops = []
    def __enter__(self): return self
    def __exit__(self, *exc):
        if exc[0] is None:
            for op, k, v in self.ops:
                self.db._apply(op, k, v)
        return False
    def put(self, k, v): self.ops.append(("put", k, v))
    def merge(self, k, v): self.ops.append(("merge", k, v))


class FakeDB:
    """Applies put/merge like the packed24 operator would, and tracks flushes."""
    def __init__(self):
        self.data = {}; self.flushes = 0; self.flush_log = []
    def _apply(self, op, k, v):
        if op == "put" or k not in self.data:
            self.data[k] = v
        else:
            self.data[k] = worker_mod.merge_packed_records(self.data[k], v)
    def write_batch(self, disable_wal=False, sync=False): return _Batch(self)
    def finalize_bulk(self):
        self.flushes += 1
        self.flush_log.append(sorted(k for k in self.data if k.startswith(PROCESSED_PREFIX)))


def _unpack(blob):
    n = len(blob) // 8; v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


def test_process_files_streams_chunks_marks_and_flushes(monkeypatch, tmp_path):
    files = {
        "http://x/1-00000-of-00003.gz": [f"a{i}_NOUN\t1990,{i},1" for i in range(700)],
        "http://x/1-00001-of-00003.gz": [f"b{i}_NOUN\t1990,{i},1" for i in range(300)]
                                        + ["shared_NOUN\t1990,5,1"],
        "http://x/1-00002-of-00003.gz": ["shared_NOUN\t1990,7,2", "drop me\t1990,1,1"],
    }
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _Resp(gzip.compress("\n".join(files[url]).encode())))

    db = FakeDB()
    ok, bad, written, batches, nbytes = process_files(
        list(files), ThreadPoolExecutor, 2, db,
        filter_pred=make_ngram_type_predicate("tagged"),
        spool_dir=str(tmp_path), chunk_entries=250,
    )

    assert bad == [] and len(ok) == 3
    assert written == 700 + 301 + 1                      # entries written, pre-merge
    assert batches == 3 + 2 + 1                          # 700->3 chunks, 301->2, 1->1
    # cross-shard duplicate summed by merge
    assert _unpack(db.data[b"shared_NOUN"]) == [(1990, 12, 3)]
    assert b"drop me" not in db.data                     # filtered out
    # every shard marked, one flush per shard, marker present at its flush
    markers = {k for k in db.data if k.startswith(PROCESSED_PREFIX)}
    assert markers == {PROCESSED_PREFIX + name.encode() for name in
                       ["1-00000-of-00003.gz", "1-00001-of-00003.gz", "1-00002-of-00003.gz"]}
    assert db.flushes == 3
    assert [len(m) for m in db.flush_log] == [1, 2, 3]
    # spool cleaned up
    assert not list(tmp_path.rglob("*.chunk")) and not list(tmp_path.rglob("*.part"))
    assert nbytes > 0


def test_failed_shard_is_reported_not_marked(monkeypatch, tmp_path):
    def dl(url, session=None):
        if "00001" in url:
            import requests
            raise requests.ConnectionError("down")
        return _Resp(gzip.compress(b"ok_NOUN\t1990,1,1"))
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", dl)
    monkeypatch.setattr(worker_mod.time, "sleep", lambda s: None) if hasattr(worker_mod, "time") else None

    db = FakeDB()
    ok, bad, written, _, _ = process_files(
        ["http://x/1-00000-of-00002.gz", "http://x/1-00001-of-00002.gz"],
        ThreadPoolExecutor, 2, db, spool_dir=str(tmp_path),
    )
    assert len(ok) == 1 and bad == ["NETWORK_ERROR: 1-00001-of-00002.gz"]
    assert written == 1
    assert PROCESSED_PREFIX + b"1-00000-of-00002.gz" in db.data
    assert PROCESSED_PREFIX + b"1-00001-of-00002.gz" not in db.data

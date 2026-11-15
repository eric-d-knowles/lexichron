"""ChunkIngestor: streams chunk files into the DB, marks the shard, persists."""
import struct

from ngramprep.ngram_acquire.batch_writer import ChunkIngestor
from ngramprep.ngram_acquire.db.metadata import processed_key, processed_stats
from ngramprep.ngram_acquire.spool import ChunkWriter
from ngramprep.ngram_acquire.worker import merge_packed_records


class FakeBatch:
    def __init__(self, store): self.store = store; self.ops = []
    def __enter__(self): return self
    def __exit__(self, *exc):
        if exc[0] is None:
            for op, k, v in self.ops:
                self.store.setdefault(k, []).append((op, v))
        return False
    def put(self, k, v): self.ops.append(("put", k, v))
    def merge(self, k, v): self.ops.append(("merge", k, v))


class FakeDB:
    def __init__(self): self.store = {}; self.finalize_calls = 0
    def write_batch(self, disable_wal=False, sync=False): return FakeBatch(self.store)
    def finalize_bulk(self): self.finalize_calls += 1


def _chunks(tmp_path, tag, entries, per_chunk):
    w = ChunkWriter(tmp_path, tag, per_chunk, merge_packed_records)
    for k, v in entries:
        w.add(k, v)
    return w.finish()


def test_ingest_file_merges_marks_and_persists(tmp_path):
    db = FakeDB()
    ing = ChunkIngestor(db)
    paths = _chunks(tmp_path, "f1", [("a", b"x"), ("b", b"y"), ("c", b"z")], per_chunk=2)
    assert len(paths) == 2

    written = ing.ingest_file("f1.gz", paths, uncompressed_bytes=12345)

    assert written == 3
    assert db.store[b"a"] == [("merge", b"x")]
    assert db.store[b"c"] == [("merge", b"z")]
    # the resume marker records what the file contributed
    (op, marker), = db.store[processed_key("f1.gz")]
    assert op == "put"
    db.get = lambda k: marker if k == processed_key("f1.gz") else None
    assert processed_stats(db, "f1.gz") == {"entries": 3, "bytes": 12345, "chunks": 2}
    assert processed_stats(db, "other.gz") is None
    db.get = lambda k: b"1"
    assert processed_stats(db, "old.gz") == {}         # marker from an older version
    assert db.finalize_calls == 1          # once per shard, after all its chunks
    assert ing.get_stats() == (3, 2)       # entries, chunks
    assert not list(tmp_path.iterdir())    # chunks removed after ingest


def test_each_file_persists_separately(tmp_path):
    db = FakeDB()
    ing = ChunkIngestor(db)
    for i in range(3):
        ing.ingest_file(f"f{i}.gz", _chunks(tmp_path, f"f{i}", [(f"k{i}", b"v")], 10))
    assert db.finalize_calls == 3
    assert ing.files_completed == 3


def test_discard_removes_chunks(tmp_path):
    paths = _chunks(tmp_path, "f", [("a", b"x")], 10)
    ChunkIngestor.discard(paths)
    assert not list(tmp_path.iterdir())


# ---- SST files: the marker rides in the same ingest call ----------------------

def _sst(tmp_path, tag, entries, monkeypatch):
    import fake_rocks
    from ngramprep.common_db import api
    from ngramprep.ngram_acquire.spool import SstChunkWriter
    monkeypatch.setattr(api, "rs", fake_rocks)
    w = SstChunkWriter(tmp_path, tag, 100, merge_packed_records)
    for k, v in entries:
        w.add(k, v)
    return w.finish()


def _val(*triplets):
    flat = [x for t in triplets for x in t]
    return struct.pack(f"<{len(flat)}Q", *flat)


def test_sst_shard_and_its_marker_arrive_in_one_ingest_call(tmp_path, monkeypatch):
    import fake_rocks
    db = fake_rocks.DB(str(tmp_path / "db"))
    calls = []
    real = db.ingest
    db.ingest = lambda paths, move=True, **kw: (calls.append([p.rsplit("/", 1)[-1] for p in paths]), real(paths, move=move))[1]
    ing = ChunkIngestor(db)
    paths = _sst(tmp_path, "f1", [(b"a", _val((1990, 1, 1))), (b"b", _val((1990, 2, 1)))], monkeypatch)

    written = ing.ingest_file("f1.gz", paths, uncompressed_bytes=99, entries=2,
                              extra={"match_total": 3, "volume_total": 2})

    assert written == 2 and ing.get_stats() == (2, 1)
    assert calls == [["f1.00000.sst", "f1.00000.sst.marker.sst"]]        # one call: records and marker together
    assert db.flushes == 0                                    # nothing went through a memtable
    assert db.data[b"a"] == _val((1990, 1, 1))
    assert processed_stats(db, "f1.gz") == {"entries": 2, "bytes": 99, "chunks": 1, "match_total": 3, "volume_total": 2}
    assert not list(tmp_path.glob("*.sst"))                   # moved into the database


def test_failed_sst_ingest_marks_nothing_and_leaves_no_marker_file(tmp_path, monkeypatch):
    import pytest
    import fake_rocks
    db = fake_rocks.DB(str(tmp_path / "db"))
    def boom(paths, move=True, **kw): raise RuntimeError("disk gone")
    db.ingest = boom
    ing = ChunkIngestor(db)
    paths = _sst(tmp_path, "f1", [(b"a", _val((1990, 1, 1)))], monkeypatch)
    with pytest.raises(RuntimeError, match="disk gone"):
        ing.ingest_file("f1.gz", paths, entries=1)
    assert db.data == {} and ing.files_completed == 0
    assert not list(tmp_path.glob("*.marker.sst"))            # the caller discards the data file


def test_shard_without_records_is_marked_through_a_batch(tmp_path):
    import fake_rocks
    db = fake_rocks.DB(str(tmp_path / "db"))
    ing = ChunkIngestor(db)
    assert ing.ingest_file("empty.gz", [], entries=0, extra={"match_total": 0, "volume_total": 0}) == 0
    assert processed_stats(db, "empty.gz")["entries"] == 0 and db.flushes == 1

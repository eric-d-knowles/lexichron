"""ChunkIngestor: streams chunk files into the DB, marks the shard, persists."""
import struct

from ngramprep.ngram_acquire.batch_writer import ChunkIngestor
from ngramprep.ngram_acquire.db.metadata import processed_key
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

    written = ing.ingest_file("f1.gz", paths)

    assert written == 3
    assert db.store[b"a"] == [("merge", b"x")]
    assert db.store[b"c"] == [("merge", b"z")]
    assert db.store[processed_key("f1.gz")] == [("put", b"1")]
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

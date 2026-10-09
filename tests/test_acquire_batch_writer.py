"""BatchWriter persists every batch, together with its resume markers."""
from ngramprep.ngram_acquire.batch_writer import BatchWriter
from ngramprep.ngram_acquire.db.metadata import processed_key


class FakeBatch:
    def __init__(self, store):
        self.store = store
        self.ops = []
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        if exc[0] is None:
            for op, k, v in self.ops:
                self.store.setdefault(k, []).append((op, v))
        return False
    def put(self, k, v):
        self.ops.append(("put", k, v))
    def merge(self, k, v):
        self.ops.append(("merge", k, v))


class FakeDB:
    """Records writes and whether they have been made durable."""
    def __init__(self):
        self.store = {}
        self.finalize_calls = 0
    def write_batch(self, disable_wal=False, sync=False):
        return FakeBatch(self.store)
    def finalize_bulk(self):
        self.finalize_calls += 1


def test_flush_merges_data_puts_markers_and_persists():
    db = FakeDB()
    bw = BatchWriter(db, max_entries=10, max_bytes=1 << 20)
    bw.add("f1.gz", {"a": b"x", "b": b"y"})
    bw.flush()

    assert db.store[b"a"] == [("merge", b"x")]
    assert db.store[b"b"] == [("merge", b"y")]
    assert db.store[processed_key("f1.gz")] == [("put", b"1")]
    assert db.finalize_calls == 1
    assert bw.get_stats() == (2, 1)


def test_each_flush_persists_separately():
    db = FakeDB()
    bw = BatchWriter(db, max_entries=1, max_bytes=1 << 20)
    for i in range(3):
        assert bw.add(f"f{i}.gz", {f"k{i}": b"v"}) is True
        bw.flush()
    assert db.finalize_calls == 3
    assert bw.get_stats() == (3, 3)


def test_empty_flush_is_a_noop():
    db = FakeDB()
    BatchWriter(db).flush()
    assert db.finalize_calls == 0

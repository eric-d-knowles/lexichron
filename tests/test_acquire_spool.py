import struct

from ngramprep.ngram_acquire.spool import ChunkWriter, read_chunk, chunk_entry_count
from ngramprep.ngram_acquire.worker import merge_packed_records


def v(*triplets):
    flat = [x for t in triplets for x in t]
    return struct.pack(f"<{len(flat)}Q", *flat)


def test_roundtrip_and_chunk_boundaries(tmp_path):
    w = ChunkWriter(tmp_path, "f", chunk_entries=2, merge_fn=merge_packed_records)
    for i in range(5):
        w.add(f"k{i}", v((1990, i, 1)))
    assert not list(tmp_path.glob("*.chunk"))        # nothing published yet
    assert len(list(tmp_path.glob("*.part"))) == 2   # 4 entries spilled, 1 pending
    paths = w.finish()
    assert len(paths) == 3 and all(p.endswith(".chunk") for p in paths)
    assert not list(tmp_path.glob("*.part"))
    assert [chunk_entry_count(p) for p in paths] == [2, 2, 1]
    got = {k: val for p in paths for k, val in read_chunk(p)}
    assert got == {f"k{i}".encode(): v((1990, i, 1)) for i in range(5)}
    assert w.entries_total == 5


def test_repeat_key_within_chunk_is_summed(tmp_path):
    w = ChunkWriter(tmp_path, "f", chunk_entries=100, merge_fn=merge_packed_records)
    w.add("k", v((1990, 500, 80)))
    w.add("k", v((1990, 20, 5)))
    (p,) = w.finish()
    assert dict(read_chunk(p)) == {b"k": v((1990, 520, 85))}


def test_discard_removes_parts(tmp_path):
    w = ChunkWriter(tmp_path, "f", chunk_entries=1, merge_fn=merge_packed_records)
    w.add("a", b"x"); w.add("b", b"y")
    assert len(list(tmp_path.glob("*.part"))) == 2
    w.discard()
    assert not list(tmp_path.iterdir())
    assert w.finish() == []


def test_read_chunk_rejects_garbage(tmp_path):
    bad = tmp_path / "x.chunk"
    bad.write_bytes(b"nope")
    import pytest
    with pytest.raises(ValueError):
        list(read_chunk(bad))

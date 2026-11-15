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


# ---- the acquisition checks -------------------------------------------------

def test_check_values_totals_and_finds_years_out_of_order():
    from ngramprep.ngram_acquire.spool import check_values
    values = [v((1990, 1, 1), (1991, 2, 2)),            # fine
              v((2000, 5, 1), (1990, 7, 2)),            # descending
              v((1990, 1, 1)),                          # one triplet: 1990 after 1990 of... the value before is not a step
              v((1990, 3, 1), (1990, 4, 1), (1995, 1, 1)),   # a year twice
              v((1800, 9, 9))]
    match, volume, bad = check_values(values)
    assert (match, volume) == (1 + 2 + 5 + 7 + 1 + 3 + 4 + 1 + 9, 1 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 9)
    assert bad == [1, 3]
    assert check_values([]) == (0, 0, [])
    # totals do not wrap: two values near the top of uint64 add up as integers do
    big = 2 ** 63
    assert check_values([v((1990, big, 0))])[0] == big


def test_check_values_refuses_what_is_not_triplets():
    import pytest
    from ngramprep.ngram_acquire.spool import check_values
    with pytest.raises(ValueError, match="24-byte"):
        check_values([v((1990, 1, 1)), b"12345"])
    with pytest.raises(ValueError, match="24-byte"):
        check_values([b""])


def test_checked_writer_keeps_totals_and_orders_years(tmp_path):
    from ngramprep.ngram_acquire.spool import CheckedChunkWriter
    w = CheckedChunkWriter(tmp_path, "t", 3, merge_packed_records)
    w.STAGE = 2                                       # several drains, one of them partial
    w.add("a", v((1990, 1, 1)))
    w.add(b"a", v((1990, 2, 1), (1991, 1, 1)))        # the same key as bytes: one key, summed
    w.add("b", v((2000, 5, 1), (1990, 7, 2)))         # out of order
    w.add("c", v((1990, 1, 1)))
    w.add("d", v((1990, 1, 1)))
    paths = w.finish()
    got = {k: val for p in paths for k, val in read_chunk(p)}
    assert got[b"a"] == v((1990, 3, 2), (1991, 1, 1))
    assert got[b"b"] == v((1990, 7, 2), (2000, 5, 1))
    assert (w.match_total, w.volume_total, w.reordered) == (1 + 3 + 12 + 1 + 1, 1 + 2 + 3 + 1 + 1, 1)
    assert w.entries_total == 4


def test_checked_writer_discard_forgets_what_was_staged(tmp_path):
    from ngramprep.ngram_acquire.spool import CheckedChunkWriter
    w = CheckedChunkWriter(tmp_path, "t", 10, merge_packed_records)
    w.add("a", v((1990, 1, 1)))
    w.discard()
    assert w.finish() == [] and not list(tmp_path.iterdir())

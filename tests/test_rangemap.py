"""The range-map engine: partition -> map in workers -> single merging writer,
with per-unit markers that make resume exact."""
import struct

import pytest

import fake_rocks
from ngramprep.common_db import api as dbapi
from ngramprep.ngram_acquire.progress import read_progress
from ngramprep.rangemap import RangeMapConfig, Mapper, run_rangemap
from ngramprep.rangemap.engine import RangeMapError
from ngramprep.rangemap.units import load_partition


def _packed(*triples):
    flat = [x for t in triples for x in t]
    return struct.pack(f"<{len(flat)}Q", *flat)


def _unpack(blob):
    n = len(blob) // 8
    v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


class LowerMapper(Mapper):
    """Lowercase the key; drop keys containing 'drop'; optionally blow up."""

    def __init__(self, poison=None):
        self.poison = poison

    def build(self):
        poison = self.poison

        def fn(key, value):
            if poison and poison in key:
                raise RuntimeError("poisoned record")
            if b"drop" in key:
                return None
            return key.lower(), value

        return fn


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def _make_source(path, n=500):
    db = fake_rocks.open(path, mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for i in range(n):
            # 'Word3' and 'word3' collapse to one key after lowercasing
            wb.put(f"Word{i}".encode(), _packed((1990, i, 1)))
            wb.put(f"word{i}".encode(), _packed((1990, 1, 1), (2000, 2, 2)))
            if i % 50 == 0:
                wb.put(f"drop{i}".encode(), _packed((1990, 9, 9)))
        wb.put(b"__processed__/marker", b"1")        # bookkeeping: must be skipped
    return db


def test_engine_maps_merges_marks_and_resumes(rocks, tmp_path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    _make_source(str(src))
    progress_path = tmp_path / "progress.json"
    from ngramprep.ngram_acquire.progress import ProgressReporter

    cfg = RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=8,
                         workers=2, smart_partitioning=False, chunk_entries=64,
                         spool_dir=tmp_path, compact_after=True,
                         progress=ProgressReporter(progress_path, stage="filter", min_interval_s=0,
                                                   unit_label="work units"))
    res = run_rangemap(cfg)

    out = fake_rocks.dump(dst)
    assert len(out) == 500                                # Word_i + word_i -> word_i; drops gone
    assert res.scanned == 500 * 2 + 10 and res.kept == 1000
    assert _unpack(out[b"word3"]) == [(1990, 4, 2), (2000, 2, 2)]   # merged across the two inputs
    assert res.units_total == 8 and res.units_done == 8 and res.units_skipped == 0 and not res.failed
    m = fake_rocks.markers(dst)
    assert len(m) == 8 and sum(d["scanned"] for d in m.values()) == 1010
    assert fake_rocks._REGISTRY[str(dst)].compactions == 1
    # progress document
    doc = read_progress(progress_path)
    assert doc["files_total"] == 8 and doc["files_done"] == 8 and doc["unit_label"] == "work units"
    assert doc["counters"] == {"scanned": 1010, "kept": 1000}
    assert doc["phase"] == "ingesting" and doc["state"] == "running"   # caller finishes it

    # resume: nothing to do, partition reused, output unchanged
    res2 = run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=99,
                                       workers=2, smart_partitioning=False, spool_dir=tmp_path))
    assert res2.units_skipped == 8 and res2.units_done == 0
    assert len(load_partition(fake_rocks._REGISTRY[str(dst)])) == 8    # not 99: the stored one wins
    assert fake_rocks.dump(dst) == out


def test_failed_unit_is_retried_on_rerun(rocks, tmp_path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    _make_source(str(src))
    base = RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(poison=b"word42"), num_units=8,
                          workers=2, smart_partitioning=False, spool_dir=tmp_path, compact_after=False)
    with pytest.raises(RangeMapError) as exc:
        run_rangemap(base)
    assert "1 of 8 work units failed" in str(exc.value)
    m = fake_rocks.markers(dst)
    assert len(m) == 7                                     # the poisoned unit has no marker
    partial = fake_rocks.dump(dst)

    # rerun with a healthy mapper: only the failed unit runs
    ok = RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=8, workers=2,
                        smart_partitioning=False, spool_dir=tmp_path, compact_after=False)
    res = run_rangemap(ok)
    assert res.units_skipped == 7 and res.units_done == 1
    full = fake_rocks.dump(dst)
    assert len(full) == 500 and set(partial) <= set(full)


def test_restart_wipes_destination(rocks, tmp_path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    _make_source(str(src), n=50)
    cfg = RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=4, workers=2,
                         smart_partitioning=False, spool_dir=tmp_path, compact_after=False)
    run_rangemap(cfg)
    # a second restart run creates a fresh destination and does all the work again
    before = fake_rocks._REGISTRY[str(dst)]
    cfg.mode = "restart"
    res = run_rangemap(cfg)
    assert res.units_done == 4 and res.units_skipped == 0
    assert fake_rocks._REGISTRY[str(dst)] is not before


def test_worker_scans_without_filling_the_cache(rocks, tmp_path):
    from ngramprep.rangemap.worker import map_unit
    src = tmp_path / "src.db"
    _make_source(str(src), n=10)
    chunks, scanned, kept, _ = map_unit(0, "u0", None, None, str(src), LowerMapper(), str(tmp_path), 100)
    assert scanned == 21 and kept == 20 and len(chunks) == 1
    assert fake_rocks._REGISTRY[str(src)].last_iterator_opts == (False, 2 << 20)


def test_units_from_sst_metadata_balance_by_size():
    from ngramprep.rangemap.units import units_from_live_files
    # 64 files of 1 MB over a..., then 64 files of 1 MB at the far end; one
    # 64 MB file of bookkeeping keys that must be ignored
    files = [(f"a{i:03d}".encode(), f"a{i:03d}z".encode(), 1 << 20, 1) for i in range(64)]
    files += [(f"z{i:03d}".encode(), f"z{i:03d}z".encode(), 1 << 20, 1) for i in range(64)]
    files.append((b"__processed__/a", b"__processed__/z", 64 << 20, 1))
    units = units_from_live_files(files, 8)
    assert len(units) == 8 and units[0].start_key is None and units[-1].end_key is None
    cuts = [u.end_key for u in units[:-1]]
    assert cuts == sorted(cuts) and len(set(cuts)) == len(cuts)
    # half the data is in the a-files, half in the z-files: the middle cut is at the boundary
    assert sum(c < b"b" for c in cuts) == 3 and cuts[3] >= b"z"
    # too few files -> None, so the caller samples instead
    assert units_from_live_files(files[:5], 8) is None


def test_plan_units_uses_sst_metadata_when_the_store_offers_it(rocks, tmp_path):
    from ngramprep.rangemap.units import plan_units
    src = tmp_path / "src.db"
    _make_source(str(src), n=5000)                     # 10k keys -> 100 fake files
    units = plan_units(src, 16, smart=True, log=lambda *_: None)
    assert len(units) == 16
    # every unit covers about the same amount of data (bytes, not keys: the
    # source's values are 24 bytes in one half of the key space, 48 in the other)
    data = {k: v for k, v in fake_rocks.dump(src).items() if not k.startswith(b"__")}
    sizes = []
    for u in units:
        lo = u.start_key or b""
        sizes.append(sum(len(k) + len(v) for k, v in data.items()
                         if k >= lo and (u.end_key is None or k < u.end_key)))
    assert sum(sizes) == sum(len(k) + len(v) for k, v in data.items())
    assert max(sizes) <= 1.5 * min(sizes), sizes


def test_plan_units_falls_back_to_sampling_without_live_files(rocks, tmp_path, monkeypatch):
    from ngramprep.rangemap import units as units_mod
    src = tmp_path / "src.db"
    _make_source(str(src), n=200)
    monkeypatch.setattr(fake_rocks.DB, "live_files", None)   # as if rocks-shim < 0.6
    called = {}
    monkeypatch.setattr(units_mod, "create_smart_work_units",
                        lambda **kw: called.update(kw) or units_mod.create_uniform_work_units(kw["num_units"]))
    units = units_mod.plan_units(src, 16, smart=True, log=lambda *_: None)
    assert len(units) == 16 and called["num_sampling_workers"] == 8   # capped pool

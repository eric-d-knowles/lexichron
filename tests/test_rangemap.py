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
    c = doc["counters"]
    assert c["scanned"] == 1010 and c["kept"] == 1000
    assert 500 <= c["inserted"] <= 1000                # records handed to the DB, after in-chunk merging
    # Entries reports records kept, not chunk-file entries: the latter depends
    # on how many duplicate output keys were merged inside a chunk, so it
    # would differ between partitions of the same source.
    assert doc["entries_written"] == 1000
    assert doc["phase"] == "ingesting" and doc["state"] == "running"   # caller finishes it

    # resume: nothing to do, partition reused, output unchanged
    res2 = run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=99,
                                       workers=2, smart_partitioning=False, spool_dir=tmp_path))
    assert res2.units_skipped == 8 and res2.units_done == 0
    assert len(load_partition(fake_rocks._REGISTRY[str(dst)])) == 8    # not 99: the stored one wins
    assert fake_rocks.dump(dst) == out
    # a run that died during its compaction is finished by a resume: the
    # destination is compacted again even though no unit was processed
    assert fake_rocks._REGISTRY[str(dst)].compactions == 2


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
    chunks, scanned, kept, _, _stats, _written, _produced, _ledger = map_unit(0, "u0", None, None, str(src), LowerMapper(), str(tmp_path), 100)
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


class ExplodeMapper(Mapper):
    """A pivot in miniature: one (ngram, packed years) -> one record per
    year, keyed year-first, lowercased. Duplicates across raw keys must be
    summed by the database."""

    def build(self):
        def fn(key, value):
            if b"drop" in key:
                return None
            out = []
            for i in range(0, len(value), 24):
                year, n, v = struct.unpack("<3Q", value[i:i + 24])
                out.append((struct.pack(">I", year) + key.lower(), value[i:i + 24]))
            return out
        return fn


def test_sst_output_with_one_to_many_mapper(rocks, tmp_path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    _make_source(str(src), n=200)                       # Word_i (1990) and word_i (1990, 2000)
    cfg = RangeMapConfig(src_db=src, dst_db=dst, mapper=ExplodeMapper(), num_units=8, workers=2,
                         smart_partitioning=False, chunk_entries=50, spool_dir=tmp_path,
                         compact_after=True, output="sst")
    res = run_rangemap(cfg)
    out = fake_rocks.dump(dst)
    # 200 words x 2 years, keyed year-first
    data = {k: v for k, v in out.items() if not k.startswith(b"__")}
    assert len(data) == 400
    assert _unpack(data[struct.pack(">I", 1990) + b"word7"]) == [(1990, 8, 2)]    # Word7 + word7 summed
    assert _unpack(data[struct.pack(">I", 2000) + b"word7"]) == [(2000, 2, 2)]
    assert res.kept == 400 and res.scanned == 404                 # 200 raw keys x 2 spellings kept; drops skipped
    assert res.produced == 600                                    # -> 1 + 2 year records each
    # each unit arrived as ONE data file (its parts merged by the worker), so
    # the destination has as many of them as units, not as chunks; the unit's
    # marker rides in the same ingest call as a second, one-key file, so that
    # records and marker are in the database together or not at all
    store = fake_rocks._REGISTRY[str(dst)]
    assert 0 < store.ingests <= len(fake_rocks.markers(dst)) and store.ingested_files == 2 * store.ingests
    assert not [p for p in tmp_path.rglob("*.sst*")]              # parts and merged files were consumed
    assert sum(d["entries"] for d in fake_rocks.markers(dst).values()) <= 600   # entries per unit, after merging


def test_underscore_tokens_are_data_not_bookkeeping():
    """Google's tables hold tokens made of underscores (OCR'd blanks such as
    "____X"); only the stages' own marker keys are skipped by a scan."""
    from ngramprep.rangemap.units import is_bookkeeping
    assert is_bookkeeping(b"__rangemap__/partition")
    assert is_bookkeeping(b"__rangemap__/unit/7")
    assert is_bookkeeping(b"__processed__/1-00003-of-00014.gz")
    assert not is_bookkeeping(b"__NOUN")
    assert not is_bookkeeping(b"____X the_DET cat_NOUN")
    assert not is_bookkeeping(b"_")


# ---- the ledger: what each unit read and emitted, and the checks built on it ----------------

def _source_totals(src):
    """Occurrences and volume counts of every data record of a source."""
    rows = [(k, v) for k, v in fake_rocks.dump(src).items()]
    return (sum(f for _k, v in rows for _y, f, _d in _unpack(v)), sum(d for _k, v in rows for _y, _f, d in _unpack(v)))


@pytest.mark.parametrize("output", ["sst", "chunks"])
def test_units_record_what_they_read_and_emitted_and_the_database_is_verified(rocks, tmp_path, output):
    from ngramprep.ngram_acquire.verify import ledger_totals, verify_database
    from ngramprep.rangemap import UNIT_MARKER_PREFIX
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _make_source(str(src))
    run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=8, workers=2,
                                smart_partitioning=False, chunk_entries=64, spool_dir=tmp_path, output=output))
    markers = fake_rocks.markers(dst)
    assert len(markers) == 8 and all("match_total" in m and "source_match" in m for m in markers.values())
    # everything read, kept or not, adds up to what the source holds: every key lies in exactly one unit
    read = (sum(m["source_match"] for m in markers.values()), sum(m["source_volume"] for m in markers.values()))
    assert read == _source_totals(src)
    # the 'drop' records were read and not emitted; the rest went out with their counts
    dropped = (10 * 9, 10 * 9)
    emitted = (sum(m["match_total"] for m in markers.values()), sum(m["volume_total"] for m in markers.values()))
    assert emitted == (read[0] - dropped[0], read[1] - dropped[1])
    assert all(m["reordered"] == 0 and m["source_disordered"] == 0 for m in markers.values())

    v = verify_database(str(dst), 2, marker_prefix=UNIT_MARKER_PREFIX)
    assert v["status"] == "verified" and (v["database"]["match_total"], v["database"]["volume_total"]) == emitted
    assert v["ledger"]["shards"] == 8 and v["ledger"]["source_match"] == read[0]
    store = fake_rocks._REGISTRY[str(dst)]
    if output == "sst":
        # a unit's records and its marker arrive in one ingest call: nothing is replayed, nothing flushed for it
        assert store.ingested_files == 2 * store.ingests and store.flushes <= 8 - store.ingests + 2

    # a unit taken in twice (its marker lost, the next run scans it again) is counted twice: the check finds it
    victim = next(k for k, m in markers.items() if m["entries"])
    lost = markers[victim]["match_total"]
    del store.data[victim.encode()]
    run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=8, workers=2,
                                smart_partitioning=False, chunk_entries=64, spool_dir=tmp_path, output=output))
    v = verify_database(str(dst), 2, marker_prefix=UNIT_MARKER_PREFIX)
    assert v["status"] == "mismatch" and v["database"]["match_total"] - v["ledger"]["match_total"] == lost


def test_emitted_values_are_put_in_year_order_and_source_disorder_is_counted(rocks, tmp_path):
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    db = fake_rocks.open(str(src), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"Fine", _packed((1990, 1, 1), (2000, 2, 2)))
        wb.put(b"Odd", _packed((2000, 5, 1), (1990, 7, 2)))          # years out of order in the source
    run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=LowerMapper(), num_units=1, workers=1,
                                smart_partitioning=False, spool_dir=tmp_path, output="sst"))
    assert _unpack(fake_rocks.dump(dst)[b"odd"]) == [(1990, 7, 2), (2000, 5, 1)]
    (marker,) = fake_rocks.markers(dst).values()
    assert marker["source_disordered"] == 1 and marker["reordered"] == 1 and marker["match_total"] == 15


def test_progress_is_republished_while_units_are_still_being_scanned(rocks, tmp_path, monkeypatch):
    """A unit of a large database takes minutes. The parent republishes the
    progress document meanwhile, with what the workers report of their units."""
    from ngramprep.ngram_acquire.progress import ProgressReporter
    from ngramprep.rangemap import engine as engine_mod, worker as worker_mod
    monkeypatch.setattr(engine_mod, "_REFRESH_EVERY_S", 0.05)
    monkeypatch.setattr(worker_mod, "_REPORT_EVERY_S", 0.0)
    monkeypatch.setattr(worker_mod, "_BATCH", 16)
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _make_source(str(src), n=200)

    class Slow(Mapper):
        def build(self):
            import time
            def fn(key, value):
                time.sleep(0.002)
                return key.lower(), value
            return fn

    progress = ProgressReporter(tmp_path / "p.json", stage="filter", min_interval_s=0, unit_label="work units")
    seen = []
    real = progress.refresh
    progress.refresh = lambda: (real(), seen.append(dict(read_progress(tmp_path / "p.json")["in_flight"])))[0]
    run_rangemap(RangeMapConfig(src_db=src, dst_db=dst, mapper=Slow(), num_units=2, workers=2,
                                smart_partitioning=False, spool_dir=tmp_path, output="sst", progress=progress))
    assert len(seen) >= 2 and max(x["lines"] for x in seen) > 0          # records scanned in units not yet finished
    assert read_progress(tmp_path / "p.json")["in_flight"] == {"lines": 0, "bytes": 0}


def test_worker_log_entries_name_the_unit_and_what_is_being_done(rocks, tmp_path, caplog, monkeypatch):
    import logging
    from ngramprep.rangemap import worker as worker_mod
    from ngramprep.rangemap.worker import map_unit
    monkeypatch.setattr(worker_mod, "_REPORT_EVERY_S", 0.0)
    monkeypatch.setattr(worker_mod, "_LOG_EVERY_S", 0.0)
    monkeypatch.setattr(worker_mod, "_BATCH", 64)
    src = tmp_path / "src.db"
    _make_source(str(src), n=100)
    with caplog.at_level(logging.INFO):
        map_unit(2, "unit_a_m", b"W", b"x", str(src), LowerMapper(), str(tmp_path), 1000, output="sst", units_total=8)
        with pytest.raises(RuntimeError):
            map_unit(3, "unit_bad", None, None, str(src), LowerMapper(poison=b"word42"), str(tmp_path), 1000)
    records = [r for r in caplog.records if r.name.startswith("rangemap.worker.")]
    text = [r.getMessage() for r in records]
    assert {r.name for r in records} == {"rangemap.worker.00"}                 # run in this process
    good = [m for m in text if m.startswith("unit unit_a_m (3 of 8): ")]
    assert good[0].startswith("unit unit_a_m (3 of 8): scanning keys b'W' to b'x'")
    assert any("still scanning after" in m and "records read" in m and "kept so far" in m for m in good)
    assert good[-1].startswith("unit unit_a_m (3 of 8): done in") and "in 1 SST file; waiting for the parent to ingest it" in good[-1]
    bad = [r for r in records if "unit_bad" in r.getMessage() and r.levelname == "ERROR"]
    assert len(bad) == 1 and "failed after" in bad[0].getMessage() and "poisoned record" in bad[0].getMessage()

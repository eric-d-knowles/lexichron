"""Work units (key ranges) for the range-map engine, and how they persist.

The partition is computed once per destination database and stored inside
it under ``PARTITION_KEY``, so a resumed run processes exactly the same
ranges and the per-unit markers (``UNIT_MARKER_PREFIX + unit_id``) stay
meaningful. A restart wipes the destination, and with it the partition.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ngramprep.common_db.api import open_db
from ngramprep.tracking import WorkUnit, create_smart_work_units, create_uniform_work_units
from ngramprep.tracking import PartitionCache, PartitionCacheKey
from ngramprep.tracking.partitioning import make_unit_id

__all__ = ["PARTITION_KEY", "UNIT_MARKER_PREFIX", "BOOKKEEPING_PREFIXES", "is_bookkeeping",
           "plan_units", "units_from_live_files", "load_partition", "store_partition", "unit_marker"]

# Bookkeeping keys the stages themselves write into a database. The test is
# exact: Google's tables contain real tokens made of underscores ("__NOUN",
# "____X" for OCR'd blanks), which must be scanned like any other record.
BOOKKEEPING_PREFIXES = (b"__rangemap__/", b"__processed__/")


def is_bookkeeping(key: bytes) -> bool:
    """True for a stage's own marker/partition key, false for data."""
    return key.startswith(BOOKKEEPING_PREFIXES)
PARTITION_KEY = b"__rangemap__/partition"   # JSON list of [start_hex|null, end_hex|null]
UNIT_MARKER_PREFIX = b"__rangemap__/unit/"  # + unit_id -> figures for a finished unit


def unit_marker(unit: WorkUnit) -> bytes:
    return UNIT_MARKER_PREFIX + unit.unit_id.encode("utf-8")


def plan_units(
    src_db: Path,
    num_units: int,
    *,
    smart: bool = True,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    num_sampling_workers: Optional[int] = None,
    samples_per_worker: int = 50_000,
    read_profile: str = "scan:packed24",
    log=print,
) -> List[WorkUnit]:
    """Cut the source key space into ``num_units`` ranges.

    With ``smart`` the ranges are balanced by data volume. The source's SST
    file metadata (key range and size of every file, from RocksDB itself) is
    used when it is available and fine enough, which costs milliseconds and
    no memory; otherwise the density is estimated by sampling, a parallel
    pre-scan of the source whose result is cached under ``cache_dir``.
    Without ``smart`` the ranges split the byte space uniformly.
    """
    if num_units < 4 or not smart:
        units = create_uniform_work_units(max(1, num_units))
        log(f"Created {len(units)} uniform work units")
        return units

    units = _units_from_sst_metadata(src_db, num_units, read_profile, log)
    if units:
        log(f"Created {len(units)} work units balanced by SST file sizes")
        return units
    log("Too few SST files to partition by their boundaries; sampling the source instead")

    # A few sampling processes are enough: each one is a full scan of its
    # share of the keys, and each holds a RocksDB handle with its block cache.
    if num_sampling_workers is None:
        num_sampling_workers = min(num_units, 8)
    cache = PartitionCache(cache_dir) if cache_dir else None
    key = PartitionCacheKey(db_path=str(src_db), num_units=num_units,
                            samples_per_worker=samples_per_worker,
                            num_sampling_workers=num_sampling_workers)
    if cache and use_cache and cache.exists(key):
        try:
            units = cache.load(key)
            if units:
                log(f"Loaded {len(units)} cached work units")
                return units
        except Exception as exc:                  # pragma: no cover - cache corruption
            log(f"Could not load cached partition ({exc}); resampling")

    log(f"Sampling the source to create {num_units} density-balanced work units...")
    try:
        units = create_smart_work_units(db_path=src_db, num_units=num_units,
                                        num_sampling_workers=num_sampling_workers,
                                        samples_per_worker=samples_per_worker,
                                        read_profile=read_profile)
    except Exception as exc:
        log(f"Density sampling failed ({exc}); using uniform work units")
        return create_uniform_work_units(num_units)
    if cache:
        try:
            cache.save(key, units)
        except Exception as exc:                  # pragma: no cover
            log(f"Could not cache the partition: {exc}")
    log(f"Created {len(units)} density-balanced work units")
    return units


def _units_from_sst_metadata(src_db, num_units, read_profile, log) -> Optional[List[WorkUnit]]:
    try:
        with open_db(src_db, mode="r", profile=read_profile) as db:
            if not callable(getattr(db, "live_files", None)):   # rocks-shim < 0.6
                return None
            files = db.live_files()
    except Exception as exc:                            # pragma: no cover - depends on the store
        log(f"Could not read SST metadata ({exc}); sampling instead")
        return None
    return units_from_live_files(files, num_units)


def units_from_live_files(files, num_units: int, *, min_files: int = 16) -> Optional[List[WorkUnit]]:
    """Balanced key ranges from SST metadata ``[(smallest_key, largest_key, size, level), ...]``.

    Each file's bytes are treated as sitting half at its first key and half
    at its last; the cuts fall where the cumulative size crosses each
    ``k/num_units`` of the total, so every unit covers about the same amount
    of data. Returns None when there are too few files for the cuts to mean
    anything (then sampling is the better estimate).
    """
    points = []                                         # (key, weight, is_file_start)
    for lo, hi, size, _level in files:
        lo, hi = bytes(lo), bytes(hi)
        if is_bookkeeping(lo) and is_bookkeeping(hi):
            continue                                    # a file of bookkeeping keys only
        half = max(int(size), 1) / 2.0
        points.append((lo, half, True))
        points.append((hi, half, False))
    if len(points) < 2 * min_files:
        return None
    points.sort(key=lambda p: (p[0], not p[2]))
    total = sum(p[1] for p in points)

    # A cut is placed at the start of a file, never at its last key, so that
    # units line up with file boundaries wherever the files do not overlap.
    cuts: List[bytes] = []
    acc = 0.0
    k = 1
    for i, (key, weight, is_start) in enumerate(points):
        acc += weight
        while k < num_units and acc >= total * k / num_units:
            cut = key if is_start else (points[i + 1][0] if i + 1 < len(points) else None)
            if cut is not None and (not cuts or cut > cuts[-1]):
                cuts.append(cut)
            k += 1
    # The first cut must leave something before it and no cut may be the very
    # first key, or the leading unit would be empty.
    first_key = points[0][0]
    cuts = [c for c in cuts if c > first_key]
    if not cuts:
        return None

    bounds = [None] + cuts + [None]
    return [WorkUnit(unit_id=make_unit_id(a, b), start_key=a, end_key=b)
            for a, b in zip(bounds[:-1], bounds[1:])]


def _hex(b: Optional[bytes]) -> Optional[str]:
    return None if b is None else b.hex()


def _unhex(h: Optional[str]) -> Optional[bytes]:
    return None if h is None else bytes.fromhex(h)


def store_partition(db, units: List[WorkUnit]) -> None:
    doc = [[u.unit_id, _hex(u.start_key), _hex(u.end_key)] for u in units]
    with db.write_batch(disable_wal=False, sync=True) as wb:
        wb.put(PARTITION_KEY, json.dumps(doc).encode("utf-8"))
    db.finalize_bulk()


def load_partition(db) -> Optional[List[WorkUnit]]:
    raw = db.get(PARTITION_KEY)
    if raw is None:
        return None
    doc = json.loads(raw.decode("utf-8"))
    return [WorkUnit(unit_id=uid, start_key=_unhex(s), end_key=_unhex(e)) for uid, s, e in doc]

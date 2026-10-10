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

from ngramprep.tracking import WorkUnit, create_smart_work_units, create_uniform_work_units
from ngramprep.tracking import PartitionCache, PartitionCacheKey

__all__ = ["PARTITION_KEY", "UNIT_MARKER_PREFIX", "METADATA_PREFIX",
           "plan_units", "load_partition", "store_partition", "unit_marker"]

METADATA_PREFIX = b"__"                     # keys that are bookkeeping, not data
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
    read_profile: str = "read:packed24",
    log=print,
) -> List[WorkUnit]:
    """Cut the source key space into ``num_units`` ranges.

    With ``smart`` the ranges are balanced by density sampling (a parallel
    pre-scan of the source), and the result is cached under ``cache_dir``
    keyed by the source path and sampling parameters, since it depends only
    on the source. Otherwise the ranges split the byte space uniformly.
    """
    if num_units < 4 or not smart:
        units = create_uniform_work_units(max(1, num_units))
        log(f"Created {len(units)} uniform work units")
        return units

    if num_sampling_workers is None:
        num_sampling_workers = min(num_units, 40)
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

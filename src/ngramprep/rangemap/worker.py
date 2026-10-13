"""Worker side of the range-map engine: scan one key range, map, spool."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Tuple

from ngramprep.common_db.api import open_db, range_scan
from ngramprep.ngram_acquire.spool import ChunkWriter
from ngramprep.ngram_acquire.worker import merge_packed_records

from .units import METADATA_PREFIX

__all__ = ["map_unit", "UnitResult", "set_inherited_mapper"]

# The parent sets this before forking the pool, so workers inherit the mapper
# (which may carry a large whitelist) instead of unpickling it per unit.
_INHERITED_MAPPER = None


def set_inherited_mapper(mapper) -> None:
    global _INHERITED_MAPPER
    _INHERITED_MAPPER = mapper

# (chunk_paths, scanned, kept, seconds)
UnitResult = Tuple[list, int, int, float]

try:
    import setproctitle as _setproctitle
except ImportError:                      # pragma: no cover
    _setproctitle = None


def map_unit(
    unit_index: int,
    unit_id: str,
    start_key: Optional[bytes],
    end_key: Optional[bytes],
    src_db: str,
    mapper,                       # Mapper (picklable); built in this process
    spool_dir: str,
    chunk_entries: int,
    read_profile: str = "scan:packed24",
    merge_fn: Callable[[bytes, bytes], bytes] = merge_packed_records,
) -> UnitResult:
    """Scan ``[start_key, end_key)`` of ``src_db``, apply ``mapper`` to every
    data record, and spool the results to chunk files under ``spool_dir``.

    Returns ``(chunk_paths, scanned, kept, seconds)``. Keys starting with
    ``METADATA_PREFIX`` are bookkeeping and are skipped. Records that map to
    the same output key within a chunk are merged with ``merge_fn``; across
    chunks and units the database's merge operator does the same.
    """
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"rmap:worker[{unit_index:04d}]")
        except Exception:
            pass
    t0 = time.perf_counter()
    tag = f"{unit_index:05d}_{unit_id}"
    try:
        open(os.path.join(spool_dir, f"{tag}.started"), "w").close()
    except OSError:
        pass

    fn = (mapper if mapper is not None else _INHERITED_MAPPER).build()
    writer = ChunkWriter(spool_dir, tag, chunk_entries, merge_fn)
    scanned = kept = 0
    try:
        with open_db(src_db, mode="r", profile=read_profile) as db:
            # One pass over the range: don't fill the block cache (rocks-shim >= 0.4),
            # and read ahead, since this is a strictly sequential scan.
            for key, value in range_scan(db, start_key or b"", end_key,
                                         fill_cache=False, readahead_size=2 << 20):
                if key.startswith(METADATA_PREFIX):
                    continue
                scanned += 1
                out = fn(key, value)
                if out is None:
                    continue
                writer.add(out[0], out[1])
                kept += 1
        chunk_paths = writer.finish()
    except BaseException:
        writer.discard()
        raise
    try:
        os.replace(os.path.join(spool_dir, f"{tag}.started"), os.path.join(spool_dir, f"{tag}.parsed"))
    except OSError:
        pass
    return chunk_paths, scanned, kept, time.perf_counter() - t0

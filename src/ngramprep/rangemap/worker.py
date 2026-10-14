"""Worker side of the range-map engine: scan one key range, map, spool."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Tuple

from ngramprep.common_db.api import open_db, range_scan
from ngramprep.ngram_acquire.spool import ChunkWriter, SstChunkWriter
from ngramprep.ngram_acquire.worker import merge_packed_records

from .units import is_bookkeeping

__all__ = ["map_unit", "UnitResult", "set_inherited_mapper"]

# The parent sets this before forking the pool, so workers inherit the mapper
# (which may carry a large whitelist) instead of unpickling it per unit.
_INHERITED_MAPPER = None


def set_inherited_mapper(mapper) -> None:
    global _INHERITED_MAPPER
    _INHERITED_MAPPER = mapper

# (chunk_paths, scanned, kept, seconds, stats, written, produced): kept is
# the number of source records the mapper kept, produced the number of
# records it made from them (more than kept for a one-to-many mapper: the
# pivot makes one per year), written the number of entries in the chunk
# files (fewer than produced: duplicates merge within a chunk); stats are
# the mapper's own counters for the unit (a dict, empty if it keeps none)
UnitResult = Tuple[list, int, int, float, dict, int, int]

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
    output: str = "chunks",
    sst_compression: str = "zstd",
) -> UnitResult:
    """Scan ``[start_key, end_key)`` of ``src_db``, apply ``mapper`` to every
    data record, and spool the results to chunk files under ``spool_dir``.

    Returns a :data:`UnitResult`. A stage's own bookkeeping keys are
    bookkeeping and are skipped. The mapper returns one ``(key, value)``, a
    list of them, or None. Records that map to the same output key within a
    chunk are merged with ``merge_fn``; across chunks and units the
    database's merge operator does the same. With ``output="sst"`` the chunks
    are sorted SST files of merge operands that the parent ingests as files.
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
    if output == "sst":
        writer = SstChunkWriter(spool_dir, tag, chunk_entries, merge_fn, compression=sst_compression)
    else:
        writer = ChunkWriter(spool_dir, tag, chunk_entries, merge_fn)
    scanned = kept = produced = 0
    try:
        with open_db(src_db, mode="r", profile=read_profile) as db:
            # One pass over the range: don't fill the block cache (rocks-shim >= 0.4),
            # and read ahead, since this is a strictly sequential scan.
            for key, value in range_scan(db, start_key or b"", end_key,
                                         fill_cache=False, readahead_size=2 << 20):
                if is_bookkeeping(key):
                    continue
                scanned += 1
                out = fn(key, value)
                if out is None:
                    continue
                kept += 1
                if type(out) is tuple:
                    writer.add(out[0], out[1])
                    produced += 1
                else:                                 # one-to-many mapper
                    for k, v in out:
                        writer.add(k, v)
                    produced += len(out)
        chunk_paths = writer.finish()
    except BaseException:
        writer.discard()
        raise
    try:
        os.replace(os.path.join(spool_dir, f"{tag}.started"), os.path.join(spool_dir, f"{tag}.parsed"))
    except OSError:
        pass
    stats_fn = getattr(fn, "stats_dict", None)
    stats = {k: int(v) for k, v in stats_fn().items()} if stats_fn else {}
    return chunk_paths, scanned, kept, time.perf_counter() - t0, stats, writer.entries_total, produced

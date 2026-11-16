"""Worker side of the range-map engine: scan one key range, map, spool."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, Tuple

from ngramprep.common_db.api import open_db, range_scan
from ngramprep.ngram_acquire.spool import CheckedChunkWriter, CheckedSstChunkWriter, check_values
from ngramprep.ngram_acquire.worker import _worker_slot, merge_packed_records

from .units import is_bookkeeping

__all__ = ["map_unit", "UnitResult", "set_inherited_mapper"]

# The parent sets this before forking the pool, so workers inherit the mapper
# (which may carry a large whitelist) instead of unpickling it per unit.
_INHERITED_MAPPER = None


def set_inherited_mapper(mapper) -> None:
    global _INHERITED_MAPPER
    _INHERITED_MAPPER = mapper

# (chunk_paths, scanned, kept, seconds, stats, written, produced, ledger):
# kept is the number of source records the mapper kept, produced the number
# of records it made from them (more than kept for a one-to-many mapper: the
# pivot makes one per year), written the number of entries in the chunk
# files (fewer than produced: duplicates merge within a chunk); stats are
# the mapper's own counters for the unit (a dict, empty if it keeps none).
# ledger holds the counts the run is verified against: ``match_total`` and
# ``volume_total`` of everything the unit emitted, ``source_match`` and
# ``source_volume`` of everything it read, ``reordered`` (emitted values
# whose years had to be put in order) and ``source_disordered`` (source
# values whose years were not in order).
UnitResult = Tuple[list, int, int, float, dict, int, int, dict]

try:
    import setproctitle as _setproctitle
except ImportError:                      # pragma: no cover
    _setproctitle = None

# Source values are totalled, and the clock looked at, once per this many records.
_BATCH = 2048
# How often a worker says how far it is (its marker file, for the progress
# display) and writes a log entry while a unit is still being scanned.
_REPORT_EVERY_S = 3.0
_LOG_EVERY_S = 60.0


def _report(marker_path: str, scanned: int) -> None:
    """Write "<records scanned> 0" into the worker's marker file (the format
    an acquisition worker uses for lines and bytes). Advisory."""
    try:
        with open(marker_path, "w") as fh:
            fh.write(f"{int(scanned)} 0")
    except OSError:
        pass


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
    units_total: Optional[int] = None,
) -> UnitResult:
    """Scan ``[start_key, end_key)`` of ``src_db``, apply ``mapper`` to every
    data record, and spool the results to chunk files under ``spool_dir``.

    Returns a :data:`UnitResult`. A stage's own bookkeeping keys are
    bookkeeping and are skipped. The mapper returns one ``(key, value)``, a
    list of them, or None. Records that map to the same output key within a
    chunk are merged with ``merge_fn``; across chunks and units the
    database's merge operator does the same. With ``output="sst"`` the chunks
    are sorted SST files of merge operands that the parent ingests as files.

    The unit keeps two sets of totals, both sums of frequencies and of
    document counts over all years. Of what it *read*: every source record,
    kept or not, so that all units together must add up to what the source
    holds (a key range skipped or read twice by the partition shows there).
    Of what it *emitted*: every record handed to the spool, before repeated
    keys are summed, so that the finished destination must hold exactly the
    units' emitted totals. Emitted values are also checked to have their
    years in ascending order, as the merge operator assumes, and put in
    order if not (see ngram_acquire.spool._Checked).
    """
    slot = _worker_slot()
    log = logging.getLogger(f"rangemap.worker.{slot:02d}")
    which = f"unit {unit_id} ({unit_index + 1}" + (f" of {units_total}" if units_total else "") + ")"
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"rmap:worker[{slot:02d}] {unit_id}")
        except Exception:
            pass
    t0 = time.perf_counter()
    tag = f"{unit_index:05d}_{unit_id}"
    marker = os.path.join(spool_dir, f"{tag}.started")
    _report(marker, 0)
    log.info("%s: scanning keys %r to %r", which, (start_key or b"")[:40], (end_key[:40] if end_key else "the end"))

    fn = (mapper if mapper is not None else _INHERITED_MAPPER).build()
    if output == "sst":
        writer = CheckedSstChunkWriter(spool_dir, tag, chunk_entries, merge_fn, compression=sst_compression)
    else:
        writer = CheckedChunkWriter(spool_dir, tag, chunk_entries, merge_fn)
    scanned = kept = produced = 0
    source_match = source_volume = source_disordered = 0
    batch = []                                            # source values not yet totalled
    last_report = last_logged = t0

    def tally() -> None:
        nonlocal source_match, source_volume, source_disordered, batch
        match, volume, bad = check_values(batch)
        source_match += match
        source_volume += volume
        source_disordered += len(bad)
        batch = []

    try:
        with open_db(src_db, mode="r", profile=read_profile) as db:
            # One pass over the range: don't fill the block cache (rocks-shim >= 0.4),
            # and read ahead, since this is a strictly sequential scan.
            for key, value in range_scan(db, start_key or b"", end_key,
                                         fill_cache=False, readahead_size=2 << 20):
                if is_bookkeeping(key):
                    continue
                scanned += 1
                batch.append(bytes(value))
                if len(batch) >= _BATCH:
                    tally()
                    now = time.perf_counter()
                    if now - last_report >= _REPORT_EVERY_S:
                        last_report = now
                        _report(marker, scanned)
                        if now - last_logged >= _LOG_EVERY_S:
                            last_logged = now
                            log.info("%s: still scanning after %.0fs: %s records read, %s kept so far",
                                     which, now - t0, f"{scanned:,}", f"{kept:,}")
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
        if batch:
            tally()
        _report(marker, scanned)                      # the whole unit, while it waits to be ingested
        scan_s = time.perf_counter() - t0
        chunk_paths = writer.finish()
    except BaseException as exc:
        writer.discard()
        log.error("%s: failed after %s records with %s: %s. Nothing of this unit was ingested",
                  which, f"{scanned:,}", type(exc).__name__, exc)
        raise
    try:
        os.replace(marker, os.path.join(spool_dir, f"{tag}.parsed"))
    except OSError:
        pass
    stats_fn = getattr(fn, "stats_dict", None)
    stats = {k: int(v) for k, v in stats_fn().items()} if stats_fn else {}
    ledger = {"match_total": writer.match_total, "volume_total": writer.volume_total, "reordered": writer.reordered,
              "source_match": source_match, "source_volume": source_volume, "source_disordered": source_disordered}
    seconds = time.perf_counter() - t0
    if chunk_paths:
        log.info("%s: done in %.0fs (scan %.0fs, spool %.0fs): read %s records, kept %s, made %s as %s entries "
                 "in %d %s file%s; waiting for the parent to ingest it", which, seconds, scan_s, seconds - scan_s,
                 f"{scanned:,}", f"{kept:,}", f"{produced:,}", f"{writer.entries_total:,}", len(chunk_paths),
                 "SST" if output == "sst" else "chunk", "" if len(chunk_paths) == 1 else "s")
    else:
        log.info("%s: done in %.0fs: read %s records, kept none; waiting for the parent to mark it processed",
                 which, seconds, f"{scanned:,}")
    if writer.reordered or source_disordered:
        log.warning("%s: %s source values had years out of order; %s emitted values were put in order",
                    which, f"{source_disordered:,}", f"{writer.reordered:,}")
    return chunk_paths, scanned, kept, seconds, stats, writer.entries_total, produced, ledger

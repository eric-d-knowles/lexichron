"""Parent side of the range-map engine."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ngramprep.common_db.api import open_db
from ngramprep.ngram_acquire.batch_writer import ChunkIngestor
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
from ngramprep.tracking import WorkUnit

from .units import UNIT_MARKER_PREFIX, load_partition, plan_units, store_partition, unit_marker
from .worker import map_unit, set_inherited_mapper

logger = logging.getLogger(__name__)

# How often the parent republishes progress while it waits for a unit to finish.
_REFRESH_EVERY_S = 5.0

__all__ = ["Mapper", "RangeMapConfig", "RangeMapResult", "RangeMapError", "run_rangemap"]


class Mapper:
    """What to do with each record. Instances are pickled to the workers, so
    they must hold only plain data; :meth:`build` runs in the worker and
    returns ``fn(key, value) -> (key, value) | None`` (None drops the record).
    Expensive resources (a lemmatizer, say) belong in ``build``."""

    def build(self) -> Callable[[bytes, bytes], Optional[tuple]]:
        raise NotImplementedError


class RangeMapError(RuntimeError):
    """Some units failed; the destination holds the completed ones."""


class InputsChanged(RangeMapError):
    """A resume was asked to finish a destination started with other inputs."""


# What the run that started a destination was given (a JSON object), so a
# resume can refuse to finish it with different inputs.
INPUTS_KEY = b"__rangemap__/inputs"


def _check_inputs(dst, inputs: Optional[Dict[str, Any]], started: bool, log) -> None:
    """Compare ``inputs`` with what the destination was started with, and
    record them. A resume finishes a destination by processing only the
    units not yet done: with different inputs (a rebuilt vocabulary, say)
    those units would be made differently from the ones already in it, and
    the result would be a mixture that every count check passes. Refused.
    A destination from before inputs were recorded cannot be checked; that
    is said, and the inputs are recorded from now on."""
    if inputs is None:
        return
    current = json.loads(json.dumps(inputs, sort_keys=True, default=str))
    raw = dst.get(INPUTS_KEY)
    if raw is not None and started:
        before = json.loads(raw.decode("utf-8"))
        changed = sorted(k for k in set(before) | set(current) if before.get(k) != current.get(k))
        if changed:
            raise InputsChanged(
                "the destination was started with different inputs ("
                + "; ".join(f"{k}: was {before.get(k)!r}, now {current.get(k)!r}" for k in changed)
                + "). Resuming would finish it with units made two ways. Run again with mode restart "
                  "to rebuild it from the start.")
        return
    if raw is None and started:
        log("The destination was started before its inputs were recorded, so a resume cannot check that they "
            "are unchanged; recording them now")
        logger.warning("Resuming a destination whose inputs were not recorded: not checked that they are unchanged")
    with dst.write_batch(disable_wal=False, sync=True) as wb:
        wb.put(INPUTS_KEY, json.dumps(current, sort_keys=True).encode("utf-8"))


@dataclass
class RangeMapConfig:
    src_db: Path
    dst_db: Path
    mapper: Mapper
    num_units: int = 256
    workers: Optional[int] = None            # default: cpu_count - 1
    mode: str = "resume"                     # "resume" | "restart"
    smart_partitioning: bool = True
    samples_per_worker: int = 50_000
    use_cached_partition: bool = True
    partition_cache_dir: Optional[Path] = None
    read_profile: str = "scan:packed24"
    write_profile: str = "write:packed24"
    chunk_entries: int = 200_000
    spool_dir: Optional[Path] = None         # default: system temp (node-local in Slurm)
    compact_after: bool = True
    output: str = "chunks"                   # "chunks" (replayed by the parent) | "sst" (ingested as files)
    sst_compression: str = "zstd"
    progress: Optional[ProgressReporter] = None
    log: Callable[[str], None] = print
    # What determines the output (a JSON object): recorded in the destination
    # and compared on a resume (see _check_inputs). None: not checked.
    inputs: Optional[Dict[str, Any]] = None


@dataclass
class RangeMapResult:
    units_total: int
    units_done: int                          # in this run
    units_skipped: int                       # already in the destination
    scanned: int                             # records read in this run
    kept: int                                # source records the mapper kept
    produced: int                            # records made from them (one per year for the pivot)
    seconds: float
    failed: List[str] = field(default_factory=list)


def _marker_stats(db, units: List[WorkUnit]) -> Dict[str, dict]:
    """unit_id -> figures for units already in the destination."""
    done = {}
    for u in units:
        raw = db.get(unit_marker(u))
        if raw is None:
            continue
        try:
            done[u.unit_id] = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            done[u.unit_id] = {}
    return done


def run_rangemap(cfg: RangeMapConfig) -> RangeMapResult:
    """Run the engine to completion (or until units fail). See module docs."""
    t_start = time.perf_counter()
    log = cfg.log
    progress = cfg.progress or ProgressReporter(None, stage="rangemap", unit_label="work units")
    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)

    if cfg.mode not in ("resume", "restart"):
        raise ValueError(f"mode must be 'resume' or 'restart', got {cfg.mode!r}")
    if cfg.mode == "restart" and Path(cfg.dst_db).exists():
        log(f"Restart: removing {cfg.dst_db}")
        if not safe_db_cleanup(cfg.dst_db):
            raise RuntimeError(f"could not remove {cfg.dst_db}")
    Path(cfg.dst_db).parent.mkdir(parents=True, exist_ok=True)

    spool_ctx = tempfile.TemporaryDirectory(prefix="rangemap_spool_",
                                            dir=str(cfg.spool_dir) if cfg.spool_dir else None)
    spool_path = spool_ctx.name
    progress.set_spool_dir(spool_path)
    failed: List[str] = []
    scanned_total = kept_total = produced_total = 0

    try:
        with open_db(cfg.dst_db, mode="rw", profile=cfg.write_profile, create_if_missing=True) as dst:
            # ---- partition: reuse the destination's own, else plan and store it
            units = load_partition(dst)
            _check_inputs(dst, cfg.inputs, started=units is not None, log=log)
            if units is None:
                with progress.long_phase("partitioning"):
                    units = plan_units(Path(cfg.src_db), cfg.num_units, smart=cfg.smart_partitioning,
                                       cache_dir=cfg.partition_cache_dir, use_cache=cfg.use_cached_partition,
                                       samples_per_worker=cfg.samples_per_worker,
                                       read_profile=cfg.read_profile, log=log)
                    store_partition(dst, units)
            else:
                log(f"Using the destination's existing partition ({len(units)} work units)")

            done = _marker_stats(dst, units)
            todo = [u for u in units if u.unit_id not in done]
            prior = {"files": len(done),
                     "entries": sum(int(d.get("kept", d.get("entries", 0))) for d in done.values()),
                     "bytes": 0, "unsized": 0}
            progress.set_totals(files_total=len(units), files_skipped=len(done), prior=prior)
            log(f"{len(units)} work units: {len(done)} already done, {len(todo)} to process "
                f"with {workers} workers")
            logger.info("Work units: %d to scan, %d already in the destination, %d workers; each spools its unit as %s "
                        "under %s", len(todo), len(done), workers,
                        "one sorted SST file" if cfg.output == "sst" else "chunk files", spool_path)

            ingestor = ChunkIngestor(dst, marker_prefix=UNIT_MARKER_PREFIX)
            index_of = {u.unit_id: i for i, u in enumerate(units)}

            set_inherited_mapper(cfg.mapper)        # forked workers inherit it
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
                it = iter(todo)
                futures: Dict[object, WorkUnit] = {}
                max_in_flight = max(1, workers * 2)

                def submit_next(n: int = 1) -> None:
                    for _ in range(n):
                        try:
                            u = next(it)
                        except StopIteration:
                            return
                        progress.file_started(u.unit_id)
                        fut = pool.submit(map_unit, index_of[u.unit_id], u.unit_id, u.start_key, u.end_key,
                                          str(cfg.src_db), None, spool_path, cfg.chunk_entries,
                                          cfg.read_profile, output=cfg.output,
                                          sst_compression=cfg.sst_compression, units_total=len(units))
                        futures[fut] = u

                submit_next(max_in_flight)
                while futures:
                    # Wake every few seconds even if no unit has finished, to
                    # republish progress: what the workers report of units
                    # still being scanned, and a fresh "updated" time.
                    ready, _ = wait(futures.keys(), timeout=_REFRESH_EVERY_S, return_when=FIRST_COMPLETED)
                    if not ready:
                        progress.refresh()
                        continue
                    for fut in ready:
                        progress.file_parsed(futures[fut].unit_id)
                    for fut in ready:
                        u = futures.pop(fut)
                        tag = f"{index_of[u.unit_id]:05d}_{u.unit_id}"
                        chunk_paths: List[str] = []
                        try:
                            chunk_paths, scanned, kept, secs, stats, entries, produced, ledger = fut.result()
                            progress.file_ingesting(u.unit_id)
                            # The marker records what the unit read and what it
                            # emitted: the run's verification adds these up and
                            # compares them with the source and the destination.
                            written = ingestor.ingest_file(u.unit_id, chunk_paths, entries=entries,
                                                           extra={"scanned": scanned, "kept": kept, **stats, **ledger})
                            scanned_total += scanned
                            kept_total += kept
                            produced_total += produced
                            # Four distinct figures: records scanned from the source,
                            # source records the mapper kept (the same whatever the
                            # partition), records it produced from them (more for a
                            # one-to-many mapper such as the pivot), and records handed
                            # to the database (`written`: fewer than produced by the
                            # duplicates merged inside a chunk). The mapper's own
                            # counters (per-rule token fates) ride along.
                            progress.file_done(u.unit_id, entries=kept, chunks=len(chunk_paths),
                                               uncompressed_bytes=0,
                                               counters={"scanned": scanned, "kept": kept, "produced": produced,
                                                         "inserted": written, **stats, **ledger})
                            logger.info("Unit %s: scanned %s, kept %s, produced %s (%s entries after in-chunk merging), %.1fs",
                                        u.unit_id, f"{scanned:,}", f"{kept:,}", f"{produced:,}", f"{written:,}", secs)
                        except Exception as exc:
                            ChunkIngestor.discard(chunk_paths)
                            msg = f"{u.unit_id}: {exc}"
                            failed.append(msg)
                            logger.error("Unit %s was not ingested and is not marked (%s: %s); a re-run will scan it again",
                                         u.unit_id, type(exc).__name__, exc)
                            progress.file_failed(u.unit_id, msg)
                        finally:
                            for ext in (".started", ".parsed"):
                                try:
                                    os.remove(os.path.join(spool_path, tag + ext))
                                except OSError:
                                    pass
                    submit_next(len(ready))

            if cfg.compact_after and not failed:
                # Also when nothing was left to do: a resume after a run that
                # died during its compaction must still compact (cheap when
                # the database is already compacted).
                with progress.long_phase("compacting"):
                    log("Compacting the destination database...")
                    logger.info("Compacting: folding the units' SST files into one sorted database and summing the "
                                "counts of records that arrived from more than one unit")
                    t_compact = time.perf_counter()
                    dst.compact_all()
                    logger.info("Compaction completed in %.0fs", time.perf_counter() - t_compact)
            if not failed:
                # bookkeeping keys: one marker per unit plus the stored partition
                progress.set_database(dst, compacted=bool(cfg.compact_after),
                                      bookkeeping=len(units) + 1 + (1 if cfg.inputs is not None else 0))   # markers, partition, inputs
    finally:
        spool_ctx.cleanup()

    result = RangeMapResult(units_total=len(units), units_done=len(todo) - len(failed),
                            units_skipped=len(done), scanned=scanned_total, kept=kept_total, produced=produced_total,
                            seconds=time.perf_counter() - t_start, failed=failed)
    logger.info("Work units and compaction finished in %.0fs: %d ingested, %d failed; %s records read, %s kept, %s made",
                result.seconds, result.units_done, len(failed), f"{scanned_total:,}", f"{kept_total:,}",
                f"{produced_total:,}")
    if failed:
        logger.error("Stopping without compacting: %d work units failed. The destination holds the other units, "
                     "each marked; a re-run scans only the missing ones", len(failed))
        progress.finish("failed", f"{len(failed)} work unit(s) failed")
        raise RangeMapError(f"{len(failed)} of {len(todo)} work units failed; completed units are "
                            f"in the database, re-run to retry the rest. First failure: {failed[0]}")
    return result

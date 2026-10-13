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
from typing import Callable, Dict, List, Optional

from ngramprep.common_db.api import open_db
from ngramprep.ngram_acquire.batch_writer import ChunkIngestor
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
from ngramprep.tracking import WorkUnit

from .units import UNIT_MARKER_PREFIX, load_partition, plan_units, store_partition, unit_marker
from .worker import map_unit, set_inherited_mapper

logger = logging.getLogger(__name__)

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
    progress: Optional[ProgressReporter] = None
    log: Callable[[str], None] = print


@dataclass
class RangeMapResult:
    units_total: int
    units_done: int                          # in this run
    units_skipped: int                       # already in the destination
    scanned: int                             # records read in this run
    kept: int                                # records written (before merging)
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
    scanned_total = kept_total = 0

    try:
        with open_db(cfg.dst_db, mode="rw", profile=cfg.write_profile, create_if_missing=True) as dst:
            # ---- partition: reuse the destination's own, else plan and store it
            units = load_partition(dst)
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
                                          cfg.read_profile)
                        futures[fut] = u

                submit_next(max_in_flight)
                while futures:
                    ready, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for fut in ready:
                        progress.file_parsed(futures[fut].unit_id)
                    for fut in ready:
                        u = futures.pop(fut)
                        tag = f"{index_of[u.unit_id]:05d}_{u.unit_id}"
                        chunk_paths: List[str] = []
                        try:
                            chunk_paths, scanned, kept, secs = fut.result()
                            progress.file_ingesting(u.unit_id)
                            written = ingestor.ingest_file(u.unit_id, chunk_paths,
                                                           extra={"scanned": scanned, "kept": kept})
                            scanned_total += scanned
                            kept_total += kept
                            # Progress reports the records kept, which is the same
                            # whatever the partition; `written` (entries in the chunk
                            # files) varies with how many duplicates met inside a chunk.
                            progress.file_done(u.unit_id, entries=kept, chunks=len(chunk_paths),
                                               uncompressed_bytes=0,
                                               counters={"scanned": scanned, "kept": kept})
                            logger.info("Unit %s: scanned %s, kept %s (%s entries after in-chunk merging), %.1fs",
                                        u.unit_id, f"{scanned:,}", f"{kept:,}", f"{written:,}", secs)
                        except Exception as exc:
                            ChunkIngestor.discard(chunk_paths)
                            msg = f"{u.unit_id}: {exc}"
                            failed.append(msg)
                            logger.error("Unit failed: %s", msg)
                            progress.file_failed(u.unit_id, msg)
                        finally:
                            for ext in (".started", ".parsed"):
                                try:
                                    os.remove(os.path.join(spool_path, tag + ext))
                                except OSError:
                                    pass
                    submit_next(len(ready))

            if cfg.compact_after and not failed and todo:
                with progress.long_phase("compacting"):
                    log("Compacting the destination database...")
                    dst.compact_all()
    finally:
        spool_ctx.cleanup()

    result = RangeMapResult(units_total=len(units), units_done=len(todo) - len(failed),
                            units_skipped=len(done), scanned=scanned_total, kept=kept_total,
                            seconds=time.perf_counter() - t_start, failed=failed)
    if failed:
        progress.finish("failed", f"{len(failed)} work unit(s) failed")
        raise RangeMapError(f"{len(failed)} of {len(todo)} work units failed; completed units are "
                            f"in the database, re-run to retry the rest. First failure: {failed[0]}")
    return result

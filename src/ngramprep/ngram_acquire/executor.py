"""Concurrent file processing and ingestion for ngram pipeline."""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from tqdm import tqdm
import rocks_shim as rs

from ngramprep.ngram_acquire.worker import process_and_ingest_file, DEFAULT_CHUNK_ENTRIES
from ngramprep.ngram_acquire.batch_writer import ChunkIngestor
from ngramprep.ngram_acquire.progress import ProgressReporter

logger = logging.getLogger(__name__)

__all__ = ["process_files"]

# How often the parent republishes progress while it waits for a file to finish.
_REFRESH_EVERY_S = 5.0


def process_files(
        urls: Iterable[str],
        executor_class: Type,
        workers: int,
        db: rs.DB,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        combined_multigrams: Optional[set] = None,
        combined_multigrams_case: Optional[set] = None,
        spool_dir: Optional[str] = None,
        chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
        write_batch_size: Optional[int] = None,  # accepted for compatibility; chunk_entries governs
        progress: Optional[ProgressReporter] = None,
        spool_format: str = "sst",
        worker_fn: Optional[Callable] = None,
        worker_kwargs: Optional[dict] = None,
) -> Tuple[List[str], List[str], int, int, int]:
    """
    Process files concurrently and ingest results into RocksDB.

    Workers download, parse and filter shards, spooling the packed entries to
    files (see :mod:`ngramprep.ngram_acquire.spool`); as each shard
    completes, this process takes its files into the database and marks the
    shard processed, the marker becoming durable with the records (see
    :class:`~ngramprep.ngram_acquire.batch_writer.ChunkIngestor`). With the
    default ``spool_format="sst"`` a worker hands over one sorted SST file
    and the parent only adopts it; with ``"chunk"`` the parent replays the
    records. Memory use is bounded by one chunk per worker plus one in this
    process, independent of shard size.

    Args:
        urls: File URLs to process
        executor_class: ThreadPoolExecutor or ProcessPoolExecutor
        workers: Number of concurrent workers
        db: RocksDB database handle
        filter_pred: Optional predicate to filter ngrams by text
        combined_multigrams: Optional word sequences to combine, matched whatever their capitalization
        combined_multigrams_case: Sequences that must match as written
        spool_dir: Directory for chunk files (default: a fresh temp directory)
        chunk_entries: Entries per chunk file
        write_batch_size: Ignored (kept so older callers do not break)
        progress: Optional reporter updated as files start, finish or fail
        spool_format: How workers hand shards over, ``"sst"`` or ``"chunk"``
        worker_fn: What a worker does with one file, in place of the Google
            Books n-gram worker: called as ``worker_fn(path_or_url, number,
            log_file_path, spool_dir=, chunk_entries=, spool_format=,
            files_total=, **worker_kwargs)`` and returning the same tuple as
            :func:`~ngramprep.ngram_acquire.worker.process_and_ingest_file`.
            Everything after the worker (ingest with the marker, the ledger)
            is the same for every reader.
        worker_kwargs: Extra keyword arguments for ``worker_fn``

    Returns:
        Tuple of (success_msgs, failure_msgs, total_entries_written,
                  write_batches, total_uncompressed_bytes)
    """
    import tempfile

    log_file_path = _get_log_file_path()
    if log_file_path:
        logger.debug("Log file path for workers: %s", log_file_path)

    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_uncompressed_bytes = 0

    ingestor = ChunkIngestor(db)
    total = len(urls) if hasattr(urls, "__len__") else None

    # Each run gets its own spool directory, removed at the end.
    if spool_dir:
        os.makedirs(spool_dir, exist_ok=True)
    spool_ctx = tempfile.TemporaryDirectory(prefix="ngram_acquire_spool_", dir=spool_dir)
    spool_path = spool_ctx.name
    logger.info("Starting %d workers; each spools its file as %s under %s (at most %s entries in memory)",
                workers, "one sorted SST file" if spool_format == "sst" else "chunk files", spool_path,
                f"{chunk_entries:,}")
    if progress:
        progress.set_spool_dir(spool_path)

    try:
        with tqdm(
            total=total,
            desc="Files Processed:",
            unit="files",
            ncols=100,
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            kwargs = {"max_workers": workers}
            if issubclass(executor_class, ProcessPoolExecutor):
                # Workers never touch the database; the parent does all writes.
                # Fork is chosen explicitly so behaviour does not change with the
                # interpreter's default start method (spawn from Python 3.14),
                # and so each worker inherits the already-imported modules
                # instead of re-importing them.
                kwargs["mp_context"] = mp.get_context("fork")
                logger.debug(
                    "Using multiprocessing start method: %s",
                    kwargs["mp_context"].get_start_method()
                )

            with executor_class(**kwargs) as executor:
                it = iter(urls)
                futures: Dict[object, Tuple[str, int]] = {}
                max_in_flight = max(1, workers * 2)
                idx = 0

                def submit_next(n: int = 1) -> None:
                    nonlocal idx
                    for _ in range(n):
                        try:
                            url = next(it)
                        except StopIteration:
                            return
                        idx += 1
                        if progress:
                            progress.file_started(PurePosixPath(url).name)
                        if worker_fn is not None:
                            fut = executor.submit(
                                worker_fn, url, idx, log_file_path,
                                spool_dir=spool_path, chunk_entries=chunk_entries,
                                spool_format=spool_format, files_total=total,
                                **(worker_kwargs or {}),
                            )
                        else:
                            fut = executor.submit(
                                process_and_ingest_file,
                                url,
                                idx,
                                filter_pred,
                                log_file_path,
                                combined_multigrams=combined_multigrams,
                                combined_multigrams_case=combined_multigrams_case,
                                spool_dir=spool_path,
                                chunk_entries=chunk_entries,
                                spool_format=spool_format,
                                files_total=total,
                            )
                        futures[fut] = (url, idx)

                submit_next(max_in_flight)

                while futures:
                    # Wake every few seconds even if no file has finished: the
                    # progress document is then rewritten with what the workers
                    # report of the files they are still reading, and its
                    # "updated" time stays fresh through a wave of large files.
                    done, _ = wait(futures.keys(), timeout=_REFRESH_EVERY_S, return_when=FIRST_COMPLETED)
                    if not done:
                        if progress:
                            progress.refresh()
                        continue
                    if progress:
                        for fut in done:
                            progress.file_parsed(PurePosixPath(futures[fut][0]).name)

                    for fut in done:
                        url, file_idx = futures.pop(fut)
                        filename = PurePosixPath(url).name
                        chunk_paths: List[str] = []
                        try:
                            result_msg, chunk_paths, uncompressed_bytes, _entries, _lines, stats = fut.result()

                            if result_msg.startswith("SUCCESS"):
                                if progress:
                                    progress.file_ingesting(filename)
                                # The marker records what the shard contributed:
                                # the run's verification adds these up and
                                # compares them with what the database holds.
                                written = ingestor.ingest_file(filename, chunk_paths,
                                                               uncompressed_bytes=uncompressed_bytes,
                                                               extra={"lines": _lines, **stats},
                                                               entries=_entries)
                                success_msgs.append(result_msg)
                                total_uncompressed_bytes += uncompressed_bytes
                                if progress:
                                    # lines: raw data lines parsed; inserted: records
                                    # handed to the database (lines that produced an
                                    # entry, after in-chunk merging)
                                    progress.file_done(filename, entries=_entries,
                                                       chunks=len(chunk_paths),
                                                       uncompressed_bytes=uncompressed_bytes,
                                                       counters={"lines": _lines, "inserted": written,
                                                                 "empty_files": 1 if not _entries else 0,
                                                                 "match_total": stats.get("match_total", 0),
                                                                 "volume_total": stats.get("volume_total", 0),
                                                                 "problem_lines": stats.get("problem_lines", 0),
                                                                 "reordered": stats.get("reordered", 0)})
                            else:
                                ChunkIngestor.discard(chunk_paths)
                                failure_msgs.append(result_msg)
                                logger.error("Not ingested, not marked: %s (the worker's entries say why); "
                                             "a re-run will fetch it again", filename)
                                if progress:
                                    progress.file_failed(filename, result_msg)

                        except Exception as exc:
                            ChunkIngestor.discard(chunk_paths)
                            msg = f"ERROR: {filename} - {exc}"
                            failure_msgs.append(msg)
                            logger.error("Could not ingest %s: %s: %s. Its spool files were discarded and it is "
                                         "not marked; a re-run will fetch it again", filename, type(exc).__name__, exc)
                            if progress:
                                progress.file_failed(filename, msg)
                        finally:
                            for ext in (".started", ".parsed"):
                                try:
                                    os.remove(os.path.join(spool_path, f"{file_idx:05d}_{filename}{ext}"))
                                except OSError:
                                    pass
                            pbar.update(1)

                    submit_next(len(done))
    finally:
        spool_ctx.cleanup()

    total_entries_written, write_batches = ingestor.get_stats()
    return (
        success_msgs,
        failure_msgs,
        total_entries_written,
        write_batches,
        total_uncompressed_bytes,
    )


def _get_log_file_path() -> Optional[str]:
    """Extract log file path from root logger's handlers."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            return handler.baseFilename
    return None

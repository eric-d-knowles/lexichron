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


def process_files(
        urls: Iterable[str],
        executor_class: Type,
        workers: int,
        db: rs.DB,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        combined_bigrams: Optional[set] = None,
        spool_dir: Optional[str] = None,
        chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
        write_batch_size: Optional[int] = None,  # accepted for compatibility; chunk_entries governs
        progress: Optional[ProgressReporter] = None,
) -> Tuple[List[str], List[str], int, int, int]:
    """
    Process files concurrently and ingest results into RocksDB.

    Workers download, parse and filter shards, spooling the packed entries to
    chunk files (see :mod:`ngramprep.ngram_acquire.spool`); as each shard
    completes, this process streams its chunks into the database one at a
    time, marks the shard processed, and flushes. Memory use is bounded by
    one chunk per worker plus one in this process, independent of shard size.

    Args:
        urls: File URLs to process
        executor_class: ThreadPoolExecutor or ProcessPoolExecutor
        workers: Number of concurrent workers
        db: RocksDB database handle
        filter_pred: Optional predicate to filter ngrams by text
        combined_bigrams: Optional set of bigrams to combine with hyphens
        spool_dir: Directory for chunk files (default: a fresh temp directory)
        chunk_entries: Entries per chunk file
        write_batch_size: Ignored (kept so older callers do not break)
        progress: Optional reporter updated as files start, finish or fail

    Returns:
        Tuple of (success_msgs, failure_msgs, total_entries_written,
                  write_batches, total_uncompressed_bytes)
    """
    import tempfile

    log_file_path = _get_log_file_path()
    if log_file_path:
        logger.info("Log file path for workers: %s", log_file_path)

    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_uncompressed_bytes = 0

    ingestor = ChunkIngestor(db)
    total = len(urls) if hasattr(urls, "__len__") else None

    # Each run gets its own spool directory, removed at the end.
    spool_ctx = tempfile.TemporaryDirectory(prefix="ngram_acquire_spool_", dir=spool_dir)
    spool_path = spool_ctx.name
    logger.info("Spool directory: %s (chunk size %s entries)", spool_path, f"{chunk_entries:,}")
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
                logger.info(
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
                        fut = executor.submit(
                            process_and_ingest_file,
                            url,
                            idx,
                            filter_pred,
                            log_file_path,
                            combined_bigrams=combined_bigrams,
                            spool_dir=spool_path,
                            chunk_entries=chunk_entries,
                        )
                        futures[fut] = (url, idx)

                submit_next(max_in_flight)

                while futures:
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    if progress:
                        for fut in done:
                            progress.file_parsed(PurePosixPath(futures[fut][0]).name)

                    for fut in done:
                        url, file_idx = futures.pop(fut)
                        filename = PurePosixPath(url).name
                        chunk_paths: List[str] = []
                        try:
                            result_msg, chunk_paths, uncompressed_bytes, _entries = fut.result()

                            if result_msg.startswith("SUCCESS"):
                                if progress:
                                    progress.file_ingesting(filename)
                                ingestor.ingest_file(filename, chunk_paths,
                                                     uncompressed_bytes=uncompressed_bytes)
                                success_msgs.append(result_msg)
                                total_uncompressed_bytes += uncompressed_bytes
                                logger.info("Processed: %s", filename)
                                if progress:
                                    progress.file_done(filename, entries=_entries,
                                                       chunks=len(chunk_paths),
                                                       uncompressed_bytes=uncompressed_bytes)
                            else:
                                ChunkIngestor.discard(chunk_paths)
                                failure_msgs.append(result_msg)
                                if progress:
                                    progress.file_failed(filename, result_msg)

                        except Exception as exc:
                            ChunkIngestor.discard(chunk_paths)
                            msg = f"ERROR: {filename} - {exc}"
                            failure_msgs.append(msg)
                            logger.error(msg)
                            if progress:
                                progress.file_failed(filename, msg)
                        finally:
                            try:
                                os.remove(os.path.join(spool_path, f"{file_idx:05d}_{filename}.started"))
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

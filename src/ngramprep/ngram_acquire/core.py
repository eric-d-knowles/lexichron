"""Main entry point for ngram acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple, Type

from ngramprep.ngram_acquire.coordinator import (
    discover_files,
    select_file_subset,
    filter_processed_files,
    processed_totals,
    randomize_file_order,
)
from ngramprep.ngram_acquire.executor import process_files
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_acquire.worker import DEFAULT_CHUNK_ENTRIES, SPOOL_FORMATS
from ngramprep.ngram_acquire.verify import verify_database, describe_verification
from ngramprep.utilities.memory import memory_report
from ngramprep.ngram_acquire.reporter import print_pipeline_header, print_final_summary
from ngramprep.ngram_acquire.io.parse import prepare_multigrams
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate
from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
from ngramprep.ngram_acquire.db.build_path import build_db_path
from ngramprep.ngram_acquire.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngramprep.ngram_acquire.logger import setup_logger
from ngramprep.common_db.api import open_db
from ngramprep.common_db.compress import compress_db

logger = logging.getLogger(__name__)

__all__ = ["download_and_ingest_to_rocksdb", "run_acquisition", "AcquisitionError"]


class AcquisitionError(RuntimeError):
    """Raised when one or more files could not be acquired. Completed files
    are persisted; re-running with overwrite_db=False retries only the rest.
    Also raised when the finished database fails its checks: the counts do
    not equal what the files contributed, or lines could not be read."""

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def download_and_ingest_to_rocksdb(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
        db_path_stub: str,
        file_range: Optional[Tuple[int, int]] = None,
        workers: Optional[int] = None,
        ngram_type: str = "all",
        overwrite_db: bool = False,
        random_seed: Optional[int] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        open_type: str = "write:packed24",
        compact_after_ingest: bool = True,
        archive_path_stub: Optional[str] = None,
        combined_multigrams: Optional[set] = None,
        combined_multigrams_case: Optional[set] = None,
        chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
        spool_dir: Optional[str] = None,
        progress_path: Optional[str] = None,
        verify_counts: bool = True,
        allow_unparsed_lines: bool = False,
        spool_format: str = "sst",
) -> None:
    """
    Main pipeline: discover, download, parse, and ingest ngram files into RocksDB.

    Orchestrates the complete ngram acquisition workflow:
    1. Builds database path from stub and parameters
    2. Discovers available files from repository
    3. Opens/creates RocksDB with specified profile
    4. Downloads and processes files concurrently; each worker hands its
       file over as one sorted SST file
    5. Adopts each file into the database together with its marker
    6. Compacts the database
    7. Verifies the counts in the database against what the files contributed
    8. Optionally archives (compresses) database to specified directory

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for database (will be expanded)
        file_range: Optional (start_idx, end_idx) to process subset of files. Use None for all files.
        workers: Number of concurrent workers (default: min(cpu_count - 1, num_files))
        ngram_type: Filter type ("all", "tagged", etc.)
        overwrite_db: If True, remove an existing database before starting.
            Default False: an existing database is resumed, skipping files
            already marked as processed.
        random_seed: Optional seed for randomizing file processing order
        write_batch_size: Deprecated; chunk_entries now governs batch size.
        open_type: RocksDB profile; "write:packed24" (default) is the bulk-ingest
            profile with the merge operator the pipeline relies on.
        compact_after_ingest: Ignored: a run that acquired every file always
            compacts. Each file arrives as its own SST file and
            auto-compaction is off in the write profile, so compaction is
            what folds those files together and sums the counts of n-grams
            that arrived from more than one file; an uncompacted database is
            an unfinished one. Kept so that older settings files still load.
        archive_path_stub: Optional archive stub directory. Creates structured path: {archive_path_stub}/{release}/{corpus}/{n}gram_files/{n}grams.db.tar.zst
        combined_multigrams: Optional set of word sequences to combine into one token as the files are
            read: an entry is two or more words, optionally followed by "=> target", the token the run
            becomes (e.g., {"working class", "a . i . => A.I."}); without a target the run's own tokens
            are joined by hyphens in the case they had. The longest match at a position wins. These match
            whatever their capitalization in the data ("Working Class", "working class").
        combined_multigrams_case: The same, for sequences that must match as written.
        chunk_entries: Entries per spooled chunk. Workers stream parsed entries
            to chunk files on disk instead of holding a whole shard in memory;
            peak memory is roughly (workers + 1) x one chunk of packed values.
            Lower this on small allocations; raise it on large nodes.
        spool_dir: Where chunk files are written (default: the system temp
            directory, e.g. $TMPDIR; node-local storage is ideal). Needs room
            for about workers x 2 chunks.
        progress_path: If given, a JSON progress document is kept up to date
            at this path for other processes to read (see
            :mod:`ngramprep.ngram_acquire.progress`).
        verify_counts: Check the finished database against the run's ledger
            (default True). Every file's marker records the total
            occurrences and volume counts of the lines kept from it; after
            compaction the whole database is read once, in parallel, and
            must hold exactly the sum of those totals, with every value's
            years in order. A file ingested twice, a lost record or a wrong
            sum all show up as a difference, and the run fails. The pass
            reads the database once; turn it off only to save that time on a
            database you will check another way.
        allow_unparsed_lines: A line whose counts cannot be read in full
            (an entry that is not year,frequency,documents in whole numbers)
            is counted, logged and left out, so its n-gram is undercounted.
            By default a run that met any such line finishes its work and
            then fails, naming how many; set this to accept them. The number
            is recorded with the database either way.
        spool_format: How workers hand a file to the parent: ``"sst"``
            (default), one sorted SST file that the parent adopts with a
            metadata operation, or ``"chunk"``, record files that the parent
            replays through a write batch. Both produce the same database;
            ``"chunk"`` is the older path, kept as the reference.

    Raises:
        AcquisitionError: If any file failed after retries (the summary is
            printed first; completed files remain in the database), if the
            compaction failed, if the counts in the database do not match
            the ledger, or if lines could not be read and
            ``allow_unparsed_lines`` is off.
    """
    logger.info("Acquisition starting: Google Books %s %s %s-grams (%s)", repo_release_id, repo_corpus_id,
                ngram_size, ngram_type)

    # Set process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("nga:main")
        except Exception:
            pass

    start_time = datetime.now()

    if spool_format not in SPOOL_FORMATS:
        raise ValueError(f"spool_format must be one of {SPOOL_FORMATS}, got {spool_format!r}")
    if not compact_after_ingest:
        print("Note: compact_after_ingest is ignored; a complete acquisition is always compacted.")

    # The sequence list is checked before anything is downloaded or created:
    # an entry of one word could never match, and an entry longer than the
    # n-grams being acquired never occurs in one line (a 1-gram file has one
    # token per line, so no list applies there).
    combined_multigrams = prepare_multigrams(combined_multigrams, exact=combined_multigrams_case)
    if combined_multigrams:
        n = int(ngram_size)
        if n == 1:
            raise ValueError("combined_multigrams cannot apply to 1-grams: each line holds one token, so no "
                             "sequence ever occurs. Combine sequences when acquiring 2-grams or larger, or clear "
                             "the list.")
        too_long = sorted(m for m in combined_multigrams.sequences() if m.count(" ") + 1 > n)
        if too_long:
            raise ValueError(f"combined_multigrams: {too_long[0]!r} has {too_long[0].count(' ') + 1} words, which "
                             f"cannot occur in a {n}-gram; drop it or acquire longer n-grams")

    # Build full database path from stub
    db_path = build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)

    # Log to a file next to the database unless the caller configured logging.
    # INFO (per-file progress, batch writes) goes to the file; the console
    # shows the pipeline's own banner and progress bar, plus warnings.
    log_file = None
    if not logging.getLogger().hasHandlers():
        log_dir = Path(os.path.dirname(db_path)) / "logs"
        log_file = setup_logger(
            log_dir, filename_prefix="ngram_acquire",
            console=True, console_level=logging.WARNING,
        )
        print(f"Log file: {log_file}")
    logger.info("Database path: %s", db_path)
    progress = ProgressReporter(progress_path, stage="acquire",
                                db_path=db_path, log_path=log_file,
                                inputs={"source": f"Google Books {repo_release_id} {repo_corpus_id} {ngram_size}-grams"
                                                  f" ({ngram_type})"},
                                # the one reader that downloads what it reads
                                terms={"reading": "downloading & parsing",
                                       "reading_now": "downloading and parsing the files"})
    # What the run goes through, and what it will check, for whoever reads
    # the progress document: "step 2 of 3", and what a problem will lead to.
    progress.set_plan(["ingesting", "compacting"] + (["verifying"] if verify_counts else [])
                      + (["archiving"] if archive_path_stub is not None else []))
    progress.set_checks(verify_counts=bool(verify_counts), allow_unparsed_lines=bool(allow_unparsed_lines))

    # Discover available files
    page_url, file_urls_available = discover_files(ngram_size, repo_release_id, repo_corpus_id)

    # Select file subset
    file_urls_to_use, start_idx, end_idx = select_file_subset(file_urls_available, file_range)

    # Validate and normalize open_type
    open_type = open_type.lower()
    valid_profiles = {"read", "write", "read:packed24", "write:packed24"}
    if open_type not in valid_profiles:
        raise ValueError(f"open_type must be one of {valid_profiles}, got {open_type!r}")
    logger.info("Using RocksDB profile: %s", open_type)

    def header(files_to_get: int, files_to_skip: int, n_workers: int) -> None:
        print_pipeline_header(
            start_time=start_time,
            page_url=page_url,
            db_path=db_path,
            start_idx=start_idx,
            end_idx=end_idx,
            total_files=len(file_urls_available),
            files_to_get=files_to_get,
            files_to_skip=files_to_skip,
            workers=n_workers,
            write_batch_size=chunk_entries,
            ngram_size=ngram_size,
            ngram_type=ngram_type,
            overwrite_db=overwrite_db,
            open_type=open_type,
        )

    run_acquisition(
        db_path, file_urls_to_use,
        sizes=getattr(file_urls_available, "sizes", None) or {},
        workers=workers, overwrite_db=overwrite_db, open_type=open_type, random_seed=random_seed,
        progress=progress, start_time=start_time, verify_counts=verify_counts,
        allow_unparsed_lines=allow_unparsed_lines, print_header=header,
        process_kwargs=dict(filter_pred=make_ngram_type_predicate(ngram_type),
                            combined_multigrams=combined_multigrams, spool_dir=spool_dir,
                            chunk_entries=chunk_entries, spool_format=spool_format),
        archive=(lambda: _archive_database(db_path, archive_path_stub, ngram_size, repo_release_id, repo_corpus_id))
        if archive_path_stub is not None else None,
    )
    return None


def run_acquisition(
        db_path: str,
        units: list,
        *,
        sizes: Optional[dict] = None,
        workers: Optional[int] = None,
        overwrite_db: bool = False,
        open_type: str = "write:packed24",
        random_seed: Optional[int] = None,
        progress: ProgressReporter,
        start_time: datetime,
        verify_counts: bool = True,
        allow_unparsed_lines: bool = False,
        print_header=None,
        process_kwargs: Optional[dict] = None,
        archive=None,
) -> None:
    """Take a list of source files into a raw database and vouch for it.

    This is the part of an acquisition that does not depend on where the
    files come from or what they look like: each of ``units`` (a URL or a
    path; its last component names it) is handed to a worker, which returns
    one sorted SST file of summed counts and the totals of what it kept; the
    file is ingested together with its marker; files already marked are
    skipped; the database is compacted; and the whole of it is read once
    and must hold exactly the totals the markers record. The reader (what a
    worker does with a file) comes in ``process_kwargs`` (see
    :func:`ngramprep.ngram_acquire.executor.process_files`): by default the
    Google Books n-gram worker, or another ``worker_fn``.

    Args:
        db_path: The raw database to create or resume
        units: The files to acquire, already narrowed to the run's selection
        sizes: Their sizes by name, where known, for progress by size
        workers: Processes (default: the allocation's CPUs less one, at most one per file)
        overwrite_db: Remove an existing database first; otherwise resume it
        open_type: RocksDB profile
        random_seed: Shuffle the order of the files with this seed
        progress: The run's progress reporter
        start_time: When the run began
        verify_counts: Check the finished database against the markers' totals
        allow_unparsed_lines: Accept lines (or rows) that could not be read in full
        print_header: Called as (files to get, files to skip, workers) to print the run's banner
        process_kwargs: Keyword arguments for ``process_files``
        archive: Called with nothing after a verified run, to archive the database

    Raises:
        AcquisitionError: as :func:`download_and_ingest_to_rocksdb`.
    """
    file_urls_to_use = list(units)
    process_kwargs = dict(process_kwargs or {})
    # Handle existing database
    if overwrite_db and os.path.exists(db_path):
        logger.info("Removing existing database for fresh start")
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Determine worker count based on available CPUs and number of files
    if workers is None:
        try:                                   # the allocation's CPUs inside a Slurm job, not the node's
            cpu_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpu_count = os.cpu_count() or 4
        # Use CPUs - 1 for overhead, capped by number of files to process
        workers = min(cpu_count - 1, len(file_urls_to_use))
        # Ensure at least 1 worker
        workers = max(1, workers)

    # Open database with specified profile (flush happens automatically on exit)
    with open_db(db_path, profile=open_type, create_if_missing=True) as db:
        # Resume mode: skip already-processed files
        files_to_skip = 0
        prior = None
        if not overwrite_db:
            prior = processed_totals(file_urls_to_use, db)
            file_urls_to_use, files_to_skip = filter_processed_files(file_urls_to_use, db)

        progress.set_totals(files_total=len(file_urls_to_use) + files_to_skip,
                            files_skipped=files_to_skip, prior=prior)
        # The listing's sizes, where it gave them: progress by how much there
        # is to fetch, since files differ enormously in what they hold.
        progress.set_sizes(sizes or {}, [PurePosixPath(u).name for u in file_urls_to_use])
        to_fetch = progress.doc.get("bytes_total")
        logger.info("Files: %d to read%s, %d already in the database, %d workers",
                    len(file_urls_to_use), f" ({to_fetch / 1e9:,.1f} GB compressed)" if to_fetch else "",
                    files_to_skip, workers)

        # Optional randomization
        if random_seed is not None:
            randomize_file_order(file_urls_to_use, random_seed)

        # Print run summary
        if print_header is not None:
            print_header(len(file_urls_to_use), files_to_skip, workers)

        # Process and ingest files (always processes, for parallelism)
        success, failure, written, batches, uncompressed_bytes = process_files(
            urls=file_urls_to_use,
            executor_class=ProcessPoolExecutor,
            workers=workers,
            db=db,
            progress=progress,
            **process_kwargs,
        )
        # Database is automatically flushed by context manager on exit
        logger.info("Files finished in %s: %d ingested, %d failed; %s entries handed to the database, "
                    "%.1f GB uncompressed read", timedelta(seconds=int((datetime.now() - start_time).total_seconds())),
                    len(success), len(failure), f"{written:,}", uncompressed_bytes / 1e9)

        # Compaction, whenever every file is in (a run with failures leaves
        # the database incomplete, and the re-run that completes it compacts).
        # On a large database this is the longest single step of the run.
        _log_memory("after the files")
        compaction_error = None
        if not failure and not success and _already_compacted(db):
            # A re-run that found every file done, on a database a previous
            # run compacted (typically one that ended during the count
            # check). Compacting again would rewrite the whole database to
            # produce the same records.
            logger.info("Compaction skipped: this run added no file and the database is already compacted "
                        "(all of its SST files are in one sorted level, so no record is held twice)")
            print("\nCompaction skipped: nothing was added and the database is already compacted.")
        elif not failure:
            with progress.long_phase("compacting"):
                compaction_error = _perform_compaction(db, db_path)
        # bookkeeping keys: one resume marker per file done, in any job
        done_files = len(success) + (prior["files"] if prior else 0)
        progress.set_database(db, compacted=not failure and compaction_error is None, bookkeeping=done_files)

    # Report final statistics
    end_time = datetime.now()
    print_final_summary(
        start_time=start_time,
        end_time=end_time,
        success=success,
        failure=failure,
        written=written,
        batches=batches,
        uncompressed_bytes=uncompressed_bytes,
    )

    # Fail loudly if any file did not make it. Completed files are already
    # persisted and marked, so a re-run (overwrite_db=False) picks up only the
    # missing ones. Archiving is skipped: the database is incomplete.
    if failure:
        for msg in failure:
            logger.error("Failed file: %s", msg)
        logger.error("Stopping without compacting or verifying: %d files failed. The database holds the other "
                     "files, each marked; a re-run reads only the missing ones", len(failure))
        progress.finish("failed", f"{len(failure)} file(s) failed")
        raise AcquisitionError(
            f"{len(failure)} of {len(success) + len(failure)} files failed "
            f"(see log for details); re-run with overwrite_db=False to retry "
            f"only the missing files. First failure: {failure[0]}"
        )

    if compaction_error is not None:
        progress.finish("failed", f"compaction failed: {compaction_error}")
        logger.error("Stopping: every file is in, but the compaction failed; a re-run compacts")
        raise AcquisitionError(
            f"Every file was acquired, but the compaction failed ({compaction_error}). The database is "
            f"complete but not compacted: counts of records that arrived from more than one file are not "
            f"summed yet. Re-run with overwrite_db=False to compact it."
        )

    # Verification: the database is closed, so readers see it whole. The
    # ledger (every file's totals) is read from the markers of every file in
    # the database, whichever run acquired it.
    if verify_counts:
        from ngramprep.utilities.display import format_banner
        with progress.long_phase("verifying"):
            print()
            print(format_banner("Count Verification"))
            sys.stdout.flush()
            t0 = time.time()
            _log_memory("before the count check")
            logger.info("Verifying the counts: reading the whole database once with up to %d processes "
                        "and comparing its totals with the files' recorded totals", workers)
            verification = verify_database(db_path, workers)
            verification["seconds"] = round(time.time() - t0, 1)
        for line in describe_verification(verification):
            print(line)
            logger.info("%s", line)
        print(f"Verification took {timedelta(seconds=int(verification['seconds']))}")
        _log_memory("after the count check")
        logger.info("Verification took %s; outcome: %s", timedelta(seconds=int(verification["seconds"])),
                    verification["status"])
        progress.set_verification(verification)
        if verification["status"] == "mismatch":
            progress.finish("failed", "the counts in the database do not match what the files contributed")
            logger.error("Stopping: the counts in the database do not match what the files contributed. "
                         "The database should not be used")
            raise AcquisitionError(
                "The counts in the database do not match what the files contributed (figures above, and in "
                "the log). The database should not be used. Nothing was deleted: re-run with "
                "overwrite_db=True to acquire it again, and please report this."
            )
        unreadable = int(verification["ledger"].get("problem_lines", 0))
    else:
        from ngramprep.ngram_acquire.verify import ledger_totals
        with open_db(db_path, mode="r", profile="read:packed24") as db:
            ledger = ledger_totals(db)
        unreadable = int(ledger.get("problem_lines", 0))
        progress.set_verification({"status": "skipped", "ledger": ledger})
        print("Count verification skipped (verify_counts is off).")
        logger.info("Count verification skipped (verify_counts is off)")

    if unreadable and not allow_unparsed_lines:
        progress.finish("failed", f"{unreadable:,} lines could not be read in full")
        logger.error("Ending with an error: %s lines could not be read in full (the workers' entries show them) "
                     "and allow_unparsed_lines is off. The database is otherwise complete and compacted",
                     f"{unreadable:,}")
        raise AcquisitionError(
            f"{unreadable:,} lines could not be read in full, so what they held is "
            f"undercounted (the log shows the lines). The database is otherwise complete and compacted. "
            f"Re-run with allow_unparsed_lines=True to accept it as it is."
        )

    # Optional archiving: compress DB to archive directory
    if archive is not None:
        with progress.long_phase("archiving"):
            archive()

    progress.finish("done")
    logger.info("Acquisition finished in %s: %s", timedelta(seconds=int((datetime.now() - start_time).total_seconds())),
                db_path)

    return None


_LEVELS = 7        # RocksDB's default number of levels


def _log_memory(when: str) -> None:
    """Log what the job's memory holds (see utilities/memory.py): at the
    start of each step, so an out-of-memory kill can be read from the log."""
    report = memory_report()
    if report:
        logger.info("Memory %s: %s", when, report)


def _already_compacted(db) -> bool:
    """True when the database provably needs no compaction: nothing is
    waiting in memory and every SST file is in one level below level 0.

    The files of such a level are a single sorted run with no key in two
    files, so every n-gram has exactly one record, its counts summed: what
    a full compaction leaves. Files a run ingested and never compacted sit
    in level 0 or are stacked over several levels wherever their key ranges
    overlap, and fail this test. Any doubt (a property that cannot be read)
    answers False, and the database is compacted."""
    try:
        if int(db.get_property("rocksdb.num-entries-active-mem-table") or 0) or \
                int(db.get_property("rocksdb.num-entries-imm-mem-tables") or 0):
            return False
        per_level = [int(db.get_property(f"rocksdb.num-files-at-level{i}")) for i in range(_LEVELS)]
    except Exception:
        return False
    occupied = [i for i, n in enumerate(per_level) if n > 0]
    return len(occupied) == 1 and occupied[0] > 0


def _perform_compaction(db, db_path: str) -> Optional[str]:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)

    Returns:
        None when the compaction completed, else the error as text (the
        caller fails the run: an uncompacted database is not a finished one).
    """
    from ngramprep.utilities.display import format_bytes, format_banner

    logger.info("Compacting: folding the per-file SST files into one sorted database and summing the "
                "counts of records that arrived from more than one file")
    print()
    print(format_banner("Post-Ingestion Compaction"))

    # Get initial size if possible
    try:
        initial_size = db.get_property("rocksdb.total-sst-files-size")
        initial_size = int(initial_size) if initial_size else None
        if initial_size:
            print(f"Initial DB size:         {format_bytes(initial_size)}")
    except Exception:
        initial_size = None

    sys.stdout.flush()

    start_time = time.time()
    try:
        db.compact_all()
        elapsed = time.time() - start_time

        print(f"Compaction completed in {timedelta(seconds=int(elapsed))}")
        logger.info("Compaction completed in %s", timedelta(seconds=int(elapsed)))

        # Get final size if possible
        try:
            final_size = db.get_property("rocksdb.total-sst-files-size")
            final_size = int(final_size) if final_size else None
            if initial_size and final_size:
                saved = initial_size - final_size
                pct = (saved / initial_size) * 100
                print(f"Size before:             {format_bytes(initial_size)}")
                print(f"Size after:              {format_bytes(final_size)}")
                print(f"Space saved:             {format_bytes(saved)} ({pct:.1f}%)")
                logger.info("Database size: %s before compaction, %s after", format_bytes(initial_size),
                            format_bytes(final_size))
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        print(f"Compaction failed: {e}")
        return str(e)
    return None


def _archive_database(
    db_path: str,
    archive_stub: str,
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str
) -> None:
    """
    Archive (compress) the database to a structured directory using zstd.

    Creates archive path following the pattern:
    {archive_stub}/{release_id}/{corpus_id}/{n}gram_files/{n}grams.db.tar.zst

    The compression is done in a temp directory on /scratch first, then
    rsync'd to the final destination for better performance.

    Args:
        db_path: Path to the database directory
        archive_stub: Base directory for archives (will be expanded with structured path)
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
    """
    import subprocess
    import tempfile
    from pathlib import Path
    from ngramprep.utilities.display import format_banner, truncate_path_to_fit

    logger.info("Archiving the database")
    print()
    print(format_banner("Database Archival"))

    # Build structured archive path (mirrors database structure)
    archive_base = Path(archive_stub)
    final_archive = archive_base / repo_release_id / repo_corpus_id / f"{ngram_size}gram_files" / f"{Path(db_path).name}.tar.zst"
    logger.info(f"Target archive location: {final_archive}")

    # Determine output archive filename
    db_name = Path(db_path).name

    try:
        # Compress into a temp directory beside the database (same filesystem),
        # or $TMPDIR if set, then copy to the final location.
        temp_base = os.environ.get("TMPDIR") or str(Path(db_path).parent)
        temp_dir = tempfile.mkdtemp(dir=temp_base, prefix="ngram_archive_")
        temp_archive = Path(temp_dir) / f"{db_name}.tar.zst"
        logger.info(f"Temporary archive location: {temp_archive}")

        try:
            # Compress the database to temp location
            print(f"Compressing database to temporary location: {truncate_path_to_fit(str(temp_archive), 'Compressing database to temporary location: ')}")
            compress_db(db_path, output_path=temp_archive)
            logger.info(f"Database compressed successfully to {temp_archive}")

            # Transfer the archive to final destination
            print(f"Transferring archive to {truncate_path_to_fit(str(final_archive), 'Transferring archive to ')}")

            # Create parent directories for destination
            final_archive.parent.mkdir(parents=True, exist_ok=True)

            # Copy the archive to final destination
            import shutil as shutil_copy
            shutil_copy.copy2(str(temp_archive), str(final_archive))

            logger.info(f"Archive transferred successfully to {final_archive}")
            print(f"Archive created: {truncate_path_to_fit(str(final_archive), 'Archive created: ')}")

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {cleanup_err}")

    except Exception as e:
        logger.error(f"Archival failed: {e}")
        print(f"Archival failed: {e}")
        raise

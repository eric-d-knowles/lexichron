"""Worker process for downloading and parsing ngram files."""
from __future__ import annotations

import gzip
import logging
import os
import struct
from contextlib import closing
from time import monotonic as _monotonic
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import requests

from ngramprep.ngram_acquire.io.download import stream_download_with_retries
from ngramprep.ngram_acquire.io.parse import parse_line_checked, prepare_multigrams
from ngramprep.ngram_acquire.spool import CheckedChunkWriter, CheckedSstChunkWriter

try:                                   # compiled fast path (see io/parse_cy.pyx)
    from ngramprep.ngram_acquire.io.parse_cy import (FALLBACK as _FALLBACK, multigram_tables as _multigram_tables,
                                                     parse_packed as _parse_packed)
except ImportError:                    # pragma: no cover - built without the extension
    _parse_packed = None
    _multigram_tables = None
    _FALLBACK = object()

_FAST_MODES = {"all": 0, "tagged": 1, "untagged": 2}
_READ_BLOCK = 1 << 20


def _iter_lines(stream):
    """Yield the lines of a binary stream (without their newline), reading in
    large blocks: much cheaper than readline() per line. A final line without
    a newline is yielded too. Byte totals are the caller's: sum the blocks."""
    tail = b""
    read = stream.read
    while True:
        block = read(_READ_BLOCK)
        if not block:
            break
        yield len(block), None               # (bytes read, no line): for byte accounting
        lines = block.split(b"\n")
        lines[0] = tail + lines[0]
        tail = lines.pop()
        for line in lines:
            yield 0, line
    if tail:
        yield 0, tail

if TYPE_CHECKING:
    from ngram_acquire.io.parse import NgramRecord

logger = logging.getLogger(__name__)

__all__ = ["process_and_ingest_file", "merge_packed_records", "DEFAULT_CHUNK_ENTRIES", "SPOOL_FORMATS"]

# How a worker hands a shard to the parent: "sst" writes sorted RocksDB files
# of merge operands that the parent adopts with one metadata operation;
# "chunk" writes plain record files that the parent replays through a write
# batch. Both carry the same records; "chunk" is the older path, kept as the
# reference the SST path is tested against.
SPOOL_FORMATS = ("sst", "chunk")

# A worker says how far it is into its file this often, by rewriting the
# marker file the parent already watches (see ``_report``).
_REPORT_EVERY_S = 3.0

# Unreadable lines are logged with their content up to this many per shard;
# the count is always complete.
_PROBLEM_LINES_LOGGED = 20

# Entries per spooled chunk. Bounds worker memory to roughly one chunk of
# packed values (a few hundred MB for unigrams, far less for 5-grams).
DEFAULT_CHUNK_ENTRIES = 200_000

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def process_and_ingest_file(
    url: str,
    worker_id: int,
    filter_pred: Optional[Callable[[str], bool]] = None,
    log_file_path: Optional[str] = None,
    *,
    session: Optional[requests.Session] = None,
    combined_multigrams: Optional[set] = None,
    combined_multigrams_case: Optional[set] = None,
    max_attempts: int = 3,
    spool_dir: Optional[str] = None,
    chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
    spool_format: str = "sst",
) -> Tuple[str, List[str], int, int, int, Dict[str, int]]:
    """
    Download, decompress, parse a gzipped ngram file, and spool it to chunks.

    Downloads the file from the given URL, decompresses it line-by-line,
    parses each line, packs the frequencies into compact binary form, and
    writes the entries to files in ``spool_dir`` (see
    :mod:`ngramprep.ngram_acquire.spool`). The worker holds at most one
    chunk in memory; the parent process ingests the files afterwards.

    Every value is checked on its way to the spool (years strictly
    ascending, which the merge operator assumes), the frequencies and
    document counts of everything kept are totalled, and lines with content
    that could not be read are counted: the figures come back in ``stats``
    and are stored with the shard's marker, where the run's verification
    reads them.

    Args:
        url: Download URL for the gzipped ngram file
        worker_id: Worker identifier for logging
        filter_pred: Optional predicate to filter ngrams by text
        log_file_path: Optional path to log file for worker output
        session: Optional requests.Session for connection pooling
        combined_multigrams: Optional word sequences to combine (matched whatever
            their capitalization), or a prepared Multigrams
        combined_multigrams_case: Sequences that must match as written
        max_attempts: How many times to attempt the whole download-and-parse
            of this file. A connection dropped part-way through a stream is
            retried from the beginning, discarding partial results (including
            chunk files written so far).
        spool_dir: Directory for chunk files (default: the system temp dir).
        chunk_entries: Entries per chunk file.
        spool_format: ``"sst"`` (one sorted SST file of merge operands for
            the shard) or ``"chunk"`` (plain chunk files); see ``SPOOL_FORMATS``.

    Returns:
        Tuple of (status_message, chunk_paths, uncompressed_bytes, entries,
        lines, stats)
        - status_message: Success or error message
        - chunk_paths: Files holding this file's entries, in order
        - uncompressed_bytes: Total bytes of uncompressed data processed
        - entries: Number of distinct keys written across the files
        - lines: Raw lines read
        - stats: ``match_total`` and ``volume_total`` (the frequencies and
          document counts of every kept line, summed), ``problem_lines``
          (lines with content that could not be read) and ``reordered``
          (values whose years had to be put in order)

    Packing Format:
        Values are packed as little-endian uint64 triplets per year:
        (year, frequency, document_count) → struct format '<{3*N}Q'
    """
    # Set up worker-specific logging
    if log_file_path:
        worker_logger = logging.getLogger(f"worker_{os.getpid()}")

        # Only add handler if not already present
        if not worker_logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file_path, mode="a")
                formatter = logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(formatter)
                worker_logger.addHandler(file_handler)
                worker_logger.setLevel(logging.INFO)
                worker_logger.propagate = False
            except Exception as e:
                # Fallback to standard logger if file logging fails
                logger.warning(
                    "Worker %s (PID %s): Could not set up file logging: %s",
                    worker_id, os.getpid(), e
                )
                worker_logger = logger
    else:
        worker_logger = logger

    # Set process title if available (helps with process monitoring)
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"nga:worker[{worker_id:03d}]")
        except Exception:
            pass

    if spool_format not in SPOOL_FORMATS:
        raise ValueError(f"spool_format must be one of {SPOOL_FORMATS}, got {spool_format!r}")
    writer_class = CheckedSstChunkWriter if spool_format == "sst" else CheckedChunkWriter

    filename = PurePosixPath(url).name
    pid = os.getpid()

    # One session per worker call, closed when done (a shared one may be passed in).
    own_session = session is None
    sess = session or requests.Session()

    worker_logger.info("Worker %s (PID %s): Processing %s", worker_id, pid, filename)

    import tempfile
    spool_root = spool_dir or tempfile.gettempdir()
    file_tag = f"{worker_id:05d}_{filename}"

    # Use the compiled parser when the filter is one it knows; it hands
    # anything unusual back to parse_line. The sequences to combine are
    # normalized once here (the entry point did it already; this is idempotent).
    combined_multigrams = prepare_multigrams(combined_multigrams, exact=combined_multigrams_case)
    fast_mode = _FAST_MODES.get(getattr(filter_pred, "ngram_type", None))
    fast = _parse_packed is not None and fast_mode is not None
    fast_suffixes = tuple(s.encode("ascii") for s in getattr(filter_pred, "suffixes", ())) if fast else ()
    fast_folded, fast_exact, fast_firsts, fast_words = _multigram_tables(combined_multigrams) if fast \
        else (None, None, None, 0)
    # Tell the parent (which reports progress) that a worker has this file.
    started_path = os.path.join(spool_root, f"{file_tag}.started")
    _report(started_path, 0, None)

    try:
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            writer = writer_class(spool_root, file_tag, chunk_entries, merge_packed_records)
            uncompressed_bytes = 0
            lines_processed = 0
            problem_lines = 0
            if attempt > 1:
                _report(started_path, 0, None)          # the file starts over: so does its count
            last_report = _monotonic()
            try:
                resp = stream_download_with_retries(url, session=sess)
                with closing(resp):
                    content_length = resp.headers.get("content-length")
                    if content_length:
                        try:
                            worker_logger.info(
                                "Worker %s (PID %s): File size: %s bytes (compressed)",
                                worker_id, pid, f"{int(content_length):,}"
                            )
                        except ValueError:
                            worker_logger.debug(
                                "Worker %s (PID %s): Non-numeric content-length=%r",
                                worker_id, pid, content_length
                            )

                    # Process gzipped content line by line
                    with gzip.GzipFile(fileobj=resp.raw, mode="rb") as gz:
                        for nread, raw in _iter_lines(gz):
                            if raw is None:
                                # Once per block of decompressed data (about 1 MB):
                                # the place to look at the clock, not once per line.
                                uncompressed_bytes += nread
                                if _monotonic() - last_report >= _REPORT_EVERY_S:
                                    last_report = _monotonic()
                                    _report(started_path, lines_processed, resp)
                                continue
                            lines_processed += 1
                            if fast:
                                hit = _parse_packed(raw, fast_mode, fast_suffixes, fast_folded, fast_exact,
                                                    fast_firsts, fast_words)
                                if hit is not _FALLBACK:
                                    if hit is not None:
                                        writer.add(hit[0], hit[1])
                                    continue
                            # Anything the fast path does not take is parsed here,
                            # including every line with content that cannot be
                            # read: those are counted, never dropped in silence.
                            problem = None
                            key = packed = None
                            try:
                                key, rec, unreadable = parse_line_checked(
                                    raw.decode("utf-8"),
                                    filter_pred=filter_pred,
                                    combined_multigrams=combined_multigrams,
                                )
                                if key and rec:
                                    packed = _pack_record(rec)
                                if unreadable:
                                    problem = "counts that are not year,frequency,documents"
                            except UnicodeDecodeError as exc:
                                problem = f"not UTF-8 ({exc})"
                            except (ValueError, struct.error) as exc:
                                # e.g. a count too large for the packed format
                                key = packed = None
                                problem = f"{type(exc).__name__}: {exc}"
                            if packed:
                                # Repeated keys within a chunk (e.g. tagged
                                # variants of a combined multigram) are summed
                                # by the writer; across chunks, by the DB.
                                # Outside the try: an error from the writer is
                                # the shard's, not this line's.
                                writer.add(key, packed)
                            if problem:
                                problem_lines += 1
                                if problem_lines <= _PROBLEM_LINES_LOGGED:
                                    worker_logger.warning(
                                        "Worker %s (PID %s): %s line %s could not be read in full (%s): %r",
                                        worker_id, pid, filename, lines_processed, problem, raw[:200]
                                    )

                    _check_complete(resp, content_length, filename)
                    _report(started_path, lines_processed, resp)     # the whole file, while it waits to be ingested

                chunk_paths = writer.finish()
                entries = writer.entries_total
                # Tell the parent's progress reporter this file is parsed, even
                # before the parent gets round to collecting the result.
                try:
                    os.replace(started_path, os.path.join(spool_root, f"{file_tag}.parsed"))
                except OSError:
                    pass
                stats = {"match_total": writer.match_total, "volume_total": writer.volume_total,
                         "problem_lines": problem_lines, "reordered": writer.reordered}
                msg = (
                    f"SUCCESS: {filename} - {lines_processed:,} lines, "
                    f"{entries:,} entries in {len(chunk_paths)} chunks, "
                    f"{uncompressed_bytes:,} uncompressed bytes"
                )
                if problem_lines:
                    msg += f", {problem_lines:,} lines not read in full"
                if writer.reordered:
                    msg += f", {writer.reordered:,} values put in year order"
                worker_logger.info("Worker %s (PID %s): %s", worker_id, pid, msg)
                return msg, chunk_paths, uncompressed_bytes, entries, lines_processed, stats

            except requests.Timeout as exc:
                writer.discard()
                last_error = f"TIMEOUT: {filename}"
                detail = str(exc)
            except (requests.RequestException, EOFError, OSError, gzip.BadGzipFile) as exc:
                # Includes connections dropped mid-stream, which surface from
                # inside the gzip loop rather than from the initial request,
                # and a body shorter than the server announced.
                writer.discard()
                last_error = f"NETWORK_ERROR: {filename}"
                detail = f"{type(exc).__name__}: {exc}"

            if attempt < max_attempts:
                worker_logger.warning(
                    "Worker %s (PID %s): %s after %s lines (attempt %d/%d: %s); retrying from start",
                    worker_id, pid, last_error, f"{lines_processed:,}", attempt, max_attempts, detail
                )
            else:
                worker_logger.error(
                    "Worker %s (PID %s): %s - giving up after %d attempts (%s)",
                    worker_id, pid, last_error, max_attempts, detail
                )
        return last_error, [], 0, 0, 0, {}

    except Exception as exc:
        # Anything else (a value that cannot be spooled, a full disk the
        # writer could not recover from) fails the shard: nothing of it has
        # been published, so nothing of it is ingested.
        try:
            writer.discard()
        except Exception:
            pass
        msg = f"ERROR: {filename} - {exc}"
        worker_logger.error("Worker %s (PID %s): Error - %s: %s", worker_id, pid, filename, exc)
        return msg, [], 0, 0, 0, {}

    finally:
        if own_session:
            try:
                sess.close()
            except Exception:
                pass


def _report(marker_path: str, lines: int, resp) -> None:
    """Write "<lines read> <compressed bytes received>" into the worker's
    marker file, for the parent's progress display (throughput while files
    are still being read, and progress weighed by size). Advisory: a failure
    to write it is ignored, and the parent ignores a marker it cannot read."""
    try:
        received = int(resp.raw.tell()) if resp is not None else 0
    except (AttributeError, OSError, TypeError, ValueError):
        received = 0
    try:
        with open(marker_path, "w") as fh:
            fh.write(f"{int(lines)} {received}")
    except OSError:
        pass


def _check_complete(resp, content_length, filename: str) -> None:
    """Raise if fewer compressed bytes arrived than the server announced.

    A gzip stream cut short already fails (no end-of-stream marker, or a bad
    checksum), and so does a body the HTTP layer knows to be incomplete; this
    compares the two counts directly where both are known, so that a shard is
    never taken as whole on the strength of the decompressor alone.
    """
    try:
        expected = int(content_length)
        received = int(resp.raw.tell())
    except (TypeError, ValueError, AttributeError, OSError):
        return                                 # no announced length, or a body that does not count
    if received < expected:
        raise OSError(f"{filename}: received {received:,} of {expected:,} compressed bytes")


def merge_packed_records(a: bytes, b: bytes) -> bytes:
    """
    Sum two packed records year by year.

    Both inputs are sequences of little-endian uint64 triplets
    ``(year, frequency, document_count)`` as produced by :func:`_pack_record`.
    The result contains each year once, with frequency and document_count
    summed, in ascending year order.
    """
    totals: dict[int, list[int]] = {}
    for blob in (a, b):
        n = len(blob) // 8
        vals = struct.unpack(f"<{n}Q", blob)
        for i in range(0, n, 3):
            year, freq, docs = vals[i], vals[i + 1], vals[i + 2]
            t = totals.get(year)
            if t is None:
                totals[year] = [freq, docs]
            else:
                t[0] += freq
                t[1] += docs
    flat: list[int] = []
    for year in sorted(totals):
        freq, docs = totals[year]
        flat.extend((year, freq, docs))
    return struct.pack(f"<{len(flat)}Q", *flat)


def _pack_record(rec: NgramRecord) -> bytes:
    """
    Pack frequency data into compact binary format.

    Converts year/frequency/document_count triplets into little-endian
    uint64 values for efficient storage.

    Args:
        rec: NgramRecord containing frequency data

    Returns:
        Packed binary data as bytes
    """
    freqs = rec.get("frequencies", [])
    if not freqs:
        return b""

    # Flatten triplets into single list (cache extend for performance)
    flat: list[int] = []
    extend = flat.extend
    for f in freqs:
        extend((
            int(f["year"]),
            int(f["frequency"]),
            int(f["document_count"])
        ))

    return struct.pack(f"<{len(flat)}Q", *flat)
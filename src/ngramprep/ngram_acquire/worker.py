"""Worker process for downloading and parsing ngram files."""
from __future__ import annotations

import gzip
import http.client
import logging
import multiprocessing as mp
import os
import struct
from contextlib import closing
from time import monotonic as _monotonic
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import requests
import urllib3.exceptions

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

# What a download that fails part-way raises. The body is read from the raw
# urllib3 stream (through gzip), so a connection that breaks or stalls while
# the file is being read raises urllib3's own errors, which are neither
# requests' nor OSError: "Connection broken: IncompleteRead(...)" is a
# urllib3 ProtocolError. Opening the download goes through requests and
# raises its errors. All of them mean the same thing here: discard what was
# read and start the file again.
_TIMEOUTS = (requests.Timeout, urllib3.exceptions.TimeoutError)
_BROKEN_DOWNLOADS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException,
                     EOFError, OSError, gzip.BadGzipFile)

# A worker says how far it is into its file this often, by rewriting the
# marker file the parent already watches (see ``_report``).
_REPORT_EVERY_S = 3.0

# While a file is being read, the worker says so in the log this often: a large
# file takes minutes, and a log that is silent meanwhile cannot tell "reading"
# from "stuck".
_LOG_EVERY_S = 60.0

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
    files_total: Optional[int] = None,
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
        worker_id: The file's number in this run (1, 2, ...). It names the
            file's spool files and appears in the log as "file 12 of 40"; it
            is not the number of the process doing the work, which the log
            gives separately (see ``_worker_slot``).
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
        files_total: How many files this run fetches, for the log ("file 12 of 40").

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
    # Every log entry of this call names the file, where it stands in the run,
    # and the process doing the work: "5-00450-of-11145.gz (file 12 of 40)" from
    # logger "acquire.worker.03". The pool's processes are numbered 1..workers
    # and each takes many files in turn.
    filename = PurePosixPath(url).name
    slot = _worker_slot()
    worker_logger = logging.getLogger(f"acquire.worker.{slot:02d}")
    if log_file_path and not worker_logger.handlers:
        try:
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            worker_logger.addHandler(file_handler)
            worker_logger.setLevel(logging.INFO)
            worker_logger.propagate = False
        except Exception as e:
            # Fall back to the module's logger if the file cannot be opened
            logger.warning("%s: could not open the log file for this worker: %s", filename, e)
            worker_logger = logger
    which = f"{filename} (file {worker_id}" + (f" of {files_total}" if files_total else "") + ")"

    def log(level: int, text: str, *args) -> None:
        worker_logger.log(level, "%s: " + text, which, *args)

    # The process title too (ps, top): which worker, and the file it has.
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"nga:worker[{slot:02d}] {filename}")
        except Exception:
            pass

    if spool_format not in SPOOL_FORMATS:
        raise ValueError(f"spool_format must be one of {SPOOL_FORMATS}, got {spool_format!r}")
    writer_class = CheckedSstChunkWriter if spool_format == "sst" else CheckedChunkWriter

    # One session per worker call, closed when done (a shared one may be passed in).
    own_session = session is None
    sess = session or requests.Session()
    started_at = _monotonic()

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
            last_report = last_logged = attempt_started = _monotonic()
            log(logging.INFO, "opening the download%s", f" (attempt {attempt} of {max_attempts})" if attempt > 1 else "")
            try:
                resp = stream_download_with_retries(url, session=sess)
                with closing(resp):
                    content_length = resp.headers.get("content-length")
                    try:
                        size_text = f"{int(content_length) / 1e6:,.1f} MB compressed"
                    except (TypeError, ValueError):
                        size_text = "size not announced"
                    log(logging.INFO, "downloading and parsing as it arrives (%s)", size_text)

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
                                    if last_report - last_logged >= _LOG_EVERY_S:
                                        last_logged = last_report
                                        log(logging.INFO, "still reading after %.0fs: %s lines, %s kept so far%s",
                                            last_report - attempt_started, f"{lines_processed:,}",
                                            f"{writer.kept:,}",
                                            _received_text(resp, content_length))
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
                                    log(logging.WARNING, "line %s could not be read in full (%s): %r",
                                        f"{lines_processed:,}", problem, raw[:200])
                                elif problem_lines == _PROBLEM_LINES_LOGGED + 1:
                                    log(logging.WARNING, "more lines could not be read in full; only the first %d "
                                        "are shown, all are counted", _PROBLEM_LINES_LOGGED)

                    _check_complete(resp, content_length, filename)
                    _report(started_path, lines_processed, resp)     # the whole file, while it waits to be ingested

                read_s = _monotonic() - attempt_started
                writer.settle()                           # so that its part count is the true one
                parts = len(writer.part_paths) + (1 if writer.pending else 0)
                log(logging.INFO, "read to the end in %.0fs: %s lines, %.1f MB uncompressed; writing %s",
                    read_s, f"{lines_processed:,}", uncompressed_bytes / 1e6,
                    "nothing (no line was kept)" if not parts else
                    (f"one SST file from {parts} sorted part{'s' if parts != 1 else ''}" if spool_format == "sst"
                     else f"{parts} chunk file{'s' if parts != 1 else ''}"))
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
                spooled = sum(os.path.getsize(p) for p in chunk_paths if os.path.exists(p))
                took = (_monotonic() - started_at, read_s, _monotonic() - attempt_started - read_s)
                if chunk_paths:
                    log(logging.INFO, "done in %.0fs (read %.0fs, spool %.0fs): %s entries from %s lines, in %d %s "
                        "file%s (%.1f MB); waiting for the parent to ingest it", *took, f"{entries:,}",
                        f"{lines_processed:,}", len(chunk_paths), "SST" if spool_format == "sst" else "chunk",
                        "" if len(chunk_paths) == 1 else "s", spooled / 1e6)
                else:
                    log(logging.INFO, "done in %.0fs (read %.0fs, spool %.0fs): nothing kept from %s lines; "
                        "waiting for the parent to mark it processed", *took, f"{lines_processed:,}")
                if problem_lines:
                    log(logging.WARNING, "%s line%s could not be read in full; n-grams on such lines are undercounted",
                        f"{problem_lines:,}", "" if problem_lines == 1 else "s")
                if writer.reordered:
                    log(logging.WARNING, "%s value%s arrived with years out of order and were put in order",
                        f"{writer.reordered:,}", "" if writer.reordered == 1 else "s")
                return msg, chunk_paths, uncompressed_bytes, entries, lines_processed, stats

            except _TIMEOUTS as exc:
                writer.discard()
                last_error = f"TIMEOUT: {filename}"
                detail = str(exc)
            except _BROKEN_DOWNLOADS as exc:
                # Includes connections dropped mid-stream, which surface from
                # inside the gzip loop rather than from the initial request,
                # and a body shorter than the server announced.
                writer.discard()
                last_error = f"NETWORK_ERROR: {filename}"
                detail = f"{type(exc).__name__}: {exc}"

            kind = "the download timed out" if last_error.startswith("TIMEOUT") else "the download broke off"
            if attempt < max_attempts:
                log(logging.WARNING, "%s after %s lines (attempt %d of %d: %s); discarded what was read, "
                    "starting the file again", kind, f"{lines_processed:,}", attempt, max_attempts, detail)
            else:
                log(logging.ERROR, "%s after %s lines (%s); giving up after %d attempts. Nothing of this file "
                    "was ingested", kind, f"{lines_processed:,}", detail, max_attempts)
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
        log(logging.ERROR, "failed with %s: %s. Nothing of this file was ingested", type(exc).__name__, exc)
        return msg, [], 0, 0, 0, {}

    finally:
        if own_session:
            try:
                sess.close()
            except Exception:
                pass


def _worker_slot() -> int:
    """Which of the pool's processes this is: 1..workers, or 0 when the
    caller runs in the main process (a thread pool, a test, a notebook)."""
    identity = getattr(mp.current_process(), "_identity", ())
    return int(identity[0]) if identity else 0


def _received_text(resp, content_length) -> str:
    """", 310.2 of 395.3 MB received" for a log line, as far as it is known."""
    try:
        received = int(resp.raw.tell())
    except (AttributeError, OSError, TypeError, ValueError):
        return ""
    try:
        return f", {received / 1e6:,.1f} of {int(content_length) / 1e6:,.1f} MB received"
    except (TypeError, ValueError):
        return f", {received / 1e6:,.1f} MB received"


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
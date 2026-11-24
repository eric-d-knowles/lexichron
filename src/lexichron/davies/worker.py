"""What one acquire worker does with one archive of a Davies corpus.

It mirrors :func:`ngramprep.ngram_acquire.worker.process_and_ingest_file`
and returns the same tuple, so the parent treats its output like any other
file's: one sorted SST file of summed counts, ingested together with a
marker that records the totals of what was kept.

A record is a unit of text (see :mod:`lexichron.davies.reader`) under its
year: the key is the unit's tokens joined by spaces, the value one
``(year, occurrences, documents)`` triplet. Within a document a unit is
counted as often as it occurs and as one document; the same unit from
other documents and other archives is summed by the writer and the
database, as n-grams are. With ``unit="tokens"`` the records are the
single tokens of those same units instead, counted the same way: the
corpus's 1-grams.
"""
from __future__ import annotations

import logging
import os
import struct
from collections import Counter
from pathlib import Path
from time import monotonic as _monotonic
from typing import Dict, List, Optional, Sequence, Tuple

from lexichron.davies.formats import corpus_format
from lexichron.davies.reader import documents, units
from ngramprep.ngram_acquire.spool import CheckedChunkWriter, CheckedSstChunkWriter
from ngramprep.ngram_acquire.worker import (DEFAULT_CHUNK_ENTRIES, SPOOL_FORMATS, _worker_slot,
                                            merge_packed_records)

__all__ = ["process_archive"]

_REPORT_EVERY_S = 3.0
_PROBLEM_ROWS_LOGGED = 20
_pack = struct.Struct("<3Q").pack


def process_archive(
    path: str,
    worker_id: int,
    log_file_path: Optional[str] = None,
    *,
    corpus: str,
    unit: str = "sentences",
    ngram_type: str = "tagged",
    genres: Optional[Sequence[str]] = None,
    max_unit_tokens: int = 500,
    spool_dir: Optional[str] = None,
    chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
    spool_format: str = "sst",
    files_total: Optional[int] = None,
) -> Tuple[str, List[str], int, int, int, Dict[str, int]]:
    """Read one archive and spool its units.

    Args:
        path: The archive (a zip of the edition ``ngram_type`` reads)
        worker_id: The archive's number in this run; names its spool files
        log_file_path: The run's log file, for this worker's entries
        corpus: Which corpus's format to read it by (``coha``)
        unit: ``sentences`` stores each unit of text as a record; ``tokens``
            stores each token of those units as a record of its own, the
            shape of a 1-gram database
        ngram_type: ``tagged`` or ``untagged``
        genres: Keep only documents of these genres (default: all)
        max_unit_tokens: Longest unit stored; longer text is cut into pieces
        spool_dir: Where the spool files go
        chunk_entries: Entries held in memory before a sorted part is written
        spool_format: ``sst`` or ``chunk``
        files_total: How many archives the run reads, for the log

    Returns:
        ``(status message, spool files, uncompressed bytes read, distinct
        records, tokens read, stats)``. ``stats`` holds the ledger's
        ``match_total`` (occurrences of every unit kept), ``volume_total``
        (their document counts), ``problem_lines`` (rows of a tagged file
        that were not word, lemma, tag) and ``reordered``, and what the
        archive held: ``documents``, ``documents_skipped`` (other genres),
        ``tokens_kept``.
    """
    filename = Path(path).name
    slot = _worker_slot()
    log = logging.getLogger(f"acquire.worker.{slot:02d}")
    if log_file_path and not log.handlers:
        try:
            handler = logging.FileHandler(log_file_path, mode="a")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                                                   datefmt="%Y-%m-%d %H:%M:%S"))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False
        except OSError:
            log = logging.getLogger(__name__)
    which = f"{filename} (file {worker_id}" + (f" of {files_total}" if files_total else "") + ")"

    if spool_format not in SPOOL_FORMATS:
        raise ValueError(f"spool_format must be one of {SPOOL_FORMATS}, got {spool_format!r}")
    import tempfile
    spool_root = spool_dir or tempfile.gettempdir()
    file_tag = f"{worker_id:05d}_{filename}"
    started_path = os.path.join(spool_root, f"{file_tag}.started")
    _report(started_path, 0, 0)

    fmt = corpus_format(corpus)
    if unit not in ("sentences", "tokens"):
        raise ValueError(f"unit must be sentences or tokens, got {unit!r}")
    by_token = unit == "tokens"
    wanted = {str(g).lower() for g in genres} if genres else None
    writer = (CheckedSstChunkWriter if spool_format == "sst" else CheckedChunkWriter)(
        spool_root, file_tag, chunk_entries, merge_packed_records)
    started = last_report = _monotonic()
    tokens_read = tokens_kept = nbytes = docs = skipped = problem_rows = 0
    try:
        log.info("%s: reading the %s edition", which, ngram_type)
        for doc in documents(path, fmt, ngram_type):
            nbytes += doc.nbytes
            tokens_read += len(doc.tokens)
            if doc.unreadable:
                problem_rows += doc.unreadable
                for row in doc.bad_rows:
                    if problem_rows <= _PROBLEM_ROWS_LOGGED:
                        log.warning("%s: %s has a row that is not word, lemma, tag: %r", which, doc.name, row)
            if wanted is not None and doc.genre not in wanted:
                skipped += 1
                continue
            docs += 1
            counts: Counter = Counter()
            if by_token:
                for text in units(doc, fmt, max_unit_tokens):
                    tokens_kept += len(text)
                    counts.update(text)
            else:
                for text in units(doc, fmt, max_unit_tokens):
                    tokens_kept += len(text)
                    counts[" ".join(text)] += 1
            year = doc.year
            for key, n in counts.items():
                writer.add(key, _pack(year, n, 1))
            if _monotonic() - last_report >= _REPORT_EVERY_S:
                last_report = _monotonic()
                _report(started_path, tokens_read, nbytes)
        _report(started_path, tokens_read, nbytes)
        read_s = _monotonic() - started
        chunk_paths = writer.finish()
        entries = writer.entries_total
        try:
            os.replace(started_path, os.path.join(spool_root, f"{file_tag}.parsed"))
        except OSError:
            pass
        stats = {"match_total": writer.match_total, "volume_total": writer.volume_total,
                 "problem_lines": problem_rows, "reordered": writer.reordered,
                 "documents": docs, "documents_skipped": skipped, "tokens_kept": tokens_kept}
        msg = (f"SUCCESS: {filename} - {docs:,} documents, {tokens_read:,} tokens, {entries:,} entries in "
               f"{len(chunk_paths)} chunks, {nbytes:,} uncompressed bytes")
        if problem_rows:
            msg += f", {problem_rows:,} rows not read"
            log.warning("%s: %s row%s could not be read as word, lemma, tag and were left out", which,
                        f"{problem_rows:,}", "" if problem_rows == 1 else "s")
        log.info("%s: done in %.0fs (read %.0fs): %s documents%s, %s tokens read, %s kept; %s %s, %s distinct; "
                 "waiting for the parent to ingest it", which, _monotonic() - started, read_s,
                 f"{docs:,}", f" ({skipped:,} of other genres left out)" if skipped else "", f"{tokens_read:,}",
                 f"{tokens_kept:,}", f"{writer.match_total:,}", "token occurrences" if by_token else "units",
                 f"{entries:,}")
        return msg, chunk_paths, nbytes, entries, tokens_read, stats
    except Exception as exc:
        # Nothing of the archive has been published, so nothing of it is ingested.
        try:
            writer.discard()
        except Exception:
            pass
        log.error("%s: failed with %s: %s. Nothing of this file was ingested", which, type(exc).__name__, exc)
        return f"ERROR: {filename} - {exc}", [], 0, 0, 0, {}


def _report(marker_path: str, tokens: int, nbytes: int) -> None:
    """"<tokens read> <bytes read>" for the parent's progress display."""
    try:
        with open(marker_path, "w") as fh:
            fh.write(f"{int(tokens)} {int(nbytes)}")
    except OSError:
        pass

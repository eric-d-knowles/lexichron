"""Check an acquired database against what the run put into it.

Every shard's marker records the totals of what its worker spooled: the
frequencies (``match_total``) and document counts (``volume_total``) of every
line it kept, summed over all years. Ingestion only ever *adds* counts
together (the same n-gram arriving from two shards, or twice within one, has
its years summed), and adding never changes a grand total. So after
compaction the database must hold exactly the sum of its markers' totals:

* a shard ingested twice makes the database larger than the ledger;
* a shard marked but not ingested, or a record lost on the way, makes it
  smaller;
* a value summed wrongly shows up either in the totals or as years that are
  no longer strictly ascending, which the scan checks for every value.

The scan reads the whole database once, in parallel over key ranges taken
from the database's own file boundaries, in streaming passes that hold a few
thousand values at a time.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ngramprep.common_db.api import open_db, prefix_scan, range_scan
from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX
from ngramprep.ngram_acquire.spool import check_values
from ngramprep.rangemap.units import is_bookkeeping

logger = logging.getLogger(__name__)

__all__ = ["ledger_totals", "scan_totals", "verify_database", "describe_verification"]

_BATCH = 4096
READ_PROFILE = "scan:packed24"


def ledger_totals(db, marker_prefix: bytes = PROCESSED_PREFIX) -> Dict[str, int]:
    """Add up the markers of every shard in ``db`` (or, with another
    ``marker_prefix``, of every work unit of a range-map run).

    ``unledgered`` counts markers that carry no totals: shards acquired by a
    version that did not record them. With any of those the database cannot
    be checked against its ledger (the scan still checks every value's form).
    A range-map unit's marker also holds what the unit *read* from its
    source (``source_match``, ``source_volume``, ``source_disordered``).
    """
    out = {"shards": 0, "unledgered": 0, "entries": 0, "lines": 0, "match_total": 0, "volume_total": 0,
           "problem_lines": 0, "reordered": 0, "scanned": 0, "kept": 0,
           "source_match": 0, "source_volume": 0, "source_disordered": 0}
    for _key, raw in prefix_scan(db, marker_prefix):
        out["shards"] += 1
        try:
            doc = json.loads(bytes(raw).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            doc = None
        if not isinstance(doc, dict) or "match_total" not in doc or "volume_total" not in doc:
            out["unledgered"] += 1
            continue
        for name in ("entries", "lines", "match_total", "volume_total", "problem_lines", "reordered", "scanned",
                     "kept", "source_match", "source_volume", "source_disordered"):
            out[name] += int(doc.get(name, 0))
    return out


def _tally(values: List[bytes], out: Dict[str, int]) -> None:
    try:
        match, volume, bad = check_values(values)
    except ValueError:
        # A value that is not whole triplets: count it, and take the rest one by one.
        for value in values:
            try:
                match, volume, bad = check_values([value])
            except ValueError:
                out["malformed"] += 1
                continue
            out["match_total"] += match
            out["volume_total"] += volume
            out["disordered"] += len(bad)
        return
    out["match_total"] += match
    out["volume_total"] += volume
    out["disordered"] += len(bad)


def _scan_range(db_path: str, lower: Optional[bytes], upper: Optional[bytes], profile: str) -> Dict[str, int]:
    out = {"records": 0, "match_total": 0, "volume_total": 0, "disordered": 0, "malformed": 0}
    batch: List[bytes] = []
    with open_db(db_path, mode="r", profile=profile) as db:
        for key, value in range_scan(db, lower or b"", upper, fill_cache=False, readahead_size=2 << 20):
            if is_bookkeeping(key):
                continue
            batch.append(bytes(value))
            if len(batch) >= _BATCH:
                out["records"] += len(batch)
                _tally(batch, out)
                batch = []
    if batch:
        out["records"] += len(batch)
        _tally(batch, out)
    return out


def _ranges(db_path: str, parts: int, profile: str) -> List[Tuple[Optional[bytes], Optional[bytes]]]:
    """Key ranges covering the whole database, cut at file boundaries so that
    after a compaction each range is a run of whole files."""
    if parts <= 1:
        return [(None, None)]
    try:
        with open_db(db_path, mode="r", profile=profile) as db:
            files = db.live_files()
    except Exception as exc:                      # a store without file metadata: one range
        logger.info("No file metadata for %s (%s); scanning it as one range", db_path, exc)
        return [(None, None)]
    starts = sorted({bytes(f[0]) for f in files})[1:]        # the first file's start is the open end
    if not starts:
        return [(None, None)]
    step = max(1, len(starts) // parts)
    cuts = starts[step - 1::step] if len(starts) > parts else starts
    bounds: List[Optional[bytes]] = [None] + cuts + [None]
    return list(zip(bounds[:-1], bounds[1:]))


def scan_totals(db_path: str, workers: int = 1, *, profile: str = READ_PROFILE) -> Dict[str, int]:
    """Read every data record of the database at ``db_path`` and return
    ``{"records", "match_total", "volume_total", "disordered", "malformed"}``:
    how many there are, what their counts add up to, and how many values have
    years out of order or are not whole triplets (both must be zero).

    The database must not be open for writing elsewhere while this runs.
    """
    workers = max(1, int(workers))
    ranges = _ranges(str(db_path), workers * 4, profile)
    total = {"records": 0, "match_total": 0, "volume_total": 0, "disordered": 0, "malformed": 0}
    if len(ranges) == 1 or workers == 1:
        results = [_scan_range(str(db_path), lo, hi, profile) for lo, hi in ranges]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(ranges)), mp_context=mp.get_context("fork")) as pool:
            results = list(pool.map(_scan_range, [str(db_path)] * len(ranges), [r[0] for r in ranges],
                                    [r[1] for r in ranges], [profile] * len(ranges)))
    for part in results:
        for name in total:
            total[name] += part[name]
    return total


def verify_database(db_path: str, workers: int = 1, *, profile: str = READ_PROFILE,
                    marker_prefix: bytes = PROCESSED_PREFIX) -> Dict[str, Any]:
    """Compare the database with its ledger. Returns a record with ``status``:

    * ``"verified"``: the counts in the database equal the sum of what every
      shard recorded, and every value is well formed;
    * ``"mismatch"``: they differ, or a value is malformed or has years out
      of order. The counts cannot be trusted;
    * ``"unverifiable"``: some shards were acquired by a version that kept no
      totals, so there is nothing to compare with (every value is still well
      formed, or the status would be ``"mismatch"``).
    """
    with open_db(db_path, mode="r", profile=profile) as db:
        ledger = ledger_totals(db, marker_prefix)
    found = scan_totals(db_path, workers, profile=profile)
    if found["disordered"] or found["malformed"]:
        status = "mismatch"
    elif ledger["unledgered"]:
        status = "unverifiable"
    elif (found["match_total"], found["volume_total"]) == (ledger["match_total"], ledger["volume_total"]):
        status = "verified"
    else:
        status = "mismatch"
    return {"status": status, "ledger": ledger, "database": found}


def describe_verification(v: Dict[str, Any]) -> List[str]:
    """The verification record in sentences, for a run's summary."""
    ledger, found = v.get("ledger") or {}, v.get("database") or {}
    lines = []
    status = v.get("status")
    if status == "verified":
        lines.append(f"Counts verified: the database's {found.get('records', 0):,} records hold "
                     f"{found.get('match_total', 0):,} occurrences in {found.get('volume_total', 0):,} "
                     f"volume counts, exactly what the {ledger.get('shards', 0):,} files contributed.")
    elif status == "unverifiable":
        lines.append(f"Counts not verifiable: {ledger.get('unledgered', 0):,} of {ledger.get('shards', 0):,} files "
                     f"were acquired by a version that kept no totals. Every value in the database is well formed.")
    elif status == "mismatch":
        lines.append("COUNTS DO NOT MATCH what the files contributed:")
        lines.append(f"  occurrences:   database {found.get('match_total', 0):,}   files {ledger.get('match_total', 0):,}"
                     f"   difference {found.get('match_total', 0) - ledger.get('match_total', 0):+,}")
        lines.append(f"  volume counts: database {found.get('volume_total', 0):,}   files {ledger.get('volume_total', 0):,}"
                     f"   difference {found.get('volume_total', 0) - ledger.get('volume_total', 0):+,}")
        if ledger.get("unledgered"):
            lines.append(f"  ({ledger['unledgered']:,} files kept no totals, so the difference above is not meaningful)")
        if found.get("disordered"):
            lines.append(f"  {found['disordered']:,} values have years out of order")
        if found.get("malformed"):
            lines.append(f"  {found['malformed']:,} values are not whole (year, frequency, documents) triplets")
    if ledger.get("problem_lines"):
        n = int(ledger["problem_lines"])
        lines.append(f"{n:,} line{'' if n == 1 else 's'} had content that could not be read; the counts of the "
                     f"n-grams on them are incomplete (the workers' log entries show the lines).")
    if ledger.get("reordered"):
        n = int(ledger["reordered"])
        lines.append(f"{n:,} value{'' if n == 1 else 's'} arrived with years out of order and "
                     f"{'was' if n == 1 else 'were'} put in order.")
    return lines

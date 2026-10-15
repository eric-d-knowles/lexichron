"""Turn a run's ``progress.json`` into numbers and text for display.

Pure functions, so the UI's progress panel can be tested without a terminal.
"""
from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["summarize", "meters", "fmt_duration", "fmt_bytes", "fmt_count", "tail_lines"]


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_count(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


LONG_PHASES = {
    "partitioning": "partitioning the source (sampling the key space)",
    "compacting": "compacting the database (all work ingested)",
    "archiving": "archiving the database",
    "whitelist": "writing the vocabulary",
    "corpus": "writing the year's corpus file(s)",
}


def summarize(doc: Dict[str, Any], now: Optional[datetime] = None, *,
              job_gone: bool = False) -> Dict[str, Any]:
    """Derived figures for one progress document.

    Returns a dict with ``done``, ``total`` (files, excluding skipped),
    ``percent`` (None until the total is known), ``elapsed_s``, ``rate``
    (entries per second), ``eta_s`` (None unless running with progress),
    ``stale`` (True when a running job has not written for a while),
    ``state_label`` (``running``, ``running?`` when stale, ``stopped`` when
    the caller knows the job has left the queue — ``job_gone`` — or the
    recorded final state) and ready-made ``headline`` / ``detail`` /
    ``current_text`` strings.
    """
    now = now or datetime.now(timezone.utc)
    state = doc.get("state", "running")
    started = _parse(doc.get("started"))
    updated = _parse(doc.get("updated"))
    finished = _parse(doc.get("finished"))
    end = finished if (state != "running" and finished) else now
    elapsed = (end - started).total_seconds() if started else None

    total = max(int(doc.get("files_total", 0)) - int(doc.get("files_skipped", 0)), 0)
    done = int(doc.get("files_done", 0))
    failed = int(doc.get("files_failed", 0))
    entries = int(doc.get("entries_written", 0))
    percent = (100.0 * done / total) if total else None
    rate = (entries / elapsed) if elapsed and elapsed > 0 else 0.0

    eta = None
    if state == "running" and total and done and elapsed:
        remaining = total - done - failed
        eta = max(remaining * (elapsed / done), 0.0)

    stale = False
    if state == "running" and updated:
        stale = (now - updated).total_seconds() > 300
    stopped = state == "running" and job_gone
    if stopped:
        # The job is gone, so the clock stopped at its last write.
        end = updated or now
        elapsed = (end - started).total_seconds() if started else elapsed
        rate = (entries / elapsed) if elapsed and elapsed > 0 else 0.0
        eta = None
    long_phase = state == "running" and (doc.get("phase") or "ingesting") in LONG_PHASES
    state_label = ("stopped" if stopped else "running?" if stale
                   else doc["phase"] if long_phase else state)

    where = doc.get("hostname") or ""
    job = doc.get("slurm_job_id")
    where_txt = ""
    if job:
        where_txt = f"job {job}" + (f" on {where}" if where else "")
    elif where:
        where_txt = f"on {where}"

    phase = doc.get("phase") or "ingesting"
    phase_started = _parse(doc.get("phase_started"))
    if stopped:
        status = "stopped (job no longer in the queue; resume by running again)"
    elif state == "running" and stale:
        status = "running? (no update for " + fmt_duration((now - updated).total_seconds()) + ")"
    elif state == "running" and phase in LONG_PHASES:
        for_how_long = fmt_duration((now - phase_started).total_seconds()) if phase_started else "-"
        status = f"{LONG_PHASES[phase]} ({for_how_long} so far)"
        eta = None
    else:
        status = state
    parts = [status + (f" {where_txt}" if where_txt else "")]
    unit = unit_label(doc)
    if total:
        parts.append(f"{done}/{total} {unit}" + (f" ({percent:.0f}%)" if percent is not None else ""))
    else:
        parts.append(f"{done} {unit}")
    if failed:
        parts.append(f"{failed} failed")
    if doc.get("files_skipped"):
        parts.append(f"{doc['files_skipped']} already done")
    headline = " · ".join(parts)

    counters = doc.get("counters") or {}
    detail = [f"{entries:,} entries"]
    if doc.get("uncompressed_bytes"):
        detail.append(f"{fmt_bytes(float(doc['uncompressed_bytes']))} parsed (uncompressed)")
    if counters.get("scanned"):
        kept = int(counters.get("kept", 0))
        pct = 100.0 * kept / int(counters["scanned"]) if counters["scanned"] else 0.0
        detail.append(f"{int(counters['scanned']):,} scanned, {kept:,} kept ({pct:.1f}%)")
    detail += [f"{fmt_count(rate)} entries/s", f"elapsed {fmt_duration(elapsed)}"]
    if eta is not None:
        detail.append(f"about {fmt_duration(eta)} left")
    current = list(doc.get("current") or [])
    phases = doc.get("phases") or {}
    # What a worker does with a unit differs by stage: acquisition downloads
    # and parses a file, filtering scans a key range.
    if doc.get("stage") == "filter":
        phase_labels = (("queued", "waiting for a worker"), ("parsing", "scanning"),
                        ("parsed", "scanned, waiting to ingest"), ("ingesting", "ingesting"))
    elif doc.get("stage") == "tune":
        phase_labels = (("queued", "waiting"), ("parsing", "training"),
                        ("parsed", "trained, waiting to score"), ("ingesting", "scoring"))
    else:
        phase_labels = (("queued", "waiting for a worker"), ("parsing", "downloading & parsing"),
                        ("parsed", "parsed, waiting to ingest"), ("ingesting", "ingesting"))
    if not current:
        current_text = ""
    elif phases:
        bits = []
        for key, label in phase_labels:
            n = int(phases.get(key, 0))
            if n:
                bits.append(f"{n} {label}")
        current_text = f"{len(current)} {unit} in flight: " + " · ".join(bits)
        if doc.get("ingesting"):
            current_text += f"  ({doc['ingesting']})"
    else:
        current_text = f"{len(current)} {unit} in flight"
    inputs = doc.get("inputs") or {}
    labels = {"source": "Source", "vocabulary": "Vocabulary", "always_include": "Always keep"}
    inputs_text = "  ·  ".join(f"{labels.get(k, k)}: {v}" for k, v in inputs.items() if v)
    if doc.get("stage") == "filter" and "vocabulary" in inputs and not inputs.get("vocabulary"):
        inputs_text += "  ·  Vocabulary: none (rule filters only)"
    corpus = doc.get("corpus") or {}
    corpus_text = ""
    # The whole-corpus total is acquisition's: files downloaded over several
    # jobs. A filter run's units are not files and carry no byte counts.
    if doc.get("stage") != "filter" and corpus and (corpus.get("files") or 0) > 0:
        all_files = int(doc.get("files_total", 0))
        n = int(corpus.get("files", 0))
        parts_c = [f"{n}/{all_files} files" if all_files else f"{n} files",
                   f"{int(corpus.get('entries', 0)):,} entries",
                   f"{fmt_bytes(float(corpus.get('bytes', 0)))} parsed (uncompressed)"]
        text = "Whole corpus so far: " + " · ".join(parts_c)
        unsized = int(corpus.get("unsized", 0))
        if unsized:
            text += f"  (excludes {unsized} files done before sizes were recorded)"
        corpus_text = text

    return {
        "state": state, "state_label": state_label, "done": done, "total": total, "failed": failed,
        "percent": percent, "elapsed_s": elapsed, "rate": rate, "eta_s": eta,
        "stale": stale, "stopped": stopped, "phase": phase,
        "headline": headline, "detail": " · ".join(detail),
        "current": current, "current_text": current_text, "corpus_text": corpus_text,
        "inputs_text": inputs_text,
        "message": doc.get("message") or "",
        "log_path": doc.get("log_path"), "db_path": doc.get("db_path"),
    }


def tail_lines(path: Optional[str | os.PathLike], n: int = 20) -> List[str]:
    """Last ``n`` lines of a text file (empty if it cannot be read)."""
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=n)]
    except (OSError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Meters: the figures of a run as labelled values, grouped, for the UI's grid.
# ---------------------------------------------------------------------------

def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else ""


def unit_label(doc: Dict[str, Any]) -> str:
    """What a run's progress counts: files for acquisition, work units (key
    ranges) for filtering. Filter documents written before the label was
    recorded still count work units."""
    return doc.get("unit_label") or ("work units" if doc.get("stage") == "filter" else "files")


def meters(doc: Dict[str, Any], info: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """Groups of ``(label, value, note)`` for one run, from its progress
    document and :func:`summarize`'s figures. The groups depend on the stage.

    Three record counts are kept apart on purpose: what was read (raw data
    lines, or source records), what was handed to the database (fewer:
    duplicates merge before insertion), and what the final database holds
    (fewer again: duplicates merge across insertions, once compacted)."""
    stage = doc.get("stage") or "acquire"
    counters = doc.get("counters") or {}
    database = doc.get("database") or {}
    unit = unit_label(doc)
    done, total, failed = info["done"], info["total"], info["failed"]

    progress_tiles = [
        (unit.capitalize(), f"{done:,}/{total:,}" if total else f"{done:,}",
         " · ".join(x for x in [(f"{info['percent']:.0f}%" if info["percent"] is not None else ""),
                                (f"{failed} failed" if failed else ""),
                                (f"{doc['files_skipped']} already done" if doc.get("files_skipped") else "")] if x)),
        (("Rate", f"{done / info['elapsed_s'] * 3600:.1f}/h" if info["elapsed_s"] else "—", "models")
         if stage == "tune" else
         ("Rate", f"{fmt_count(info['rate'])}/s", "entries kept" if stage == "filter" else "entries")),
        ("Elapsed", fmt_duration(info["elapsed_s"]), ""),
        ("Remaining", fmt_duration(info["eta_s"]) if info["eta_s"] is not None else "—", "estimate"),
    ]

    def db_tile():
        n = database.get("records")
        if n is None:
            return ("In database", "—", "after compaction")
        return ("In database", f"{n:,}",
                "after compaction" if database.get("compacted") else "estimate: not yet compacted")

    groups = [("Progress", progress_tiles)]
    current = list(doc.get("current") or [])
    phases = doc.get("phases") or {}
    # Shown for the whole of a running run, count zero included, so the group
    # does not come and go between work units (the UI rebuilds its tiles
    # when the set of groups changes).
    if phases and (current or info.get("state") == "running"):
        if stage == "filter":
            labels = (("queued", "Waiting for a worker"), ("parsing", "Scanning"),
                      ("parsed", "Scanned, waiting to ingest"), ("ingesting", "Ingesting"))
        elif stage == "tune":
            labels = (("queued", "Waiting"), ("parsing", "Training"),
                      ("parsed", "Trained, waiting to score"), ("ingesting", "Scoring"))
        else:
            labels = (("queued", "Waiting for a worker"), ("parsing", "Downloading & parsing"),
                      ("parsed", "Parsed, waiting to ingest"), ("ingesting", "Ingesting"))
        tiles = [(label, f"{int(phases.get(key, 0)):,}", "") for key, label in labels]
        groups.append((f"In flight: {len(current):,} {unit}", tiles))
    if stage == "tune":
        groups.append(("Search", [
            ("Models trained", f"{int(counters.get('models', 0)):,}", "this run"),
            ("Training time", fmt_duration(float(counters.get("train_seconds", 0))), "summed over models"),
            ("N-grams used", f"{int(counters.get('ngrams_used', 0)):,}", "after the <UNK> policy"),
            ("Corpus lines", f"{int(counters.get('corpus_lines', 0)):,}", "weighted repeats"),
        ]))
    elif stage == "filter":
        scanned = int(counters.get("scanned", 0))
        kept = int(counters.get("kept", 0))
        produced = int(counters.get("produced", kept))
        tiles = [("Scanned", f"{scanned:,}", "source records"),
                 ("Kept", f"{kept:,}", _pct(kept, scanned) + " of scanned" if scanned else "")]
        if produced > kept:
            # the pivot makes one record per n-gram and year from each kept n-gram
            tiles.append(("Year records", f"{produced:,}", f"{produced / kept:.1f} per n-gram" if kept else ""))
        tiles += [("Inserted", f"{int(counters.get('inserted', 0)):,}", "to the database"), db_tile()]
        groups.append(("Records", tiles))
        tokens = int(counters.get("tokens", 0))
        ngrams = int(counters.get("ngrams", 0))
        if tokens:
            # With a vocabulary the kernel applies only the vocabulary rule;
            # without one, only the three others. Show the rules that ran.
            with_vocab = bool((doc.get("inputs") or {}).get("vocabulary"))
            rules = ([("Not in vocabulary", "tokens_not_in_vocab")] if with_vocab else
                     [("Not alphabetic", "tokens_not_alpha"), ("Too short", "tokens_too_short"),
                      ("Stopword", "tokens_stopword")])
            tiles = [(label, f"{int(counters.get(key, 0)):,}", _pct(int(counters.get(key, 0)), tokens) + " of tokens")
                     for label, key in rules]
            dropped = int(counters.get("ngrams_dropped_context", 0))
            tiles.append(("Too few real tokens", f"{dropped:,}", _pct(dropped, ngrams) + " of n-grams"))
            groups.append(("Dropped by rule", tiles))
    else:
        # This session's figures, then the running totals over every
        # acquisition job that has contributed to the database. The database
        # itself is cumulative, so its record count belongs with the latter.
        lines = int(counters.get("lines", 0))
        groups.append(("Records \u2013 this session", [
            ("Lines read", f"{lines:,}", "raw data lines"),
            ("Parsed", fmt_bytes(float(doc.get("uncompressed_bytes") or 0)), "uncompressed"),
            ("Inserted", f"{int(counters.get('inserted', doc.get('entries_written', 0))):,}", "to the database"),
        ]))
        corpus = doc.get("corpus") or {}
        tiles = []
        if "lines" in corpus:
            tiles.append(("Lines read", f"{int(corpus.get('lines', 0)):,}", "raw data lines"))
        tiles += [("Parsed", fmt_bytes(float(corpus.get("bytes", 0))), "uncompressed"),
                  ("Inserted", f"{int(corpus.get('entries', 0)):,}", "to the database"),
                  db_tile()]
        unsized = int(corpus.get("unsized", 0))
        if unsized:
            tiles.append(("Unsized", f"{unsized:,}", "files done before sizes were recorded"))
        groups.append(("Records \u2013 entire corpus", tiles))
    return groups

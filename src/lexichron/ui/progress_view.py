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
    "verifying": "verifying the counts (reading the whole database once)",
    "archiving": "archiving the database",
    "whitelist": "writing the vocabulary",
    "corpus": "writing the year's corpus file(s)",
    "training": "training the year's models",
}

# The steps of a run that declares them (acquire: see ProgressReporter.set_plan):
# what each is called while it runs, and when it is still to come.
STEP_NOW = {"ingesting": "downloading and parsing the files", **LONG_PHASES}
STEP_LATER = {"compacting": "compaction", "verifying": "the count check", "archiving": "archiving"}


def summarize(doc: Dict[str, Any], now: Optional[datetime] = None, *,
              job_gone: bool = False) -> Dict[str, Any]:
    """Derived figures for one progress document.

    Returns a dict with ``done``, ``total`` (files, excluding skipped),
    ``percent`` (None until the total is known), ``fraction`` (0-1: by size
    where the run knows its files' sizes, else by count), ``elapsed_s``,
    ``rate`` and ``rate_unit`` (entries per second; for an acquisition, the
    raw n-grams scanned per second while the files were being read),
    ``eta_s`` (None unless running with progress; of the files, not of the
    steps that follow them: ``then_text`` names those), ``step_text``
    ("step 2 of 3" for a run that declares its steps), ``warnings`` (what
    has gone wrong so far and what it will lead to),
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
    counters = doc.get("counters") or {}
    in_flight = doc.get("in_flight") or {}

    stale = False
    if state == "running" and updated:
        stale = (now - updated).total_seconds() > 300
    stopped = state == "running" and job_gone
    if stopped:
        # The job is gone, so the clock stopped at its last write.
        end = updated or now
        elapsed = (end - started).total_seconds() if started else elapsed

    # The time spent on the files themselves. A run that declares its steps
    # goes on to others (compaction, the count check) that are single long
    # calls: they are not part of any rate, and no rate predicts them.
    steps = [str(x) for x in (doc.get("steps") or [])]
    work_elapsed = elapsed
    first = (doc.get("phase_times") or {}).get(steps[0]) if steps else None
    if first and started and _parse(first.get("finished")):
        work_elapsed = (min(_parse(first["finished"]), end) - started).total_seconds()

    # Throughput. An acquisition reads raw n-grams (one per line) and keeps
    # some: what it reads per second is the steady figure, whatever the
    # files in flight happen to yield. Workers report as they go, so files
    # not yet finished count too.
    # (Known from the first frame: a run that reports files in flight says so
    # before any line is read, so the screen does not change its tiles later.)
    scanning = doc.get("stage", "acquire") == "acquire" and ("lines" in counters or "in_flight" in doc)
    if scanning:
        scanned = int(counters.get("lines", 0)) + int(in_flight.get("lines", 0))
        rate = (scanned / work_elapsed) if work_elapsed and work_elapsed > 0 else 0.0
        rate_unit = "n-grams scanned/s"
    else:
        rate = (entries / elapsed) if elapsed and elapsed > 0 else 0.0
        rate_unit = "entries/s"

    # How far along: by size where the sizes are known (files differ by
    # orders of magnitude in what they hold), else by count.
    bytes_total = doc.get("bytes_total")
    fetched = 0.0
    if bytes_total:
        fetched = min(float(doc.get("bytes_done", 0)) + float(in_flight.get("bytes", 0)), float(bytes_total))
        fraction = fetched / float(bytes_total)
    else:
        fraction = ((done + failed) / total) if total else None
    by_size = bool(bytes_total)

    eta = None
    if state == "running" and not stopped and work_elapsed:
        if by_size and fetched > 0:
            eta = max((float(bytes_total) - fetched) * (work_elapsed / fetched), 0.0)
        elif not by_size and total and done:
            remaining = total - done - failed
            eta = max(remaining * (work_elapsed / done), 0.0)
    long_phase = state == "running" and (doc.get("phase") or "ingesting") in LONG_PHASES
    state_label = ("stopped" if stopped else "running?" if stale
                   else doc["phase"] if long_phase else state)

    where = doc.get("hostname") or ""
    job = doc.get("slurm_job_id")
    array = doc.get("array") or {}
    task_txt = f"task {int(array['task']) + 1} of {array['count']}" if array.get("count") else ""
    where_txt = ""
    if job:
        where_txt = f"job {job}" + (f" ({task_txt})" if task_txt else "") + (f" on {where}" if where else "")
    elif where:
        where_txt = (f"{task_txt} " if task_txt else "") + f"on {where}"
    if task_txt and array.get("years"):
        ys = [int(y) for y in array["years"]]
        where_txt += f"; years {', '.join(str(y) for y in ys)}" if len(ys) <= 8 else f"; {len(ys)} years"

    phase = doc.get("phase") or "ingesting"
    phase_started = _parse(doc.get("phase_started"))
    # "step 2 of 3", and what follows the files, for a run that declares its steps
    step_text = f"step {steps.index(phase) + 1} of {len(steps)}" if phase in steps else ""
    later = [STEP_LATER.get(x, x) for x in steps[steps.index(phase) + 1:]] if phase in steps else []
    if step_text and state == "done":
        step_text, later = f"all {len(steps)} steps done", []
    elif step_text and (state != "running" or stopped):
        step_text, later = f"ended in {step_text}", []
    then_text = ("then " + (", ".join(later[:-1]) + " and " + later[-1] if len(later) > 1 else later[0])) if later else ""
    if stopped:
        status = "stopped (job no longer in the queue; resume by running again)"
    elif state == "running" and stale:
        status = "running? (no update for " + fmt_duration((now - updated).total_seconds()) + ")"
    elif state == "running" and phase in LONG_PHASES:
        for_how_long = fmt_duration((now - phase_started).total_seconds()) if phase_started else "-"
        status = (f"{step_text}: " if step_text else "") + f"{LONG_PHASES[phase]} ({for_how_long} so far)"
        eta = None
    elif state == "running" and step_text:
        status = f"{step_text}: {STEP_NOW.get(phase, phase)}"
    else:
        status = state
    parts = [status + (f" {where_txt}" if where_txt else "")]
    unit = unit_label(doc)
    if total and by_size:
        parts.append(f"{done}/{total} {unit} ({100.0 * fraction:.0f}% by size)")
    elif total:
        parts.append(f"{done}/{total} {unit}" + (f" ({percent:.0f}%)" if percent is not None else ""))
    else:
        parts.append(f"{done} {unit}")
    if failed:
        parts.append(f"{failed} failed")
    if doc.get("files_skipped"):
        parts.append(f"{doc['files_skipped']} already done")
    headline = " · ".join(parts)

    detail = [f"{entries:,} entries"]
    if doc.get("uncompressed_bytes"):
        detail.append(f"{fmt_bytes(float(doc['uncompressed_bytes']))} parsed (uncompressed)")
    if counters.get("scanned"):
        kept = int(counters.get("kept", 0))
        pct = 100.0 * kept / int(counters["scanned"]) if counters["scanned"] else 0.0
        detail.append(f"{int(counters['scanned']):,} scanned, {kept:,} kept ({pct:.1f}%)")
    detail += [f"{fmt_count(rate)} {rate_unit}", f"elapsed {fmt_duration(elapsed)}"]
    if eta is not None:
        detail.append(f"about {fmt_duration(eta)} left" + (f" for the files, {then_text}" if then_text else ""))
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
    elif doc.get("stage") == "train":
        phase_labels = (("queued", "waiting"), ("parsing", "training"),
                        ("parsed", "trained, year not yet complete"), ("ingesting", "writing the year's record"))
    elif doc.get("stage") == "measure":
        phase_labels = (("queued", "waiting"), ("parsing", "measuring"),
                        ("parsed", "measured"), ("ingesting", "writing"))
    else:
        # The parent only adopts a finished file (one metadata operation), so
        # "parsed" and "being ingested" are one short wait: for the database.
        phases = dict(phases, handover=int(phases.get("parsed", 0)) + int(phases.get("ingesting", 0))) if phases else phases
        phase_labels = (("queued", "waiting for a worker"), ("parsing", "downloading & parsing"),
                        ("handover", "waiting for the database"))
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
    labels = {"source": "Source", "vocabulary": "Vocabulary", "always_include": "Always keep",
              "always_include_case": "Always keep (match case)"}
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
        "percent": percent, "fraction": fraction, "by_size": by_size,
        "elapsed_s": elapsed, "rate": rate, "rate_unit": rate_unit, "scanning": scanning, "eta_s": eta,
        "step_text": step_text, "then_text": then_text, "warnings": warnings(doc, state=state, stopped=stopped),
        "stale": stale, "stopped": stopped, "phase": phase,
        "headline": headline, "detail": " · ".join(detail),
        "current": current, "current_text": current_text, "corpus_text": corpus_text,
        "inputs_text": inputs_text,
        "message": doc.get("message") or "",
        "log_path": doc.get("log_path"), "db_path": doc.get("db_path"),
    }


def warnings(doc: Dict[str, Any], *, state: Optional[str] = None, stopped: bool = False) -> List[str]:
    """What has gone wrong in a run so far, each with what it leads to, so
    that nobody learns at the end of a long job what the screen could have
    said at the start. For an acquisition: files that failed, lines whose
    counts could not be read, and a count check that did not come out even."""
    state = state or doc.get("state", "running")
    running = state == "running" and not stopped
    out: List[str] = []
    if (doc.get("stage") or "acquire") != "acquire":
        return out
    failed = int(doc.get("files_failed", 0))
    if failed:
        names = [f.get("file", "") for f in (doc.get("failed_files") or [])][:3]
        named = (" (" + ", ".join(names) + (", ..." if failed > len(names) else "") + ")") if names else ""
        out.append(f"{failed:,} file{'s' if failed != 1 else ''} failed{named}. "
                   + ("The run will finish the other files and then stop with an error, without compacting or "
                      "checking the counts; run it again to retry them." if running else
                      "The database is incomplete and was not compacted; run again to retry them."))
    corpus = doc.get("corpus") or {}
    counters = doc.get("counters") or {}
    ledger = (doc.get("verification") or {}).get("ledger") or {}
    unreadable = int(ledger.get("problem_lines", corpus.get("problem_lines", counters.get("problem_lines", 0))))
    if unreadable:
        accepted = bool((doc.get("checks") or {}).get("allow_unparsed_lines"))
        text = (f"{unreadable:,} line{'s' if unreadable != 1 else ''} had counts that could not be read in full, "
                f"so their n-grams are undercounted (the log shows the lines). ")
        if accepted:
            text += "Accepted: allow_unparsed_lines is on."
        elif running:
            text += "The run will finish its work and then end with an error; allow_unparsed_lines accepts them."
        else:
            text += "The run ended with an error for it; allow_unparsed_lines accepts them."
        out.append(text)
    verification = doc.get("verification") or {}
    if verification.get("status") == "mismatch":
        found = verification.get("database") or {}
        text = "The counts in the database do not match what the files contributed"
        if not ledger.get("unledgered"):
            text += (f": occurrences differ by {int(found.get('match_total', 0)) - int(ledger.get('match_total', 0)):+,}, "
                     f"volume counts by {int(found.get('volume_total', 0)) - int(ledger.get('volume_total', 0)):+,}")
        bad = int(found.get("disordered", 0)) + int(found.get("malformed", 0))
        if bad:
            text += f"; {bad:,} values are malformed or have years out of order"
        out.append(text + ". Do not use this database.")
    return out


def notice_text(info: Dict[str, Any]) -> str:
    """The lines shown under a run's headline: its warnings, then the run's
    own last message unless a warning already says it (a run that ends over
    failed files, unread lines or a count mismatch records that as its
    message too)."""
    lines = list(info.get("warnings") or [])
    message = info.get("message") or ""
    covered = lines and any(key in message for key in ("failed", "could not be read", "do not match", "_ERROR", "TIMEOUT"))
    if message and not covered:
        lines.append(message)
    return "\n".join(lines)


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

    # How far: by size where the run knows its files' sizes (an acquisition
    # does, from the listing), since a count of very unequal files misleads.
    if info.get("by_size") and info.get("fraction") is not None:
        how_far = f"{100.0 * info['fraction']:.0f}% by size"
    else:
        how_far = f"{info['percent']:.0f}%" if info["percent"] is not None else ""
    # What the estimate covers. A run that declares its steps has more to do
    # after its files, in single long calls that nothing predicts.
    then = info.get("then_text") or ""
    if info["eta_s"] is not None:
        remaining_note = f"for the files; {then}" if then else "estimate"
    elif info.get("state") == "running" and doc.get("phase") in LONG_PHASES and info.get("step_text"):
        remaining_note = "no estimate for this step" + (f"; {then}" if then else "")
    else:
        remaining_note = "estimate"
    progress_tiles = [
        (unit.capitalize(), f"{done:,}/{total:,}" if total else f"{done:,}",
         " · ".join(x for x in [how_far,
                                (f"{failed} failed" if failed else ""),
                                (f"{doc['files_skipped']} already done" if doc.get("files_skipped") else "")] if x)),
        (("Rate", f"{done / info['elapsed_s'] * 3600:.1f}/h" if info["elapsed_s"] else "—",
          "years" if stage == "measure" else "models")
         if stage in ("tune", "train", "measure") else
         ("Scan rate", f"{fmt_count(info['rate'])}/s", "raw n-grams read, while reading files")
         if info.get("scanning") else
         ("Rate", f"{fmt_count(info['rate'])}/s", "entries kept" if stage == "filter" else "entries")),
        ("Elapsed", fmt_duration(info["elapsed_s"]), info.get("step_text") or ""),
        ("Remaining", fmt_duration(info["eta_s"]) if info["eta_s"] is not None else "—", remaining_note),
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
        elif stage == "train":
            labels = (("queued", "Waiting"), ("parsing", "Training"),
                      ("parsed", "Trained, year not yet complete"), ("ingesting", "Writing the year's record"))
        elif stage == "measure":
            labels = (("queued", "Waiting"), ("parsing", "Measuring"), ("parsed", "Measured"), ("ingesting", "Writing"))
        else:
            # A finished file is adopted by the database in one short step,
            # so "parsed" and "being ingested" are a single wait. It should
            # stay near zero: a number that grows means the parent has
            # fallen behind its workers.
            phases = dict(phases, handover=int(phases.get("parsed", 0)) + int(phases.get("ingesting", 0)))
            labels = (("queued", "Waiting for a worker"), ("parsing", "Downloading & parsing"),
                      ("handover", "Waiting for the database"))
        tiles = [(label, f"{int(phases.get(key, 0)):,}", "") for key, label in labels]
        groups.append((f"In flight: {len(current):,} {unit}", tiles))
    if stage == "tune":
        groups.append(("Search", [
            ("Models trained", f"{int(counters.get('models', 0)):,}", "this run"),
            ("Training time", fmt_duration(float(counters.get("train_seconds", 0))), "summed over models"),
            ("N-grams used", f"{int(counters.get('ngrams_used', 0)):,}", "after the <UNK> policy"),
            ("Corpus lines", f"{int(counters.get('corpus_lines', 0)):,}", "weighted repeats"),
        ]))
    elif stage == "train":
        years, years_done = int(counters.get("years", 0)), int(counters.get("years_done", 0))
        groups.append(("Ensemble", [
            ("Models trained", f"{int(counters.get('models', 0)):,}", "this run"),
            ("Years complete", f"{years_done:,}" + (f"/{years:,}" if years else ""), "records written"),
            ("Training time", fmt_duration(float(counters.get("train_seconds", 0))), "summed over models"),
            ("Corpus lines", f"{int(counters.get('corpus_lines', 0)):,}", "weighted repeats, all resamples"),
        ]))
        groups += ensemble_groups(doc.get("ensemble") or {})
    elif stage == "measure":
        years, years_done = int(counters.get("years", 0)), int(counters.get("years_done", 0))
        groups.append(("Measure", [
            ("Years measured", f"{years_done:,}" + (f"/{years:,}" if years else ""), "this run"),
            ("Years pending", f"{int(counters.get('years_pending', 0)):,}", "models not all trained yet"),
            ("Models read", f"{int(counters.get('models', 0)):,}", "this run"),
            ("Words", f"{int(counters.get('words', 0)):,}", "series"),
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
        groups += acquire_groups(doc, info, db_tile())
    return groups


def acquire_groups(doc: Dict[str, Any], info: Dict[str, Any], db_tile) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """An acquisition's records and checks.

    Records: this session's figures, then the running totals over every
    acquisition job that has added to the database (the database is
    cumulative, so its record count belongs with the latter). In each:
    the raw n-grams *scanned* (one per data line), how many were *kept* by
    the token-type filter, and the files that held nothing to keep, which
    for tagged n-grams is most of them and does not mean anything is wrong.

    Checks: what the run verifies of its own counts (see
    ngramprep.ngram_acquire.verify), shown from the start so that it is
    plain what is pending, and what has already gone wrong.
    """
    counters = doc.get("counters") or {}
    corpus = doc.get("corpus") or {}
    in_flight = doc.get("in_flight") or {}
    groups: List[Tuple[str, List[Tuple[str, str, str]]]] = []

    def kept_note(kept: int, scanned: int) -> str:
        return (_pct(kept, scanned) + " of the n-grams in finished files") if scanned else ""

    lines = int(counters.get("lines", 0))
    flying = int(in_flight.get("lines", 0))
    kept = int(counters.get("inserted", doc.get("entries_written", 0)))
    session = [
        ("N-grams scanned", f"{lines + flying:,}",
         f"raw data lines; {flying:,} of them in files still being read" if flying else "raw data lines"),
        ("Kept", f"{kept:,}", kept_note(kept, lines)),
    ]
    if "empty_files" in counters or doc.get("checks") is not None:
        session.append(("Files with nothing kept", f"{int(counters.get('empty_files', 0)):,}",
                        f"of {int(doc.get('files_done', 0)):,} finished"))
    session.append(("Parsed", fmt_bytes(float(doc.get("uncompressed_bytes") or 0)), "uncompressed"))
    groups.append(("Records \u2013 this session", session))

    tiles = []
    if "lines" in corpus:
        tiles.append(("N-grams scanned", f"{int(corpus.get('lines', 0)):,}", "raw data lines, finished files"))
    tiles.append(("Kept", f"{int(corpus.get('entries', 0)):,}",
                  kept_note(int(corpus.get("entries", 0)), int(corpus.get("lines", 0)))))
    if "empty_files" in corpus or doc.get("checks") is not None:
        tiles.append(("Files with nothing kept", f"{int(corpus.get('empty_files', 0)):,}",
                      f"of {int(corpus.get('files', 0)):,} finished"))
    tiles += [("Parsed", fmt_bytes(float(corpus.get("bytes", 0))), "uncompressed"), db_tile]
    unsized = int(corpus.get("unsized", 0))
    if unsized:
        tiles.append(("Unsized", f"{unsized:,}", "files done before sizes were recorded"))
    groups.append(("Records \u2013 entire corpus", tiles))

    checks = doc.get("checks")
    verification = doc.get("verification") or {}
    if checks is None and not verification:
        return groups                              # a run from before the checks existed
    checks = checks or {}
    ledger, found = verification.get("ledger") or {}, verification.get("database") or {}
    status = verification.get("status")
    state, phase = info.get("state"), doc.get("phase")
    if status == "verified":
        counts = ("verified", "the database holds exactly what the files gave")
    elif status == "mismatch":
        note = "the database differs from what the files gave"
        if not ledger.get("unledgered") and "match_total" in found:
            note = (f"occurrences {int(found.get('match_total', 0)) - int(ledger.get('match_total', 0)):+,}; "
                    f"volume counts {int(found.get('volume_total', 0)) - int(ledger.get('volume_total', 0)):+,}")
        counts = ("DO NOT MATCH", note)
    elif status == "unverifiable":
        counts = ("not verifiable", f"{int(ledger.get('unledgered', 0)):,} files predate the count ledger")
    elif status == "skipped" or checks.get("verify_counts") is False:
        counts = ("not checked", "verify_counts is off")
    elif state == "running" and phase == "verifying":
        counts = ("checking", "reading the whole database once")
    elif state == "running":
        counts = ("pending", "checked after compaction")
    else:
        counts = ("not reached", "the run ended before the check")
    occurrences = int(found.get("match_total", ledger.get("match_total", corpus.get("match_total", 0))))
    unreadable = int(ledger.get("problem_lines", corpus.get("problem_lines", counters.get("problem_lines", 0))))
    reordered = int(ledger.get("reordered", corpus.get("reordered", counters.get("reordered", 0))))
    tiles = [
        ("Counts", counts[0], counts[1]),
        ("Occurrences", f"{occurrences:,}",
         "in the database" if "match_total" in found else "in the files read so far (the ledger)"),
        ("Lines not read in full", f"{unreadable:,}",
         "their n-grams are undercounted; see the log" if unreadable else "every line's counts were read"),
        ("Values put in year order", f"{reordered:,}",
         "arrived out of order; corrected" if reordered else "all arrived in order"),
    ]
    unledgered = int(ledger.get("unledgered", corpus.get("unledgered", 0)))
    if unledgered:
        tiles.append(("Files without totals", f"{unledgered:,}", "acquired before the count ledger"))
    groups.append(("Checks", tiles))
    return groups


def ensemble_groups(ens: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """For a train run: the years whose replicates are still coming in, each
    with how many of the design's models exist so far. The noise profile of
    the finished years is a result, and lives on the Inspect tab."""
    in_hand = {int(y): int(n) for y, n in (ens.get("in_hand") or {}).items()}
    if not in_hand:
        return []
    per_year = int(ens.get("per_year") or 0)
    design = f"{ens.get('corpora', '?')} resamples × {ens.get('seeds', '?')} seeds"
    tiles = [(str(y), f"{n:,}/{per_year:,}" if per_year else f"{n:,}", f"{100.0 * n / per_year:.0f}%" if per_year else "")
             for y, n in sorted(in_hand.items())[:6]]
    more = len(in_hand) - len(tiles)
    return [(f"Replicates in hand ({design})" + (f", {more} more years" if more > 0 else ""), tiles)]

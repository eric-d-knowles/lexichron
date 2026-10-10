"""Turn a run's ``progress.json`` into numbers and text for display.

Pure functions, so the UI's progress panel can be tested without a terminal.
"""
from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["summarize", "fmt_duration", "fmt_bytes", "fmt_count", "tail_lines"]


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
    state_label = "stopped" if stopped else ("running?" if stale else state)

    where = doc.get("hostname") or ""
    job = doc.get("slurm_job_id")
    where_txt = ""
    if job:
        where_txt = f"job {job}" + (f" on {where}" if where else "")
    elif where:
        where_txt = f"on {where}"

    if stopped:
        status = "stopped (job no longer in the queue; resume by running again)"
    elif state == "running" and stale:
        status = "running? (no update for " + fmt_duration((now - updated).total_seconds()) + ")"
    else:
        status = state
    parts = [status + (f" {where_txt}" if where_txt else "")]
    if total:
        parts.append(f"{done}/{total} files" + (f" ({percent:.0f}%)" if percent is not None else ""))
    else:
        parts.append(f"{done} files")
    if failed:
        parts.append(f"{failed} failed")
    if doc.get("files_skipped"):
        parts.append(f"{doc['files_skipped']} already done")
    headline = " · ".join(parts)

    detail = [
        f"{entries:,} entries",
        f"{fmt_bytes(float(doc.get('uncompressed_bytes', 0)))} parsed (uncompressed)",
        f"{fmt_count(rate)} entries/s",
        f"elapsed {fmt_duration(elapsed)}",
    ]
    if eta is not None:
        detail.append(f"about {fmt_duration(eta)} left")
    current = list(doc.get("current") or [])
    phases = doc.get("phases") or {}
    if not current:
        current_text = ""
    elif phases:
        bits = []
        for key, label in (("queued", "waiting for a worker"), ("parsing", "downloading & parsing"),
                           ("parsed", "parsed, waiting to ingest"), ("ingesting", "ingesting")):
            n = int(phases.get(key, 0))
            if n:
                bits.append(f"{n} {label}")
        current_text = f"{len(current)} files in flight: " + " · ".join(bits)
        if doc.get("ingesting"):
            current_text += f"  ({doc['ingesting']})"
    else:
        current_text = f"{len(current)} files in flight"
    corpus = doc.get("corpus") or {}
    corpus_text = ""
    if corpus and (corpus.get("files") or 0) > 0:
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
        "stale": stale, "stopped": stopped, "headline": headline, "detail": " · ".join(detail),
        "current": current, "current_text": current_text, "corpus_text": corpus_text,
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
    except OSError:
        return []

"""Machine-readable progress for a pipeline run.

A :class:`ProgressReporter` keeps a small JSON document up to date while a
stage runs, so that other processes (the ``lexichron ui`` jobs pane, a Slurm
monitor, a notebook) can read the state of a run without parsing its log.
Writes are atomic (write to a temp file, then rename) and rate-limited, so a
reader never sees a partial document and the pipeline is not slowed down.

Document::

    {
      "stage": "acquire", "state": "running" | "done" | "failed",
      "started": "...", "updated": "...", "finished": "..." | null,
      "files_total": 14, "files_done": 3, "files_failed": 0, "files_skipped": 0,
      "entries_written": 4914496, "uncompressed_bytes": 4731..., "chunks": 25,
      "current": ["1-00003-of-00014.gz", "1-00004-of-00014.gz"],
      "message": "..."                      # last error, if any
    }
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

__all__ = ["ProgressReporter", "read_progress"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressReporter:
    def __init__(self, path: Optional[str | os.PathLike], *, stage: str,
                 min_interval_s: float = 2.0) -> None:
        self.path = Path(path) if path else None
        self.min_interval_s = min_interval_s
        self._last_write = 0.0
        self.doc: Dict[str, Any] = {
            "stage": stage, "state": "running",
            "started": _now(), "updated": _now(), "finished": None,
            "files_total": 0, "files_done": 0, "files_failed": 0, "files_skipped": 0,
            "entries_written": 0, "uncompressed_bytes": 0, "chunks": 0,
            "current": [], "message": "",
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(force=True)

    # -- updates --------------------------------------------------------------
    def set_totals(self, files_total: int, files_skipped: int = 0) -> None:
        self.doc["files_total"] = files_total
        self.doc["files_skipped"] = files_skipped
        self._write(force=True)

    def file_started(self, filename: str) -> None:
        cur = self.doc["current"]
        if filename not in cur:
            cur.append(filename)
        self._write()

    def file_done(self, filename: str, *, entries: int, chunks: int, uncompressed_bytes: int) -> None:
        self._drop_current(filename)
        self.doc["files_done"] += 1
        self.doc["entries_written"] += entries
        self.doc["chunks"] += chunks
        self.doc["uncompressed_bytes"] += uncompressed_bytes
        self._write(force=True)

    def file_failed(self, filename: str, message: str) -> None:
        self._drop_current(filename)
        self.doc["files_failed"] += 1
        self.doc["message"] = message
        self._write(force=True)

    def finish(self, state: str = "done", message: str = "") -> None:
        self.doc["state"] = state
        self.doc["finished"] = _now()
        self.doc["current"] = []
        if message:
            self.doc["message"] = message
        self._write(force=True)

    # -- internals ------------------------------------------------------------
    def _drop_current(self, filename: str) -> None:
        try:
            self.doc["current"].remove(filename)
        except ValueError:
            pass

    def _write(self, force: bool = False) -> None:
        if not self.path:
            return
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval_s:
            return
        self.doc["updated"] = _now()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=1)
        os.replace(tmp, self.path)
        self._last_write = now


def read_progress(path: str | os.PathLike) -> Optional[Dict[str, Any]]:
    """Read a progress document, or None if it does not exist or is unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

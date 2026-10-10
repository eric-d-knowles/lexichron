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
      "current": ["1-00003-of-00014.gz", "1-00004-of-00014.gz"],  # in flight
      "phases": {"queued": 40, "parsing": 36, "parsed": 3, "ingesting": 1},
      "ingesting": "1-00003-of-00014.gz" | null,
      "corpus": {"files": 8309, "entries": 794..., "bytes": 7.9e12, "unsized": 0},
                                            # whole database incl. earlier runs
      "message": "...",                     # last error, if any
      "db_path": "...", "log_path": "...",  # where the output and log are
      "slurm_job_id": "12345" | null,       # when run under Slurm
      "hostname": "..."
    }
"""
from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

__all__ = ["ProgressReporter", "read_progress"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressReporter:
    def __init__(self, path: Optional[str | os.PathLike], *, stage: str,
                 min_interval_s: float = 2.0,
                 db_path: Optional[str | os.PathLike] = None,
                 log_path: Optional[str | os.PathLike] = None,
                 spool_dir: Optional[str | os.PathLike] = None) -> None:
        self.path = Path(path) if path else None
        self.min_interval_s = min_interval_s
        self._last_write = 0.0
        # Which in-flight files are past which phase. Workers announce that
        # they have picked a file up by touching <tag>.started in the spool
        # directory (they run in other processes); the parent tells us the rest.
        self.spool_dir = Path(spool_dir) if spool_dir else None
        self._parsed: set = set()
        self._ingesting: Optional[str] = None
        self._prior: Dict[str, int] = {"files": 0, "entries": 0, "bytes": 0, "unsized": 0}
        self.doc: Dict[str, Any] = {
            "stage": stage, "state": "running",
            "started": _now(), "updated": _now(), "finished": None,
            "files_total": 0, "files_done": 0, "files_failed": 0, "files_skipped": 0,
            "entries_written": 0, "uncompressed_bytes": 0, "chunks": 0,
            "current": [], "message": "",
            "phases": {"queued": 0, "parsing": 0, "parsed": 0, "ingesting": 0},
            "ingesting": None,
            "corpus": {"files": 0, "entries": 0, "bytes": 0, "unsized": 0},
            "db_path": str(db_path) if db_path else None,
            "log_path": str(log_path) if log_path else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
        }
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(force=True)

    # -- updates --------------------------------------------------------------
    def set_totals(self, files_total: int, files_skipped: int = 0,
                   prior: Optional[Dict[str, int]] = None) -> None:
        """``prior`` is what files already in the database contributed
        (see :func:`coordinator.processed_totals`); the ``corpus`` totals in
        the document are prior plus this run, so they span resumed runs."""
        self.doc["files_total"] = files_total
        self.doc["files_skipped"] = files_skipped
        self._prior = dict(prior or {"files": 0, "entries": 0, "bytes": 0, "unsized": 0})
        self._write(force=True)

    def file_started(self, filename: str) -> None:
        cur = self.doc["current"]
        if filename not in cur:
            cur.append(filename)
        self._write()

    def set_spool_dir(self, spool_dir: Optional[str | os.PathLike]) -> None:
        self.spool_dir = Path(spool_dir) if spool_dir else None

    def file_parsed(self, filename: str) -> None:
        """A worker has finished with the file; it waits for ingestion."""
        self._parsed.add(filename)
        self._write()

    def file_ingesting(self, filename: str) -> None:
        self._parsed.discard(filename)
        self._ingesting = filename
        self._write(force=True)

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
        self._parsed.discard(filename)
        if self._ingesting == filename:
            self._ingesting = None

    def _started_files(self) -> set:
        """Files whose worker has touched its .started marker."""
        if not self.spool_dir:
            return set()
        found = set()
        try:
            with os.scandir(self.spool_dir) as it:
                for entry in it:
                    name = entry.name
                    if name.endswith(".started"):
                        # <idx>_<filename>.started
                        found.add(name[:-len(".started")].split("_", 1)[-1])
        except OSError:
            pass
        return found

    def _update_phases(self) -> None:
        current = list(self.doc["current"])
        started = self._started_files()
        ingesting = {self._ingesting} if self._ingesting else set()
        parsed = {f for f in current if f in self._parsed} - ingesting
        parsing = {f for f in current if f in started} - parsed - ingesting
        queued = set(current) - parsing - parsed - ingesting
        self.doc["phases"] = {"queued": len(queued), "parsing": len(parsing),
                              "parsed": len(parsed), "ingesting": len(ingesting)}
        self.doc["ingesting"] = self._ingesting
        self.doc["corpus"] = {
            "files": self._prior["files"] + self.doc["files_done"],
            "entries": self._prior["entries"] + self.doc["entries_written"],
            "bytes": self._prior["bytes"] + self.doc["uncompressed_bytes"],
            "unsized": self._prior["unsized"],
        }

    def _write(self, force: bool = False) -> None:
        if not self.path:
            return
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval_s:
            return
        self.doc["updated"] = _now()
        self._update_phases()
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

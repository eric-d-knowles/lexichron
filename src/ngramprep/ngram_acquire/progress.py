"""Machine-readable progress for a pipeline run.

A :class:`ProgressReporter` keeps a small JSON document up to date while a
stage runs, so that other processes (the ``lexichron ui`` jobs pane, a Slurm
monitor, a notebook) can read the state of a run without parsing its log.
Writes are atomic (write to a temp file, then rename) and rate-limited, so a
reader never sees a partial document and the pipeline is not slowed down.

Document::

    {
      "stage": "acquire", "state": "running" | "done" | "failed",
      "inputs": {"source": ..., "vocabulary": ...},   # what the run reads (stage-specific)
      "phase": "ingesting" | "compacting" | "archiving",   # what a running job is doing
      "phase_started": "...",
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
      "slurm_job_id": "12345" | null,       # when run under Slurm (12345_3 for an array task)
      "array": {"task": 3, "count": 8} | null,   # a task of a job array: set by the stage
      "hostname": "..."
    }
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

__all__ = ["ProgressReporter", "read_progress"]


def slurm_job_id() -> Optional[str]:
    """The job as ``squeue`` names it: ``12345``, or ``12345_3`` for a task
    of a job array (whose own ``SLURM_JOB_ID`` the queue does not show)."""
    array_job, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and task:
        return f"{array_job}_{task}"
    return os.environ.get("SLURM_JOB_ID")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProgressReporter:
    def __init__(self, path: Optional[str | os.PathLike], *, stage: str,
                 min_interval_s: float = 2.0,
                 db_path: Optional[str | os.PathLike] = None,
                 log_path: Optional[str | os.PathLike] = None,
                 spool_dir: Optional[str | os.PathLike] = None,
                 unit_label: str = "files",
                 inputs: Optional[Dict[str, Any]] = None) -> None:
        self.path = Path(path) if path else None
        self.min_interval_s = min_interval_s
        self._write_failed = False
        self._last_write = 0.0
        # Which in-flight files are past which phase. Workers announce that
        # they have picked a file up by touching <tag>.started in the spool
        # directory (they run in other processes); the parent tells us the rest.
        self.spool_dir = Path(spool_dir) if spool_dir else None
        self._parsed: set = set()
        self._active: set = set()          # told to us directly (no marker file)
        self._ingesting: Optional[str] = None
        self._prior: Dict[str, int] = {"files": 0, "entries": 0, "bytes": 0, "unsized": 0}
        self.doc: Dict[str, Any] = {
            "stage": stage, "state": "running",
            "phase": "ingesting", "phase_started": _now(),
            "started": _now(), "updated": _now(), "finished": None,
            "files_total": 0, "files_done": 0, "files_failed": 0, "files_skipped": 0,
            "entries_written": 0, "uncompressed_bytes": 0, "chunks": 0,
            "current": [], "message": "",
            "phases": {"queued": 0, "parsing": 0, "parsed": 0, "ingesting": 0},
            "ingesting": None,
            "corpus": {"files": 0, "entries": 0, "bytes": 0, "unsized": 0},
            "unit_label": unit_label,          # what files_total/files_done count
            "counters": {},                    # stage-specific running totals
            "database": {},                    # after compaction: records, bytes (exact once compacted)
            "inputs": {k: (str(v) if v is not None else None) for k, v in (inputs or {}).items()},
            "db_path": str(db_path) if db_path else None,
            "log_path": str(log_path) if log_path else None,
            "slurm_job_id": slurm_job_id(),
            "array": None,
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

    def file_active(self, filename: str) -> None:
        """The unit is being worked on (the 'parsing' phase), reported by the
        parent itself rather than by a worker's marker file."""
        cur = self.doc["current"]
        if filename not in cur:
            cur.append(filename)
        self._active.add(filename)
        self._write()

    def file_parsed(self, filename: str) -> None:
        """A worker has finished with the file; it waits for ingestion."""
        self._parsed.add(filename)
        self._active.discard(filename)
        self._write()

    def file_ingesting(self, filename: str) -> None:
        self._parsed.discard(filename)
        self._ingesting = filename
        self._write(force=True)

    def file_done(self, filename: str, *, entries: int, chunks: int, uncompressed_bytes: int,
                  counters: Optional[Dict[str, int]] = None) -> None:
        self._drop_current(filename)
        self.doc["files_done"] += 1
        self.doc["entries_written"] += entries
        self.doc["chunks"] += chunks
        self.doc["uncompressed_bytes"] += uncompressed_bytes
        for k, v in (counters or {}).items():
            self.doc["counters"][k] = self.doc["counters"].get(k, 0) + int(v)
        self._write(force=True)

    def file_failed(self, filename: str, message: str) -> None:
        self._drop_current(filename)
        self.doc["files_failed"] += 1
        self.doc["message"] = message
        self._write(force=True)

    def set_phase(self, phase: str) -> None:
        self.doc["phase"] = phase
        self.doc["phase_started"] = _now()
        self._write(force=True)

    @contextmanager
    def long_phase(self, phase: str, heartbeat_s: float = 30.0):
        """Mark a long single-call phase (compaction, archiving) and keep the
        document's ``updated`` time fresh from a background thread while it
        runs, so readers can tell "still compacting" from "died"."""
        self.set_phase(phase)
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(heartbeat_s):
                try:
                    self._write(force=True)
                except Exception:
                    pass

        t = threading.Thread(target=beat, name=f"progress-{phase}", daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=5)
            self.set_phase("ingesting" if self.doc["state"] == "running" else self.doc["phase"])

    def set_database(self, db, *, compacted: bool, bookkeeping: int = 0) -> None:
        """Record the destination's data-record count and size. After a full
        compaction every key is in one SST entry, so RocksDB's key count is
        exact; before that it is an estimate and is marked as such.
        ``bookkeeping`` is the number of marker keys the stage itself keeps
        in the database (resume markers, the partition), left out of the
        count."""
        try:
            records = db.get_property("rocksdb.estimate-num-keys")
            size = db.get_property("rocksdb.total-sst-files-size")
        except Exception:                       # pragma: no cover - store without properties
            return
        self.doc["database"] = {"records": max(int(records) - int(bookkeeping), 0) if records else None,
                                "bytes": int(size) if size else None, "compacted": bool(compacted)}
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
        self._active.discard(filename)
        if self._ingesting == filename:
            self._ingesting = None

    def _marker_files(self) -> tuple:
        """(started, parsed): files whose worker has touched <idx>_<file>.started,
        and those it has since renamed to .parsed."""
        started: set = set()
        parsed: set = set()
        if not self.spool_dir:
            return started, parsed
        try:
            with os.scandir(self.spool_dir) as it:
                for entry in it:
                    name = entry.name
                    if name.endswith(".started"):
                        started.add(name[:-len(".started")].split("_", 1)[-1])
                    elif name.endswith(".parsed"):
                        parsed.add(name[:-len(".parsed")].split("_", 1)[-1])
        except OSError:
            pass
        return started, parsed

    def _update_phases(self) -> None:
        current = list(self.doc["current"])
        started, worker_parsed = self._marker_files()
        ingesting = {self._ingesting} if self._ingesting else set()
        parsed = {f for f in current if f in self._parsed or f in worker_parsed} - ingesting
        parsing = {f for f in current if f in started or f in self._active} - parsed - ingesting
        queued = set(current) - parsing - parsed - ingesting
        self.doc["phases"] = {"queued": len(queued), "parsing": len(parsing),
                              "parsed": len(parsed), "ingesting": len(ingesting)}
        self.doc["ingesting"] = self._ingesting
        self.doc["corpus"] = {
            "files": self._prior["files"] + self.doc["files_done"],
            "entries": self._prior["entries"] + self.doc["entries_written"],
            "bytes": self._prior["bytes"] + self.doc["uncompressed_bytes"],
            "lines": self._prior.get("lines", 0) + int(self.doc["counters"].get("lines", 0)),
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
        # The document is advisory: a run must never die because it could
        # not be written. Its directory may have been removed under a
        # running job (a UI clearing finished runs, a tidy-up); recreate it,
        # and if the write still fails, say so once and carry on.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.doc, fh, indent=1)
            os.replace(tmp, self.path)
        except OSError as exc:
            if not self._write_failed:
                self._write_failed = True
                import logging
                logging.getLogger(__name__).warning("could not write the progress document %s: %s", self.path, exc)
            return
        self._write_failed = False
        self._last_write = now


def read_progress(path: str | os.PathLike) -> Optional[Dict[str, Any]]:
    """Read a progress document, or None if it does not exist or is unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

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
      "phase": "ingesting" | "compacting" | "verifying" | "archiving",   # what a running job is doing
      "phase_started": "...",
      "steps": ["ingesting", "compacting", "verifying"],   # the phases this run goes through, in order
                                            # (a stage that declares them: see set_plan)
      "phase_times": {"ingesting": {"started": "...", "finished": "..." | null}, ...},
      "started": "...", "updated": "...", "finished": "..." | null,
      "files_total": 14, "files_done": 3, "files_failed": 0, "files_skipped": 0,
      "entries_written": 4914496, "uncompressed_bytes": 4731..., "chunks": 25,
      "bytes_total": 2.1e12 | null,         # size of the files this run has to fetch, where the
      "bytes_done": 8.4e11,                 # listing gave sizes, and of those finished (or failed)
      "in_flight": {"lines": 7300000, "bytes": 3.1e9},   # what workers report of unfinished files
      "failed_files": [{"file": "...", "message": "..."}],
      "checks": {"verify_counts": true, "allow_unparsed_lines": false},   # acquire: what the run will check
      "verification": {...},                # acquire: the count check's outcome (ngram_acquire.verify)
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
                 inputs: Optional[Dict[str, Any]] = None,
                 terms: Optional[Dict[str, str]] = None) -> None:
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
        self._sizes: Dict[str, int] = {}
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
            # This run's words for what it reads, for the progress display, which
            # otherwise says "records" and "reading" (lexichron.ui.progress_view.TERMS).
            "terms": dict(terms or {}),
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

    def set_plan(self, steps: Iterable[str]) -> None:
        """Declare the phases this run goes through, in order, so that a
        reader can say "step 2 of 3" and what is still to come. The first is
        the phase the run is in now. With a plan, a long phase that ends
        stays the current phase until the next one starts."""
        self.doc["steps"] = [str(s) for s in steps]
        self.doc["phase_times"] = {self.doc["phase"]: {"started": self.doc["phase_started"], "finished": None}}
        self._write(force=True)

    def set_sizes(self, sizes: Dict[str, int], filenames: Iterable[str]) -> None:
        """Sizes in bytes of the files this run has to fetch (as the listing
        gave them; compressed). Progress and time remaining are then weighed
        by size: files differ by orders of magnitude in what they hold, so a
        count of files says little. Left unset when any size is unknown."""
        names = list(filenames)
        self._sizes = {n: int(sizes[n]) for n in names if n in sizes}
        self.doc["bytes_total"] = sum(self._sizes.values()) if names and len(self._sizes) == len(names) else None
        self.doc["bytes_done"] = 0
        self._write(force=True)

    def set_checks(self, **checks: Any) -> None:
        """What the run will check of its own work (acquire: verify_counts,
        allow_unparsed_lines), so that a reader can say what is pending and
        what a problem met along the way will lead to."""
        self.doc["checks"] = dict(checks)
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
        if "bytes_done" in self.doc:
            self.doc["bytes_done"] += self._sizes.get(filename, 0)
        self._write(force=True)

    def file_failed(self, filename: str, message: str) -> None:
        self._drop_current(filename)
        self.doc["files_failed"] += 1
        self.doc["message"] = message
        failed = self.doc.setdefault("failed_files", [])
        if len(failed) < 50:                       # the count is in files_failed; the log has them all
            failed.append({"file": filename, "message": message})
        if "bytes_done" in self.doc:
            self.doc["bytes_done"] += self._sizes.get(filename, 0)
        self._write(force=True)

    def set_phase(self, phase: str) -> None:
        now = _now()
        times = self.doc.get("phase_times")
        if times is not None:
            previous = times.get(self.doc.get("phase"))
            if previous and self.doc.get("phase") != phase and not previous.get("finished"):
                previous["finished"] = now
            # (a phase entered again, e.g. the units after partitioning, is not finished)
            times.setdefault(phase, {"started": now})["finished"] = None
        self.doc["phase"] = phase
        self.doc["phase_started"] = now
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
            if phase in (self.doc.get("steps") or ()):
                # A planned step: it is over, and the run is between steps
                # until the next one starts (or the run finishes).
                self.doc["phase_times"].setdefault(phase, {"started": self.doc["phase_started"]})["finished"] = _now()
                self._write(force=True)
            else:
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

    def refresh(self) -> None:
        """Rewrite the document with nothing new from the caller: what the
        workers report of files in flight, and a fresh ``updated`` time. For
        a parent that is waiting on work that takes minutes per unit."""
        self._write()

    def set_verification(self, verification: Dict[str, Any]) -> None:
        """Record the outcome of the count verification (see
        :mod:`ngramprep.ngram_acquire.verify`): its status, the ledger's
        totals and what the scan of the database found."""
        self.doc["verification"] = verification
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
        """(started, parsed, reports): files whose worker has touched
        <idx>_<file>.started, those it has since renamed to .parsed, and what
        each marker says of the work so far. A worker may write
        "<lines read> <bytes downloaded>" into its marker as it goes; a
        marker that is empty, or caught half-written, reports nothing."""
        started: set = set()
        parsed: set = set()
        reports: Dict[str, tuple] = {}
        if not self.spool_dir:
            return started, parsed, reports
        try:
            with os.scandir(self.spool_dir) as it:
                for entry in it:
                    name = entry.name
                    if name.endswith(".started"):
                        unit = name[:-len(".started")].split("_", 1)[-1]
                        started.add(unit)
                    elif name.endswith(".parsed"):
                        unit = name[:-len(".parsed")].split("_", 1)[-1]
                        parsed.add(unit)
                    else:
                        continue
                    try:
                        with open(entry.path, "r") as fh:
                            fields = fh.read(64).split()
                        if len(fields) == 2:
                            reports[unit] = (int(fields[0]), int(fields[1]))
                    except (OSError, ValueError):
                        pass
        except OSError:
            pass
        return started, parsed, reports

    def _update_phases(self) -> None:
        current = list(self.doc["current"])
        started, worker_parsed, reports = self._marker_files()
        # Only files still in flight: a finished file's figures are in the
        # counters, and its marker may outlive it by a moment.
        self.doc["in_flight"] = {"lines": sum(reports[f][0] for f in current if f in reports),
                                 "bytes": sum(reports[f][1] for f in current if f in reports)}
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
        # Acquisition's ledger, over every job that has added to the database.
        for name in ("empty_files", "match_total", "volume_total", "problem_lines", "reordered"):
            if name in self.doc["counters"] or name in self._prior:
                self.doc["corpus"][name] = int(self._prior.get(name, 0)) + int(self.doc["counters"].get(name, 0))
        if self._prior.get("unledgered"):
            self.doc["corpus"]["unledgered"] = int(self._prior["unledgered"])

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

"""Client for the host bridge (container/host_bridge.sh).

Code inside the container cannot run Slurm commands, because ``sbatch`` and
friends live on the host. The bridge is a small shell loop on the host that
watches a directory for request files and writes responses; this module is
the client side. See the shell script for the protocol.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = ["BridgeResult", "HostBridge", "BridgeUnavailable", "default_bridge_dir"]


class BridgeUnavailable(RuntimeError):
    """No host bridge is running for this directory."""


@dataclass
class BridgeResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def default_bridge_dir() -> Optional[Path]:
    """Where the launcher told us the bridge lives (LEXICHRON_BRIDGE_DIR)."""
    env = os.environ.get("LEXICHRON_BRIDGE_DIR")
    return Path(env) if env else None


class HostBridge:
    def __init__(self, bridge_dir: str | os.PathLike, *, poll_s: float = 0.2) -> None:
        self.dir = Path(bridge_dir)
        self.poll_s = poll_s

    def available(self) -> bool:
        pid_file = self.dir / "bridge.pid"
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            return False
        # Inside the container the host PID may not be visible; existence of
        # the pid file and the req/ directory is the best check we have.
        return (self.dir / "req").is_dir() and pid > 0

    def run(self, cmd: str, args: Sequence[str] = (), *, timeout_s: float = 60.0) -> BridgeResult:
        if not self.available():
            raise BridgeUnavailable(
                f"no host bridge at {self.dir}; start the UI through the project's "
                f".venv/lexichron-ui launcher to enable Slurm actions"
            )
        req_id = uuid.uuid4().hex
        req_dir, res_dir = self.dir / "req", self.dir / "res"
        req_dir.mkdir(parents=True, exist_ok=True)
        tmp = req_dir / f"{req_id}.json.tmp"
        tmp.write_text(json.dumps({"cmd": cmd, "args": list(args)}))
        os.replace(tmp, req_dir / f"{req_id}.json")

        res_path = res_dir / f"{req_id}.json"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if res_path.exists():
                try:
                    data = json.loads(res_path.read_text())
                except ValueError:
                    time.sleep(self.poll_s)   # being written
                    continue
                res_path.unlink(missing_ok=True)
                return BridgeResult(int(data.get("rc", 1)), data.get("stdout", ""), data.get("stderr", ""))
            time.sleep(self.poll_s)
        raise TimeoutError(f"host bridge did not answer '{cmd}' within {timeout_s:.0f}s")

    # -- convenience wrappers ---------------------------------------------------
    def sbatch(self, script: str | os.PathLike) -> str:
        """Submit a script; return the job id."""
        res = self.run("sbatch", ["--parsable", str(script)])
        if not res.ok:
            raise RuntimeError(res.stderr.strip() or f"sbatch failed (rc={res.rc})")
        return res.stdout.strip().split(";")[0]

    def squeue(self, user: Optional[str] = None) -> List[dict]:
        args = ["--noheader", "--format=%i|%j|%T|%M|%l|%D|%R"]
        if user:
            args += ["--user", user]
        res = self.run("squeue", args)
        rows = []
        for line in res.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 7:
                rows.append(dict(zip(["job_id", "name", "state", "elapsed", "limit", "nodes", "reason"], parts)))
        return rows

    def sacct(self, job_id: str) -> List[dict]:
        res = self.run("sacct", ["-j", job_id, "--noheader", "--parsable2",
                                 "--format=JobID,State,Elapsed,MaxRSS,ExitCode"])
        rows = []
        for line in res.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 5:
                rows.append(dict(zip(["job_id", "state", "elapsed", "max_rss", "exit_code"], parts)))
        return rows

    def scancel(self, job_id: str) -> BridgeResult:
        return self.run("scancel", [job_id])

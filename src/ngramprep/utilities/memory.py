"""What a job's memory holds right now, in one line for the log.

A Slurm job is killed when its control group exceeds its memory limit, and
the kill names no cause. The group's own accounting does: how much is
process memory (which cannot be dropped) and how much is file cache (which
can), next to the limit. A stage logs this line at the start of each step,
so that a kill can be explained from the log of the run that suffered it.

Everything here is advisory: on a machine without cgroup v2, or where a
file cannot be read, the part is left out, and nothing raises.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

__all__ = ["memory_report", "cgroup_memory", "process_memory"]

_GB = float(1 << 30)


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fh:
            return fh.read()
    except OSError:
        return None


def cgroup_memory(root: str = "/sys/fs/cgroup", proc: str = "/proc/self") -> Optional[Dict[str, Optional[int]]]:
    """The memory accounting of the control group that limits this process:
    ``{"current", "anon", "file", "limit", "oom_kills"}`` in bytes (``limit``
    None when no group above this process sets one), or None without cgroup
    v2. The group is the nearest one, from this process's own upward, that
    has a limit: under Slurm that is the job's, which every step shares."""
    text = _read(os.path.join(proc, "cgroup"))
    if not text:
        return None
    rel = None
    for line in text.splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip()
    if rel is None:
        return None
    path = os.path.normpath(os.path.join(root, rel.lstrip("/")))
    root = os.path.normpath(root)
    own, limit, chosen = path, None, None
    while path.startswith(root):
        raw = (_read(os.path.join(path, "memory.max")) or "").strip()
        if raw.isdigit():
            limit, chosen = int(raw), path
            break
        if path == root:
            break
        path = os.path.dirname(path)
    chosen = chosen or own
    current = (_read(os.path.join(chosen, "memory.current")) or "").strip()
    if not current.isdigit():
        return None
    stat = {}
    for line in (_read(os.path.join(chosen, "memory.stat")) or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            stat[parts[0]] = int(parts[1])
    kills = None
    for line in (_read(os.path.join(chosen, "memory.events")) or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill" and parts[1].isdigit():
            kills = int(parts[1])
    return {"current": int(current), "anon": stat.get("anon"), "file": stat.get("file"),
            "limit": limit, "oom_kills": kills}


def _rss_anon(pid) -> Optional[int]:
    for line in (_read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("RssAnon:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _children(pid) -> List[int]:
    found: List[int] = []
    for path in glob.glob(f"/proc/{pid}/task/*/children"):
        for word in (_read(path) or "").split():
            if word.isdigit():
                found.append(int(word))
    return found


def process_memory() -> Dict[str, Optional[int]]:
    """This process's own memory and its descendants': ``{"own", "children",
    "children_bytes"}`` (memory that belongs to a process alone, not pages it
    shares with files)."""
    todo, seen, total = _children(os.getpid()), set(), 0
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _rss_anon(pid) or 0
        todo.extend(_children(pid))
    return {"own": _rss_anon(os.getpid()), "children": len(seen), "children_bytes": total}


def memory_report(root: str = "/sys/fs/cgroup", proc: str = "/proc/self") -> str:
    """One line: the job's memory against its limit, split into process
    memory and file cache, then this process and its children. Empty when
    nothing could be read."""
    parts: List[str] = []
    try:
        group = cgroup_memory(root, proc)
        if group:
            text = f"job {group['current'] / _GB:.1f} GB"
            if group["limit"]:
                text += f" of {group['limit'] / _GB:.1f} GB"
            split = []
            if group["anon"] is not None:
                split.append(f"{group['anon'] / _GB:.1f} GB held by processes")
            if group["file"] is not None:
                split.append(f"{group['file'] / _GB:.1f} GB file cache, which the kernel can drop")
            if split:
                text += " (" + ", ".join(split) + ")"
            if group["oom_kills"]:
                text += f"; {group['oom_kills']} out-of-memory kill(s) so far"
            parts.append(text)
        mine = process_memory()
        if mine["own"] is not None:
            parts.append(f"this process {mine['own'] / _GB:.1f} GB")
        if mine["children"]:
            parts.append(f"{mine['children']} child process(es) {mine['children_bytes'] / _GB:.1f} GB")
    except Exception:
        pass
    return "; ".join(parts)

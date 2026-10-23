"""Warn before a stage deletes work that is already on disk.

Three settings make a stage start by deleting its own earlier output:
``acquire.overwrite_db`` removes the raw database, ``filter.mode = restart``
removes the processed or pivoted database, ``tune.mode = restart`` removes a
named search (its models and scores), ``train.mode = restart`` a named
ensemble of final models. Each is the right thing sometimes and
a day of lost compute the rest of the time, and a 5-gram database is hundreds
of gigabytes. ``overwrite_risk`` works out, from the arguments a stage is
about to be called with, whether it would delete anything and how much; the
CLI asks before going ahead (``--yes`` answers for a batch job), and the UI
puts the same question in a dialog before Run, Submit and Write script.

The check is deliberately outside the stage functions, which stay callable
from Python and tests without a prompt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["OverwriteRisk", "overwrite_risk", "overwrite_target", "dir_size", "format_bytes"]


@dataclass
class OverwriteRisk:
    path: Path             # what the stage would delete
    what: str              # "the raw 5-gram database"
    setting: str           # "acquire.overwrite_db = true"
    nbytes: int
    nfiles: int

    @property
    def keep_hint(self) -> str:
        """How to keep the data: the setting change that makes the stage resume instead."""
        key, _, value = self.setting.partition(" = ")
        return f"turn {key} off" if value == "true" else f"set {key} to resume"

    @property
    def size(self) -> str:
        return format_bytes(self.nbytes)

    def describe(self) -> str:
        files = f"{self.nfiles:,} file{'s' if self.nfiles != 1 else ''}"
        return (f"{self.setting} deletes {self.what} before starting:\n"
                f"  {self.path}  ({self.size} in {files})")


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1000 or unit == "PB":
            return f"{size:.0f} {unit}" if unit == "B" or size >= 100 else f"{size:.1f} {unit}"
        size /= 1000
    return f"{n} B"


def dir_size(path: Path) -> tuple[int, int]:
    """(bytes, files) under ``path``, symlinks not followed; a file counts itself."""
    if path.is_file():
        return path.stat().st_size, 1
    total = files = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files


def _target(stage: str, kwargs: Dict[str, Any]) -> Optional[tuple[Path, str, str]]:
    """(path, what, setting) the stage would delete, or None when its settings
    do not ask for deletion."""
    n = kwargs.get("ngram_size")
    release, lang, stub = kwargs.get("repo_release_id"), kwargs.get("repo_corpus_id"), kwargs.get("db_path_stub")
    if not (n and release and lang and stub):
        return None
    if stage == "acquire":
        if not kwargs.get("overwrite_db"):
            return None
        from ngramprep.ngram_acquire.db.build_path import build_db_path
        return Path(build_db_path(str(stub), int(n), str(release), str(lang))), f"the raw {n}-gram database", \
            "acquire.overwrite_db = true"
    if stage == "filter":
        if kwargs.get("mode", "resume") != "restart":
            return None
        from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths
        paths = derive_paths(str(stub), int(n), str(release), str(lang))
        pivot = kwargs.get("pivot")
        if pivot is None:
            pivot = int(n) > 1
        if pivot:
            return Path(paths["pivoted_db"]), f"the pivoted {n}-gram database", "filter.mode = restart"
        return Path(paths["dst_db"]), f"the processed {n}-gram database", "filter.mode = restart"
    if stage == "tune":
        if kwargs.get("mode", "resume") != "restart" or not kwargs.get("name"):
            return None                     # an unnamed search gets a fresh timestamp directory
        from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths
        paths = derive_paths(str(stub), int(n), str(release), str(lang))
        return Path(paths["pivoted_db"]).parent / "tuning" / str(kwargs["name"]), \
            f"the tuning search '{kwargs['name']}' (its models and scores)", "tune.mode = restart"
    if stage == "train":
        if kwargs.get("mode", "resume") != "restart" or not kwargs.get("name"):
            return None                     # an unnamed ensemble gets a fresh timestamp directory
        from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths
        paths = derive_paths(str(stub), int(n), str(release), str(lang))
        return Path(paths["pivoted_db"]).parent / "models" / str(kwargs["name"]), \
            f"the model ensemble '{kwargs['name']}' (every year's replicates)", "train.mode = restart"
    return None


def overwrite_target(stage: str, kwargs: Dict[str, Any]) -> Optional[Path]:
    """The path the stage would delete first with these (resolved)
    arguments, whether or not it exists; None when its settings delete
    nothing. Cheap: nothing is walked."""
    target = _target(stage, kwargs)
    return target[0] if target else None


def overwrite_risk(stage: str, kwargs: Dict[str, Any]) -> Optional[OverwriteRisk]:
    """What running ``stage`` with these (resolved) arguments would delete,
    or None if nothing on disk is at stake."""
    target = _target(stage, kwargs)
    if target is None:
        return None
    path, what, setting = target
    if not path.exists():
        return None
    nbytes, nfiles = dir_size(path)
    return OverwriteRisk(path=path, what=what, setting=setting, nbytes=nbytes, nfiles=nfiles)

"""Describe a stage's settings as form fields, derived from its Python signature.

The form in ``lexichron ui`` is generated from this, so a new keyword argument
on a stage function appears in the UI without any UI code. Help text comes
from the function's docstring ``Args:`` section.
"""
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lexichron.config import CORPUS_KEY_MAP

__all__ = ["Field", "Section", "stage_sections", "SLURM_FIELDS", "parse_args_doc"]

# Settings that are not meaningful in a form (set by the CLI / runtime).
HIDDEN = {"progress_path", "write_batch_size"}

# Known choice lists for arguments whose type is a plain str/int.
CHOICES: Dict[str, List[str]] = {
    "ngram_size": ["1", "2", "3", "4", "5"],
    "ngram_type": ["tagged", "untagged", "all"],
    "open_type": ["write:packed24", "write", "read:packed24", "read"],
    "language": ["eng", "eng-us", "eng-gb", "eng-fiction", "chi-sim", "fre", "ger", "heb", "ita", "rus", "spa"],
}


@dataclass
class Field:
    name: str                 # key in the project file
    kind: str                 # "text" | "int" | "float" | "bool" | "choice" | "list" | "path"
    default: Any = None
    required: bool = False
    choices: Optional[List[str]] = None
    help: str = ""
    api_name: Optional[str] = None  # python argument name if different from `name`


@dataclass
class Section:
    name: str
    fields: List[Field] = field(default_factory=list)


def parse_args_doc(doc: Optional[str]) -> Dict[str, str]:
    """Return {arg_name: help} from a Google-style ``Args:`` docstring section."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    out: Dict[str, str] = {}
    in_args = False
    current = None
    for line in lines:
        if re.match(r"^\s*Args:\s*$", line):
            in_args = True
            continue
        if in_args and re.match(r"^\S", line):      # next section header
            break
        if not in_args:
            continue
        m = re.match(r"^\s{1,8}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            out[current] = m.group(2).strip()
        elif current and line.strip():
            out[current] += " " + line.strip()
    return out


def _kind_for(annotation: Any, default: Any, name: str) -> str:
    if name in CHOICES:
        return "choice"
    if name.endswith(("_stub", "_dir", "_path")):
        return "path"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union and type(None) in args:   # Optional[X]
        inner = [a for a in args if a is not type(None)]
        annotation = inner[0] if len(inner) == 1 else annotation
        origin = typing.get_origin(annotation); args = typing.get_args(annotation)
    text = str(annotation)
    if annotation is bool or isinstance(default, bool):
        return "bool"
    if annotation is int or isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if annotation is float or isinstance(default, float):
        return "float"
    if origin in (tuple, list, set) or any(w in text for w in ("Tuple", "List", "Set", "tuple[", "list[", "set[")) or annotation is set:
        return "list"
    return "text"


def stage_sections(func: Callable[..., Any], stage: str) -> List[Section]:
    """Build the ``corpus`` and ``<stage>`` sections for a stage function."""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    docs = parse_args_doc(func.__doc__)
    api_to_corpus = {v: k for k, v in CORPUS_KEY_MAP.items()}

    corpus = Section("corpus")
    stage_sec = Section(stage)
    for pname, p in sig.parameters.items():
        if pname in HIDDEN:
            continue
        ann = hints.get(pname, p.annotation)
        default = None if p.default is inspect.Parameter.empty else p.default
        required = p.default is inspect.Parameter.empty
        if pname in api_to_corpus:
            key = api_to_corpus[pname]
            f = Field(key, _kind_for(ann, default, key), default, required,
                      CHOICES.get(key), docs.get(pname, ""), api_name=pname)
            corpus.fields.append(f)
        else:
            f = Field(pname, _kind_for(ann, default, pname), default, required,
                      CHOICES.get(pname), docs.get(pname, ""))
            stage_sec.fields.append(f)
    return [corpus, stage_sec]


# Slurm resources live in the project file too, under `slurm:`.
SLURM_FIELDS = Section("slurm", [
    Field("account", "text", "", True, help="Slurm account, e.g. torch_pr_600_general"),
    Field("partition", "text", "", False, help="Partition (leave empty for the default)"),
    Field("cpus", "int", 20, True, help="CPUs for the job (also used as the worker count unless set)"),
    Field("mem", "text", "16G", True, help="Memory, e.g. 16G or 64G"),
    Field("time", "text", "12:00:00", True, help="Time limit, HH:MM:SS"),
    Field("job_name", "text", "lexichron", False, help="Job name shown in squeue"),
])

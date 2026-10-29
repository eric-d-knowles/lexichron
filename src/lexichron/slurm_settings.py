"""Where a job's Slurm settings come from.

Three layers, each kept where its owner would look for it:

1. lexichron's defaults per stage (``STAGE_DEFAULTS``): acquisition stops
   gaining around 40 workers; a tuning search or a replicate ensemble wants
   a 64-core slice of a node for a day or two, split over a job array.
2. the user's defaults (``~/.lexichron/slurm.yaml``, or under
   ``$LEXICHRON_STATE_DIR``): the account and partition, which are about the
   person, not the corpus, and so the same for every settings file.
3. the corpus's settings file, ``slurm:`` section: keys at the top level
   apply to every stage (how files written before 0.9.3 look), and a block
   named after a stage (``slurm.train``) applies to that stage alone and
   wins.

``resolve`` combines them and says where each value came from, so the
Submit tab can show "64 CPUs (this corpus, train) · account X (your
defaults)"; ``stage_override`` is what Save writes back.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

__all__ = ["ARRAY_STAGES", "STAGE_DEFAULTS", "STAGE_KEYS", "USER_KEYS", "describe_sources", "read_user_defaults",
           "resolve", "seed_user_defaults", "stage_override", "state_dir", "user_defaults_path",
           "write_user_defaults"]

# Stages whose years can be split over the tasks of a job array.
ARRAY_STAGES = ("tune", "train", "measure")
STAGE_NAMES = ("acquire", "filter", "tune", "train", "measure")
USER_KEYS = ("account", "partition")                   # about the person
STAGE_KEYS = ("cpus", "mem", "time", "jobs", "job_name")   # about the stage

STAGE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Acquisition throughput flattens out around 40 workers (the single merging
    # parent becomes the limit); 28G covers 40 workers on 5-grams. Filtering
    # streams the database with the same shape of job.
    "acquire": {"cpus": 40, "mem": "28G", "time": "12:00:00"},
    "filter": {"cpus": 40, "mem": "28G", "time": "12:00:00"},
    # A 64-core slice starts within minutes where a whole node waits hours;
    # the memory is page cache from the database scan and reclaimable.
    "tune": {"cpus": 64, "mem": "64G", "time": "2-00:00:00", "jobs": 8},
    "train": {"cpus": 64, "mem": "64G", "time": "1-00:00:00", "jobs": 13},
    # Measuring reads each model once and does a few hundred dot products:
    # a small job, limited by reading the models; 8 in memory at a time.
    "measure": {"cpus": 8, "mem": "16G", "time": "04:00:00", "jobs": 1},
}


def state_dir() -> Path:
    """Per-user state: ~/.lexichron, or $LEXICHRON_STATE_DIR."""
    return Path(os.environ.get("LEXICHRON_STATE_DIR") or Path.home() / ".lexichron")


def user_defaults_path() -> Path:
    return state_dir() / "slurm.yaml"


def read_user_defaults() -> Dict[str, Any]:
    """The user's account and partition, if they have been saved."""
    try:
        data = yaml.safe_load(user_defaults_path().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {k: v for k, v in data.items() if k in USER_KEYS and v not in (None, "")} if isinstance(data, dict) else {}


def write_user_defaults(values: Mapping[str, Any]) -> Path:
    """Save the user's account and partition for every corpus (atomically)."""
    keep = {k: values[k] for k in USER_KEYS if values.get(k) not in (None, "")}
    path = user_defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text("# lexichron: your Slurm defaults, used by every settings file\n"
                   + yaml.safe_dump(keep, sort_keys=False, default_flow_style=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def seed_user_defaults(values: Mapping[str, Any]) -> bool:
    """The first account a user saves becomes their default, once; after
    that the file changes only on request (write_user_defaults). True when
    the file was written."""
    if read_user_defaults() or not values.get("account"):
        return False
    write_user_defaults(values)
    return True


def resolve(stage: str, section: Optional[Mapping[str, Any]], user: Optional[Mapping[str, Any]] = None,
            ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The Slurm settings a job of ``stage`` gets, and the source of each:
    ``"lexichron's default for <stage>"``, ``"your defaults"``,
    ``"this corpus"`` (a top-level key of the section) or
    ``"this corpus, <stage>"`` (the stage's own block)."""
    values: Dict[str, Any] = dict(STAGE_DEFAULTS.get(stage) or STAGE_DEFAULTS["acquire"])
    sources = {k: f"lexichron's default for {stage}" for k in values}
    user = read_user_defaults() if user is None else user
    for key in USER_KEYS:
        if user.get(key) not in (None, ""):
            values[key] = user[key]
            sources[key] = "your defaults"
    section = section or {}
    for key, value in section.items():
        if key in STAGE_NAMES or value in (None, ""):
            continue
        values[key] = value
        sources[key] = "this corpus"
    for key, value in (section.get(stage) or {}).items():
        if value in (None, ""):
            continue
        values[key] = value
        sources[key] = f"this corpus, {stage}"
    if stage not in ARRAY_STAGES:
        values.pop("jobs", None)
        sources.pop("jobs", None)
    return values, sources


def stage_override(section: Optional[Mapping[str, Any]], stage: str, form: Mapping[str, Any],
                   user: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings file's ``slurm`` section after the Submit tab's values
    for ``stage`` are saved: the resources go in the stage's own block; the
    account and partition stay out of the file while they match the user's
    defaults and are written at the top level when they differ (a corpus
    charged to another project); other stages' blocks, and top-level keys
    from an older file, are kept."""
    user = read_user_defaults() if user is None else user
    out: Dict[str, Any] = {k: v for k, v in (section or {}).items()}
    for key in USER_KEYS:
        value = form.get(key)
        if value not in (None, "") and value != user.get(key):
            out[key] = value
        else:
            out.pop(key, None)
    block = {k: form[k] for k in STAGE_KEYS if form.get(k) not in (None, "")}
    if stage not in ARRAY_STAGES:
        block.pop("jobs", None)
    if block:
        out[stage] = block
    else:
        out.pop(stage, None)
    return out


def describe_sources(values: Mapping[str, Any], sources: Mapping[str, str]) -> str:
    """One line for the Submit tab: each setting with where it came from,
    grouped by source."""
    labels = {"account": "account", "partition": "partition", "cpus": "CPUs", "mem": "memory", "time": "time",
              "jobs": "jobs", "job_name": "job name"}
    by_source: Dict[str, list] = {}
    for key in ("account", "partition", "cpus", "mem", "time", "jobs", "job_name"):
        if key in values:
            by_source.setdefault(sources.get(key, "?"), []).append(f"{labels[key]} {values[key]}")
    parts = [f"{', '.join(items)} ({source})" for source, items in by_source.items()]
    if "account" not in values:
        parts.insert(0, "no account: fill one in (it becomes your default for every corpus)")
    if "partition" not in values:
        parts.append("partition: the cluster's default")
    return " · ".join(parts)

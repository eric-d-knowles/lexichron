"""Where a job's Slurm settings come from.

Three layers, each kept where its owner would look for it:

1. lexichron's defaults per stage (``STAGE_DEFAULTS``): acquisition stops
   gaining around 40 workers; a tuning search or a replicate ensemble wants
   a 64-core slice of a node for a day or two, split over a job array. A
   Davies corpus is a few gigabytes, not terabytes, and has its own
   (``DAVIES_DEFAULTS``, by stage and unit).
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

__all__ = ["ARRAY_STAGES", "DAVIES_DEFAULTS", "STAGE_DEFAULTS", "defaults_for", "guidance", "STAGE_KEYS", "USER_KEYS", "describe_sources", "read_user_defaults",
           "resolve", "seed_user_defaults", "stage_override", "state_dir", "user_defaults_path",
           "write_user_defaults"]

# Stages whose years can be split over the tasks of a job array.
ARRAY_STAGES = ("tune", "train", "measure")
STAGE_NAMES = ("acquire", "filter", "tune", "train", "measure")
USER_KEYS = ("account", "partition")                   # about the person
STAGE_KEYS = ("cpus", "mem", "time", "jobs", "job_name")   # about the stage

STAGE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # 28G covers 40 acquisition workers on 5-grams. Filtering has the same
    # shape of job but a heavier worker when it pivots: at the end of each
    # unit it merges a thousand or two spooled parts, about 0.9G at the peak
    # (see resources.py, where it was measured), so 40 workers need 40G.
    "acquire": {"cpus": 40, "mem": "28G", "time": "12:00:00"},
    "filter": {"cpus": 40, "mem": "40G", "time": "12:00:00"},
    # A 64-core slice starts within minutes where a whole node waits hours;
    # the memory is page cache from the database scan and reclaimable.
    "tune": {"cpus": 64, "mem": "64G", "time": "2-00:00:00", "jobs": 8},
    "train": {"cpus": 64, "mem": "64G", "time": "1-00:00:00", "jobs": 13},
    # Measuring reads each model once and does a few hundred dot products:
    # a small job, limited by reading the models; 8 in memory at a time.
    "measure": {"cpus": 8, "mem": "16G", "time": "04:00:00", "jobs": 1},
}

# A Davies corpus, sized from COHA (2026-11-24; 20 archives an edition, about
# 400 million words): acquisition reads one archive per worker, so CPUs past
# the archive count sit idle, and took 1.5 minutes with 20 or 40 workers and
# under 6G by Slurm's MaxRSS (which overcounts shared pages); the token filter
# 30 seconds and 2.5G with 10 workers, the sentence filter (pivoting) 2
# minutes and 6.4G with 20. Times leave room for a corpus several times
# COHA's size. A bin of COHA sentences is millions of tokens, so its models
# train in minutes: a search or an ensemble fits a 32-core job, and train's
# array splits the bins (9 five-year bins over three jobs).
DAVIES_DEFAULTS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("acquire", "tokens"): {"cpus": 20, "mem": "12G", "time": "01:00:00"},
    ("acquire", "sentences"): {"cpus": 20, "mem": "12G", "time": "01:00:00"},
    ("filter", "tokens"): {"cpus": 10, "mem": "8G", "time": "00:30:00"},
    ("filter", "sentences"): {"cpus": 20, "mem": "16G", "time": "00:30:00"},
    ("tune", "sentences"): {"cpus": 32, "mem": "32G", "time": "12:00:00", "jobs": 1},
    ("train", "sentences"): {"cpus": 32, "mem": "32G", "time": "12:00:00", "jobs": 3},
    ("measure", "sentences"): {"cpus": 4, "mem": "8G", "time": "01:00:00", "jobs": 1},
}


def defaults_for(stage: str, source: str = "ngrams", unit: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """lexichron's default resources for a job of ``stage`` on this kind of
    corpus, and how to name where they came from."""
    if source == "davies":
        unit = unit or "sentences"
        found = DAVIES_DEFAULTS.get((stage, unit)) or DAVIES_DEFAULTS.get((stage, "sentences"))
        if found:
            return dict(found), f"lexichron's default for {stage}, Davies {unit}"
    return dict(STAGE_DEFAULTS.get(stage) or STAGE_DEFAULTS["acquire"]), f"lexichron's default for {stage}"


def guidance(stage: str, source: str = "ngrams", unit: Optional[str] = None,
             archives: Optional[int] = None) -> str:
    """A few sentences for the Submit tab on sizing a job of ``stage`` for
    this kind of corpus; ``archives`` is the acquire run's archive count
    when it is known."""
    if source != "davies":
        return {
            "acquire": "Sizing: acquisition gains little past 40 workers. 5-grams take hours; 1-grams minutes.",
            "filter": "Sizing: the 5-gram filter (it pivots) runs for hours and needs about 0.9G a worker; "
                      "the 1-gram filter is much lighter.",
            "tune": "Sizing: a search trains (cells x sampled years x models per year); split it over a job "
                    "array with Jobs so each task takes some of the years.",
            "train": "Sizing: an ensemble trains (corpus replicates x seeds) models for every year; Jobs splits "
                     "the years over a job array.",
            "measure": "Sizing: measuring reads each model once; one small job is usually enough.",
        }.get(stage, "")
    unit = unit or "sentences"
    if stage == "acquire":
        count = (f"This edition has {archives} archive{'s' if archives != 1 else ''}, so CPUs past "
                 f"{archives} sit idle. " if archives else "CPUs past the number of archives sit idle. ")
        return ("Sizing (Davies): one worker reads one archive. " + count +
                "COHA (20 archives) took about 1.5 minutes and under 6G. A larger corpus takes longer in "
                "proportion: raise the time, not the CPUs.")
    if stage == "filter":
        if unit == "tokens":
            return ("Sizing (Davies tokens): COHA's token filter took 30 seconds and 2.5G with 10 workers; "
                    "more CPUs gain little.")
        return ("Sizing (Davies sentences): COHA's sentence filter (it pivots) took 2 minutes and 6.4G with "
                "20 workers.")
    if stage == "tune":
        return ("Sizing (Davies): a year or bin of a Davies corpus is millions of tokens, so each model "
                "trains in minutes; one 32-core job runs a search of a few cells on six sampled bins in "
                "hours. Larger vectors and more epochs take longer in proportion.")
    if stage == "train":
        return ("Sizing (Davies): every bin gets (corpus replicates x seeds) models of a few minutes each; "
                "Jobs splits the bins over a job array (3 jobs for 9 five-year bins). With annual years, "
                "raise Jobs rather than the time.")
    if stage == "measure":
        return "Sizing (Davies): the models are small; a few CPUs and an hour are plenty."
    return ""


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
            *, source: str = "ngrams", unit: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The Slurm settings a job of ``stage`` gets, and the source of each:
    ``"lexichron's default for <stage>"``, ``"your defaults"``,
    ``"this corpus"`` (a top-level key of the section) or
    ``"this corpus, <stage>"`` (the stage's own block). ``source`` and
    ``unit`` choose lexichron's defaults (a Davies corpus has its own)."""
    values, origin = defaults_for(stage, source, unit)
    sources = {k: origin for k in values}
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

"""Does a job's memory fit what the stage will use?

A batch script sets the worker count from the job's CPUs, so asking for
many CPUs and little memory is an easy mistake and an expensive one: the
job waits in the queue, runs for a few minutes, and Slurm kills it for
exceeding its memory. ``memory_estimate`` puts the stage's rough memory need
next to the allocation before the job is submitted. The per-process figures
are generous working numbers from runs on 5-grams, not measurements of this
corpus, so the estimate is a warning, not a limit.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["MemoryEstimate", "memory_estimate", "parse_mem_mb", "fmt_gb", "available_cpus"]


def available_cpus() -> int:
    """CPUs this process may run on: the allocation's share inside a Slurm
    job (``os.cpu_count()`` reports the whole node's), the machine's
    otherwise."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, n)

# Working figures, in MB. Acquire: a worker streams one file through
# on-disk chunks of ``chunk_entries`` packed entries (the default is 200k)
# plus its decompression buffers; the parent ingests. Tune: a training
# process holds the vocabulary and the model; the corpus writer and the
# scorer are in the parent.
#
# Filter, measured on the eng 5-grams (2026-11-17, each worker's own memory
# sampled every 2 s as PSS, which does not count shared pages twice):
# a worker holds about 280 MB while it scans (the lemmatizer's model, the
# source opened with a small cache, the batch it is filling), and at the end
# of its unit merges the sorted parts it spooled into one file, all of them
# open at once: about 0.25 MB a part on top. A pivoted run makes one record
# per n-gram and year, so a unit of the default partition (4 units a worker)
# spools 750 to 2,300 parts of 200k records and peaks at 500 to 840 MB; the
# fewer the workers, the larger the units and the higher each worker's peak.
# A unigram run's units are a few parts each. 100 workers at 250 MB each (the
# figure here before) ran out of memory in 30G.
#
# Slurm's own MaxRSS reads far higher than this for the same job (about 2.4
# times while scanning): it adds up every process's resident pages, and
# forked workers share most of theirs. A job's cgroup also fills with page
# cache from the spool files, which is reclaimed under pressure and is not
# part of the need.
ACQUIRE_BASE, ACQUIRE_PER_WORKER, ACQUIRE_CHUNK_ENTRIES = 3_000, 600, 200_000
FILTER_BASE = 4_000
FILTER_PER_WORKER = 400            # a run keyed by n-gram (the unigram run): scanning, and a small merge
FILTER_PER_WORKER_PIVOT = 900      # a pivoted run: the end-of-unit merge of a thousand or two parts
# A Davies corpus (COHA, 2026-11-24): acquisition with 20 workers peaked under
# 4.7G and the pivoting sentence filter with 20 under 6.4G by Slurm's MaxRSS,
# which overcounts; a worker's documents and its spooled units are small.
DAVIES_ACQUIRE_BASE, DAVIES_ACQUIRE_PER_WORKER = 2_000, 300
DAVIES_FILTER_PER_WORKER_PIVOT = 500
TUNE_BASE, TUNE_PER_TRAINER = 3_000, 600
TRAIN_PER_WRITER = 500          # a corpus writer streams the pivoted database with a small cache

_UNITS = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}


def parse_mem_mb(text: Any) -> Optional[int]:
    """Slurm's ``--mem`` syntax (``28G``, ``16000M``, ``0.5T``, bare MB) in MB,
    or None when it cannot be read."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", str(text or ""), re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


def fmt_gb(mb: int) -> str:
    gb = mb / 1024
    return f"{gb:.0f}G" if gb >= 10 or gb == int(gb) else f"{gb:.1f}G"


@dataclass
class MemoryEstimate:
    need_mb: int
    have_mb: Optional[int]        # None when the memory setting could not be read
    processes: int                # workers (acquire, filter) or trainers (tune)
    noun: str                     # "worker" / "training process"
    per_mb: int
    base_mb: int
    fix: str                      # how to make it fit

    @property
    def fits(self) -> bool:
        return self.have_mb is None or self.need_mb <= self.have_mb

    def describe(self) -> str:
        plural = "es" if self.noun.endswith("s") else "s"
        count = f"{self.processes} {self.noun}{plural if self.processes != 1 else ''}"
        line = f"{count} x {fmt_gb(self.per_mb)} + {fmt_gb(self.base_mb)} ≈ {fmt_gb(self.need_mb)}"
        if self.have_mb is None:
            return f"Memory: about {fmt_gb(self.need_mb)} needed ({line}); the memory setting could not be read"
        if self.fits:
            return f"Memory: about {fmt_gb(self.need_mb)} of {fmt_gb(self.have_mb)} ({line})"
        return (f"Memory: about {fmt_gb(self.need_mb)} needed ({line}) but the job asks for "
                f"{fmt_gb(self.have_mb)}; Slurm would kill it. {self.fix}")


def _int(value: Any, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def memory_estimate(stage: str, kwargs: Dict[str, Any], slurm: Dict[str, Any]) -> Optional[MemoryEstimate]:
    """The stage's rough memory need against the job's ``mem``, from the
    stage's resolved arguments and the ``slurm`` settings. None for a stage
    without a model."""
    cpus = max(1, _int(slurm.get("cpus"), 1))
    have = parse_mem_mb(slurm.get("mem"))
    davies = kwargs.get("source") == "davies"
    if stage == "acquire":
        workers = _int(kwargs.get("workers"), cpus)      # the script sets workers from the CPUs unless set
        base, per_worker = (DAVIES_ACQUIRE_BASE, DAVIES_ACQUIRE_PER_WORKER) if davies \
            else (ACQUIRE_BASE, ACQUIRE_PER_WORKER)
        per = int(per_worker * max(0.25, _int(kwargs.get("chunk_entries"), ACQUIRE_CHUNK_ENTRIES)
                                   / ACQUIRE_CHUNK_ENTRIES))
        need = base + per * workers
        fit_workers = max(1, ((have or 0) - base) // per) if have else 0
        fix = (f"Ask for {fmt_gb(need)}, or set acquire.workers to {fit_workers} (Advanced) so the script "
               f"does not tie the worker count to the {cpus} CPUs.")
        return MemoryEstimate(need, have, workers, "worker", per, base, fix)
    if stage == "filter":
        workers = _int(kwargs.get("workers"), cpus)
        # Pivoted (one record per n-gram and year) unless told otherwise: the
        # stage's own default is to pivot everything longer than a unigram.
        pivot = kwargs.get("pivot")
        if pivot is None:
            pivot = kwargs.get("unit", "sentences") != "tokens" if kwargs.get("source") == "davies" \
                else _int(kwargs.get("ngram_size"), 5) > 1
        per = (DAVIES_FILTER_PER_WORKER_PIVOT if davies else FILTER_PER_WORKER_PIVOT) if pivot \
            else FILTER_PER_WORKER
        need = FILTER_BASE + per * workers
        fit_workers = max(1, ((have or 0) - FILTER_BASE) // per) if have else 0
        fix = (f"Ask for {fmt_gb(need)}, or set filter.workers to {fit_workers} so the script does not tie "
               f"the worker count to the {cpus} CPUs.")
        return MemoryEstimate(need, have, workers, "worker", per, FILTER_BASE, fix)
    if stage in ("tune", "train"):
        total = _int(kwargs.get("workers"), cpus)
        threads = max(1, _int(kwargs.get("workers_per_model"), 4))
        trainers = max(1, total // threads)
        writers = max(1, _int(kwargs.get("corpus_writers"), 2)) if stage == "train" else 1
        base = TUNE_BASE + TRAIN_PER_WRITER * (writers - 1)        # tune's base already holds one writer
        need = base + TUNE_PER_TRAINER * trainers
        fit_threads = max(threads, -(-total // max(1, ((have or 0) - base) // TUNE_PER_TRAINER))) if have else threads
        fix = (f"Ask for {fmt_gb(need)}, or raise {stage}.workers_per_model to {fit_threads} so fewer models "
               f"train at once.")
        return MemoryEstimate(need, have, trainers, "training process", TUNE_PER_TRAINER, base, fix)
    return None

"""A job's memory against what its workers will use, before it is submitted."""
import os

from lexichron.resources import available_cpus, fmt_gb, memory_estimate, parse_mem_mb


def test_parse_slurm_mem():
    assert parse_mem_mb("28G") == 28 * 1024 and parse_mem_mb("16000M") == 16000
    assert parse_mem_mb("0.5T") == 512 * 1024 and parse_mem_mb("4096") == 4096 and parse_mem_mb(" 8gb ") == 8192
    assert parse_mem_mb("lots") is None and parse_mem_mb("") is None and parse_mem_mb(None) is None
    assert fmt_gb(28 * 1024) == "28G" and fmt_gb(1536) == "1.5G" and fmt_gb(47_800) == "47G"


def test_acquire_workers_follow_the_cpus_and_can_outgrow_the_memory():
    # this morning's job: 64 CPUs (so 64 workers) in 16G
    est = memory_estimate("acquire", {}, {"cpus": 64, "mem": "16G"})
    assert est.processes == 64 and not est.fits
    text = est.describe()
    assert "64 workers" in text and "16G" in text and "Slurm would kill it" in text
    assert "set acquire.workers to 22" in text          # (16384 - 3000) // 600
    # the README's 40 workers in 28G fit
    est = memory_estimate("acquire", {}, {"cpus": 40, "mem": "28G"})
    assert est.fits and est.describe().startswith("Memory: about ")
    # an explicit worker count overrides the CPUs; smaller chunks need less
    assert memory_estimate("acquire", {"workers": 20}, {"cpus": 64, "mem": "16G"}).fits
    assert memory_estimate("acquire", {"chunk_entries": 50_000}, {"cpus": 64, "mem": "16G"}).need_mb < est.need_mb + 20_000
    # an unreadable memory setting is reported, not treated as too small
    est = memory_estimate("acquire", {}, {"cpus": 64, "mem": "plenty"})
    assert est.fits and "could not be read" in est.describe()


def test_filter_and_tune_models():
    est = memory_estimate("filter", {}, {"cpus": 40, "mem": "28G"})
    assert est.fits and est.processes == 40
    est = memory_estimate("filter", {}, {"cpus": 96, "mem": "8G"})
    assert not est.fits and "filter.workers to 16" in est.describe()
    # tune: trainers = workers / threads per model
    est = memory_estimate("tune", {"workers_per_model": 16}, {"cpus": 64, "mem": "16G"})
    assert est.processes == 4 and est.fits and "training processes" in est.describe()
    est = memory_estimate("tune", {"workers_per_model": 1}, {"cpus": 64, "mem": "8G"})
    assert not est.fits and "raise tune.workers_per_model to 8" in est.describe()
    assert memory_estimate("davies", {}, {"cpus": 4, "mem": "8G"}) is None


def test_available_cpus_respects_the_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert available_cpus() == min(2, len(os.sched_getaffinity(0)))
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert available_cpus() == len(os.sched_getaffinity(0))

"""The one-line memory report a stage logs at each step (utilities/memory.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ngramprep.utilities.memory import cgroup_memory, memory_report  # noqa: E402


def _tree(tmp_path, limit_at_job="45097156608"):
    """A Slurm-like cgroup v2 tree: the limit is on the job, the process is in a step below it."""
    root = tmp_path / "cgroup"
    job = root / "system.slice" / "slurmstepd.scope" / "job_7"
    step = job / "step_batch" / "user" / "task_0"
    step.mkdir(parents=True)
    (step / "memory.max").write_text("max\n")
    (step / "memory.current").write_text("1000\n")
    (job / "step_batch" / "memory.max").write_text("max\n")
    (job / "memory.max").write_text(limit_at_job + "\n")
    (job / "memory.current").write_text(str(40 << 30) + "\n")
    (job / "memory.stat").write_text(f"anon {3 << 30}\nfile {37 << 30}\nfile_dirty 0\n")
    (job / "memory.events").write_text("low 0\noom 2\noom_kill 1\n")
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cgroup").write_text("0::/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0\n")
    return str(root), str(proc)


def test_the_report_reads_the_group_that_holds_the_limit(tmp_path):
    root, proc = _tree(tmp_path)
    group = cgroup_memory(root, proc)
    assert group == {"current": 40 << 30, "anon": 3 << 30, "file": 37 << 30, "limit": 45097156608, "oom_kills": 1}
    line = memory_report(root, proc)
    assert line.startswith("job 40.0 GB of 42.0 GB (3.0 GB held by processes, 37.0 GB file cache, which the "
                           "kernel can drop); 1 out-of-memory kill(s) so far")
    assert "this process" in line


def test_without_a_limit_or_without_cgroups_the_report_says_what_it_can(tmp_path):
    root, proc = _tree(tmp_path, limit_at_job="max")
    group = cgroup_memory(root, proc)
    assert group["limit"] is None and group["current"] == 1000          # the process's own group
    assert memory_report(root, proc).startswith("job 0.0 GB")
    assert cgroup_memory(str(tmp_path / "nowhere"), str(tmp_path / "nowhere")) is None
    assert "job" not in memory_report(str(tmp_path / "nowhere"), str(tmp_path / "nowhere"))     # and it does not raise

"""Write a Slurm batch script that runs a lexichron stage on a project file.

With ``slurm.jobs`` above 1 the script is a job array: Slurm runs that many
tasks, and the stage splits its years over them (see ``lexichron.arrays``).
Only the stages that work year by year can be split; the others run as one
job.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["stage_command", "write_sbatch", "ARRAY_STAGES", "array_size"]

from lexichron.config import ConfigError
from lexichron.slurm_settings import ARRAY_STAGES  # stages whose years split over a job array (re-exported)


def array_size(stage: str, slurm: Dict[str, Any]) -> int:
    """How many tasks the job runs: ``slurm.jobs`` for a stage that can be
    split, 1 otherwise or when unset."""
    if stage not in ARRAY_STAGES:
        return 1
    try:
        return max(1, int(slurm.get("jobs") or 1))
    except (TypeError, ValueError):
        return 1


def stage_command(project_yaml: Path, stage: str) -> List[str]:
    """How a batch job should invoke the stage.

    Prefer the project's host launcher (written by ``new-project``): it runs
    inside the image with the project's environment and bind mounts. Fall
    back to the image we are running in, then to a bare ``lexichron`` on PATH.
    """
    project_dir = project_yaml.parent
    launcher = project_dir / ".venv" / "host-python"
    if launcher.exists():
        return [str(launcher), "-m", "lexichron.cli", stage, str(project_yaml)]
    image = os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER")
    if image:
        return ["apptainer", "exec", image, "lexichron", stage, str(project_yaml)]
    return ["lexichron", stage, str(project_yaml)]


def write_sbatch(
    project_yaml: str | os.PathLike,
    stage: str,
    slurm: Dict[str, Any],
    *,
    out_path: Optional[str | os.PathLike] = None,
    extra_sets: Optional[List[str]] = None,
    assume_yes: bool = False,
) -> Path:
    """Write ``<project>.<stage>.sbatch`` next to the project file and return its path.

    ``slurm`` holds account, partition, cpus, mem, time, job_name, jobs (see
    ``schema.SLURM_FIELDS``). Missing optional keys are omitted from the
    script. ``extra_sets`` are passed through as ``--set`` overrides.
    With ``jobs`` above 1, for a stage in ``ARRAY_STAGES``, the script is a
    job array of that many tasks, each running the stage on its share of
    the years (``array_task`` / ``array_count`` set from Slurm's task
    variables) in resume mode: a restart would have every task delete the
    directory its siblings are filling, so the caller does the deleting
    before the array starts (the UI does, after asking) and the tasks
    resume into the empty directory.
    ``assume_yes`` adds ``--yes``: the stage then deletes earlier output its
    settings tell it to without asking, which a batch job cannot do (the UI
    sets it after the user confirmed in its dialog); without it a job whose
    settings would delete something stops at once instead.
    """
    if not slurm.get("account"):
        # Slurm would fall back to the user's default account, charging the
        # job to a project nobody chose; stop here instead.
        raise ConfigError("no Slurm account: fill in Account on the Submit tab (the first one you save becomes "
                          "your default for every corpus)")
    project_yaml = Path(project_yaml).resolve()
    out = Path(out_path) if out_path else project_yaml.with_suffix(f".{stage}.sbatch")
    log_dir = project_yaml.parent / ".lexichron" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    name = slurm.get("job_name") or f"lexichron-{stage}"
    tasks = array_size(stage, slurm)
    lines = ["#!/bin/bash", f"#SBATCH --job-name={name}"]
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    lines += [
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 1))}",
        f"#SBATCH --mem={slurm.get('mem', '8G')}",
        f"#SBATCH --time={slurm.get('time', '01:00:00')}",
    ]
    if tasks > 1:
        # one log per task, named by the array's job id and the task index
        lines += [f"#SBATCH --array=0-{tasks - 1}",
                  f"#SBATCH --output={log_dir}/%x_%A_%a.out",
                  f"#SBATCH --error={log_dir}/%x_%A_%a.err"]
    else:
        lines += [f"#SBATCH --output={log_dir}/%x_%j.out",
                  f"#SBATCH --error={log_dir}/%x_%j.err"]
    lines += [
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "# Spool and temp files on node-local disk when Slurm provides it",
        'export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}"',
        "",
    ]
    cmd = stage_command(project_yaml, stage)
    # Let the job's CPU count drive the worker count unless the project sets it.
    sets = list(extra_sets or [])
    if stage in ("acquire", "filter", "tune", "train") and not any(s.startswith(f"{stage}.workers=") for s in sets):
        sets.append(f"{stage}.workers=$SLURM_CPUS_PER_TASK")
    if tasks > 1:
        sets = [s for s in sets if not s.startswith((f"{stage}.array_task=", f"{stage}.array_count=",
                                                     f"{stage}.mode="))]
        sets += [f"{stage}.array_task=$SLURM_ARRAY_TASK_ID", f"{stage}.array_count=$SLURM_ARRAY_TASK_COUNT",
                 f"{stage}.mode=resume"]
    for s in sets:
        cmd += ["--set", s]
    if assume_yes:
        cmd.append("--yes")
    lines.append(" ".join(c if "=$SLURM_" in c else shlex.quote(c) for c in cmd))
    lines.append("")
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    return out

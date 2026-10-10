"""Write a Slurm batch script that runs a lexichron stage on a project file."""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["stage_command", "write_sbatch"]


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
) -> Path:
    """Write ``<project>.<stage>.sbatch`` next to the project file and return its path.

    ``slurm`` holds account, partition, cpus, mem, time, job_name (see
    ``schema.SLURM_FIELDS``). Missing optional keys are omitted from the
    script. ``extra_sets`` are passed through as ``--set`` overrides.
    """
    project_yaml = Path(project_yaml).resolve()
    out = Path(out_path) if out_path else project_yaml.with_suffix(f".{stage}.sbatch")
    log_dir = project_yaml.parent / ".lexichron" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    name = slurm.get("job_name") or f"lexichron-{stage}"
    lines = ["#!/bin/bash", f"#SBATCH --job-name={name}"]
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    lines += [
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 1))}",
        f"#SBATCH --mem={slurm.get('mem', '8G')}",
        f"#SBATCH --time={slurm.get('time', '01:00:00')}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
        f"#SBATCH --error={log_dir}/%x_%j.err",
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
    if stage == "acquire" and not any(s.startswith("acquire.workers=") for s in sets):
        sets.append("acquire.workers=$SLURM_CPUS_PER_TASK")
    for s in sets:
        cmd += ["--set", s]
    lines.append(" ".join(shlex.quote(c) if not c.startswith("acquire.workers=$") else c for c in cmd))
    lines.append("")
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    return out

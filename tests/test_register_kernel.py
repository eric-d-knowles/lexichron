"""The launcher register_kernel.sh writes, run with a fake `apptainer` on PATH."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "container" / "register_kernel.sh"
if not REGISTER.exists():                        # running inside the image
    REGISTER = Path("/opt/lexichron/register_kernel.sh")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason="needs bash and python3")


@pytest.fixture
def launcher(tmp_path):
    """Register a kernel for a fake image; return the launcher and a run helper."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    (fake_bin / "apptainer").write_text('#!/usr/bin/env bash\necho "apptainer $*"\n')
    (fake_bin / "apptainer").chmod(0o755)
    keep = [d for d in os.environ["PATH"].split(os.pathsep)
            if d and not any((Path(d) / r).exists() for r in ("apptainer", "singularity"))]
    env = dict(os.environ, PATH=os.pathsep.join([str(fake_bin)] + keep),
               APPTAINER_CONTAINER=str(tmp_path / "lexichron-9.9.9.sif"),
               JUPYTER_DATA_DIR=str(tmp_path / "jupyter"))
    env.pop("LEXICHRON_NV", None)
    path = tmp_path / "host-python"
    subprocess.run(["bash", str(REGISTER), "--launcher", str(path),
                    "--env", "CMDSTAN=/x/cmdstan"],
                   env=env, check=True, capture_output=True, text=True)

    def run(**extra):
        out = subprocess.run([str(path), "-c", "pass"], env=dict(env, **extra),
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()

    return path, run


def test_launcher_runs_python_in_the_image(launcher):
    _, run = launcher
    out = run(LEXICHRON_NV="0")
    assert out.startswith("apptainer exec ")
    assert "--env CMDSTAN=/x/cmdstan" in out
    assert out.endswith("lexichron-9.9.9.sif python -c pass")


def test_gpu_flag_can_be_forced_on_and_off(launcher):
    _, run = launcher
    on = run(LEXICHRON_NV="1")
    assert on.startswith("apptainer exec --nv ") and "JAX_PLATFORMS" not in on
    off = run(LEXICHRON_NV="0")
    assert "--nv" not in off.split()


def test_without_a_gpu_jax_is_pointed_at_the_cpu(launcher):
    _, run = launcher
    assert run(LEXICHRON_NV="0").startswith("apptainer exec --env JAX_PLATFORMS=cpu ")
    # a choice the user made is left alone (apptainer passes it through)
    assert "JAX_PLATFORMS" not in run(LEXICHRON_NV="0", JAX_PLATFORMS="cuda")


def test_gpu_flag_follows_the_host_by_default(launcher):
    _, run = launcher
    has_gpu = Path("/dev/nvidiactl").exists()
    assert ("--nv" in run().split()) == has_gpu

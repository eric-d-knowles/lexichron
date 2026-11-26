"""new_project.sh run outside the image: its /opt/lexichron paths point at the
repository's container/ scripts, and the environment's Python stands in for
the image's. Checks what a second run (an update to a new release) keeps."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "container"
if not (CONTAINER / "new_project.sh").exists():              # running inside the image
    CONTAINER = Path("/opt/lexichron")
sys.path.insert(0, str(CONTAINER))
import launcher_settings  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("python3") is None,
                                reason="needs bash and python3")


@pytest.fixture
def new_project(tmp_path):
    """A runner for a copy of new_project.sh that reads its helpers from container/."""
    script = tmp_path / "new_project.sh"
    script.write_text((CONTAINER / "new_project.sh").read_text().replace("/opt/lexichron/", f"{CONTAINER}/"))
    bindir = tmp_path / "bin"                                  # `python` must be this interpreter
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV", "VIRTUAL_ENV")}
    env.update(PATH=f"{bindir}{os.pathsep}{env['PATH']}", JUPYTER_DATA_DIR=str(tmp_path / "jupyter"),
               APPTAINER_CONTAINER=str(tmp_path / "lexichron-9.9.9.sif"))
    project = tmp_path / "proj"

    def run(*args):
        out = subprocess.run(["bash", str(script), str(project), "--no-install", *args],
                             env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stdout + out.stderr
        return out.stdout, (project / ".venv" / "host-python").read_text()

    return run, project, tmp_path


def test_settings_are_read_from_a_generated_launcher(new_project):
    run, project, tmp_path = new_project
    _, launcher = run("--env", "CMDSTAN=/x/cmd stan", "--bind", "/opt/extra")
    envs, binds = launcher_settings.settings(launcher, str(project))
    assert envs == ["CMDSTAN=/x/cmd stan"]
    assert binds == ["/opt/extra"]                             # not the standard roots, not the project


def test_an_update_keeps_the_launchers_settings(new_project):
    run, project, _ = new_project
    run("--env", "CMDSTAN=/x/cmdstan", "--env", "A=1", "--bind", "/opt/extra")
    out, launcher = run("--env", "A=2")                          # the update: only A given again
    assert "Keeping --env CMDSTAN=/x/cmdstan" in out and "Keeping --bind /opt/extra" in out
    envs, binds = launcher_settings.settings(launcher, str(project))
    assert envs == ["CMDSTAN=/x/cmdstan", "A=2"]                 # the new value wins, order kept
    assert binds == ["/opt/extra"]


def test_reset_drops_them(new_project):
    run, project, _ = new_project
    run("--env", "CMDSTAN=/x/cmdstan")
    _, launcher = run("--reset")
    assert launcher_settings.settings(launcher, str(project)) == ([], [])


def test_a_hand_edit_of_the_launcher_is_kept(new_project):
    run, project, _ = new_project
    _, launcher = run()
    path = project / ".venv" / "host-python"
    path.write_text(launcher.replace('$binds "$img"', '$binds --env CMDSTAN=/hand/made "$img"'))
    _, launcher = run()
    assert launcher_settings.settings(launcher, str(project))[0] == ["CMDSTAN=/hand/made"]


def test_the_kernel_runs_host_kernel_and_a_users_own_is_kept(new_project):
    run, project, tmp_path = new_project
    run()
    kernel = tmp_path / "jupyter" / "kernels" / "proj-lexichron-9.9.9" / "kernel.json"
    host_kernel = project / ".venv" / "host-kernel"
    assert json.loads(kernel.read_text())["argv"][0] == str(host_kernel)
    assert (project / ".lexichron" / "host_bridge.sh").exists()
    host_kernel.write_text("#!/usr/bin/env bash\n# mine\nexec \"$(dirname \"$0\")/host-python\" \"$@\"\n")
    out, _ = run()
    assert "Keeping your own" in out and "# mine" in host_kernel.read_text()

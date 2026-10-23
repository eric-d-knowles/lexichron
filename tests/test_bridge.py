"""The host bridge: shell helper + Python client, with fake Slurm commands."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from lexichron.bridge import HostBridge, BridgeUnavailable

BRIDGE_SH = Path(__file__).resolve().parents[1] / "container" / "host_bridge.sh"
if not BRIDGE_SH.exists():                       # inside the image the repo is not present
    BRIDGE_SH = Path("/opt/lexichron/host_bridge.sh")


@pytest.fixture
def bridge(tmp_path):
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "sbatch").write_text('#!/bin/sh\necho "4242;torch"\n')
    (fake / "squeue").write_text('#!/bin/sh\necho "4242|lexichron-acquire|RUNNING|00:01:00|12:00:00|1|cs604"\n')
    (fake / "scancel").write_text('#!/bin/sh\nexit 0\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")
    proc = subprocess.Popen(["bash", str(BRIDGE_SH), str(tmp_path / "bridge")], env=env)
    b = HostBridge(tmp_path / "bridge", poll_s=0.05)
    for _ in range(100):
        if b.available():
            break
        time.sleep(0.05)
    yield b
    (tmp_path / "bridge" / "stop").touch()
    proc.wait(timeout=5)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bridge_round_trip(bridge):
    assert bridge.available()
    assert bridge.sbatch("/x/project.acquire.sbatch") == "4242"
    rows = bridge.squeue(user="me")
    assert rows and rows[0]["job_id"] == "4242" and rows[0]["state"] == "RUNNING"
    assert bridge.scancel("4242").ok


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bridge_refuses_other_commands(bridge):
    res = bridge.run("rm", ["-rf", "/"])
    assert res.rc == 126 and "not allowed" in res.stderr


def test_missing_bridge_raises(tmp_path):
    with pytest.raises(BridgeUnavailable):
        HostBridge(tmp_path / "nope").run("squeue")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_two_bridges_on_one_dir_run_each_request_once(tmp_path):
    """Requests are claimed atomically: a second bridge on the same directory
    (two UI sessions) must not run sbatch a second time."""
    fake = tmp_path / "bin"; fake.mkdir()
    count = tmp_path / "count"
    (fake / "sbatch").write_text(f'#!/bin/sh\necho x >> {count}\nsleep 0.5\necho "77;torch"\n')
    (fake / "sbatch").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")
    procs = [subprocess.Popen(["bash", str(BRIDGE_SH), str(tmp_path / "bridge")], env=env) for _ in range(2)]
    b = HostBridge(tmp_path / "bridge", poll_s=0.05)
    for _ in range(100):
        if b.available():
            break
        time.sleep(0.05)
    try:
        for _ in range(5):
            assert b.sbatch("/x/p.sbatch") == "77"
        time.sleep(1.0)
        assert count.read_text().count("x") == 5
    finally:
        (tmp_path / "bridge" / "stop").touch()
        for p in procs:
            p.wait(timeout=5)


# -- direct backend: sbatch on this machine's PATH ----------------------------
@pytest.fixture
def fake_slurm_on_path(tmp_path, monkeypatch):
    fake = tmp_path / "bin"; fake.mkdir()
    log = tmp_path / "calls"
    (fake / "sbatch").write_text(f'#!/bin/sh\necho "sbatch $*" >> {log}\necho "4242;torch"\n')
    (fake / "squeue").write_text(f'#!/bin/sh\necho "squeue $*" >> {log}\n'
                                 'echo "4242|lexichron-acquire|RUNNING|00:01:00|12:00:00|1|cs604"\n')
    (fake / "sacct").write_text('#!/bin/sh\necho "4242|COMPLETED|00:10:00|1000K|0:0"\n')
    (fake / "scancel").write_text(f'#!/bin/sh\necho "scancel $*" >> {log}\nexit 0\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.delenv("LEXICHRON_BRIDGE_DIR", raising=False)
    return log


def test_local_slurm_runs_commands_directly(fake_slurm_on_path):
    from lexichron.bridge import LocalSlurm
    local = LocalSlurm()
    assert local.available()
    assert local.sbatch("/x/project.acquire.sbatch") == "4242"
    rows = local.squeue(user="me")
    assert rows and rows[0]["job_id"] == "4242" and rows[0]["state"] == "RUNNING"
    assert local.sacct("4242")[0]["state"] == "COMPLETED"
    assert local.scancel("4242").ok
    calls = fake_slurm_on_path.read_text().splitlines()
    assert calls == ["sbatch --parsable /x/project.acquire.sbatch",
                     "squeue --noheader --format=%i|%j|%T|%M|%l|%D|%R --user me",
                     "scancel 4242"]


def test_local_slurm_refuses_other_commands(fake_slurm_on_path):
    from lexichron.bridge import LocalSlurm
    res = LocalSlurm().run("rm", ["-rf", "/"])
    assert res.rc == 126 and "not allowed" in res.stderr


def test_local_slurm_reports_sbatch_failure(fake_slurm_on_path, tmp_path):
    from lexichron.bridge import LocalSlurm
    (tmp_path / "bin" / "sbatch").write_text('#!/bin/sh\necho "sbatch: error: invalid partition" >&2\nexit 1\n')
    with pytest.raises(RuntimeError, match="invalid partition"):
        LocalSlurm().sbatch("/x/p.sbatch")


def test_local_slurm_unavailable_without_sbatch(tmp_path, monkeypatch):
    from lexichron.bridge import LocalSlurm
    empty = tmp_path / "empty"; empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    local = LocalSlurm()
    assert not local.available()
    with pytest.raises(BridgeUnavailable):
        local.run("squeue")


def test_backend_prefers_bridge_dir_then_path(fake_slurm_on_path, tmp_path, monkeypatch):
    from lexichron.bridge import LocalSlurm, slurm_backend
    # sbatch on PATH and no LEXICHRON_BRIDGE_DIR: run directly
    assert isinstance(slurm_backend(), LocalSlurm)
    # the launcher set LEXICHRON_BRIDGE_DIR: use the bridge even though sbatch is on PATH
    monkeypatch.setenv("LEXICHRON_BRIDGE_DIR", str(tmp_path / "bridge"))
    b = slurm_backend()
    assert isinstance(b, HostBridge) and b.dir == tmp_path / "bridge"
    # neither: nothing
    monkeypatch.delenv("LEXICHRON_BRIDGE_DIR")
    empty = tmp_path / "empty"; empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert slurm_backend() is None

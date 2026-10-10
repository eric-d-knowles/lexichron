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

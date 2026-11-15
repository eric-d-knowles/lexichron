"""Killing an acquisition must never change a count.

Ingestion sums: a shard taken into the database twice is counted twice, and
nothing downstream could tell. What prevents it is that a shard's records and
its "processed" marker become durable together. These tests run the stage as
a separate process against the real RocksDB (see ``acquire_kill_driver.py``),
kill it at chosen points and at random ones, resume it, and require the
finished database to equal the one an uninterrupted run builds. The resumed
run's own verification (the database against the files' totals) has to pass
as well, or the driver would not exit with status 0.
"""
import gzip
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

rocks_shim = pytest.importorskip("rocks_shim")

from ngramprep.ngram_acquire.db.build_path import build_db_path  # noqa: E402

DRIVER = Path(__file__).with_name("acquire_kill_driver.py")
SHARDS = 24


def _make_shards(root: Path) -> None:
    """Shards whose keys overlap heavily: every n-gram arrives from three
    shards, so taking any one shard in twice changes hundreds of counts."""
    (root / "shards").mkdir(parents=True)
    for s in range(SHARDS):
        lines = [f"w{(s * 60 + i) % (SHARDS * 60):05d}_NOUN x_VERB\t1990,{s + i + 1},1\t2000,{i + 1},{s + 1}"
                 for i in range(180)]
        lines.append(f"shared_NOUN x_VERB\t1990,{s + 1},1")
        with gzip.open(root / "shards" / f"2-{s:05d}-of-{SHARDS:05d}.gz", "wt") as fh:
            fh.write("\n".join(lines) + "\n")


def _start(root: Path, *args: str) -> subprocess.Popen:
    """Start a run in its own process group, its output appended to
    ``<root>/driver.log`` (a file, not a pipe: workers orphaned by a kill
    would hold a pipe open)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(root / "driver.log", "a") as log:
        return subprocess.Popen([sys.executable, str(DRIVER), str(root), *args], env=env, start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)


def _reap(proc: subprocess.Popen) -> None:
    """Kill whatever is left of the run's process group (orphaned workers)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run(root: Path, *args: str, expect: int = 0) -> None:
    proc = _start(root, *args)
    try:
        proc.wait(timeout=300)
    finally:
        _reap(proc)
    assert proc.returncode == expect, (root / "driver.log").read_text()[-3000:]


def _dump(root: Path) -> dict:
    path = build_db_path(str(root / "out"), 2, "20200217", "eng")
    db = rocks_shim.open(path, mode="r", profile="read:packed24")
    try:
        out, markers = {}, {}
        it = db.iterator()
        it.seek(b"")
        while it.valid():
            key, value = bytes(it.key()), bytes(it.value())
            (markers if key.startswith(b"__processed__/") else out)[key] = value
            it.next()
        del it
    finally:
        db.close()
    assert len(markers) == SHARDS
    return out


def _progress(root: Path) -> dict:
    return json.loads((root / "progress.json").read_text())


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    _make_shards(root)
    _run(root)
    doc = _progress(root)
    assert doc["verification"]["status"] == "verified" and doc["database"]["compacted"]
    data = _dump(root)
    assert len(data) == SHARDS * 60 + 1
    return data


@pytest.mark.parametrize("spool_format", ["sst", "chunk"])
def test_death_right_after_a_shard_goes_in_does_not_count_it_twice(tmp_path, reference, spool_format):
    """The window that matters: the shard's records are in the database and
    the process dies before doing anything else. The marker went in with
    them, so the next run skips the shard."""
    _make_shards(tmp_path)
    _run(tmp_path, "--spool-format", spool_format, "--die-after-ingest", "7", expect=9)
    _run(tmp_path, "--spool-format", spool_format)
    doc = _progress(tmp_path)
    assert doc["files_skipped"] == 7                       # exactly the shards that went in, each with its marker
    assert doc["verification"]["status"] == "verified"
    assert _dump(tmp_path) == reference


def test_death_before_a_shard_goes_in_loses_nothing(tmp_path, reference):
    _make_shards(tmp_path)
    _run(tmp_path, "--die-before-ingest", "5", expect=9)
    _run(tmp_path)
    assert _progress(tmp_path)["files_skipped"] == 4
    assert _dump(tmp_path) == reference


@pytest.mark.parametrize("spool_format", ["sst", "chunk"])
def test_kills_at_random_moments_never_change_the_counts(tmp_path, reference, spool_format):
    _make_shards(tmp_path)
    rnd = random.Random(20200217)
    killed = 0
    for _ in range(3):
        started = tmp_path / "started"
        if started.exists():
            started.unlink()
        proc = _start(tmp_path, "--spool-format", spool_format, "--slow-ms", "40")
        try:
            deadline = time.time() + 120
            while not started.exists() and proc.poll() is None and time.time() < deadline:
                time.sleep(0.01)
            time.sleep(rnd.uniform(0.0, 0.45))             # somewhere in the ~0.4 s of ingestion
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                killed += 1
            proc.wait(timeout=60)
        finally:
            _reap(proc)
    assert killed >= 2                                      # the kills landed on running jobs
    _run(tmp_path, "--spool-format", spool_format)
    assert _progress(tmp_path)["verification"]["status"] == "verified"
    assert _dump(tmp_path) == reference

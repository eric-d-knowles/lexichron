from datetime import datetime, timedelta, timezone

from lexichron.ui.progress_view import fmt_bytes, fmt_count, fmt_duration, summarize, tail_lines
from ngramprep.ngram_acquire.progress import ProgressReporter, read_progress


def _doc(**over):
    now = datetime(2026, 10, 9, 22, 40, tzinfo=timezone.utc)
    d = dict(stage="acquire", state="running",
             started=(now - timedelta(minutes=10)).isoformat(timespec="seconds"),
             updated=now.isoformat(timespec="seconds"), finished=None,
             files_total=14, files_done=4, files_failed=0, files_skipped=0,
             entries_written=6_000_000, uncompressed_bytes=3 * 1024 ** 3, chunks=10,
             current=["a.gz"], message="", db_path="/x/1grams.db", log_path=None,
             slurm_job_id="123", hostname="cs001")
    d.update(over)
    return d, now


def test_formatters():
    assert fmt_duration(5) == "5s" and fmt_duration(65) == "1m05s" and fmt_duration(3700) == "1h01m"
    assert fmt_duration(None) == "-"
    assert fmt_bytes(512) == "512 B" and fmt_bytes(3 * 1024 ** 3) == "3.0 GB"
    assert fmt_count(950) == "950" and fmt_count(48_800) == "48.8k" and fmt_count(2_500_000) == "2.5M"


def test_summarize_running_with_eta():
    doc, now = _doc()
    s = summarize(doc, now)
    assert s["done"] == 4 and s["total"] == 14 and round(s["percent"]) == 29
    assert s["elapsed_s"] == 600 and s["rate"] == 10_000
    # 10 files left at 150 s each
    assert s["eta_s"] == 1500 and not s["stale"]
    assert s["headline"] == "running job 123 on cs001 · 4/14 files (29%)"
    assert "6,000,000 entries" in s["detail"] and "about 25m00s left" in s["detail"]
    assert s["current"] == ["a.gz"]


def test_summarize_done_uses_finished_time_and_skipped():
    doc, now = _doc(state="done", files_total=14, files_skipped=11, files_done=3)
    doc["finished"] = (now - timedelta(minutes=5)).isoformat(timespec="seconds")
    s = summarize(doc, now)
    assert s["total"] == 3 and s["percent"] == 100 and s["elapsed_s"] == 300
    assert s["eta_s"] is None and "11 already done" in s["headline"]
    assert s["headline"].startswith("done job 123")


def test_summarize_flags_stale_running_job():
    doc, now = _doc()
    s = summarize(doc, now + timedelta(minutes=20))
    assert s["stale"] and s["headline"].startswith("running? (no update for 20m00s)")


def test_summarize_before_totals_known():
    doc, now = _doc(files_total=0, files_done=0, entries_written=0, slurm_job_id=None)
    s = summarize(doc, now)
    assert s["percent"] is None and s["eta_s"] is None
    assert s["headline"] == "running on cs001 · 0 files"


def test_reporter_records_where_it_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    p = tmp_path / "progress.json"
    ProgressReporter(p, stage="acquire", db_path="/db/1grams.db", log_path=tmp_path / "x.log")
    doc = read_progress(p)
    assert doc["slurm_job_id"] == "999" and doc["db_path"] == "/db/1grams.db"
    assert doc["log_path"].endswith("x.log") and doc["hostname"]


def test_tail_lines(tmp_path):
    f = tmp_path / "log"
    f.write_text("".join(f"line {i}\n" for i in range(50)))
    assert tail_lines(f, 3) == ["line 47", "line 48", "line 49"]
    assert tail_lines(tmp_path / "missing") == [] and tail_lines(None) == []


def test_summarize_stopped_when_job_left_queue():
    doc, now = _doc()
    s = summarize(doc, now + timedelta(hours=3), job_gone=True)
    assert s["stopped"] and s["state_label"] == "stopped" and s["eta_s"] is None
    assert s["headline"].startswith("stopped (job no longer in the queue")
    # the clock stopped at the last write, not at 'now'
    assert s["elapsed_s"] == 600
    # a finished run is never 'stopped', whatever the queue says
    doc["state"] = "done"; doc["finished"] = doc["updated"]
    assert summarize(doc, now, job_gone=True)["state_label"] == "done"


def test_state_label_and_current_text():
    doc, now = _doc()
    assert summarize(doc, now)["state_label"] == "running"
    assert summarize(doc, now + timedelta(hours=1))["state_label"] == "running?"
    assert summarize(doc, now)["current_text"] == "1 files in flight"      # no phase data
    doc["current"] = [f"5-{i:05d}-of-11145.gz" for i in range(80)]
    doc["phases"] = {"queued": 40, "parsing": 36, "parsed": 3, "ingesting": 1}
    doc["ingesting"] = "5-00000-of-11145.gz"
    assert summarize(doc, now)["current_text"] == (
        "80 files in flight: 40 waiting for a worker · 36 downloading & parsing · "
        "3 parsed, waiting to ingest · 1 ingesting  (5-00000-of-11145.gz)")
    doc["current"] = []
    assert summarize(doc, now)["current_text"] == ""


def test_reporter_tracks_phases(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0, spool_dir=spool)
    for name in ("a.gz", "b.gz", "c.gz", "d.gz"):
        r.file_started(name)
    assert read_progress(p)["phases"] == {"queued": 4, "parsing": 0, "parsed": 0, "ingesting": 0}
    # workers pick up a, b, c (markers as the worker writes them: <idx>_<name>.started)
    for i, name in enumerate(("a.gz", "b.gz", "c.gz"), 1):
        (spool / f"{i:05d}_{name}.started").touch()
    r.file_parsed("a.gz"); r.file_parsed("b.gz")
    r.file_ingesting("a.gz")
    doc = read_progress(p)
    assert doc["phases"] == {"queued": 1, "parsing": 1, "parsed": 1, "ingesting": 1}
    assert doc["ingesting"] == "a.gz"
    r.file_done("a.gz", entries=1, chunks=1, uncompressed_bytes=1)
    (spool / "00001_a.gz.started").unlink()
    doc = read_progress(p)
    assert doc["ingesting"] is None and doc["phases"]["ingesting"] == 0
    assert doc["current"] == ["b.gz", "c.gz", "d.gz"]

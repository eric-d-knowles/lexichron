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
    # a filter run counts work units and reports scanned/kept
    doc2, _ = _doc(unit_label="work units", counters={"scanned": 1000, "kept": 250}, uncompressed_bytes=0)
    s2 = summarize(doc2, now)
    assert "4/14 work units" in s2["headline"] and "1,000 scanned, 250 kept (25.0%)" in s2["detail"]
    assert "parsed" not in s2["detail"]
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
    r.file_parsed("a.gz")
    (spool / "00002_b.gz.started").rename(spool / "00002_b.gz.parsed")   # worker finished b
    r.file_ingesting("a.gz")
    doc = read_progress(p)
    assert doc["phases"] == {"queued": 1, "parsing": 1, "parsed": 1, "ingesting": 1}
    assert doc["ingesting"] == "a.gz"
    r.file_done("a.gz", entries=1, chunks=1, uncompressed_bytes=1)
    (spool / "00001_a.gz.started").unlink()
    doc = read_progress(p)
    assert doc["ingesting"] is None and doc["phases"]["ingesting"] == 0
    assert doc["current"] == ["b.gz", "c.gz", "d.gz"]


def test_corpus_totals_span_runs():
    doc, now = _doc(files_total=14, files_skipped=11, files_done=3,
                    entries_written=100, uncompressed_bytes=1024 ** 3,
                    corpus={"files": 14, "entries": 1100, "bytes": 5 * 1024 ** 3, "unsized": 0})
    s = summarize(doc, now)
    assert s["corpus_text"] == "Whole corpus so far: 14/14 files · 1,100 entries · 5.0 GB parsed (uncompressed)"
    doc["corpus"]["unsized"] = 4
    assert summarize(doc, now)["corpus_text"].endswith("(excludes 4 files done before sizes were recorded)")
    doc["corpus"] = {"files": 0, "entries": 0, "bytes": 0, "unsized": 0}
    assert summarize(doc, now)["corpus_text"] == ""
    assert summarize({"state": "running"}, now)["corpus_text"] == ""     # old progress files


def test_reporter_adds_prior_to_corpus_totals(tmp_path):
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0)
    r.set_totals(files_total=5, files_skipped=2, prior={"files": 2, "entries": 10, "bytes": 1000, "unsized": 1})
    r.file_started("x.gz")
    r.file_done("x.gz", entries=5, chunks=1, uncompressed_bytes=500)
    doc = read_progress(p)
    assert doc["corpus"] == {"files": 3, "entries": 15, "bytes": 1500, "lines": 0, "unsized": 1}


def test_compacting_phase_is_reported():
    doc, now = _doc(files_total=14, files_done=14, current=[], phase="compacting")
    doc["phase_started"] = (now - timedelta(minutes=12)).isoformat(timespec="seconds")
    s = summarize(doc, now)
    assert s["state_label"] == "compacting" and s["phase"] == "compacting"
    assert s["headline"].startswith("compacting the database (all work ingested) (12m00s so far)")
    assert s["eta_s"] is None and "left" not in s["detail"]
    # a compaction whose heartbeat stopped is flagged like any other silent run
    assert summarize(doc, now + timedelta(minutes=20))["state_label"] == "running?"


def test_long_phase_heartbeat_keeps_updated_fresh(tmp_path):
    import time
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0)
    with r.long_phase("compacting", heartbeat_s=0.05):
        first = read_progress(p)
        assert first["phase"] == "compacting" and first["phase_started"]
        time.sleep(0.2)
        # the document was rewritten from the heartbeat thread meanwhile
        assert p.stat().st_mtime >= p.stat().st_mtime   # file exists and is readable
        assert read_progress(p)["phase"] == "compacting"
    assert read_progress(p)["phase"] == "ingesting"       # back to normal after the phase
    r.finish("done")
    assert read_progress(p)["state"] == "done"


def test_filter_runs_describe_work_units_not_files():
    from lexichron.ui.progress_view import summarize
    doc = {"stage": "filter", "state": "running", "unit_label": "work units", "files_total": 80,
           "files_done": 10, "entries_written": 1000, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:01:00+00:00", "current": ["u1", "u2", "u3"],
           "phases": {"parsing": 2, "parsed": 1}}
    s = summarize(doc, now=__import__("datetime").datetime.fromisoformat("2026-10-12T00:01:00+00:00"), job_gone=False)
    assert s["current_text"] == "3 work units in flight: 2 scanning · 1 scanned, waiting to ingest"
    assert "10/80 work units" in s["headline"]


def test_inputs_line_names_the_vocabulary_in_use():
    from lexichron.ui.progress_view import summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T00:01:00+00:00")
    base = {"stage": "filter", "state": "running", "unit_label": "work units", "files_total": 8,
            "files_done": 1, "entries_written": 10, "started": "2026-10-12T00:00:00+00:00",
            "updated": "2026-10-12T00:01:00+00:00"}
    s = summarize(dict(base, inputs={"source": "/c/5grams.db", "vocabulary": "/c/1grams_processed.db/whitelist.txt",
                                     "always_include": "he, she"}), now=now, job_gone=False)
    assert s["inputs_text"] == ("Source: /c/5grams.db  ·  Vocabulary: /c/1grams_processed.db/whitelist.txt"
                                "  ·  Always keep: he, she")
    s = summarize(dict(base, inputs={"source": "/c/1grams.db", "vocabulary": None}), now=now, job_gone=False)
    assert s["inputs_text"] == "Source: /c/1grams.db  ·  Vocabulary: none (rule filters only)"
    assert summarize(base, now=now, job_gone=False)["inputs_text"] == ""     # older progress files


def test_filter_runs_have_no_whole_corpus_line():
    from lexichron.ui.progress_view import summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T00:01:00+00:00")
    doc = {"stage": "filter", "state": "running", "unit_label": "work units", "files_total": 80,
           "files_done": 3, "entries_written": 10, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:01:00+00:00", "corpus": {"files": 3, "entries": 10, "bytes": 0, "unsized": 0}}
    assert summarize(doc, now=now, job_gone=False)["corpus_text"] == ""


def test_meters_keep_the_three_record_counts_apart():
    from lexichron.ui.progress_view import meters, summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T00:10:00+00:00")
    doc = {"stage": "filter", "state": "done", "unit_label": "work units", "files_total": 80, "files_done": 80,
           "entries_written": 15_436_675, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:06:00+00:00", "finished": "2026-10-12T00:06:00+00:00",
           "counters": {"scanned": 26_291_248, "kept": 15_436_675, "inserted": 10_924_676,
                        "tokens": 26_291_248, "tokens_kept": 15_436_675, "tokens_not_alpha": 5_000_000,
                        "tokens_too_short": 1_000_000, "tokens_stopword": 4_854_573, "tokens_not_in_vocab": 0,
                        "ngrams": 26_291_248, "ngrams_dropped_context": 10_854_573},
           "database": {"records": 8_210_776, "bytes": 1 << 30, "compacted": True}}
    groups = dict(meters(doc, summarize(doc, now=now, job_gone=False)))
    records = {label: (value, note) for label, value, note in groups["Records"]}
    assert records["Scanned"] == ("26,291,248", "source records")
    assert records["Kept"] == ("15,436,675", "58.7% of scanned")
    assert records["Inserted"] == ("10,924,676", "to the database")
    assert records["In database"] == ("8,210,776", "after compaction")
    rules = {label: value for label, value, _ in groups["Dropped by rule"]}
    assert rules == {"Not alphabetic": "5,000,000", "Too short": "1,000,000", "Stopword": "4,854,573",
                     "Too few real tokens": "10,854,573"}
    # with a vocabulary only that rule runs, so only it is shown
    doc["inputs"] = {"vocabulary": "/w.txt"}
    rules = [label for label, _, _ in dict(meters(doc, summarize(doc, now=now, job_gone=False)))["Dropped by rule"]]
    assert rules == ["Not in vocabulary", "Too few real tokens"]
    assert groups["Progress"][0][:2] == ("Work units", "80/80")

    acq = {"stage": "acquire", "state": "running", "files_total": 100, "files_done": 10, "entries_written": 500,
           "uncompressed_bytes": 5 << 30, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:10:00+00:00", "counters": {"lines": 1234, "inserted": 500},
           "corpus": {"files": 40, "entries": 2000, "bytes": 20 << 30, "lines": 5000, "unsized": 0}}
    groups = dict(meters(acq, summarize(acq, now=now, job_gone=False)))
    session = {label: value for label, value, _ in groups["Records \u2013 this session"]}
    assert session == {"Lines read": "1,234", "Parsed": "5.0 GB", "Inserted": "500"}
    # the database is cumulative over acquisition jobs, so it sits with the corpus totals; no file count there
    corpus = {label: value for label, value, _ in groups["Records \u2013 entire corpus"]}
    assert corpus == {"Lines read": "5,000", "Parsed": "20.0 GB", "Inserted": "2,000", "In database": "\u2014"}
    # a filter document written before the unit label was recorded still counts work units
    old = {"stage": "filter", "state": "running", "files_total": 8, "files_done": 1, "entries_written": 1,
           "started": "2026-10-12T00:00:00+00:00", "updated": "2026-10-12T00:10:00+00:00"}
    assert dict(meters(old, summarize(old, now=now, job_gone=False)))["Progress"][0][0] == "Work units"


def test_in_flight_meters_follow_the_stage():
    from lexichron.ui.progress_view import meters, summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T00:01:00+00:00")
    doc = {"stage": "filter", "state": "running", "unit_label": "work units", "files_total": 80,
           "files_done": 3, "entries_written": 10, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:01:00+00:00", "current": ["a", "b", "c"],
           "phases": {"queued": 1, "parsing": 2, "parsed": 0, "ingesting": 0}}
    groups = dict(meters(doc, summarize(doc, now=now, job_gone=False)))
    assert groups["In flight: 3 work units"] == [("Waiting for a worker", "1", ""), ("Scanning", "2", ""),
                                                 ("Scanned, waiting to ingest", "0", ""), ("Ingesting", "0", "")]
    doc.update(stage="acquire", unit_label="files")
    assert [l for l, _, _ in dict(meters(doc, summarize(doc, now=now, job_gone=False)))["In flight: 3 files"]][1] == "Downloading & parsing"

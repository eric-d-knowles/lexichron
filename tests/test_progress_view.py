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
    # a finished file is adopted by the database in one short step: one wait, not two
    assert summarize(doc, now)["current_text"] == (
        "80 files in flight: 40 waiting for a worker · 36 downloading & parsing · "
        "4 waiting for the database  (5-00000-of-11145.gz)")
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
    assert s["current_text"] == "3 work units in flight: 2 scanning · 1 waiting for the database"
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
    assert records["Scanned"] == ("26,291,248", "")
    assert records["Kept"] == ("15,436,675", "58.7%")
    assert records["Inserted"] == ("10,924,676", "")
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
    assert session == {"N-grams scanned": "1,234", "Kept": "500", "Parsed": "5.0 GB"}
    # the database is cumulative over acquisition jobs, so it sits with the corpus totals; no file count there
    corpus = {label: value for label, value, _ in groups["Records \u2013 entire corpus"]}
    assert corpus == {"N-grams scanned": "5,000", "Kept": "2,000", "Parsed": "20.0 GB", "In database": "\u2014"}
    assert dict((l, n) for l, _v, n in groups["Records \u2013 entire corpus"])["Kept"] == "40.0%"
    assert "Checks" not in groups                       # a run from before the checks existed
    # a filter document written before the unit label was recorded still counts work units
    old = {"stage": "filter", "state": "running", "files_total": 8, "files_done": 1, "entries_written": 1,
           "started": "2026-10-12T00:00:00+00:00", "updated": "2026-10-12T00:10:00+00:00"}
    assert dict(meters(old, summarize(old, now=now, job_gone=False)))["Progress"][0][0] == "Work units"


def test_pivot_runs_show_year_records_between_kept_and_inserted():
    from lexichron.ui.progress_view import meters, summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T00:10:00+00:00")
    doc = {"stage": "filter", "state": "running", "unit_label": "work units", "files_total": 160,
           "files_done": 23, "entries_written": 85_000_000, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T00:10:00+00:00",
           "counters": {"scanned": 186_205_646, "kept": 85_079_727, "produced": 2_978_013_677,
                        "inserted": 2_103_834_198}}
    records = [(label, value) for label, value, _ in dict(meters(doc, summarize(doc, now=now, job_gone=False)))["Records"]]
    assert records == [("Scanned", "186,205,646"), ("Kept", "85,079,727"), ("Year records", "2,978,013,677"),
                       ("Inserted", "2,103,834,198"), ("In database", "\u2014")]
    # a one-to-one mapper (unigrams) has no such tile
    doc["counters"]["produced"] = doc["counters"]["kept"]
    assert "Year records" not in [l for l, _, _ in dict(meters(doc, summarize(doc, now=now, job_gone=False)))["Records"]]


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
                                                 ("Waiting for the database", "0", "")]
    doc.update(stage="acquire", unit_label="files", phases={"queued": 1, "parsing": 2, "parsed": 3, "ingesting": 1})
    assert dict(meters(doc, summarize(doc, now=now, job_gone=False)))["In flight: 3 files"] == [
        ("Waiting for a worker", "1", ""), ("Downloading & parsing", "2", ""), ("Waiting for the database", "4", "")]


def test_train_runs_read_as_ensembles_not_acquisitions():
    """A train run counts models and years; nothing about it is a download,
    a parsed byte or a database record."""
    from lexichron.ui.progress_view import meters, summarize
    import datetime
    now = datetime.datetime.fromisoformat("2026-10-12T01:00:00+00:00")
    doc = {"stage": "train", "state": "running", "unit_label": "models", "files_total": 320,
           "files_done": 16, "entries_written": 16 * 30000, "started": "2026-10-12T00:00:00+00:00",
           "updated": "2026-10-12T01:00:00+00:00", "current": ["y1972/c08_s0", "y1972/c08_s1", "y1972/c09_s0"],
           "phases": {"queued": 1, "parsing": 2, "parsed": 0, "ingesting": 0},
           "counters": {"models": 16, "train_seconds": 7200, "corpus_lines": 900, "years": 4, "years_done": 0},
           "array": {"task": 4, "count": 13, "years": [1972, 1985, 1998, 2011]}}
    info = summarize(doc, now=now, job_gone=False)
    assert "3 models in flight: 1 waiting · 2 training" in info["current_text"]
    assert "16/320 models" in info["headline"] and "years 1972, 1985, 1998, 2011" in info["headline"]
    groups = meters(doc, info)
    assert [title for title, _ in groups] == ["Progress", "In flight: 3 models", "Ensemble"]
    tiles = dict(groups)
    assert tiles["Progress"][1] == ("Rate", "16.0/h", "models")
    assert tiles["In flight: 3 models"][1] == ("Training", "2", "")
    assert tiles["Ensemble"][:3] == [("Models trained", "16", "this run"), ("Years complete", "0/4", "records written"),
                                     ("Training time", "2h00m", "summed over models")]
    text = repr(groups)
    for word in ("Download", "Parsed", "Inserted", "In database", "entries"):
        assert word not in text, word
    # the years whose replicates are coming in; the noise profile itself is the Inspect tab's
    doc["ensemble"] = {"corpora": 40, "seeds": 2, "per_year": 80, "in_hand": {"1972": 16, "1985": 2},
                       "years": {"1968": {"axis_se_mean": 0.002}}}
    groups = dict(meters(doc, summarize(doc, now=now, job_gone=False)))
    assert groups["Replicates in hand (40 resamples × 2 seeds)"] == [("1972", "16/80", "20%"), ("1985", "2/80", "2%")]
    assert not any("profile" in title.lower() for title in groups)



# ---- an acquisition's screen: size-weighted progress, steps, scan rate, checks --------

def _acq(**over):
    """A run 10 minutes into its files: 3 of 10 done, which is 60% of the bytes."""
    now = datetime(2026, 11, 15, 12, 10, tzinfo=timezone.utc)
    start = (now - timedelta(minutes=10)).isoformat(timespec="seconds")
    d = dict(stage="acquire", state="running", unit_label="files", phase="ingesting", phase_started=start,
             started=start, updated=now.isoformat(timespec="seconds"), finished=None,
             steps=["ingesting", "compacting", "verifying"],
             phase_times={"ingesting": {"started": start, "finished": None}},
             files_total=10, files_done=3, files_failed=0, files_skipped=0, entries_written=90_000,
             uncompressed_bytes=3 << 30, bytes_total=1000, bytes_done=500, in_flight={"lines": 600_000, "bytes": 100},
             counters={"lines": 3_000_000, "inserted": 90_000, "empty_files": 1, "match_total": 777,
                       "volume_total": 70, "problem_lines": 0, "reordered": 0},
             corpus={"files": 3, "entries": 90_000, "bytes": 3 << 30, "lines": 3_000_000, "unsized": 0,
                     "empty_files": 1, "match_total": 777, "volume_total": 70, "problem_lines": 0, "reordered": 0},
             checks={"verify_counts": True, "allow_unparsed_lines": False},
             current=["a.gz", "b.gz"], phases={"queued": 0, "parsing": 2, "parsed": 0, "ingesting": 0},
             message="", db_path="/x/5grams.db", log_path=None, slurm_job_id=None, hostname="cs001")
    d.update(over)
    return d, now


def test_acquire_progress_is_weighed_by_size_and_time_left_covers_the_files_only():
    from lexichron.ui.progress_view import meters
    doc, now = _acq()
    info = summarize(doc, now)
    # 500 bytes of finished files + 100 reported by workers mid-file, of 1000
    assert info["by_size"] and abs(info["fraction"] - 0.6) < 1e-9 and info["percent"] == 30.0
    assert abs(info["eta_s"] - 400) < 1e-6                    # 400 bytes left at 600 bytes in 600 s
    assert info["step_text"] == "step 1 of 3" and info["then_text"] == "then compaction and the count check"
    assert info["headline"].startswith("step 1 of 3: downloading and parsing the files on cs001 · 3/10 files (60% by size)")
    # what is read per second, files still in flight included; not what happens to be kept
    assert info["scanning"] and info["rate_unit"] == "n-grams scanned/s" and info["rate"] == 3_600_000 / 600
    tiles = {label: (value, note) for label, value, note in dict(meters(doc, info))["Progress"]}
    assert tiles["Files"] == ("3/10", "60% by size")
    assert tiles["Scan rate"][0] == "6.0k/s"
    assert tiles["Elapsed"][1] == "step 1 of 3"
    assert tiles["Remaining"] == ("6m40s", "files only") and tiles["Scan rate"][1] == "n-grams"
    # without sizes (an index page gives none) it falls back to counting files
    doc, now = _acq(bytes_total=None, in_flight={"lines": 0, "bytes": 0})
    info = summarize(doc, now)
    assert not info["by_size"] and info["fraction"] == 0.3 and abs(info["eta_s"] - 7 * 200) < 1e-6
    assert "3/10 files (30%)" in info["headline"]


def test_acquire_tiles_are_the_same_from_the_first_frame():
    """Before any file is read the screen already has every tile it will
    have, at zero: nothing appears or is renamed as the run gets going."""
    from lexichron.ui.progress_view import meters
    doc, now = _acq()
    late = [[label for label, _, _ in tiles] for _title, tiles in meters(doc, summarize(doc, now))]
    doc.update(files_done=0, bytes_done=0, entries_written=0, uncompressed_bytes=0, counters={},
               in_flight={"lines": 0, "bytes": 0}, current=[], phases={"queued": 0, "parsing": 0, "parsed": 0, "ingesting": 0},
               corpus={"files": 0, "entries": 0, "bytes": 0, "lines": 0, "unsized": 0})
    info = summarize(doc, now)
    early = [[label for label, _, _ in tiles] for _title, tiles in meters(doc, info)]
    assert early == late and info["scanning"] and info["rate"] == 0


def test_acquire_steps_after_the_files_are_named_and_stop_the_clock_on_the_rate():
    from lexichron.ui.progress_view import meters
    doc, now = _acq(phase="compacting", files_done=10, bytes_done=1000, in_flight={"lines": 0, "bytes": 0},
                    counters={"lines": 6_000_000, "inserted": 100})
    start = datetime.fromisoformat(doc["started"])
    files_end = (start + timedelta(minutes=5)).isoformat(timespec="seconds")
    doc["phase_started"] = files_end
    doc["phase_times"] = {"ingesting": {"started": doc["started"], "finished": files_end},
                          "compacting": {"started": files_end, "finished": None}}
    info = summarize(doc, now)
    assert info["headline"].startswith("step 2 of 3: compacting the database (all work ingested) (5m00s so far)")
    assert info["eta_s"] is None and info["then_text"] == "then the count check"
    assert info["rate"] == 6_000_000 / 300                    # over the five minutes of reading, not the ten elapsed
    tiles = {label: (value, note) for label, value, note in dict(meters(doc, info))["Progress"]}
    assert tiles["Remaining"] == ("\u2014", "")
    assert tiles["Elapsed"][1] == "step 2 of 3"
    doc["phase"] = "verifying"
    assert summarize(doc, now)["then_text"] == "" and summarize(doc, now)["step_text"] == "step 3 of 3"
    # once it is over, the steps are not "in progress" any more
    assert summarize(dict(doc, state="done", finished=doc["updated"]), now)["step_text"] == "all 3 steps done"
    ended = summarize(dict(doc, state="failed", phase="compacting", finished=doc["updated"]), now)
    assert ended["step_text"] == "ended in step 2 of 3" and ended["then_text"] == ""
    assert summarize(dict(doc, slurm_job_id="9"), now, job_gone=True)["step_text"] == "ended in step 3 of 3"


def test_acquire_records_say_scanned_kept_and_files_with_nothing():
    from lexichron.ui.progress_view import meters
    doc, now = _acq()
    groups = dict(meters(doc, summarize(doc, now)))
    assert groups["Records \u2013 this session"] == [
        ("N-grams scanned", "3,600,000", ""),
        ("Kept", "90,000", "3.0%"),
        ("Nothing kept", "1", "files"),
        ("Parsed", "3.0 GB", "uncompressed")]
    corpus = {label: value for label, value, _ in groups["Records \u2013 entire corpus"]}
    # the corpus total includes the files in flight too: it never trails the session's
    assert corpus == {"N-grams scanned": "3,600,000", "Kept": "90,000", "Nothing kept": "1",
                      "Parsed": "3.0 GB", "In database": "\u2014"}


def test_acquire_checks_are_shown_from_the_start():
    from lexichron.ui.progress_view import meters

    def checks(doc, now):
        return {label: (value, note) for label, value, note in dict(meters(doc, summarize(doc, now)))["Checks"]}

    doc, now = _acq()
    tiles = checks(doc, now)
    # a note is a figure or a caveat, never an explanation: a plain state has none
    assert tiles == {"Counts": ("pending", ""), "Occurrences": ("777", ""), "Unread lines": ("0", ""),
                     "Reordered values": ("0", "")}
    doc["phase"] = "verifying"
    assert checks(doc, now)["Counts"][0] == "checking"
    doc["checks"]["verify_counts"] = False
    assert checks(doc, now)["Counts"] == ("not checked", "verify_counts off")
    doc["checks"]["verify_counts"] = True

    doc.update(state="done", verification={"status": "verified", "ledger": {"match_total": 900, "reordered": 2},
                                           "database": {"match_total": 900, "volume_total": 80}})
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("verified", "") and tiles["Occurrences"] == ("900", "")
    assert tiles["Reordered values"] == ("2", "corrected")
    doc["verification"]["ledger"]["problem_lines"] = 3
    assert checks(doc, now)["Unread lines"] == ("3", "see log")

    doc.update(state="failed", verification={"status": "mismatch",
                                             "ledger": {"match_total": 900, "volume_total": 80},
                                             "database": {"match_total": 1350, "volume_total": 95}})
    assert checks(doc, now)["Counts"] == ("DO NOT MATCH", "occurrences +450; volume counts +15")
    doc["verification"] = {"status": "unverifiable", "ledger": {"unledgered": 4}, "database": {"match_total": 5}}
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("not verifiable", "") and tiles["Files without totals"] == ("4", "pre-ledger")
    del doc["verification"]
    assert checks(doc, now)["Counts"][0] == "not reached"       # e.g. a run that ended over failed files


def test_acquire_warnings_say_what_a_problem_will_lead_to():
    from lexichron.ui.progress_view import notice_text
    doc, now = _acq()
    assert summarize(doc, now)["warnings"] == [] and notice_text(summarize(doc, now)) == ""

    doc["counters"]["problem_lines"] = doc["corpus"]["problem_lines"] = 3
    (w,) = summarize(doc, now)["warnings"]
    assert w.startswith("3 lines had counts that could not be read in full")
    assert "will finish its work and then end with an error" in w and "allow_unparsed_lines accepts them" in w
    doc["checks"]["allow_unparsed_lines"] = True
    assert summarize(doc, now)["warnings"][0].endswith("Accepted: allow_unparsed_lines is on.")
    doc["counters"]["problem_lines"] = doc["corpus"]["problem_lines"] = 0

    doc.update(files_failed=4, message="NETWORK_ERROR: d.gz",
               failed_files=[{"file": f, "message": "NETWORK_ERROR"} for f in ("a.gz", "b.gz", "c.gz", "d.gz")])
    (w,) = summarize(doc, now)["warnings"]
    assert w.startswith("4 files failed (a.gz, b.gz, c.gz, ...). The run will finish the other files and then stop")
    assert "without compacting or checking the counts" in w
    assert notice_text(summarize(doc, now)) == w                # the last failure's message adds nothing
    doc.update(state="failed", message="4 file(s) failed")
    assert "The database is incomplete and was not compacted; run again to retry them." in summarize(doc, now)["warnings"][0]

    doc.update(files_failed=0, failed_files=[], message="the counts in the database do not match what the files contributed",
               verification={"status": "mismatch", "ledger": {"match_total": 900, "volume_total": 80},
                             "database": {"match_total": 1350, "volume_total": 95, "disordered": 1, "malformed": 0}})
    (w,) = summarize(doc, now)["warnings"]
    assert w == ("The counts in the database do not match what the files contributed: occurrences differ by +450, "
                 "volume counts by +15; 1 values are malformed or have years out of order. Do not use this database.")
    # a message no warning covers is still shown; other stages have no warnings of this kind
    info = summarize(dict(doc, verification={}, message="compaction failed: no space"), now)
    assert notice_text(info) == "compaction failed: no space"
    assert summarize(dict(doc, stage="tune"), now)["warnings"] == []


def test_reporter_keeps_the_figures_the_screen_needs(tmp_path):
    """The plan of steps and their times, sizes of the files to fetch, what
    workers say of files in flight, failed files by name, and the ledger."""
    spool = tmp_path / "spool"
    spool.mkdir()
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0, spool_dir=spool)
    r.set_plan(["ingesting", "compacting", "verifying"])
    r.set_checks(verify_counts=True, allow_unparsed_lines=False)
    r.set_totals(files_total=5, files_skipped=1,
                 prior={"files": 1, "entries": 10, "bytes": 100, "lines": 50, "unsized": 0, "empty_files": 1,
                        "match_total": 1000, "volume_total": 10, "problem_lines": 0, "reordered": 0, "unledgered": 0})
    r.set_sizes({"a.gz": 100, "b.gz": 300, "c.gz": 600, "z.gz": 5}, ["a.gz", "b.gz", "c.gz", "d.gz"])
    assert read_progress(p)["bytes_total"] is None                # d.gz has no size: no weighing by size
    r.set_sizes({"a.gz": 100, "b.gz": 300, "c.gz": 600, "d.gz": 1000}, ["a.gz", "b.gz", "c.gz", "d.gz"])
    doc = read_progress(p)
    assert (doc["bytes_total"], doc["bytes_done"]) == (2000, 0) and doc["steps"][0] == "ingesting"
    assert doc["checks"] == {"verify_counts": True, "allow_unparsed_lines": False}

    for name in ("a.gz", "b.gz", "c.gz", "d.gz"):
        r.file_started(name)
    (spool / "00001_a.gz.started").write_text("700 90")           # as a worker reports: lines, bytes received
    (spool / "00002_b.gz.parsed").write_text("4000 300")          # finished reading, waiting for the database
    (spool / "00003_c.gz.started").write_text("12")               # caught half-written: ignored
    (spool / "00004_d.gz.started").touch()                        # just picked up
    r.file_active("a.gz")
    assert read_progress(p)["in_flight"] == {"lines": 4700, "bytes": 390}

    r.file_done("b.gz", entries=0, chunks=0, uncompressed_bytes=9,
                counters={"lines": 4000, "inserted": 0, "empty_files": 1, "match_total": 0, "volume_total": 0})
    doc = read_progress(p)                                        # b's marker is still there: not counted twice
    assert doc["in_flight"] == {"lines": 700, "bytes": 90} and doc["bytes_done"] == 300
    assert doc["corpus"]["empty_files"] == 2 and doc["corpus"]["match_total"] == 1000 and doc["corpus"]["lines"] == 4050
    r.file_failed("c.gz", "NETWORK_ERROR: c.gz")
    doc = read_progress(p)
    assert doc["failed_files"] == [{"file": "c.gz", "message": "NETWORK_ERROR: c.gz"}] and doc["bytes_done"] == 900

    with r.long_phase("compacting"):
        doc = read_progress(p)
        assert doc["phase"] == "compacting" and doc["phase_times"]["ingesting"]["finished"]
    doc = read_progress(p)
    # a planned step stays the current phase when it ends: the run is between steps, not back at its files
    assert doc["phase"] == "compacting" and doc["phase_times"]["compacting"]["finished"]
    r.set_verification({"status": "verified"})
    assert read_progress(p)["verification"] == {"status": "verified"}


# ---- a filter run's screen: steps, scan rate, checks ---------------------------------------

def _flt(**over):
    """A filter run 10 minutes into its work units: 20 of 80 done."""
    now = datetime(2026, 11, 16, 12, 10, tzinfo=timezone.utc)
    start = (now - timedelta(minutes=10)).isoformat(timespec="seconds")
    d = dict(stage="filter", state="running", unit_label="work units", phase="ingesting", phase_started=start,
             started=start, updated=now.isoformat(timespec="seconds"), finished=None,
             steps=["ingesting", "compacting", "verifying", "whitelist"],
             phase_times={"ingesting": {"started": start, "finished": None}},
             files_total=80, files_done=20, files_failed=0, files_skipped=0, entries_written=500_000,
             in_flight={"lines": 400_000, "bytes": 0},
             counters={"scanned": 2_000_000, "kept": 500_000, "produced": 500_000, "inserted": 450_000,
                       "match_total": 9_000, "volume_total": 800, "source_match": 30_000, "source_volume": 2_000,
                       "reordered": 0, "source_disordered": 0, "dropped_match": 21_000, "dropped_volume": 1_200},
             checks={"verify_counts": True, "source_known": True},
             current=["u1", "u2"], phases={"queued": 0, "parsing": 2, "parsed": 0, "ingesting": 0},
             inputs={"source": "/x/5grams.db"}, message="", db_path="/x/5grams_pivoted.db", log_path=None,
             slurm_job_id=None, hostname="cs001")
    d.update(over)
    return d, now


def test_filter_progress_names_its_steps_and_reports_what_is_being_scanned():
    from lexichron.ui.progress_view import meters
    doc, now = _flt()
    info = summarize(doc, now)
    assert info["headline"].startswith("step 1 of 4: scanning and filtering the work units on cs001 · 20/80 work units (25%)")
    assert info["then_text"] == "then compaction, the count check and the vocabulary"
    # records read per second, units still being scanned included
    assert info["scanning"] and info["rate_unit"] == "records scanned/s" and info["rate"] == 2_400_000 / 600
    groups = dict(meters(doc, info))
    tiles = {label: (value, note) for label, value, note in groups["Progress"]}
    assert tiles["Scan rate"] == ("4.0k/s", "records") and tiles["Remaining"][1] == "units only"
    assert groups["In flight: 2 work units"] == [("Waiting for a worker", "0", ""), ("Scanning", "2", ""),
                                                 ("Waiting for the database", "0", "")]
    assert groups["Records"][:2] == [("Scanned", "2,400,000", ""), ("Kept", "500,000", "25.0%")]
    # a run recorded before workers reported keeps its old figures
    old = {k: v for k, v in doc.items() if k not in ("in_flight", "steps", "phase_times", "checks")}
    info = summarize(old, now)
    assert not info["scanning"] and info["rate_unit"] == "entries/s" and "Checks" not in dict(meters(old, info))


def test_filter_checks_cover_the_destination_and_the_source():
    from lexichron.ui.progress_view import meters, notice_text

    def checks(doc, now):
        return {label: (value, note) for label, value, note in dict(meters(doc, summarize(doc, now)))["Checks"]}

    doc, now = _flt()
    assert checks(doc, now) == {"Counts": ("pending", ""), "Source": ("pending", ""),
                                "Kept + dropped": ("pending", "70.0% dropped"),      # the share is shown as the run goes
                                "Occurrences": ("9,000", ""), "Reordered values": ("0", "")}
    doc["checks"]["source_known"] = False                       # a source made before counts were verified
    assert checks(doc, now)["Source"] == ("total not known", "")
    doc["checks"]["source_known"] = True
    doc["phase"] = "verifying"
    assert checks(doc, now)["Counts"] == ("checking", "")

    good = {"status": "verified", "ledger": {"shards": 80, "match_total": 36_000, "volume_total": 3_000},
            "database": {"records": 10, "match_total": 36_000, "volume_total": 3_000, "disordered": 0, "malformed": 0},
            "source": {"status": "verified", "read": {"match_total": 120_000, "volume_total": 8_000},
                       "expected": {"match_total": 120_000, "volume_total": 8_000}, "disordered": 0},
            "balance": {"status": "verified", "read": {"match_total": 120_000, "volume_total": 8_000},
                        "emitted": {"match_total": 36_000, "volume_total": 3_000},
                        "dropped": {"match_total": 84_000, "volume_total": 5_000},
                        "difference": {"match_total": 0, "volume_total": 0}}}
    good["ledger"].update(source_match=120_000, dropped_match=84_000)
    doc.update(state="done", verification=good)
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("verified", "") and tiles["Source"] == ("read in full", "")
    assert tiles["Kept + dropped"] == ("adds up", "70.0% dropped")
    # a count lost inside the filter: the other two checks pass, this one does not
    lost = dict(good, status="mismatch", balance=dict(good["balance"], status="mismatch",
                                                      difference={"match_total": -500, "volume_total": -20}))
    doc.update(state="failed", verification=lost)
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("verified", "") and tiles["Source"] == ("read in full", "")
    assert tiles["Kept + dropped"] == ("DOES NOT ADD UP", "occurrences -500; volume counts -20")
    (w,) = summarize(doc, now)["warnings"]
    assert w.startswith("What the filter kept and what it dropped do not add up to what it read: occurrences differ by -500")
    doc.update(state="done", verification=good)
    assert tiles["Occurrences"] == ("36,000", "") and summarize(doc, now)["warnings"] == []

    # the destination holds more than the units emitted: a unit taken in twice
    bad = dict(good, status="mismatch", database=dict(good["database"], match_total=36_450, volume_total=3_015))
    doc.update(state="failed", verification=bad)
    assert checks(doc, now)["Counts"] == ("DO NOT MATCH", "occurrences +450; volume counts +15")
    (w,) = summarize(doc, now)["warnings"]
    assert w.startswith("The counts in the database do not match what the work units emitted: occurrences differ by +450")

    # the units read less than the source holds: the partition skipped keys
    gap = dict(good, status="mismatch", source=dict(good["source"], status="mismatch",
                                                    read={"match_total": 119_000, "volume_total": 7_900}))
    doc["verification"] = gap
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("verified", "") and tiles["Source"] == ("NOT READ ONCE", "occurrences -1,000; volume counts -100")
    (w,) = summarize(doc, now)["warnings"]
    assert "The source was not read exactly once" in w and "-1,000" in w

    doc["verification"] = dict(good, status="unverifiable", ledger=dict(good["ledger"], unledgered=3),
                               source={"status": "unknown", "read": {}, "expected": None, "disordered": 0})
    tiles = checks(doc, now)
    assert tiles["Counts"] == ("not verifiable", "") and tiles["Source"] == ("total not known", "")
    assert tiles["Units without totals"] == ("3", "pre-ledger")
    doc.update(verification={"status": "skipped"}); doc["checks"]["verify_counts"] = False
    assert checks(doc, now)["Counts"] == ("not checked", "verify_counts off")

    # failed work units are named, with what follows
    doc, now = _flt(files_failed=2, failed_files=[{"file": "unit_a", "message": "x"}, {"file": "unit_b", "message": "y"}])
    (w,) = summarize(doc, now)["warnings"]
    assert w.startswith("2 work units failed (unit_a, unit_b). The run will finish the other work units and then stop")
    assert notice_text(summarize(doc, now)) == w


def test_the_bar_is_not_full_while_a_later_step_is_still_running():
    """Once every file is in, the bar by size is at 100%, but compaction and
    the count check are still to run: a full bar there reads as a finished
    run. It shows activity without an amount until the run is done."""
    from lexichron.ui.progress_view import bar_state
    doc, now = _acq(phase="ingesting", files_done=5, bytes_done=500, in_flight={"lines": 0, "bytes": 100})
    assert bar_state(doc, summarize(doc, now)) == (1000, 600)
    for phase in ("compacting", "verifying"):
        doc, now = _acq(phase=phase, files_done=10, bytes_done=1000, in_flight={"lines": 0, "bytes": 0})
        doc["phase_started"] = doc["started"]
        assert bar_state(doc, summarize(doc, now)) == (None, 0)
    doc, now = _acq(phase="verifying", state="done", files_done=10, bytes_done=1000,
                    in_flight={"lines": 0, "bytes": 0}, finished=now.isoformat(timespec="seconds"))
    assert bar_state(doc, summarize(doc, now)) == (1000, 1000)
    # a run with no declared steps, counted by files
    doc, now = _doc()
    assert bar_state(doc, summarize(doc, now)) == (14, 4)


def test_a_measure_run_says_which_year_it_is_measuring_and_how_many_are_left():
    """Measure takes one year at a time and writes its record at once, so the
    queue tiles other stages show (waiting, done but unwritten, writing)
    could only say 0. It shows the year in progress and the years left."""
    from lexichron.ui.progress_view import meters
    doc, now = _doc(stage="measure", unit_label="years", files_total=52, files_done=20, current=["y1988"],
                    phases={"queued": 0, "parsing": 1, "parsed": 0, "ingesting": 0})
    groups = dict(meters(doc, summarize(doc, now)))
    assert groups["In flight"] == [("Measuring", "1988", "year"), ("Years left", "32 of 52", "this run")]
    doc["current"] = []
    assert dict(meters(doc, summarize(doc, now)))["In flight"][0] == ("Measuring", "none", "year")

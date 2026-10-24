"""Headless tests of the Textual app (only where textual is installed)."""
import asyncio

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import Collapsible, Select  # noqa: E402

from lexichron.ui.app import FixedValue, LexichronApp  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep the 'last settings file' memory out of the real home directory,
    and the app off this machine's Slurm (a login node has sbatch on its
    PATH, and the tests' made-up job ids are not in its queue)."""
    monkeypatch.setenv("LEXICHRON_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("lexichron.ui.app.slurm_backend", lambda: None)


def _assert_clean(app):
    # run_test re-raises panics, but be explicit: the app must not have errored.
    assert getattr(app, "_exception", None) is None, app._exception
    assert app.return_code in (None, 0), app.return_code


def test_app_builds_form_and_preview(tmp_path):
    proj = tmp_path / "project.yaml"
    proj.write_text(
        "corpus:\n  release: '20200217'\n  language: eng-us\n  ngram_size: 2\n  db_path_stub: /data/\n"
        "acquire:\n  ngram_type: tagged\n  workers: 4\n  file_range: [3, 7]\n"
    )

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = str(app.query_one("#call").content)
            assert "download_and_ingest_to_rocksdb(" in text
            assert "ngram_size=2" in text and "repo_corpus_id='eng-us'" in text
            assert "file_range=(3, 7)" in text
            assert str(app.query_one("#status").content) == ""
            # the file range is two boxes
            assert app.query_one("#f-acquire-file_range").value == "3"
            assert app.query_one("#f-acquire-file_range-end").value == "7"
            # the only release is fixed text, not a menu
            assert isinstance(app.query_one("#f-corpus-release"), FixedValue)
            # rarely-used settings and the call are collapsed by default
            assert app.query_one("#advanced", Collapsible).collapsed
            assert app.query_one("#call-box", Collapsible).collapsed
            assert app.query_one("#f-acquire-spool_dir") is not None
            # labels are plain language, not parameter names
            labels = [str(l.content) for l in app.query(".row Label")]
            assert "Corpus directory *" in labels and "Token type" in labels
            assert not any(l.startswith("db_path_stub") for l in labels)
            # help is hidden until F1
            assert not app.screen.has_class("show-help")
            await pilot.press("f1")
            assert app.screen.has_class("show-help")
            # change a field; preview follows
            app.query_one("#f-acquire-workers").value = "8"
            await pilot.pause()
            assert "workers=8" in str(app.query_one("#call").content)
            # save writes the file
            app.action_save()
            await pilot.pause()
        _assert_clean(app)
        import yaml
        saved = yaml.safe_load(proj.read_text())
        assert saved["acquire"]["workers"] == 8 and saved["corpus"]["language"] == "eng-us"
        assert saved["acquire"]["file_range"] == [3, 7]
        assert saved["corpus"]["release"] == "20200217"

    asyncio.run(scenario())


def test_app_without_project_derives_path_from_db_stub(tmp_path):
    async def scenario():
        app = LexichronApp(None)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "(fill in the corpus directory, language and n-gram size)" in str(app.query_one("#dest").content)
            assert app.query_one("#f-corpus-language", Select).value is Select.NULL
            app.query_one("#f-corpus-db_path_stub").value = str(tmp_path / "corpora")
            app.query_one("#f-corpus-language").value = "eng"
            app.query_one("#f-corpus-ngram_size").value = "1"
            await pilot.pause()
            # the settings file goes beside the database it describes
            assert app.project_path == tmp_path / "corpora" / "20200217" / "eng" / "1gram_files" / "lexichron.yaml"
            assert "repo_corpus_id='eng'" in str(app.query_one("#call").content)
            # a half-filled range is reported in plain words, no call shown
            app.query_one("#f-acquire-file_range").value = "0"
            await pilot.pause()
            assert "first and the last file" in str(app.query_one("#status").content)
            app.query_one("#f-acquire-file_range-end").value = "0"
            await pilot.pause()
            assert "file_range=(0, 0)" in str(app.query_one("#call").content)
            app.action_save()
            await pilot.pause()
        _assert_clean(app)
        assert (tmp_path / "corpora" / "20200217" / "eng" / "1gram_files" / "lexichron.yaml").exists()

    asyncio.run(scenario())


def test_missing_required_reported_with_labels(tmp_path):
    async def scenario():
        app = LexichronApp(None)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(app.query_one("#status").content)
            assert "Corpus directory" in status and "db_path_stub" not in status
        _assert_clean(app)

    asyncio.run(scenario())


def test_progress_tab_shows_selected_run(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  db_path_stub: {tmp_path}\n"
                    "acquire:\n  ngram_size: 1\n")
    now = datetime.now(timezone.utc)
    log = tmp_path / "run.log"
    log.write_text("first\nsecond\nthird\n")
    run = tmp_path / ".lexichron" / "runs" / "20261009_120000_acquire"
    run.mkdir(parents=True)
    def write_running(current, phases):
        run.joinpath("progress.json").write_text(json.dumps(dict(
            stage="acquire", state="running", started=(now - timedelta(minutes=5)).isoformat(timespec="seconds"),
            updated=now.isoformat(timespec="seconds"), finished=None, files_total=10, files_done=2,
            files_failed=0, files_skipped=0, entries_written=1000, uncompressed_bytes=10, chunks=2,
            current=current, phases=phases, message="", db_path="/db", log_path=str(log), slurm_job_id="42",
            hostname="n1")))
    write_running(["f.gz"], {"queued": 0, "parsing": 1, "parsed": 0, "ingesting": 0})
    older = tmp_path / ".lexichron" / "runs" / "20261009_110000_acquire"
    older.mkdir()
    older.joinpath("progress.json").write_text(json.dumps(dict(
        stage="acquire", state="done", started=(now - timedelta(hours=2)).isoformat(timespec="seconds"),
        updated=(now - timedelta(hours=1)).isoformat(timespec="seconds"),
        finished=(now - timedelta(hours=1)).isoformat(timespec="seconds"), files_total=1, files_done=1,
        files_failed=0, files_skipped=0, entries_written=5, uncompressed_bytes=1, chunks=1,
        current=[], message="", db_path=None, log_path=None, slurm_job_id=None, hostname="n1",
        database={"records": 3, "bytes": 100, "compacted": True})))

    async def scenario():
        from textual.widgets import DataTable, ProgressBar
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            runs = app.query_one("#runs", DataTable)
            assert runs.row_count == 2 and app.selected_run == run       # newest first, selected
            # Records column: kept so far for a running run, the compacted database for a finished one
            assert str(runs.get_row_at(0)[3]) == "1,000" and str(runs.get_row_at(1)[3]) == "3*"
            assert "running job 42 on n1 · 2/10 files (20%)" in str(app.query_one("#run-headline").content)
            bar = app.query_one("#run-bar", ProgressBar)
            assert bar.total == 10 and bar.progress == 2
            from lexichron.ui.app import Meters
            assert "1 files in flight" not in str(app.query_one("#run-headline").content)
            assert app.query_one("#run-log-box").collapsed            # log tail is folded away by default
            assert app._log_shown[1] == ("first", "second", "third")
            meters = app.query_one("#meters", Meters)
            tiles_before = list(meters.query(".meter"))
            assert str(meters._titles[1].content) == "In flight: 1 files"
            # the log pane follows the file; a changed in-flight count updates
            # the meters in place rather than rebuilding (and blanking) them
            log.write_text("first\nsecond\nthird\nfourth\n")
            write_running(["f.gz", "g.gz"], {"queued": 1, "parsing": 1, "parsed": 0, "ingesting": 0})
            app.refresh_jobs()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._log_shown[1][-1] == "fourth"
            assert str(meters._titles[1].content) == "In flight: 2 files"
            assert list(meters.query(".meter")) == tiles_before
            # selecting the older run switches the panel
            runs.move_cursor(row=1)
            for _ in range(20):                 # the highlight message and a periodic refresh may interleave
                await pilot.pause()
                if app.selected_run == older:
                    break
            assert app.selected_run == older
            assert str(app.query_one("#run-headline").content).startswith("done on n1 · 1/1 files (100%)")
            # clearing removes finished runs and keeps the running one
            app._clear_runs_pressed()           # the button's handler (the tab is not on screen)
            await pilot.pause()
            assert not older.exists() and run.exists() and runs.row_count == 1
        _assert_clean(app)

    asyncio.run(scenario())


def test_reopening_without_argument_restores_last_settings(tmp_path):
    proj = tmp_path / "corp" / "20200217" / "eng" / "1gram_files" / "lexichron.yaml"

    async def first():
        app = LexichronApp(None)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.query_one("#f-corpus-db_path_stub").value = str(tmp_path / "corp")
            app.query_one("#f-corpus-language").value = "eng"
            app.query_one("#f-corpus-ngram_size").value = "1"
            await pilot.pause()
            app.action_save()
            await pilot.pause()
        _assert_clean(app)

    async def second():
        app = LexichronApp(None)          # no argument, like plain `lexichron-ui`
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.reopened and app.project_path == proj
            assert app.query_one("#f-corpus-language").value == "eng"
            assert "(reopened from last time)" in str(app.query_one("#dest").content)
            # changing which corpus it describes moves the settings file with it
            app.query_one("#f-corpus-ngram_size").value = "2"
            await pilot.pause()
            assert app.project_path == tmp_path / "corp" / "20200217" / "eng" / "2gram_files" / "lexichron.yaml"
        _assert_clean(app)

    asyncio.run(first())
    assert proj.exists()
    asyncio.run(second())


def test_run_whose_job_left_the_queue_is_stopped(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  db_path_stub: {tmp_path}\n"
                    "acquire:\n  ngram_size: 1\n")
    now = datetime.now(timezone.utc)

    def run(name, job, minutes_ago):
        d = tmp_path / ".lexichron" / "runs" / name
        d.mkdir(parents=True)
        d.joinpath("progress.json").write_text(json.dumps(dict(
            stage="acquire", state="running",
            started=(now - timedelta(minutes=minutes_ago + 5)).isoformat(timespec="seconds"),
            updated=(now - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds"), finished=None,
            files_total=10, files_done=3, files_failed=0, files_skipped=0, entries_written=100,
            uncompressed_bytes=1, chunks=1, current=[], message="", db_path=None, log_path=None,
            slurm_job_id=job, hostname="n1")))

    run("20261010_090000_acquire", "200", 0)      # alive, in the queue
    run("20261010_010000_acquire", "100", 60)     # job gone, stale for an hour

    class FakeBridge:
        def available(self):
            return True

        def squeue(self, user=None):
            return [dict(job_id="999", name="other", state="RUNNING", elapsed="1:00", limit="2:00",
                         nodes="1", reason="n5"),
                    dict(job_id="200", name="lexichron", state="RUNNING", elapsed="0:05", limit="1:00",
                         nodes="1", reason="n1")]

    async def scenario():
        from textual.widgets import DataTable
        app = LexichronApp(proj)
        app.bridge = FakeBridge()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            runs = app.query_one("#runs", DataTable)
            states = [str(runs.get_row_at(i)[1]) for i in range(runs.row_count)]
            assert states == ["running", "stopped"]
            # lexichron's own job is listed first
            jobs = app.query_one("#jobs", DataTable)
            assert str(jobs.get_row_at(0)[0]) == "200"
        _assert_clean(app)

    asyncio.run(scenario())


def test_stage_selector_switches_form_and_keeps_other_sections(tmp_path):
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n"
                    "acquire:\n  workers: 4\nfilter:\n  min_len: 4\n")

    async def scenario():
        from textual.widgets import Select
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.stage == "acquire"
            assert app.query_one("#stage-acquire").display and not app.query_one("#stage-filter").display
            assert "download_and_ingest_to_rocksdb(" in str(app.query_one("#call").content)
            from textual.widgets import Button
            assert app.query_one("#stage-btn-acquire", Button).variant == "primary"
            await pilot.click("#stage-btn-filter")
            await pilot.pause()
            assert app.stage == "filter"
            assert app.query_one("#stage-btn-filter", Button).variant == "primary"
            assert app.query_one("#stage-btn-acquire", Button).variant == "default"
            assert app.query_one("#stage-filter").display and not app.query_one("#stage-acquire").display
            call = str(app.query_one("#call").content)
            assert "filter_ngrams(" in call and "min_len=4" in call and "ngram_size=1" in call
            assert app.project_path == proj          # explicitly opened file stays the target
            app.action_save()
            await pilot.pause()
        _assert_clean(app)
        import yaml
        saved = yaml.safe_load(proj.read_text())
        # both stages' settings survive a save made while the other stage is shown
        assert saved["acquire"]["workers"] == 4 and saved["filter"]["min_len"] == 4
        assert saved["corpus"]["ngram_size"] == 1

    asyncio.run(scenario())


def test_stale_file_handle_does_not_crash_refresh(tmp_path, monkeypatch):
    """A networked filesystem can answer stat() with ESTALE while a run
    rewrites its progress file; the Progress tab must shrug it off."""
    import errno
    import json
    import pathlib
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n")
    run = tmp_path / ".lexichron" / "runs" / "20261010_144558_acquire"
    run.mkdir(parents=True)
    run.joinpath("progress.json").write_text(json.dumps({"state": "running", "started": "2026-10-10T14:45:58+00:00",
                                                         "updated": "2026-10-10T14:46:58+00:00"}))
    real_stat = pathlib.Path.stat

    def flaky_stat(self, *a, **kw):
        if self.name == "progress.json":
            raise OSError(errno.ESTALE, "Stale file handle", str(self))
        return real_stat(self, *a, **kw)

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            monkeypatch.setattr(pathlib.Path, "stat", flaky_stat)
            app.refresh_jobs()          # must not raise
            await app.workers.wait_for_complete()
            await pilot.pause()
            monkeypatch.setattr(pathlib.Path, "stat", real_stat)
            app.refresh_jobs()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.selected_run == run
        _assert_clean(app)

    asyncio.run(scenario())


def test_inspect_tab_lists_databases_and_shows_records(tmp_path, monkeypatch):
    import struct
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    import fake_rocks
    from ngramprep.common_db import api as dbapi
    from textual.widgets import DataTable, Input, Static
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    corpus = tmp_path / "corp" / "20200217" / "eng" / "5gram_files"
    corpus.mkdir(parents=True)
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "filter: {}\n")
    db = fake_rocks.open(str(corpus / "5grams_pivoted.db"), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for year, ng, count in ((1990, b"cat run", 5), (2011, b"nuclear <UNK>", 9), (2011, b"nuclear power", 4)):
            wb.put(struct.pack(">I", year) + ng, struct.pack("<QQQ", year, count, count))
    (corpus / "5grams_pivoted.db" / "CURRENT").write_text("MANIFEST-000001\n")

    async def scenario():
        app = LexichronApp(str(proj), "filter")
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            sel = app.query_one("#inspect-db", Select)
            assert [v for _, v in sel._options if v is not Select.NULL] == [str(corpus / "5grams_pivoted.db")]
            sel.value = str(corpus / "5grams_pivoted.db")            # choosing fills the path and describes
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#inspect-path", Input).value == str(corpus / "5grams_pivoted.db")
            assert "3 records" in str(app.query_one("#inspect-summary", Static).content)
            assert "keys look pivoted" in str(app.query_one("#inspect-summary", Static).content)
            app.query_one("#inspect-key", Input).value = "[2011] nuclear"
            app._inspect("prefix")
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.query_one("#inspect-table", DataTable)
            assert table.row_count == 2 and str(table.get_row_at(0)[0]) == "[2011] nuclear <UNK>"
            assert "2 record(s) with prefix '[2011] nuclear'" in str(app.query_one("#inspect-note", Static).content)
            # a key without the year is refused with the reason, not a crash
            app.query_one("#inspect-key", Input).value = "nuclear"
            app._inspect("key")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "start with the year" in str(app.query_one("#inspect-note", Static).content)
        _assert_clean(app)

    asyncio.run(scenario())


def test_inspect_tab_shows_provenance_and_year_series(tmp_path, monkeypatch):
    import struct
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    import fake_rocks
    from ngramprep.common_db import api as dbapi
    from lexichron import provenance
    from textual.widgets import DataTable, Input, Static
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    corpus = tmp_path / "corp" / "20200217" / "eng" / "1gram_files"
    corpus.mkdir(parents=True)
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "filter: {}\n")
    db_dir = corpus / "1grams_processed.db"
    db = fake_rocks.open(str(db_dir), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"apple", struct.pack("<QQQ", 1990, 5, 2) + struct.pack("<QQQ", 2000, 7, 3))
    db.compact_all()
    (db_dir / "CURRENT").write_text("MANIFEST-000001\n")
    doc = {"stage": "filter", "state": "done", "db_path": str(db_dir), "files_total": 8, "hostname": "n1",
           "started": "2026-10-14T00:00:00+00:00", "finished": "2026-10-14T00:05:00+00:00",
           "counters": {"scanned": 8, "kept": 5, "produced": 5, "inserted": 5, "tokens": 8},
           "database": {"records": 1, "bytes": 100, "compacted": True},
           "inputs": {"source": str(corpus / "1grams.db"), "vocabulary": None}}
    provenance.write_provenance(db_dir, provenance.build_record("filter", {"ngram_size": 1, "mode": "restart"}, doc))

    async def scenario():
        app = LexichronApp(str(proj), "filter")
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one("#inspect-path", Input).value = str(db_dir)
            app._inspect("head")
            await app.workers.wait_for_complete()
            await pilot.pause()
            box = app.query_one("#inspect-provenance-box", Collapsible)
            assert box.display
            text = str(app.query_one("#inspect-provenance", Static).content)
            assert "Made by lexichron" in text and "stage filter" in text and "8 scanned, 5 kept" in text
            table = app.query_one("#inspect-table", DataTable)
            assert table.row_count == 1
            # selecting the row shows its year series
            table.move_cursor(row=0)
            app._inspect_row_selected(DataTable.RowSelected(table, 0, table.coordinate_to_cell_key((0, 0)).row_key))
            await pilot.pause()
            detail = str(app.query_one("#inspect-detail", Static).content)
            assert "2 years" in detail and "1990: 5/2" in detail and "2000: 7/3" in detail
        _assert_clean(app)

    asyncio.run(scenario())


def test_tune_form_toggles_build_the_grid(tmp_path):
    from textual.widgets import Static
    from lexichron.ui.app import ToggleRow
    corpus = tmp_path / "corp" / "20200217" / "eng" / "5gram_files"
    corpus.mkdir(parents=True)
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "tune:\n  vector_size: [100, 300]\n  name: t\nslurm:\n  cpus: 40\n")

    async def scenario():
        app = LexichronApp(str(proj), "tune")
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause()
            row = app.query_one("#f-tune-vector_size", ToggleRow)
            assert row.value == [100, 300]
            assert app.config["tune"]["vector_size"] == [100, 300]
            summary = str(app.query_one("#grid-summary", Static).content)
            assert "2 vector sizes" in summary and "= 2 cells" in summary and "64 models" in summary
            assert "10 at a time x 4 threads on 40 CPUs" in summary
            # toggle 300 off, 50 on; add a custom value
            await pilot.click("#f-tune-vector_size Button.toggle", offset=(0, 0))      # the first button: 50
            await pilot.pause()
            assert row.value == [50, 100, 300]
            for b in row.query("Button.toggle"):
                if b.name == "300":
                    b.press()
            await pilot.pause()
            assert row.value == [50, 100] and app.config["tune"]["vector_size"] == [50, 100]
            adder = row.query_one("Input.toggle-add")
            adder.value = "150"
            await pilot.pause()
            await adder.action_submit()
            await pilot.pause()
            assert row.value == [50, 100, 150] and "3 vector sizes" in str(app.query_one("#grid-summary", Static).content)
            # the last lit button stays lit
            win = app.query_one("#f-tune-window", ToggleRow)
            assert win.value == [2]
            for b in win.query("Button.toggle"):
                if b.name == "2":
                    b.press()
            await pilot.pause()
            assert win.value == [2]
            # the resolved call carries the lists
            assert "vector_size=[50, 100, 150]" in str(app.query_one("#call").content)
        _assert_clean(app)

    asyncio.run(scenario())


def test_tune_inspect_shows_the_search_scores(tmp_path):
    import json
    from textual.widgets import DataTable, Static
    corpus = tmp_path / "corp" / "20200217" / "eng" / "5gram_files"
    search = corpus / "tuning" / "t1"
    search.mkdir(parents=True)
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "tune:\n  name: t1\n")
    (search / "search.json").write_text(json.dumps({"grid": {"vector_size": [100, 300], "window": [2]},
                                                     "sample_years": [1990, 1991, 2010, 2011], "seeds": 2,
                                                     "split_half": True}))
    cells = [{"cell": "vs100_w2_mc5_e5_sg1_wbfreq", "vector_size": 100, "window": 2, "min_count": 5, "epochs": 5,
              "approach": "skip-gram", "weight_by": "freq", "years": 2, "vocab_size": 25000, "coverage": 0.91,
              "simlex_rho": 0.31, "wordsim_rho": 0.55, "analogy_accuracy": 0.2, "seed_overlap": 0.72,
              "split_half_overlap": 0.6, "adjacent_overlap": 0.65, "distant_overlap": None, "signal_to_noise": None},
             {"cell": "vs300_w2_mc5_e5_sg1_wbfreq", "vector_size": 300, "window": 2, "min_count": 5, "epochs": 5,
              "approach": "skip-gram", "weight_by": "freq", "years": 2, "vocab_size": 25000, "coverage": 0.91,
              "simlex_rho": 0.35, "wordsim_rho": 0.58, "analogy_accuracy": 0.3, "seed_overlap": 0.81,
              "split_half_overlap": 0.7, "adjacent_overlap": 0.75, "seed_overlap_sd": 0.02, "change_ratio": None}]
    (search / "summary.json").write_text(json.dumps({"cells": cells, "best_cell": "vs300_w2_mc5_e5_sg1_wbfreq",
                                                      "best_rule": "most stable among cells near the best"}))
    (search / "results.tsv").write_text("cell\tyear\tvocab_size\tcoverage\tseed_overlap\n"
                                        "vs100_w2_mc5_e5_sg1_wbfreq\t1990\t24000\t0.9000\t0.7000\n"
                                        "vs100_w2_mc5_e5_sg1_wbfreq\t1991\t26000\t0.9200\t0.7400\n"
                                        "vs300_w2_mc5_e5_sg1_wbfreq\t1990\t24000\t0.9000\t0.8000\n"
                                        "vs300_w2_mc5_e5_sg1_wbfreq\t1991\t26000\t0.9200\t0.8200\n")
    (search / "pairs.tsv").write_text("cell\tyear_a\tyear_b\tkind\toverlap\tcosine\n"
                                      "vs300_w2_mc5_e5_sg1_wbfreq\t1990\t1991\tadjacent\t0.7500\t0.9000\n")

    async def scenario():
        app = LexichronApp(str(proj), "tune")
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#inspect-search").display and not app.query_one("#inspect-generic").display
            tiles = [str(t.content) for t in app.query("#search-meters .meter")]
            assert any("Cells" in t and "2" in t for t in tiles) and any("2/4" in t and "complete for this grid" in t for t in tiles)
            assert any("vs300_w2_mc5_e5_sg1_wbfreq" in t for t in tiles)
            grid = str(app.query_one("#search-grid", Static).content)
            assert grid.startswith("Grid: vector size 100/300") and "1990, 1991 scored so far" in grid
            table = app.query_one("#search-table", DataTable)
            assert table.row_count == 2
            headings = [str(c.label) for c in table.columns.values()]
            assert headings[:3] == ["", "Dims", "Years"]                     # window is fixed: not a column
            rows = [[str(c).strip() for c in table.get_row_at(i)] for i in range(2)]    # padded to the column
            assert rows[0][:2] == ["", "100"] and rows[1][:2] == ["★", "300"]
            assert "25,000" in rows[1] and "0.810 ±.020" in rows[1]
            seed = headings.index("Seed")
            assert table.get_row_at(1)[seed].style == "bold" and table.get_row_at(0)[seed].style == ""   # column best
            assert str(table.get_row_at(1)[seed]).strip() == "0.810 ±.020"                              # with its spread
            # the suggested cell's years are shown to begin with
            assert str(app.query_one("#search-cell", Static).content) == "vs300_w2_mc5_e5_sg1_wbfreq by year"
            assert app.query_one("#search-years", DataTable).row_count == 2 and table.cursor_row == 1
            # sorting by a score column puts the best first; selecting a row shows its years
            app._sort_search_table("Seed", reverse=True)           # the heading-click sort; the ± text sorts by its mean
            await pilot.pause()
            assert str(table.get_row_at(0)[0]).strip() == "★"
            app._search_row_selected(DataTable.RowSelected(table, 0, table.coordinate_to_cell_key((0, 0)).row_key))
            await pilot.pause()
            assert str(app.query_one("#search-cell", Static).content) == "vs300_w2_mc5_e5_sg1_wbfreq by year"
            years = app.query_one("#search-years", DataTable)
            assert years.row_count == 2 and [str(c).strip() for c in years.get_row_at(1)] == ["1991", "26,000", "0.920", "0.820"]
            assert "adjacent 1990–1991 0.750 / 0.900" in str(app.query_one("#search-pairs", Static).content)
            # a narrower view, and the spreads switched off; the tile follows the family
            app.query_one("#search-view", Select).value = "neighbourhoods"
            await pilot.pause()
            headings = [str(c.label) for c in table.columns.values()]
            assert "SimLex ρ" not in headings and "Seed" in headings
            from lexichron.ui.app import Meters
            meters = app.query_one("#search-meters", Meters)
            tile = next(str(t.content) for t in meters.query(".meter") if "Suggested cell" in str(t.content))
            assert "by the neighbourhoods rule" in tile and "benchmarks" not in tile      # that family's pick alone
            app.query_one("#search-spreads").value = False
            await pilot.pause()
            seed = [str(c.label) for c in table.columns.values()].index("Seed")
            plain = str(table.get_row_at(0)[seed]).strip()
            assert "±" not in plain and float(plain) > 0                           # no spread any more
            app.query_one("#search-view", Select).value = "all"
            app.query_one("#search-spreads").value = True
            await pilot.pause()
            # the other stages keep the record inspector
            app.select_stage("filter")
            await pilot.pause()
            assert app.query_one("#inspect-generic").display and not app.query_one("#inspect-search").display
        _assert_clean(app)

    asyncio.run(scenario())


def test_run_asks_before_deleting_earlier_output(tmp_path, monkeypatch):
    """Run in restart mode with a database on disk opens a dialog; Keep it
    runs nothing, confirming runs the stage with --yes."""
    from pathlib import Path
    from textual.widgets import Button, Static
    from ngramprep.ngram_acquire.db.build_path import build_db_path
    from lexichron.ui.app import ConfirmOverwrite
    stub = tmp_path / "data"
    db = Path(build_db_path(str(stub), 5, "20200217", "eng-us")).parent / "5grams_pivoted.db"
    db.mkdir(parents=True)
    (db / "000001.sst").write_bytes(b"x" * 4000); (db / "CURRENT").write_text("MANIFEST\n")
    proj = tmp_path / "project.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng-us\n  ngram_size: 5\n  db_path_stub: {stub}\n"
                    "filter:\n  mode: restart\n")
    started = []

    async def scenario():
        app = LexichronApp(proj, stage="filter")
        monkeypatch.setattr(app, "_run_stage", lambda log, project, assume_yes=False: started.append(assume_yes))
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app._run_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, ConfirmOverwrite)
            body = str(dialog.query_one("#confirm-body", Static).content)
            assert "filter.mode = restart" in body and "5grams_pivoted.db" in body and "4.0 KB in 2 files" in body
            assert str(dialog.query_one("#confirm-go", Button).label) == "Delete 4.0 KB and start over"
            await pilot.click("#confirm-keep")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmOverwrite) and started == []
            # confirming runs the stage, with --yes so the CLI does not ask again
            app._run_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#confirm-go")
            await pilot.pause()
            assert started == [True]
            # in resume mode there is no dialog and no --yes
            app.query_one("#f-filter-mode", Select).value = "resume"
            await pilot.pause()
            app._run_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmOverwrite) and started == [True, False]
        _assert_clean(app)

    asyncio.run(scenario())


def test_clear_runs_keeps_every_run_still_going(tmp_path):
    """A tune run spends hours in its 'training' phase, which the runs table
    labels by phase rather than 'running'; clearing finished runs must not
    take it with them. Nor a Slurm run that has not written for a while
    (the queue decides whether it is gone), while a local run that has
    gone quiet with no live process is dead and goes."""
    import json
    from datetime import datetime, timedelta, timezone
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  db_path_stub: {tmp_path}\n"
                    "acquire:\n  ngram_size: 1\n")
    now = datetime.now(timezone.utc)
    runs = tmp_path / ".lexichron" / "runs"

    def write(name, **fields):
        d = runs / name
        d.mkdir(parents=True)
        doc = dict(stage="tune", state="running", started=(now - timedelta(hours=2)).isoformat(timespec="seconds"),
                   updated=now.isoformat(timespec="seconds"), finished=None, files_total=229, files_done=21,
                   files_failed=0, files_skipped=59, entries_written=0, uncompressed_bytes=0, chunks=0, current=[],
                   phases={"queued": 0, "parsing": 28, "parsed": 0, "ingesting": 0}, message="", db_path="/db",
                   log_path=None, slurm_job_id=None, hostname="n1")
        doc.update(fields)
        d.joinpath("progress.json").write_text(json.dumps(doc))
        return d

    stale = (now - timedelta(minutes=20)).isoformat(timespec="seconds")
    training = write("20261020_010000_tune", phase="training", slurm_job_id="7")      # in a long phase
    quiet_job = write("20261020_020000_tune", updated=stale, slurm_job_id="8")        # Slurm run, no write for 20 min
    dead_local = write("20261020_030000_tune", updated=stale)                         # local run, gone quiet, no process
    finished = write("20261020_040000_tune", state="done", finished=now.isoformat(timespec="seconds"))

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app._clear_runs_pressed()
            await pilot.pause()
            assert training.exists() and quiet_job.exists()
            assert not dead_local.exists() and not finished.exists()
        _assert_clean(app)

    asyncio.run(scenario())


def test_submit_warns_when_memory_is_too_small_for_the_workers(tmp_path, monkeypatch):
    from textual.widgets import Button, Static
    from lexichron.ui import app as app_module
    from lexichron.ui.app import ConfirmMemory
    stub = tmp_path / "data"
    proj = tmp_path / "project.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng-us\n  ngram_size: 5\n  db_path_stub: {stub}\n"
                    "acquire: {}\nslurm:\n  account: acct\n  cpus: 64\n  mem: 16G\n")
    written = []
    monkeypatch.setattr(app_module, "write_sbatch", lambda *a, **k: written.append(k.get("assume_yes")) or (tmp_path / "x.sbatch"))

    async def scenario():
        app = LexichronApp(proj, stage="acquire")
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            line = str(app.query_one("#submit-memory", Static).content)
            assert line.startswith("Memory: about ") and "64 workers" in line and "Slurm would kill it" in line
            assert app.query_one("#submit-memory").has_class("over")
            # Write batch script: no data at stake, but the memory question comes up
            app._write_sbatch_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmMemory)
            assert str(app.screen.query_one("#confirm-go", Button).label) == "Submit anyway"
            await pilot.click("#confirm-keep")
            await pilot.pause()
            assert written == []
            app._write_sbatch_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#confirm-go")
            await pilot.pause()
            assert written == [False]
            # enough memory: the line is plain and nothing is asked
            app.query_one("#f-slurm-mem").value = "48G"
            await pilot.pause()
            assert not app.query_one("#submit-memory").has_class("over")
            app._write_sbatch_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmMemory) and written == [False, False]
        _assert_clean(app)

    asyncio.run(scenario())


def test_restart_submitted_as_a_job_array_deletes_before_the_tasks_start(tmp_path, monkeypatch):
    """A tune restart with Jobs above 1: the dialog asks as usual; confirming
    deletes the search here (the tasks cannot: each would delete the
    others' work) and the script sends the tasks in resume mode. The grid
    summary says how the years split."""
    from textual.widgets import Static
    from lexichron.ui.app import ConfirmOverwrite
    corpus = tmp_path / "corp" / "20200217" / "eng" / "5gram_files"
    search = corpus / "tuning" / "t"
    search.mkdir(parents=True)
    (search / "search.json").write_text("{}"); (search / "models").mkdir()
    (search / "models" / "a.npz").write_bytes(b"x" * 1000)
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "tune:\n  vector_size: [100, 300]\n  name: t\n  mode: restart\n"
                    "slurm:\n  account: acct\n  cpus: 40\n  mem: 64G\n  jobs: 4\n")

    async def scenario():
        app = LexichronApp(str(proj), "tune")
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            summary = str(app.query_one("#grid-summary", Static).content)
            assert "Split over 4 jobs" in summary and "every 4th year" in summary and "2 years and 16 models" in summary
            app._write_sbatch_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmOverwrite)
            await pilot.click("#confirm-keep")
            await pilot.pause()
            assert search.exists()
            app._write_sbatch_pressed()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#confirm-go")
            await pilot.pause()
            status = str(app.query_one("#submit-status", Static).content)
            assert not search.exists() and "Deleted" in status and "4 tasks resume" in status
            script = (corpus / "lexichron.tune.sbatch").read_text()
            assert "#SBATCH --array=0-3" in script and "--set tune.mode=resume" in script
            assert "--set tune.array_task=$SLURM_ARRAY_TASK_ID" in script
        _assert_clean(app)

    asyncio.run(scenario())


def test_train_form_summary_tuned_cell_and_inspect(tmp_path):
    """The train stage: its settings form with the ensemble summary, the
    button that copies a search's suggested cell into the six fields, and
    the Inspect tab's per-year table of an ensemble."""
    import json
    from textual.widgets import DataTable, Input, Static
    corpus = tmp_path / "corp" / "20200217" / "eng" / "5gram_files"
    search = corpus / "tuning" / "t1"
    search.mkdir(parents=True)
    (search / "search.json").write_text(json.dumps({"grid": {"vector_size": [100, 300]}, "sample_years": [1990, 1991],
                                                     "seeds": 2, "split_half": True}))
    cells = [{"cell": "vs300_w5_mc10_e20_sg0_wbdoc_freq", "vector_size": 300, "window": 5, "min_count": 10,
              "epochs": 20, "approach": "cbow", "weight_by": "doc_freq", "years": 2, "coverage": 0.9,
              "seed_overlap": 0.8}]
    (search / "summary.json").write_text(json.dumps({"cells": cells, "best_cell": cells[0]["cell"],
                                                      "best_rule": "the most stable"}))
    ens = corpus / "models" / "e1"
    (ens / "y1990").mkdir(parents=True)
    (ens / "y1991").mkdir()
    (ens / "train.json").write_text(json.dumps({"cell_stem": "vs100_w2_mc5_e10_sg1_wbnone", "years": [1990, 1991],
                                                 "years_with_data": [1990, 1991], "corpus_replicates": 2,
                                                 "seeds_per_corpus": 2, "resampling": "poisson", "noise_seed": 0}))
    for i in range(4):
        (ens / "y1990" / f"c{i // 2:02d}_s{i % 2}.npz").write_bytes(b"x")
    (ens / "y1990" / "year.json").write_text(json.dumps({
        "year": 1990, "models_on_disk": 4, "vocab_size": 12000, "train_seconds": 480, "axis_words": 3000,
        "axis_var_seed": 0.0004, "axis_var_corpus": 0.0012, "axis_var_replicate": None, "axis_se_mean": 0.026,
        "axis_var_words": 0.02, "axis_noise_share": 0.034,
        "neigh_seed_overlap": 0.61, "neigh_corpus_overlap": 0.44, "neigh_probes": 900,
        "simlex_rho": 0.35, "simlex_se_mean": 0.004, "wordsim_rho": 0.66, "wordsim_se_mean": 0.006}))
    (ens / "y1991" / "c00_s0.npz").write_bytes(b"x")
    proj = corpus / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 5\n  db_path_stub: {tmp_path / 'corp'}\n"
                    "train:\n  name: e1\n  years: [1990, 1991]\n  corpus_replicates: 2\n  seeds_per_corpus: 2\n"
                    "slurm:\n  cpus: 32\n  jobs: 2\n")

    async def scenario():
        app = LexichronApp(str(proj), "train")
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#inspect-ensemble").display and not app.query_one("#inspect-generic").display
            summary = str(app.query_one("#train-summary", Static).content)
            assert "2 years (1990-1991) x 2 corpus x 2 seed replicates = 4 models per year, 8 in all" in summary
            assert "8 at a time x 4 threads on 32 CPUs" in summary and "Split over 2 jobs" in summary
            # the tuned cell lands in the form
            assert app.query_one("#f-train-vector_size", Input).value == "100"
            app._tuned_cell_pressed()
            await pilot.pause()
            assert app.query_one("#f-train-vector_size", Input).value == "300"
            assert app.query_one("#f-train-window", Input).value == "5"
            assert app.query_one("#f-train-epochs", Input).value == "20"
            assert app.query_one("#f-train-approach").value == "cbow"
            assert app.query_one("#f-train-weight_by").value == "doc_freq"
            assert app.config["train"]["vector_size"] == 300 and app.config["train"]["weight_by"] == "doc_freq"
            # the Inspect tab shows the ensemble's years
            tiles = [str(t.content) for t in app.query("#ensemble-meters .meter")]
            assert any("1/2" in t and "complete" in t for t in tiles) and any("5/8" in t for t in tiles)
            table = app.query_one("#ensemble-table", DataTable)
            assert table.row_count == 2
            cols = [str(c.label) for c in table.columns.values()]
            assert cols[:3] == ["Year", "Models", "Vocab"] and "SE of mean" in cols and "Replicate var" not in cols
            first = [str(c) for c in table.get_row_at(0)]
            assert first[0].strip() == "1990" and first[1].strip() == "4/4" and "12,000" in first[2]
            second = [str(c) for c in table.get_row_at(1)]
            assert second[1].strip() == "1/4 (training)"
            # the profile of the years complete so far, in all three families
            tiles = [str(t.content) for t in app.query("#ensemble-meters .meter")]
            assert any("Axis SE of mean" in t and "2.60e-02" in t for t in tiles)
            assert any("Seed overlap" in t and "0.610" in t for t in tiles)
            assert any("SimLex ρ" in t and "0.350 ±0.004" in t for t in tiles)
            # the Columns selector narrows the table to one family
            app.query_one("#ensemble-view", Select).value = "neighbourhoods"
            await pilot.pause()
            cols = [str(c.label) for c in app.query_one("#ensemble-table", DataTable).columns.values()]
            assert cols == ["Year", "Models", "Seed overlap", "Corpus overlap", "Probes"]
            tiles = [str(t.content) for t in app.query("#ensemble-meters .meter")]
            assert any("Seed overlap" in t for t in tiles) and not any("SimLex" in t for t in tiles)
        _assert_clean(app)

    asyncio.run(scenario())


def test_slurm_backend_chosen_at_startup(tmp_path, monkeypatch):
    """Outside the container with sbatch on the PATH the UI runs Slurm commands
    itself; the 'no Slurm' note appears only when there is no bridge and no
    sbatch either."""
    import os
    from textual.widgets import DataTable, Static
    from lexichron import bridge
    from lexichron.bridge import LocalSlurm
    monkeypatch.setattr("lexichron.ui.app.slurm_backend", bridge.slurm_backend)   # the real chooser
    proj = tmp_path / "project.yaml"
    proj.write_text(
        "corpus:\n  release: '20200217'\n  language: eng-us\n  ngram_size: 2\n  db_path_stub: /data/\n"
        "acquire:\n  ngram_type: tagged\n  workers: 4\n"
    )
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "sbatch").write_text('#!/bin/sh\necho "4242;torch"\n')
    (fake / "squeue").write_text('#!/bin/sh\necho "4242|lexichron-acquire|RUNNING|00:01:00|12:00:00|1|cs604"\n')
    for f in fake.iterdir():
        f.chmod(0o755)
    monkeypatch.delenv("LEXICHRON_BRIDGE_DIR", raising=False)

    async def with_sbatch_on_path():
        app = LexichronApp(proj)
        assert isinstance(app.bridge, LocalSlurm)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "No Slurm here" not in str(app.query_one("#submit-status", Static).content)
            jobs = app.query_one("#jobs", DataTable)
            assert [str(jobs.get_row_at(i)[0]) for i in range(jobs.row_count)] == ["4242"]
        _assert_clean(app)

    async def with_nothing(expect: str):
        app = LexichronApp(proj)
        assert app.bridge is None
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            note = str(app.query_one("#submit-status", Static).content)
            assert "No Slurm here" in note and expect in note, note
        _assert_clean(app)

    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    asyncio.run(with_sbatch_on_path())
    empty = tmp_path / "empty"; empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)      # CI runs the suite inside the image
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    asyncio.run(with_nothing("sbatch is not on this machine's PATH"))
    # inside the image without the helper (the plain launcher, or a shell where
    # ~/.local/bin shadows the environment's command), the note says to use lexichron-ui
    monkeypatch.setenv("APPTAINER_CONTAINER", "/images/lexichron.sif")
    asyncio.run(with_nothing("inside the image without the host helper"))


def test_save_asks_before_replacing_a_file_this_session_did_not_open(tmp_path):
    """The save path follows the corpus fields. Pointing them at a corpus
    whose settings file already exists, in a session that never opened that
    file, must not overwrite it silently: the form was not filled from it,
    so saving would drop whatever it holds."""
    from textual.widgets import Button
    from lexichron.ui.app import ConfirmReplaceSettings
    opened = tmp_path / "a.yaml"
    opened.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n")
    other = tmp_path / "20200217" / "eng" / "2gram_files" / "lexichron.yaml"
    other.parent.mkdir(parents=True)
    original = (f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 2\n  db_path_stub: {tmp_path}\n"
                "filter:\n  min_len: 9\ntune:\n  name: precious\n")
    other.write_text(original)

    async def scenario():
        app = LexichronApp(opened)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.query_one("#f-corpus-ngram_size").value = "2"       # now describes the other corpus
            await pilot.pause()
            assert app.project_path == other
            app.action_save()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmReplaceSettings) and app.screen.path == other
            await pilot.click("#confirm-keep")
            await pilot.pause()
            assert other.read_text() == original                   # kept
            app.action_save()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmReplaceSettings)
            await pilot.click("#confirm-go")
            await pilot.pause()
            assert "written by lexichron ui" in other.read_text() and "precious" not in other.read_text()
            app.action_save()                                      # written by this session: no question now
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmReplaceSettings)
            # a path with no file there yet never asks
            app.query_one("#f-corpus-ngram_size").value = "3"
            await pilot.pause()
            fresh = tmp_path / "20200217" / "eng" / "3gram_files" / "lexichron.yaml"
            assert app.project_path == fresh and not fresh.exists()
            app.action_save()
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmReplaceSettings) and fresh.exists()
        _assert_clean(app)

    asyncio.run(scenario())


def test_submit_tab_says_what_a_job_would_run(tmp_path, monkeypatch):
    from textual.widgets import Static
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n")

    async def scenario(expect: str):
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert expect in str(app.query_one("#submit-command", Static).content)
        _assert_clean(app)

    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    asyncio.run(scenario("on the job's PATH"))                      # a development environment
    monkeypatch.setenv("APPTAINER_CONTAINER", "/images/lexichron-9.9.sif")
    asyncio.run(scenario("Jobs run in the image /images/lexichron-9.9.sif"))


def test_jobs_field_appears_only_for_stages_that_split_over_an_array(tmp_path):
    """Acquire and filter write one database and run as one job, so the Jobs
    row would be a no-op there; it is shown for tune and train."""
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n")

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            shown = {}
            for stage in ("acquire", "filter", "tune", "train"):
                app.select_stage(stage)
                await pilot.pause()
                shown[stage] = (app.query_one("#row-slurm-jobs").display, app.query_one("#help-slurm-jobs").display)
            assert shown == {"acquire": (False, False), "filter": (False, False),
                             "tune": (True, True), "train": (True, True)}
            assert app.query_one("#row-slurm-cpus").display                   # the other resources stay
        _assert_clean(app)

    asyncio.run(scenario())


def test_submit_tab_resolves_slurm_settings_per_stage_and_saves_them_in_layers(tmp_path):
    """Each stage shows its own resources; the first account saved becomes
    the user's default and stays out of the corpus file; the resources go
    into the stage's block; a value typed for one stage survives a visit to
    another."""
    import yaml
    from textual.widgets import Input, Static
    from lexichron.slurm_settings import read_user_defaults
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng\n  ngram_size: 1\n  db_path_stub: {tmp_path}\n"
                    "slurm:\n  mem: 30G\n")                                 # an older file: one block for every stage

    async def scenario():
        app = LexichronApp(str(proj))
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            cpus, mem, acct = (app.query_one(f"#f-slurm-{k}", Input) for k in ("cpus", "mem", "account"))
            assert cpus.value == "40" and mem.value == "30G" and acct.value == ""
            assert "no account" in str(app.query_one("#submit-sources", Static).content)
            app.select_stage("train")
            await pilot.pause()
            assert cpus.value == "64" and mem.value == "30G" and app.query_one("#f-slurm-jobs", Input).value == "13"
            line = str(app.query_one("#submit-sources", Static).content)
            assert "memory 30G (this corpus)" in line and "CPUs 64" in line and "lexichron's default for train" in line
            cpus.value = "96"
            acct.value = "proj_a"
            await pilot.pause()
            app.select_stage("acquire")
            await pilot.pause()
            assert cpus.value == "40"                                     # acquire's own figure
            app.select_stage("train")
            await pilot.pause()
            assert cpus.value == "96"                                     # the edit for train survived the trip
            app.action_save()
            await pilot.pause()
            saved = yaml.safe_load(proj.read_text())
            assert saved["slurm"]["train"] == {"cpus": 96, "mem": "30G", "time": "1-00:00:00", "jobs": 13}
            assert "account" not in saved["slurm"] and saved["slurm"]["mem"] == "30G"
            assert read_user_defaults() == {"account": "proj_a"}          # seeded from the first save
            assert "account proj_a (your defaults)" in str(app.query_one("#submit-sources", Static).content)
            # another account for this corpus alone goes into the file; the button makes it the default instead
            acct.value = "proj_b"
            await pilot.pause()
            app.action_save()
            await pilot.pause()
            assert yaml.safe_load(proj.read_text())["slurm"]["account"] == "proj_b"
            app._slurm_defaults_pressed()
            await pilot.pause()
            assert read_user_defaults() == {"account": "proj_b"}
            app.action_save()
            await pilot.pause()
            assert "account" not in yaml.safe_load(proj.read_text())["slurm"]
        _assert_clean(app)

    asyncio.run(scenario())

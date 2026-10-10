"""Headless tests of the Textual app (only where textual is installed)."""
import asyncio

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import Collapsible, Select  # noqa: E402

from lexichron.ui.app import FixedValue, LexichronApp  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Keep the 'last settings file' memory out of the real home directory."""
    monkeypatch.setenv("LEXICHRON_STATE_DIR", str(tmp_path / "state"))


def _assert_clean(app):
    # run_test re-raises panics, but be explicit: the app must not have errored.
    assert getattr(app, "_exception", None) is None, app._exception
    assert app.return_code in (None, 0), app.return_code


def test_app_builds_form_and_preview(tmp_path):
    proj = tmp_path / "project.yaml"
    proj.write_text(
        "corpus:\n  release: '20200217'\n  language: eng-us\n  db_path_stub: /data/\n"
        "acquire:\n  ngram_size: 2\n  ngram_type: tagged\n  workers: 4\n  file_range: [3, 7]\n"
    )

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
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
            assert "(fill in the corpus directory, language and n-gram size)" in str(app.query_one("#dest").content)
            assert app.query_one("#f-corpus-language", Select).value is Select.NULL
            app.query_one("#f-corpus-db_path_stub").value = str(tmp_path / "corpora")
            app.query_one("#f-corpus-language").value = "eng"
            app.query_one("#f-acquire-ngram_size").value = "1"
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
    run.joinpath("progress.json").write_text(json.dumps(dict(
        stage="acquire", state="running", started=(now - timedelta(minutes=5)).isoformat(timespec="seconds"),
        updated=now.isoformat(timespec="seconds"), finished=None, files_total=10, files_done=2,
        files_failed=0, files_skipped=0, entries_written=1000, uncompressed_bytes=10, chunks=2,
        current=["f.gz"], message="", db_path="/db", log_path=str(log), slurm_job_id="42", hostname="n1")))
    older = tmp_path / ".lexichron" / "runs" / "20261009_110000_acquire"
    older.mkdir()
    older.joinpath("progress.json").write_text(json.dumps(dict(
        stage="acquire", state="done", started=(now - timedelta(hours=2)).isoformat(timespec="seconds"),
        updated=(now - timedelta(hours=1)).isoformat(timespec="seconds"),
        finished=(now - timedelta(hours=1)).isoformat(timespec="seconds"), files_total=1, files_done=1,
        files_failed=0, files_skipped=0, entries_written=5, uncompressed_bytes=1, chunks=1,
        current=[], message="", db_path=None, log_path=None, slurm_job_id=None, hostname="n1")))

    async def scenario():
        from textual.widgets import DataTable, ProgressBar
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            runs = app.query_one("#runs", DataTable)
            assert runs.row_count == 2 and app.selected_run == run       # newest first, selected
            assert "running job 42 on n1 · 2/10 files (20%)" in str(app.query_one("#run-headline").content)
            bar = app.query_one("#run-bar", ProgressBar)
            assert bar.total == 10 and bar.progress == 2
            assert "1 files in flight" in str(app.query_one("#run-current").content)
            assert app._log_shown[1] == ("first", "second", "third")
            # the log pane follows the file
            log.write_text("first\nsecond\nthird\nfourth\n")
            app.refresh_jobs()
            await pilot.pause()
            assert app._log_shown[1][-1] == "fourth"
            # selecting the older run switches the panel
            runs.move_cursor(row=1)
            await pilot.pause()
            assert app.selected_run == older
            assert str(app.query_one("#run-headline").content).startswith("done on n1 · 1/1 files (100%)")
        _assert_clean(app)

    asyncio.run(scenario())


def test_reopening_without_argument_restores_last_settings(tmp_path):
    proj = tmp_path / "corp" / "20200217" / "eng" / "1gram_files" / "lexichron.yaml"

    async def first():
        app = LexichronApp(None)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            app.query_one("#f-corpus-db_path_stub").value = str(tmp_path / "corp")
            app.query_one("#f-corpus-language").value = "eng"
            app.query_one("#f-acquire-ngram_size").value = "1"
            await pilot.pause()
            app.action_save()
            await pilot.pause()
        _assert_clean(app)

    async def second():
        app = LexichronApp(None)          # no argument, like plain `lexichron-ui`
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert app.reopened and app.project_path == proj
            assert app.query_one("#f-corpus-language").value == "eng"
            assert "(reopened from last time)" in str(app.query_one("#dest").content)
            # changing which corpus it describes moves the settings file with it
            app.query_one("#f-acquire-ngram_size").value = "2"
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
            runs = app.query_one("#runs", DataTable)
            states = [str(runs.get_row_at(i)[1]) for i in range(runs.row_count)]
            assert states == ["running", "stopped"]
            # lexichron's own job is listed first
            jobs = app.query_one("#jobs", DataTable)
            assert str(jobs.get_row_at(0)[0]) == "200"
        _assert_clean(app)

    asyncio.run(scenario())

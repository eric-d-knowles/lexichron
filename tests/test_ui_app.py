"""Headless smoke test of the Textual app (only where textual is installed)."""
import asyncio

import pytest

textual = pytest.importorskip("textual")

from lexichron.ui.app import LexichronApp  # noqa: E402


def _assert_clean(app):
    # run_test re-raises panics, but be explicit: the app must not have errored.
    assert getattr(app, "_exception", None) is None, app._exception
    assert app.return_code in (None, 0), app.return_code


def test_app_builds_form_and_preview(tmp_path):
    proj = tmp_path / "project.yaml"
    proj.write_text(
        "corpus:\n  release: '20200217'\n  language: eng-us\n  db_path_stub: /data/\n"
        "acquire:\n  ngram_size: 2\n  ngram_type: tagged\n  workers: 4\n"
    )

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            call = app.query_one("#call").content
            text = str(call)
            assert "download_and_ingest_to_rocksdb(" in text
            assert "ngram_size=2" in text and "repo_corpus_id='eng-us'" in text
            assert app.query_one("#status").content == "" or str(app.query_one("#status").content) == ""
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

    asyncio.run(scenario())


def test_app_without_project_derives_path_from_db_stub(tmp_path):
    async def scenario():
        app = LexichronApp(None)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert "(set corpus.db_path_stub)" in str(app.query_one("#call").content)
            from textual.widgets import Select
            assert app.query_one("#f-corpus-language", Select).value is Select.NULL
            app.query_one("#f-corpus-db_path_stub").value = str(tmp_path / "corpora")
            app.query_one("#f-corpus-release").value = "20200217"
            app.query_one("#f-corpus-language").value = "eng"
            app.query_one("#f-acquire-ngram_size").value = "1"
            await pilot.pause()
            assert app.project_path == tmp_path / "corpora" / "project.yaml"
            assert "repo_corpus_id='eng'" in str(app.query_one("#call").content)
            app.action_save()
            await pilot.pause()
        _assert_clean(app)
        assert (tmp_path / "corpora" / "project.yaml").exists()

    asyncio.run(scenario())

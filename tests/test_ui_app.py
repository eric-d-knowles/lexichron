"""Headless smoke test of the Textual app (only where textual is installed)."""
import asyncio

import pytest

textual = pytest.importorskip("textual")

from lexichron.ui.app import LexichronApp  # noqa: E402


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
            call = app.query_one("#call").renderable
            text = str(call)
            assert "download_and_ingest_to_rocksdb(" in text
            assert "ngram_size=2" in text and "repo_corpus_id='eng-us'" in text
            assert app.query_one("#status").renderable == "" or str(app.query_one("#status").renderable) == ""
            # change a field; preview follows
            app.query_one("#f-acquire-workers").value = "8"
            await pilot.pause()
            assert "workers=8" in str(app.query_one("#call").renderable)
            # save writes the file
            app.action_save()
            await pilot.pause()
        import yaml
        saved = yaml.safe_load(proj.read_text())
        assert saved["acquire"]["workers"] == 8 and saved["corpus"]["language"] == "eng-us"

    asyncio.run(scenario())

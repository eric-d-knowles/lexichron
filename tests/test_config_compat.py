"""Settings files written by earlier versions keep loading."""
import textwrap

from lexichron.config import build_call, load_project
from lexichron.ui.app import LexichronApp
from ngramprep.ngram_acquire.core import download_and_ingest_to_rocksdb


def _old_file(tmp_path):
    p = tmp_path / "lexichron.yaml"
    p.write_text(textwrap.dedent(f"""
        # written by lexichron ui 0.2.0: ngram_size lived in the acquire section
        corpus:
          release: '20200217'
          language: eng
          db_path_stub: {tmp_path}
        acquire:
          ngram_size: 5
          workers: 20
        """))
    return p


def test_ngram_size_in_acquire_section_is_moved_to_corpus(tmp_path):
    cfg = load_project(_old_file(tmp_path))
    assert cfg["corpus"]["ngram_size"] == 5 and "ngram_size" not in cfg["acquire"]
    kwargs = build_call(download_and_ingest_to_rocksdb, cfg, "acquire")
    assert kwargs["ngram_size"] == 5 and kwargs["workers"] == 20


def test_ui_resaves_old_file_in_the_new_layout(tmp_path):
    import asyncio
    import yaml
    proj = _old_file(tmp_path)

    async def scenario():
        app = LexichronApp(proj)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert "ngram_size=5" in str(app.query_one("#call").content)
            app.action_save()
            await pilot.pause()
    asyncio.run(scenario())
    saved = yaml.safe_load(proj.read_text())
    assert saved["corpus"]["ngram_size"] == 5 and "ngram_size" not in saved["acquire"]


def test_dropped_filter_keys_are_ignored(tmp_path):
    p = tmp_path / "lexichron.yaml"
    p.write_text(textwrap.dedent(f"""
        corpus:
          release: '20200217'
          language: eng-us
          ngram_size: 1
          db_path_stub: {tmp_path}
        filter:
          smart_partitioning: true
          samples_per_worker: 50000
          min_len: 4
        """))
    cfg = load_project(p)
    assert cfg["filter"] == {"min_len": 4}

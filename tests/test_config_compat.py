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
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "ngram_size=5" in str(app.query_one("#call").content)
            app.action_save()
            await pilot.pause()
    asyncio.run(scenario())
    saved = yaml.safe_load(proj.read_text())
    assert saved["corpus"]["ngram_size"] == 5 and "ngram_size" not in saved["acquire"]
    assert saved["corpus"]["name"] == "eng" and "language" not in saved["corpus"]      # the key's new name


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
          work_units: 300
          min_len: 4
        """))
    cfg = load_project(p)
    assert cfg["filter"] == {"min_len": 4}


def test_renamed_acquire_keys_still_load(tmp_path):
    """combined_bigrams became combined_multigrams (and its case switch
    followed) when the setting grew to sequences of any length: a settings
    file from before loads under the new names."""
    from lexichron.config import load_project
    path = tmp_path / "lexichron.yaml"
    path.write_text("acquire:\n  combined_bigrams: [working class]\n  bigram_case_sensitive: true\n  workers: 4\n")
    cfg = load_project(path)
    # the global case switch became a per-sequence choice: on, every sequence was exact
    assert cfg["acquire"] == {"combined_multigrams_case": ["working class"], "workers": 4}
    path.write_text("acquire:\n  combined_multigrams: [working class]\n  multigram_case_sensitive: false\n")
    assert load_project(path)["acquire"] == {"combined_multigrams": ["working class"]}
    path.write_text("acquire:\n  combined_bigrams: [old]\n  combined_multigrams: [new]\n")
    assert load_project(path)["acquire"] == {"combined_multigrams": ["new"]}      # the new name wins


def test_the_corpus_language_key_is_now_its_name(tmp_path):
    """The corpus's identifier (eng-us, eng-fiction) was kept under
    ``language``, which it is not always and which a corpus from another
    source is not at all. A file with the old key loads as before, an
    override under the old name still reaches the corpus, and a file that
    has both keeps the new one."""
    from lexichron.config import apply_overrides
    cfg = load_project(_old_file(tmp_path))
    assert cfg["corpus"]["name"] == "eng" and "language" not in cfg["corpus"]
    assert build_call(download_and_ingest_to_rocksdb, cfg, "acquire")["repo_corpus_id"] == "eng"
    apply_overrides(cfg, ["corpus.language=eng-us"])
    assert cfg["corpus"]["name"] == "eng-us" and "language" not in cfg["corpus"]
    apply_overrides(cfg, ["corpus.name=eng-gb"])
    assert build_call(download_and_ingest_to_rocksdb, cfg, "acquire")["repo_corpus_id"] == "eng-gb"
    both = tmp_path / "both.yaml"
    both.write_text(f"corpus:\n  release: '20200217'\n  name: eng-fiction\n  language: eng\n"
                    f"  ngram_size: 5\n  db_path_stub: {tmp_path}\n")
    assert load_project(both)["corpus"] == {"release": "20200217", "name": "eng-fiction", "ngram_size": 5,
                                            "db_path_stub": str(tmp_path)}

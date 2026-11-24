"""The terminal UI with a Davies corpus (COHA): listed beside the n-gram
corpora, created from the New corpus dialog, its own Acquire form, and a
settings file that keeps what kind of corpus it describes."""
import asyncio

import pytest
import yaml

textual = pytest.importorskip("textual")

from textual.widgets import Button, Input, Select, Static  # noqa: E402

from lexichron.ui.app import LexichronApp, NewCorpus  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXICHRON_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("lexichron.ui.app.slurm_backend", lambda: None)


def _db(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "CURRENT").write_text("MANIFEST-000001\n")


def _clean(app):
    assert getattr(app, "_exception", None) is None, app._exception
    assert app.return_code in (None, 0), app.return_code


async def _settle(app, pilot):
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def test_a_davies_corpus_is_listed_and_its_stages_follow_what_exists(tmp_path):
    """Tokens and sentences are two entries. The tokens offer Acquire and
    Filter; the sentences need the tokens' vocabulary before Filter, and the
    pivoted database before Tune and Train."""
    root = tmp_path / "corpora"
    tokens, sentences = root / "coha" / "token_files", root / "coha" / "sentence_files"
    _db(tokens / "tokens.db")
    _db(sentences / "sentences.db")
    (sentences / "lexichron.yaml").write_text(
        f"corpus:\n  source: davies\n  name: coha\n  db_path_stub: {root}\n"
        f"acquire:\n  source_dir: /data/COHA\n  ngram_type: untagged\n  genres: [fic, mag]\n"
        f"filter:\n  min_len: 4\n")
    _db(root / "20200217" / "eng" / "5gram_files" / "5grams.db")

    async def scenario():
        app = LexichronApp(None)
        async with app.run_test(size=(160, 50)) as pilot:
            await _settle(app, pilot)
            app.query_one("#corpus-root", Input).value = str(root)
            await pilot.pause()
            sel = app.query_one("#corpus-select", Select)
            assert [label for label, _ in sel._options if label] == [
                "COHA · Davies · tokens", "COHA · Davies · sentences", "eng · 20200217 · 5-grams"]
            sel.value = str(sentences)
            await _settle(app, pilot)
            assert app.project_path == sentences / "lexichron.yaml"
            assert app.config["corpus"] == {"source": "davies", "name": "coha", "db_path_stub": str(root)}
            # the Acquire form is the Davies one, holding the file's acquire section
            assert app.query_one("#stage-acquire_davies").display and not app.query_one("#stage-acquire").display
            assert app.query_one("#f-acquire_davies-source_dir", Input).value == "/data/COHA"
            assert app.query_one("#f-acquire_davies-ngram_type", Select).value == "untagged"
            assert app.query_one("#f-acquire_davies-genres", Input).value == "fic, mag"
            assert app.query_one("#f-filter-min_len", Input).value == "4"
            assert "your licensed" in str(app.query_one("#stage-desc", Static).content) or \
                "under licence" in str(app.query_one("#stage-desc", Static).content)
            # the resolved call is the Davies reader's
            call = str(app.query_one("#call").content)
            assert "acquire_davies(" in call and "source_dir='/data/COHA'" in call and "ngram_size" not in call
            # sentences are acquired but there is no vocabulary yet. Filter is offered all the same, as
            # for n-grams: with No vocabulary the rules alone do the filtering; the state line says which
            state = str(app.query_one("#corpus-state", Static).content)
            assert state.startswith("COHA · Davies · sentences: acquired · not filtered (no token vocabulary yet")
            assert "No vocabulary" in state
            assert not app.query_one("#stage-btn-filter", Button).disabled
            assert app.query_one("#stage-btn-tune", Button).disabled
            # filtering against a vocabulary that is not there is refused by the stage, which says what to do
            app._show_stage("filter")
            await pilot.pause()
            app.query_one("#f-filter-no_vocabulary").value = True
            await pilot.pause()
            assert "no_vocabulary=True" in str(app.query_one("#call").content)
            app.query_one("#f-filter-no_vocabulary").value = False
            await pilot.pause()
            # the tokens: Filter is live, the model stages point to the sentences
            sel.value = str(tokens)
            await _settle(app, pilot)
            assert app.config["corpus"] == {"source": "davies", "name": "coha", "unit": "tokens",
                                            "db_path_stub": str(root)}
            assert app.project_path == tokens / "lexichron.yaml"
            assert not app.query_one("#stage-btn-filter", Button).disabled
            tune = app.query_one("#stage-btn-tune", Button)
            assert tune.disabled and "sentences" in tune.tooltip
            assert "no vocabulary yet" in str(app.query_one("#corpus-state", Static).content)
            assert app.query_one("#f-acquire_davies-source_dir", Input).value == ""      # nothing carried over
            # once the tokens are filtered the sentences can be, and once pivoted, tuned
            _db(tokens / "tokens_processed.db")
            (tokens / "tokens_processed.db" / "whitelist.txt").write_text("word\t1\n")
            _db(sentences / "sentences_pivoted.db")
            sel.value = str(sentences)
            await _settle(app, pilot)
            assert "no token vocabulary" not in str(app.query_one("#corpus-state", Static).content)
            assert not app.query_one("#stage-btn-tune", Button).disabled
            assert app.query_one("#stage-btn-measure", Button).disabled
            # an n-gram corpus gets the n-gram Acquire form back
            sel.value = str(root / "20200217" / "eng" / "5gram_files")
            await _settle(app, pilot)
            assert app.query_one("#stage-acquire").display or app.stage != "acquire"
            app._show_stage("acquire")
            await pilot.pause()
            assert app.query_one("#stage-acquire").display and not app.query_one("#stage-acquire_davies").display
            assert app.config["corpus"] == {"db_path_stub": str(root), "release": "20200217", "name": "eng",
                                            "ngram_size": 5}
        _clean(app)

    asyncio.run(scenario())


def test_new_davies_corpus_and_its_settings_file(tmp_path):
    """The dialog asks what kind of corpus; a Davies corpus needs a name and
    whether it is the tokens or the sentences. What is saved says so, and
    has no release or n-gram size."""
    root = tmp_path / "corpora"
    root.mkdir()

    async def scenario():
        app = LexichronApp(None, "tune")
        async with app.run_test(size=(160, 50)) as pilot:
            await _settle(app, pilot)
            app.query_one("#corpus-root", Input).value = str(root)
            await pilot.pause()
            app._corpus_new_pressed()
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, NewCorpus)
            assert dialog.query_one("#new-row-size").display and not dialog.query_one("#new-row-unit").display
            dialog.query_one("#new-source", Select).value = "davies"
            await pilot.pause()
            assert not dialog.query_one("#new-row-size").display and not dialog.query_one("#new-row-release").display
            assert dialog.query_one("#new-row-unit").display
            assert dialog.query_one("#new-name", Select).value == "coha"            # the only one there is
            assert dialog.query_one("#new-unit", Select).value == "tokens"          # what is done first
            assert "read twice" in str(dialog.query_one("#new-note", Static).content)
            dialog._create()
            await _settle(app, pilot)
            assert not isinstance(app.screen, NewCorpus)
            assert app.stage == "acquire"
            assert app.config["corpus"] == {"source": "davies", "name": "coha", "unit": "tokens",
                                            "db_path_stub": str(root)}
            assert app.project_path == root / "coha" / "token_files" / "lexichron.yaml"
            assert app.query_one("#stage-acquire_davies").display
            assert app.query_one("#stage-btn-filter", Button).disabled
            assert "COHA · Davies · tokens" in str(app.query_one("#corpus-state", Static).content)
            assert app.query_one("#corpus-select", Select).value == str(root / "coha" / "token_files")
            # fill in the Acquire form and save
            app.query_one("#f-acquire_davies-source_dir", Input).value = "/data/english-corpora/COHA"
            app.query_one("#f-acquire_davies-workers", Input).value = "20"
            await pilot.pause()
            call = str(app.query_one("#call").content)
            assert "unit='tokens'" in call and "source='davies'" in call
            app.action_save()
            await _settle(app, pilot)
        _clean(app)

    asyncio.run(scenario())
    saved = yaml.safe_load((root / "coha" / "token_files" / "lexichron.yaml").read_text())
    assert saved["corpus"] == {"source": "davies", "name": "coha", "unit": "tokens", "db_path_stub": str(root)}
    assert saved["acquire"] == {"source_dir": "/data/english-corpora/COHA", "workers": 20}
    # and it runs from the command line as saved
    from lexichron.cli import _resolve, stage_entry
    from lexichron.config import build_call, corpus_source, load_project
    cfg = load_project(root / "coha" / "token_files" / "lexichron.yaml")
    call = build_call(_resolve(stage_entry("acquire", corpus_source(cfg))), cfg, "acquire")
    assert call["unit"] == "tokens" and call["repo_corpus_id"] == "coha" and call["workers"] == 20


def test_davies_form_fields_have_labels_and_help():
    from lexichron.cli import _resolve, stage_entry
    from lexichron.ui.schema import stage_sections
    corpus, sec = stage_sections(_resolve(stage_entry("acquire", "davies")), "acquire_davies")
    assert [f.name for f in corpus.fields] == ["source", "name", "unit", "db_path_stub"]
    names = [f.name for f in sec.fields]
    assert names[:6] == ["source_dir", "ngram_type", "genres", "file_range", "overwrite_db", "workers"]
    by = {f.name: f for f in sec.fields}
    assert by["ngram_type"].choices == ["tagged", "untagged"] and by["ngram_type"].title == "Edition"
    assert by["source_dir"].title == "Corpus files" and by["source_dir"].kind == "path" and by["source_dir"].required
    assert by["max_unit_tokens"].advanced and all(f.help for f in sec.fields)
    assert {f.name: f.title for f in corpus.fields}["unit"] == "Records"

"""A corpus directory written with $VARIABLES or ~ (one settings file for
everyone on a cluster) is expanded where it is used, and kept as written."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lexichron import config, corpora  # noqa: E402
from ngramprep.ngram_acquire.db.build_path import build_db_path  # noqa: E402
from ngramprep.utilities.paths import UnsetVariable, expand_path  # noqa: E402


def test_variables_and_home_are_expanded_and_an_unset_variable_is_an_error(monkeypatch):
    monkeypatch.setenv("USER", "abc123")
    monkeypatch.setenv("HOME", "/home/abc123")
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)
    assert expand_path("/scratch/$USER/corpora") == "/scratch/abc123/corpora"
    assert expand_path("/scratch/${USER}/corpora") == "/scratch/abc123/corpora"
    assert expand_path("~/corpora") == "/home/abc123/corpora"
    assert expand_path("/plain/path") == "/plain/path"
    with pytest.raises(UnsetVariable, match=r"\$NO_SUCH_VAR is not set"):
        expand_path("/scratch/$NO_SUCH_VAR/corpora")
    # never a directory literally named $USER
    assert build_db_path("/scratch/$USER/NLP", 5, "20200217", "eng-us") == \
        "/scratch/abc123/NLP/20200217/eng-us/5gram_files/5grams.db"
    with pytest.raises(ValueError):
        build_db_path("/scratch/$NO_SUCH_VAR/NLP", 5, "20200217", "eng-us")


def test_a_stage_receives_the_expanded_directory_and_the_file_keeps_it_as_written(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "abc123")
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)

    def stage(ngram_size, repo_release_id, repo_corpus_id, db_path_stub, archive_path_stub=None):
        pass

    text = "corpus:\n  release: '20200217'\n  name: eng-us\n  ngram_size: 5\n  db_path_stub: /scratch/$USER/NLP\n"
    proj = tmp_path / "lexichron.yaml"; proj.write_text(text)
    cfg = config.load_project(proj)
    assert cfg["corpus"]["db_path_stub"] == "/scratch/$USER/NLP"                 # as written, for saving back
    kwargs = config.build_call(stage, cfg, "acquire")
    assert kwargs["db_path_stub"] == "/scratch/abc123/NLP"                        # as the job sees it
    cfg["corpus"]["archive_path_stub"] = "/archive/$USER"
    assert config.build_call(stage, cfg, "acquire")["archive_path_stub"] == "/archive/abc123"
    cfg["corpus"]["db_path_stub"] = "/scratch/$NO_SUCH_VAR/NLP"
    with pytest.raises(config.ConfigError, match=r"corpus.db_path_stub: .*\$NO_SUCH_VAR is not set"):
        config.build_call(stage, cfg, "acquire")


def test_the_corpus_bar_reads_an_expanded_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path))
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)
    d = tmp_path / "20200217" / "eng-us" / "5gram_files"
    d.mkdir(parents=True)
    (d / "lexichron.yaml").write_text("corpus: {}\n")
    info = corpora.corpus_info("$CORPUS_ROOT", "20200217", "eng-us", 5)
    assert info.dir == d and info.settings == d / "lexichron.yaml"
    assert [i.dir for i in corpora.scan_corpora("$CORPUS_ROOT")] == [d]
    assert corpora.scan_corpora("/x/$NO_SUCH_VAR") == []

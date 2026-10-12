"""lexichron.stages.filter: plain settings -> FilterConfig/PipelineConfig -> engine."""
import pytest

core = pytest.importorskip("ngramprep.ngram_filter.filters.core_cy")

import fake_rocks  # noqa: E402
from ngramprep.common_db import api as dbapi  # noqa: E402
from lexichron.stages import filter as stage  # noqa: E402
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths  # noqa: E402
from test_filter_stage import FakeLemmatizer, _raw_unigrams, _unpack  # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


@pytest.fixture
def no_spacy(monkeypatch):
    import ngramprep.ngram_filter.stopwords as sw
    import ngramprep.ngram_filter.lemmatizer as lm
    monkeypatch.setattr(sw, "load_stopwords", lambda lang: ({"the", "of"}, lang))
    monkeypatch.setattr(lm, "CachedSpacyLemmatizer", lambda language=None, **kw: FakeLemmatizer())


def test_text_language_derived_from_corpus():
    assert stage.text_language_for("eng-us") == "en" and stage.text_language_for("rus") == "ru"
    with pytest.raises(ValueError):
        stage.text_language_for("xyz")


def test_filter_ngrams_from_plain_settings(rocks, no_spacy, tmp_path):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    out = stage.filter_ngrams(1, "20200217", "eng", str(stub), workers=2, mode="restart",
                              output_vocabulary_top_n=2,
                              spool_dir=str(tmp_path), progress_path=str(tmp_path / "p.json"))
    data = fake_rocks.dump(out)
    assert set(data) == {b"cat", b"woman", b"run"}
    assert _unpack(data[b"cat"]) == [(1990, 18, 6), (2000, 5, 1)]
    assert [l.split("\t")[0] for l in paths["whitelist"].read_text().splitlines()] == ["cat", "woman"]


def test_vocabulary_mode_lemmatizes_before_lookup(rocks, no_spacy, tmp_path):
    stub = tmp_path / "corpora"
    paths = derive_paths(str(stub), 1, "20200217", "eng")
    _raw_unigrams(str(paths["src_db"]))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("cat\t100\nwoman\t5\n")
    out = stage.filter_ngrams(1, "20200217", "eng", str(stub), vocabulary=str(vocab),
                              always_include=["the"], workers=2, mode="restart",
                              output_vocabulary=False, spool_dir=str(tmp_path))
    data = fake_rocks.dump(out)
    # cats -> cat (in vocab), women -> woman (in vocab), "the" kept via always_include,
    # running -> run (not in vocab) and the rest -> <UNK> -> dropped (min_context_tokens=1)
    assert set(data) == {b"cat", b"woman", b"the"}

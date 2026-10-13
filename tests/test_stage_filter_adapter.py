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


def _raw_bigrams(path):
    import struct
    db = fake_rocks.open(path, mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for key in (b"the cat", b"cats run", b"women run", b"the the"):
            wb.put(key, struct.pack("<3Q", 1990, 2, 1))
    db.close()


def test_multigrams_default_to_the_corpus_unigram_vocabulary(rocks, no_spacy, tmp_path):
    stub = tmp_path / "corpora"
    paths2 = derive_paths(str(stub), 2, "20200217", "eng")
    _raw_bigrams(str(paths2["src_db"]))
    # no unigram whitelist yet: refuse rather than filter 2-grams without a vocabulary
    with pytest.raises(FileNotFoundError) as exc:
        stage.filter_ngrams(2, "20200217", "eng", str(stub), workers=2, mode="restart", spool_dir=str(tmp_path))
    assert "unigram filter" in str(exc.value)
    # the explicit override: rule filters only (the_the -> dropped, cats_run -> cat run)
    out = stage.filter_ngrams(2, "20200217", "eng", str(stub), no_vocabulary=True, workers=2, mode="restart",
                              pivot=False, spool_dir=str(tmp_path))
    assert b"cat run" in fake_rocks.dump(out)
    # the unigram stage's output is picked up automatically
    wl = derive_paths(str(stub), 1, "20200217", "eng")["whitelist"]
    wl.parent.mkdir(parents=True)
    wl.write_text("cat\t100\nrun\t50\n")
    out = stage.filter_ngrams(2, "20200217", "eng", str(stub), workers=2, mode="restart", spool_dir=str(tmp_path),
                              pivot=False, progress_path=str(tmp_path / "p.json"))
    data = fake_rocks.dump(out)
    # the run records which vocabulary it applied, for the Progress tab
    import json
    assert json.loads((tmp_path / "p.json").read_text())["inputs"]["vocabulary"] == str(wl)
    # min_context_tokens defaults to 2 for bigrams: only n-grams with two vocabulary lemmas survive
    assert set(data) == {b"cat run"}
    # and no whitelist is written for multigrams by default
    assert not paths2["whitelist"].exists()


def test_multigrams_are_filtered_and_pivoted_in_one_pass(rocks, no_spacy, tmp_path):
    import struct
    from ngramprep.ngram_pivot.encoding import decode_year_ngram_key, decode_year_stats
    stub = tmp_path / "corpora"
    paths2 = derive_paths(str(stub), 2, "20200217", "eng")
    db = fake_rocks.open(str(paths2["src_db"]), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        # 'Cats run' and 'cats run' collapse onto 'cat run': their years must be summed
        wb.put(b"Cats run", struct.pack("<3Q", 1990, 2, 1) + struct.pack("<3Q", 2000, 1, 1))
        wb.put(b"cats run", struct.pack("<3Q", 1990, 5, 2))
        wb.put(b"the cat", struct.pack("<3Q", 1990, 9, 9))        # 'the' not in vocab -> 1 real token -> dropped
    db.close()
    wl = derive_paths(str(stub), 1, "20200217", "eng")["whitelist"]
    wl.parent.mkdir(parents=True)
    wl.write_text("cat\t100\nrun\t50\n")

    out = stage.filter_ngrams(2, "20200217", "eng", str(stub), workers=2, mode="restart",
                              spool_dir=str(tmp_path), progress_path=str(tmp_path / "p.json"))
    assert out.endswith("2grams_pivoted.db") and not paths2["dst_db"].exists()
    data = {k: v for k, v in fake_rocks.dump(out).items() if not k.startswith(b"__")}
    decoded = {decode_year_ngram_key(k): v for k, v in data.items()}
    assert set(decoded) == {(1990, b"cat run"), (2000, b"cat run")}
    # 24-byte values: (year, count, volumes), summed across the two raw spellings
    assert struct.unpack("<3Q", decoded[(1990, b"cat run")]) == (1990, 7, 3)
    assert decode_year_stats(decoded[(1990, b"cat run")]) == (7, 3)      # what training reads
    assert decode_year_stats(decoded[(2000, b"cat run")]) == (1, 1)
    import json
    doc = json.loads((tmp_path / "p.json").read_text())
    assert doc["db_path"].endswith("2grams_pivoted.db") and doc["counters"]["kept"] == 3   # year records
    # asking for a vocabulary from a pivoted output is refused
    with pytest.raises(ValueError):
        stage.filter_ngrams(2, "20200217", "eng", str(stub), output_vocabulary=True, spool_dir=str(tmp_path))

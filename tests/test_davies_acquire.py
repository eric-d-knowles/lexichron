"""Acquiring a Davies corpus: the COHA reader (both editions), the worker on
the n-gram acquisition's own ingest path, and the project-file plumbing.
The fixture is a made-up corpus in COHA's layout, with the quirks the real
files have; no licensed text is used."""
import json
import struct
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from lexichron.config import ConfigError, build_call, load_project
from lexichron.corpora import corpus_layout
from lexichron.davies.formats import corpus_format, google_tag
from lexichron.davies.reader import ReadError, archives, documents, units
from lexichron.davies.worker import process_archive
from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX
from ngramprep.ngram_acquire.executor import process_files

COHA = corpus_format("coha")

# word, CLAWS tag. "@" x10 is a copyright gap; <p> a paragraph mark.
DOC_A = [("The", "at"), ("buildings", "nn2"), ("stood", "vvd"), (".", "y"),
         ("He", "pphs1"), ("was", "vbdz"), ("building", "vvg_nn1"), ("a", "at1"), ("better", "jjr"), ("one", "pn1"),
         ("?", "y"), ("<p>", "null")] + [("@", "ii")] * 10 + \
        [("leaves", "nn2"), ("fall", "vv0"), (",", "y"), ("slowly", "rr"), ("!", "y"),
         ("The", "at"), ("buildings", "nn2"), ("stood", "vvd"), (".", "y")]
DOC_B = [("txt", "nnu"), ("ART", "nn1"), (".", "y"), ("The", "at"), ("buildings", "nn2"), ("stood", "vvd")]


def _wlp(doc_id, rows, header_end="\n", padding=0):
    lines = [f"@@{doc_id}\t##\tfo"]
    body = [f"{w}\t{w.lower()}\t{t}" for w, t in rows] + ["q!\tq!\tq!"] * padding
    return (lines[0] + header_end + "\r\n".join(body) + "\r\n").encode("latin-1")


def _text(doc_id, rows):
    return (f"@@{doc_id}\r\n\n" + " ".join(w for w, _ in rows) + " ").encode("latin-1")


@pytest.fixture
def coha(tmp_path):
    """<tmp>/COHA with text/ and wlp/: two decades, three documents."""
    root = tmp_path / "COHA"
    docs = {"1810s": {"mag_1815_1.txt": DOC_A, "fic_1817_2.txt": DOC_B}, "1820s": {"news_1821_3.txt": DOC_B}}
    for edition, make in (("text", _text), ("wlp", _wlp)):
        (root / edition).mkdir(parents=True)
        for decade, members in docs.items():
            with zipfile.ZipFile(root / edition / f"{edition}_{decade}_abc.zip", "w") as zf:
                for name, rows in members.items():
                    doc_id = name.split("_")[2][:-4]
                    if edition == "wlp":
                        # the real files: a header ending in a bare line feed, a wrong id and padding in fiction
                        fiction = name.startswith("fic")
                        zf.writestr(name, _wlp("999" if fiction else doc_id, rows, padding=20 if fiction else 0))
                    else:
                        zf.writestr(name, _text(doc_id, rows))
    return root


def test_claws_tags_become_the_coarse_tags_the_filter_knows():
    assert [google_tag(t) for t in ("nn1", "nn2", "np1", "nnt1")] == ["NOUN"] * 4
    assert [google_tag(t) for t in ("vvz", "vbdz", "vm", "vhd")] == ["VERB"] * 4
    assert google_tag("jj") == "ADJ" and google_tag("rr") == "ADV" and google_tag("xx") == "ADV"
    assert google_tag("jj_nn1") == "ADJ" and google_tag("nn1_jj") == "NOUN"          # the first alternative
    assert google_tag("jj@_nn1@") == "ADJ" and google_tag("vvz@_np1") == "VERB" and google_tag("rr21") == "ADV"
    assert google_tag("NP1_VV0                  ") == "NOUN"                         # upper case, padded
    assert google_tag("y") == "." and google_tag("null") == "." and google_tag('"') == "."
    assert google_tag("at1") == "DET" and google_tag("ii") == "ADP" and google_tag("appge") == "PRON"
    assert google_tag("") is None and google_tag("   ") is None
    from ngramprep.ngram_acquire.utils.filters import DEFAULT_TAGS
    assert {google_tag(t) for t in ("nn1", "vv0", "jj", "rr", "at", "ii", "pn1", "mc", "cc", "fo", "y")} <= set(DEFAULT_TAGS)


def test_documents_come_out_the_same_from_either_edition(coha):
    tagged = {d.name: d for a in archives(coha, COHA, "tagged") for d in documents(a, COHA, "tagged")}
    plain = {d.name: d for a in archives(coha, COHA, "untagged") for d in documents(a, COHA, "untagged")}
    assert set(tagged) == set(plain) == {"mag_1815_1.txt", "fic_1817_2.txt", "news_1821_3.txt"}
    for name in tagged:
        assert tagged[name].tokens == plain[name].tokens, name       # the padding and the header are gone
        assert (tagged[name].year, tagged[name].genre) == (plain[name].year, plain[name].genre)
        assert plain[name].tags is None and len(tagged[name].tags) == len(tagged[name].tokens)
    assert (tagged["fic_1817_2.txt"].year, tagged["fic_1817_2.txt"].genre) == (1817, "fic")   # from the name
    assert tagged["mag_1815_1.txt"].tokens[0] == "The"               # not lost with a header that ends in LF
    assert tagged["fic_1817_2.txt"].tokens[0] == "ART"               # the leftover "txt" is dropped
    # the same units, tags apart
    for name in tagged:
        a = [[t.rsplit("_", 1)[0] for t in u] for u in units(tagged[name], COHA)]
        assert a == list(units(plain[name], COHA)), name


def test_units_end_at_sentences_paragraphs_and_gaps():
    from lexichron.davies.reader import Document
    words = [w for w, _ in DOC_A]
    doc = Document("mag_1815_1.txt", 1815, "mag", words, [google_tag(t) for _, t in DOC_A])
    assert list(units(doc, COHA)) == [
        ["The_DET", "buildings_NOUN", "stood_VERB"],
        ["He_PRON", "was_VERB", "building_VERB", "a_DET", "better_ADJ", "one_PRON"],
        ["leaves_NOUN", "fall_VERB", ",_.", "slowly_ADV"],
        ["The_DET", "buildings_NOUN", "stood_VERB"],
    ]
    plain = Document("mag_1815_1.txt", 1815, "mag", words, None)
    assert list(units(plain, COHA))[2] == ["leaves", "fall", ",", "slowly"]
    # text with no sentence punctuation is cut at max_tokens
    long = Document("x", 1900, "mag", [f"w{i}" for i in range(11)], None)
    assert [len(u) for u in units(long, COHA, max_tokens=4)] == [4, 4, 3]


def test_a_tagged_row_that_cannot_be_read_is_counted_and_odd_words_are_handled(tmp_path):
    z = tmp_path / "wlp_1810s_x.zip"
    body = ("@@1\t##\tfo\nGood\tgood\tjj\r\njust a word\r\n - -  The\t---the\tat\r\n\x00\tp11\tfo\r\n"
            "end\tend\tnn1\t\r\n")
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("mag_1815_1.txt", body.encode())
    (doc,) = documents(z, COHA, "tagged")
    assert doc.unreadable == 1 and doc.bad_rows == ["just a word"]
    # "- - The" is three tokens, the tag belonging to the word; the NUL row (a page mark) is not a token
    assert doc.tokens == ["Good", "-", "-", "The", "end"] and doc.tags == ["ADJ", None, None, "DET", "NOUN"]
    # a member that is not named like a document fails the archive
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", b"hello")
    with pytest.raises(ReadError, match="readme.txt"):
        list(documents(z, COHA, "tagged"))


def test_the_edition_is_looked_for_where_the_setting_says(coha, tmp_path):
    assert [p.name for p in archives(coha, COHA, "tagged")] == ["wlp_1810s_abc.zip", "wlp_1820s_abc.zip"]
    assert [p.name for p in archives(coha, COHA, "untagged")] == ["text_1810s_abc.zip", "text_1820s_abc.zip"]
    only_text = tmp_path / "only"; (only_text / "text").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="It has the untagged edition"):
        archives(only_text, COHA, "tagged")
    with pytest.raises(ValueError, match="ngram_type"):
        archives(coha, COHA, "all")


@pytest.fixture
def fake_store(monkeypatch):
    import fake_rocks
    from ngramprep.common_db import api
    fake_rocks.reset()
    monkeypatch.setattr(api, "rs", fake_rocks)
    return fake_rocks


def _unpack(blob):
    n = len(blob) // 8; v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


@pytest.mark.parametrize("ngram_type", ["tagged", "untagged"])
def test_archives_are_ingested_with_their_markers_and_counts_are_summed(coha, tmp_path, fake_store, ngram_type):
    """The n-gram acquisition's own path: a sorted file per archive, ingested
    with a marker that holds the totals of what was kept."""
    files = [str(p) for p in archives(coha, COHA, ngram_type)]
    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, written, batches, nbytes = process_files(
        files, ThreadPoolExecutor, 2, db, spool_dir=str(tmp_path / "spool"), chunk_entries=3,
        worker_fn=process_archive, worker_kwargs=dict(corpus="coha", ngram_type=ngram_type))
    assert bad == [] and len(ok) == 2
    sentence = b"The_DET buildings_NOUN stood_VERB" if ngram_type == "tagged" else b"The buildings stood"
    # twice in one document of 1815, once in one of 1817, once in one of 1821: occurrences and documents by year
    assert _unpack(db.data[sentence]) == [(1815, 2, 1), (1817, 1, 1), (1821, 1, 1)]
    art = b"ART_NOUN" if ngram_type == "tagged" else b"ART"
    assert _unpack(db.data[art]) == [(1817, 1, 1), (1821, 1, 1)]
    data = {k: v for k, v in db.data.items() if not k.startswith(PROCESSED_PREFIX)}
    assert len(data) == 4 and not any(b"@" in k or b"q!" in k or b"<p>" in k or b"txt" in k for k in data)
    markers = {k[len(PROCESSED_PREFIX):].decode(): json.loads(v) for k, v in db.data.items()
               if k.startswith(PROCESSED_PREFIX)}
    first, second = (markers[p.rsplit("/", 1)[1]] for p in files)
    # 1810s: document A has 4 units (3 distinct), document B 2; 1820s: document B again
    assert (first["match_total"], first["volume_total"], first["documents"]) == (6, 5, 2)
    assert (second["match_total"], second["volume_total"], second["documents"]) == (2, 2, 1)
    assert first["problem_lines"] == 0 and first["tokens_kept"] == 3 + 6 + 4 + 3 + 1 + 3
    # the ledger: what the database holds is what the markers say went in
    total = [sum(t[i] for v in data.values() for t in _unpack(v)) for i in (1, 2)]
    assert total == [first["match_total"] + second["match_total"], first["volume_total"] + second["volume_total"]]
    assert not [p for p in (tmp_path / "spool").rglob("*") if p.suffix in (".chunk", ".part", ".sst")]


def test_genres_can_be_left_out_and_a_bad_archive_is_not_marked(coha, tmp_path, fake_store):
    files = [str(p) for p in archives(coha, COHA, "tagged")]
    with zipfile.ZipFile(coha / "wlp" / "wlp_1830s_abc.zip", "w") as zf:
        zf.writestr("notes.txt", b"not a document")
    files.append(str(coha / "wlp" / "wlp_1830s_abc.zip"))
    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, *_ = process_files(files, ThreadPoolExecutor, 2, db, spool_dir=str(tmp_path / "spool"),
                                worker_fn=process_archive,
                                worker_kwargs=dict(corpus="coha", ngram_type="tagged", genres=["mag", "news"]))
    assert len(ok) == 2 and len(bad) == 1 and "wlp_1830s_abc.zip" in bad[0] and "notes.txt" in bad[0]
    assert PROCESSED_PREFIX + b"wlp_1830s_abc.zip" not in db.data          # a re-run reads it again
    assert _unpack(db.data[b"The_DET buildings_NOUN stood_VERB"]) == [(1815, 2, 1), (1821, 1, 1)]   # no fiction
    marker = json.loads(db.data[PROCESSED_PREFIX + b"wlp_1810s_abc.zip"])
    assert marker["documents"] == 1 and marker["documents_skipped"] == 1


def test_a_davies_corpus_in_the_project_file(tmp_path, coha):
    """source: davies and a name; no release or n-gram size. The acquire
    stage is then the Davies reader, and the corpus lives at <stub>/<name>."""
    from lexichron.cli import _resolve, stage_entry
    from lexichron.config import corpus_source
    proj = tmp_path / "lexichron.yaml"
    proj.write_text(f"corpus:\n  source: davies\n  name: coha\n  db_path_stub: {tmp_path / 'corpora'}\n"
                    f"acquire:\n  source_dir: {coha}\n  ngram_type: untagged\n  genres: [fic, mag]\n")
    cfg = load_project(proj)
    assert corpus_source(cfg) == "davies"
    func = _resolve(stage_entry("acquire", "davies"))
    assert func.__name__ == "acquire_davies"
    assert build_call(func, cfg, "acquire") == {
        "source": "davies", "repo_corpus_id": "coha", "db_path_stub": str(tmp_path / "corpora"),
        "source_dir": str(coha), "ngram_type": "untagged", "genres": ["fic", "mag"]}
    # an n-gram corpus has an n-gram size, not a unit
    with pytest.raises(ConfigError, match="corpus.unit belongs to a Davies corpus"):
        build_call(func, {"corpus": {"name": "eng", "unit": "tokens"}}, "acquire")
    # the corpus lives at <stub>/<name>, with its tokens (the 1-grams' counterpart, which give the
    # vocabulary) and its sentences (what is trained on) side by side
    root = tmp_path / "corpora" / "coha"
    text = corpus_layout(tmp_path / "corpora", None, None, "COHA", source="davies")
    words = corpus_layout(tmp_path / "corpora", None, None, "coha", source="davies", unit="tokens")
    assert text.dir == root / "sentence_files" and text.raw_db == text.dir / "sentences.db"
    assert text.training_db == text.dir / "sentences_pivoted.db" and text.models == text.dir / "models"
    assert words.dir == root / "token_files" and words.raw_db == words.dir / "tokens.db"
    assert words.vocabulary == words.dir / "tokens_processed.db" / "whitelist.txt"
    assert text.shared_vocabulary == words.shared_vocabulary == words.vocabulary
    cfg["corpus"]["unit"] = "tokens"
    assert build_call(func, cfg, "acquire")["unit"] == "tokens"
    cfg["corpus"]["unit"] = "paragraphs"
    with pytest.raises(ConfigError, match="sentences or tokens"):
        build_call(func, cfg, "acquire")
    del cfg["corpus"]["unit"]
    with pytest.raises(ValueError, match="unit must be one of"):
        corpus_layout(tmp_path, None, None, "coha", source="davies", unit="paragraphs")
    # a release or an n-gram size does not belong to it; nor does an unknown source or corpus
    cfg["corpus"]["release"] = "20200217"
    with pytest.raises(ConfigError, match="davies, which has no release"):
        build_call(func, cfg, "acquire")
    del cfg["corpus"]["release"]
    # every other stage is the n-gram stage, told the kind of corpus
    from lexichron.cli import STAGES
    assert all(stage_entry(st, "davies") == STAGES[st][0] for st in ("filter", "tune", "train", "measure"))
    from lexichron.stages.tune import tune_models
    call = build_call(tune_models, cfg, "tune")
    assert (call["source"], call["ngram_size"], call["repo_release_id"]) == ("davies", None, None)
    # the filter is the n-gram filter; a Davies corpus has no n-gram size or release to give it
    from lexichron.stages.filter import filter_ngrams
    assert _resolve(stage_entry("filter", "davies")) is filter_ngrams
    call = build_call(filter_ngrams, cfg, "filter")
    assert call["ngram_size"] is None and call["repo_release_id"] is None and call["source"] == "davies"
    cfg["corpus"]["source"] = "gutenberg"
    with pytest.raises(ConfigError, match="ngrams or davies"):
        corpus_source(cfg)
    with pytest.raises(ValueError, match="no Davies corpus called 'coca'"):
        corpus_format("coca")
    # an n-gram project file is untouched by all this
    assert stage_entry("acquire") == "ngramprep.ngram_acquire:download_and_ingest_to_rocksdb"


def test_overwriting_a_davies_database_is_asked_about(tmp_path):
    from lexichron.safeguards import overwrite_target
    kwargs = {"source": "davies", "repo_corpus_id": "coha", "db_path_stub": str(tmp_path), "overwrite_db": True}
    assert overwrite_target("acquire", kwargs) == tmp_path / "coha" / "sentence_files" / "sentences.db"
    assert overwrite_target("acquire", dict(kwargs, unit="tokens")) == tmp_path / "coha" / "token_files" / "tokens.db"
    assert overwrite_target("acquire", dict(kwargs, overwrite_db=False)) is None
    kwargs = {"source": "davies", "repo_corpus_id": "coha", "db_path_stub": str(tmp_path), "mode": "restart"}
    assert overwrite_target("filter", kwargs) == tmp_path / "coha" / "sentence_files" / "sentences_pivoted.db"
    assert overwrite_target("filter", dict(kwargs, unit="tokens")) == \
        tmp_path / "coha" / "token_files" / "tokens_processed.db"


def test_tokens_are_the_corpus_s_one_grams(coha, tmp_path, fake_store):
    """unit="tokens": every token of the same units as a record of its own,
    with its occurrences and the documents it is in, by year. This is what
    the vocabulary is built from, as from Google's 1-grams."""
    files = [str(p) for p in archives(coha, COHA, "tagged")]
    db = fake_store.DB(str(tmp_path / "db"))
    ok, bad, *_ = process_files(files, ThreadPoolExecutor, 2, db, spool_dir=str(tmp_path / "spool"),
                                worker_fn=process_archive,
                                worker_kwargs=dict(corpus="coha", ngram_type="tagged", unit="tokens"))
    assert bad == [] and len(ok) == 2
    data = {k: _unpack(v) for k, v in db.data.items() if not k.startswith(PROCESSED_PREFIX)}
    assert data[b"buildings_NOUN"] == [(1815, 2, 1), (1817, 1, 1), (1821, 1, 1)]
    assert data[b"building_VERB"] == [(1815, 1, 1)] and data[b",_."] == [(1815, 1, 1)]
    assert all(b" " not in k for k in data) and b"@_ADP" not in data and b"._." not in data
    markers = [json.loads(v) for k, v in db.data.items() if k.startswith(PROCESSED_PREFIX)]
    # every token kept is one occurrence: the ledger's total is the tokens kept
    assert sum(m["match_total"] for m in markers) == sum(m["tokens_kept"] for m in markers) == 20 + 4
    assert sum(t[1] for v in data.values() for t in v) == 24


def test_a_davies_corpus_goes_through_the_n_gram_filter(coha, tmp_path, monkeypatch):
    """Tokens first, which give the vocabulary; then the sentences, filtered
    against it and pivoted by year: the n-gram filter's two runs, on a
    database made from COHA's layout."""
    import fake_rocks
    import ngramprep.ngram_filter.lemmatizer as lm
    import ngramprep.ngram_filter.stopwords as sw
    from ngramprep.common_db import api
    from lexichron.stages.filter import filter_ngrams
    from ngramprep.ngram_pivot.encoding import decode_year_ngram_key
    fake_rocks.reset()
    monkeypatch.setattr(api, "rs", fake_rocks)

    class Lemmatizer:
        def __init__(self, language=None, use_tags=True, **kw):
            self._use_tags = use_tags
        def lemmatize(self, word, pos=None):
            table = {("buildings", "NOUN"): "building", ("building", "VERB"): "build", ("stood", "VERB"): "stand",
                     ("leaves", "NOUN"): "leaf"}
            return table.get((word, pos), word)

    monkeypatch.setattr(sw, "load_stopwords", lambda lang: ({"the", "was", "one"}, lang))
    monkeypatch.setattr(lm, "CachedSpacyLemmatizer", Lemmatizer)
    stub = tmp_path / "corpora"
    for unit in ("tokens", "sentences"):
        layout = corpus_layout(stub, None, None, "coha", source="davies", unit=unit)
        db = fake_rocks.open(str(layout.raw_db), mode="rw", create_if_missing=True)
        process_files([str(p) for p in archives(coha, COHA, "tagged")], ThreadPoolExecutor, 2, db,
                      spool_dir=str(tmp_path / "spool"), worker_fn=process_archive,
                      worker_kwargs=dict(corpus="coha", ngram_type="tagged", unit=unit))
        db.compact_all()                         # the filter refuses a source that was never compacted
        db.close()
        (layout.raw_db / "lexichron.json").write_text(json.dumps(
            {"format": 1, "stage": "acquire", "settings": {"ngram_type": "tagged", "source": "davies"}}))
    # sentences before tokens: there is no vocabulary yet, and the message says what to do
    with pytest.raises(FileNotFoundError, match="Filter this corpus's tokens first"):
        filter_ngrams(None, None, "coha", str(stub), source="davies", workers=2, spool_dir=str(tmp_path))
    # ... or the sentences are filtered with the rules alone: stopwords, short words and punctuation
    # become <UNK>, nothing is checked against a vocabulary, and lemmas are made as for the tokens
    out = filter_ngrams(None, None, "coha", str(stub), source="davies", no_vocabulary=True, workers=2,
                        mode="restart", spool_dir=str(tmp_path), verify_counts=False)
    rules_only = sorted(decode_year_ngram_key(k)[1].decode() for k in fake_rocks.dump(out))
    assert "<UNK> building stand" in rules_only and "leaf fall <UNK> slowly" in rules_only
    assert "<UNK> <UNK> build <UNK> better <UNK>" in rules_only
    filter_ngrams(None, None, "coha", str(stub), source="davies", unit="tokens", workers=2, mode="restart",
                  spool_dir=str(tmp_path), verify_counts=False)
    words = corpus_layout(stub, None, None, "coha", source="davies", unit="tokens")
    vocabulary = [line.split("\t")[0] for line in words.vocabulary.read_text().splitlines()]
    # lemmas by tag, lowercased; stopwords, punctuation and short words are not vocabulary
    assert set(vocabulary) == {"building", "build", "stand", "better", "leaf", "fall", "slowly", "art"}
    out = filter_ngrams(None, None, "coha", str(stub), source="davies", workers=2, mode="restart",
                        spool_dir=str(tmp_path), verify_counts=False)
    text = corpus_layout(stub, None, None, "coha", source="davies")
    assert str(out) == str(text.training_db)
    rows = sorted((decode_year_ngram_key(k)[0], decode_year_ngram_key(k)[1].decode())
                  for k in fake_rocks.dump(text.training_db))
    assert rows == [(1815, "<UNK> <UNK> build <UNK> better <UNK>"), (1815, "<UNK> building stand"),
                    (1815, "leaf fall <UNK> slowly"), (1817, "<UNK> building stand"), (1821, "<UNK> building stand")]


def test_models_are_tuned_and_trained_on_a_davies_corpus_s_sentences(tmp_path, monkeypatch):
    """tune, train and measure are told the kind of corpus and find its
    pivoted sentences, its token vocabulary, and a place for searches and
    ensembles beside them; they refuse to be pointed at the tokens."""
    import numpy as np
    import fake_rocks
    from ngramprep.common_db import api
    from lexichron.stages import measure, train, tune
    from test_tune_stage import _datasets, fake_train
    fake_rocks.reset()
    monkeypatch.setattr(api, "rs", fake_rocks)
    stub = tmp_path / "corpora"
    text = corpus_layout(stub, None, None, "coha", source="davies")
    db = fake_rocks.open(str(text.training_db), mode="rw", create_if_missing=True)
    words = [f"w{i:02d}" for i in range(40)]
    rng = np.random.default_rng(0)
    with db.write_batch() as wb:
        for y in (1990, 1991, 2000, 2001):
            for i in range(300):                        # sentences of varying length, counted once
                toks = list(rng.choice(words, size=int(rng.integers(3, 12))))
                wb.put(struct.pack(">I", y) + " ".join(toks).encode(), struct.pack("<QQQ", y, 1, 1))
    db.close()
    text.shared_vocabulary.parent.mkdir(parents=True)
    text.shared_vocabulary.write_text("\n".join(words[:30]) + "\n")
    data = _datasets(tmp_path)
    common = dict(source="davies", workers=1, workers_per_model=1, probes_per_decile=3, train_fn=fake_train,
                  similarity_dataset=str(data / "simlex999.txt"), axis_lexicon=str(data / "vader_lexicon.txt"),
                  axis_poles=str(data / "poles.txt"))
    out = tune.tune_models(None, None, "coha", str(stub), years=(1990, 2001), year_pairs=2, vector_size=[8],
                           window=[2], min_count=[2], epochs=[1], approach=["skip-gram"], weight_by=["none"],
                           seeds=2, split_half=True, name="t1", mode="restart",
                           analogy_dataset=str(data / "questions-words.txt"), **common)
    assert Path(out) == text.dir / "tuning" / "t1"
    assert len(list((Path(out) / "scores").glob("*.json"))) == 4          # one cell, four sampled years
    ens = train.train_models(None, None, "coha", str(stub), years=(1990, 1991), vector_size=8, window=2,
                             min_count=2, epochs=1, corpus_replicates=2, seeds_per_corpus=1, name="e1",
                             mode="restart", **common)
    assert Path(ens) == text.dir / "models" / "e1" and (Path(ens) / "train.json").is_file()
    series = measure.measure_series(None, None, "coha", str(stub), ensemble="e1", name="m1", kind="frequency",
                                    words=("w01", "w02"), source="davies", workers=1)
    assert Path(series) == Path(ens) / "measures" / "m1"
    for stage, call in ((tune.tune_models, {}), (train.train_models, {}),
                        (measure.measure_series, {"ensemble": "e1", "name": "m2"})):
        with pytest.raises(ValueError, match="trained on a Davies corpus's sentences"):
            stage(None, None, "coha", str(stub), source="davies", unit="tokens", **call)

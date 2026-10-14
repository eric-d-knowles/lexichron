"""lexichron.inspection: a few decoded records from any database or vocabulary file."""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                      # noqa: E402
from ngramprep.common_db import api as dbapi            # noqa: E402
from lexichron import inspection                        # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def _packed(*triples):
    return b"".join(struct.pack("<QQQ", *t) for t in triples)


def _ngram_db(path):
    db = fake_rocks.open(str(path), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"__processed__/1-00000-of-00014.gz", b"marker")
        wb.put(b"apple", _packed((1990, 5, 2), (2000, 7, 3)))
        wb.put(b"apple pie", _packed((2000, 1, 1)))
        wb.put(b"zebra", _packed((1950, 2, 2)))
    db.compact_all()
    return db


def _pivoted_db(path):
    db = fake_rocks.open(str(path), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"__rangemap__/partition", b"[]")
        for year, ng, count in ((1990, b"cat run", 5), (2011, b"nuclear <UNK>", 9), (2011, b"nuclear power", 4)):
            wb.put(struct.pack(">I", year) + ng, struct.pack("<QQQ", year, count, count))
    return db


def test_ngram_database_is_sniffed_and_decoded(rocks, tmp_path):
    _ngram_db(tmp_path / "1grams_processed.db")
    info = inspection.describe(str(tmp_path / "1grams_processed.db"))
    assert info["kind"] == "rocksdb" and info["layout"] == "ngram" and info["compacted"]
    assert info["records"] == 3 and info["bookkeeping"] == 1        # the marker is not a record
    layout, found = inspection.rows(str(tmp_path / "1grams_processed.db"), "head", n=10)
    assert layout == "ngram"
    assert found == [("apple", "12 in 5 volumes (1990-2000, 2 years)"),
                     ("apple pie", "1 in 1 volumes (2000)"),
                     ("zebra", "2 in 2 volumes (1950)")]           # bookkeeping key skipped
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "prefix", "apple", 10)[1] == found[:2]
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "key", "b", 10)[1] == found[2:]
    # the raw layout shows everything, bookkeeping included
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "head", n=1, layout="raw")[1][0][0].startswith("__processed__/")


def test_pivoted_database_keys_are_typed_with_the_year(rocks, tmp_path):
    db = str(tmp_path / "5grams_pivoted.db")
    _pivoted_db(db)
    info = inspection.describe(db)
    assert info["layout"] == "pivoted" and not info["compacted"] and info["records"] == 3
    assert "not compacted" in inspection.format_summary(info)
    _, found = inspection.rows(db, "prefix", "[2011] nuclear", 10)
    assert found == [("[2011] nuclear <UNK>", "9 in 9 volumes"), ("[2011] nuclear power", "4 in 4 volumes")]
    assert inspection.rows(db, "key", "[2011]", 1)[1] == [("[2011] nuclear <UNK>", "9 in 9 volumes")]
    with pytest.raises(ValueError):
        inspection.rows(db, "key", "nuclear", 10)             # no year: not a pivoted key
    with pytest.raises(ValueError):
        inspection.rows(db, "key", "[soon] nuclear", 10)


def test_vocabulary_file_by_rank_token_and_prefix(tmp_path):
    wl = tmp_path / "whitelist.txt"
    wl.write_text("the\nof\napple\napricot\nzebra\n")
    info = inspection.describe(str(wl))
    assert info["kind"] == "text" and info["lines"] == 5
    assert inspection.rows(str(wl), "head", n=2) == ("text", [("1", "the"), ("2", "of")])
    assert inspection.rows(str(wl), "prefix", "ap", 10)[1] == [("3", "apple"), ("4", "apricot")]
    assert inspection.rows(str(wl), "key", "apricot", 10)[1] == [("4", "apricot"), ("5", "zebra")]


def test_missing_and_unopenable_paths_are_reported(rocks, tmp_path):
    assert inspection.describe(str(tmp_path / "nope.db"))["kind"] == "missing"
    (tmp_path / "notadb").mkdir()
    info = inspection.describe(str(tmp_path / "notadb"))
    assert info["kind"] == "rocksdb" and "cannot open" in info["error"]


def test_candidates_list_the_corpus_directory(rocks, tmp_path):
    _ngram_db(tmp_path / "1grams_processed.db")
    (tmp_path / "1grams_processed.db" / "CURRENT").write_text("MANIFEST-000001\n")
    (tmp_path / "1grams_processed.db" / "whitelist.txt").write_text("the\n")
    (tmp_path / "scratch.db").mkdir()                                  # no CURRENT: not a database
    assert inspection.candidates(str(tmp_path)) == [
        ("1grams_processed.db", str(tmp_path / "1grams_processed.db")),
        ("1grams_processed.db/whitelist.txt", str(tmp_path / "1grams_processed.db" / "whitelist.txt"))]
    assert inspection.candidates(None) == [] and inspection.candidates(str(tmp_path / "missing")) == []


def test_cli_inspect_prints_records(rocks, tmp_path):
    import contextlib
    import io
    from lexichron.cli import main
    _ngram_db(tmp_path / "1grams_processed.db")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["inspect", str(tmp_path / "1grams_processed.db"), "--prefix", "apple", "--rows", "5"]) == 0
    out = buf.getvalue()
    assert "3 records" in out and "apple pie" in out and "zebra" not in out
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["inspect", str(tmp_path / "none.db")]) == 1

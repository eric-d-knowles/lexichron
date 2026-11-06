"""The compiled parser must agree with parse_line + _pack_record on every
line, or hand the line back (FALLBACK). Skipped where the extension is not built."""
import gzip
import io
import random
from pathlib import Path

import pytest

parse_cy = pytest.importorskip("ngramprep.ngram_acquire.io.parse_cy")

from ngramprep.ngram_acquire.io.parse import parse_line, prepare_bigrams  # noqa: E402
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate  # noqa: E402
from ngramprep.ngram_acquire import worker as worker_mod  # noqa: E402
from ngramprep.ngram_acquire.worker import _pack_record  # noqa: E402

MODES = {"all": parse_cy.MODE_ALL, "tagged": parse_cy.MODE_TAGGED, "untagged": parse_cy.MODE_UNTAGGED}


def _reference(raw: bytes, pred, bigrams=None, cs=False):
    """What the Python path produces for this raw line (None = rejected,
    'raises' = an exception the worker would log and skip)."""
    try:
        key, rec = parse_line(raw.decode("utf-8"), filter_pred=pred, combined_bigrams=bigrams,
                              bigram_case_sensitive=cs)
        if not (key and rec):
            return None
        return key, _pack_record(rec)
    except Exception:
        return "raises"


def _check(raw: bytes, bigrams=None, cs=False):
    """Fast path output equals the reference, or is FALLBACK. Returns the
    number of fallbacks (0 or 1) so tests can assert on it."""
    fb = 0
    prepared = prepare_bigrams(bigrams, cs)
    tables = parse_cy.bigram_tables(prepared)
    for mode, m in MODES.items():
        pred = make_ngram_type_predicate(mode)
        sfx = tuple(s.encode() for s in pred.suffixes)
        got = parse_cy.parse_packed(raw, m, sfx, tables[0], tables[1], cs)
        if got is parse_cy.FALLBACK:
            fb = 1
            continue
        assert got == _reference(raw, pred, prepared, cs), (raw, mode, bigrams, cs)
    return fb


ORDINARY = [
    b"hello\t2000,100,50\t2001,150,60\n",
    b"the_DET working_VERB class_NOUN in_ADP america_PROPN\t1900,1,1\n",
    b"the working class in america\t1900,1,1\t1900,2,2\n",          # duplicate year kept twice
    b"mixed_NOUN plain\t1950,3,3\n",
    b"a  b\t2000,1,1\n",                                             # double space -> empty token
    b"trailing \t2000,1,1\n",
    b"x_.\t2000,1,1\n",                                              # '.' tag
    b"caf\xc3\xa9 na\xc3\xafve\t2000,1,1\n",                         # UTF-8 in the middle
    b"\xc3\xbcber alles\t2000,1,1\n",                                # non-ASCII first char (not whitespace)
    b"word\t2000,1,1\t\t2001,2,2\n",                                 # empty entry between tabs
    b"word\t2000,1\t2001,2,2\n",                                     # 2-field entry skipped
    b"word\t2000,1,1,1\t2001,2,2\n",                                 # 4-field entry skipped
    b"word\t2000,abc,1\t2001,2,2\n",                                 # non-numeric -> entry skipped
    b"word\t2000,1.5,1\n",                                           # float -> entry skipped -> no entries
    b"word\t,,\n",
    b"word\t2000,,1\n",
    b"word\n",                                                       # no tab
    b"\n", b"", b"   \n", b"\t2000,1,1\n", b" \t2000,1,1\n",
    b"word\t2000,1,1\t\n",                                           # trailing tab stripped
    b"\r\nword\t2000,1,1\r\n",
    b"word\t18446744073709551615,1,1\n",                             # max uint64
    b"word\t0,0,0\n",
    b"word\t2000,1,1 \n",                                            # trailing space stripped
    b"word\t2000,1,1\tx\n",                                          # junk entry skipped
]

UNUSUAL = [
    b"\xc2\xa0word\t2000,1,1\n",                                     # NBSP first: str.strip() removes it
    b"word\t2000,+1,1\n", b"word\t2000, 1,1\n", b"word\t2000,1_0,1\n",   # int() accepts these
    b"word\t\xd9\xa3,1,1\n",                                         # Arabic-Indic digit
    b"word\t18446744073709551616,1,1\n",                             # 2**64: struct.pack raises
    b"word\t99999999999999999999999,1,1\n",
    b"w\xffrd\t2000,1,1\n",                                          # invalid UTF-8
    b"\x1cword\t2000,1,1\n",                                         # \x1c is whitespace for str.strip()
    b"word\t2000,1,1\xc2\xa0\n",
]


def test_ordinary_lines_match_without_fallback():
    for raw in ORDINARY:
        assert _check(raw) == 0, raw


def test_unusual_lines_fall_back_or_match():
    # They may match (if the fast path handles them) but must never differ.
    fallbacks = sum(_check(raw) for raw in UNUSUAL)
    assert fallbacks >= len(UNUSUAL) - 2


def test_random_lines_match():
    rnd = random.Random(7)
    words = ["the", "of", "and", "class", "Mr.", "x_NOUN", "y_VERB", "café", "—", "a_b_c", "_NOUN", "__."]
    for _ in range(5000):
        toks = [rnd.choice(words) for _ in range(rnd.randint(1, 5))]
        sep = rnd.choice([" ", " ", " ", "  "])
        years = "\t".join(f"{rnd.randint(1500, 2019)},{rnd.randint(0, 10**6)},{rnd.randint(0, 10**4)}"
                          for _ in range(rnd.randint(0, 6)))
        line = sep.join(toks) + "\t" + years + rnd.choice(["\n", "\n", "\r\n", ""])
        _check(line.encode("utf-8"))


PAIRS = {"working class", "Middle Class", "new york"}
COMBINING = [
    b"the working class in america\t1900,1,1\n",
    b"The Working Class in America\t1900,1,1\n",                       # capitalized: combines, keeps its case
    b"WORKING CLASS\t1900,1,1\n",
    b"the_DET working_VERB class_NOUN in_ADP\t1900,1,1\n",             # tagged: the first tag survives
    b"The_DET Working_NOUN Class_NOUN\t1900,1,1\n",
    b"middle class working class\t1900,1,1\n",                         # two pairs
    b"lower middle class\t1900,1,1\n",                                 # overlapping: left to right
    b"working  class\t1900,1,1\n",                                     # empty token between: no pair
    b"working class\t1900,1,1\n",
    b"class working\t1900,1,1\n",
    b"New York City\t1900,1,1\n",
    b"new_PROPN york_PROPN\t1900,1,1\n",
    b"_NOUN class_NOUN\t1900,1,1\n",                                   # a bare tag: empty base
    b"working_NOUN_X class\t1900,1,1\n",                               # several underscores: the last is the tag
    b"Working Class\t1990,500,80\t2000,600,90\n",
    b"a b c d e f g\t1900,1,1\n",
]
NON_ASCII_PAIRS = {"caf\u00e9 society", "sch\u00f6ne welt"}
NON_ASCII_LINES = [
    b"caf\xc3\xa9 society\t1900,1,1\n",
    b"Caf\xc3\xa9 Society\t1900,1,1\n",
    b"CAF\xc3\x89 SOCIETY\t1900,1,1\n",                               # an uppercase accented letter
    b"sch\xc3\xb6ne_ADJ welt_NOUN\t1900,1,1\n",
]


def test_combining_matches_python_without_fallback():
    for raw in COMBINING:
        assert _check(raw, PAIRS) == 0, raw
        assert _check(raw, PAIRS, cs=True) == 0, raw
    # a capitalized pair combines by default and not under the switch
    tables = parse_cy.bigram_tables(prepare_bigrams(PAIRS))
    key, _ = parse_cy.parse_packed(COMBINING[1], parse_cy.MODE_ALL, (), tables[0], tables[1], False)
    assert key == "The Working-Class in America"
    tables = parse_cy.bigram_tables(prepare_bigrams(PAIRS, True))
    key, _ = parse_cy.parse_packed(COMBINING[1], parse_cy.MODE_ALL, (), tables[0], tables[1], True)
    assert key == "The Working Class in America"
    key, _ = parse_cy.parse_packed(b"the Middle Class\t1900,1,1", parse_cy.MODE_ALL, (), tables[0], tables[1], True)
    assert key == "the Middle-Class"


def test_non_ascii_pairs_fall_back_only_where_case_folding_needs_python():
    for raw in NON_ASCII_LINES:
        _check(raw, NON_ASCII_PAIRS)                      # may fall back; never differs
        assert _check(raw, NON_ASCII_PAIRS, cs=True) == 0, raw      # exact bytes: no folding needed
    assert _check(NON_ASCII_LINES[2], NON_ASCII_PAIRS) == 1          # CAFÉ: only str.lower() can fold É
    # a non-ASCII token that no pair could start with is dismissed without Python
    assert _check(b"\xc3\xbcber alles working class\t1900,1,1\n", PAIRS) == 0


def test_random_lines_with_pairs_match():
    rnd = random.Random(11)
    words = ["the", "The", "working", "Working", "class", "Class", "CLASS", "middle", "x_NOUN", "class_NOUN",
             "Working_NOUN", "café", "_NOUN", "a_b_c", ""]
    pairs = {"working class", "middle class", "class class", "the working"}
    for _ in range(4000):
        toks = [rnd.choice(words) for _ in range(rnd.randint(1, 6))]
        line = " ".join(toks) + "\t" + f"{rnd.randint(1500, 2019)},{rnd.randint(0, 10**6)},{rnd.randint(0, 10**4)}"
        line += rnd.choice(["\n", "", "\r\n"])
        _check(line.encode("utf-8"), pairs, cs=rnd.random() < 0.3)


def test_worker_combines_on_the_fast_path_like_python(tmp_path, monkeypatch):
    lines = ORDINARY + COMBINING + NON_ASCII_LINES
    payload = gzip.compress(b"".join(lines))

    class Resp:
        def __init__(self): self.raw = io.BytesIO(payload); self.headers = {}
        def close(self): pass

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda url, session=None: Resp())
    from ngramprep.ngram_acquire.spool import read_chunk

    def run(fast, mode, cs):
        monkeypatch.setattr(worker_mod, "_parse_packed", parse_cy.parse_packed if fast else None)
        import tempfile
        d = Path(tempfile.mkdtemp(prefix=f"{'fast' if fast else 'slow'}-{mode}-{cs}-", dir=tmp_path))
        msg, chunks, _, _, _ = worker_mod.process_and_ingest_file(
            "http://example/f.gz", 1, make_ngram_type_predicate(mode), spool_dir=str(d), chunk_entries=1000,
            combined_bigrams=PAIRS | NON_ASCII_PAIRS, bigram_case_sensitive=cs)
        assert msg.startswith("SUCCESS"), msg
        return {k: v for c in chunks for k, v in read_chunk(c)}

    for mode in MODES:
        for cs in (False, True):
            out = run(True, mode, cs)
            assert out == run(False, mode, cs), (mode, cs)
    keys = {k.decode("utf-8") for k in run(True, "all", False)}
    assert "The Working-Class in America" in keys and "Caf\u00e9-Society" in keys and "CAF\u00c9-SOCIETY" in keys
    keys = {k.decode("utf-8") for k in run(True, "all", True)}
    assert "The Working Class in America" in keys and "the working-class in america" in keys


def test_worker_uses_fast_path_and_matches_python(tmp_path, monkeypatch):
    """Whole-worker check: same chunks with the fast path on and off."""
    lines = ORDINARY + UNUSUAL + [b"z_NOUN q_VERB\t1999,5,5\n"] * 3
    payload = gzip.compress(b"".join(lines))

    class Resp:
        def __init__(self): self.raw = io.BytesIO(payload); self.headers = {}
        def close(self): pass

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda url, session=None: Resp())
    from ngramprep.ngram_acquire.spool import read_chunk

    def run(fast, mode):
        monkeypatch.setattr(worker_mod, "_parse_packed", parse_cy.parse_packed if fast else None)
        d = tmp_path / f"{'fast' if fast else 'slow'}-{mode}"
        d.mkdir()
        msg, chunks, nbytes, entries, _lines = worker_mod.process_and_ingest_file(
            "http://example/f.gz", 1, make_ngram_type_predicate(mode), spool_dir=str(d), chunk_entries=1000)
        assert msg.startswith("SUCCESS"), msg
        out = {}
        for c in chunks:
            for k, v in read_chunk(c):
                out[k] = v
        return out, nbytes, entries

    for mode in MODES:
        assert run(True, mode) == run(False, mode), mode

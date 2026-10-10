"""The compiled parser must agree with parse_line + _pack_record on every
line, or hand the line back (FALLBACK). Skipped where the extension is not built."""
import gzip
import io
import random

import pytest

parse_cy = pytest.importorskip("ngramprep.ngram_acquire.io.parse_cy")

from ngramprep.ngram_acquire.io.parse import parse_line  # noqa: E402
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate  # noqa: E402
from ngramprep.ngram_acquire import worker as worker_mod  # noqa: E402
from ngramprep.ngram_acquire.worker import _pack_record  # noqa: E402

MODES = {"all": parse_cy.MODE_ALL, "tagged": parse_cy.MODE_TAGGED, "untagged": parse_cy.MODE_UNTAGGED}


def _reference(raw: bytes, pred):
    """What the Python path produces for this raw line (None = rejected,
    'raises' = an exception the worker would log and skip)."""
    try:
        key, rec = parse_line(raw.decode("utf-8"), filter_pred=pred)
        if not (key and rec):
            return None
        return key, _pack_record(rec)
    except Exception:
        return "raises"


def _check(raw: bytes):
    """Fast path output equals the reference, or is FALLBACK. Returns the
    number of fallbacks (0 or 1) so tests can assert on it."""
    fb = 0
    for mode, m in MODES.items():
        pred = make_ngram_type_predicate(mode)
        sfx = tuple(s.encode() for s in pred.suffixes)
        got = parse_cy.parse_packed(raw, m, sfx)
        if got is parse_cy.FALLBACK:
            fb = 1
            continue
        assert got == _reference(raw, pred), (raw, mode)
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
        msg, chunks, nbytes, entries = worker_mod.process_and_ingest_file(
            "http://example/f.gz", 1, make_ngram_type_predicate(mode), spool_dir=str(d), chunk_entries=1000)
        assert msg.startswith("SUCCESS"), msg
        out = {}
        for c in chunks:
            for k, v in read_chunk(c):
                out[k] = v
        return out, nbytes, entries

    for mode in MODES:
        assert run(True, mode) == run(False, mode), mode

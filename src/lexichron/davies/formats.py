"""How each Davies corpus's files are laid out, and how its tags map onto
the ones the filter knows.

A corpus is distributed in editions. Two matter here: the *tagged* edition
(``wlp/``: one token per line as word, lemma, part of speech) and the
*untagged* one (``text/``: the running text, already tokenized, tokens
separated by spaces). The acquire stage's ``ngram_type`` chooses between
them, as it chooses between tagged and untagged rows of a Google Books
file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Pattern, Tuple

__all__ = ["CorpusFormat", "FORMATS", "corpus_format", "google_tag", "EDITIONS"]

# ngram_type -> the edition it reads
EDITIONS = {"tagged": "wlp", "untagged": "text"}


@dataclass(frozen=True)
class CorpusFormat:
    name: str                            # the corpus key's value: coha
    label: str                           # for people: COHA
    text_language: str                   # spaCy code, for stopwords and lemmas
    archive_glob: Dict[str, str]         # edition -> the archives' file-name pattern
    # One archive member per document, named by this pattern (groups: genre, year, id).
    member: Pattern[str]
    genres: Tuple[str, ...]
    # Tokens that are not text: where the corpus cut ten words out of every
    # two hundred for copyright (one "@" per word removed), and marks of
    # structure. All end a unit of text; none is kept.
    gap: str = "@"
    breaks: FrozenSet[str] = frozenset({"<p>", "<h>"})          # compared in lower case
    # Rows the tagged edition adds that the text does not have.
    padding: FrozenSet[str] = frozenset()
    # A first token that is a leftover of the file's own name.
    leading_junk: FrozenSet[str] = frozenset()


FORMATS: Dict[str, CorpusFormat] = {
    # The Corpus of Historical American English, in the layout of its 2015
    # files: twenty archives per edition, one per decade (text_1810s_kso.zip,
    # wlp_1810s_ksw.zip), each holding one file per document named
    # <genre>_<year>_<id>.txt. Year and genre are read from that name. The
    # tagged files' own first line also gives an id, but for the fiction
    # files it is a different number and is not used.
    "coha": CorpusFormat(
        name="coha", label="COHA", text_language="en",
        archive_glob={"wlp": "wlp_*.zip", "text": "text_*.zip"},
        member=re.compile(r"^(?P<genre>[a-z]+)_(?P<year>\d{4})_(?P<id>\d+)\.txt$"),
        genres=("fic", "mag", "news", "nf"),
        padding=frozenset({"q!"}),                # runs of q! at the end of tagged fiction files
        leading_junk=frozenset({"txt"}),
    ),
}


def corpus_format(name: str) -> CorpusFormat:
    try:
        return FORMATS[str(name).lower()]
    except KeyError:
        raise ValueError(f"no Davies corpus called {name!r} is known; known: {', '.join(sorted(FORMATS))}") from None


# CLAWS7 tags (the tagged editions' third column) to the coarse tags Google
# Books uses and the filter recognizes (ngramprep.ngram_acquire.utils.filters
# DEFAULT_TAGS). Only nouns, verbs, adjectives and adverbs change a lemma;
# the rest say what the token is. The first prefix that matches decides.
# Proper nouns are NOUN, as in Google's files.
_CLAWS_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("appge", "PRON"),                   # my, your
    ("at", "DET"),                       # the, a
    ("cc", "CONJ"), ("cs", "CONJ"),
    ("d", "DET"),                        # this, all, which
    ("ex", "DET"),                       # existential there
    ("fo", "X"), ("fu", "X"), ("fw", "X"),
    ("ge", "X"),                         # 's
    ("i", "ADP"),                        # in, of, with
    ("j", "ADJ"),
    ("m", "NUM"),
    ("n", "NOUN"),                       # nn1, nn2, np1, nnt1, ...
    ("p", "PRON"),
    ("r", "ADV"),                        # rr, rg, rl, rp, rt, ...
    ("to", "ADP"),                       # infinitive marker
    ("uh", "X"),
    ("v", "VERB"),                       # lexical verbs, be, have, do, modals
    ("xx", "ADV"),                       # not, n't
    ("z", "X"),                          # letters of the alphabet
)
_PUNCTUATION = "."
_tag_cache: Dict[str, Optional[str]] = {}


def google_tag(claws: str) -> Optional[str]:
    """The coarse tag for a CLAWS7 tag as the files write it, or None when
    the field is empty. A tag that offers alternatives (``jj_nn1``) is read
    as its first, the tagger's preferred one; the marks of rarity (``@``,
    ``%``) and the digits of a multiword tag (``rr21``) are ignored. A tag
    that names no part of speech (punctuation, ``null``) is ``.``."""
    hit = _tag_cache.get(claws)
    if hit is not None or claws in _tag_cache:
        return hit
    tag = claws.strip().lower().split("_", 1)[0].rstrip("@%")
    if not tag:
        out: Optional[str] = None
    elif tag == "null" or not tag[0].isalpha() or tag == "y":
        out = _PUNCTUATION
    else:
        out = "X"
        for prefix, coarse in _CLAWS_PREFIXES:
            if tag.startswith(prefix):
                out = coarse
                break
    if len(_tag_cache) < 100_000:
        _tag_cache[claws] = out
    return out

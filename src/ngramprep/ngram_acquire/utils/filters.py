from __future__ import annotations
from typing import Callable

DEFAULT_TAGS: tuple[str, ...] = (
    "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "NUM", "CONJ", "X", ".",
)


def _is_tagged(tok: str, suffixes: tuple[str, ...]) -> bool:
    return tok.endswith(suffixes)


def _pred_all_nonempty(line: str) -> bool:
    # Preserve existing behavior for "all": any non-empty, non-whitespace line
    return bool(line) and not line.isspace()


def _pred_tagged(line: str, *, suffixes: tuple[str, ...]) -> bool:
    toks = line.split(" ")
    return toks != [] and all(_is_tagged(t, suffixes) for t in toks)


def _pred_untagged(line: str, *, suffixes: tuple[str, ...]) -> bool:
    toks = line.split(" ")
    return toks != [] and all(not _is_tagged(t, suffixes) for t in toks)


_PREDICATES = {"all", "tagged", "untagged"}


class NgramTypePredicate:
    """predicate(text) -> bool for one n-gram type, carrying its own
    description (``ngram_type``, ``suffixes``) so the compiled parser can
    apply the same filter without calling back into Python (see
    io/parse_cy.pyx).

    It is a class, not a ``functools.partial`` with attributes set on it,
    because it travels to the worker processes: multiprocessing pickles a
    partial with its own reducer, which rebuilds it from the function and its
    arguments and drops everything else. The description was lost on the way
    and every worker ran the Python parser."""

    __slots__ = ("ngram_type", "suffixes")

    def __init__(self, ngram_type: str, suffixes: tuple[str, ...]):
        self.ngram_type = ngram_type
        self.suffixes = tuple(suffixes)

    def __call__(self, line: str) -> bool:
        if self.ngram_type == "tagged":
            return _pred_tagged(line, suffixes=self.suffixes)
        if self.ngram_type == "untagged":
            return _pred_untagged(line, suffixes=self.suffixes)
        return _pred_all_nonempty(line)

    def __reduce__(self):
        return (NgramTypePredicate, (self.ngram_type, self.suffixes))

    def __repr__(self) -> str:
        return f"NgramTypePredicate({self.ngram_type!r})"


def make_ngram_type_predicate(
    ngram_type: str,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> Callable[[str], bool]:
    """
    Return predicate(line) -> bool for 'tagged' | 'untagged' | 'all'.

    Assumptions
    -----------
    - Tokens are separated by single spaces.
    - A token is 'tagged' iff it ends with '_<TAG>' and <TAG> ∈ tags.
      (Base may be any string, including underscores; e.g., '__NOUN' is tagged.)
      Examples:
        '__NOUN'      -> tagged
        'cat_VERB'    -> tagged
        '_NOUN_'      -> NOT tagged (tag not at token end)
    """
    suffixes = tuple(f"_{t}" for t in tags)
    if ngram_type not in _PREDICATES:
        raise ValueError("ngram_type must be 'tagged', 'untagged', or 'all'.")
    return NgramTypePredicate(ngram_type, suffixes)

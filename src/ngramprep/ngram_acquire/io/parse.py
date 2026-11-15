"""Parser for Google Ngrams data format."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["YearFreq", "NgramRecord", "parse_line", "parse_line_checked", "prepare_multigrams", "parse_multigram_entry",
           "Multigrams", "TARGET_ARROW"]


class Multigrams(dict):
    """The word sequences to combine, as :func:`prepare_multigrams` returns
    them. The dict maps each case-folded sequence ("w1 w2 ...", lowercased)
    to the token that replaces it, or None for the default (the sequence's
    own tokens joined by hyphens, in the case they had); ``exact`` maps the
    sequences whose capitalization must match, as written. ``max_words`` is
    the longest sequence in either, so the per-line scan knows how far to
    look."""
    max_words: int = 2

    def __init__(self, entries=(), exact=None, max_words: int = 2):
        super().__init__(entries)
        self.exact: Dict[str, Optional[str]] = dict(exact or {})
        self.max_words = max_words

    def __bool__(self) -> bool:
        return bool(len(self) or self.exact)

    def sequences(self):
        """Every sequence, folded and exact."""
        return list(self) + list(self.exact)


TARGET_ARROW = "=>"


def parse_multigram_entry(entry) -> Tuple[List[str], Optional[str]]:
    """One settings entry -> (words, target). An entry is a string of words
    separated by spaces, optionally followed by ``=> target``: the token
    the sequence becomes. Spaces in the target become hyphens (it must be
    one token); a blank target means the default. A pair or list
    ``(words, target)`` is accepted too."""
    if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], (list, tuple)):
        words, target = [str(w) for w in entry[0]], entry[1]
    else:
        text = str(entry)
        left, arrow, right = text.partition(TARGET_ARROW)
        words = left.split()
        target = right if arrow else None
    if target is not None:
        target = "-".join(str(target).split()) or None
    return words, target


def prepare_multigrams(multigrams, case_sensitive: bool = False, exact=None) -> Optional[Multigrams]:
    """The word sequences to combine, as the parser wants them: each entry's
    words joined by single spaces, mapped to the token they become (None:
    the default). ``multigrams`` match whatever their capitalization in the
    data and are held lowercased; ``exact`` must match as written
    (``case_sensitive`` puts the first list there too, for older callers).
    None for no entries. A one-word entry could never match, so it is
    refused rather than silently ignored."""
    if isinstance(multigrams, Multigrams) and not exact:
        return multigrams
    folded: Dict[str, Optional[str]] = {}
    strict: Dict[str, Optional[str]] = {}
    longest = 2
    for entries, into, fold in ((multigrams, strict if case_sensitive else folded, not case_sensitive),
                                (exact, strict, False)):
        for entry in entries or ():
            words, target = parse_multigram_entry(entry)
            if len(words) < 2:
                raise ValueError(f"a combined multigram must be two or more words separated by spaces, not {entry!r}")
            phrase = " ".join(words)
            into[phrase.lower() if fold else phrase] = target
            longest = max(longest, len(words))
    if not folded and not strict:
        return None
    return Multigrams(folded, strict, longest)


def _max_words(multigrams) -> int:
    n = getattr(multigrams, "max_words", None)
    if n:
        return int(n)
    return max((len(str(e).split(" ")) for e in multigrams), default=2)


def _combine_multigrams_in_ngram(ngram_text: str, multigrams, case_sensitive: bool = False) -> str:
    """
    Replace runs of tokens matching an entry of ``multigrams`` with one token.

    Scans the n-gram tokens (split on single spaces, as the tagged/untagged
    filter does) from left to right; at each token the longest matching
    sequence wins, and the scan continues after it. A sequence on the
    ``exact`` side of a :class:`Multigrams` must match as written; the others
    match whatever the data's capitalization (``case_sensitive`` makes a
    plain set exact, for older callers). The run becomes the entry's target
    token when it has one, else its own tokens joined by hyphens in the case
    they had. For POS-tagged data, strips tags before matching but preserves
    the first token's tag in the output.

    Examples:
        >>> _combine_multigrams_in_ngram("the working class in america", {"working class"})
        'the working-class in america'
        >>> _combine_multigrams_in_ngram("The Working Class", {"working class"})
        'The Working-Class'
        >>> _combine_multigrams_in_ngram("in New York City today", {"new york", "new york city"})
        'in New-York-City today'
        >>> _combine_multigrams_in_ngram("the_DET working_VERB class_NOUN", {"working class"})
        'the_DET working-class_VERB'
        >>> _combine_multigrams_in_ngram("A_DET ._. I_PRON ._.", {"a . i .": "A.I."})
        'A.I._DET'
    """
    tokens = ngram_text.split(" ")
    n = len(tokens)
    if n < 2:
        return ngram_text
    longest = _max_words(multigrams)
    if isinstance(multigrams, Multigrams):
        folded, strict = multigrams, multigrams.exact
    elif case_sensitive:
        folded, strict = {}, multigrams
    else:
        folded, strict = multigrams, {}
    targets_folded = folded if isinstance(folded, dict) else {}
    targets_strict = strict if isinstance(strict, dict) else {}
    # the base forms, without the POS tag an underscore introduces
    bases = [t.rsplit('_', 1)[0] if '_' in t else t for t in tokens]
    lowered = [b.lower() for b in bases]
    result_tokens = []
    i = 0
    while i < n:
        hit = 0
        target = None
        for length in range(min(longest, n - i), 1, -1):
            raw = " ".join(bases[i:i + length])
            if raw in strict:
                hit, target = length, targets_strict.get(raw)
                break
            low = " ".join(lowered[i:i + length])
            if low in folded:
                hit, target = length, targets_folded.get(low)
                break
        if hit:
            first = tokens[i]
            tag = "_" + first.rsplit('_', 1)[1] if '_' in first else ""     # the first token's tag survives
            result_tokens.append((target if target else "-".join(bases[i:i + hit])) + tag)
            i += hit
        else:
            result_tokens.append(tokens[i])
            i += 1
    return " ".join(result_tokens)


prepare_bigrams = prepare_multigrams                        # the earlier names
_combine_bigrams_in_ngram = _combine_multigrams_in_ngram


class YearFreq(TypedDict):
    """Year-level frequency data for an n-gram."""
    year: int
    frequency: int
    document_count: int


class NgramRecord(TypedDict):
    """Complete n-gram record with all year frequencies."""
    frequencies: List[YearFreq]


def parse_line(
        line: str,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        combined_multigrams: Optional[set] = None,
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """
    Parse Google Ngrams line format into structured data.

    Format: "ngram\\tYEAR,FREQ,DOC\\tYEAR,FREQ,DOC..."

    Args:
        line: Raw line from ngrams file
        filter_pred: Optional predicate to filter n-grams by text
                     (applied before parsing frequency data)
        combined_multigrams: Optional word sequences to combine (a plain set, or a
                         Multigrams from prepare_multigrams with its exact-case side)
                         (e.g., {"working class", "new york city"})
                         When found within an n-gram, consecutive tokens matching
                         a listed sequence are replaced with one hyphenated token,
                         reducing the n-gram size by the sequence's length less 1.

    Returns:
        (ngram_text, record) if valid and passes filter, else (None, None)

    Notes:
        - Returns (None, None) for empty/malformed lines or filtered n-grams
        - Skips malformed frequency tuples silently
        - Requires at least one valid frequency tuple to succeed
        - If combined_multigrams is provided, matching runs of tokens
          within the n-gram will be hyphenated (e.g., "the working class in america"
          becomes "the working-class in america")

    Examples:
        >>> parse_line("hello\\t2000,100,50\\t2001,150,60")
        ('hello', {'frequencies': [
            {'year': 2000, 'frequency': 100, 'document_count': 50},
            {'year': 2001, 'frequency': 150, 'document_count': 60}
        ]})
        >>> parse_line("the working class in america\\t2000,100,50", combined_multigrams={"working class"})
        ('the working-class in america', {'frequencies': [{'year': 2000, 'frequency': 100, 'document_count': 50}]})
    """
    key, record, _problem = parse_line_checked(line, filter_pred=filter_pred,
                                               combined_multigrams=combined_multigrams)
    return key, record


def parse_line_checked(
        line: str,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        combined_multigrams: Optional[set] = None,
) -> Tuple[Optional[str], Optional[NgramRecord], bool]:
    """:func:`parse_line`, also saying whether the line held anything that
    could not be read.

    Returns ``(ngram_text, record, problem)``. ``problem`` is True when the
    line is not blank and either has no tab-separated counts at all, or passed
    the filter with a count entry that is not ``year,frequency,documents``
    in whole numbers. Such an entry is left out (the rest of the line is
    kept, as :func:`parse_line` always did), so the n-gram's counts are short
    by it: the caller counts these lines, since a database that silently
    undercounts is worse than one that says so. A line the filter rejects is
    never a problem, whatever its counts look like.
    """
    s = line.strip()
    if not s:
        return None, None, False

    # Split into n-gram text and frequency data
    parts = s.split("\t", 1)
    if len(parts) != 2:
        return None, None, True

    ngram_text, freq_blob = parts

    # Combine the listed word sequences into single hyphenated tokens
    if combined_multigrams is not None:
        ngram_text = _combine_multigrams_in_ngram(ngram_text, combined_multigrams)

    # Apply filter before parsing frequency data (optimization)
    if filter_pred is not None and not filter_pred(ngram_text):
        return None, None, False

    frequencies: List[YearFreq] = []
    problem = False

    # Parse frequency tuples: "year,frequency,document_count"
    for entry in freq_blob.split("\t"):
        parts = entry.split(",")
        if len(parts) != 3:
            problem = True
            continue

        try:
            frequencies.append({
                "year": int(parts[0]),
                "frequency": int(parts[1]),
                "document_count": int(parts[2]),
            })
        except ValueError:
            # A malformed number: the entry is left out, and reported
            problem = True
            continue

    # Only return if we got at least one valid frequency
    if not frequencies:
        return None, None, True

    return ngram_text, {"frequencies": frequencies}, problem

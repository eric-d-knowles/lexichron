"""Parser for Google Ngrams data format."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["YearFreq", "NgramRecord", "parse_line", "prepare_multigrams", "parse_multigram_entry",
           "Multigrams", "TARGET_ARROW"]


class Multigrams(dict):
    """The word sequences to combine, as :func:`prepare_multigrams` returns
    them: a dict from "w1 w2 ..." (the sequence, normalized) to the token
    that replaces it, or None for the default (the sequence's own tokens
    joined by hyphens, in the case they had). It also knows the longest
    sequence's length, so the per-line scan knows how far to look."""
    max_words: int = 2

    def __init__(self, entries=(), max_words: int = 2):
        super().__init__(entries)
        self.max_words = max_words


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


def prepare_multigrams(multigrams, case_sensitive: bool = False) -> Optional[Multigrams]:
    """The word sequences to combine, as the parser wants them: each entry's
    words joined by single spaces, lowercased unless matching is
    case-sensitive, mapped to the token they become (None: the default).
    None for no entries. A one-word entry could never match, so it is
    refused rather than silently ignored."""
    if not multigrams:
        return None
    if isinstance(multigrams, Multigrams):
        return multigrams
    out: Dict[str, Optional[str]] = {}
    longest = 2
    for entry in multigrams:
        words, target = parse_multigram_entry(entry)
        if len(words) < 2:
            raise ValueError(f"a combined multigram must be two or more words separated by spaces, not {entry!r}")
        phrase = " ".join(words)
        out[phrase if case_sensitive else phrase.lower()] = target
        longest = max(longest, len(words))
    return Multigrams(out, longest)


prepare_bigrams = prepare_multigrams          # the earlier name


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
    sequence wins, and the scan continues after it. The run becomes the
    entry's target token when it has one, else its own tokens joined by
    hyphens in the case they had. For POS-tagged data, strips tags before
    matching but preserves the first token's tag in the output. Matching
    ignores case unless ``case_sensitive``. ``multigrams`` is as
    :func:`prepare_multigrams` returns it (lowercased when matching ignores
    case); a plain set of sequences means every target is the default.

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
    targets = multigrams if isinstance(multigrams, dict) else {}
    # the base forms, without the POS tag an underscore introduces
    bases = [t.rsplit('_', 1)[0] if '_' in t else t for t in tokens]
    keys = bases if case_sensitive else [b.lower() for b in bases]
    result_tokens = []
    i = 0
    while i < n:
        hit = 0
        for length in range(min(longest, n - i), 1, -1):
            key = " ".join(keys[i:i + length])
            if key in multigrams:
                hit = length
                break
        if hit:
            first = tokens[i]
            tag = "_" + first.rsplit('_', 1)[1] if '_' in first else ""     # the first token's tag survives
            target = targets.get(key)
            result_tokens.append((target if target else "-".join(bases[i:i + hit])) + tag)
            i += hit
        else:
            result_tokens.append(tokens[i])
            i += 1
    return " ".join(result_tokens)


_combine_bigrams_in_ngram = _combine_multigrams_in_ngram      # the earlier name


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
        multigram_case_sensitive: bool = False,
) -> Tuple[Optional[str], Optional[NgramRecord]]:
    """
    Parse Google Ngrams line format into structured data.

    Format: "ngram\\tYEAR,FREQ,DOC\\tYEAR,FREQ,DOC..."

    Args:
        line: Raw line from ngrams file
        filter_pred: Optional predicate to filter n-grams by text
                     (applied before parsing frequency data)
        combined_multigrams: Optional set of word sequences to combine with hyphens
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
    s = line.strip()
    if not s:
        return None, None

    # Split into n-gram text and frequency data
    parts = s.split("\t", 1)
    if len(parts) != 2:
        return None, None

    ngram_text, freq_blob = parts

    # Combine the listed word sequences into single hyphenated tokens
    if combined_multigrams is not None:
        ngram_text = _combine_multigrams_in_ngram(ngram_text, combined_multigrams, multigram_case_sensitive)

    # Apply filter before parsing frequency data (optimization)
    if filter_pred is not None and not filter_pred(ngram_text):
        return None, None

    frequencies: List[YearFreq] = []

    # Parse frequency tuples: "year,frequency,document_count"
    for entry in freq_blob.split("\t"):
        parts = entry.split(",")
        if len(parts) != 3:
            continue

        try:
            frequencies.append({
                "year": int(parts[0]),
                "frequency": int(parts[1]),
                "document_count": int(parts[2]),
            })
        except ValueError:
            # Skip malformed numeric values
            continue

    # Only return if we got at least one valid frequency
    if not frequencies:
        return None, None

    return ngram_text, {"frequencies": frequencies}
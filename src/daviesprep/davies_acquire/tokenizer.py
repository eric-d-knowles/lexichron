"""Sentence tokenization for Davies corpus text."""
from __future__ import annotations

from typing import Iterator, List, Optional, Set
import re

__all__ = [
    "tokenize_sentences",
    "simple_sentence_tokenizer",
    "combine_multigrams_in_tokens",
    "combine_bigrams_in_tokens",
]


def combine_multigrams_in_tokens(tokens: List[str], combined_multigrams,
                                 case_sensitive: bool = False) -> List[str]:
    """
    Replace runs of tokens matching an entry of combined_multigrams with one token.

    Scans the token list from left to right; at each token the longest
    matching sequence wins and the scan continues after it. The run becomes
    the entry's target token when it has one (a dict of sequence -> target,
    as ngramprep's ``prepare_multigrams`` returns), else its own tokens
    joined by hyphens in the case they had. Matching ignores case unless
    ``case_sensitive``.

    Args:
        tokens: List of tokens (e.g., ["the", "working", "class", "family"])
        combined_multigrams: Sequences to combine, words joined by single spaces
                            (e.g., {"working class", "new york city"}), lowercase unless
                            matching is case-sensitive; a dict maps each to its target token
        case_sensitive: Match the sequences' case exactly

    Examples:
        >>> combine_multigrams_in_tokens(["the", "working", "class"], {"working class"})
        ['the', 'working-class']
        >>> combine_multigrams_in_tokens(["in", "New", "York", "City"], {"new york", "new york city"})
        ['in', 'New-York-City']
        >>> combine_multigrams_in_tokens(["A", ".", "I", "."], {"a . i .": "A.I."})
        ['A.I.']
    """
    if not combined_multigrams or len(tokens) < 2:
        return tokens
    longest = getattr(combined_multigrams, "max_words", None) or \
        max((len(m.split(" ")) for m in combined_multigrams), default=2)
    targets = combined_multigrams if isinstance(combined_multigrams, dict) else {}
    keys = tokens if case_sensitive else [t.lower() for t in tokens]
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        hit = 0
        for length in range(min(longest, n - i), 1, -1):
            key = " ".join(keys[i:i + length])
            if key in combined_multigrams:
                hit = length
                break
        if hit:
            target = targets.get(key)
            result.append(target if target else "-".join(tokens[i:i + hit]))
            i += hit
        else:
            result.append(tokens[i])
            i += 1
    return result


combine_bigrams_in_tokens = combine_multigrams_in_tokens       # the earlier name


def simple_sentence_tokenizer(
    text: str,
    combined_multigrams: Optional[Set[str]] = None,
    multigram_case_sensitive: bool = False,
) -> Iterator[List[str]]:
    """
    Simple sentence tokenizer for Davies corpus text.

    Splits on sentence-ending punctuation (., !, ?) and tokenizes
    each sentence into words.

    Args:
        text: Raw text to tokenize
        combined_multigrams: Optional set of word sequences to combine with hyphens
                         (e.g., {"working class", "middle class"})

    Yields:
        Lists of tokens (words) for each sentence

    Example:
        >>> text = "Hello world. This is a test!"
        >>> list(simple_sentence_tokenizer(text))
        [['Hello', 'world'], ['This', 'is', 'a', 'test']]
        >>> text = "The working class family. The middle class home."
        >>> list(simple_sentence_tokenizer(text, {"working class", "middle class"}))
        [['The', 'working-class', 'family'], ['The', 'middle-class', 'home']]
    """
    # Split into sentences on sentence-ending punctuation
    # Keep the punctuation for now
    sentences = re.split(r'([.!?]+)', text)

    # Combine sentences with their punctuation
    current_sentence = ""
    for i, part in enumerate(sentences):
        if re.match(r'[.!?]+', part):
            # This is punctuation, add to current sentence
            current_sentence += part
            # Yield the sentence
            if current_sentence.strip():
                tokens = tokenize_sentence(current_sentence, combined_multigrams=combined_multigrams,
                                           multigram_case_sensitive=multigram_case_sensitive)
                if len(tokens) >= 2:  # Require at least 2 tokens
                    yield tokens
            current_sentence = ""
        else:
            # This is text, accumulate
            current_sentence += part

    # Don't forget the last sentence if it doesn't end with punctuation
    if current_sentence.strip():
        tokens = tokenize_sentence(current_sentence, combined_multigrams=combined_multigrams,
                                           multigram_case_sensitive=multigram_case_sensitive)
        if len(tokens) >= 2:
            yield tokens


def tokenize_sentence(
    sentence: str,
    combined_multigrams: Optional[Set[str]] = None,
    multigram_case_sensitive: bool = False,
) -> List[str]:
    """
    Tokenize a single sentence into words.

    Uses simple whitespace splitting with some cleanup.
    Filters out COHA markup symbols like @ and other standalone punctuation.

    Args:
        sentence: Sentence text
        combined_multigrams: Optional set of word sequences to combine with hyphens
                         (e.g., {"working class", "middle class"})

    Returns:
        List of tokens

    Example:
        >>> tokenize_sentence("Hello, world!")
        ['Hello', 'world']
        >>> tokenize_sentence("The working class family", {"working class"})
        ['The', 'working-class', 'family']
    """
    # Simple whitespace tokenization
    # Remove leading/trailing whitespace and split
    tokens = sentence.strip().split()

    # Remove empty tokens and clean up
    tokens = [t for t in tokens if t]

    # Remove standalone punctuation and COHA markup symbols (@)
    tokens = [t for t in tokens if not re.match(r'^[.!?,;:"\'\-@]+$', t)]

    # Apply bigram combination if specified
    if combined_multigrams:
        tokens = combine_bigrams_in_tokens(tokens, combined_multigrams, multigram_case_sensitive)

    return tokens


def tokenize_sentences(
    text: str,
    min_tokens: int = 2,
    combined_multigrams: Optional[Set[str]] = None,
    chunk_on: str = "sentence",
    multigram_case_sensitive: bool = False,
) -> Iterator[List[str]]:
    """
    Tokenize text into sentences with word tokens.

    This is the main entry point for tokenization.

    Args:
        text: Raw text to tokenize
        min_tokens: Minimum number of tokens per sentence
        combined_multigrams: Optional set of word sequences to combine with hyphens
                         (e.g., {"working class", "middle class"})
        chunk_on: Chunking mode. "sentence" (default) splits on sentence-ending punctuation;
              "scene" or "document" keeps the entire chunk intact (no punctuation-based splitting).
        multigram_case_sensitive: Match the sequences' case exactly (default: a sequence matches whatever
              its capitalization; the set is then expected in lowercase)

    Yields:
        Lists of tokens for each sentence or chunk (depending on chunk_on)

    Example:
        >>> text = "The cat sat. On the mat!"
        >>> list(tokenize_sentences(text))
        [['The', 'cat', 'sat'], ['On', 'the', 'mat']]
        >>> text = "The working class family. The middle class home."
        >>> list(tokenize_sentences(text, combined_multigrams={"working class", "middle class"}))
        [['The', 'working-class', 'family'], ['The', 'middle-class', 'home']]
    """
    mode = (chunk_on or "sentence").lower()

    if mode in ("scene", "document"):
        tokens = tokenize_sentence(text, combined_multigrams=combined_multigrams,
                                   multigram_case_sensitive=multigram_case_sensitive)
        if len(tokens) >= min_tokens:
            yield tokens
        return

    # Default: sentence-level splitting on . ! ?
    for tokens in simple_sentence_tokenizer(text, combined_multigrams=combined_multigrams,
                                            multigram_case_sensitive=multigram_case_sensitive):
        if len(tokens) >= min_tokens:
            yield tokens

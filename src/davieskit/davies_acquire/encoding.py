"""Encoding utilities for Davies corpus database format.

Davies corpora are stored in the same pivoted format as ngrams:
    Key: (year, sentence_tokens)
    Value: () or dummy value (no frequency data)

This allows the same training code to work for both corpora.
"""
from __future__ import annotations

from typing import List

# Import encoding utilities from ngramkit
from ngramkit.ngram_pivot.encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
)

__all__ = [
    "encode_sentence_key",
    "decode_sentence_key",
    "EMPTY_VALUE",
]

# Davies entries have no frequency data, so use empty value
EMPTY_VALUE = b''


def encode_sentence_key(year: int, tokens: List[str]) -> bytes:
    """
    Encode a sentence into a database key.

    Uses the same format as pivoted ngrams: (year, tokens).

    Args:
        year: Year for this sentence
        tokens: List of word tokens

    Returns:
        Encoded key bytes

    Example:
        >>> tokens = ["the", "cat", "sat"]
        >>> key = encode_sentence_key(1950, tokens)
        >>> isinstance(key, bytes)
        True
    """
    # Join tokens with spaces to create the "ngram"
    sentence_bytes = ' '.join(tokens).encode('utf-8')

    # Use ngram encoding from ngramkit
    return encode_year_ngram_key(year, sentence_bytes)


def decode_sentence_key(key: bytes) -> tuple[int, List[str]]:
    """
    Decode a database key into year and tokens.

    Args:
        key: Encoded key bytes

    Returns:
        Tuple of (year, tokens)

    Example:
        >>> key = encode_sentence_key(1950, ["the", "cat", "sat"])
        >>> year, tokens = decode_sentence_key(key)
        >>> year
        1950
        >>> tokens
        ['the', 'cat', 'sat']
    """
    # Decode using ngram encoding
    year, sentence_bytes = decode_year_ngram_key(key)

    # Split back into tokens
    tokens = sentence_bytes.decode('utf-8').split()

    return year, tokens

"""Sentence tokenization for Davies corpus text."""
from __future__ import annotations

from typing import Iterator, List
import re

__all__ = [
    "tokenize_sentences",
    "simple_sentence_tokenizer",
]


def simple_sentence_tokenizer(text: str) -> Iterator[List[str]]:
    """
    Simple sentence tokenizer for Davies corpus text.

    Splits on sentence-ending punctuation (., !, ?) and tokenizes
    each sentence into words.

    Args:
        text: Raw text to tokenize

    Yields:
        Lists of tokens (words) for each sentence

    Example:
        >>> text = "Hello world. This is a test!"
        >>> list(simple_sentence_tokenizer(text))
        [['Hello', 'world'], ['This', 'is', 'a', 'test']]
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
                tokens = tokenize_sentence(current_sentence)
                if len(tokens) >= 2:  # Require at least 2 tokens
                    yield tokens
            current_sentence = ""
        else:
            # This is text, accumulate
            current_sentence += part

    # Don't forget the last sentence if it doesn't end with punctuation
    if current_sentence.strip():
        tokens = tokenize_sentence(current_sentence)
        if len(tokens) >= 2:
            yield tokens


def tokenize_sentence(sentence: str) -> List[str]:
    """
    Tokenize a single sentence into words.

    Uses simple whitespace splitting with some cleanup.
    Filters out COHA markup symbols like @ and other standalone punctuation.

    Args:
        sentence: Sentence text

    Returns:
        List of tokens

    Example:
        >>> tokenize_sentence("Hello, world!")
        ['Hello', 'world']
    """
    # Simple whitespace tokenization
    # Remove leading/trailing whitespace and split
    tokens = sentence.strip().split()

    # Remove empty tokens and clean up
    tokens = [t for t in tokens if t]

    # Remove standalone punctuation and COHA markup symbols (@)
    tokens = [t for t in tokens if not re.match(r'^[.!?,;:"\'\-@]+$', t)]

    return tokens


def tokenize_sentences(
    text: str,
    min_tokens: int = 2,
) -> Iterator[List[str]]:
    """
    Tokenize text into sentences with word tokens.

    This is the main entry point for tokenization.

    Args:
        text: Raw text to tokenize
        min_tokens: Minimum number of tokens per sentence

    Yields:
        Lists of tokens for each sentence

    Example:
        >>> text = "The cat sat. On the mat!"
        >>> list(tokenize_sentences(text))
        [['The', 'cat', 'sat'], ['On', 'the', 'mat']]
    """
    for tokens in simple_sentence_tokenizer(text):
        if len(tokens) >= min_tokens:
            yield tokens

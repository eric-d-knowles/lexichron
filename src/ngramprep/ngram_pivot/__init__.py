# ngram_pivot/__init__.py
"""Year-first ("pivoted") n-gram databases: key encoding and streaming reads.

The pivoted database itself is written by the filter stage (``lexichron
filter`` / ``lexichron.stages.filter.filter_ngrams``) in the same pass that
filters the n-grams; this package holds what readers of it need.
"""

from .stream import stream_year_ngrams, stream_year_range, get_ngram_stats
from .encoding import (
    encode_year_ngram_key,
    decode_year_ngram_key,
    year_prefix,
    encode_year_stats,
    decode_year_stats,
)

__all__ = [
    # Query interface
    "stream_year_ngrams",
    "stream_year_range",
    "get_ngram_stats",
    # Low-level encoding utilities
    "encode_year_ngram_key",
    "decode_year_ngram_key",
    "year_prefix",
    "encode_year_stats",
    "decode_year_stats",
]

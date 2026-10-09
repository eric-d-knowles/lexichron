"""URL and pattern construction for Google Ngrams repository."""
from __future__ import annotations

import re
from typing import Tuple

BASE_URL = "https://storage.googleapis.com/books/ngrams/books"
SUPPORTED_RELEASES = ("20200217",)

__all__ = ["BASE_URL", "SUPPORTED_RELEASES", "build_location_info"]


def build_location_info(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
) -> Tuple[str, re.Pattern[str]]:
    """
    Build the listing URL and filename pattern for Google Ngrams files.

    Returns a URL that can be fetched (either HTML index or GCS XML listing)
    and a regex pattern that matches against filenames.

    Supports the V3 release (20200217): GCS XML API listing with files named
    "{n}-{shard}-of-{total}.gz", one line per n-gram with all years on it.

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us", "fre")

    Returns:
        Tuple of (listing_url, filename_pattern)
        - listing_url: URL to fetch for file discovery
        - filename_pattern: Regex that matches just the filename (not full URL)

    Raises:
        ValueError: If parameters are invalid or the release is not supported

    Examples:
        >>> url, pattern = build_location_info(1, "20200217", "eng")
        >>> url
        'https://books.storage.googleapis.com/?prefix=ngrams/books/20200217/eng/1-'
        >>> pattern.match("1-00012-of-00024.gz")
        <re.Match object...>
    """
    # Validate ngram_size
    if ngram_size not in (1, 2, 3, 4, 5):
        raise ValueError(f"ngram_size must be 1-5, got {ngram_size}")

    # Validate release ID format
    if not re.fullmatch(r"\d{8}", repo_release_id):
        raise ValueError(
            f"repo_release_id must be 8-digit YYYYMMDD, got {repo_release_id!r}"
        )

    # Validate corpus ID format
    if not re.fullmatch(r"[A-Za-z0-9-]+", repo_corpus_id):
        raise ValueError(
            f"repo_corpus_id must contain only [A-Za-z0-9-], got {repo_corpus_id!r}"
        )

    # Only the 2020 release (V3) is supported. Earlier releases use different
    # file formats (2012: one line per n-gram-year; 2009: zipped CSV) that the
    # worker and parser do not handle.
    if repo_release_id not in SUPPORTED_RELEASES:
        raise ValueError(
            f"Unsupported release {repo_release_id!r}. "
            f"Supported: {', '.join(SUPPORTED_RELEASES)} (Google Books Ngrams V3)."
        )

    # V3: GCS bucket XML API listing; files named "1-00012-of-00024.gz"
    prefix = f"ngrams/books/{repo_release_id}/{repo_corpus_id}/{ngram_size}-"
    listing_url = f"https://books.storage.googleapis.com/?prefix={prefix}"
    filename_pattern = re.compile(rf"^{ngram_size}-\d{{5}}-of-\d{{5}}\.gz$")

    return listing_url, filename_pattern
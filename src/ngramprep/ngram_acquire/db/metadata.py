"""Metadata tracking for processed ngram files in RocksDB."""
from __future__ import annotations

import logging

import rocks_shim as rs

logger = logging.getLogger(__name__)

PROCESSED_PREFIX = b"__processed__/"

__all__ = [
    "PROCESSED_PREFIX",
    "processed_key",
    "is_file_processed",
]


def processed_key(filename: str) -> bytes:
    """Generate metadata key for a processed file."""
    return PROCESSED_PREFIX + filename.encode("utf-8")


def is_file_processed(db: rs.DB, filename: str) -> bool:
    """
    Check if a file has been processed (O(1) lookup).
    
    Args:
        db: RocksDB database handle
        filename: Name of file to check
        
    Returns:
        True if file has been marked as processed
    """
    try:
        return db.get(processed_key(filename)) is not None
    except Exception:
        logger.exception("Failed to check processed status for %s", filename)
        return False

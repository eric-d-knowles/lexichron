"""Metadata tracking for processed ngram files in RocksDB."""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import rocks_shim as rs

logger = logging.getLogger(__name__)

PROCESSED_PREFIX = b"__processed__/"

__all__ = [
    "PROCESSED_PREFIX",
    "processed_key",
    "processed_value",
    "is_file_processed",
    "processed_stats",
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


def processed_value(*, entries: int, uncompressed_bytes: int, chunks: int = 0) -> bytes:
    """Marker value recording what the file contributed (older databases hold
    ``b"1"``, which still counts as processed but carries no figures)."""
    return json.dumps({"entries": int(entries), "bytes": int(uncompressed_bytes),
                       "chunks": int(chunks)}).encode("utf-8")


def processed_stats(db: rs.DB, filename: str) -> Optional[Dict[str, int]]:
    """Figures stored with a processed file's marker: ``{"entries", "bytes",
    "chunks"}``. ``{}`` for an old marker without figures; None if the file
    has not been processed."""
    try:
        raw = db.get(processed_key(filename))
    except Exception:
        logger.exception("Failed to read processed marker for %s", filename)
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, UnicodeDecodeError):
        return {}

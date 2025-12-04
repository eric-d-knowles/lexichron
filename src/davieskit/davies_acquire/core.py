"""Main entry point for Davies corpus acquisition pipeline."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = ["ingest_davies_corpus"]


def ingest_davies_corpus(
    corpus_name: str,
    corpus_path: str,
    db_path: str,
    overwrite_db: bool = True,
    workers: Optional[int] = None,
) -> None:
    """
    Main pipeline: read Davies corpus text files and ingest into RocksDB.

    Orchestrates the complete Davies acquisition workflow:
    1. Discovers text files in corpus directory
    2. Opens/creates RocksDB in pivoted format
    3. Reads and tokenizes text files
    4. Writes sentences directly to pivoted DB: (year, tokens) -> ()

    Args:
        corpus_name: Name of corpus (e.g., "COHA", "COCA")
        corpus_path: Path to corpus directory (containing text/ subdirectory)
        db_path: Path for output database
        overwrite_db: If True, remove existing database before starting
        workers: Number of concurrent workers (default: cpu_count - 1)
    """
    logger.info("Starting Davies corpus acquisition pipeline")

    start_time = datetime.now()

    # Validate corpus path
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise ValueError(f"Corpus path does not exist: {corpus_path}")

    text_dir = corpus_path / "text"
    if not text_dir.exists():
        raise ValueError(f"Text directory not found: {text_dir}")

    # Handle existing database
    db_path = Path(db_path)
    if overwrite_db and db_path.exists():
        logger.info("Removing existing database for fresh start")
        # Use safe cleanup from ngramkit
        from ngramkit.ngram_acquire.utils.cleanup import safe_db_cleanup
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine worker count
    if workers is None:
        cpu_count = os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

    logger.info(f"Processing {corpus_name} corpus")
    logger.info(f"Corpus path: {corpus_path}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Workers: {workers}")

    # TODO: Implement the actual pipeline
    # 1. Discover text files
    # 2. Open database
    # 3. Process files in parallel
    # 4. Write to database in pivoted format

    raise NotImplementedError("Davies acquisition pipeline not yet implemented")

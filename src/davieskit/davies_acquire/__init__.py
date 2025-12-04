"""
Davies corpus acquisition pipeline for ingesting local corpus files.

This module provides a pipeline for reading, parsing, and ingesting
text data from Mark Davies' corpora (COHA, COCA, etc.) into RocksDB.

Main entry point:
    ingest_davies_corpus() - Full pipeline orchestration

Key components:
    - core: Main pipeline orchestration
    - reader: Text file reading and parsing
    - tokenizer: Sentence tokenization
    - writer: Batched database writes
"""

from .core import ingest_davies_corpus
from .config import (
    CorpusConfig,
    AcquisitionConfig,
    build_coha_config,
    build_db_path,
)

__all__ = [
    "ingest_davies_corpus",
    "CorpusConfig",
    "AcquisitionConfig",
    "build_coha_config",
    "build_db_path",
]

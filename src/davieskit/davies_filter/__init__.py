"""
Davies corpus filtering pipeline for preprocessing and cleaning.

This module provides filtering and preprocessing capabilities for
Davies corpus data stored in RocksDB, preparing it for word2vec training.

Main entry point:
    build_filtered_db() - Full filtering pipeline

Key components:
    - config: Configuration for filtering options
    - processor: Apply filters (lowercase, lemmatization, stopwords)
    - pipeline: Parallel processing orchestration
"""

__all__ = []

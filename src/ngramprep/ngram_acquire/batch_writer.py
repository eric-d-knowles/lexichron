"""Ingest spooled chunk files into RocksDB, one chunk at a time."""
from __future__ import annotations

import logging
import os
from typing import Iterable, List

import rocks_shim as rs

from ngramprep.ngram_acquire.db.metadata import processed_key
from ngramprep.ngram_acquire.spool import read_chunk

logger = logging.getLogger(__name__)

__all__ = ["ChunkIngestor"]


class ChunkIngestor:
    """
    Writes a worker's chunk files into the database and marks the file done.

    Each chunk is read from disk, written in one batch with ``merge`` (so a
    key that already exists, e.g. from another chunk of the same shard, has
    its per-year counts summed by the packed24 merge operator), and deleted.
    Only one chunk is in memory at a time.

    When all chunks of a shard are in, the shard's resume marker is written
    and the memtables are flushed, so a completed shard is on disk together
    with its marker even if the process is killed afterwards.
    """

    def __init__(self, db: rs.DB, *, disable_wal: bool = True) -> None:
        self.db = db
        self.disable_wal = disable_wal
        self.total_entries_written = 0
        self.write_batches = 0
        self.files_completed = 0

    def ingest_file(self, filename: str, chunk_paths: Iterable[str]) -> int:
        """Ingest all chunks of one shard, then mark it processed and persist."""
        written = 0
        for path in chunk_paths:
            written += self._ingest_chunk(path)

        with self.db.write_batch(disable_wal=self.disable_wal, sync=False) as wb:
            wb.put(processed_key(filename), b"1")
        self.db.finalize_bulk()

        self.files_completed += 1
        logger.info(
            "Completed %s: %s entries written (persisted to disk)", filename, f"{written:,}"
        )
        return written

    def _ingest_chunk(self, path: str) -> int:
        n = 0
        try:
            with self.db.write_batch(disable_wal=self.disable_wal, sync=False) as wb:
                for key, value in read_chunk(path):
                    wb.merge(key, value)
                    n += 1
        except Exception:
            logger.error("DB write error ingesting chunk %s; aborting to prevent data loss", path)
            raise
        self.total_entries_written += n
        self.write_batches += 1
        logger.info("Ingested chunk %s: %s entries", os.path.basename(path), f"{n:,}")
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Could not remove ingested chunk %s: %s", path, exc)
        return n

    @staticmethod
    def discard(chunk_paths: Iterable[str]) -> None:
        """Remove chunk files that will not be ingested."""
        for path in chunk_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def get_stats(self) -> tuple[int, int]:
        """Return (total_entries_written, write_batches)."""
        return self.total_entries_written, self.write_batches

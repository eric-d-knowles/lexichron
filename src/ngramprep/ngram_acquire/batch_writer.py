"""Ingest spooled files into RocksDB and mark their shard done.

Everything here is written with *merge*: a key that is already in the
database has its per-year counts summed by the packed24 operator. That is
what makes a shard's records add up with every other shard's, and it is also
why a shard must never be ingested twice: the second time would be summed
too. The marker that says "this shard is in" therefore has to become durable
together with the shard's records, never before and never after.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Optional

import rocks_shim as rs

from ngramprep.ngram_acquire.db.metadata import PROCESSED_PREFIX, processed_key, processed_value
from ngramprep.ngram_acquire.spool import read_chunk

logger = logging.getLogger(__name__)

__all__ = ["ChunkIngestor"]


class ChunkIngestor:
    """
    Writes a worker's chunk files into the database and marks the file done.

    A shard spooled as SST files (``.sst``, see SstChunkWriter) is adopted
    with a single ingest call that also carries its marker, written as a
    one-key SST file: RocksDB applies the whole call as one manifest edit, so
    the records and the marker are in the database together or not at all,
    wherever the process is killed. Nothing passes through a memtable and
    nothing needs flushing.

    A shard spooled as plain chunk files is replayed: each chunk is read from
    disk, written in one batch with ``merge``, and deleted, one chunk in
    memory at a time. When all chunks are in, the marker is written and the
    memtables are flushed. The marker is durable with the records as long as
    no memtable is flushed while the shard is being replayed, which holds for
    a shard smaller than the write buffers.
    """

    def __init__(self, db: rs.DB, *, disable_wal: bool = True,
                 marker_prefix: bytes = PROCESSED_PREFIX) -> None:
        self.db = db
        self.disable_wal = disable_wal
        self.marker_prefix = marker_prefix
        self.total_entries_written = 0
        self.write_batches = 0
        self.files_completed = 0

    def ingest_file(self, filename: str, chunk_paths: Iterable[str], *,
                    uncompressed_bytes: int = 0, extra: Optional[dict] = None,
                    entries: Optional[int] = None) -> int:
        """Ingest all chunks of one shard and mark it processed (recording
        its entry count, uncompressed size and any ``extra`` figures), the
        marker becoming durable with the records. A chunk is replayed through
        a write batch, or, when it is an SST file (``.sst``, see
        SstChunkWriter), ingested as a file; the caller passes ``entries``
        for those, since a file ingest does not count them."""
        chunk_paths = list(chunk_paths)
        t0 = time.perf_counter()
        written = 0
        sst = [p for p in chunk_paths if p.endswith(".sst")]
        for path in chunk_paths:
            if not path.endswith(".sst"):
                written += self._ingest_chunk(path)
        if entries is not None:
            # A file ingest counts nothing, so the caller's figure stands for
            # the SST files; replayed chunks were counted as they went in.
            if sst:
                self.total_entries_written += max(int(entries) - written, 0)
            written = int(entries)
        key = self.marker_prefix + filename.encode("utf-8")
        value = processed_value(entries=written, uncompressed_bytes=uncompressed_bytes,
                                chunks=len(chunk_paths), extra=extra)

        if sst and len(sst) == len(chunk_paths):
            self._ingest_sst(sst, marker=(key, value))
            how = (f"adopted {len(sst)} SST file{'s' if len(sst) != 1 else ''} and the marker "
                   f"in one ingest call")
        else:
            if sst:                               # a mixed shard: no writer produces one
                self._ingest_sst(sst)
            with self.db.write_batch(disable_wal=self.disable_wal, sync=False) as wb:
                wb.put(key, value)
            self.db.finalize_bulk()
            how = (f"replayed {len(chunk_paths)} chunk file{'s' if len(chunk_paths) != 1 else ''}, "
                   f"wrote the marker and flushed" if chunk_paths else "nothing to ingest; wrote the marker and flushed")

        self.files_completed += 1
        # One entry per unit, saying what was done to make it durable.
        logger.info("Ingested %s: %s entries; %s (%.2fs)", filename, f"{written:,}", how, time.perf_counter() - t0)
        return written

    def _ingest_chunk(self, path: str) -> int:
        n = 0
        try:
            with self.db.write_batch(disable_wal=self.disable_wal, sync=False) as wb:
                if hasattr(wb, "merge_batch"):
                    # One call into C++ for the whole chunk (rocks-shim >= merge_batch).
                    items = list(read_chunk(path))
                    wb.merge_batch(items)
                    n = len(items)
                    del items
                else:
                    for key, value in read_chunk(path):
                        wb.merge(key, value)
                        n += 1
        except Exception:
            logger.error("DB write error ingesting chunk %s; aborting to prevent data loss", path)
            raise
        self.total_entries_written += n
        self.write_batches += 1
        logger.debug("Replayed chunk %s: %s entries", os.path.basename(path), f"{n:,}")
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Could not remove ingested chunk %s: %s", path, exc)
        return n

    def _ingest_sst(self, paths: List[str], marker: Optional[tuple] = None) -> None:
        # All of a unit's files in one call (one manifest write); RocksDB
        # accepts files that overlap one another in key space, placing them
        # in level 0. The files are moved, not copied. With ``marker`` (key,
        # value) the unit's marker rides in the same call as a one-key file
        # holding a put: the marker is never a merge operand, which the
        # counts' operator would try to sum.
        paths = list(paths)
        if marker is not None:
            from ngramprep.common_db.api import rs    # the store in use (or a test double)
            marker_path = paths[0] + ".marker.sst"
            writer = rs.SstFileWriter(compression="none")
            writer.open(marker_path)
            writer.put(marker[0], marker[1])
            writer.finish()
            paths.append(marker_path)
        try:
            self.db.ingest(paths, move=True)
        except Exception:
            logger.error("SST ingest error for %s; aborting to prevent data loss", paths)
            if marker is not None:
                try:
                    os.remove(paths[-1])
                except OSError:
                    pass
            raise
        self.write_batches += 1
        logger.debug("Ingested %d SST file(s): %s", len(paths), ", ".join(os.path.basename(p) for p in paths))
        for path in paths:
            if os.path.exists(path):              # a store that copied rather than moved
                try:
                    os.remove(path)
                except OSError:
                    pass

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

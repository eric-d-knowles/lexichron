"""On-disk spool of parsed entries, written by workers and read by the parent.

A worker parses a shard into fixed-size *chunks* and writes each chunk to a
file as soon as it is full, so the worker never holds more than one chunk in
memory. Chunks are written as ``.part`` files and only renamed to ``.chunk``
once the whole shard has been parsed successfully; a failed attempt removes
its ``.part`` files. The parent therefore never sees a chunk from a shard that
may still be retried, which keeps merge-based ingestion from double counting.

Chunk file layout (little-endian)::

    magic   b"NGC1"
    repeat: u32 key_len, u32 value_len, key bytes, value bytes
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

__all__ = ["MAGIC", "ChunkWriter", "read_chunk", "chunk_entry_count"]

MAGIC = b"NGC1"
_HDR = struct.Struct("<II")


class ChunkWriter:
    """Accumulates (key, packed value) pairs and spills them to chunk files.

    Keys that repeat within a chunk are merged by ``merge_fn`` (summing year
    counts); keys that repeat across chunks are merged later by the database.
    """

    def __init__(
        self,
        spool_dir: str | os.PathLike,
        file_tag: str,
        chunk_entries: int,
        merge_fn,
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.file_tag = file_tag
        self.chunk_entries = max(1, int(chunk_entries))
        self.merge_fn = merge_fn
        self.pending: Dict[object, bytes] = {}
        self.part_paths: List[Path] = []
        self.entries_total = 0
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    def add(self, key, value: bytes) -> None:
        """``key`` may be str (encoded as UTF-8 on spill) or bytes."""
        existing = self.pending.get(key)
        if existing is None:
            self.pending[key] = value
            if len(self.pending) >= self.chunk_entries:
                self._spill()
        else:
            self.pending[key] = self.merge_fn(existing, value)

    def _spill(self) -> None:
        if not self.pending:
            return
        idx = len(self.part_paths)
        path = self.spool_dir / f"{self.file_tag}.{idx:05d}.part"
        pack = _HDR.pack
        with open(path, "wb") as fh:
            fh.write(MAGIC)
            buf: List[bytes] = []
            n = 0
            for key, value in self.pending.items():
                kb = key if isinstance(key, bytes) else key.encode("utf-8")
                buf += (pack(len(kb), len(value)), kb, value)
                n += 1
                if n >= 8192:                     # one write per few thousand records
                    fh.write(b"".join(buf))
                    buf = []
                    n = 0
            if buf:
                fh.write(b"".join(buf))
        self.part_paths.append(path)
        self.entries_total += len(self.pending)
        self.pending = {}

    def finish(self) -> List[str]:
        """Spill the remainder and publish all parts as ``.chunk`` files.

        Returns the chunk paths in order. After this call the writer is empty.
        """
        self._spill()
        published: List[str] = []
        for part in self.part_paths:
            final = part.with_suffix(".chunk")
            os.replace(part, final)
            published.append(str(final))
        self.part_paths = []
        return published

    def discard(self) -> None:
        """Remove any parts written so far (used when a shard is retried)."""
        for part in self.part_paths:
            try:
                os.remove(part)
            except FileNotFoundError:
                pass
        self.part_paths = []
        self.pending = {}
        self.entries_total = 0


def read_chunk(path: str | os.PathLike) -> Iterator[Tuple[bytes, bytes]]:
    """Yield (key, value) pairs from a chunk file."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"not a chunk file: {path}")
        hdr = _HDR
        while True:
            h = fh.read(hdr.size)
            if not h:
                return
            if len(h) != hdr.size:
                raise ValueError(f"truncated chunk file: {path}")
            klen, vlen = hdr.unpack(h)
            key = fh.read(klen)
            value = fh.read(vlen)
            if len(key) != klen or len(value) != vlen:
                raise ValueError(f"truncated chunk file: {path}")
            yield key, value


def chunk_entry_count(path: str | os.PathLike) -> int:
    """Count entries without materializing values (used for reporting/tests)."""
    n = 0
    for _ in read_chunk(path):
        n += 1
    return n


class SstChunkWriter(ChunkWriter):
    """Like :class:`ChunkWriter`, but each chunk is written as a sorted SST
    file of *merge* operands (rocks-shim >= 0.8), so the parent ingests it as
    a file instead of replaying its records through a write batch. Meant for
    stages whose output is many times larger than their input: the parent's
    single-writer path would otherwise be the bottleneck. Duplicate keys
    across chunks and units are summed by the database's merge operator at
    compaction, exactly as with replayed chunks.
    """

    def __init__(self, spool_dir, file_tag: str, chunk_entries: int, merge_fn, *,
                 compression: str = "zstd") -> None:
        super().__init__(spool_dir, file_tag, chunk_entries, merge_fn)
        self.compression = compression

    def _spill(self) -> None:
        if not self.pending:
            return
        from ngramprep.common_db.api import rs       # the store in use (or a test double)
        idx = len(self.part_paths)
        path = self.spool_dir / f"{self.file_tag}.{idx:05d}.sst.part"
        writer = rs.SstFileWriter(compression=self.compression)
        writer.open(str(path))
        for key in sorted(self.pending):            # SST files are written in key order
            kb = key if isinstance(key, bytes) else key.encode("utf-8")
            writer.merge(kb, self.pending[key])
        writer.finish()
        self.part_paths.append(path)
        self.entries_total += len(self.pending)
        self.pending = {}

    def finish(self) -> List[str]:
        """Merge the parts into one SST file (rocks-shim >= 0.9) and publish
        it. The parent then ingests one file per unit: a compaction holds
        every level-0 file open at once, so their number must stay in the
        hundreds. Without merge support the parts are published as they are."""
        self._spill()
        if not self.part_paths:
            return []
        from ngramprep.common_db.api import rs
        merge = getattr(rs, "merge_sst_files", None)
        if merge is not None and len(self.part_paths) > 1:
            merged = self.spool_dir / f"{self.file_tag}.sst.part"
            self.entries_total = int(merge([str(p) for p in self.part_paths], str(merged),
                                           compression=self.compression))
            for part in self.part_paths:
                try:
                    os.remove(part)
                except OSError:
                    pass
            self.part_paths = [merged]
        published: List[str] = []
        for part in self.part_paths:
            final = part.with_name(part.name[:-len(".part")])      # x.00001.sst.part -> x.00001.sst
            os.replace(part, final)
            published.append(str(final))
        self.part_paths = []
        return published

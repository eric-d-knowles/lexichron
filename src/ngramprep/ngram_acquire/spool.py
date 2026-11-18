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

:class:`SstChunkWriter` spools the same records as sorted RocksDB SST files
instead, which the parent ingests as files; the ``Checked`` writers are the
ones acquisition uses, and add its checks on every value (see
:class:`_Checked`).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

__all__ = ["MAGIC", "ChunkWriter", "SstChunkWriter", "CheckedChunkWriter", "CheckedSstChunkWriter",
           "check_values", "read_chunk", "chunk_entry_count"]

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
    a file instead of replaying its records through a write batch: the
    parent's single-writer path is otherwise the bottleneck, for a stage
    whose output is many times its input (the filter's pivot) and for one
    with many small units (acquisition's shards). Duplicate keys across
    chunks and units are summed by the database's merge operator at
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
            # The operator is named, not left to the store's default: the
            # parts hold packed24 operands, and a key met in several parts
            # must be summed the way the database will sum it.
            self.entries_total = int(merge([str(p) for p in self.part_paths], str(merged),
                                           compression=self.compression, merge_operator="packed24"))
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


class _Checked:
    """Mixin for the acquisition writers: checks every value before it is
    spooled and keeps the totals a run is later verified against.

    Values arrive as the parser packed them, ``(year, frequency, documents)``
    in little-endian uint64. Two things are done to each, in vectorized
    batches of ``STAGE`` values:

    * its frequencies and document counts are added to ``match_total`` and
      ``volume_total``. Summing repeated keys never changes these totals, so
      what the finished database holds must equal the totals of everything
      spooled into it (see :mod:`ngramprep.ngram_acquire.verify`);
    * its years are checked to be strictly ascending. The packed24 merge
      operator assumes this and checks nothing in a release build: a value
      with years out of order or repeated would be summed wrongly without
      any error. Such a value is rewritten in order (``reordered`` counts
      them), so the assumption is enforced here instead of trusted.

    Keys are encoded to bytes on arrival, so that one n-gram is one key
    however the parser handed it over, and so that sorting them is sorting
    what the database compares.
    """

    STAGE = 4096

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stage: List[Tuple[bytes, bytes]] = []
        self.match_total = 0
        self.volume_total = 0
        self.reordered = 0

    def add(self, key, value: bytes) -> None:
        self._stage.append((key if isinstance(key, bytes) else key.encode("utf-8"), value))
        if len(self._stage) >= self.STAGE:
            self._drain()

    def _drain(self) -> None:
        stage, self._stage = self._stage, []
        if not stage:
            return
        values = [v for _k, v in stage]
        match, volume, bad = check_values(values)
        self.match_total += match
        self.volume_total += volume
        add = super().add
        if bad:
            self.reordered += len(bad)
            wrong = set(bad)
            for i, (key, value) in enumerate(stage):
                add(key, self.merge_fn(value, b"") if i in wrong else value)
        else:
            for key, value in stage:
                add(key, value)

    def settle(self) -> None:
        """Check and pass on everything staged so far, so that the writer's
        own figures (``pending``, ``part_paths``, the totals) are current."""
        self._drain()

    @property
    def kept(self) -> int:
        """Values taken so far, before repeated keys are summed across parts."""
        return self.entries_total + len(self.pending) + len(self._stage)

    def finish(self) -> List[str]:
        self._drain()
        return super().finish()

    def discard(self) -> None:
        self._stage = []
        super().discard()


def check_values(values: List[bytes]) -> Tuple[int, int, List[int]]:
    """Totals and year order of a batch of packed values.

    Returns ``(frequency total, document-count total, indices of the values
    whose years are not strictly ascending)``. Raises ValueError for a value
    that is empty or not a whole number of 24-byte triplets: nothing
    downstream could read it.
    """
    import numpy as np

    n = len(values)
    if not n:
        return 0, 0, []
    sizes = np.fromiter(map(len, values), dtype=np.int64, count=n)
    if (sizes == 0).any() or (sizes % 24).any():
        i = int(np.flatnonzero((sizes == 0) | (sizes % 24 != 0))[0])
        raise ValueError(f"packed value of {int(sizes[i])} bytes is not a sequence of 24-byte "
                         f"(year, frequency, documents) triplets")
    rows = np.frombuffer(b"".join(values), dtype="<u8").reshape(-1, 3)
    # Python integers for the totals: a corpus total can exceed what a batch
    # sum is safe to hold only in theory, but the ledger must never wrap.
    match = int(rows[:, 1].sum(dtype=np.uint64))
    volume = int(rows[:, 2].sum(dtype=np.uint64))
    years = rows[:, 0]
    ascending = years[1:] > years[:-1]
    ends = np.cumsum(sizes // 24)                 # row index one past each value's last triplet
    ascending[ends[:-1] - 1] = True               # the step from one value to the next is not a step in time
    if ascending.all():
        return match, volume, []
    rows_bad = np.flatnonzero(~ascending) + 1     # a row whose year is not above the one before it
    return match, volume, sorted(set(np.searchsorted(ends, rows_bad, side="right").tolist()))


class CheckedChunkWriter(_Checked, ChunkWriter):
    """:class:`ChunkWriter` with the acquisition checks (see :class:`_Checked`)."""


class CheckedSstChunkWriter(_Checked, SstChunkWriter):
    """:class:`SstChunkWriter` with the acquisition checks (see :class:`_Checked`)."""

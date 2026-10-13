"""An in-memory stand-in for rocks_shim with packed24 merge semantics.

Databases live in a process-wide registry keyed by path, so a test can open
the same "database" several times. Forked workers inherit a copy (reads
only); the parent's writes stay in the parent, which is what the engine
needs. Install with ``monkeypatch.setattr(api, "rs", fake_rocks)``.
"""
from __future__ import annotations

import builtins

import bisect
import json
import os

from ngramprep.ngram_acquire.worker import merge_packed_records

_REGISTRY: dict = {}


class _Iterator:
    def __init__(self, db):
        self._keys = sorted(db.data)
        self._db = db
        self._i = 0

    def seek(self, lower):
        self._i = bisect.bisect_left(self._keys, lower)

    def valid(self):
        return self._i < len(self._keys)

    def key(self):
        return self._keys[self._i]

    def value(self):
        return self._db.data[self._keys[self._i]]

    def next(self):
        self._i += 1


class _Batch:
    def __init__(self, db):
        self.db = db
        self.ops = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            for op, k, v in self.ops:
                self.db._apply(op, k, v)
        return False

    def put(self, k, v):
        self.ops.append(("put", bytes(k), bytes(v)))

    def merge(self, k, v):
        self.ops.append(("merge", bytes(k), bytes(v)))

    def merge_batch(self, items):
        for k, v in items:
            self.ops.append(("merge", bytes(k), bytes(v)))


class DB:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.flushes = 0
        self.compactions = 0
        self.closed = False

    def _apply(self, op, k, v):
        if op == "put" or k not in self.data:
            self.data[k] = v
        else:
            self.data[k] = merge_packed_records(self.data[k], v)

    def iterator(self, fill_cache=True, readahead_size=0):
        self.last_iterator_opts = (fill_cache, readahead_size)
        return _Iterator(self)

    def get(self, k):
        return self.data.get(bytes(k))

    def write_batch(self, disable_wal=False, sync=False):
        return _Batch(self)

    def finalize_bulk(self):
        self.flushes += 1

    def compact_all(self):
        self.compactions += 1

    def ingest(self, paths, move=True, write_global_seqno=False):
        import os
        import pickle
        for p in paths:
            with builtins.open(p, "rb") as fh:
                _compression, entries = pickle.load(fh)
            for kind, k, v in entries:
                self._apply(kind, k, v)
            self.ingests = getattr(self, "ingests", 0) + 1
            if move:
                os.remove(p)

    def get_property(self, name):
        if name == "rocksdb.total-sst-files-size":
            return str(sum(len(k) + len(v) for k, v in self.data.items()))
        return str(len(self.data))        # estimate-num-keys: exact for the fake

    # SST metadata as rocks-shim 0.6 reports it: the fake pretends the sorted
    # keys were flushed into files of `file_keys` entries each.
    file_keys = 100

    def live_files(self):
        keys = sorted(self.data)
        files = []
        for i in range(0, len(keys), self.file_keys):
            chunk = keys[i:i + self.file_keys]
            size = sum(len(k) + len(self.data[k]) for k in chunk)
            files.append((chunk[0], chunk[-1], size, 1))
        return files

    def close(self):
        self.closed = True


class SstFileWriter:
    """Writes a 'file' that the fake DB can ingest: a pickled list of
    (kind, key, value) entries, kind 'put' or 'merge', in sorted key order."""

    def __init__(self, *, compression="none", compression_level=3):
        if compression not in ("none", "lz4", "zstd"):
            raise ValueError(f"Unknown compression: {compression}")
        self.compression = compression
        self.path = None
        self.entries = []

    def open(self, path):
        self.path = path

    def put(self, k, v):
        self._add("put", k, v)

    def merge(self, k, v):
        self._add("merge", k, v)

    def _add(self, kind, k, v):
        if self.entries and bytes(k) <= self.entries[-1][1]:
            raise RuntimeError("keys must be added in increasing order")
        self.entries.append((kind, bytes(k), bytes(v)))

    def finish(self):
        import pickle
        with builtins.open(self.path, "wb") as fh:
            pickle.dump((self.compression, self.entries), fh)

    def file_size(self):
        import os
        return os.path.getsize(self.path) if self.path else 0


def open(path, mode="rw", profile="", create_if_missing=False, **_):
    path = str(path)
    if path not in _REGISTRY or not os.path.isdir(path):   # removed on disk = gone
        if not create_if_missing and mode == "r":
            raise FileNotFoundError(path)
        _REGISTRY[path] = DB(path)
        os.makedirs(path, exist_ok=True)     # so Path(...).exists() behaves
    return _REGISTRY[path]


def reset():
    _REGISTRY.clear()


def dump(path):
    """All data keys (no bookkeeping) as a dict, for comparisons."""
    db = _REGISTRY[str(path)]
    return {k: v for k, v in db.data.items() if not k.startswith(b"__")}


def markers(path):
    db = _REGISTRY[str(path)]
    return {k.decode(): json.loads(v) for k, v in db.data.items() if k.startswith(b"__rangemap__/unit/")}

"""An in-memory stand-in for rocks_shim with packed24 merge semantics.

Databases live in a process-wide registry keyed by path, so a test can open
the same "database" several times. Forked workers inherit a copy (reads
only); the parent's writes stay in the parent, which is what the engine
needs. Install with ``monkeypatch.setattr(api, "rs", fake_rocks)``.
"""
from __future__ import annotations

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

    def get_property(self, name):
        return str(len(self.data))

    def close(self):
        self.closed = True


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

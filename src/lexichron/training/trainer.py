"""One word2vec model from one corpus file.

The model is handed around as :class:`Vectors` (words in frequency order
and a matrix), which the stability and benchmark code work on directly;
gensim is needed only to train. ``train_fn`` can be replaced (tests use a
deterministic stand-in).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .grid import Cell

__all__ = ["Vectors", "TrainResult", "train_model", "gensim_train", "save_vectors", "load_vectors", "load_words"]


@dataclass
class Vectors:
    words: List[str]                   # most frequent first
    matrix: np.ndarray                 # (len(words), dim)
    counts: Optional[np.ndarray] = None
    _index: Dict[str, int] = field(default_factory=dict, repr=False)
    _unit: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {w: i for i, w in enumerate(self.words)}

    def __contains__(self, word: str) -> bool:
        return word in self._index

    def __len__(self) -> int:
        return len(self.words)

    def index(self, word: str) -> int:
        return self._index[word]

    def unit(self) -> np.ndarray:
        """Rows normalized to unit length (zero rows left zero). Computed
        once and kept: the axis, neighbourhood and benchmark measurements
        each want it, and renormalizing a large matrix for each was the
        bulk of a year's record-writing time. Callers must not write to it."""
        if self._unit is None:
            self._unit = _normalized(self.matrix)
        return self._unit

    def unit_rows(self, words: Iterable[str]) -> Tuple[List[str], np.ndarray]:
        """The words of ``words`` the model has (in that order, once each)
        and their unit vectors as a (n, dim) array: normalizes only those
        rows, so a measurement of a few hundred words never touches the
        whole matrix."""
        found: List[str] = []
        seen = set()
        for w in words:
            if w in self._index and w not in seen:
                found.append(w)
                seen.add(w)
        if not found:
            return found, np.zeros((0, self.matrix.shape[1]), dtype=self.matrix.dtype)
        if self._unit is not None:
            return found, self._unit[[self._index[w] for w in found]]
        return found, _normalized(self.matrix[[self._index[w] for w in found]])


def _normalized(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


@dataclass
class TrainResult:
    vectors: Vectors
    seconds: float
    vocab_size: int
    corpus_lines: int = 0


TrainFn = Callable[[str, Cell, int, int], Vectors]


def train_model(corpus_file: str, cell: Cell, seed: int, workers: int,
                train_fn: Optional[TrainFn] = None) -> TrainResult:
    t0 = time.perf_counter()
    vectors = (train_fn or gensim_train)(corpus_file, cell, seed, workers)
    return TrainResult(vectors=vectors, seconds=time.perf_counter() - t0, vocab_size=len(vectors))


def gensim_train(corpus_file: str, cell: Cell, seed: int, workers: int) -> Vectors:
    """gensim's Word2Vec on the corpus file (its multi-threaded path).
    ``seed`` fixes initialization; with workers > 1 gensim's thread
    scheduling still adds a little run-to-run variation."""
    from gensim.models import Word2Vec
    model = Word2Vec(corpus_file=corpus_file, vector_size=cell.vector_size, window=cell.window,
                     min_count=cell.min_count, sg=cell.sg, epochs=cell.epochs, workers=workers,
                     seed=seed)
    kv = model.wv
    counts = np.array([kv.get_vecattr(w, "count") for w in kv.index_to_key], dtype=np.int64)
    return Vectors(words=list(kv.index_to_key), matrix=np.asarray(kv.vectors, dtype=np.float32), counts=counts)


def save_vectors(vectors: Vectors, path: str) -> None:
    """``.npz`` with the words, the matrix and the counts: readable with
    numpy alone, from any environment."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".part")
    np.savez_compressed(tmp, words=np.array(vectors.words, dtype=object), matrix=vectors.matrix,
                        counts=vectors.counts if vectors.counts is not None else np.zeros(0, dtype=np.int64))
    # np.savez_compressed appends .npz to a name without it
    written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    written.replace(p)


def load_vectors(path: str) -> Vectors:
    with np.load(path, allow_pickle=True) as z:
        words = [str(w) for w in z["words"]]
        counts = z["counts"] if z["counts"].size else None
        return Vectors(words=words, matrix=np.asarray(z["matrix"], dtype=np.float32), counts=counts)


def load_words(path: str) -> List[str]:
    """Only the word list of a saved model, most frequent first. An ``.npz``
    is a zip archive and a member is decompressed on access, so this reads
    a few megabytes where :func:`load_vectors` would decompress the matrix
    (hundreds of megabytes) just to find the vocabulary."""
    with np.load(path, allow_pickle=True) as z:
        return [str(w) for w in z["words"]]

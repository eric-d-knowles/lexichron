"""How much two models agree about which words are close.

For each probe word, the *k* nearest neighbours (by cosine) are found in
each model within the vocabulary the two models share, and the overlap of
the two neighbour sets is scored; the mean over probes is the agreement.
Probes are sampled per frequency decile so that rare words, where models
disagree most, are represented but do not dominate.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .trainer import Vectors

__all__ = ["probe_words", "neighbour_overlap", "aligned_cosine"]


def probe_words(models: Sequence[Vectors], per_decile: int = 100, seed: int = 0,
                reference: Optional[Vectors] = None) -> List[str]:
    """``per_decile`` words from each frequency decile of the reference
    model's vocabulary (its words are in frequency order), among the words
    every model has. The same seed gives the same probes for every cell."""
    ref = reference or models[0]
    common = set(ref.words)
    for m in models[1:]:
        common &= set(m.words)
    ranked = [w for w in ref.words if w in common]
    if not ranked:
        return []
    rng = np.random.default_rng(seed)
    out: List[str] = []
    n = len(ranked)
    for d in range(10):
        lo, hi = (n * d) // 10, (n * (d + 1)) // 10
        bucket = ranked[lo:hi]
        if not bucket:
            continue
        take = min(per_decile, len(bucket))
        out.extend(bucket[i] for i in sorted(rng.choice(len(bucket), size=take, replace=False)))
    return out


def neighbour_overlaps(a: Vectors, b: Vectors, probes: Sequence[str], k: int = 10) -> np.ndarray:
    """Jaccard overlap of the k nearest neighbours of each probe in the two
    models, within their shared vocabulary, one value per usable probe (1.0:
    the models agree completely about that probe's neighbourhood). Empty
    when no probe is usable."""
    common = [w for w in a.words if w in b]
    if len(common) <= k + 1:
        return np.empty(0)
    ia = np.array([a.index(w) for w in common])
    ib = np.array([b.index(w) for w in common])
    ua = a.unit()[ia]
    ub = b.unit()[ib]
    pos = {w: i for i, w in enumerate(common)}
    scores = []
    for w in probes:
        if w not in pos:
            continue
        p = pos[w]
        na = _top_k(ua @ ua[p], p, k)
        nb = _top_k(ub @ ub[p], p, k)
        scores.append(len(na & nb) / len(na | nb))
    return np.array(scores, dtype=float)


def neighbour_overlap(a: Vectors, b: Vectors, probes: Sequence[str], k: int = 10) -> float:
    """Mean of :func:`neighbour_overlaps`; NaN with no usable probe."""
    scores = neighbour_overlaps(a, b, probes, k)
    return float(scores.mean()) if scores.size else float("nan")


def standard_error(scores: np.ndarray) -> float:
    """Standard error of a mean over probes (NaN below two values)."""
    return float(scores.std(ddof=1) / np.sqrt(scores.size)) if scores.size > 1 else float("nan")


def _top_k(sims: np.ndarray, self_index: int, k: int) -> set:
    sims = sims.copy()
    sims[self_index] = -np.inf
    idx = np.argpartition(-sims, k)[:k]
    return set(int(i) for i in idx)


def aligned_cosine(a: Vectors, b: Vectors, probes: Sequence[str]) -> float:
    """Mean cosine between a probe's vector in ``a`` and in ``b`` after an
    orthogonal Procrustes alignment of ``b`` onto ``a`` over the shared
    vocabulary. A second, geometry-based view of the same agreement."""
    common = [w for w in a.words if w in b]
    if len(common) < 2:
        return float("nan")
    ua = a.unit()[[a.index(w) for w in common]]
    ub = b.unit()[[b.index(w) for w in common]]
    if ua.shape[1] != ub.shape[1]:
        return float("nan")
    u, _s, vt = np.linalg.svd(ub.T @ ua)
    r = u @ vt
    aligned = ub @ r
    pos = {w: i for i, w in enumerate(common)}
    cos = [float(ua[pos[w]] @ aligned[pos[w]]) for w in probes if w in pos]
    return float(np.mean(cos)) if cos else float("nan")

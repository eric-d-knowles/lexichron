"""Quality scores for a model: agreement with human similarity judgements,
analogy accuracy, and how much of a vocabulary list the model covers.

Datasets are the plain-text formats gensim ships (``wordsim353.tsv``:
``word1 <tab> word2 <tab> score``; ``questions-words.txt``: ``a b c d``
lines under ``: section`` headers); lines beginning with ``#`` are
comments. Words are lowercased before lookup, since the models are.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .trainer import Vectors

__all__ = ["similarity_score", "analogy_accuracy", "coverage", "bundled_dataset"]


def bundled_dataset(name: str) -> Optional[str]:
    """Path of a bundled dataset (``simlex999.txt``, ``wordsim353.tsv``,
    ``questions-words.txt``): the image's copy ($LEXICHRON_DATA_DIR, then
    /opt/lexichron/data), else the one gensim ships, else None."""
    import os
    for d in (os.environ.get("LEXICHRON_DATA_DIR"), "/opt/lexichron/data"):
        if d and (Path(d) / name).is_file():
            return str(Path(d) / name)
    try:
        from gensim.test.utils import datapath
        p = Path(datapath(name))
        return str(p) if p.exists() else None
    except Exception:
        return None


def similarity_sets(given: Optional[str] = None) -> Dict[str, str]:
    """The similarity datasets to score, by label: the file given (a copy of
    a known dataset keeps its label, so its scores land in the SimLex /
    WordSim columns), else WordSim-353 and SimLex-999 where bundled."""
    if given:
        stem = Path(given).stem.lower()
        label = "simlex" if stem.startswith("simlex") else "wordsim" if stem.startswith("wordsim") else stem
        return {label: given}
    sets: Dict[str, str] = {}
    for label, fname in (("wordsim", "wordsim353.tsv"), ("simlex", "simlex999.txt")):
        p = bundled_dataset(fname)
        if p:
            sets[label] = p
    return sets


def similarity_score(vectors: Vectors, dataset: str) -> Tuple[float, float]:
    """Spearman correlation between the model's cosine similarities and the
    dataset's scores over the pairs the model covers, and the fraction of
    pairs covered. ``(nan, 0.0)`` when fewer than three pairs are covered."""
    from scipy.stats import spearmanr
    pairs = list(_read_pairs(dataset))
    if not pairs:
        return float("nan"), 0.0
    unit = vectors.unit()
    ours, theirs = [], []
    for a, b, s in pairs:
        if a in vectors and b in vectors:
            ours.append(float(unit[vectors.index(a)] @ unit[vectors.index(b)]))
            theirs.append(s)
    cov = len(ours) / len(pairs)
    if len(ours) < 3:
        return float("nan"), cov
    rho = spearmanr(ours, theirs).correlation
    return float(rho), cov


def analogy_accuracy(vectors: Vectors, dataset: str, restrict: int = 30_000) -> Tuple[float, float]:
    """3CosAdd accuracy on the questions the model covers (searching the
    ``restrict`` most frequent words), and the fraction covered."""
    questions = list(_read_questions(dataset))
    if not questions:
        return float("nan"), 0.0
    n = min(restrict, len(vectors))
    unit = vectors.unit()[:n]
    usable = [(a, b, c, d) for a, b, c, d in questions
              if all(w in vectors and vectors.index(w) < n for w in (a, b, c, d))]
    cov = len(usable) / len(questions)
    if not usable:
        return float("nan"), cov
    correct = 0
    for a, b, c, d in usable:
        ia, ib, ic, idx = (vectors.index(w) for w in (a, b, c, d))
        target = unit[ib] - unit[ia] + unit[ic]
        sims = unit @ target
        sims[[ia, ib, ic]] = -np.inf
        if int(np.argmax(sims)) == idx:
            correct += 1
    return correct / len(usable), cov


def coverage(vectors: Vectors, words: Iterable[str]) -> float:
    """Fraction of ``words`` present in the model."""
    words = list(words)
    if not words:
        return float("nan")
    return sum(1 for w in words if w in vectors) / len(words)


def _read_pairs(path: str) -> Iterable[Tuple[str, str, float]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split("\t") if "\t" in line or "," in line else line.split()
            if len(parts) < 3:
                continue
            score = _first_number(parts[2:])        # SimLex-999 has a POS column before its score
            if score is not None:
                yield parts[0].strip().lower(), parts[1].strip().lower(), score


def _first_number(fields) -> Optional[float]:
    for f in fields:
        try:
            return float(f)
        except ValueError:
            continue
    return None


def _read_questions(path: str) -> Iterable[Tuple[str, str, str, str]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(":") or line.startswith("#"):
                continue
            parts = line.lower().split()
            if len(parts) == 4:
                yield parts[0], parts[1], parts[2], parts[3]

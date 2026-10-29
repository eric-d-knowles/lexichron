"""Scalars measured inside one model, for a time series across an ensemble.

The measure stage runs one of these over every replicate of every year.
Each takes a model and a specification and returns a value per word the
model has, with a note on how well the specification was covered. All of
them are rotation-invariant (a cosine, or a count), which is why nothing
needs aligning across replicates or years before the ensemble's mean and
spread are taken: that is the border the measure stage keeps to. Anything
that compares vectors between two models (a lexeme's vector over time, a
centroid tracked across years) is a different operation that needs an
alignment first, and does not belong here.

- ``projection``: cosine with the axis from the negative pole's centroid to
  the positive pole's, both built inside the model (Kozlowski, Taddy &
  Evans 2019; the valence yardstick of ``tune`` and ``train`` is this
  measure with fixed poles).
- ``similarity``: cosine with the centroid of a set of target words (one
  word: the plain cosine with it).
- ``frequency``: the word's share of the model's token count, from the
  counts word2vec kept; it varies across corpus resamples and not across
  seeds, so its error is the text's alone.

A word of the list may be a group, ``surgeon|physician|doctor``: one series
whose value in each model is the mean of the members' values (members the
model lacks are left out). Because the averaging happens inside the model,
the group's standard error across the ensemble is exact, where averaging
the members' series afterwards would have to assume their errors
independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lexichron.training.axis import axis_from_poles
from lexichron.training.trainer import Vectors

__all__ = ["KINDS", "Spec", "Measured", "measure", "read_word_list", "read_pole_lists", "spec_digest",
           "projection", "similarity", "frequency", "GROUP_SEP", "group_members", "grouped"]


@dataclass(frozen=True)
class Spec:
    """What to measure: the kind, the words measured, and the kind's own
    words (poles for a projection, the target of a similarity)."""
    kind: str
    words: Tuple[str, ...]
    positive: Tuple[str, ...] = ()
    negative: Tuple[str, ...] = ()
    target: Tuple[str, ...] = ()

    @property
    def lexemes(self) -> List[str]:
        """Every lexeme the list asks for, groups expanded, once each."""
        return list(dict.fromkeys(m for w in self.words for m in group_members(w)))

    def validate(self) -> None:
        if any(not group_members(w) for w in self.words):
            raise ValueError(f"empty word or group in the list: {[w for w in self.words if not group_members(w)]}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {tuple(KINDS)}, not {self.kind!r}")
        if not self.words:
            raise ValueError("no words to measure: give words, or a words_file")
        if self.kind == "projection" and (len(self.positive) < MIN_POLE_WORDS or len(self.negative) < MIN_POLE_WORDS):
            raise ValueError(f"a projection needs at least {MIN_POLE_WORDS} words on each pole "
                             f"(positive and negative, or a poles_file)")
        if self.kind == "similarity" and not self.target:
            raise ValueError("a similarity needs a target word or words")

    def as_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"kind": self.kind, "words": list(self.words)}
        if self.kind == "projection":
            d["positive"], d["negative"] = list(self.positive), list(self.negative)
        if self.kind == "similarity":
            d["target"] = list(self.target)
        return d


@dataclass
class Measured:
    """One model's values, word -> value (words the model lacks are absent),
    and the coverage of the specification: how many of the kind's own words
    the model had (both poles, or the target), and a note when the measure
    could not be built at all."""
    values: Dict[str, float]
    coverage: Dict[str, int] = field(default_factory=dict)
    note: Optional[str] = None


MIN_POLE_WORDS = 3          # per pole, in the model (as the valence yardstick requires)


# ---------------------------------------------------------------- the kinds
GROUP_SEP = "|"           # "surgeon|physician|doctor": one series, the mean of the members' values in each model


def group_members(word: str) -> List[str]:
    """The lexemes a word of the list stands for: itself, or the members of
    a ``a|b|c`` group."""
    return [w.strip() for w in word.split(GROUP_SEP) if w.strip()]


def grouped(values: Dict[str, float], words: Sequence[str]) -> Dict[str, float]:
    """Per-lexeme values folded to the word list: a plain word keeps its
    value, a group gets the mean over the members the model has (absent
    members are left out, as the old synonym averaging did). Averaging the
    scalars inside the model is what makes the group's ensemble error
    exact: its spread across replicates is the spread of that mean."""
    out: Dict[str, float] = {}
    for w in words:
        have = [values[m] for m in group_members(w) if m in values]
        if have:
            out[w] = float(np.mean(have))
    return out


def projection(vectors: Vectors, spec: Spec) -> Measured:
    ax, pos, neg = axis_from_poles(vectors, spec.positive, spec.negative, MIN_POLE_WORDS)
    cov = {"positive": len(pos), "negative": len(neg)}
    if ax is None:
        return Measured({}, cov, f"axis not built: {len(pos)} of {len(spec.positive)} positive and {len(neg)} of "
                                 f"{len(spec.negative)} negative pole words in the model (at least "
                                 f"{MIN_POLE_WORDS} each needed)")
    found, rows = vectors.unit_rows(spec.lexemes)
    return Measured(grouped(dict(zip(found, (rows @ ax).astype(float).tolist())), spec.words), cov)


def similarity(vectors: Vectors, spec: Spec) -> Measured:
    tfound, trows = vectors.unit_rows(spec.target)
    cov = {"target": len(tfound)}
    if not tfound:
        return Measured({}, cov, f"none of the {len(spec.target)} target words is in the model")
    centroid = trows.mean(axis=0)
    n = np.linalg.norm(centroid)
    if n == 0:
        return Measured({}, cov, "the target words' centroid is the zero vector")
    centroid = centroid / n
    found, rows = vectors.unit_rows(spec.lexemes)
    return Measured(grouped(dict(zip(found, (rows @ centroid).astype(float).tolist())), spec.words), cov)


def frequency(vectors: Vectors, spec: Spec) -> Measured:
    if vectors.counts is None or not len(vectors.counts):
        return Measured({}, {}, "the model carries no word counts")
    total = float(vectors.counts.sum())
    if total <= 0:
        return Measured({}, {}, "the model's word counts sum to zero")
    out = {w: float(vectors.counts[vectors.index(w)]) / total for w in spec.lexemes if w in vectors}
    return Measured(grouped(out, spec.words), {})


KINDS: Dict[str, Tuple[Callable[[Vectors, Spec], Measured], str]] = {
    "projection": (projection, "cosine with the axis between two poles' centroids, built inside the model"),
    "similarity": (similarity, "cosine with the centroid of the target words, built inside the model"),
    "frequency": (frequency, "the word's share of the model's tokens (from the counts word2vec kept)"),
}


def measure(vectors: Vectors, spec: Spec) -> Measured:
    """Measure ``spec`` on one model."""
    return KINDS[spec.kind][0](vectors, spec)


# ---------------------------------------------------------------- inputs
def read_word_list(path: str | Path) -> List[str]:
    """One word per line (or several separated by commas or spaces); blank
    lines and ``#`` comments skipped; lowercased, in order, once each."""
    words: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            words += [w.strip().lower() for w in line.replace(",", " ").split() if w.strip()]
    return list(dict.fromkeys(words))


def read_pole_lists(path: str | Path) -> Tuple[List[str], List[str]]:
    """Pole words from a file in either of two layouts: two lines (the
    positive words, then the negative, separated by spaces or commas, as
    the ``tune`` and ``train`` poles file), or one pair per line
    (``he<TAB>she`` or ``he, she``), which is the same axis: the mean of
    the pairwise differences equals the difference of the two means."""
    lines = [l.split("#", 1)[0].strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        raise ValueError(f"{path}: no pole words")
    pairs = [[w.strip().lower() for w in l.replace("\t", ",").split(",") if w.strip()] for l in lines]
    if all(len(p) == 2 for p in pairs) and (len(lines) != 2 or all(" " not in l for l in lines)):
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if len(lines) != 2:
        raise ValueError(f"{path}: expected two lines of pole words (positive, then negative), or one "
                         f"positive,negative pair per line")
    split = lambda l: [w.strip().lower() for w in l.replace(",", " ").split() if w.strip()]
    return split(lines[0]), split(lines[1])


def spec_digest(spec: Spec) -> str:
    """A short fingerprint of a specification: a measure directory holds one
    specification, and a resume checks it has not changed under the name."""
    import hashlib
    import json
    text = json.dumps(spec.as_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

"""A general-purpose semantic axis as a yardstick for a model.

Much diachronic work measures words by projecting them onto an axis between
two sets of pole words (gender, class, morality, ...). How well a model
supports that kind of measurement is a property of the model that the
neighbourhood scores do not capture, and it can be checked without
assuming anyone's research question by using the one axis every
embedding space has and every rater agrees on: valence. The pole words are
the pleasant/unpleasant lists of the implicit-association literature
(Greenwald, McGhee & Schwartz 1998, as used by Caliskan, Bryson &
Narayanan 2017), and the yardstick is a lexicon of human valence ratings.

Three questions, each answered by projecting the rated words onto the
model's own axis (the mean of the pleasant vectors minus the mean of the
unpleasant ones):

- validity: do the projections agree with the human ratings? (Spearman)
- reliability: do two models (seeds, halves, adjacent years) put the same
  words in the same places on the axis? (Pearson correlation of their
  projections, each model building its own axis)
- change: do periods differ by more than adjacent years do?

Any ``word <tab> rating`` file serves as the lexicon; any two-line file
(pleasant words, then unpleasant words) as the poles.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lexichron.training.trainer import Vectors

__all__ = ["PLEASANT", "UNPLEASANT", "Axis", "load_axis", "read_ratings", "read_poles", "axis_vector",
           "projections", "validity", "agreement"]

# Greenwald, McGhee & Schwartz (1998), as listed by Caliskan et al. (2017), WEAT 1
PLEASANT = ("caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
            "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise",
            "family", "happy", "laughter", "paradise", "vacation")
UNPLEASANT = ("abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink",
              "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer",
              "kill", "rotten", "vomit", "agony", "prison")
MIN_POLE_WORDS = 3          # per side, in the model's vocabulary
MIN_TARGETS = 20            # rated words in the vocabulary for a correlation


class Axis:
    """The yardstick: pole word lists and a lexicon of ratings (words that
    are poles are left out of the lexicon, so validity is not circular)."""

    def __init__(self, pleasant: Sequence[str], unpleasant: Sequence[str], ratings: Dict[str, float],
                 lexicon_path: Optional[str] = None, poles_path: Optional[str] = None) -> None:
        self.pleasant = tuple(pleasant)
        self.unpleasant = tuple(unpleasant)
        poles = set(self.pleasant) | set(self.unpleasant)
        self.ratings = {w: r for w, r in ratings.items() if w not in poles}
        self.lexicon_path, self.poles_path = lexicon_path, poles_path

    @property
    def targets(self) -> List[str]:
        return list(self.ratings)


def read_ratings(path: str | Path) -> Dict[str, float]:
    """``word <tab> rating`` (further fields ignored; a header line, comments
    and tokens that are not plain words — emoticons, punctuation — are skipped). Reads VADER's
    ``vader_lexicon.txt`` (token, mean, sd, ratings), Warriner et al.'s CSV
    (Word, V.Mean.Sum, ...) and any similar file."""
    out: Dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "\t" if "\t" in sample else ","
        for row in csv.reader(fh, delimiter=delim):
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            if not word.isalnum() or word.startswith("#"):       # no emoticons, punctuation, phrases
                continue
            try:
                out[word] = float(row[1])
            except ValueError:
                continue                     # the header
    return out


def read_poles(path: str | Path) -> Tuple[List[str], List[str]]:
    """Two non-empty lines: the pleasant (positive) words, then the
    unpleasant (negative) ones, separated by spaces or commas."""
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        raise ValueError(f"{path}: expected two lines of pole words (positive, then negative)")
    split = lambda l: [w.strip().lower() for w in l.replace(",", " ").split() if w.strip()]
    return split(lines[0]), split(lines[1])


def load_axis(lexicon: Optional[str], poles: Optional[str]) -> Optional[Axis]:
    """The yardstick from a ratings file and an optional poles file; None
    when there is no lexicon."""
    if not lexicon:
        return None
    pos, neg = read_poles(poles) if poles else (list(PLEASANT), list(UNPLEASANT))
    return Axis(pos, neg, read_ratings(lexicon), lexicon, poles)


def axis_vector(vectors: Vectors, pleasant: Iterable[str], unpleasant: Iterable[str]) -> Optional[np.ndarray]:
    """Unit vector from the unpleasant centroid to the pleasant one, over the
    pole words the model has; None with too few on either side."""
    unit = vectors.unit()
    pos = [unit[vectors.index(w)] for w in pleasant if w in vectors]
    neg = [unit[vectors.index(w)] for w in unpleasant if w in vectors]
    if len(pos) < MIN_POLE_WORDS or len(neg) < MIN_POLE_WORDS:
        return None
    d = np.mean(pos, axis=0) - np.mean(neg, axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def projections(vectors: Vectors, axis: np.ndarray, words: Iterable[str]) -> Dict[str, float]:
    """Cosine of each word (that the model has) with the axis."""
    unit = vectors.unit()
    return {w: float(unit[vectors.index(w)] @ axis) for w in words if w in vectors}


def validity(vectors: Vectors, yard: Axis) -> Tuple[float, float]:
    """(Spearman correlation between the rated words' projections and their
    ratings, share of the rated words the model has); NaN with too few."""
    from scipy.stats import spearmanr
    ax = axis_vector(vectors, yard.pleasant, yard.unpleasant)
    have = [w for w in yard.targets if w in vectors]
    cov = len(have) / len(yard.targets) if yard.targets else float("nan")
    if ax is None or len(have) < MIN_TARGETS:
        return float("nan"), cov
    proj = projections(vectors, ax, have)
    rho = spearmanr([proj[w] for w in have], [yard.ratings[w] for w in have]).correlation
    return float(rho), cov


def agreement(a: Vectors, b: Vectors, yard: Axis) -> float:
    """Pearson correlation between two models' projections of the rated
    words they share, each onto its own axis: 1 means they place every word
    identically along it. NaN with too few words or no axis."""
    ax_a = axis_vector(a, yard.pleasant, yard.unpleasant)
    ax_b = axis_vector(b, yard.pleasant, yard.unpleasant)
    have = [w for w in yard.targets if w in a and w in b]
    if ax_a is None or ax_b is None or len(have) < MIN_TARGETS:
        return float("nan")
    pa, pb = projections(a, ax_a, have), projections(b, ax_b, have)
    x = np.array([pa[w] for w in have])
    y = np.array([pb[w] for w in have])
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

"""The replicate ensemble a year's final models form, and what it measures.

For each year the train stage fits one hyperparameter cell many times: on
``corpora`` resamples of the year's corpus (a Poisson draw of every
n-gram's repeat count, so each resample is another same-sized sample of
books: with the ``none`` weighting a count-1 n-gram survives with
probability 1 - 1/e) and, on each resample, with ``seeds`` training seeds.
The nested design (seeds within resamples: a seed index names a different
fit on every resample) separates the two sources of noise in anything
measured on a model — the sample of text and the fit — and the per-replicate models
are the product: a scalar measured inside each replicate (a projection on
an axis, say) is unaffected by the arbitrary rotation between fits, so
nothing needs aligning before the replicates are averaged or their spread
is used as the measurement's error.

Every replicate is reproducible from its coordinates alone: its resampling
seed is derived from ``(noise_seed, year, corpus)`` and its training seed
from ``(noise_seed, year, corpus, seed)`` by numpy's ``SeedSequence``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lexichron.training.grid import Cell

__all__ = ["RESAMPLINGS", "ReplicaTask", "plan_replicas", "resample_seed", "training_seed",
           "variance_components", "VarianceComponents", "projection_cube", "word_components"]

RESAMPLINGS = ("poisson", "none")

# Salts that keep the resampling and training seed streams apart. Fixed:
# changing them would silently change every ensemble trained after.
_SALT_RESAMPLE = 0xA17E5A11
_SALT_TRAIN = 0x5EEDF00D


def resample_seed(noise_seed: int, year: int, corpus: int) -> int:
    """The corpus resampling seed of one (year, corpus replicate): the same
    for every seed trained on that replicate, so the corpus stays fixed
    while the fit varies."""
    ss = np.random.SeedSequence([int(noise_seed), int(year), int(corpus), _SALT_RESAMPLE])
    return int(ss.generate_state(1)[0])


def training_seed(noise_seed: int, year: int, corpus: int, seed: int) -> int:
    """The word2vec seed of one replicate (year, corpus, seed). gensim takes
    a 32-bit seed."""
    ss = np.random.SeedSequence([int(noise_seed), int(year), int(corpus), int(seed), _SALT_TRAIN])
    return int(ss.generate_state(1)[0] % (2 ** 31 - 1))


@dataclass(frozen=True)
class ReplicaTask:
    """One model of the ensemble: the cell, a year, a corpus replicate and
    a seed index. Its file is ``<models>/y1990/c03_s1.npz``."""
    cell: Cell
    year: int
    corpus: int
    seed_index: int
    noise_seed: int

    @property
    def stem(self) -> str:
        return f"y{self.year}/c{self.corpus:02d}_s{self.seed_index}"

    @property
    def name(self) -> str:
        return f"c{self.corpus:02d}_s{self.seed_index}"

    @property
    def seed(self) -> int:
        return training_seed(self.noise_seed, self.year, self.corpus, self.seed_index)

    @property
    def corpus_key(self) -> Tuple[int, int]:
        return (self.year, self.corpus)


def plan_replicas(cell: Cell, years: Sequence[int], corpora: int, seeds: int, noise_seed: int) -> List[ReplicaTask]:
    """The ensemble's models in training order: year by year, corpus by
    corpus (so a corpus file is needed by consecutive models and can go
    as soon as they are done), seed by seed."""
    if corpora < 1 or seeds < 1:
        raise ValueError("corpus_replicates and seeds_per_corpus must be at least 1")
    return [ReplicaTask(cell, int(y), c, s, int(noise_seed))
            for y in years for c in range(corpora) for s in range(seeds)]


@dataclass
class VarianceComponents:
    """The nested random-effects decomposition (seeds within corpus
    resamples) of a scalar measured on every replicate of a year (one value
    per model and word), summarized over the words."""
    words: int                        # words measured in every replicate
    corpora: int
    seeds: int
    var_seed: float                   # within-corpus, across seeds (the fit), averaged over words
    var_corpus: float                 # across corpora, net of the seed part (the sample of text)
    var_replicate: float              # across all replicates, as one source (when the design is one-way)
    se_mean: float                    # standard error of the ensemble mean per word, averaged over words
    var_words: float                  # across words, of the ensemble means (the signal)
    noise_share: float                # mean se_mean^2 / var_words: the part of the spread across words that is noise

    def as_dict(self, prefix: str = "axis_") -> Dict[str, float]:
        """The components under ``prefix`` (the family's column names)."""
        return {f"{prefix}words": self.words, f"{prefix}var_seed": self.var_seed,
                f"{prefix}var_corpus": self.var_corpus, f"{prefix}var_replicate": self.var_replicate,
                f"{prefix}se_mean": self.se_mean, f"{prefix}var_words": self.var_words,
                f"{prefix}noise_share": self.noise_share}


def variance_components(values: np.ndarray) -> VarianceComponents:
    """``values`` is (corpora, seeds, words). With seeds nested in corpora
    and at least two levels of each: the seed component is the variance across
    seeds within a corpus, averaged; the corpus component the variance of
    the corpus means less the seed part they carry (seed variance / seeds),
    floored at zero. With one level of either, the other's variance is all
    that can be measured and is reported as the replicate variance. The
    standard error of the ensemble mean is sqrt(corpus / C + seed / (C S))
    per word, and the noise share compares its square with the variance
    of the ensemble means across words."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError("values must be (corpora, seeds, words)")
    C, S, W = v.shape
    nan = float("nan")
    if W == 0:
        return VarianceComponents(0, C, S, nan, nan, nan, nan, nan, nan)
    grand = v.mean(axis=(0, 1))                                        # (W,)
    var_words = float(grand.var(ddof=1)) if W > 1 else nan
    var_seed = var_corpus = var_rep = se2 = nan
    if S > 1:
        var_seed = float(v.var(axis=1, ddof=1).mean())                 # within corpus, over seeds; mean over c, w
    if C > 1:
        corpus_means = v.mean(axis=1)                                  # (C, W)
        between = corpus_means.var(axis=0, ddof=1)                     # (W,)
        if S > 1:
            var_corpus = float(np.maximum(between - v.var(axis=1, ddof=1).mean(axis=0) / S, 0.0).mean())
        else:
            var_rep = float(between.mean())                            # one seed: corpus and fit together
    elif S > 1:
        var_rep = var_seed                                             # one corpus: the seeds are the replicates
    if C > 1 and S > 1:
        se2 = var_corpus / C + var_seed / (C * S)
    elif var_rep == var_rep:
        se2 = var_rep / (C * S)
    se_mean = float(np.sqrt(se2)) if se2 == se2 else nan
    noise_share = float(se2 / var_words) if se2 == se2 and var_words == var_words and var_words > 0 else nan
    return VarianceComponents(W, C, S, var_seed, var_corpus, var_rep, se_mean, var_words, noise_share)


def projection_cube(projections: Dict[Tuple[int, int], Dict[str, float]], corpora: int, seeds: int,
                    words: Optional[Sequence[str]] = None) -> Tuple[List[str], np.ndarray]:
    """From each replicate's ``{word: value}`` (keyed by (corpus, seed)),
    a (corpora, seeds, words) cube. Without ``words``, over the words every
    replicate has, sorted (the profile's complete cube). With ``words``, in
    that order, NaN where a replicate lacks a word (a rare word can miss a
    resample's min count): :func:`word_components` handles the gaps."""
    if words is None:
        if not projections:
            return [], np.zeros((corpora, seeds, 0))
        common: Optional[set] = None
        for d in projections.values():
            common = set(d) if common is None else common & set(d)
        words = sorted(common or [])
    words = list(words)
    cube = np.full((corpora, seeds, len(words)), np.nan)
    for (c, s), d in projections.items():
        cube[c, s] = [d.get(w, np.nan) for w in words]
    return words, cube


def word_components(values: np.ndarray) -> Dict[str, np.ndarray]:
    """Per word, the same decomposition as :func:`variance_components`
    (which averages it over the words), tolerating replicates that lack a
    word: ``mean``, ``se`` (standard error of the ensemble mean),
    ``var_seed``, ``var_corpus``, ``n`` (replicates with a value), each a
    (words,) array over the (corpora, seeds, words) cube. The seed variance
    pools the resamples that have two or more values for the word; the
    corpus variance is the variance of the resample means less the seed
    part they carry, floored at zero; the standard error is
    sqrt(var_corpus / C_w + var_seed / n_w) with C_w the resamples and n_w
    the replicates that have the word. With one seed per resample only
    the replicate variance can be measured and stands in for var_corpus,
    with var_seed NaN; with one resample the reverse. A word found in a
    single resample of a larger design gets var_corpus NaN and the
    standard error of its seeds alone (an understatement, flagged by n)."""
    import warnings
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError("values must be (corpora, seeds, words)")
    C, S, W = v.shape
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # nanvar over a single value
        have = ~np.isnan(v)                                            # (C, S, W)
        n = have.sum(axis=(0, 1))                                      # (W,)
        mean = np.where(n > 0, np.nansum(v, axis=(0, 1)) / np.maximum(n, 1), np.nan)
        n_cs = have.sum(axis=1)                                        # (C, W) seeds with a value
        corpus_means = np.where(n_cs > 0, np.nansum(v, axis=1) / np.maximum(n_cs, 1), np.nan)   # (C, W)
        C_w = (n_cs > 0).sum(axis=0)                                   # (W,) resamples with a value
        nan = np.full(W, np.nan)
        var_seed = nan.copy()
        if S > 1:
            # within-resample variance, over resamples with two or more seeds (ddof 1 each), pooled
            dev2 = np.nansum((v - corpus_means[:, None, :]) ** 2, axis=1)           # (C, W)
            dof = np.where(n_cs > 1, n_cs - 1, 0)
            pooled_dof = dof.sum(axis=0)
            var_seed = np.where(pooled_dof > 0, dev2.sum(axis=0) / np.maximum(pooled_dof, 1), np.nan)
        var_corpus = nan.copy()
        between = nan.copy()
        if C > 1:
            ok = C_w > 1
            between = np.where(ok, np.nanvar(corpus_means, axis=0, ddof=1), np.nan)
            seeds_per = np.where(C_w > 0, n_cs.sum(axis=0) / np.maximum(C_w, 1), np.nan)   # mean seeds per resample
            if S > 1:
                var_corpus = np.where(ok & ~np.isnan(var_seed), np.maximum(between - var_seed / seeds_per, 0.0),
                                      np.where(ok, between, np.nan))
            else:
                var_corpus = between                                   # one seed: text and fit together
        if C > 1 and S > 1:
            se2 = var_corpus / np.maximum(C_w, 1) + var_seed / np.maximum(n, 1)
            se2 = np.where(np.isnan(var_seed), between / np.maximum(C_w, 1), se2)
            # a word in one resample only: the text's share cannot be measured; the fit's can
            se2 = np.where(np.isnan(var_corpus) & ~np.isnan(var_seed), var_seed / np.maximum(n, 1), se2)
        elif C > 1:
            se2 = between / np.maximum(C_w, 1)
        elif S > 1:
            se2 = var_seed / np.maximum(n, 1)
        else:
            se2 = nan.copy()
        se = np.sqrt(np.where(n > 1, se2, np.nan))
    return {"mean": mean, "se": se, "var_seed": var_seed, "var_corpus": var_corpus, "n": n.astype(int)}

"""A year's corpus file, written from the pivoted database in one pass.

Each kept n-gram becomes a line, repeated according to the weighting
(``freq``: floor(log5(count + 1)) times; ``doc_freq``: the same on the
volume count; ``none``: once). ``<UNK>`` tokens are handled by the
``unk_mode`` policy: ``strip`` (default) drops them and keeps the n-gram if
at least two real tokens remain, ``reject`` drops any n-gram containing
one, ``retain`` keeps them as tokens.

The same pass can write two *half* files alongside the full one: every
line is assigned to one half at random, so the halves are two independent
samples of the year's corpus (split-half reliability), and a Poisson
resample of the line counts for replica training.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import floor, log
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from ngramprep.ngram_pivot.encoding import decode_year_ngram_key, decode_year_stats, year_prefix

__all__ = ["CorpusStats", "write_corpus", "weight_for", "UNK_MODES", "WEIGHTINGS"]

UNK = "<UNK>"
UNK_MODES = ("strip", "reject", "retain")
WEIGHTINGS = ("freq", "doc_freq", "none")


@dataclass
class CorpusStats:
    year: int
    weight_by: str
    unk_mode: str
    ngrams_read: int = 0          # records of the year in the database
    ngrams_used: int = 0          # kept after the <UNK> policy
    lines: int = 0                # lines in the full file (weighted repeats)
    tokens: int = 0               # tokens in the full file
    lines_half_a: int = 0
    lines_half_b: int = 0

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def weight_for(count: int, base: float = 5) -> int:
    """How many times an n-gram with this count appears in the corpus."""
    return max(1, floor(log(count + 1, base)))


def write_corpus(db_path: str, year: int, out_path: str, *, weight_by: str = "freq",
                 unk_mode: str = "strip", log_base: float = 5.0,
                 halves: Optional[Tuple[str, str]] = None, split_seed: int = 0,
                 resample_seed: Optional[int] = None) -> CorpusStats:
    """Write the year's corpus to ``out_path`` (and, with ``halves``, two
    half files). Files are written under a temporary name and renamed when
    complete, so a file that exists is a whole one."""
    if weight_by not in WEIGHTINGS:
        raise ValueError(f"weight_by must be one of {WEIGHTINGS}")
    if unk_mode not in UNK_MODES:
        raise ValueError(f"unk_mode must be one of {UNK_MODES}")
    stats = CorpusStats(year=year, weight_by=weight_by, unk_mode=unk_mode)
    split_rng = np.random.default_rng(split_seed) if halves else None
    resample_rng = np.random.default_rng(resample_seed) if resample_seed is not None else None

    tmp_full = out_path + ".part"
    files = [open(tmp_full, "w", encoding="utf-8")]
    tmp_halves = []
    if halves:
        for p in halves:
            tmp_halves.append(p + ".part")
            files.append(open(p + ".part", "w", encoding="utf-8"))
    try:
        full = files[0]
        for ngram, count, volumes in _year_records(db_path, year):
            stats.ngrams_read += 1
            tokens = ngram.decode("utf-8", errors="replace").split()
            if unk_mode == "reject":
                if UNK in tokens:
                    continue
            elif unk_mode == "strip":
                tokens = [t for t in tokens if t != UNK]
                if len(tokens) < 2:
                    continue
            if weight_by == "freq":
                w = weight_for(count, log_base)
            elif weight_by == "doc_freq":
                w = weight_for(volumes, log_base)
            else:
                w = 1
            if resample_rng is not None:
                w = int(resample_rng.poisson(w))
                if w == 0:
                    continue
            stats.ngrams_used += 1
            line = " ".join(tokens) + "\n"
            full.write(line * w)
            stats.lines += w
            stats.tokens += w * len(tokens)
            if split_rng is not None:
                # each repeat lands in one half independently
                a = int(split_rng.binomial(w, 0.5))
                if a:
                    files[1].write(line * a)
                if w - a:
                    files[2].write(line * (w - a))
                stats.lines_half_a += a
                stats.lines_half_b += w - a
    finally:
        for f in files:
            f.close()
    os.replace(tmp_full, out_path)
    if halves:
        for tmp, final in zip(tmp_halves, halves):
            os.replace(tmp, final)
    return stats


def _year_records(db_path: str, year: int) -> Iterator[Tuple[bytes, int, int]]:
    """``(ngram, count, volumes)`` for every record of the year, read with
    the scan profile (a small, fixed memory footprint per process)."""
    from ngramprep.common_db.api import open_db, range_scan
    lo, hi = year_prefix(year), year_prefix(year + 1)
    with open_db(db_path, mode="r", profile="scan:packed24") as db:
        for key, value in range_scan(db, lo, hi, fill_cache=False, readahead_size=2 << 20):
            _y, ngram = decode_year_ngram_key(bytes(key))
            count, volumes = decode_year_stats(bytes(value))
            yield ngram, count, volumes


def corpus_file_name(year: int, weight_by: str, unk_mode: str, part: str = "") -> str:
    """``corpus_y1990_wbfreq_strip.txt`` (``_a``/``_b`` for the halves)."""
    return f"corpus_y{year}_wb{weight_by}_{unk_mode}{('_' + part) if part else ''}.txt"

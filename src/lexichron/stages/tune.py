"""``lexichron tune``: the hyperparameter search.

A grid of word2vec settings is trained on a few sampled years of the
pivoted database and scored for *stability* (how much two models agree
about which words are close: same corpus with another seed, the two halves
of one year's corpus, adjacent years) and *quality* (similarity and
analogy benchmarks, vocabulary coverage). The result is a table per cell,
with a suggested cell that is the most stable among those close to the
best on coverage and quality; the choice is yours.

The sampled years are chosen as adjacent pairs spread over the study
range, so adjacent-year agreement is measured alongside the agreement
between the first and last sampled years (the signal to compare it with).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lexichron.training import benchmarks, stability
from lexichron.training.corpus import UNK_MODES, CorpusStats, corpus_file_name, write_corpus
from lexichron.training.grid import AXES, Cell, Grid, corpus_variants
from lexichron.training.trainer import TrainFn, Vectors, load_vectors, save_vectors, train_model
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths

__all__ = ["tune_models", "sample_years", "plan_models", "summarize_results"]

logger = logging.getLogger(__name__)

RESULTS_FILE = "results.tsv"      # one row per cell and year
PAIRS_FILE = "pairs.tsv"          # one row per cell and year pair
SUMMARY_FILE = "summary.tsv"      # one row per cell
SUMMARY_JSON = "summary.json"


def tune_models(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
    db_path_stub: str,
    years: Tuple[int, int] = (1968, 2019),
    sample_years: Optional[List[int]] = None,
    year_pairs: int = 4,
    vector_size: Any = None,
    window: Any = None,
    min_count: Any = None,
    epochs: Any = None,
    approach: Any = None,
    weight_by: Any = None,
    seeds: int = 2,
    split_half: bool = True,
    unk_mode: str = "strip",
    neighbours: int = 10,
    probes_per_decile: int = 100,
    similarity_dataset: Optional[str] = None,
    analogy_dataset: Optional[str] = None,
    name: Optional[str] = None,
    mode: str = "resume",
    workers: Optional[int] = None,
    workers_per_model: int = 4,
    temp_dir: Optional[str] = None,
    keep_models: bool = True,
    progress_path: Optional[str] = None,
    train_fn: Optional[TrainFn] = None,
) -> str:
    """
    Train a grid of word2vec models on sampled years and score them.

    Args:
        ngram_size: N-gram size of the pivoted database to train on (2-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng-us")
        db_path_stub: Base directory of the corpora
        years: The study period, [first, last]: sampled years are drawn from it
        sample_years: The years to train on, given explicitly; adjacent pairs
            (1970, 1971, ...) let adjacent-year stability be measured
        year_pairs: Without sample_years, this many adjacent pairs spread
            evenly over the study period (4 -> 8 years)
        vector_size: Grid axis: dimensions of the vectors (one or more values)
        window: Grid axis: context words on each side
        min_count: Grid axis: words seen fewer times in a year are left out
        epochs: Grid axis: passes over the corpus
        approach: Grid axis: skip-gram or cbow
        weight_by: Grid axis: freq (repeat an n-gram log5 of its count times),
            doc_freq (the same on its volume count) or none (once)
        seeds: Models trained per cell and year with different seeds (2 or
            more measures seed stability)
        split_half: Also train one model on each random half of the year's
            corpus and score their agreement (split-half reliability)
        unk_mode: What to do with <UNK> tokens: strip (drop them, keep the
            n-gram if two real tokens remain), reject (drop the n-gram) or
            retain (keep <UNK> as a token)
        neighbours: Neighbours per probe word compared between models
        probes_per_decile: Probe words sampled from each frequency decile
        similarity_dataset: Word-pair similarity file (default: WordSim-353
            and SimLex-999 as bundled with gensim)
        analogy_dataset: Analogy questions file (default: gensim's
            questions-words.txt)
        name: Name of this search; its directory is
            <corpus>/tuning/<name>/ (default: a timestamp)
        mode: "resume" keeps models already trained; "restart" deletes the
            search directory first
        workers: CPUs to use in total (default: all)
        workers_per_model: Threads per model; workers / workers_per_model
            models train at once
        temp_dir: Where corpus files are written (default: inside the search
            directory; a job's $TMPDIR is faster)
        keep_models: Keep the trained models (.npz) after scoring
        progress_path: Progress document for the UI
    """
    paths = derive_paths(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)
    db_path = Path(paths["pivoted_db"])
    if not db_path.is_dir():
        raise FileNotFoundError(f"pivoted database not found: {db_path} (run the filter stage first)")
    if unk_mode not in UNK_MODES:
        raise ValueError(f"unk_mode must be one of {UNK_MODES}")
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    grid = Grid.from_settings({"vector_size": vector_size, "window": window, "min_count": min_count,
                               "epochs": epochs, "approach": approach, "weight_by": weight_by})
    cells = grid.cells()
    sampled = list(sample_years) if sample_years else sample_years_in(years, year_pairs)
    sampled = sorted(set(int(y) for y in sampled))
    if not sampled:
        raise ValueError("no years to sample")
    corpus_dir = db_path.parent
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = corpus_dir / "tuning" / name
    if mode == "restart" and out_dir.exists():
        shutil.rmtree(out_dir)
    elif mode not in ("resume", "restart"):
        raise ValueError("mode must be 'resume' or 'restart'")
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    spool = Path(temp_dir) if temp_dir else out_dir / "corpus"
    spool.mkdir(parents=True, exist_ok=True)

    total_cpus = workers or os.cpu_count() or 1
    per_model = max(1, int(workers_per_model))
    parallel = max(1, total_cpus // per_model)

    whitelist = derive_paths(db_path_stub, 1, repo_release_id, repo_corpus_id)["whitelist"]
    vocab_words = _read_vocabulary(whitelist) if Path(whitelist).is_file() else []
    sim_sets = _similarity_sets(similarity_dataset)
    ana_set = analogy_dataset or benchmarks.bundled_dataset("questions-words.txt")
    if not sim_sets:
        raise FileNotFoundError("no similarity dataset: none given and neither SimLex-999 nor WordSim-353 was "
                                "found (the image keeps them in $LEXICHRON_DATA_DIR; set similarity_dataset otherwise)")
    for label, path in sim_sets.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"similarity dataset not found: {path}")
    if ana_set and not Path(ana_set).is_file():
        raise FileNotFoundError(f"analogy dataset not found: {ana_set}")

    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    _setup_log(log_dir / f"tune_{datetime.now():%Y%m%d_%H%M%S}.log")
    inputs: Dict[str, Any] = {"source": str(db_path), "vocabulary": str(whitelist) if vocab_words else None}
    for label, path in sim_sets.items():
        inputs[f"{label}_dataset"] = path
    if ana_set:
        inputs["analogy_dataset"] = ana_set
    progress = ProgressReporter(progress_path, stage="tune", db_path=out_dir, unit_label="models", inputs=inputs)
    tasks = plan_models(cells, sampled, seeds, split_half)
    existing = {t.stem for t in tasks if (models_dir / f"{t.stem}.npz").exists()}
    progress.set_totals(len(tasks), files_skipped=len(existing))
    print(f"Hyperparameter search '{name}': {grid.describe()}; years {sampled}; "
          f"{seeds} seed(s){' + split halves' if split_half else ''} -> {len(tasks)} models "
          f"({len(existing)} already trained); {parallel} at a time x {per_model} threads")
    logger.info("grid %s; years %s; %d models; %d already trained", grid.describe(), sampled, len(tasks), len(existing))

    settings_record = {"grid": grid.as_settings(), "years": list(years), "sample_years": sampled,
                       "seeds": seeds, "split_half": split_half, "unk_mode": unk_mode,
                       "neighbours": neighbours, "probes_per_decile": probes_per_decile,
                       "similarity_datasets": sim_sets, "analogy_dataset": ana_set}
    (out_dir / "search.json").write_text(json.dumps(settings_record, indent=2))

    t_start = time.perf_counter()
    corpus_stats: Dict[Tuple[int, str], CorpusStats] = {}
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}      # (cell stem, year) -> result row
    try:
        for year in sampled:
            year_tasks = [t for t in tasks if t.year == year]
            todo = [t for t in year_tasks if t.stem not in existing]
            for t in year_tasks:
                if t.stem not in existing:
                    progress.file_started(t.stem)
            # corpus files: one pass per weighting, full + halves
            files: Dict[Tuple[str, str], Path] = {}
            if todo:
                with progress.long_phase("corpus"):
                    for wb in corpus_variants(cells):
                        full = spool / corpus_file_name(year, wb, unk_mode)
                        halves = (spool / corpus_file_name(year, wb, unk_mode, "a"),
                                  spool / corpus_file_name(year, wb, unk_mode, "b")) if split_half else None
                        if not full.exists() or (halves and not all(h.exists() for h in halves)):
                            st = write_corpus(str(db_path), year, str(full), weight_by=wb, unk_mode=unk_mode,
                                              halves=tuple(str(h) for h in halves) if halves else None,
                                              split_seed=year)
                            corpus_stats[(year, wb)] = st
                            logger.info("corpus %s: %s", full.name, st.as_dict())
                        files[(wb, "full")] = full
                        if halves:
                            files[(wb, "a")], files[(wb, "b")] = halves
                    progress.doc["counters"]["corpus_lines"] = progress.doc["counters"].get("corpus_lines", 0) + \
                        sum(s.lines for (y, _), s in corpus_stats.items() if y == year)
                    progress.doc["counters"]["ngrams_used"] = progress.doc["counters"].get("ngrams_used", 0) + \
                        sum(s.ngrams_used for (y, _), s in corpus_stats.items() if y == year)
            # train this year's models
            results: Dict[str, Vectors] = {}
            for t in year_tasks:
                if t.stem in existing:
                    results[t.stem] = load_vectors(str(models_dir / f"{t.stem}.npz"))
            if todo:
                _train_all(todo, files, models_dir, per_model, parallel, progress, train_fn, results)
            # score this year's cells
            for cell in cells:
                progress.file_ingesting(_stem(cell, year, "s0"))
                row = _score_year(cell, year, results, seeds, split_half, neighbours, probes_per_decile,
                                  vocab_words, sim_sets, ana_set)
                rows[(cell.stem, year)] = row
                for t in (x for x in year_tasks if x.cell == cell):
                    if t.stem not in existing:
                        progress.file_done(t.stem, entries=row.get("vocab_size", 0), chunks=0, uncompressed_bytes=0,
                                           counters={"models": 1})
            _write_results(out_dir / RESULTS_FILE, rows)
            # keep the seed-0 model of every cell for the year-pair comparisons; the rest may go
            if not keep_models:
                for t in year_tasks:
                    if t.kind != "s0":
                        (models_dir / f"{t.stem}.npz").unlink(missing_ok=True)
            if not temp_dir:
                for f in files.values():
                    f.unlink(missing_ok=True)
        # year pairs: adjacent and distant, per cell
        pairs = _score_pairs(cells, sampled, models_dir, neighbours, probes_per_decile)
        _write_pairs(out_dir / PAIRS_FILE, pairs)
        summary = summarize_results(rows, pairs, cells)
        _write_summary(out_dir / SUMMARY_FILE, out_dir / SUMMARY_JSON, summary)
        if not keep_models:
            shutil.rmtree(models_dir, ignore_errors=True)
        progress.doc["counters"]["cells"] = len(cells)
        progress.doc["counters"]["years"] = len(sampled)
        progress.finish("done", f"best cell: {summary['best_cell']}" if summary.get("best_cell") else "")
    except BaseException as exc:
        progress.finish("failed", str(exc))
        raise
    minutes = (time.perf_counter() - t_start) / 60
    print(f"Done in {minutes:.1f} min: {out_dir}")
    if summary.get("best_cell"):
        print(f"Suggested cell: {summary['best_cell']} ({summary['best_rule']})")
    return str(out_dir)


# ---------------------------------------------------------------- planning
class ModelTask:
    """One model to train: a cell, a year, and a kind: ``s<i>`` (seed i on
    the full corpus) or ``ha``/``hb`` (seed 0 on a half)."""
    __slots__ = ("cell", "year", "kind")

    def __init__(self, cell: Cell, year: int, kind: str) -> None:
        self.cell, self.year, self.kind = cell, year, kind

    @property
    def stem(self) -> str:
        return _stem(self.cell, self.year, self.kind)

    @property
    def seed(self) -> int:
        return int(self.kind[1:]) if self.kind.startswith("s") else 0

    @property
    def corpus_part(self) -> str:
        return {"ha": "a", "hb": "b"}.get(self.kind, "full")


def _stem(cell: Cell, year: int, kind: str) -> str:
    return f"y{year}_{cell.stem}_{kind}"


def plan_models(cells: Sequence[Cell], years: Sequence[int], seeds: int, split_half: bool) -> List[ModelTask]:
    tasks = []
    for year in years:
        for cell in cells:
            for s in range(seeds):
                tasks.append(ModelTask(cell, year, f"s{s}"))
            if split_half:
                tasks.append(ModelTask(cell, year, "ha"))
                tasks.append(ModelTask(cell, year, "hb"))
    return tasks


def sample_years_in(years: Tuple[int, int], pairs: int) -> List[int]:
    """``pairs`` adjacent year pairs spread evenly over [first, last]:
    (1968, 2019, 4) -> 1968, 1969, 1984, 1985, 2001, 2002, 2018, 2019."""
    first, last = int(years[0]), int(years[1])
    if last < first:
        raise ValueError("years must be [first, last] with first <= last")
    span = last - first
    if span < 1:
        return [first]
    pairs = max(1, int(pairs))
    if pairs == 1:
        starts = [first + (span - 1) // 2]
    else:
        starts = [round(first + i * (span - 1) / (pairs - 1)) for i in range(pairs)]
    out: List[int] = []
    for s in starts:
        for y in (s, s + 1):
            if first <= y <= last and y not in out:
                out.append(y)
    return out


sample_years = sample_years_in


# ---------------------------------------------------------------- training
def _train_one(args):
    """Runs in a worker process (or inline): train and save one model."""
    corpus_file, cell, seed, threads, out_path, train_fn = args
    res = train_model(corpus_file, cell, seed, threads, train_fn)
    save_vectors(res.vectors, out_path)
    return out_path, res.seconds, res.vocab_size


def _train_all(todo: List[ModelTask], files: Dict[Tuple[str, str], Path], models_dir: Path, per_model: int,
               parallel: int, progress: ProgressReporter, train_fn: Optional[TrainFn],
               results: Dict[str, Vectors]) -> None:
    jobs = []
    for t in todo:
        corpus_file = files[(t.cell.weight_by, t.corpus_part)]
        jobs.append((str(corpus_file), t.cell, t.seed, per_model, str(models_dir / f"{t.stem}.npz"), train_fn))
    if parallel == 1 or train_fn is not None:
        for t, job in zip(todo, jobs):
            progress.file_active(t.stem)
            out_path, secs, n = _train_one(job)
            logger.info("trained %s: %d words, %.0fs", t.stem, n, secs)
            results[t.stem] = load_vectors(out_path)
            progress.doc["counters"]["train_seconds"] = progress.doc["counters"].get("train_seconds", 0) + int(secs)
            progress.file_parsed(t.stem)
        return
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for t, job in zip(todo, jobs):
            futures[pool.submit(_train_one, job)] = t
            progress.file_active(t.stem)
        for fut in as_completed(futures):
            t = futures[fut]
            out_path, secs, n = fut.result()
            logger.info("trained %s: %d words, %.0fs", t.stem, n, secs)
            results[t.stem] = load_vectors(out_path)
            progress.doc["counters"]["train_seconds"] = progress.doc["counters"].get("train_seconds", 0) + int(secs)
            progress.file_parsed(t.stem)


# ---------------------------------------------------------------- scoring
def _score_year(cell: Cell, year: int, results: Dict[str, Vectors], seeds: int, split_half: bool,
                k: int, per_decile: int, vocab_words: List[str], sim_sets: Dict[str, str],
                ana_set: Optional[str]) -> Dict[str, Any]:
    main = results[_stem(cell, year, "s0")]
    row: Dict[str, Any] = {"cell": cell.stem, "year": year, **cell.as_dict(),
                           "vocab_size": len(main),
                           "coverage": benchmarks.coverage(main, vocab_words) if vocab_words else float("nan")}
    if seeds > 1:
        others = [results[_stem(cell, year, f"s{s}")] for s in range(1, seeds)]
        probes = stability.probe_words([main] + others, per_decile=per_decile, seed=year)
        row["seed_overlap"] = statistics.fmean(stability.neighbour_overlap(main, o, probes, k) for o in others)
        row["seed_cosine"] = statistics.fmean(stability.aligned_cosine(main, o, probes) for o in others)
    else:
        row["seed_overlap"] = row["seed_cosine"] = float("nan")
    if split_half:
        a, b = results[_stem(cell, year, "ha")], results[_stem(cell, year, "hb")]
        probes = stability.probe_words([a, b], per_decile=per_decile, seed=year)
        row["split_half_overlap"] = stability.neighbour_overlap(a, b, probes, k)
        row["split_half_cosine"] = stability.aligned_cosine(a, b, probes)
    else:
        row["split_half_overlap"] = row["split_half_cosine"] = float("nan")
    for label, path in sim_sets.items():
        rho, cov = benchmarks.similarity_score(main, path)
        row[f"{label}_rho"] = rho
        row[f"{label}_coverage"] = cov
    if ana_set:
        acc, cov = benchmarks.analogy_accuracy(main, ana_set)
        row["analogy_accuracy"] = acc
        row["analogy_coverage"] = cov
    return row


def _score_pairs(cells: Sequence[Cell], years: List[int], models_dir: Path, k: int,
                 per_decile: int) -> List[Dict[str, Any]]:
    """Adjacent pairs (y, y+1 both sampled) and the distant pair (first, last)
    per cell, on the seed-0 models."""
    pairs: List[Tuple[int, int, str]] = []
    for y in years:
        if y + 1 in years:
            pairs.append((y, y + 1, "adjacent"))
    if len(years) >= 2 and years[-1] - years[0] > 1:
        pairs.append((years[0], years[-1], "distant"))
    out = []
    cache: Dict[str, Vectors] = {}

    def model(cell: Cell, year: int) -> Optional[Vectors]:
        stem = _stem(cell, year, "s0")
        if stem not in cache:
            p = models_dir / f"{stem}.npz"
            cache[stem] = load_vectors(str(p)) if p.exists() else None
        return cache[stem]

    for cell in cells:
        for a, b, kind in pairs:
            ma, mb = model(cell, a), model(cell, b)
            if ma is None or mb is None:
                continue
            probes = stability.probe_words([ma, mb], per_decile=per_decile, seed=a * 10000 + b)
            out.append({"cell": cell.stem, "year_a": a, "year_b": b, "kind": kind,
                        "overlap": stability.neighbour_overlap(ma, mb, probes, k),
                        "cosine": stability.aligned_cosine(ma, mb, probes)})
        cache.clear()
    return out


def summarize_results(rows: Dict[Tuple[str, int], Dict[str, Any]], pairs: List[Dict[str, Any]],
                      cells: Sequence[Cell]) -> Dict[str, Any]:
    """Per-cell means over years, the year-pair figures, a signal-to-noise
    ratio, and the suggested cell."""
    per_cell: List[Dict[str, Any]] = []
    for cell in cells:
        cell_rows = [r for (stem, _), r in rows.items() if stem == cell.stem]
        if not cell_rows:
            continue
        s: Dict[str, Any] = {"cell": cell.stem, **cell.as_dict(), "years": len(cell_rows)}
        for key in ("vocab_size", "coverage", "seed_overlap", "seed_cosine", "split_half_overlap",
                    "split_half_cosine", "wordsim_rho", "simlex_rho", "analogy_accuracy"):
            vals = [r[key] for r in cell_rows if key in r and r[key] == r[key]]      # drop NaN
            s[key] = statistics.fmean(vals) if vals else float("nan")
        adj = [p["overlap"] for p in pairs if p["cell"] == cell.stem and p["kind"] == "adjacent" and p["overlap"] == p["overlap"]]
        dist = [p["overlap"] for p in pairs if p["cell"] == cell.stem and p["kind"] == "distant" and p["overlap"] == p["overlap"]]
        s["adjacent_overlap"] = statistics.fmean(adj) if adj else float("nan")
        s["distant_overlap"] = statistics.fmean(dist) if dist else float("nan")
        # change over the study span relative to change over one year: >1 means
        # the span moves neighbourhoods more than a year does
        noise = 1.0 - s["adjacent_overlap"] if adj else float("nan")
        signal = 1.0 - s["distant_overlap"] if dist else float("nan")
        s["signal_to_noise"] = (signal / noise) if adj and dist and noise > 0 else float("nan")
        per_cell.append(s)
    best, rule = _suggest(per_cell)
    return {"cells": per_cell, "best_cell": best, "best_rule": rule}


def _suggest(per_cell: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """The most stable cell among those within 0.05 of the best coverage and
    within 0.05 of the best similarity correlation (when measured)."""
    if not per_cell:
        return None, ""
    def ok(v):
        return v == v                                  # not NaN
    cov_best = max((c["coverage"] for c in per_cell if ok(c["coverage"])), default=None)
    rho_key = "simlex_rho" if any(ok(c.get("simlex_rho", float("nan"))) for c in per_cell) else "wordsim_rho"
    rho_best = max((c[rho_key] for c in per_cell if ok(c.get(rho_key, float("nan")))), default=None)
    eligible = [c for c in per_cell
                if (cov_best is None or not ok(c["coverage"]) or c["coverage"] >= cov_best - 0.05)
                and (rho_best is None or not ok(c.get(rho_key, float("nan"))) or c[rho_key] >= rho_best - 0.05)]
    if not eligible:
        eligible = per_cell

    def stab(c):
        parts = [c[k] for k in ("seed_overlap", "split_half_overlap", "adjacent_overlap") if ok(c[k])]
        return statistics.fmean(parts) if parts else -1.0
    best = max(eligible, key=stab)
    rule = ("most stable (mean of seed, split-half and adjacent-year neighbour overlap) among cells within "
            "0.05 of the best coverage" + (f" and of the best {rho_key.replace('_rho', '')} correlation" if rho_best is not None else ""))
    return best["cell"], rule


# ---------------------------------------------------------------- files
def _write_results(path: Path, rows: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    ordered = [rows[k] for k in sorted(rows, key=lambda k: (k[1], k[0]))]
    _write_tsv(path, ordered)


def _write_pairs(path: Path, pairs: List[Dict[str, Any]]) -> None:
    _write_tsv(path, pairs)


def _write_summary(tsv: Path, js: Path, summary: Dict[str, Any]) -> None:
    _write_tsv(tsv, summary["cells"])
    js.write_text(json.dumps(summary, indent=2, default=_json_num))


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(_cell_text(r.get(c)) for c in cols))
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def _cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4f}"
    return str(v)


def _json_num(v):
    if isinstance(v, float) and v != v:
        return None
    raise TypeError(str(type(v)))


def _read_vocabulary(path) -> List[str]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tok = line.split("\t")[0].split()[0] if line.split() else ""
            if tok:
                out.append(tok)
    return out


def _similarity_sets(given: Optional[str]) -> Dict[str, str]:
    if given:
        return {Path(given).stem.lower(): given}
    sets = {}
    for label, fname in (("wordsim", "wordsim353.tsv"), ("simlex", "simlex999.txt")):
        p = benchmarks.bundled_dataset(fname)
        if p:
            sets[label] = p
    return sets


def _setup_log(path: Path) -> None:
    root = logging.getLogger("lexichron.stages.tune")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)

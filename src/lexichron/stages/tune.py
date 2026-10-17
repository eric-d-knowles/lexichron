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
import multiprocessing
import os
import shutil
import statistics
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lexichron.inspection import read_tsv_rows
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
        workers_per_model: Threads per model, at least: workers /
            workers_per_model models train at once, and when a year has fewer
            models than that, each gets more threads so the CPUs stay busy
        temp_dir: Where corpus files are written (default: inside the search
            directory; a job's $TMPDIR is faster, and may be written as "$TMPDIR")
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
    spool = Path(os.path.expandvars(os.path.expanduser(temp_dir))) if temp_dir else out_dir / "corpus"   # $TMPDIR works
    spool.mkdir(parents=True, exist_ok=True)

    total_cpus = workers or os.cpu_count() or 1
    per_model = max(1, int(workers_per_model))
    parallel = max(1, total_cpus // per_model)

    def threads_for(n_models: int) -> int:
        # spare CPUs go to the models in flight: a one-cell search trains its
        # four models with every thread it has rather than four each
        return max(per_model, total_cpus // max(1, min(n_models, parallel)))

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
    log_file = log_dir / f"tune_{datetime.now():%Y%m%d_%H%M%S}.log"
    _setup_log(log_file)
    inputs: Dict[str, Any] = {"source": str(db_path), "vocabulary": str(whitelist) if vocab_words else None}
    for label, path in sim_sets.items():
        inputs[f"{label}_dataset"] = path
    if ana_set:
        inputs["analogy_dataset"] = ana_set
    progress = ProgressReporter(progress_path, stage="tune", db_path=out_dir, log_path=log_file,
                                unit_label="models", inputs=inputs)
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
    # results of an earlier run of this search, so a resume rescores only
    # what it retrains; both tables are rewritten after every addition
    rows = _read_results(out_dir / RESULTS_FILE, sampled)         # (cell stem, year) -> row
    pairs = _read_pairs(out_dir / PAIRS_FILE, sampled)            # (cell stem, year a, year b) -> row
    kinds = [f"s{s}" for s in range(seeds)] + (["ha", "hb"] if split_half else [])
    trained: set = set()                                          # stems trained in this run
    variants = corpus_variants(cells)
    # One process writes corpus files; the next year's are written while this
    # year's models train, so the (single-threaded) writing overlaps training.
    # (with a stand-in trainer, as in the tests, the files are written inline:
    # the stand-in's database lives in this process)
    writer = None if train_fn is not None else \
        ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pending: Dict[int, List[Future]] = {}

    def needs_corpus(year: int) -> bool:
        return any(t.stem not in existing for t in tasks if t.year == year)

    def corpus_paths(year: int) -> Dict[Tuple[str, str], Path]:
        files: Dict[Tuple[str, str], Path] = {}
        for wb in variants:
            files[(wb, "full")] = spool / corpus_file_name(year, wb, unk_mode)
            if split_half:
                files[(wb, "a")] = spool / corpus_file_name(year, wb, unk_mode, "a")
                files[(wb, "b")] = spool / corpus_file_name(year, wb, unk_mode, "b")
        return files

    def submit_corpus(year: int) -> None:
        files = corpus_paths(year)
        futs = []
        for wb in variants:
            full = files[(wb, "full")]
            halves = (files[(wb, "a")], files[(wb, "b")]) if split_half else None
            if full.exists() and (not halves or all(h.exists() for h in halves)):
                continue
            args = (str(db_path), year, str(full))
            kw = dict(weight_by=wb, unk_mode=unk_mode, halves=tuple(str(h) for h in halves) if halves else None,
                      split_seed=year)
            if writer is None:
                fut: Future = Future()
                fut.set_result(write_corpus(*args, **kw))
                futs.append(fut)
            else:
                futs.append(writer.submit(write_corpus, *args, **kw))
        pending[year] = futs

    def pairs_ending(year: int) -> List[Tuple[int, int, str]]:
        """The year pairs whose later year is ``year``: the adjacent pair, and
        the distant pair (first sampled year, last) once the last is reached."""
        out = []
        if year - 1 in sampled:
            out.append((year - 1, year, "adjacent"))
        if year == sampled[-1] and year - sampled[0] > 1:
            out.append((sampled[0], year, "distant"))
        return out

    def score_cell(cell: Cell, year: int) -> None:
        """Score a cell as soon as its models for the year are all on disk:
        the row (loading just this cell's models, then dropping them) and
        the year pairs that end at this year. Skipped when an earlier run
        scored it and nothing of it was retrained."""
        key = (cell.stem, year)
        stems = {kind: _stem(cell, year, kind) for kind in kinds}
        if key not in rows or any(s in trained for s in stems.values()):
            progress.file_ingesting(stems["s0"])
            models = {kind: load_vectors(str(models_dir / f"{s}.npz")) for kind, s in stems.items()}
            rows[key] = _score_year(cell, year, models, seeds, split_half, neighbours, probes_per_decile,
                                    vocab_words, sim_sets, ana_set)
            del models
            _write_results(out_dir / RESULTS_FILE, rows)
            logger.info("scored %s y%s: %s", cell.stem, year,
                        {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[key].items()
                         if k in ("vocab_size", "coverage", "seed_overlap", "split_half_overlap")})
        for s in stems.values():
            if s in trained:
                progress.file_done(s, entries=rows[key].get("vocab_size", 0), chunks=0, uncompressed_bytes=0,
                                   counters={"models": 1})
        for a, b, kind in pairs_ending(year):
            pk = (cell.stem, a, b)
            if pk not in pairs or stems["s0"] in trained or _stem(cell, a, "s0") in trained:
                row = _score_pair(cell, a, b, kind, models_dir, neighbours, probes_per_decile)
                if row is not None:
                    pairs[pk] = row
                    _write_pairs(out_dir / PAIRS_FILE, pairs)
                    logger.info("scored %s %s pair y%s-y%s: overlap %.4f", cell.stem, kind, a, b, row["overlap"])
        # the summary follows every addition, so the per-cell table never lags the rows
        _write_summary(out_dir / SUMMARY_FILE, out_dir / SUMMARY_JSON,
                       summarize_results(rows, list(pairs.values()), cells))

    summary: Dict[str, Any] = {}
    try:
        todo_years = [y for y in sampled if needs_corpus(y)]
        if todo_years:
            submit_corpus(todo_years[0])
        for i, year in enumerate(sampled):
            year_tasks = [t for t in tasks if t.year == year]
            todo = [t for t in year_tasks if t.stem not in existing]
            for t in todo:
                progress.file_started(t.stem)
            files = corpus_paths(year)
            if todo:
                # wait for this year's corpus files (written ahead, or now)
                if year not in pending:
                    submit_corpus(year)
                with progress.long_phase("corpus"):
                    for fut in pending.pop(year):
                        st = fut.result()
                        corpus_stats[(year, st.weight_by)] = st
                        logger.info("corpus y%s %s: %s", year, st.weight_by, st.as_dict())
                progress.doc["counters"]["corpus_lines"] = progress.doc["counters"].get("corpus_lines", 0) + \
                    sum(s.lines for (y, _), s in corpus_stats.items() if y == year)
                progress.doc["counters"]["ngrams_used"] = progress.doc["counters"].get("ngrams_used", 0) + \
                    sum(s.ngrams_used for (y, _), s in corpus_stats.items() if y == year)
                # start writing the next year's files before training this year's models
                later = [y for y in sampled[i + 1:] if needs_corpus(y) and y not in pending]
                if later:
                    submit_corpus(later[0])
            # which of each cell's models for this year are on disk already
            ready: Dict[str, set] = {cell.stem: set() for cell in cells}
            for t in year_tasks:
                if t.stem in existing:
                    ready[t.cell.stem].add(t.kind)
            for cell in cells:                           # complete from an earlier run: score if needed
                if len(ready[cell.stem]) == len(kinds):
                    score_cell(cell, year)

            def on_trained(t: ModelTask, year: int = year, ready: Dict[str, set] = ready) -> None:
                trained.add(t.stem)
                ready[t.cell.stem].add(t.kind)
                if len(ready[t.cell.stem]) == len(kinds):
                    score_cell(t.cell, year)             # while the other cells' models still train

            if todo:
                with progress.long_phase("training"):
                    _train_all(todo, files, models_dir, threads_for(len(todo)), parallel, progress, train_fn,
                               on_trained)
            summary = summarize_results(rows, list(pairs.values()), cells)
            _write_summary(out_dir / SUMMARY_FILE, out_dir / SUMMARY_JSON, summary)
            # keep the seed-0 model of every cell for the year-pair comparisons; the rest may go
            if not keep_models:
                for t in year_tasks:
                    if t.kind != "s0":
                        (models_dir / f"{t.stem}.npz").unlink(missing_ok=True)
            if not temp_dir:
                for f in files.values():
                    f.unlink(missing_ok=True)
        if not keep_models:
            shutil.rmtree(models_dir, ignore_errors=True)
        progress.doc["counters"]["cells"] = len(cells)
        progress.doc["counters"]["years"] = len(sampled)
        progress.finish("done", f"best cell: {summary['best_cell']}" if summary.get("best_cell") else "")
    except BaseException as exc:
        progress.finish("failed", str(exc))
        raise
    finally:
        if writer is not None:
            writer.shutdown(wait=False, cancel_futures=True)
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
               on_trained: Callable[[ModelTask], None]) -> None:
    """Train the models, ``parallel`` at a time with ``per_model`` threads
    each, calling ``on_trained`` in this process as each finishes (the model
    is on disk; nothing is kept here)."""
    logger.info("training %d models, %d at a time x %d threads", len(todo), min(parallel, len(todo)), per_model)
    jobs = []
    for t in todo:
        corpus_file = files[(t.cell.weight_by, t.corpus_part)]
        jobs.append((str(corpus_file), t.cell, t.seed, per_model, str(models_dir / f"{t.stem}.npz"), train_fn))

    def finished(t: ModelTask, secs: float, n: int) -> None:
        logger.info("trained %s: %d words, %.0fs", t.stem, n, secs)
        progress.doc["counters"]["train_seconds"] = progress.doc["counters"].get("train_seconds", 0) + int(secs)
        progress.file_parsed(t.stem)
        on_trained(t)

    if parallel == 1 or train_fn is not None:
        for t, job in zip(todo, jobs):
            progress.file_active(t.stem)
            _, secs, n = _train_one(job)
            finished(t, secs, n)
        return
    # spawn, not fork: the parent holds numpy/BLAS and RocksDB threads, which a
    # forked child can inherit in a locked state
    with ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {}
        for t, job in zip(todo, jobs):
            futures[pool.submit(_train_one, job)] = t
            progress.file_active(t.stem)
        progress._write(force=True)
        for fut in as_completed(futures):
            t = futures[fut]
            _, secs, n = fut.result()
            finished(t, secs, n)


# ---------------------------------------------------------------- scoring
def _score_year(cell: Cell, year: int, models: Dict[str, Vectors], seeds: int, split_half: bool,
                k: int, per_decile: int, vocab_words: List[str], sim_sets: Dict[str, str],
                ana_set: Optional[str]) -> Dict[str, Any]:
    """The row for one cell and year from its models, keyed by kind (``s0``,
    ``s1``, ..., ``ha``, ``hb``)."""
    main = models["s0"]
    row: Dict[str, Any] = {"cell": cell.stem, "year": year, **cell.as_dict(),
                           "vocab_size": len(main),
                           "coverage": benchmarks.coverage(main, vocab_words) if vocab_words else float("nan")}
    if seeds > 1:
        others = [models[f"s{s}"] for s in range(1, seeds)]
        probes = stability.probe_words([main] + others, per_decile=per_decile, seed=year)
        row["seed_overlap"] = statistics.fmean(stability.neighbour_overlap(main, o, probes, k) for o in others)
        row["seed_cosine"] = statistics.fmean(stability.aligned_cosine(main, o, probes) for o in others)
    else:
        row["seed_overlap"] = row["seed_cosine"] = float("nan")
    if split_half:
        a, b = models["ha"], models["hb"]
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


def _score_pair(cell: Cell, a: int, b: int, kind: str, models_dir: Path, k: int,
                per_decile: int) -> Optional[Dict[str, Any]]:
    """Agreement between the seed-0 models of two years (``adjacent`` or
    ``distant``), loaded for the comparison and dropped; None when either
    model is not on disk."""
    pa, pb = models_dir / f"{_stem(cell, a, 's0')}.npz", models_dir / f"{_stem(cell, b, 's0')}.npz"
    if not (pa.exists() and pb.exists()):
        return None
    ma, mb = load_vectors(str(pa)), load_vectors(str(pb))
    probes = stability.probe_words([ma, mb], per_decile=per_decile, seed=a * 10000 + b)
    return {"cell": cell.stem, "year_a": a, "year_b": b, "kind": kind,
            "overlap": stability.neighbour_overlap(ma, mb, probes, k),
            "cosine": stability.aligned_cosine(ma, mb, probes)}


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
    # only cells scored on the same years are comparable: a cell added to a
    # grid later, with its early (more stable) years in, must not win on them
    most_years = max(c.get("years", 0) for c in per_cell)
    per_cell = [c for c in per_cell if c.get("years", 0) == most_years]
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
            "0.05 of the best coverage" + (f" and of the best {rho_key.replace('_rho', '')} correlation" if rho_best is not None else "")
            + f", of the cells scored on {most_years} year(s)")
    return best["cell"], rule


# ---------------------------------------------------------------- files
def _write_results(path: Path, rows: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    ordered = [rows[k] for k in sorted(rows, key=lambda k: (k[1], k[0]))]
    _write_tsv(path, ordered)


def _write_pairs(path: Path, pairs: Dict[Tuple[str, int, int], Dict[str, Any]]) -> None:
    ordered = [pairs[k] for k in sorted(pairs, key=lambda k: (k[1], k[2], k[0]))]
    _write_tsv(path, ordered)


def _read_results(path: Path, years: Sequence[int]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Rows of an earlier run, for the sampled years only."""
    return {(r["cell"], int(r["year"])): r for r in read_tsv_rows(path)
            if "cell" in r and r.get("year") in years}


def _read_pairs(path: Path, years: Sequence[int]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    return {(r["cell"], int(r["year_a"]), int(r["year_b"])): r for r in read_tsv_rows(path)
            if "cell" in r and r.get("year_a") in years and r.get("year_b") in years}


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

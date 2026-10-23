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

A search too long for one job is split over the tasks of a Slurm job array
(``array_task`` / ``array_count``, which the batch script sets): each task
trains and scores its share of the sampled years into the same search
directory, and a year pair is scored by whichever task finds both years'
models on disk.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import statistics
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from lexichron.arrays import ArrayTask, task_share
from lexichron.arrays import array_task as _array_task
from lexichron.resources import available_cpus
from lexichron.training import axis, benchmarks, stability
from lexichron.training.corpus import UNK_MODES, CorpusStats, corpus_file_name, write_corpus
from lexichron.training.grid import AXES, Cell, Grid, corpus_variants
from lexichron.training.scheduler import train_across_years
from lexichron.training.scores import (PAIRS_FILE, RESULTS_FILE, SUMMARY_FILE, SUMMARY_JSON, ScoreStore,
                                       read_tsv_rows, write_json)
from lexichron.training.trainer import TrainFn, Vectors, load_vectors, save_vectors, train_model
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths

__all__ = ["tune_models", "sample_years", "plan_models", "summarize_results"]

logger = logging.getLogger(__name__)


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
    axis_lexicon: Optional[str] = None,
    axis_poles: Optional[str] = None,
    name: Optional[str] = None,
    mode: str = "resume",
    workers: Optional[int] = None,
    workers_per_model: int = 4,
    temp_dir: Optional[str] = None,
    keep_models: bool = True,
    array_task: Optional[int] = None,
    array_count: Optional[int] = None,
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
        axis_lexicon: Valence ratings (word, rating per line) for the axis
            scores: how well words projected onto the model's pleasant-
            unpleasant axis match human ratings, and how reliably models
            place them (default: the VADER lexicon the image carries; "none"
            skips the axis scores, as does a blank with no lexicon bundled)
        axis_poles: Two lines of pole words, pleasant then unpleasant
            (default: the implicit-association lists)
        name: Name of this search; its directory is
            <corpus>/tuning/<name>/ (default: a timestamp)
        mode: "resume" keeps models already trained; "restart" deletes the
            search directory first
        workers: CPUs to use in total (default: all)
        workers_per_model: Threads per model, at least: workers /
            workers_per_model models train at once, across years (a slot
            freed by one year's model takes the next year's), and when fewer
            models remain than that, each gets more threads so the CPUs
            stay busy
        temp_dir: Where corpus files are written (default: inside the search
            directory; a job's $TMPDIR is faster, and may be written as "$TMPDIR")
        keep_models: Keep the trained models (.npz) after scoring
        array_task: This run's task in a job array (0-based), with
            array_count: it trains and scores every array_count-th sampled
            year from this one. Set by the batch script from Slurm's
            $SLURM_ARRAY_TASK_ID; blank outside an array
        array_count: How many tasks the job array has
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
    task = _array_task(array_task, array_count)
    mine = task_share(sampled, task)
    if mode not in ("resume", "restart"):
        raise ValueError("mode must be 'resume' or 'restart'")
    if mode == "restart" and task is not None:
        raise ValueError("a search cannot be restarted by the tasks of a job array (each would delete what the "
                         "others are writing): delete the search directory first, then run the array in resume "
                         "mode, or run the restart as one job")
    corpus_dir = db_path.parent
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = corpus_dir / "tuning" / name
    if mode == "restart" and out_dir.exists():
        shutil.rmtree(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    spool = Path(os.path.expandvars(os.path.expanduser(temp_dir))) if temp_dir else out_dir / "corpus"   # $TMPDIR works
    spool.mkdir(parents=True, exist_ok=True)
    store = ScoreStore(out_dir)
    if store.import_tables():
        logger.info("imported the scores of an earlier version of this search into %s", store.scores_dir)

    total_cpus = workers or available_cpus()      # the allocation's CPUs, not the node's
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
    if isinstance(axis_lexicon, str) and axis_lexicon.strip().lower() in ("none", "off", "false"):
        lexicon = None                                   # switched off (an unset value is the default, not off)
    else:
        lexicon = axis_lexicon or benchmarks.bundled_dataset("vader_lexicon.txt")
    if lexicon and not Path(lexicon).is_file():
        raise FileNotFoundError(f"axis lexicon not found: {lexicon}")
    if axis_poles and not Path(axis_poles).is_file():
        raise FileNotFoundError(f"axis poles file not found: {axis_poles}")
    yard = axis.load_axis(lexicon, axis_poles)
    if yard is None:
        logger.warning("no valence lexicon (axis_lexicon unset and none bundled): the axis scores are skipped")

    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"tune_{datetime.now():%Y%m%d_%H%M%S}{'_' + task.suffix if task else ''}.log"
    _setup_log(log_file)
    inputs: Dict[str, Any] = {"source": str(db_path), "vocabulary": str(whitelist) if vocab_words else None}
    for label, path in sim_sets.items():
        inputs[f"{label}_dataset"] = path
    if ana_set:
        inputs["analogy_dataset"] = ana_set
    if yard is not None:
        inputs["axis_lexicon"] = yard.lexicon_path
        inputs["axis_poles"] = yard.poles_path or "built-in pleasant/unpleasant lists"
    progress = ProgressReporter(progress_path, stage="tune", db_path=out_dir, log_path=log_file,
                                unit_label="models", inputs=inputs)
    if task is not None:
        progress.doc["array"] = {"task": task.index, "count": task.count, "years": mine}
    tasks = plan_models(cells, mine, seeds, split_half)
    existing = {t.stem for t in tasks if (models_dir / f"{t.stem}.npz").exists()}
    progress.set_totals(len(tasks), files_skipped=len(existing))
    share = f" ({task.label}: years {mine})" if task else ""
    print(f"Hyperparameter search '{name}': {grid.describe()}; years {sampled}{share}; "
          f"{seeds} seed(s){' + split halves' if split_half else ''} -> {len(tasks)} models "
          f"({len(existing)} already trained); {parallel} at a time x {per_model} threads")
    logger.info("grid %s; years %s%s; %d models; %d already trained", grid.describe(), sampled, share,
                len(tasks), len(existing))

    settings_record = {"grid": grid.as_settings(), "years": list(years), "sample_years": sampled,
                       "seeds": seeds, "split_half": split_half, "unk_mode": unk_mode,
                       "neighbours": neighbours, "probes_per_decile": probes_per_decile,
                       "similarity_datasets": sim_sets, "analogy_dataset": ana_set,
                       "axis_lexicon": yard.lexicon_path if yard else None,
                       "axis_poles": yard.poles_path if yard else None}
    write_json(out_dir / "search.json", settings_record)
    if task is not None and not mine:
        progress.finish("done", f"{task.label}: no sampled year falls to this task")
        print(f"{task.label}: no sampled year falls to this task (there are {len(sampled)}); nothing to do")
        return str(out_dir)

    t_start = time.perf_counter()
    corpus_stats: Dict[Tuple[int, str], CorpusStats] = {}
    # results of earlier runs of this search (any task's), so a resume
    # rescores only what it retrains
    rows = store.load_rows(sampled)                               # (cell stem, year) -> row
    pairs = store.load_pairs(sampled)                             # (cell stem, year a, year b) -> row
    kinds = [f"s{s}" for s in range(seeds)] + (["ha", "hb"] if split_half else [])
    trained: set = set()                                          # stems trained in this run
    variants = corpus_variants(cells)
    # One process writes corpus files; the next year's are written while this
    # year's models train, so the (single-threaded) writing overlaps training.
    # (with a stand-in trainer, as in the tests, the files are written inline:
    # the stand-in's database lives in this process)
    writer = None if (train_fn is not None and _pool is _process_pool) else _pool(1)
    pending: Dict[Hashable, List[Future]] = {}

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

    def corpus_written(year: int, results: List[CorpusStats]) -> None:
        for st in results:
            corpus_stats[(year, st.weight_by)] = st
            logger.info("corpus y%s %s: %s", year, st.weight_by, st.as_dict())
        progress.doc["counters"]["corpus_lines"] = progress.doc["counters"].get("corpus_lines", 0) + \
            sum(s.lines for (y, _), s in corpus_stats.items() if y == year)
        progress.doc["counters"]["ngrams_used"] = progress.doc["counters"].get("ngrams_used", 0) + \
            sum(s.ngrams_used for (y, _), s in corpus_stats.items() if y == year)

    all_pairs = year_pairs_of(sampled)                            # every task's years: pairs cross tasks
    scored_cells: set = set()                                     # (cell stem, year) with a row this run
    paired_now: set = set()                                       # pairs scored in this run
    written: List[str] = []                                       # score files this run wrote
    written_pairs: List[str] = []

    def export() -> Dict[str, Any]:
        """The tables, from every score file on disk (a sibling task's too):
        the summary always, the row tables once this run has scored
        something or when one is missing (a run with nothing to rescore
        leaves them untouched)."""
        fresh_rows, fresh_pairs = store.load_rows(sampled), store.load_pairs(sampled)
        summary = summarize_results(fresh_rows, list(fresh_pairs.values()), cells)
        store.export(fresh_rows, fresh_pairs, summary,
                     rows_table=bool(written) or not (out_dir / RESULTS_FILE).exists(),
                     pairs_table=bool(written_pairs) or not (out_dir / PAIRS_FILE).exists())
        return summary

    def score_cell(cell: Cell, year: int) -> None:
        """Score a cell as soon as its models for the year are all on disk:
        the row (loading just this cell's models, then dropping them) and
        the year pairs that end at this year. Skipped when an earlier run
        scored it and nothing of it was retrained."""
        key = (cell.stem, year)
        stems = {kind: _stem(cell, year, kind) for kind in kinds}
        if key not in rows or any(s in trained for s in stems.values()) or _lacks_axis(rows[key], yard):
            progress.file_ingesting(stems["s0"])
            models = {kind: load_vectors(str(models_dir / f"{s}.npz")) for kind, s in stems.items()}
            rows[key] = _score_year(cell, year, models, seeds, split_half, neighbours, probes_per_decile,
                                    vocab_words, sim_sets, ana_set, yard)
            del models
            written.append(str(store.put_row(rows[key])))
            logger.info("scored %s y%s: %s", cell.stem, year,
                        {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[key].items()
                         if k in ("vocab_size", "coverage", "seed_overlap", "split_half_overlap")})
        for s in stems.values():
            if s in trained:
                progress.file_done(s, entries=rows[key].get("vocab_size", 0), chunks=0, uncompressed_bytes=0,
                                   counters={"models": 1})
        scored_cells.add(key)
        score_pairs(cell)
        # the tables follow every addition, so a reader of the tables never lags the rows
        export()

    def score_pairs(cell: Cell) -> None:
        """The cell's year pairs whose two seed-0 models are both final: on
        disk from an earlier run or a sibling task, or trained in this one.
        Years train side by side, so either model may land first; the pair
        is scored when the second does (or up front, on a resume, when an
        earlier run left it)."""
        for a, b, kind in all_pairs:
            pk = (cell.stem, a, b)
            sa, sb = _stem(cell, a, "s0"), _stem(cell, b, "s0")
            if pk in paired_now:
                continue
            if not all(st in trained or (models_dir / f"{st}.npz").exists() for st in (sa, sb)):
                continue                                 # on disk: this grid's, or an earlier grid's
            if pk in pairs and sa not in trained and sb not in trained and not _lacks_axis(pairs[pk], yard):
                continue                                 # scored by an earlier run, nothing retrained
            row = _score_pair(cell, a, b, kind, models_dir, neighbours, probes_per_decile, yard)
            if row is not None:
                pairs[pk] = row
                paired_now.add(pk)
                written_pairs.append(str(store.put_pair(row)))
                logger.info("scored %s %s pair y%s-y%s: overlap %.4f", cell.stem, kind, a, b, row["overlap"])

    summary: Dict[str, Any] = {}
    partial = False
    scored_years: List[int] = []
    try:
        todo = [t for t in tasks if t.stem not in existing]           # in year order
        for t in todo:
            progress.file_started(t.stem)
        # cells complete from an earlier run are scored first (only if unscored)
        ready: Dict[Tuple[str, int], set] = {(cell.stem, y): set() for cell in cells for y in mine}
        for t in tasks:
            if t.stem in existing:
                ready[(t.cell.stem, t.year)].add(t.kind)
        for year in mine:
            for cell in cells:
                if len(ready[(cell.stem, year)]) == len(kinds):
                    score_cell(cell, year)
        # cells of earlier grids of this search (rows, and seed-0 models on disk,
        # but not in this grid) get any year pairs they lack, e.g. the period
        # pairs a version before them did not score
        for stem in sorted({stem for stem, _ in rows} - {c.stem for c in cells}):
            try:
                score_pairs(Cell.from_stem(stem))
            except ValueError:
                pass

        def finish_year(year: int) -> None:
            """A year's models are all in: the summary, then what the pairs do
            not need goes (every model but seed 0, and the corpus)."""
            nonlocal summary
            summary = export()
            if not keep_models:
                for t in tasks:
                    if t.year == year and t.kind != "s0":
                        (models_dir / f"{t.stem}.npz").unlink(missing_ok=True)
            if not temp_dir:
                for f in corpus_paths(year).values():
                    f.unlink(missing_ok=True)

        for year in mine:
            if not any(t.year == year for t in todo):
                finish_year(year)

        def on_trained(t: ModelTask, secs: float, n: int) -> None:
            trained.add(t.stem)
            ready[(t.cell.stem, t.year)].add(t.kind)
            if len(ready[(t.cell.stem, t.year)]) == len(kinds):
                score_cell(t.cell, t.year)               # while the other models still train
            else:
                score_pairs(t.cell)                      # a seed-0 model a scored later year was waiting for

        if todo:
            inline = parallel == 1 or (train_fn is not None and _pool is _process_pool)
            with progress.long_phase("training"):
                train_across_years(todo, models_dir=models_dir, parallel=parallel, threads_for=threads_for,
                                   corpus_file=lambda t: corpus_paths(t.year)[(t.cell.weight_by, t.corpus_part)],
                                   submit_corpus=submit_corpus, pending=pending, on_corpus_written=corpus_written,
                                   on_trained=on_trained, on_corpus_done=lambda key: None,
                                   on_year_done=finish_year, progress=progress, train_fn=train_fn,
                                   pool_factory=_pool, inline=inline)
        # a sibling task's seed-0 model that landed while this task's model
        # for the other year of a pair was already on disk: the pair is this
        # task's to score if the sibling has not (it is written either way)
        if task is not None:
            for cell in cells:
                score_pairs(cell)
            if written_pairs:
                export()
        if not keep_models:
            if task is None:
                shutil.rmtree(models_dir, ignore_errors=True)
            else:
                logger.info("keep_models is off: the seed-0 models stay for the year pairs of the other tasks")
        summary = export()
        progress.doc["counters"]["cells"] = len(cells)
        progress.doc["counters"]["years"] = len(mine)
        scored_years = sorted({y for _, y in store.load_rows(sampled)})
        partial = task is not None and len(scored_years) < len(sampled)
        note = (f"best cell so far: {summary['best_cell']} ({len(scored_years)} of {len(sampled)} years scored)"
                if partial else f"best cell: {summary['best_cell']}") if summary.get("best_cell") else ""
        progress.finish("done", note)
    except BaseException as exc:
        progress.finish("failed", str(exc))
        raise
    finally:
        if writer is not None:
            writer.shutdown(wait=False, cancel_futures=True)
    minutes = (time.perf_counter() - t_start) / 60
    print(f"Done in {minutes:.1f} min: {out_dir}")
    if summary.get("best_cell"):
        print(f"Suggested cell{' so far' if partial else ''}: {summary['best_cell']} ({summary['best_rule']})"
              + (f"; {len(scored_years)} of {len(sampled)} years scored, other tasks may still be running"
                 if partial else ""))
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

    @property
    def corpus_key(self) -> int:
        return self.year                                  # one write covers the year's files


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
def _process_pool(n: int) -> ProcessPoolExecutor:
    # spawn, not fork: the parent holds numpy/BLAS and RocksDB threads, which a
    # forked child can inherit in a locked state
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))


_pool = _process_pool      # tests swap in a thread pool to run the scheduler with a stand-in trainer


# ---------------------------------------------------------------- scoring
def _score_year(cell: Cell, year: int, models: Dict[str, Vectors], seeds: int, split_half: bool,
                k: int, per_decile: int, vocab_words: List[str], sim_sets: Dict[str, str],
                ana_set: Optional[str], yard: Optional[axis.Axis] = None) -> Dict[str, Any]:
    """The row for one cell and year from its models, keyed by kind (``s0``,
    ``s1``, ..., ``ha``, ``hb``)."""
    main = models["s0"]
    row: Dict[str, Any] = {"cell": cell.stem, "year": year, **cell.as_dict(),
                           "vocab_size": len(main),
                           "coverage": benchmarks.coverage(main, vocab_words) if vocab_words else float("nan")}
    # overlap scores keep their standard error across probe words (``_se``),
    # the precision of the per-year figure
    if seeds > 1:
        others = [models[f"s{s}"] for s in range(1, seeds)]
        probes = stability.probe_words([main] + others, per_decile=per_decile, seed=year)
        per_probe = [stability.neighbour_overlaps(main, o, probes, k) for o in others]
        usable = [x for x in per_probe if x.size]
        pooled = np.mean(usable, axis=0) if usable and len({x.size for x in usable}) == 1 else \
            np.concatenate(usable) if usable else np.empty(0)
        row["seed_overlap"] = float(pooled.mean()) if pooled.size else float("nan")
        row["seed_overlap_se"] = stability.standard_error(pooled)
        row["seed_cosine"] = statistics.fmean(stability.aligned_cosine(main, o, probes) for o in others)
    else:
        row["seed_overlap"] = row["seed_overlap_se"] = row["seed_cosine"] = float("nan")
    if split_half:
        a, b = models["ha"], models["hb"]
        probes = stability.probe_words([a, b], per_decile=per_decile, seed=year)
        per_probe = stability.neighbour_overlaps(a, b, probes, k)
        row["split_half_overlap"] = float(per_probe.mean()) if per_probe.size else float("nan")
        row["split_half_overlap_se"] = stability.standard_error(per_probe)
        row["split_half_cosine"] = stability.aligned_cosine(a, b, probes)
    else:
        row["split_half_overlap"] = row["split_half_overlap_se"] = row["split_half_cosine"] = float("nan")
    for label, path in sim_sets.items():
        rho, cov = benchmarks.similarity_score(main, path)
        row[f"{label}_rho"] = rho
        row[f"{label}_coverage"] = cov
    if ana_set:
        acc, cov = benchmarks.analogy_accuracy(main, ana_set)
        row["analogy_accuracy"] = acc
        row["analogy_coverage"] = cov
    if yard is not None:
        # the valence axis as a yardstick: validity against human ratings, and
        # the same reliability contrasts as the neighbourhoods, for projections
        row["axis_rho"], row["axis_coverage"] = axis.validity(main, yard)
        row["axis_seed"] = statistics.fmean(axis.agreement(main, models[f"s{i}"], yard) for i in range(1, seeds)) \
            if seeds > 1 else float("nan")
        row["axis_split_half"] = axis.agreement(models["ha"], models["hb"], yard) if split_half else float("nan")
    return row


def _lacks_axis(record: Dict[str, Any], yard: Optional[axis.Axis]) -> bool:
    """Whether a row or pair from an earlier run predates the axis scores
    this run would add (a resume then rescores it from the models on disk).
    A score file is self-describing: the key is there when the axis was
    scored, even when the score itself was NaN (too few of the lexicon's
    words in a small corpus), so a NaN is not rescored on every resume."""
    if yard is None:
        return False
    return ("axis_coverage" if "year" in record else "axis") not in record


def year_pairs_of(sampled: Sequence[int]) -> List[Tuple[int, int, str]]:
    """The year pairs a search compares, as (earlier, later, kind): every
    ``adjacent`` pair (noise: one year apart), every ``period`` pair (the
    first year of each adjacent pair against the others: change across the
    span, whatever its shape), and ``distant`` (first vs last sampled year)
    for reference."""
    out: List[Tuple[int, int, str]] = [(y - 1, y, "adjacent") for y in sampled if y - 1 in sampled]
    firsts = [y for y in sampled if y - 1 not in sampled]
    out += [(a, b, "period") for i, a in enumerate(firsts) for b in firsts[i + 1:] if b - a > 1]
    if sampled[-1] - sampled[0] > 1 and (sampled[0], sampled[-1], "period") not in out:
        out.append((sampled[0], sampled[-1], "distant"))
    return out


def _score_pair(cell: Cell, a: int, b: int, kind: str, models_dir: Path, k: int,
                per_decile: int, yard: Optional[axis.Axis] = None) -> Optional[Dict[str, Any]]:
    """Agreement between the seed-0 models of two years (``adjacent``,
    ``period`` or ``distant``), loaded for the comparison and dropped; None
    when either model is not on disk."""
    pa, pb = models_dir / f"{_stem(cell, a, 's0')}.npz", models_dir / f"{_stem(cell, b, 's0')}.npz"
    if not (pa.exists() and pb.exists()):
        return None
    ma, mb = load_vectors(str(pa)), load_vectors(str(pb))
    probes = stability.probe_words([ma, mb], per_decile=per_decile, seed=a * 10000 + b)
    row = {"cell": cell.stem, "year_a": a, "year_b": b, "kind": kind,
           "overlap": stability.neighbour_overlap(ma, mb, probes, k),
           "cosine": stability.aligned_cosine(ma, mb, probes)}
    if yard is not None:
        row["axis"] = axis.agreement(ma, mb, yard)
    return row


def summarize_results(rows: Dict[Tuple[str, int], Dict[str, Any]], pairs: List[Dict[str, Any]],
                      cells: Sequence[Cell]) -> Dict[str, Any]:
    """Per-cell means over years (with their spread across years, ``_sd``),
    the year-pair figures, the change ratio, and the suggested cell. Every cell with rows is summarized, the
    grid's cells first: a grid narrowed on resume keeps its earlier cells in
    the table (the rows are there; a search's tables are cumulative)."""
    known = {c.stem for c in cells}
    extra = []
    for stem, _ in rows:
        if stem not in known:
            known.add(stem)
            try:
                extra.append(Cell.from_stem(stem))
            except Exception:
                pass
    cells = list(cells) + sorted(extra, key=lambda c: c.stem)
    per_cell: List[Dict[str, Any]] = []
    for cell in cells:
        cell_rows = [r for (stem, _), r in rows.items() if stem == cell.stem]
        if not cell_rows:
            continue
        s: Dict[str, Any] = {"cell": cell.stem, **cell.as_dict(), "years": len(cell_rows)}
        rho_keys = sorted({k for r in cell_rows for k in r if k.endswith("_rho")})     # any similarity dataset
        for key in ("vocab_size", "coverage", "seed_overlap", "seed_cosine", "split_half_overlap",
                    "split_half_cosine", *rho_keys, "analogy_accuracy",
                    "axis_coverage", "axis_seed", "axis_split_half"):
            vals = [r[key] for r in cell_rows if key in r and r[key] == r[key]]      # drop NaN
            s[key] = statistics.fmean(vals) if vals else float("nan")
            s[f"{key}_sd"] = statistics.stdev(vals) if len(vals) > 1 else float("nan")   # spread across years
        by_kind: Dict[str, List[float]] = {"adjacent": [], "period": [], "distant": []}
        axis_by_kind: Dict[str, List[float]] = {"adjacent": [], "period": []}
        for p in pairs:
            if p["cell"] != cell.stem or p["kind"] not in by_kind:
                continue
            if p["overlap"] == p["overlap"]:
                by_kind[p["kind"]].append(p["overlap"])
            v = p.get("axis")
            if p["kind"] in axis_by_kind and isinstance(v, (int, float)) and v == v:
                axis_by_kind[p["kind"]].append(v)
        for kind, vals in by_kind.items():
            s[f"{kind}_overlap"] = statistics.fmean(vals) if vals else float("nan")
            s[f"{kind}_overlap_sd"] = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        for kind, vals in axis_by_kind.items():
            s[f"axis_{kind}"] = statistics.fmean(vals) if vals else float("nan")
            s[f"axis_{kind}_sd"] = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        a_noise = 1.0 - s["axis_adjacent"] if axis_by_kind["adjacent"] else float("nan")
        a_change = 1.0 - s["axis_period"] if axis_by_kind["period"] else float("nan")
        s["axis_change_ratio"] = (a_change / a_noise) if axis_by_kind["adjacent"] and axis_by_kind["period"] \
            and a_noise > 0 else float("nan")
        # change between periods relative to change between adjacent years: the
        # between/within ratio of a one-way analysis of the neighbourhood
        # dissimilarities, >1 when periods differ by more than noise, whatever
        # the shape of the change over the span
        noise = 1.0 - s["adjacent_overlap"] if by_kind["adjacent"] else float("nan")
        change = 1.0 - s["period_overlap"] if by_kind["period"] else float("nan")
        s["change_ratio"] = (change / noise) if by_kind["adjacent"] and by_kind["period"] and noise > 0 \
            else float("nan")
        per_cell.append(s)
    picks = suggest_by_family(per_cell)
    # the overall pick: the axis family's when axis scores exist, else the neighbourhoods'
    overall = picks.get("axis") or picks.get("neighbourhoods") or {}
    return {"cells": per_cell, "best_cell": overall.get("cell"), "best_rule": overall.get("rule", ""),
            "suggestions": picks}


def _suggest(per_cell: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """The neighbourhood family's pick (kept under this name for the tests
    and older callers); see :func:`suggest_by_family`."""
    picks = suggest_by_family(per_cell)
    chosen = picks.get("neighbourhoods") or {}
    return chosen.get("cell"), chosen.get("rule", "")


SB = lambda r: 2 * r / (1 + r) if r == r else r           # Spearman-Brown: half-corpus agreement -> full corpus


def suggest_by_family(per_cell: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One suggested cell per family of scores, each with the rule it was
    chosen by, so the tool's opinion is legible and family-specific:

    - ``benchmarks``: the best general-purpose space — highest mean of the
      benchmark correlations and accuracy (SimLex, WordSim, analogy, axis
      validity where measured) among cells within 0.02 of the best coverage.
    - ``neighbourhoods``: the most stable neighbourhoods — mean of seed,
      split-half and adjacent-year overlap — among cells within 0.05 of the
      best coverage and of the best similarity correlation.
    - ``axis``: the most reliable axis measurement — mean of the split-half
      agreement (Spearman-Brown corrected to the full corpus) and the
      adjacent-year agreement — among cells within 0.02 of the best coverage
      and 0.01 of the best axis validity; of the cells within one standard
      error of the best, the cheapest to train (fewest epochs, then
      dimensions). Only when axis scores exist.

    Only cells scored on the most years are compared: a cell added to a
    grid later, with its early (more stable) years in, must not win on them.
    """
    if not per_cell:
        return {}
    most_years = max(c.get("years", 0) for c in per_cell)
    cells = [c for c in per_cell if c.get("years", 0) == most_years]
    scope = f", of the cells scored on {most_years} year(s)"

    def ok(v):
        return isinstance(v, (int, float)) and v == v

    def best_of(key):
        return max((c[key] for c in cells if ok(c.get(key))), default=None)

    def within(c, key, best, tol):
        return best is None or not ok(c.get(key)) or c[key] >= best - tol

    out: Dict[str, Dict[str, Any]] = {}
    cov_best = best_of("coverage")

    # benchmarks
    bench_keys = [k for k in ("simlex_rho", "wordsim_rho", "analogy_accuracy", "axis_rho")
                  if any(ok(c.get(k)) for c in cells)]
    if bench_keys:
        eligible = [c for c in cells if within(c, "coverage", cov_best, 0.02)] or cells

        def bench(c):
            parts = [c[k] for k in bench_keys if ok(c.get(k))]
            return statistics.fmean(parts) if parts else -1.0
        best = max(eligible, key=bench)
        names = {"simlex_rho": "SimLex", "wordsim_rho": "WordSim", "analogy_accuracy": "analogy",
                 "axis_rho": "axis validity"}
        out["benchmarks"] = {"cell": best["cell"], "rule": "highest mean of " + ", ".join(names.get(k, k) for k in bench_keys)
                             + " among cells within 0.02 of the best coverage" + scope}

    # neighbourhoods (the original rule)
    rho_key = "simlex_rho" if any(ok(c.get("simlex_rho")) for c in cells) else "wordsim_rho"
    rho_best = best_of(rho_key)
    eligible = [c for c in cells if within(c, "coverage", cov_best, 0.05) and within(c, rho_key, rho_best, 0.05)] \
        or cells

    def stab(c):
        parts = [c[k] for k in ("seed_overlap", "split_half_overlap", "adjacent_overlap") if ok(c.get(k))]
        return statistics.fmean(parts) if parts else -1.0
    best = max(eligible, key=stab)
    out["neighbourhoods"] = {"cell": best["cell"], "rule": (
        "most stable (mean of seed, split-half and adjacent-year neighbour overlap) among cells within 0.05 of "
        "the best coverage" + (f" and of the best {rho_key.replace('_rho', '')} correlation" if rho_best is not None
                               else "") + scope)}

    # axis
    if any(ok(c.get("axis_split_half")) or ok(c.get("axis_adjacent")) for c in cells):
        rho_best = best_of("axis_rho")
        eligible = [c for c in cells if within(c, "coverage", cov_best, 0.02) and within(c, "axis_rho", rho_best, 0.01)] \
            or cells

        def reliability(c):
            parts = [v for v in (SB(c.get("axis_split_half", float("nan"))), c.get("axis_adjacent", float("nan")))
                     if ok(v)]
            return statistics.fmean(parts) if parts else -1.0

        def se(c):
            # the score's standard error from the recorded spreads across years / pairs
            n_years, n_pairs = max(1, c.get("years", 1)), max(1, c.get("years", 1) // 2)
            parts = [v for v in (c.get("axis_split_half_sd", float("nan")) / n_years ** 0.5,
                                 c.get("axis_adjacent_sd", float("nan")) / n_pairs ** 0.5) if ok(v)]
            return statistics.fmean(parts) if parts else 0.0
        top = max(eligible, key=reliability)
        floor = reliability(top) - se(top)
        tied = [c for c in eligible if reliability(c) >= floor]
        best = min(tied, key=lambda c: (c.get("epochs", 0), c.get("vector_size", 0), -reliability(c)))
        out["axis"] = {"cell": best["cell"], "rule": (
            "most reliable axis measurement (mean of split-half agreement, corrected to the full corpus, and "
            "adjacent-year agreement) among cells within 0.02 of the best coverage and 0.01 of the best axis "
            "validity; of the cells within one standard error of the best, the cheapest to train" + scope)}
    return out


# ---------------------------------------------------------------- files


# ---------------------------------------------------------------- files
def _read_results(path: Path, years: Sequence[int]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Rows of a search (its score files, or its table), for ``years``;
    ``path`` is the table's, as older callers passed it."""
    return ScoreStore(Path(path).parent).load_rows(years)


def _read_pairs(path: Path, years: Sequence[int]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    return ScoreStore(Path(path).parent).load_pairs(years)


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
        stem = Path(given).stem.lower()
        # a user's copy of a known dataset keeps its name, so its scores land
        # in the summary's SimLex / WordSim columns
        label = "simlex" if stem.startswith("simlex") else "wordsim" if stem.startswith("wordsim") else stem
        return {label: given}
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

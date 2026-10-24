"""``lexichron train``: the final models, as a replicate ensemble per year.

One hyperparameter cell (the one a tuning search suggested, or any other)
is trained for every year of the study: ``corpus_replicates`` Poisson
resamples of the year's corpus, each fitted with ``seeds_per_corpus``
seeds (see :mod:`lexichron.training.ensemble`). The per-replicate models
are the product, written as ``<corpus>/models/<name>/y1990/c03_s1.npz``
with a ``train.json`` manifest and, per year, a ``year.json`` of the
corpus and model facts and the year's noise profile: the valence
yardstick's projections measured across the replicates and decomposed
into their seed and corpus variance (the error of anything measured the
same way on these models). Nothing in the stage assumes a research
question; projecting your own words onto your own axis is a later stage.

A run this size spans jobs: ``mode: resume`` trains only the models not on
disk, and ``array_task`` / ``array_count`` (the Submit tab's *Jobs*) split
the years over the tasks of a job array, each writing its own years.
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
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from lexichron.arrays import task_share
from lexichron.arrays import array_task as _array_task
from lexichron.resources import available_cpus
from lexichron.training import axis, benchmarks
from lexichron.training.corpus import UNK_MODES, CorpusStats, write_corpus, years_with_data
from lexichron.training.ensemble import (RESAMPLINGS, ReplicaTask, plan_replicas, projection_cube,
                                         resample_seed, variance_components)
from lexichron.training.grid import AXIS_BY_NAME, Cell
from lexichron.training.scheduler import train_across_years
from lexichron.training.scores import write_json
from lexichron.training.trainer import TrainFn, load_vectors
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths

__all__ = ["train_models", "MANIFEST_FILE", "YEAR_FILE"]

logger = logging.getLogger(__name__)

MANIFEST_FILE = "train.json"
YEAR_FILE = "year.json"


def train_models(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
    db_path_stub: str,
    years: Tuple[int, int] = (1968, 2019),
    vector_size: int = 100,
    window: int = 2,
    min_count: int = 5,
    epochs: int = 10,
    approach: str = "skip-gram",
    weight_by: str = "none",
    corpus_replicates: int = 20,
    seeds_per_corpus: int = 2,
    resampling: str = "poisson",
    noise_seed: int = 0,
    unk_mode: str = "strip",
    axis_lexicon: Optional[str] = None,
    axis_poles: Optional[str] = None,
    name: Optional[str] = None,
    mode: str = "resume",
    workers: Optional[int] = None,
    workers_per_model: int = 4,
    corpus_writers: int = 2,
    temp_dir: Optional[str] = None,
    array_task: Optional[int] = None,
    array_count: Optional[int] = None,
    progress_path: Optional[str] = None,
    train_fn: Optional[TrainFn] = None,
) -> str:
    """
    Train the final models: a replicate ensemble of one cell for every year.

    Args:
        ngram_size: N-gram size of the pivoted database to train on (2-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng-us")
        db_path_stub: Base directory of the corpora
        years: The study period, [first, last]: every year of it with data
            in the database gets an ensemble
        vector_size: Dimensions of the vectors
        window: Context words on each side
        min_count: Words seen fewer times in a year's corpus are left out
        epochs: Passes over the corpus
        approach: skip-gram or cbow
        weight_by: freq (repeat an n-gram log5 of its count times), doc_freq
            (the same on its volume count) or none (once)
        corpus_replicates: Resamples of each year's corpus (1 with resampling
            none)
        seeds_per_corpus: Models trained on each resample, with different
            seeds; 2 or more separate the fit's noise from the text's
        resampling: poisson (every n-gram's repeat count redrawn from a
            Poisson with that mean: another same-sized sample of books) or
            none (the corpus as it is; the seeds are the only replicates)
        noise_seed: Seed every replicate's seeds derive from, so the ensemble
            is reproducible from this number
        unk_mode: What to do with <UNK> tokens: strip (drop them, keep the
            n-gram if two real tokens remain), reject (drop the n-gram) or
            retain (keep <UNK> as a token)
        axis_lexicon: Valence ratings (word, rating per line) for the noise
            profile (default: the VADER lexicon the image carries; "none"
            skips it)
        axis_poles: Two lines of pole words, pleasant then unpleasant
            (default: the implicit-association lists)
        name: Name of this ensemble; its directory is <corpus>/models/<name>/
            (default: a timestamp)
        mode: "resume" keeps models already trained; "restart" deletes the
            directory first
        workers: CPUs to use in total (default: all)
        workers_per_model: Threads per model, at least: workers /
            workers_per_model models train at once; the last few get the
            spare CPUs
        corpus_writers: Processes writing corpus resamples ahead of training
        temp_dir: Where corpus files are written (default: inside the
            directory; a job's $TMPDIR is faster, and may be written as "$TMPDIR")
        array_task: This run's task in a job array (0-based), with
            array_count: it trains every array_count-th year from this one.
            Set by the batch script from Slurm's $SLURM_ARRAY_TASK_ID; blank
            outside an array
        array_count: How many tasks the job array has
        progress_path: Progress document for the UI
    """
    paths = derive_paths(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)
    db_path = Path(paths["pivoted_db"])
    if not db_path.is_dir():
        raise FileNotFoundError(f"pivoted database not found: {db_path} (run the filter stage first)")
    if unk_mode not in UNK_MODES:
        raise ValueError(f"unk_mode must be one of {UNK_MODES}")
    if resampling not in RESAMPLINGS:
        raise ValueError(f"resampling must be one of {RESAMPLINGS}")
    cell = Cell(vector_size=AXIS_BY_NAME["vector_size"].parse(vector_size)[0],
                window=AXIS_BY_NAME["window"].parse(window)[0],
                min_count=AXIS_BY_NAME["min_count"].parse(min_count)[0],
                epochs=AXIS_BY_NAME["epochs"].parse(epochs)[0],
                approach=AXIS_BY_NAME["approach"].parse(approach)[0],
                weight_by=AXIS_BY_NAME["weight_by"].parse(weight_by)[0])
    corpora, seeds = int(corpus_replicates), int(seeds_per_corpus)
    if corpora < 1 or seeds < 1:
        raise ValueError("corpus_replicates and seeds_per_corpus must be at least 1")
    if resampling == "none" and corpora != 1:
        logger.info("resampling is none: one corpus per year (corpus_replicates %d ignored)", corpora)
        corpora = 1
    first, last = int(years[0]), int(years[1])
    if last < first:
        raise ValueError("years must be [first, last] with first <= last")
    task = _array_task(array_task, array_count)
    if mode not in ("resume", "restart"):
        raise ValueError("mode must be 'resume' or 'restart'")
    if mode == "restart" and task is not None:
        raise ValueError("an ensemble cannot be restarted by the tasks of a job array (each would delete what the "
                         "others are writing): delete the directory first, then run the array in resume mode, "
                         "or run the restart as one job")
    corpus_dir = db_path.parent
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = corpus_dir / "models" / name
    if mode == "restart" and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spool = Path(os.path.expandvars(os.path.expanduser(temp_dir))) if temp_dir else out_dir / "corpus"
    spool.mkdir(parents=True, exist_ok=True)

    total_cpus = workers or available_cpus()
    per_model = max(1, int(workers_per_model))
    parallel = max(1, total_cpus // per_model)

    def threads_for(n_models: int) -> int:
        return max(per_model, total_cpus // max(1, min(n_models, parallel)))

    if isinstance(axis_lexicon, str) and axis_lexicon.strip().lower() in ("none", "off", "false"):
        lexicon = None
    else:
        lexicon = axis_lexicon or benchmarks.bundled_dataset("vader_lexicon.txt")
    if lexicon and not Path(lexicon).is_file():
        raise FileNotFoundError(f"axis lexicon not found: {lexicon}")
    if axis_poles and not Path(axis_poles).is_file():
        raise FileNotFoundError(f"axis poles file not found: {axis_poles}")
    yard = axis.load_axis(lexicon, axis_poles)
    if yard is None:
        logger.warning("no valence lexicon (axis_lexicon unset and none bundled): the noise profile is skipped")

    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}{'_' + task.suffix if task else ''}.log"
    _setup_log(log_file)
    inputs: Dict[str, Any] = {"source": str(db_path)}
    if yard is not None:
        inputs["axis_lexicon"] = yard.lexicon_path
        inputs["axis_poles"] = yard.poles_path or "built-in pleasant/unpleasant lists"
    progress = ProgressReporter(progress_path, stage="train", db_path=out_dir, log_path=log_file,
                                unit_label="models", inputs=inputs)

    all_years = years_with_data(str(db_path), range(first, last + 1))
    skipped = (last - first + 1) - len(all_years)
    if not all_years:
        progress.finish("failed", f"no year of {first}-{last} has data in {db_path}")
        raise ValueError(f"no year of {first}-{last} has data in {db_path}")
    mine = task_share(all_years, task)
    manifest = {"cell": cell.as_dict(), "cell_stem": cell.stem, "years": [first, last], "years_with_data": all_years,
                "corpus_replicates": corpora, "seeds_per_corpus": seeds, "resampling": resampling,
                "noise_seed": int(noise_seed), "unk_mode": unk_mode,
                "axis_lexicon": yard.lexicon_path if yard else None, "axis_poles": yard.poles_path if yard else None,
                "layout": "y<year>/c<corpus>_s<seed>.npz"}
    write_json(out_dir / MANIFEST_FILE, manifest)
    if task is not None:
        progress.doc["array"] = {"task": task.index, "count": task.count, "years": mine}
    tasks = plan_replicas(cell, mine, corpora, seeds, int(noise_seed))
    for y in mine:
        (out_dir / f"y{y}").mkdir(exist_ok=True)
    existing = {t.stem for t in tasks if (out_dir / f"{t.stem}.npz").exists()}
    progress.set_totals(len(tasks), files_skipped=len(existing))
    progress.doc["counters"]["years"] = len(mine)
    share = f" ({task.label}: {len(mine)} years)" if task else ""
    print(f"Ensemble '{name}': {cell.stem}; years {first}-{last} ({len(all_years)} with data"
          f"{f', {skipped} without' if skipped else ''}){share}; {corpora} corpus x {seeds} seed replicates "
          f"= {corpora * seeds} models per year -> {len(tasks)} models ({len(existing)} already trained); "
          f"{parallel} at a time x {per_model} threads")
    logger.info("cell %s; years %s%s; %d models; %d already trained", cell.stem, mine, share, len(tasks),
                len(existing))
    if task is not None and not mine:
        progress.finish("done", f"{task.label}: no year falls to this task")
        print(f"{task.label}: no year falls to this task (there are {len(all_years)}); nothing to do")
        return str(out_dir)

    t_start = time.perf_counter()
    corpus_stats: Dict[Tuple[int, int], CorpusStats] = {}
    model_facts: Dict[str, Dict[str, Any]] = {}                     # stem -> seconds, vocab
    inline = parallel == 1 or (train_fn is not None and _pool is _process_pool)
    writer = None if (train_fn is not None and _pool is _process_pool) else _pool(max(1, int(corpus_writers)))
    pending: Dict[Hashable, List[Future]] = {}

    def corpus_path(key: Tuple[int, int]) -> Path:
        year, c = key
        return spool / f"corpus_y{year}_wb{cell.weight_by}_{unk_mode}_c{c:02d}.txt"

    def submit_corpus(key: Tuple[int, int]) -> None:
        year, c = key
        path = corpus_path(key)
        if path.exists():
            pending[key] = []
            return
        rs = resample_seed(int(noise_seed), year, c) if resampling == "poisson" else None
        args = (str(db_path), year, str(path))
        kw = dict(weight_by=cell.weight_by, unk_mode=unk_mode, resample_seed=rs)
        if writer is None:
            fut: Future = Future()
            fut.set_result(write_corpus(*args, **kw))
            pending[key] = [fut]
        else:
            pending[key] = [writer.submit(write_corpus, *args, **kw)]

    def corpus_written(key: Tuple[int, int], results: List[CorpusStats]) -> None:
        for st in results:
            corpus_stats[key] = st
            logger.info("corpus y%s c%02d: %s", key[0], key[1], st.as_dict())
            progress.doc["counters"]["corpus_lines"] = progress.doc["counters"].get("corpus_lines", 0) + st.lines

    def on_trained(t: ReplicaTask, secs: float, n: int) -> None:
        model_facts[t.stem] = {"model": t.name, "corpus": t.corpus, "seed": t.seed_index, "vocab_size": n,
                               "seconds": round(secs, 1)}
        progress.file_done(t.stem, entries=n, chunks=0, uncompressed_bytes=0, counters={"models": 1})

    def on_corpus_done(key: Tuple[int, int]) -> None:
        if not temp_dir:
            corpus_path(key).unlink(missing_ok=True)

    def finish_year(year: int) -> None:
        """A year's models are all in: its record, with the noise profile
        measured across them."""
        progress.file_ingesting(f"y{year}")
        record = _year_record(out_dir, year, cell, corpora, seeds, corpus_stats, model_facts, yard)
        write_json(out_dir / f"y{year}" / YEAR_FILE, record)
        progress.doc["counters"]["years_done"] = progress.doc["counters"].get("years_done", 0) + 1
        logger.info("year %s: %s", year, {k: v for k, v in record.items() if k.startswith("axis_") or k == "models_on_disk"})

    years_done: List[int] = []
    partial, note = False, ""
    try:
        todo = [t for t in tasks if t.stem not in existing]
        for t in todo:
            progress.file_started(t.stem)
        for year in mine:
            if not any(t.year == year for t in todo) and not (out_dir / f"y{year}" / YEAR_FILE).exists():
                finish_year(year)                     # complete from an earlier run, its record missing

        def year_done(year: int) -> None:
            finish_year(year)
            years_done.append(year)

        if todo:
            with progress.long_phase("training"):
                train_across_years(todo, models_dir=out_dir, parallel=parallel, threads_for=threads_for,
                                   corpus_file=lambda t: corpus_path(t.corpus_key),
                                   submit_corpus=submit_corpus, pending=pending, on_corpus_written=corpus_written,
                                   on_trained=on_trained, on_corpus_done=on_corpus_done, on_year_done=year_done,
                                   progress=progress, train_fn=train_fn, pool_factory=_pool, inline=inline,
                                   corpus_ahead=max(2, int(corpus_writers) + 1))
        done_all = sorted(int(p.parent.name[1:]) for p in out_dir.glob("y*/" + YEAR_FILE))
        partial = task is not None and len(done_all) < len(all_years)
        note = (f"{len(done_all)} of {len(all_years)} years complete" + (" so far" if partial else ""))
        progress.finish("done", note)
    except BaseException as exc:
        progress.finish("failed", str(exc))
        raise
    finally:
        if writer is not None:
            writer.shutdown(wait=False, cancel_futures=True)
    minutes = (time.perf_counter() - t_start) / 60
    print(f"Done in {minutes:.1f} min: {out_dir}" + (f"; {note}, other tasks may still be running" if partial else ""))
    return str(out_dir)


# ---------------------------------------------------------------- per year
def _year_record(out_dir: Path, year: int, cell: Cell, corpora: int, seeds: int,
                 corpus_stats: Dict[Tuple[int, int], CorpusStats], model_facts: Dict[str, Dict[str, Any]],
                 yard: Optional[axis.Axis]) -> Dict[str, Any]:
    """The year's ``year.json``: what was trained and the noise profile. The
    models are loaded one at a time from disk (a resume's earlier models
    included) and dropped."""
    year_dir = out_dir / f"y{year}"
    models = sorted(year_dir.glob("c*_s*.npz"))
    # facts of the models: this run's, else the earlier record's, else from the model itself
    earlier: Dict[str, Dict[str, Any]] = {}
    try:
        earlier = {m["model"]: m for m in json.loads((year_dir / YEAR_FILE).read_text()).get("models", [])}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    facts: List[Dict[str, Any]] = []
    proj: Dict[Tuple[int, int], Dict[str, float]] = {}
    for p in models:
        c, s = int(p.stem[1:3]), int(p.stem.split("_s")[1])
        fact = dict(model_facts.get(f"y{year}/{p.stem}") or earlier.get(p.stem)
                    or {"model": p.stem, "corpus": c, "seed": s})
        if yard is not None or "vocab_size" not in fact:
            v = load_vectors(str(p))
            fact.setdefault("vocab_size", len(v))
            if yard is not None and c < corpora and s < seeds:          # not a model of a larger earlier design
                ax = axis.axis_vector(v, yard.pleasant, yard.unpleasant)
                if ax is not None:
                    proj[(c, s)] = axis.projections(v, ax, yard.targets)
            del v
        facts.append(fact)
    corpora_now = [st.as_dict() for (y, _), st in sorted(corpus_stats.items()) if y == year]
    if not corpora_now:
        try:
            corpora_now = json.loads((year_dir / YEAR_FILE).read_text()).get("corpora", [])
        except (OSError, ValueError, AttributeError):
            corpora_now = []
    record: Dict[str, Any] = {"year": year, "cell": cell.stem, "corpus_replicates": corpora, "seeds_per_corpus": seeds,
                              "models_on_disk": len(models), "models": facts, "corpora": corpora_now}
    vocab = [f["vocab_size"] for f in facts if f.get("vocab_size") is not None]
    secs = [f["seconds"] for f in facts if f.get("seconds") is not None]
    record["vocab_size"] = round(statistics.fmean(vocab)) if vocab else None
    record["train_seconds"] = round(sum(secs)) if secs else None
    if yard is not None and models:
        if len(proj) == corpora * seeds:
            words, cube = projection_cube(proj, corpora, seeds)
            record.update(variance_components(cube).as_dict())
        else:
            record["axis_note"] = f"{len(proj)} of {corpora * seeds} models could build the axis"
    return record


def _process_pool(n: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"))


_pool = _process_pool      # tests swap in a thread pool to run the scheduler with a stand-in trainer


def _setup_log(path: Path) -> None:
    root = logging.getLogger("lexichron.stages.train")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)

"""``lexichron measure``: a time series, with its error, from an ensemble.

The train stage leaves one ensemble of models per year. This stage measures
one scalar of your choosing inside every one of those models (see
:mod:`lexichron.training.measures`: a projection onto an axis between two
sets of pole words, a cosine with a target word or set of words, a relative
frequency) for the words you name, and gives each word's value per year as
the ensemble mean with its standard error and the two parts of its variance,
the sample of text and the fit. Because every value is computed inside one
model, nothing is aligned: the per-replicate models are used as they are.

The output is a directory of files under the ensemble,
``models/<ensemble>/measures/<name>/``: ``measure.json`` (what was measured
and on what), one ``y<year>.json`` of per-word statistics and one
``y<year>.npz`` of the raw (corpus, seed, word) values per year, and
``series.tsv`` (long: year, word, mean, se, ...) and ``summary.tsv``
rebuilt from them after every year. A year whose models are not all on
disk yet is left pending; a later run in resume mode fills it in. Several
measures of one ensemble sit side by side under their names.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lexichron.arrays import array_task as _array_task
from lexichron.arrays import task_share
from lexichron.resources import available_cpus
from lexichron.training.ensemble import projection_cube, variance_components, word_components
from lexichron.training.measures import (KINDS, Measured, Spec, measure as measure_one, read_pole_lists,
                                         read_word_list, spec_digest)
from lexichron.training.scores import write_json
from lexichron.training.trainer import load_vectors
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths

__all__ = ["measure_series", "MANIFEST_FILE", "SERIES_FILE", "SUMMARY_FILE", "SERIES_COLUMNS", "year_file",
           "year_values_file"]

logger = logging.getLogger(__name__)

MANIFEST_FILE = "measure.json"
SERIES_FILE = "series.tsv"
SUMMARY_FILE = "summary.tsv"
SERIES_COLUMNS = ("year", "word", "mean", "se", "n", "var_seed", "var_corpus")
SUMMARY_COLUMNS = ("year", "models", "words_all", "words_some", "words_none", "se_mean", "noise_share",
                   "coverage", "note")
DEFAULT_WORKERS = 8         # models read at once; each is a whole model in memory


def year_file(out_dir: Path, year: int) -> Path:
    return out_dir / f"y{year}.json"


def year_values_file(out_dir: Path, year: int) -> Path:
    return out_dir / f"y{year}.npz"


def measure_series(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
    db_path_stub: str,
    ensemble: str,
    name: str,
    kind: str = "projection",
    words: Tuple[str, ...] = (),
    words_file: Optional[str] = None,
    positive: Tuple[str, ...] = (),
    negative: Tuple[str, ...] = (),
    poles_file: Optional[str] = None,
    target: Tuple[str, ...] = (),
    years: Optional[Tuple[int, int]] = None,
    mode: str = "resume",
    workers: Optional[int] = None,
    out_dir: Optional[str] = None,
    array_task: Optional[int] = None,
    array_count: Optional[int] = None,
    progress_path: Optional[str] = None,
) -> str:
    """Measure one scalar for a list of words inside every model of an
    ensemble, year by year, and write the series with its error.

    Args:
        ngram_size: N-gram size of the corpus
        repo_release_id: Google Books release (e.g. 20200217)
        repo_corpus_id: Language/corpus id (e.g. eng-us)
        db_path_stub: Root directory of the corpora
        ensemble: The ensemble to measure: its name under <corpus>/models/,
            or the directory itself
        name: Name of this measure; its files go to
            <ensemble>/measures/<name>/. One name holds one specification:
            to measure something else, use another name
        kind: What is measured inside each model: projection (cosine with
            the axis from the negative pole's centroid to the positive
            pole's), similarity (cosine with the centroid of the target
            words), or frequency (the word's share of the model's tokens)
        words: The words measured, one series each. A group written
            surgeon|physician|doctor is one series, the mean of its members'
            values inside each model (its error is then exact)
        words_file: A file of words to measure, one per line (added to words)
        positive: Pole words of the positive end of a projection's axis (at
            least 3)
        negative: Pole words of the negative end (at least 3)
        poles_file: A file of pole words: two lines (positive words, then
            negative), or one positive,negative pair per line. Added to the
            lists
        target: The word or words a similarity is measured against (their
            centroid, built in each model)
        years: First and last year to measure; blank: every year of the
            ensemble
        mode: resume keeps the years already measured and fills in the
            rest; restart deletes this measure's directory first
        workers: Models read at once (threads); each holds a whole model in
            memory. Blank: 8, or fewer where there are fewer CPUs
        out_dir: Where to write instead of <ensemble>/measures/<name>/
        array_task: This run's task in a job array (0-based), with
            array_count: it measures every array_count-th year from this one.
            Set by the batch script from Slurm's $SLURM_ARRAY_TASK_ID; blank
            outside an array
        array_count: How many tasks the job array has
        progress_path: Progress document for the UI
    """
    paths = derive_paths(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)
    corpus_dir = Path(paths["pivoted_db"]).parent
    ens_dir = _ensemble_dir(corpus_dir, ensemble)
    manifest = json.loads((ens_dir / "train.json").read_text())
    corpora, seeds = int(manifest.get("corpus_replicates") or 1), int(manifest.get("seeds_per_corpus") or 1)
    all_years = [int(y) for y in manifest.get("years_with_data") or []]
    if not all_years:
        all_years = sorted(int(p.name[1:]) for p in ens_dir.glob("y[0-9]*") if p.is_dir())
    if not name or not str(name).strip():
        raise ValueError("name is required: the measure's files go under it")
    name = str(name).strip()
    if mode not in ("resume", "restart"):
        raise ValueError("mode must be 'resume' or 'restart'")
    task = _array_task(array_task, array_count)
    if mode == "restart" and task is not None:
        raise ValueError("a measure cannot be restarted by the tasks of a job array (each would delete what the "
                         "others are writing): delete the directory first, then run the array in resume mode")

    spec = _spec(kind, words, words_file, positive, negative, poles_file, target)
    spec.validate()
    digest = spec_digest(spec)

    out = Path(os.path.expandvars(os.path.expanduser(out_dir))) if out_dir else ens_dir / "measures" / name
    if mode == "restart" and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    existing_manifest = _read_json(out / MANIFEST_FILE)
    if existing_manifest and existing_manifest.get("spec_digest") not in (None, digest):
        raise ValueError(f"'{name}' already holds a different measure ({existing_manifest.get('kind')} of "
                         f"{len(existing_manifest.get('words') or [])} words) in {out}: give this one another "
                         f"name, or set mode: restart to replace it")

    if years:
        first, last = int(years[0]), int(years[1])
        if last < first:
            raise ValueError("years must be [first, last] with first <= last")
        all_years = [y for y in all_years if first <= y <= last]
    mine = task_share(all_years, task)

    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"measure_{datetime.now():%Y%m%d_%H%M%S}{'_' + task.suffix if task else ''}.log"
    _setup_log(log_file)
    inputs: Dict[str, Any] = {"source": str(ens_dir)}
    if words_file:
        inputs["words_file"] = words_file
    if poles_file:
        inputs["poles_file"] = poles_file
    progress = ProgressReporter(progress_path, stage="measure", db_path=out, log_path=log_file,
                                unit_label="years", inputs=inputs)
    write_json(out / MANIFEST_FILE, {
        **spec.as_dict(), "name": name, "spec_digest": digest, "ensemble": str(ens_dir),
        "ensemble_name": ens_dir.name, "cell": manifest.get("cell_stem"), "corpus_replicates": corpora,
        "seeds_per_corpus": seeds, "years": [all_years[0], all_years[-1]] if all_years else None,
        "years_with_data": all_years, "created": existing_manifest.get("created") if existing_manifest
        else datetime.now().isoformat(timespec="seconds"), "layout": "y<year>.json, y<year>.npz",
        "description": KINDS[spec.kind][1]})
    if task is not None:
        progress.doc["array"] = {"task": task.index, "count": task.count, "years": mine}

    done_before = [y for y in mine if _record_current(year_file(out, y), digest)]
    progress.set_totals(len(mine), files_skipped=len(done_before))
    progress.doc["counters"]["years"] = len(mine)
    progress.doc["counters"]["words"] = len(spec.words)
    share = f" ({task.label}: {len(mine)} years)" if task else ""
    print(f"Measure '{name}': {spec.kind} of {len(spec.words)} words on ensemble '{ens_dir.name}' "
          f"({corpora} x {seeds} models per year); {len(all_years)} years{share}, {len(done_before)} already measured")
    logger.info("%s; spec %s; years %s", spec.as_dict(), digest, mine)
    if task is not None and not mine:
        progress.finish("done", f"{task.label}: no year falls to this task")
        print(f"{task.label}: no year falls to this task (there are {len(all_years)}); nothing to do")
        return str(out)

    n_threads = max(1, min(int(workers) if workers else DEFAULT_WORKERS, available_cpus()))
    pending: List[int] = []
    measured = 0
    try:
        for year in mine:
            if year in done_before:
                continue
            models = _design_models(ens_dir, year, corpora, seeds)
            if len(models) < corpora * seeds:
                pending.append(year)
                progress.doc["counters"]["years_pending"] = len(pending)
                logger.info("year %s: %d of %d models on disk; left pending", year, len(models), corpora * seeds)
                progress.file_done(f"y{year}", entries=0, chunks=0, uncompressed_bytes=0)
                continue
            progress.file_started(f"y{year}")
            progress.file_active(f"y{year}")
            record, cube_words, cube = _measure_year(models, year, spec, corpora, seeds, n_threads)
            record["spec_digest"] = digest
            np.savez_compressed(year_values_file(out, year).with_suffix(".part.npz"), values=cube.astype(np.float32),
                                words=np.array(cube_words, dtype=object),
                                models=np.array([f"c{c:02d}_s{s}" for c, s, _ in models], dtype=object))
            os.replace(year_values_file(out, year).with_suffix(".part.npz"), year_values_file(out, year))
            write_json(year_file(out, year), record)
            measured += 1
            progress.doc["counters"]["years_done"] = measured
            progress.file_done(f"y{year}", entries=len(record["words"]), chunks=0, uncompressed_bytes=0,
                               counters={"models": len(models)})
            _export(out, spec)
            logger.info("year %s: %d words, se %.4g, noise share %s", year, record["summary"].get("words"),
                        record["summary"].get("se_mean") or float("nan"), record["summary"].get("noise_share"))
    except Exception as exc:
        progress.finish("failed", str(exc))
        raise
    _export(out, spec)
    note = f"{measured} years measured" + (f", {len(done_before)} already done" if done_before else "") + \
        (f", {len(pending)} pending (models not all trained yet): {pending}" if pending else "")
    progress.finish("done", note)
    print(f"Measure '{name}': {note}. Files in {out}")
    return str(out)


# ---------------------------------------------------------------- pieces
def _ensemble_dir(corpus_dir: Path, ensemble: str) -> Path:
    if not ensemble or not str(ensemble).strip():
        raise ValueError("ensemble is required: the name of an ensemble under models/, or its directory")
    given = Path(os.path.expandvars(os.path.expanduser(str(ensemble).strip())))
    candidates = [given] if given.is_absolute() or "/" in str(ensemble) else []
    candidates.append(corpus_dir / "models" / str(ensemble).strip())
    for d in candidates:
        if (d / "train.json").is_file():
            return d
    have = sorted(p.name for p in (corpus_dir / "models").glob("*") if (p / "train.json").is_file()) \
        if (corpus_dir / "models").is_dir() else []
    raise FileNotFoundError(f"no ensemble '{ensemble}' (no train.json in {candidates[-1]})"
                            + (f"; the corpus has: {', '.join(have)}" if have else "; the corpus has none: run train first"))


def _words(given: Sequence[Any]) -> List[str]:
    if isinstance(given, str):
        given = given.replace(",", " ").split()
    return [str(w).strip().lower() for w in (given or ()) if str(w).strip()]


def _spec(kind: str, words, words_file, positive, negative, poles_file, target) -> Spec:
    kind = str(kind or "projection").strip().lower()
    ws = _words(words)
    if words_file:
        p = Path(os.path.expandvars(os.path.expanduser(str(words_file))))
        if not p.is_file():
            raise FileNotFoundError(f"words file not found: {p}")
        ws += read_word_list(p)
    pos, neg = _words(positive), _words(negative)
    if poles_file:
        p = Path(os.path.expandvars(os.path.expanduser(str(poles_file))))
        if not p.is_file():
            raise FileNotFoundError(f"poles file not found: {p}")
        fp, fn = read_pole_lists(p)
        pos, neg = pos + fp, neg + fn
    dedupe = lambda xs: tuple(dict.fromkeys(xs))
    return Spec(kind=kind, words=dedupe(ws), positive=dedupe(pos), negative=dedupe(neg), target=dedupe(_words(target)))


def _design_models(ens_dir: Path, year: int, corpora: int, seeds: int) -> List[Tuple[int, int, Path]]:
    """The year's models that belong to the design, as (corpus, seed, path)."""
    out = []
    for p in sorted((ens_dir / f"y{year}").glob("c*_s*.npz")):
        try:
            c, s = int(p.stem[1:3]), int(p.stem.split("_s")[1])
        except ValueError:
            continue
        if c < corpora and s < seeds:
            out.append((c, s, p))
    return out


def _measure_year(models: List[Tuple[int, int, Path]], year: int, spec: Spec, corpora: int, seeds: int,
                  threads: int) -> Tuple[Dict[str, Any], List[str], np.ndarray]:
    """Measure every model of the year (``threads`` at a time) and build the
    year's record (per-word statistics, coverage and the summary) and the
    (corpora, seeds, words) cube of raw values."""
    def one(item: Tuple[int, int, Path]) -> Tuple[Tuple[int, int], Measured]:
        c, s, p = item
        v = load_vectors(str(p))
        try:
            return (c, s), measure_one(v, spec)
        finally:
            del v

    results: Dict[Tuple[int, int], Measured] = {}
    if threads > 1 and len(models) > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for key, m in pool.map(one, models):
                results[key] = m
    else:
        for item in models:
            key, m = one(item)
            results[key] = m
    values = {key: m.values for key, m in results.items()}
    cube_words, cube = projection_cube(values, corpora, seeds, words=list(spec.words))
    comps = word_components(cube)
    rows = []
    for i, w in enumerate(cube_words):
        rows.append({"word": w, "mean": _num(comps["mean"][i]), "se": _num(comps["se"][i]),
                     "var_seed": _num(comps["var_seed"][i]), "var_corpus": _num(comps["var_corpus"][i]),
                     "n": int(comps["n"][i])})
    complete = [i for i in range(len(cube_words)) if comps["n"][i] == corpora * seeds]
    summary = variance_components(cube[:, :, complete]).as_dict("") if complete else {}
    summary = {k: (int(v) if k == "words" else _num(v)) for k, v in summary.items()}
    coverage: Dict[str, List[int]] = {}
    for m in results.values():
        for k, n in m.coverage.items():
            lo, hi = coverage.get(k, [n, n])
            coverage[k] = [min(lo, n), max(hi, n)]
    notes = [{"model": f"c{c:02d}_s{s}", "note": m.note} for (c, s), m in sorted(results.items()) if m.note]
    record = {"year": year, "kind": spec.kind, "corpus_replicates": corpora, "seeds_per_corpus": seeds,
              "models_used": len(models), "words": rows,
              "words_all": len(complete), "words_some": int(((comps["n"] > 0) & (comps["n"] < corpora * seeds)).sum()),
              "words_none": int((comps["n"] == 0).sum()), "coverage": coverage, "notes": notes, "summary": summary}
    return record, cube_words, cube


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _record_current(path: Path, digest: str) -> bool:
    rec = _read_json(path)
    return bool(rec) and rec.get("spec_digest") == digest and "words" in rec


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _export(out: Path, spec: Spec) -> None:
    """``series.tsv`` (long) and ``summary.tsv`` (a row per year), rebuilt
    from the year files so they are current whichever task wrote last."""
    records = []
    for p in sorted(out.glob("y[0-9]*.json")):
        rec = _read_json(p)
        if rec.get("words") is not None:
            records.append(rec)
    records.sort(key=lambda r: int(r.get("year", 0)))
    order = {w: i for i, w in enumerate(spec.words)}
    series = ["\t".join(SERIES_COLUMNS)]
    for rec in records:
        for row in sorted(rec["words"], key=lambda r: order.get(r["word"], len(order))):
            series.append("\t".join([str(rec["year"]), row["word"], _fmt(row.get("mean")), _fmt(row.get("se")),
                                     str(row.get("n", 0)), _fmt(row.get("var_seed")), _fmt(row.get("var_corpus"))]))
    _write_text(out / SERIES_FILE, "\n".join(series) + "\n")
    summary = ["\t".join(SUMMARY_COLUMNS)]
    for rec in records:
        s = rec.get("summary") or {}
        cov = "; ".join(f"{k} {lo}-{hi}" if lo != hi else f"{k} {lo}" for k, (lo, hi) in (rec.get("coverage") or {}).items())
        notes = rec.get("notes") or []
        summary.append("\t".join([str(rec["year"]), str(rec.get("models_used", "")), str(rec.get("words_all", "")),
                                  str(rec.get("words_some", "")), str(rec.get("words_none", "")),
                                  _fmt(s.get("se_mean")), _fmt(s.get("noise_share")), cov,
                                  f"{len(notes)} models could not measure" if notes else ""]))
    _write_text(out / SUMMARY_FILE, "\n".join(summary) + "\n")


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return f"{float(v):.6g}"


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _setup_log(path: Path) -> None:
    root = logging.getLogger("lexichron.stages.measure")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(h)

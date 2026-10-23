"""Training many models across years on one node.

The models to train are a list of tasks in year order. ``parallel`` of
them train at a time in worker processes, each with ``threads`` CPUs; a
slot that frees up takes the next task whose corpus is ready, so the last
models of one year and the first of the next train side by side instead of
the tail of each year idling the node. Corpus files are written in the
background, ahead of training, and dropped when their last model lands.

A task's corpus is named by its ``corpus_key``: the year for the tune
stage (one write produces every file the year's models read), the year
and the replicate for the train stage (a file per replicate). The caller
supplies the writing (``submit_corpus(key)`` starts it, ``pending[key]`` is
its futures) and the file to train from (``corpus_file(task)``); the
scheduler keeps the queue's next ``corpus_ahead`` keys written or being
written, and calls
``on_trained(task)`` as each model lands (it is on disk), ``on_corpus_done
(key)`` when a key's last model lands, and ``on_year_done(year)`` when a
year's last model lands, all in the calling process.
"""
from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Protocol, Sequence

from lexichron.training.grid import Cell
from lexichron.training.trainer import TrainFn, save_vectors, train_model
from ngramprep.ngram_acquire.progress import ProgressReporter

__all__ = ["Task", "train_across_years", "train_one"]

logger = logging.getLogger(__name__)


class Task(Protocol):
    """What the scheduler needs of a model to train."""
    cell: Cell
    year: int

    @property
    def stem(self) -> str: ...        # the model's file name, without .npz

    @property
    def seed(self) -> int: ...

    @property
    def corpus_key(self) -> Hashable: ...


def train_one(args):
    """Runs in a worker process (or inline): train and save one model."""
    corpus_file, cell, seed, threads, out_path, train_fn = args
    res = train_model(corpus_file, cell, seed, threads, train_fn)
    save_vectors(res.vectors, out_path)
    return out_path, res.seconds, res.vocab_size


def train_across_years(todo: Sequence[Task], *, models_dir: Path, parallel: int,
                       threads_for: Callable[[int], int], corpus_file: Callable[[Any], Path],
                       submit_corpus: Callable[[Hashable], None], pending: Dict[Hashable, List[Future]],
                       on_corpus_written: Callable[[Hashable, List[Any]], None],
                       on_trained: Callable[[Any, float, int], None],
                       on_corpus_done: Callable[[Hashable], None], on_year_done: Callable[[int], None],
                       progress: ProgressReporter, train_fn: Optional[TrainFn],
                       pool_factory: Callable[[int], Any], inline: bool, corpus_ahead: int = 2) -> None:
    """Train ``todo`` as described in the module docstring. ``threads_for(n)``
    gives the threads per model when ``n`` models remain, so the final few
    get the spare CPUs. ``on_corpus_written(key, results)`` receives the
    results of the key's write futures once they are all done. With
    ``inline`` (tests, or one slot) models train one at a time in this
    process, in the order the pool would start them."""
    logger.info("training %d models, %d at a time x %d threads", len(todo), min(parallel, len(todo)),
                threads_for(len(todo)))
    queue = list(todo)
    in_flight: Dict[Future, Any] = {}
    ready: set = set()
    left_in_key: Dict[Hashable, int] = {}
    left_in_year: Dict[int, int] = {}
    for t in todo:
        left_in_key[t.corpus_key] = left_in_key.get(t.corpus_key, 0) + 1
        left_in_year[t.year] = left_in_year.get(t.year, 0) + 1

    def keys_ahead() -> List[Hashable]:
        """The next ``corpus_ahead`` distinct keys the queue will need (the
        in-flight models' corpora are ready already, so they do not use up
        the window)."""
        seen: List[Hashable] = []
        for t in queue:
            if t.corpus_key not in seen:
                seen.append(t.corpus_key)
                if len(seen) == corpus_ahead:
                    break
        return seen

    def keep_corpora_coming() -> None:
        for key in keys_ahead():
            if key not in ready and key not in pending:
                submit_corpus(key)

    def corpus_ready(key: Hashable) -> bool:
        if key in ready:
            return True
        futs = pending.get(key)
        if futs is None or not all(f.done() for f in futs):
            return False
        on_corpus_written(key, [f.result() for f in pending.pop(key)])
        ready.add(key)
        return True

    def finished(t: Any, secs: float, n: int) -> None:
        logger.info("trained %s: %d words, %.0fs", t.stem, n, secs)
        progress.doc["counters"]["train_seconds"] = progress.doc["counters"].get("train_seconds", 0) + int(secs)
        progress.file_parsed(t.stem)
        on_trained(t, secs, n)
        left_in_key[t.corpus_key] -= 1
        if left_in_key[t.corpus_key] == 0:
            on_corpus_done(t.corpus_key)
        left_in_year[t.year] -= 1
        if left_in_year[t.year] == 0:
            on_year_done(t.year)

    def job_for(t: Any, threads: int):
        return (str(corpus_file(t)), t.cell, t.seed, threads, str(models_dir / f"{t.stem}.npz"), train_fn)

    pool = None if inline else pool_factory(parallel)
    try:
        while queue or in_flight:
            keep_corpora_coming()
            # fill the free slots from the front of the queue; threads per model
            # from what remains at the start of this pass, so models that start
            # together get the same share
            threads = threads_for(len(queue) + len(in_flight))
            while queue and len(in_flight) < parallel and corpus_ready(queue[0].corpus_key):
                t = queue.pop(0)
                progress.file_active(t.stem)
                if inline:
                    _, secs, n = train_one(job_for(t, threads))
                    finished(t, secs, n)
                    keep_corpora_coming()
                else:
                    in_flight[pool.submit(train_one, job_for(t, threads))] = t
            if not queue and not in_flight:
                break
            progress._write(force=True)
            # wait for a model to finish, or for the corpus the next model needs
            waits: List[Future] = list(in_flight)
            if queue and not corpus_ready(queue[0].corpus_key):
                waits += pending.get(queue[0].corpus_key, [])
                if not in_flight:
                    progress.set_phase("corpus")
                    logger.info("waiting for the corpus of %s", queue[0].corpus_key)
            if not waits:
                continue
            done, _ = wait(waits, return_when=FIRST_COMPLETED)
            progress.set_phase("training")
            for fut in done:
                if fut in in_flight:
                    t = in_flight.pop(fut)
                    _, secs, n = fut.result()
                    finished(t, secs, n)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

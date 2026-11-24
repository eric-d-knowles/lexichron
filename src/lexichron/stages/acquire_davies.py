"""``lexichron acquire`` for a Davies corpus (``corpus.source: davies``).

The corpus's own files, which the user holds under licence, are read into
a raw database of text units by year: the same shape as a raw n-gram
database, with a sentence where an n-gram would be, so the filter, tune,
train and measure stages work on it as they do on n-grams. The machinery
is the n-gram acquisition's: one sorted SST file per archive ingested with
its marker, a mandatory compaction, and a count check of the finished
database against the markers' totals.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = ["acquire_davies"]

logger = logging.getLogger(__name__)


def acquire_davies(
    repo_corpus_id: str,
    db_path_stub: str,
    source_dir: str,
    unit: str = "sentences",
    ngram_type: str = "tagged",
    genres: Optional[List[str]] = None,
    file_range: Optional[Tuple[int, int]] = None,
    workers: Optional[int] = None,
    overwrite_db: bool = False,
    max_unit_tokens: int = 500,
    chunk_entries: int = 200_000,
    spool_dir: Optional[str] = None,
    verify_counts: bool = True,
    allow_unparsed_lines: bool = False,
    progress_path: Optional[str] = None,
    source: str = "davies",
) -> None:
    """Read a Davies corpus's files into its raw database.

    Args:
        repo_corpus_id: Which corpus (``coha``)
        db_path_stub: Directory under which the corpus's databases live, as
            ``<db_path_stub>/<name>/``. Environment variables and ~ are expanded.
        source_dir: The corpus's own directory, as distributed: the one
            holding ``wlp/`` and ``text/``. It is only read.
        unit: What a record is. ``sentences`` (default): a unit of running
            text, which the models are trained on. ``tokens``: a single
            token, the counterpart of Google's 1-grams, from which the
            filter builds the vocabulary. A corpus is acquired once as
            each, into ``<name>/sentence_files/`` and ``<name>/token_files/``;
            use the same ngram_type and genres for both.
        ngram_type: ``tagged`` reads the word/lemma/part-of-speech edition
            (``wlp/``) and stores each token with a coarse tag (``building_NOUN``),
            which the filter uses for lemmas as it does with tagged n-grams;
            ``untagged`` reads the plain-text edition (``text/``) and stores
            bare words. The two editions hold the same text.
        genres: Keep only documents of these genres (COHA: fic, mag, news,
            nf). Default: all. The choice is fixed in the database: another
            selection is another corpus directory.
        file_range: (first, last) of the edition's archives in name order,
            to read a part. Default: all.
        workers: Processes (default: the allocation's CPUs less one, at most one per archive)
        overwrite_db: Remove an existing raw database first. Default False:
            it is resumed, and archives already in it are skipped.
        max_unit_tokens: The longest unit of text stored. Text is cut at
            sentence ends, paragraph marks and the corpus's copyright gaps;
            a stretch with none of these for this many tokens is cut here.
        chunk_entries: Records a worker holds in memory before writing a sorted part
        spool_dir: Where workers write their files (default: the system temp
            directory; node-local disk is ideal)
        verify_counts: Read the finished database once and require its
            totals to equal what the archives contributed (default True)
        allow_unparsed_lines: Accept rows of the tagged edition that are not
            word, lemma, tag (they are left out and counted). By default a
            run that met any finishes its work and then fails, saying how many.
        progress_path: Progress document for this run (set by the CLI)
        source: The kind of corpus; always ``davies`` here (set from ``corpus.source``)
    """
    from lexichron.corpora import corpus_layout
    from lexichron.davies.formats import EDITIONS, corpus_format
    from lexichron.davies.reader import archives
    from lexichron.davies.worker import process_archive
    from ngramprep.ngram_acquire.coordinator import select_file_subset
    from ngramprep.ngram_acquire.core import run_acquisition
    from ngramprep.ngram_acquire.logger import setup_logger
    from ngramprep.ngram_acquire.progress import ProgressReporter
    from ngramprep.utilities.display import format_banner
    from ngramprep.utilities.paths import expand_path

    if source != "davies":
        raise ValueError(f"acquire_davies reads Davies corpora; source is {source!r}")
    fmt = corpus_format(repo_corpus_id)
    if ngram_type not in EDITIONS:
        raise ValueError(f"ngram_type must be one of {tuple(EDITIONS)}, got {ngram_type!r}")
    if genres:
        genres = [str(g).lower() for g in genres]
        unknown = sorted(set(genres) - set(fmt.genres))
        if unknown:
            raise ValueError(f"{fmt.label} has no genre {unknown[0]!r}; its genres are {', '.join(fmt.genres)}")
    if int(max_unit_tokens) < 2:
        raise ValueError("max_unit_tokens must be at least 2")
    start_time = datetime.now()
    source_dir = expand_path(source_dir)
    found = archives(source_dir, fmt, ngram_type)
    selected, first, last = select_file_subset([str(p) for p in found], tuple(file_range) if file_range else None)

    layout = corpus_layout(db_path_stub, None, None, fmt.name, source="davies", unit=unit)
    db_path = str(layout.raw_db)
    log_file = None
    if not logging.getLogger().hasHandlers():
        log_file = setup_logger(layout.dir / "logs", filename_prefix="davies_acquire",
                                console=True, console_level=logging.WARNING)
        print(f"Log file: {log_file}")
    what = f"{fmt.label} {unit}, {ngram_type} edition" + (f", genres {', '.join(genres)}" if genres else "")
    logger.info("Acquisition starting: %s from %s into %s", what, source_dir, db_path)
    progress = ProgressReporter(progress_path, stage="acquire", db_path=db_path, log_path=log_file,
                                inputs={"source": f"{what} ({Path(source_dir) / EDITIONS[ngram_type]})"})
    progress.set_plan(["ingesting", "compacting"] + (["verifying"] if verify_counts else []))
    progress.set_checks(verify_counts=bool(verify_counts), allow_unparsed_lines=bool(allow_unparsed_lines))

    def header(to_get: int, to_skip: int, n_workers: int) -> None:
        print(format_banner(f"{fmt.label} ACQUISITION", style="━"))
        print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
        print()
        print(format_banner("Configuration"))
        print(f"Source:               {Path(source_dir) / EDITIONS[ngram_type]}")
        print(f"Records:              {unit}")
        print(f"Edition:              {ngram_type}")
        print(f"Genres:               {', '.join(genres) if genres else 'all'}")
        print(f"DB path:              {db_path}")
        print(f"Archives:             {first}-{last} of {len(found)} ({to_get} to read, {to_skip} already in)")
        print(f"Workers:              {n_workers}")
        print(f"Overwrite:            {overwrite_db}")
        print()

    run_acquisition(
        db_path, selected,
        sizes={Path(p).name: os.path.getsize(p) for p in selected},
        workers=workers, overwrite_db=overwrite_db, progress=progress, start_time=start_time,
        verify_counts=verify_counts, allow_unparsed_lines=allow_unparsed_lines, print_header=header,
        process_kwargs=dict(worker_fn=process_archive, spool_dir=spool_dir, chunk_entries=int(chunk_entries),
                            worker_kwargs=dict(corpus=fmt.name, unit=unit, ngram_type=ngram_type, genres=genres,
                                               max_unit_tokens=int(max_unit_tokens))),
    )
    return None

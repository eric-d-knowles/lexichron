"""Filter stage: raw n-gram database -> processed database (+ whitelist).

Runs on the range-map engine (:mod:`ngramprep.rangemap`): the source key
space is partitioned, workers map their ranges to chunk files, and one
parent process merges the chunks into the destination with a resume marker
per unit. Compared with the previous orchestrator this removes the per-worker
RocksDB instances (the main memory cost), the SQLite work tracker (the main
source of contention) and the re-read of every shard at ingest.

``build_processed_db`` keeps its previous signature. ``PipelineConfig``
fields that belonged to the old design (flush intervals, claim order,
ingest readers, ...) are accepted and ignored.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Set, Union

from ..config import FilterConfig, PipelineConfig
from ngramprep.ngram_acquire.progress import ProgressReporter
from ngramprep.rangemap import RangeMapConfig, run_rangemap
from ngramprep.utilities.display import format_banner, truncate_path_to_fit
from .mapper import FilterMapper, PivotFilterMapper
from .setup import PipelineSetup
from .whitelist import write_whitelist

logger = logging.getLogger(__name__)

__all__ = ["build_processed_db", "run_filter_pipeline", "derive_paths"]


def _to_bytes_set(s: Optional[Set[Any]]) -> Optional[Set[bytes]]:
    if s is None:
        return None
    out = set()
    for item in s:
        if isinstance(item, bytes):
            out.add(item)
        elif isinstance(item, str):
            out.add(item.encode("utf-8"))
        else:
            raise TypeError(f"Expected str or bytes, got {type(item)}")
    return out


def derive_paths(db_path_stub: str, ngram_size: int, repo_release_id: str, repo_corpus_id: str) -> dict:
    """Source, destination, temp and default whitelist paths for a corpus."""
    from ngramprep.ngram_acquire.db.build_path import build_db_path
    src = Path(build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id))
    base = src.parent
    dst = base / f"{ngram_size}grams_processed.db"
    return {"src_db": src, "dst_db": dst, "tmp_dir": base / "processing_tmp",
            "whitelist": dst / "whitelist.txt",
            "pivoted_db": base / f"{ngram_size}grams_pivoted.db"}


def run_filter_pipeline(filter_config: FilterConfig, pipeline_config: PipelineConfig,
                        *, progress_path: Optional[str] = None,
                        spool_dir: Optional[str] = None, chunk_entries: int = 200_000) -> str:
    """Run the stage from fully resolved configs (paths filled in)."""
    pc = pipeline_config
    if pc.src_db is None or pc.dst_db is None:
        raise ValueError("pipeline_config needs src_db and dst_db (see derive_paths)")
    t0 = time.perf_counter()

    pivot = pc.output == "pivoted"
    if pivot and pc.output_whitelist_path:
        raise ValueError("a vocabulary (output whitelist) is built from a processed database, not a pivoted one")
    print(format_banner("N-gram filtering" + (" and pivoting" if pivot else "")))
    print(f"Source:       {pc.src_db}")
    print(f"Destination:  {pc.dst_db}" + ("  (keyed year-first)" if pivot else ""))
    print(f"Mode:         {pc.mode}")
    print(f"Workers:      {pc.num_workers}")

    # Whitelist loading, vocabulary index, directories.
    setup = PipelineSetup(pc, filter_config)
    paths, filter_config = setup.prepare_all()

    mode = "restart" if pc.mode in ("restart", "reprocess") else "resume"
    # Log to a file next to the database unless the caller configured
    # logging (a notebook); the unit-by-unit lines go there, warnings to the
    # console, and the UI's Progress tab shows the file's tail.
    log_file = None
    if not logging.getLogger().hasHandlers():
        from ngramprep.ngram_acquire.logger import setup_logger
        log_file = setup_logger(Path(pc.dst_db).parent / "logs", filename_prefix="ngram_filter",
                                console=True, console_level=logging.WARNING)
        print(f"Log file:     {log_file}")
    logger.info("Source %s -> destination %s (mode %s, %s workers)", pc.src_db, pc.dst_db, pc.mode, pc.num_workers)

    vocab = getattr(filter_config, "whitelist_path", None)
    progress = ProgressReporter(progress_path, stage="filter", db_path=pc.dst_db, log_path=log_file,
                                unit_label="work units",
                                inputs={"source": pc.src_db, "vocabulary": vocab or None,
                                        "always_include": ", ".join(sorted(
                                            x.decode() if isinstance(x, bytes) else str(x)
                                            for x in (filter_config.always_include or ()))) or None})
    num_units = pc.num_initial_work_units or max(4 * pc.num_workers, 64)

    # The pivoted output is many times the size of its input (one record per
    # year), so its chunks go to the database as SST files rather than
    # through the parent's write path.
    cfg = RangeMapConfig(
        src_db=Path(pc.src_db), dst_db=Path(pc.dst_db),
        mapper=PivotFilterMapper(filter_config) if pivot else FilterMapper(filter_config),
        output="sst" if pivot else "chunks",
        num_units=num_units, workers=pc.num_workers, mode=mode,
        smart_partitioning=pc.use_smart_partitioning, samples_per_worker=pc.samples_per_worker,
        use_cached_partition=pc.use_cached_partitions and pc.cache_partitions,
        partition_cache_dir=paths["cache_dir"],
        read_profile=pc.reader_profile, write_profile=pc.ingest_write_profile,
        chunk_entries=chunk_entries, spool_dir=Path(spool_dir) if spool_dir else None,
        compact_after=pc.compact_after_ingest, progress=progress,
    )
    result = run_rangemap(cfg)
    print(f"Filtered {result.scanned:,} records -> {result.kept:,} {'year records' if pivot else 'kept'} in "
          f"{result.units_done} work units ({result.units_skipped} already done) "
          f"in {result.seconds / 60:.1f} min")

    if pc.output_whitelist_path:
        wl = Path(pc.output_whitelist_path)
        print(format_banner("Output whitelist"))
        print("  Path: " + truncate_path_to_fit(wl, "  Path: "))
        with progress.long_phase("whitelist"):
            write_whitelist(db_or_path=pc.dst_db, dest=wl, top=pc.output_whitelist_top_n,
                            decode=True, sep="\t", spell_check=pc.output_whitelist_spell_check,
                            spell_check_language=pc.output_whitelist_spell_check_language,
                            year_range=pc.output_whitelist_year_range,
                            bin_size=filter_config.bin_size)
        try:
            with open(wl, "r", encoding="utf-8") as fh:
                n = sum(1 for _ in fh)
            print(f"  {n:,} tokens")
        except OSError:
            pass

    progress.finish("done")
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min: {pc.dst_db}")
    return str(pc.dst_db)


def build_processed_db(
    filter_config: Optional[FilterConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    # Path construction parameters (used if pipeline_config has no paths)
    ngram_size: Optional[int] = None,
    repo_release_id: Optional[str] = None,
    repo_corpus_id: Optional[str] = None,
    db_path_stub: Optional[str] = None,
    # Filter configuration parameters (used if filter_config not provided)
    stop_set: Optional[Set[str]] = None,
    lemma_gen: Any = None,
    min_len: int = 2,
    whitelist_path: Optional[Union[str, Path]] = None,
    whitelist_min_count: int = 1,
    whitelist_top_n: Optional[int] = None,
    always_include: Optional[Set[str]] = None,
    bin_size: int = 1,
    min_context_tokens: int = 2,
    # Pipeline execution parameters
    num_workers: Optional[int] = None,
    mode: Optional[str] = None,
    use_smart_partitioning: Optional[bool] = None,
    num_initial_work_units: Optional[int] = None,
    cache_partitions: Optional[bool] = None,
    use_cached_partitions: Optional[bool] = None,
    samples_per_worker: Optional[int] = None,
    work_unit_claim_order: Optional[str] = None,          # ignored (legacy)
    flush_interval_s: Optional[float] = None,             # ignored (legacy)
    progress_every_s: Optional[float] = None,             # ignored (legacy)
    ingest_num_readers: Optional[int] = None,             # ignored (legacy)
    ingest_batch_items: Optional[int] = None,             # ignored (legacy)
    ingest_queue_size: Optional[int] = None,              # ignored (legacy)
    compact_after_ingest: Optional[bool] = None,
    # Output whitelist generation
    output_whitelist_path: Optional[Union[str, Path]] = "default",
    output_whitelist_top_n: Optional[int] = None,
    output_whitelist_year_range: Optional[tuple[int, int]] = None,
    output_whitelist_spell_check: bool = False,
    output_whitelist_spell_check_language: str = "en_US",
    # Engine
    progress_path: Optional[str] = None,
    spool_dir: Optional[str] = None,
    chunk_entries: int = 200_000,
) -> str:
    """
    Filter a raw n-gram database into a processed one.

    Either pass ``filter_config``/``pipeline_config`` objects (the notebooks'
    way) or the plain keyword arguments. Paths are derived from
    ``db_path_stub``/``ngram_size``/``repo_release_id``/``repo_corpus_id``
    unless ``pipeline_config`` already carries ``src_db``/``dst_db``.

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for databases
        stop_set: Set of stopwords to filter
        lemma_gen: Lemmatizer instance (e.g. CachedSpacyLemmatizer(language="en"))
        min_len: Minimum token length
        whitelist_path: Input whitelist (tokens not in it become <UNK>)
        whitelist_min_count: Minimum count for whitelist tokens
        whitelist_top_n: Limit the whitelist to its top N tokens
        always_include: Tokens kept regardless of the whitelist
        bin_size: Aggregate years into bins (1 = annual)
        min_context_tokens: Reject n-grams with fewer non-<UNK> tokens than this
        num_workers: Worker processes (default: CPUs - 1)
        mode: "resume" (default: continue from the destination's markers) or
            "restart" (wipe the destination first); "reprocess" is treated as restart
        use_smart_partitioning: Balance work units by sampling the source
        num_initial_work_units: Number of work units (default: 4 x workers, at least 64)
        cache_partitions: Cache the sampled partition beside the database
        use_cached_partitions: Reuse a cached partition when available
        samples_per_worker: Reservoir size per sampling worker
        compact_after_ingest: Compact the destination at the end
        output_whitelist_path: Whitelist to write from the result ("default" =
            <destination>/whitelist.txt, None = none)
        output_whitelist_top_n: Keep the top N tokens
        output_whitelist_year_range: Only tokens present in every year of (start, end)
        output_whitelist_spell_check: Only correctly spelled tokens
        output_whitelist_spell_check_language: Spell-check dictionary, e.g. en_US
        progress_path: Where to keep a progress.json for this run
        spool_dir: Where workers write chunk files (default: $TMPDIR)
        chunk_entries: Entries per chunk file (bounds worker memory)

    Returns:
        Path to the processed database directory
    """
    if filter_config is None:
        filter_config = FilterConfig(
            stop_set=stop_set, lemma_gen=lemma_gen, min_len=min_len,
            whitelist_path=whitelist_path, whitelist_min_count=whitelist_min_count,
            whitelist_top_n=whitelist_top_n,
            always_include=_to_bytes_set(always_include) if always_include else None,
            bin_size=bin_size, min_context_tokens=min_context_tokens,
        )

    pc = pipeline_config
    if pc is not None and pc.src_db is not None and pc.dst_db is not None:
        if pc.tmp_dir is None:
            pc = replace(pc, tmp_dir=Path(pc.dst_db).parent / "processing_tmp")
        if pc.output_whitelist_path == "default":
            pc = replace(pc, output_whitelist_path=None if pc.output == "pivoted" else Path(pc.dst_db) / "whitelist.txt")
        return run_filter_pipeline(filter_config, pc, progress_path=progress_path,
                                   spool_dir=spool_dir, chunk_entries=chunk_entries)

    # Paths from the stub, either from pipeline_config or the keyword arguments.
    if pc is not None and pc.db_path_stub is not None:
        ngram_size, repo_release_id = pc.ngram_size, pc.repo_release_id
        repo_corpus_id, db_path_stub = pc.repo_corpus_id, pc.db_path_stub
    if ngram_size is None or repo_release_id is None or repo_corpus_id is None or db_path_stub is None:
        raise ValueError("provide a PipelineConfig with paths, or ngram_size, repo_release_id, "
                         "repo_corpus_id and db_path_stub")
    paths = derive_paths(db_path_stub, int(ngram_size), str(repo_release_id), str(repo_corpus_id))

    if pc is None:
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        kwargs = {"num_workers": num_workers}
        for key, value in {
            "num_initial_work_units": num_initial_work_units, "mode": mode,
            "use_smart_partitioning": use_smart_partitioning, "cache_partitions": cache_partitions,
            "use_cached_partitions": use_cached_partitions, "samples_per_worker": samples_per_worker,
            "compact_after_ingest": compact_after_ingest,
            "output_whitelist_top_n": output_whitelist_top_n,
            "output_whitelist_year_range": output_whitelist_year_range,
            "output_whitelist_spell_check": output_whitelist_spell_check,
            "output_whitelist_spell_check_language": output_whitelist_spell_check_language,
        }.items():
            if value is not None:
                kwargs[key] = value
        pc = PipelineConfig(output_whitelist_path=output_whitelist_path, **kwargs)

    wl = pc.output_whitelist_path
    if pc.output == "pivoted":
        dst = paths["pivoted_db"]
        wl = None if wl == "default" else wl
    else:
        dst = paths["dst_db"]
        wl = paths["whitelist"] if wl == "default" else wl
    pc = replace(pc, src_db=paths["src_db"], dst_db=dst, tmp_dir=paths["tmp_dir"], output_whitelist_path=wl)
    return run_filter_pipeline(filter_config, pc, progress_path=progress_path,
                               spool_dir=spool_dir, chunk_entries=chunk_entries)

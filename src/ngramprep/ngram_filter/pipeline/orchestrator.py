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
from ngramprep.ngram_acquire.verify import verify_database
from ngramprep.rangemap import RangeMapConfig, UNIT_MARKER_PREFIX, run_rangemap
from ngramprep.utilities.display import format_banner, truncate_path_to_fit
from .mapper import FilterMapper, PivotFilterMapper
from .setup import PipelineSetup
from .whitelist import write_whitelist

logger = logging.getLogger(__name__)

# How a run made its lemmas, as recorded in its progress document and provenance.
LEMMAS_BY_TAG = "by part-of-speech tag, else lookup table"
LEMMAS_LOOKUP = "lookup table"

__all__ = ["build_processed_db", "run_filter_pipeline", "derive_paths", "FilterError",
           "describe_filter_verification"]


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


class FilterError(RuntimeError):
    """Raised when a filter run cannot vouch for its output: the source is
    not a finished database, or the counts do not add up."""


def _file_digest(path) -> Optional[str]:
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _run_inputs(filter_config, vocab, lemmas, pivot: bool) -> dict:
    """What decides a filter run's output, for the range-map engine to record
    in the destination and compare on a resume: the vocabulary by its
    content (a rebuilt vocabulary at the same path is a different input),
    how many of its words are used, how lemmas are made, the keep lists and
    the rules. Not the workers or other settings that change only how the
    work is done."""
    def words(xs):
        return sorted(x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in (xs or ()))
    fc = filter_config
    return {
        "vocabulary": _file_digest(vocab) if vocab else None,
        "vocabulary_top_n": getattr(fc, "whitelist_top_n", None),
        "vocabulary_min_count": getattr(fc, "whitelist_min_count", None) if vocab else None,
        "lemmas": lemmas,
        "always_include": words(fc.always_include),
        "always_include_case": words(fc.always_include_case),
        "lemma_exempt": words(getattr(fc, "lemma_exempt", None)),
        "stopwords": words(fc.stop_set) if fc.filter_stops else None,
        "rules": {"lowercase": fc.lowercase, "alpha_only": fc.alpha_only, "ascii_alpha_only": fc.ascii_alpha_only,
                  "filter_short": fc.filter_short, "min_len": fc.min_len,
                  "min_context_tokens": fc.min_context_tokens, "bin_size": fc.bin_size},
        "pivot": bool(pivot),
    }


def _require_compacted_source(src_db, read_profile: str) -> None:
    """Refuse a source database that was never compacted.

    An acquisition that did not finish (a file failed, or the job ended
    before compaction) leaves one SST file per downloaded file, every one
    overlapping the others: counts of an n-gram that arrived from several
    files are not summed yet, and a range of keys touches nearly every file.
    The stage's partition and its workers assume a compacted source."""
    from ngramprep.common_db.api import open_db
    try:
        with open_db(str(src_db), mode="r", profile=read_profile) as db:
            level0 = db.get_property("rocksdb.num-files-at-level0")
    except Exception:
        return                                   # not openable: the run reports that itself
    try:
        level0 = int(level0 or 0)
    except (TypeError, ValueError):
        return
    if level0 > 0:
        raise FilterError(
            f"The source database {src_db} has not been compacted ({level0:,} files still waiting to be "
            f"folded in): the run that made it did not finish. Run that stage again; it resumes, and "
            f"compacts when every file is in.")


def _check_source(ledger: dict, source_totals: Optional[dict]) -> dict:
    """What the work units read, against what the source is known to hold.

    Every data record of the source lies in exactly one unit's key range, so
    the units' totals of everything they read (kept or not) must equal the
    source's own. A difference means the partition skipped keys or gave some
    to two units. ``unknown`` when the source's totals were never verified
    (a database acquired before the count ledger), or some units predate
    their own."""
    read = {"match_total": int(ledger.get("source_match", 0)), "volume_total": int(ledger.get("source_volume", 0))}
    out = {"status": "unknown", "read": read, "expected": None,
           "disordered": int(ledger.get("source_disordered", 0))}
    if out["disordered"]:
        out["status"] = "mismatch"               # the source itself holds values the merge operator cannot sum
    if not source_totals or ledger.get("unledgered"):
        return out
    out["expected"] = {"match_total": int(source_totals["match_total"]),
                       "volume_total": int(source_totals["volume_total"])}
    if out["status"] != "mismatch":
        out["status"] = "verified" if read == out["expected"] else "mismatch"
    return out


def _check_balance(ledger: dict) -> dict:
    """What the work units read, against what they emitted plus what they dropped.

    Filtering only moves counts: a kept n-gram goes out with every count it
    came in with (lowercased, lemmatized, binned, split into one record per
    year, but not a count more or less), and a dropped one goes nowhere. The
    three totals are taken separately, the dropped one from the dropped
    records themselves, so read = emitted + dropped is a test and not a
    definition. It fails if the mapper loses or invents a count (in binning,
    in the pivot), or a record is neither emitted nor dropped. ``unknown``
    when some units predate the dropped totals."""
    read = (int(ledger.get("source_match", 0)), int(ledger.get("source_volume", 0)))
    emitted = (int(ledger.get("match_total", 0)), int(ledger.get("volume_total", 0)))
    dropped = (int(ledger.get("dropped_match", 0)), int(ledger.get("dropped_volume", 0)))
    out = {"read": {"match_total": read[0], "volume_total": read[1]},
           "emitted": {"match_total": emitted[0], "volume_total": emitted[1]},
           "dropped": {"match_total": dropped[0], "volume_total": dropped[1]},
           "difference": {"match_total": emitted[0] + dropped[0] - read[0],
                          "volume_total": emitted[1] + dropped[1] - read[1]}}
    if ledger.get("unledgered") or ledger.get("undropped"):
        out["status"] = "unknown"
    else:
        out["status"] = "verified" if out["difference"] == {"match_total": 0, "volume_total": 0} else "mismatch"
    return out


def describe_filter_verification(v: dict) -> list:
    """The verification record of a filter run in sentences."""
    ledger, found, source = v.get("ledger") or {}, v.get("database") or {}, v.get("source") or {}
    lines = []
    units = int(ledger.get("shards", 0))
    if ledger.get("unledgered"):
        lines.append(f"Counts not verifiable: {int(ledger['unledgered']):,} of {units:,} work units were filtered by "
                     f"a version that kept no totals. Every value in the database is well formed."
                     if not (found.get("disordered") or found.get("malformed")) else
                     f"{int(found.get('disordered', 0)) + int(found.get('malformed', 0)):,} values in the database "
                     f"are malformed or have years out of order.")
    elif (found.get("match_total"), found.get("volume_total")) == (ledger.get("match_total"), ledger.get("volume_total")) \
            and not (found.get("disordered") or found.get("malformed")):
        lines.append(f"Counts verified: the database's {int(found.get('records', 0)):,} records hold "
                     f"{int(found.get('match_total', 0)):,} occurrences in {int(found.get('volume_total', 0)):,} "
                     f"volume counts, exactly what the {units:,} work units emitted.")
    else:
        lines.append("COUNTS DO NOT MATCH what the work units emitted:")
        lines.append(f"  occurrences:   database {int(found.get('match_total', 0)):,}   units {int(ledger.get('match_total', 0)):,}"
                     f"   difference {int(found.get('match_total', 0)) - int(ledger.get('match_total', 0)):+,}")
        lines.append(f"  volume counts: database {int(found.get('volume_total', 0)):,}   units {int(ledger.get('volume_total', 0)):,}"
                     f"   difference {int(found.get('volume_total', 0)) - int(ledger.get('volume_total', 0)):+,}")
        if found.get("disordered") or found.get("malformed"):
            lines.append(f"  {int(found.get('disordered', 0)) + int(found.get('malformed', 0)):,} values are malformed "
                         f"or have years out of order")
    read, expected = source.get("read") or {}, source.get("expected")
    if source.get("status") == "verified":
        lines.append(f"Source read in full: the units read {int(read.get('match_total', 0)):,} occurrences, "
                     f"exactly what the source database holds.")
    elif source.get("status") == "mismatch" and expected:
        lines.append(f"THE SOURCE WAS NOT READ EXACTLY ONCE: the units read {int(read.get('match_total', 0)):,} "
                     f"occurrences and the source holds {int(expected['match_total']):,} "
                     f"(difference {int(read.get('match_total', 0)) - int(expected['match_total']):+,}).")
    elif source.get("status") == "unknown":
        lines.append(f"Source total not known (the source was made before counts were verified): the units read "
                     f"{int(read.get('match_total', 0)):,} occurrences; nothing to compare that with.")
    if source.get("disordered"):
        lines.append(f"{int(source['disordered']):,} values in the SOURCE have years out of order.")
    balance = v.get("balance") or {}
    if balance.get("status") == "verified":
        r, e, d = (int(balance[k]["match_total"]) for k in ("read", "emitted", "dropped"))
        lines.append(f"Filtering accounted for: of {r:,} occurrences read, {e:,} were kept and {d:,} dropped "
                     f"({100.0 * d / r:.1f}%), and the two add up." if r else "Filtering accounted for: nothing was read.")
    elif balance.get("status") == "mismatch":
        r, e, d = (int(balance[k]["match_total"]) for k in ("read", "emitted", "dropped"))
        lines.append(f"KEPT AND DROPPED DO NOT ADD UP TO WHAT WAS READ: read {r:,} occurrences, kept {e:,}, dropped "
                     f"{d:,} (difference {int(balance['difference']['match_total']):+,}; volume counts "
                     f"{int(balance['difference']['volume_total']):+,}). Counts were lost or made inside the filter.")
    elif balance.get("status") == "unknown" and not ledger.get("unledgered"):
        lines.append("Kept and dropped not checked against what was read: some work units were filtered before "
                     "dropped counts were kept.")
    if ledger.get("reordered"):
        lines.append(f"{int(ledger['reordered']):,} emitted values had years out of order and were put in order.")
    return lines


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
    print(format_banner("Filtering" + (" and pivoting" if pivot else "")))
    print(f"Source:       {pc.src_db}")
    print(f"Destination:  {pc.dst_db}" + ("  (keyed year-first)" if pivot else ""))
    print(f"Mode:         {pc.mode}")
    print(f"Workers:      {pc.num_workers}")

    _require_compacted_source(pc.src_db, pc.reader_profile)

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
    # How lemmas were made is part of what this run's output is: a vocabulary
    # built one way must not be used to filter n-grams lemmatized the other
    # (lexichron.stages.filter reads this back from the vocabulary's record).
    lemma_gen = filter_config.lemma_gen if filter_config.apply_lemmatization else None
    lemmas = None
    if lemma_gen is not None:
        by_tag = getattr(lemma_gen, "_use_tags", getattr(lemma_gen, "use_tags", False))
        lemmas = LEMMAS_BY_TAG if by_tag else LEMMAS_LOOKUP
    progress = ProgressReporter(progress_path, stage="filter", db_path=pc.dst_db, log_path=log_file,
                                unit_label="work units",
                                inputs={"source": pc.src_db, "vocabulary": vocab or None, "lemmas": lemmas,
                                        "always_include": ", ".join(sorted(
                                            x.decode() if isinstance(x, bytes) else str(x)
                                            for x in (filter_config.always_include or ()))) or None,
                                        "always_include_case": ", ".join(sorted(
                                            x.decode() if isinstance(x, bytes) else str(x)
                                            for x in (filter_config.always_include_case or ()))) or None})
    num_units = pc.num_initial_work_units or max(4 * pc.num_workers, 64)
    # What the run goes through and what it will check, for whoever reads the
    # progress document ("step 2 of 4", what a problem will lead to).
    progress.set_plan(["ingesting", "compacting"] + (["verifying"] if pc.verify_counts else [])
                      + (["whitelist"] if pc.output_whitelist_path else []))
    progress.set_checks(verify_counts=bool(pc.verify_counts),
                        source_known=bool(pc.source_totals))
    if not pc.compact_after_ingest:
        print("Note: compact_after_ingest is ignored; a complete filter run is always compacted.")

    # Every unit is handed over as one sorted SST file of merge operands and
    # adopted by the database together with its marker in a single ingest
    # call: the unit's records and the fact that it is done are durable
    # together or not at all, and the parent replays nothing. (The pivoted
    # output is many times the size of its input, one record per year, and
    # was the first to go this way; the unigram run follows for the marker.)
    cfg = RangeMapConfig(
        src_db=Path(pc.src_db), dst_db=Path(pc.dst_db),
        inputs=_run_inputs(filter_config, vocab, lemmas, pivot),
        mapper=PivotFilterMapper(filter_config) if pivot else FilterMapper(filter_config),
        output="sst",
        num_units=num_units, workers=pc.num_workers, mode=mode,
        smart_partitioning=pc.use_smart_partitioning, samples_per_worker=pc.samples_per_worker,
        use_cached_partition=pc.use_cached_partitions and pc.cache_partitions,
        partition_cache_dir=paths["cache_dir"],
        read_profile=pc.reader_profile, write_profile=pc.ingest_write_profile,
        chunk_entries=chunk_entries, spool_dir=Path(spool_dir) if spool_dir else None,
        compact_after=True, progress=progress,
    )
    result = run_rangemap(cfg)
    year_records = f" ({result.produced:,} year records)" if pivot else ""
    print(f"Filtered {result.scanned:,} records -> {result.kept:,} kept{year_records} in "
          f"{result.units_done} work units ({result.units_skipped} already done) "
          f"in {result.seconds / 60:.1f} min")

    # The count check: the destination against what the units recorded
    # emitting, and what they recorded reading against what the source holds.
    if pc.verify_counts:
        with progress.long_phase("verifying"):
            print(format_banner("Count Verification"))
            logger.info("Verifying the counts: reading the whole destination once with up to %d processes and "
                        "comparing its totals with the work units' recorded totals", pc.num_workers)
            t_verify = time.perf_counter()
            verification = verify_database(str(pc.dst_db), pc.num_workers, marker_prefix=UNIT_MARKER_PREFIX)
            verification["source"] = _check_source(verification["ledger"], pc.source_totals)
            verification["balance"] = _check_balance(verification["ledger"])
            if "mismatch" in (verification["source"]["status"], verification["balance"]["status"]):
                verification["status"] = "mismatch"
            verification["seconds"] = round(time.perf_counter() - t_verify, 1)
        for line in describe_filter_verification(verification):
            print(line)
            logger.info("%s", line)
        logger.info("Verification took %.0fs; outcome: %s", verification["seconds"], verification["status"])
        progress.set_verification(verification)
        if verification["status"] == "mismatch":
            progress.finish("failed", "the counts in the database do not match what the work units recorded")
            logger.error("Stopping: the counts do not match. The database should not be used")
            raise FilterError(
                "The filtered database's counts do not match what the run recorded (figures above, and in the "
                "log). The database should not be used. Nothing was deleted: run again with mode: restart to "
                "build it again, and please report this.")
    else:
        progress.set_verification({"status": "skipped"})
        print("Count verification skipped (verify_counts is off).")
        logger.info("Count verification skipped (verify_counts is off)")

    if pc.output_whitelist_path:
        wl = Path(pc.output_whitelist_path)
        print(format_banner("Output whitelist"))
        print("  Path: " + truncate_path_to_fit(wl, "  Path: "))
        # The words the user asked to keep are in the vocabulary whatever
        # their rank: a later filter against it then keeps them without being
        # told again, and the file lists every word that corpus can hold.
        from ..filters.processor_factory import kept_forms
        keep = kept_forms(filter_config)
        with progress.long_phase("whitelist"):
            write_whitelist(db_or_path=pc.dst_db, dest=wl, top=pc.output_whitelist_top_n,
                            decode=True, sep="\t", spell_check=pc.output_whitelist_spell_check,
                            spell_check_language=pc.output_whitelist_spell_check_language,
                            year_range=pc.output_whitelist_year_range,
                            bin_size=filter_config.bin_size, keep=keep)
        try:
            with open(wl, "r", encoding="utf-8") as fh:
                words = [line.split("\t", 1)[0] for line in fh]
            n = len(words)
            top = pc.output_whitelist_top_n
            added = words[top:] if top and n > top else []
            if added:
                shown = ", ".join(added[:20]) + (f" and {len(added) - 20} more" if len(added) > 20 else "")
                print(f"  {n:,} tokens: the {top:,} most frequent and {len(added)} kept by request below them ({shown})")
                logger.info("Vocabulary: %d tokens, the %d most frequent and %d kept by request (always_include) "
                            "that rank below them: %s", n, top, len(added), shown)
            else:
                print(f"  {n:,} tokens")
        except OSError:
            pass

    progress.finish("done")
    logger.info("Filter finished in %.1f min: %s", (time.perf_counter() - t0) / 60, pc.dst_db)
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
    always_include_case: Optional[Set[str]] = None,
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
        always_include: Spellings kept whatever the rules say, matched case-folded
        always_include_case: The same, matched exactly
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
            always_include_case=_to_bytes_set(always_include_case) if always_include_case else None,
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

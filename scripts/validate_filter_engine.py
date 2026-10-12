#!/usr/bin/env python
"""Run the filter stage with the legacy pipeline and with the range-map engine
on the same corpus, and compare the two processed databases key by key.

    apptainer exec lexichron.sif python /opt/lexichron/scripts/validate_filter_engine.py \\
        --stub /scratch/$USER/NLP_corpora/Google_Books --release 20200217 --language eng-us \\
        --ngram-size 1 --workers 20

The legacy result goes to <n>grams_processed.legacy.db and the engine's to
<n>grams_processed.engine.db, so neither touches an existing
<n>grams_processed.db. Pass --skip-legacy / --skip-engine to reuse one side
from a previous run. Exit code 0 means identical.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path


def _configs(args, stub_lang):
    from ngramprep.ngram_filter.config import FilterConfig
    from ngramprep.ngram_filter.lemmatizer import CachedSpacyLemmatizer
    from ngramprep.ngram_filter.stopwords import load_stopwords

    stop_set, lang = load_stopwords(stub_lang)
    return FilterConfig(stop_set=stop_set, stop_words_language=lang,
                        lemma_gen=CachedSpacyLemmatizer(language=stub_lang),
                        ascii_alpha_only=True, min_context_tokens=1, min_len=3, bin_size=1)


def run_legacy(args, src, dst, tmp, filter_config):
    from ngramprep.ngram_filter.config import PipelineConfig
    from ngramprep.ngram_filter.pipeline.legacy_orchestrator import build_processed_db
    pc = PipelineConfig(src_db=src, dst_db=dst, tmp_dir=tmp, mode="restart", num_workers=args.workers,
                        num_initial_work_units=args.units, cache_partitions=False,
                        use_cached_partitions=False, progress_every_s=30, compact_after_ingest=True,
                        output_whitelist_path=dst / "whitelist.txt", output_whitelist_top_n=30_000,
                        output_whitelist_year_range=(1968, 2019), output_whitelist_spell_check=False)
    t = time.time()
    build_processed_db(filter_config=filter_config, pipeline_config=pc)
    return time.time() - t


def run_engine(args, src, dst, tmp, filter_config):
    from ngramprep.ngram_filter.config import PipelineConfig
    from ngramprep.ngram_filter.pipeline.orchestrator import build_processed_db
    pc = PipelineConfig(src_db=src, dst_db=dst, tmp_dir=tmp, mode="restart", num_workers=args.workers,
                        num_initial_work_units=args.units, cache_partitions=False,
                        use_cached_partitions=False, compact_after_ingest=True,
                        output_whitelist_path=dst / "whitelist.txt", output_whitelist_top_n=30_000,
                        output_whitelist_year_range=(1968, 2019), output_whitelist_spell_check=False)
    t = time.time()
    build_processed_db(filter_config=filter_config, pipeline_config=pc, spool_dir=args.spool_dir)
    return time.time() - t


def _records(blob):
    n = len(blob) // 8
    v = struct.unpack(f"<{n}Q", blob)
    return sorted((v[i], v[i + 1], v[i + 2]) for i in range(0, n, 3))


def compare(a_path, b_path, limit=20):
    from ngramprep.common_db.api import open_db, scan_all
    diffs = 0
    only_a = only_b = common = 0
    with open_db(a_path, mode="r", profile="read") as a, open_db(b_path, mode="r", profile="read") as b:
        ia, ib = scan_all(a), scan_all(b)
        ka = next(ia, None)
        kb = next(ib, None)
        while ka is not None or kb is not None:
            if ka is not None and ka[0].startswith(b"__"):
                ka = next(ia, None); continue
            if kb is not None and kb[0].startswith(b"__"):
                kb = next(ib, None); continue
            if kb is None or (ka is not None and ka[0] < kb[0]):
                only_a += 1
                if only_a <= limit: print(f"  only in legacy: {ka[0]!r}")
                ka = next(ia, None)
            elif ka is None or kb[0] < ka[0]:
                only_b += 1
                if only_b <= limit: print(f"  only in engine: {kb[0]!r}")
                kb = next(ib, None)
            else:
                common += 1
                if ka[1] != kb[1] and _records(ka[1]) != _records(kb[1]):
                    diffs += 1
                    if diffs <= limit: print(f"  value differs: {ka[0]!r}")
                ka = next(ia, None); kb = next(ib, None)
    print(f"compared {common:,} common keys; {diffs:,} value differences; "
          f"{only_a:,} only in legacy; {only_b:,} only in engine")
    return diffs == 0 and only_a == 0 and only_b == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--release", default="20200217")
    ap.add_argument("--language", default="eng-us")
    ap.add_argument("--ngram-size", type=int, default=1)
    ap.add_argument("--text-language", default="en")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--units", type=int, default=300)
    ap.add_argument("--spool-dir", default=None)
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--skip-engine", action="store_true")
    args = ap.parse_args()

    from ngramprep.ngram_acquire.db.build_path import build_db_path
    src = Path(build_db_path(args.stub, args.ngram_size, args.release, args.language))
    if not (src / "CURRENT").exists():
        release_dir = Path(args.stub) / args.release
        found = sorted(p.parent.parent.name + "/" + p.parent.name
                       for p in release_dir.glob(f"*/{args.ngram_size}gram_files/{args.ngram_size}grams.db/CURRENT"))
        sys.exit(f"no acquired database at {src}\n"
                 f"corpora with {args.ngram_size}-gram databases under {release_dir}: "
                 + (", ".join(found) if found else "none") + "\n(check --language and --release)")
    base = src.parent
    legacy_dst = base / f"{args.ngram_size}grams_processed.legacy.db"
    engine_dst = base / f"{args.ngram_size}grams_processed.engine.db"
    fc = _configs(args, args.text_language)

    if not args.skip_legacy:
        print(f"== legacy pipeline -> {legacy_dst}")
        secs = run_legacy(args, src, legacy_dst, base / "processing_tmp_legacy", fc)
        print(f"== legacy took {secs / 60:.1f} min")
    if not args.skip_engine:
        print(f"== range-map engine -> {engine_dst}")
        secs = run_engine(args, src, engine_dst, base / "processing_tmp_engine", fc)
        print(f"== engine took {secs / 60:.1f} min")

    print("== comparing databases")
    same = compare(legacy_dst, engine_dst)
    wl_a = (legacy_dst / "whitelist.txt").read_text().splitlines() if (legacy_dst / "whitelist.txt").exists() else None
    wl_b = (engine_dst / "whitelist.txt").read_text().splitlines() if (engine_dst / "whitelist.txt").exists() else None
    if wl_a is not None and wl_b is not None:
        print(f"whitelists: {len(wl_a):,} vs {len(wl_b):,} lines, {'identical' if wl_a == wl_b else 'DIFFERENT'}")
        same = same and wl_a == wl_b
    print("RESULT:", "identical" if same else "DIFFERENT")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()

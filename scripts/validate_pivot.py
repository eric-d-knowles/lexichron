#!/usr/bin/env python
"""Check a pivoted database against the processed database of the same corpus.

    apptainer run lexichron.sif /opt/lexichron/scripts/validate_pivot.py \\
        --stub /scratch/$USER/NLP_corpora/Google_Books --release 20200217 \\
        --language eng-us --ngram-size 5 [--samples 200] [--per-sample 100]

Both databases are filtered views of the same raw data, so for every n-gram
the processed record's (year, count, volumes) triples must appear in the
pivoted database as one record per year, and every pivoted record must be
one of those triples. The check samples `--samples` points spread across
each database's key space (from its SST file boundaries) and verifies
`--per-sample` consecutive records at each point, in both directions, with
point lookups. Exit code 0 means every sampled record agreed.
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path

PACK_YEAR = struct.Struct(">I")


def _sample_points(db, n, rng):
    files = sorted(f[0] for f in db.live_files() if not bytes(f[0]).startswith(b"__"))
    if not files:
        return [b""]
    return [files[rng.randrange(len(files))] for _ in range(n)]


def _records(db, start, limit):
    from ngramprep.common_db.api import range_scan
    out = []
    for key, value in range_scan(db, start, None, fill_cache=False):
        if key.startswith(b"__"):
            continue
        out.append((key, value))
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stub", required=True)
    ap.add_argument("--release", default="20200217")
    ap.add_argument("--language", default="eng-us")
    ap.add_argument("--ngram-size", type=int, default=5)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--per-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from ngramprep.common_db.api import open_db
    from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths
    paths = derive_paths(args.stub, args.ngram_size, args.release, args.language)
    processed, pivoted = paths["dst_db"], paths["pivoted_db"]
    for p in (processed, pivoted):
        if not (p / "CURRENT").exists():
            sys.exit(f"no database at {p}")
    rng = random.Random(args.seed)
    checked = missing = differing = 0
    examples = []

    with open_db(processed, mode="r", profile="scan:packed24") as pdb, \
            open_db(pivoted, mode="r", profile="scan:packed24") as vdb:
        print(f"== processed -> pivoted: {args.samples} x {args.per_sample} n-grams")
        for start in _sample_points(pdb, args.samples, rng):
            for ngram, value in _records(pdb, start, args.per_sample):
                for i in range(0, len(value), 24):
                    triple = value[i:i + 24]
                    year = int.from_bytes(triple[:8], "little")
                    got = vdb.get(PACK_YEAR.pack(year) + ngram)
                    checked += 1
                    if got is None:
                        missing += 1
                        if len(examples) < 20:
                            examples.append(f"missing in pivoted: {year} {ngram!r}")
                    elif got != triple:
                        differing += 1
                        if len(examples) < 20:
                            examples.append(f"differs: {year} {ngram!r}: processed {struct.unpack('<3Q', triple)} "
                                            f"pivoted {struct.unpack('<3Q', got) if len(got) == 24 else got!r}")
        print(f"   {checked:,} year records checked: {missing:,} missing, {differing:,} differing")

        print(f"== pivoted -> processed: {args.samples} x {args.per_sample} records")
        checked2 = extra = differing2 = 0
        for start in _sample_points(vdb, args.samples, rng):
            for key, value in _records(vdb, start, args.per_sample):
                year, ngram = PACK_YEAR.unpack(key[:4])[0], key[4:]
                pv = pdb.get(ngram)
                checked2 += 1
                triples = {} if pv is None else {int.from_bytes(pv[i:i + 8], "little"): pv[i:i + 24]
                                                 for i in range(0, len(pv), 24)}
                if year not in triples:
                    extra += 1
                    if len(examples) < 20:
                        examples.append(f"not in processed: {year} {ngram!r}")
                elif triples[year] != value:
                    differing2 += 1
                    if len(examples) < 20:
                        examples.append(f"differs: {year} {ngram!r}: pivoted {struct.unpack('<3Q', value)} "
                                        f"processed {struct.unpack('<3Q', triples[year])}")
        print(f"   {checked2:,} records checked: {extra:,} not in processed, {differing2:,} differing")
        for p, name in ((pdb, "processed"), (vdb, "pivoted")):
            print(f"   {name}: ~{int(p.get_property('rocksdb.estimate-num-keys') or 0):,} keys, "
                  f"{int(p.get_property('rocksdb.total-sst-files-size') or 0) / 2**30:.1f} GiB")

    for e in examples:
        print("   " + e)
    bad = missing + differing + extra + differing2
    print("RESULT: " + ("consistent" if bad == 0 else f"{bad:,} disagreements"))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()

"""Run the acquire stage on local shards, for tests that kill it.

Not a test module: ``test_acquire_kill.py`` starts this as a separate process
against the real RocksDB, so that the process can be killed outright and the
database reopened by the next run, as happens to a job that hits its time
limit or is cancelled.

    python acquire_kill_driver.py <dir> [--spool-format sst|chunk] [--slow-ms N]
                                  [--die-after-ingest N | --die-before-ingest N]

``<dir>/shards/*.gz`` stand in for the repository (the listing and the
download are replaced; nothing touches the network) and the database is built
under ``<dir>/out``. ``--die-after-ingest N`` makes the parent exit, as if
killed, the moment its N-th shard has been handed to the database and before
it does anything else; ``--die-before-ingest N`` just before the N-th.
Exit status: 0 done, 3 the stage raised (the message is printed), 9 died as asked.
"""
import argparse
import os
import sys
import time
from pathlib import Path


class _LocalResponse:
    def __init__(self, path):
        self.raw = open(path, "rb")
        self.headers = {"content-length": str(os.path.getsize(path))}

    def close(self):
        self.raw.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--spool-format", default="sst")
    ap.add_argument("--slow-ms", type=int, default=0)
    ap.add_argument("--die-after-ingest", type=int, default=0)
    ap.add_argument("--die-before-ingest", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)

    from ngramprep.ngram_acquire import core, worker
    from ngramprep.ngram_acquire.batch_writer import ChunkIngestor

    names = sorted(p.name for p in (root / "shards").glob("*.gz"))

    def discover(n, release, corpus):
        (root / "started").touch()                      # the run is about to open its database
        return "http://local/", [f"http://local/{name}" for name in names]

    def download(url, session=None):
        if args.slow_ms:
            time.sleep(args.slow_ms / 1000.0)
        return _LocalResponse(root / "shards" / url.rsplit("/", 1)[-1])

    core.discover_files = discover
    worker.stream_download_with_retries = download      # inherited by the forked workers

    if args.die_after_ingest or args.die_before_ingest:
        ingest_file = ChunkIngestor.ingest_file
        seen = {"n": 0}

        def hooked(self, *a, **kw):
            seen["n"] += 1
            if seen["n"] == args.die_before_ingest:
                os._exit(9)
            out = ingest_file(self, *a, **kw)
            if seen["n"] == args.die_after_ingest:
                os._exit(9)                             # the shard is in; nothing after it has happened
            return out

        ChunkIngestor.ingest_file = hooked

    try:
        core.download_and_ingest_to_rocksdb(
            ngram_size=2, repo_release_id="20200217", repo_corpus_id="eng", db_path_stub=str(root / "out"),
            ngram_type="all", workers=3, chunk_entries=150, spool_dir=str(root / "spool"),
            spool_format=args.spool_format, progress_path=str(root / "progress.json"))
    except core.AcquisitionError as exc:
        print(f"ACQUISITION ERROR: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

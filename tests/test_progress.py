import json

from ngramprep.ngram_acquire.progress import ProgressReporter, read_progress


def test_progress_document_lifecycle(tmp_path):
    p = tmp_path / "runs" / "x" / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0)
    assert read_progress(p)["state"] == "running"
    r.set_totals(14, files_skipped=2)
    r.file_started("a.gz"); r.file_started("b.gz")
    assert read_progress(p)["current"] == ["a.gz", "b.gz"]
    r.file_done("a.gz", entries=10, chunks=1, uncompressed_bytes=100)
    r.file_failed("b.gz", "NETWORK_ERROR: b.gz")
    d = read_progress(p)
    assert (d["files_total"], d["files_skipped"], d["files_done"], d["files_failed"]) == (14, 2, 1, 1)
    assert d["entries_written"] == 10 and d["current"] == [] and "NETWORK_ERROR" in d["message"]
    r.finish("failed", "1 file(s) failed")
    d = read_progress(p)
    assert d["state"] == "failed" and d["finished"]
    assert not list(p.parent.glob("*.tmp"))          # atomic writes leave nothing behind


def test_reporter_without_path_is_inert(tmp_path):
    r = ProgressReporter(None, stage="acquire")
    r.set_totals(1); r.file_done("a", entries=1, chunks=1, uncompressed_bytes=1); r.finish()
    assert read_progress(tmp_path / "missing.json") is None


def test_rate_limiting_keeps_forced_writes(tmp_path):
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=1000)
    r.file_started("a.gz")                      # not forced: may be skipped
    r.file_done("a.gz", entries=5, chunks=1, uncompressed_bytes=1)   # forced
    assert read_progress(p)["files_done"] == 1


def test_database_figure_leaves_out_bookkeeping_keys(tmp_path):
    import fake_rocks
    from ngramprep.ngram_acquire.progress import ProgressReporter, read_progress
    db = fake_rocks.open(str(tmp_path / "db"), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        for i in range(10):
            wb.put(f"k{i}".encode(), b"v")
        wb.put(b"__processed__/a.gz", b"1")
        wb.put(b"__processed__/b.gz", b"1")
    p = tmp_path / "progress.json"
    r = ProgressReporter(p, stage="acquire", min_interval_s=0)
    r.set_database(db, compacted=True, bookkeeping=2)
    assert read_progress(p)["database"] == {"records": 10, "bytes": read_progress(p)["database"]["bytes"],
                                            "compacted": True}
    r.set_database(db, compacted=False, bookkeeping=2)
    assert read_progress(p)["database"]["compacted"] is False


def test_write_survives_its_directory_being_removed(tmp_path):
    """A UI clearing runs can remove a running job's run directory from
    under it; the next write puts the directory back rather than raising,
    so the run reappears and, above all, the job goes on."""
    import shutil
    p = tmp_path / "runs" / "r1" / "progress.json"
    r = ProgressReporter(p, stage="tune", min_interval_s=0)
    shutil.rmtree(p.parent)
    r.set_totals(10)                                   # would have raised FileNotFoundError
    assert read_progress(p)["files_total"] == 10
    # a write that cannot succeed at all is reported, not raised
    r.path = tmp_path / "nowhere"
    r.path.write_text("a file where the directory should be")
    r.path = r.path / "progress.json"
    r.set_totals(11)
    assert r._write_failed

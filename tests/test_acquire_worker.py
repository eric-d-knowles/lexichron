"""End-to-end test of a worker processing one file, with the download stubbed."""
import gzip
import io
import struct
from pathlib import Path

import pytest

from ngramprep.ngram_acquire import worker as worker_mod


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.raw = io.BytesIO(payload)
        self.headers = {"content-length": str(len(payload))}

    def close(self):
        pass


def _gz(lines):
    return gzip.compress("\n".join(lines).encode("utf-8"))


def _unpack(blob):
    n = len(blob) // 8
    v = struct.unpack(f"<{n}Q", blob)
    return [tuple(v[i:i + 3]) for i in range(0, n, 3)]


from ngramprep.ngram_acquire.spool import read_chunk


def _collect(chunk_paths):
    out = {}
    for p in chunk_paths:
        for k, v in read_chunk(p):
            out[k.decode()] = v
    return out


def test_worker_sums_colliding_tagged_variants(monkeypatch, tmp_path):
    payload = _gz([
        "working_NOUN class_NOUN in_ADP\t1990,500,80\t2000,600,90",
        "working_NOUN class_VERB in_ADP\t1990,20,5",
        "other_NOUN thing_NOUN here_ADV\t1990,1,1",
    ])
    monkeypatch.setattr(
        worker_mod, "stream_download_with_retries",
        lambda url, session=None: _FakeResponse(payload),
    )

    msg, chunks, nbytes, entries, _lines, _stats = worker_mod.process_and_ingest_file(
        "http://example/3-00000-of-00001.gz", 1,
        filter_pred=None, log_file_path=None,
        combined_multigrams={"working class"}, spool_dir=str(tmp_path), spool_format="chunk",
    )

    assert msg.startswith("SUCCESS")
    data = _collect(chunks)
    assert set(data) == {"working-class_NOUN in_ADP", "other_NOUN thing_NOUN here_ADV"}
    assert _unpack(data["working-class_NOUN in_ADP"]) == [(1990, 520, 85), (2000, 600, 90)]
    assert nbytes > 0 and entries == 2


def test_worker_splits_into_chunks_and_bounds_memory(monkeypatch, tmp_path):
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(1000)])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _FakeResponse(payload))
    msg, chunks, _, entries, _lines, _stats = worker_mod.process_and_ingest_file(
        "http://example/f.gz", 1, spool_dir=str(tmp_path), chunk_entries=300, spool_format="chunk")
    assert msg.startswith("SUCCESS")
    assert len(chunks) == 4 and entries == 1000          # 300+300+300+100
    assert len(_collect(chunks)) == 1000
    assert not list(tmp_path.glob("*.part"))


def test_worker_without_combining_keeps_variants_separate(monkeypatch, tmp_path):
    payload = _gz([
        "working_NOUN class_NOUN in_ADP\t1990,500,80",
        "working_NOUN class_VERB in_ADP\t1990,20,5",
    ])
    monkeypatch.setattr(
        worker_mod, "stream_download_with_retries",
        lambda url, session=None: _FakeResponse(payload),
    )
    _, chunks, _, entries, _lines, _stats = worker_mod.process_and_ingest_file(
        "http://example/f.gz", 1, spool_dir=str(tmp_path), spool_format="chunk")
    assert entries == 2 and len(_collect(chunks)) == 2


class _DroppingResponse:
    """Streams a few bytes, then the connection dies."""
    def __init__(self, payload: bytes, drop_after: int):
        import requests
        self._exc = requests.ConnectionError("connection reset by peer")
        self.headers = {}
        self.raw = self._Raw(payload, drop_after, self._exc)
    class _Raw:
        def __init__(self, payload, drop_after, exc):
            self.buf = io.BytesIO(payload); self.drop_after = drop_after; self.exc = exc; self.read_bytes = 0
        def read(self, n=-1):
            if self.read_bytes >= self.drop_after:
                raise self.exc
            chunk = self.buf.read(n if n and n > 0 else 1024)
            self.read_bytes += len(chunk)
            return chunk
        def readinto(self, b):
            data = self.read(len(b)); b[:len(data)] = data; return len(data)
        def readable(self): return True
        def close(self): pass
    def close(self): pass


def test_worker_retries_a_connection_dropped_mid_stream(monkeypatch, tmp_path):
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)])
    calls = {"n": 0}

    def fake_download(url, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _DroppingResponse(payload, drop_after=64)
        return _FakeResponse(payload)

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    msg, chunks, _, entries, _lines, _stats = worker_mod.process_and_ingest_file(
        "http://example/f.gz", 1, spool_dir=str(tmp_path), chunk_entries=500, spool_format="chunk")
    assert msg.startswith("SUCCESS")
    assert calls["n"] == 2
    # The failed attempt's partial chunks were discarded: no double counting.
    assert entries == 2000 and len(chunks) == 4
    assert len(_collect(chunks)) == 2000
    # Only the chunks and the worker's "I have this file" marker remain.
    assert sorted(tmp_path.iterdir()) == sorted(list(map(type(tmp_path), chunks)) + [tmp_path / "00001_f.gz.parsed"])


def test_worker_gives_up_after_max_attempts(monkeypatch, tmp_path):
    payload = _gz(["a_NOUN\t1990,1,1"])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _DroppingResponse(payload, drop_after=0))
    msg, chunks, nbytes, entries, _lines, _stats = worker_mod.process_and_ingest_file(
        "http://example/f.gz", 1, max_attempts=2, spool_dir=str(tmp_path), spool_format="chunk")
    assert msg == "NETWORK_ERROR: f.gz"
    assert chunks == [] and nbytes == 0 and entries == 0
    assert [p.name for p in tmp_path.iterdir()] == ["00001_f.gz.started"]   # the parent removes it


def test_session_is_closed_when_worker_creates_it(monkeypatch, tmp_path):
    import requests
    closed = {"n": 0}
    class S:
        def close(self): closed["n"] += 1
    monkeypatch.setattr(requests, "Session", S)
    payload = _gz(["a_NOUN\t1990,1,1"])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _FakeResponse(payload))
    worker_mod.process_and_ingest_file("http://example/f.gz", 1, spool_dir=str(tmp_path), spool_format="chunk")
    assert closed["n"] == 1


# ---- the checks a worker makes on what it spools --------------------------

def _run(monkeypatch, tmp_path, lines, **kw):
    payload = lines if isinstance(lines, bytes) else _gz(lines)
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _FakeResponse(payload))
    kw.setdefault("spool_format", "chunk")
    return worker_mod.process_and_ingest_file("http://example/f.gz", 1, spool_dir=str(tmp_path), **kw)


@pytest.mark.parametrize("fast", [True, False])
def test_worker_totals_what_it_keeps(monkeypatch, tmp_path, fast):
    """match_total and volume_total are the frequencies and document counts
    of every kept line, before any summing of repeated keys: the figures the
    finished database is checked against."""
    if not fast:
        monkeypatch.setattr(worker_mod, "_parse_packed", None)
    from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate
    msg, chunks, _, entries, lines, stats = _run(monkeypatch, tmp_path, [
        "working_NOUN class_NOUN in_ADP\t1990,500,80\t2000,600,90",
        "working_NOUN class_VERB in_ADP\t1990,20,5",              # collapses onto the line above
        "other_NOUN thing_NOUN here_ADV\t1990,1,1",
        "untagged words here\t1990,1000,1000",                    # filtered out: not in the totals
    ], filter_pred=make_ngram_type_predicate("tagged"), combined_multigrams={"working class"})
    assert msg.startswith("SUCCESS") and entries == 2 and lines == 4
    assert stats == {"match_total": 500 + 600 + 20 + 1, "volume_total": 80 + 90 + 5 + 1,
                     "problem_lines": 0, "reordered": 0}
    data = _collect(chunks)
    assert sum(f for v in data.values() for _y, f, _d in _unpack(v)) == stats["match_total"]


@pytest.mark.parametrize("fast", [True, False])
def test_worker_counts_lines_it_cannot_read_in_full(monkeypatch, tmp_path, fast):
    """A line with unreadable counts is never dropped in silence: what can be
    read of it is kept, and the line is counted and logged."""
    if not fast:
        monkeypatch.setattr(worker_mod, "_parse_packed", None)
    payload = gzip.compress(b"\n".join([
        b"good_NOUN\t1990,1,1",
        b"short_NOUN\t1990,5\t1991,2,2",                           # one entry of two fields
        b"junk_NOUN\t1990,x,1",                                    # nothing readable
        b"notab_NOUN",                                             # no counts at all
        b"big_NOUN\t1990,18446744073709551616,1",                   # does not fit the packed format
        b"bad\xff_NOUN\t1990,1,1",                                 # not UTF-8
        b"",                                                       # a blank line is not a problem
        b"fine_NOUN\t1990,2,2",
    ]))
    msg, chunks, _, entries, lines, stats = _run(monkeypatch, tmp_path, payload)
    assert msg.startswith("SUCCESS") and "5 lines not read in full" in msg
    assert stats["problem_lines"] == 5
    data = _collect(chunks)
    assert set(data) == {"good_NOUN", "short_NOUN", "fine_NOUN"}
    assert _unpack(data["short_NOUN"]) == [(1991, 2, 2)]          # the readable entry is kept
    assert stats["match_total"] == 1 + 2 + 2


def test_worker_puts_years_in_order(monkeypatch, tmp_path):
    """The merge operator assumes ascending years and checks nothing; a value
    that arrives otherwise is rewritten in order (repeated years summed) and
    counted, with its totals unchanged."""
    msg, chunks, _, entries, _lines, stats = _run(monkeypatch, tmp_path, [
        "a_NOUN\t1990,1,1\t1991,2,2",
        "b_NOUN\t2000,5,1\t1990,7,2",                              # out of order
        "c_NOUN\t1990,1,1\t1990,2,2",                              # a year twice
    ])
    assert msg.startswith("SUCCESS") and "2 values put in year order" in msg
    data = _collect(chunks)
    assert _unpack(data["b_NOUN"]) == [(1990, 7, 2), (2000, 5, 1)]
    assert _unpack(data["c_NOUN"]) == [(1990, 3, 3)]
    assert stats == {"match_total": 3 + 12 + 3, "volume_total": 3 + 3 + 3, "problem_lines": 0, "reordered": 2}


def test_worker_refuses_a_body_shorter_than_announced(monkeypatch, tmp_path):
    """A complete gzip stream that is nevertheless shorter than the server's
    Content-Length is not taken as the whole file."""
    payload = _gz(["a_NOUN\t1990,1,1"])

    class Short(_FakeResponse):
        def __init__(self):
            super().__init__(payload)
            self.headers = {"content-length": str(len(payload) + 10)}

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda url, session=None: Short())
    msg, chunks, *_ = worker_mod.process_and_ingest_file(
        "http://example/f.gz", 1, max_attempts=2, spool_dir=str(tmp_path), spool_format="chunk")
    assert msg == "NETWORK_ERROR: f.gz" and chunks == []
    assert not list(tmp_path.glob("*.chunk")) and not list(tmp_path.glob("*.part"))


def test_worker_writes_one_sorted_sst_per_shard(monkeypatch, tmp_path):
    """The default spool: the shard's parts are merged into a single SST file
    of merge operands, keys in byte order, repeated keys summed."""
    import fake_rocks
    from ngramprep.common_db import api
    monkeypatch.setattr(api, "rs", fake_rocks)
    lines = [f"w{i:04d}_NOUN\t1990,{i},1" for i in range(1000)] + ["w0007_NOUN\t1990,100,1", "\u00e9t\u00e9_NOUN\t1990,1,1"]
    msg, paths, _, entries, _lines, stats = _run(monkeypatch, tmp_path, lines, chunk_entries=300, spool_format="sst")
    assert msg.startswith("SUCCESS") and len(paths) == 1 and paths[0].endswith(".sst")
    import pickle
    with open(paths[0], "rb") as fh:
        _compression, records = pickle.load(fh)
    assert {kind for kind, _k, _v in records} == {"merge"}
    keys = [k for _kind, k, _v in records]
    assert keys == sorted(keys) and len(keys) == len(set(keys)) == entries == 1001
    assert _unpack(dict((k, v) for _kind, k, v in records)[b"w0007_NOUN"]) == [(1990, 107, 2)]
    assert "\u00e9t\u00e9_NOUN".encode("utf-8") in keys
    assert stats["match_total"] == sum(range(1000)) + 100 + 1
    assert [p.name for p in tmp_path.iterdir() if p.suffix not in (".sst", ".parsed")] == []


def test_worker_reports_how_far_it_is_into_its_file(monkeypatch, tmp_path):
    """Every few seconds, at a block boundary, the worker rewrites its marker
    with the lines read and the compressed bytes received, so the progress
    display can count files that are still being read. The finished file's
    marker holds the whole file."""
    monkeypatch.setattr(worker_mod, "_REPORT_EVERY_S", 0.0)         # report at every block
    monkeypatch.setattr(worker_mod, "_READ_BLOCK", 4096)
    seen = []
    real = worker_mod._report
    monkeypatch.setattr(worker_mod, "_report", lambda path, lines, resp: (real(path, lines, resp),
                                                                         seen.append(open(path).read()))[0])
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)])
    msg, chunks, _, entries, lines, _stats = _run(monkeypatch, tmp_path, payload)
    assert msg.startswith("SUCCESS") and lines == 2000
    counts = [tuple(int(x) for x in text.split()) for text in seen]
    assert counts[0] == (0, 0) and len(counts) > 3                 # picked up, then several reports
    assert [c[0] for c in counts] == sorted(c[0] for c in counts)  # lines only go up
    assert counts[-1] == (2000, len(payload))                      # the whole file: lines, compressed bytes
    assert (tmp_path / "00001_f.gz.parsed").read_text() == f"2000 {len(payload)}"


def test_worker_report_starts_over_with_a_retry(monkeypatch, tmp_path):
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)])
    calls = {"n": 0}

    def fake_download(url, session=None):
        calls["n"] += 1
        return _DroppingResponse(payload, drop_after=4096) if calls["n"] == 1 else _FakeResponse(payload)

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    seen = []
    real = worker_mod._report
    monkeypatch.setattr(worker_mod, "_report", lambda path, lines, resp: (real(path, lines, resp), seen.append(lines))[0])
    msg, *_ = worker_mod.process_and_ingest_file("http://example/f.gz", 1, spool_dir=str(tmp_path), spool_format="chunk")
    assert msg.startswith("SUCCESS") and calls["n"] == 2
    assert seen[0] == 0 and 0 in seen[1:] and seen[-1] == 2000     # back to zero for the second attempt


# ---- the log: every entry says which file, where it stands in the run, and which worker ----

def test_worker_log_entries_name_the_file_and_what_is_being_done(monkeypatch, tmp_path, caplog):
    import logging
    monkeypatch.setattr(worker_mod, "_REPORT_EVERY_S", 0.0)
    monkeypatch.setattr(worker_mod, "_LOG_EVERY_S", 0.0)
    monkeypatch.setattr(worker_mod, "_READ_BLOCK", 4096)
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)] + ["bad_NOUN\t1990,x,1"])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda url, session=None: _FakeResponse(payload))
    with caplog.at_level(logging.INFO):
        worker_mod.process_and_ingest_file("http://example/5-00450-of-11145.gz", 12, spool_dir=str(tmp_path),
                                           spool_format="chunk", chunk_entries=700, files_total=40)
    records = [r for r in caplog.records if r.name.startswith("acquire.worker.")]
    text = [r.getMessage() for r in records]
    # run in this process, so it is "worker 00"; a pool's processes are 01, 02, ...
    assert {r.name for r in records} == {"acquire.worker.00"}
    assert all(m.startswith("5-00450-of-11145.gz (file 12 of 40): ") for m in text)
    steps = [m.split("): ", 1)[1] for m in text]
    assert steps[0] == "opening the download"
    assert steps[1] == f"downloading and parsing as it arrives ({len(payload) / 1e6:,.1f} MB compressed)"
    assert any(m.startswith("still reading after") and "lines" in m and "kept so far" in m and "MB received" in m
               for m in steps)
    assert any(m.startswith("line 2,001 could not be read in full") and "bad_NOUN" in m for m in steps)
    assert any(m.startswith("read to the end in") and "2,001 lines" in m and m.endswith("writing 3 chunk files")
               for m in steps)
    assert any(m.startswith("done in") and "2,000 entries from 2,001 lines, in 3 chunk files" in m
               and m.endswith("waiting for the parent to ingest it") for m in steps)
    assert steps[-1] == "1 line could not be read in full; n-grams on such lines are undercounted"
    # what it says while reading is true: lines only grow, and so does what was kept
    reading = [m for m in steps if m.startswith("still reading")]
    assert "1,877 lines, 1,877 kept so far" in reading[-1]
    assert [r.levelname for r in records if "could not be read" in r.getMessage()] == ["WARNING", "WARNING"]


def test_worker_log_says_when_nothing_was_kept_and_when_a_download_breaks_off(monkeypatch, tmp_path, caplog):
    import logging
    from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate
    payload = _gz(["plain words\t1990,1,1"] * 3)
    calls = {"n": 0}

    def fake_download(url, session=None):
        calls["n"] += 1
        return _DroppingResponse(payload, drop_after=0) if calls["n"] == 1 else _FakeResponse(payload)

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    with caplog.at_level(logging.INFO):
        worker_mod.process_and_ingest_file("http://example/f.gz", 3, make_ngram_type_predicate("tagged"),
                                           spool_dir=str(tmp_path), spool_format="chunk")
    steps = [r.getMessage().split("): ", 1)[1] for r in caplog.records if r.name.startswith("acquire.worker.")]
    assert any(m.startswith("the download broke off after 0 lines (attempt 1 of 3:") and
               m.endswith("discarded what was read, starting the file again") for m in steps)
    assert "opening the download (attempt 2 of 3)" in steps
    assert any("writing nothing (no line was kept)" in m for m in steps)
    assert any("nothing kept from 3 lines; waiting for the parent to mark it processed" in m for m in steps)


def test_workers_and_the_parent_can_share_one_log_file(tmp_path):
    """The parent used to open its log for plain writing: it kept its own
    position in the file and overwrote every line the workers appended, so
    no worker's entry ever survived. All handles append now."""
    import logging
    from ngramprep.ngram_acquire.logger import setup_logger
    root = logging.getLogger()
    saved, level = root.handlers[:], root.level
    root.handlers = []
    try:
        log_path = setup_logger(tmp_path, filename_prefix="t")
        logging.getLogger("parent").info("before the workers")
        other = logging.FileHandler(log_path, mode="a")            # as a worker process opens it
        other.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        worker = logging.getLogger("acquire.worker.test")
        worker.addHandler(other); worker.setLevel(logging.INFO); worker.propagate = False
        for i in range(50):
            worker.info("worker entry %d", i)
        logging.getLogger("parent").info("after the workers")
        other.close(); worker.removeHandler(other)
        for h in root.handlers:
            h.close()
        text = Path(log_path).read_text()
    finally:
        root.handlers, _ = saved, root.setLevel(level)
    assert text.count("worker entry") == 50 and "before the workers" in text and "after the workers" in text


@pytest.mark.parametrize("error, kind", [
    ("protocol", "the download broke off"),          # "Connection broken: IncompleteRead(...)", as the network raises it
    ("incomplete", "the download broke off"),
    ("read_timeout", "the download timed out"),
])
def test_worker_retries_what_a_broken_stream_really_raises(monkeypatch, tmp_path, caplog, error, kind):
    """The body is read from the raw urllib3 stream, so a connection that
    breaks or stalls mid-file raises urllib3's errors, not requests'. They
    used to slip past the retry and fail the file on its first attempt."""
    import http.client
    import logging
    import urllib3.exceptions as u3
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)])
    raised = {
        "protocol": u3.ProtocolError("Connection broken: IncompleteRead(64 bytes read, 999 more expected)",
                                     u3.IncompleteRead(64, 999)),
        "incomplete": http.client.IncompleteRead(b"", 999),
        "read_timeout": u3.ReadTimeoutError(None, "http://example/f.gz", "Read timed out."),
    }[error]
    calls = {"n": 0}

    def fake_download(url, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            resp = _DroppingResponse(payload, drop_after=64)
            resp.raw.exc = raised
            return resp
        return _FakeResponse(payload)

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    with caplog.at_level(logging.INFO):
        msg, chunks, _, entries, *_ = worker_mod.process_and_ingest_file(
            "http://example/f.gz", 1, spool_dir=str(tmp_path), chunk_entries=500, spool_format="chunk")
    assert msg.startswith("SUCCESS") and calls["n"] == 2
    assert entries == 2000 and len(_collect(chunks)) == 2000        # the first attempt left nothing behind
    steps = [r.getMessage().split("): ", 1)[1] for r in caplog.records if r.name.startswith("acquire.worker.")]
    assert any(m.startswith(kind) and m.endswith("starting the file again") for m in steps)


def test_worker_gives_up_on_a_stream_that_keeps_breaking(monkeypatch, tmp_path):
    import urllib3.exceptions as u3
    payload = _gz(["a_NOUN\t1990,1,1"])

    def fake_download(url, session=None):
        resp = _DroppingResponse(payload, drop_after=0)
        resp.raw.exc = u3.ProtocolError("Connection broken", u3.IncompleteRead(0, 9))
        return resp

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    msg, chunks, *_ = worker_mod.process_and_ingest_file("http://example/f.gz", 1, max_attempts=3,
                                                         spool_dir=str(tmp_path), spool_format="chunk")
    assert msg == "NETWORK_ERROR: f.gz" and chunks == []           # after three attempts, not one
    assert not list(tmp_path.glob("*.chunk")) and not list(tmp_path.glob("*.part"))

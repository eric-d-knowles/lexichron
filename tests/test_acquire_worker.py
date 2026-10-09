"""End-to-end test of a worker processing one file, with the download stubbed."""
import gzip
import io
import struct

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


def test_worker_sums_colliding_tagged_variants(monkeypatch):
    payload = _gz([
        "working_NOUN class_NOUN in_ADP\t1990,500,80\t2000,600,90",
        "working_NOUN class_VERB in_ADP\t1990,20,5",
        "other_NOUN thing_NOUN here_ADV\t1990,1,1",
    ])
    monkeypatch.setattr(
        worker_mod, "stream_download_with_retries",
        lambda url, session=None: _FakeResponse(payload),
    )

    msg, data, nbytes = worker_mod.process_and_ingest_file(
        "http://example/3-00000-of-00001.gz", 1,
        filter_pred=None, log_file_path=None,
        combined_bigrams={"working class"},
    )

    assert msg.startswith("SUCCESS")
    assert set(data) == {"working-class_NOUN in_ADP", "other_NOUN thing_NOUN here_ADV"}
    assert _unpack(data["working-class_NOUN in_ADP"]) == [(1990, 520, 85), (2000, 600, 90)]
    assert nbytes > 0


def test_worker_without_combining_keeps_variants_separate(monkeypatch):
    payload = _gz([
        "working_NOUN class_NOUN in_ADP\t1990,500,80",
        "working_NOUN class_VERB in_ADP\t1990,20,5",
    ])
    monkeypatch.setattr(
        worker_mod, "stream_download_with_retries",
        lambda url, session=None: _FakeResponse(payload),
    )
    _, data, _ = worker_mod.process_and_ingest_file("http://example/f.gz", 1)
    assert len(data) == 2


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


def test_worker_retries_a_connection_dropped_mid_stream(monkeypatch):
    payload = _gz([f"w{i}_NOUN\t1990,{i},1" for i in range(2000)])
    calls = {"n": 0}

    def fake_download(url, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _DroppingResponse(payload, drop_after=64)
        return _FakeResponse(payload)

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", fake_download)
    msg, data, _ = worker_mod.process_and_ingest_file("http://example/f.gz", 1)
    assert msg.startswith("SUCCESS")
    assert calls["n"] == 2
    assert len(data) == 2000


def test_worker_gives_up_after_max_attempts(monkeypatch):
    payload = _gz(["a_NOUN\t1990,1,1"])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _DroppingResponse(payload, drop_after=0))
    msg, data, nbytes = worker_mod.process_and_ingest_file("http://example/f.gz", 1, max_attempts=2)
    assert msg == "NETWORK_ERROR: f.gz"
    assert data == {} and nbytes == 0


def test_session_is_closed_when_worker_creates_it(monkeypatch):
    import requests
    closed = {"n": 0}
    class S:
        def close(self): closed["n"] += 1
    monkeypatch.setattr(requests, "Session", S)
    payload = _gz(["a_NOUN\t1990,1,1"])
    monkeypatch.setattr(worker_mod, "stream_download_with_retries",
                        lambda url, session=None: _FakeResponse(payload))
    worker_mod.process_and_ingest_file("http://example/f.gz", 1)
    assert closed["n"] == 1

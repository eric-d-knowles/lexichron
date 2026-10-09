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

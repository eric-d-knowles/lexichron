import random

from ngramprep.ngram_acquire.coordinator import randomize_file_order, select_file_subset


def test_randomize_is_deterministic_and_leaves_global_random_alone():
    random.seed(12345)
    before = random.random()
    urls = [f"u{i}" for i in range(20)]
    a, b = list(urls), list(urls)
    randomize_file_order(a, seed=7)
    randomize_file_order(b, seed=7)
    assert a == b and a != urls
    random.seed(12345)
    assert random.random() == before  # global generator untouched by the shuffle


def test_select_file_subset():
    urls = ["a", "b", "c", "d"]
    assert select_file_subset(urls, None) == (urls, 0, 3)
    assert select_file_subset(urls, (1, 2)) == (["b", "c"], 1, 2)
    import pytest
    with pytest.raises(ValueError):
        select_file_subset(urls, (2, 9))


def test_processed_totals_sums_marker_figures():
    from ngramprep.ngram_acquire.coordinator import processed_totals
    from ngramprep.ngram_acquire.db.metadata import processed_key, processed_value

    class DB:
        store = {
            processed_key("a.gz"): processed_value(entries=10, uncompressed_bytes=1000, chunks=1),
            processed_key("b.gz"): processed_value(entries=5, uncompressed_bytes=500, chunks=1),
            processed_key("old.gz"): b"1",
        }
        def get(self, k): return self.store.get(k)

    urls = [f"http://x/{n}" for n in ("a.gz", "b.gz", "old.gz", "new.gz")]
    assert processed_totals(urls, DB()) == {"files": 3, "entries": 15, "bytes": 1500, "unsized": 1}

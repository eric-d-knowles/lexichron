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

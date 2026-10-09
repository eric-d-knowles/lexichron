import pytest

from ngramprep.ngram_acquire.io.locations import build_location_info, SUPPORTED_RELEASES


def test_v3_listing_and_pattern():
    url, pat = build_location_info(2, "20200217", "eng-us")
    assert url == "https://books.storage.googleapis.com/?prefix=ngrams/books/20200217/eng-us/2-"
    assert pat.match("2-00012-of-00024.gz")
    assert not pat.match("3-00012-of-00024.gz")
    assert not pat.match("googlebooks-eng-all-2gram-20120701-qu.gz")


@pytest.mark.parametrize("release", ["20120701", "20090715", "20250101"])
def test_pre_2020_releases_are_rejected(release):
    with pytest.raises(ValueError, match="Unsupported release"):
        build_location_info(1, release, "eng")


@pytest.mark.parametrize("bad", [dict(ngram_size=0), dict(ngram_size=6),
                                 dict(repo_release_id="2020-02-17"), dict(repo_corpus_id="eng us")])
def test_invalid_parameters_are_rejected(bad):
    kwargs = dict(ngram_size=1, repo_release_id="20200217", repo_corpus_id="eng")
    kwargs.update(bad)
    with pytest.raises(ValueError):
        build_location_info(**kwargs)


def test_supported_releases_constant():
    assert SUPPORTED_RELEASES == ("20200217",)

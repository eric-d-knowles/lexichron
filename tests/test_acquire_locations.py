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


def test_bucket_listing_gives_urls_and_sizes(monkeypatch):
    """A bucket listing states each file's size; the run weighs its progress
    by them. Pages are followed, and files that do not match are left out."""
    import re
    from ngramprep.ngram_acquire.io import fetch

    pages = {
        None: """<?xml version='1.0'?><ListBucketResult><IsTruncated>true</IsTruncated>
                 <NextMarker>ngrams/books/20200217/eng/2-00001-of-00003.gz</NextMarker>
                 <Contents><Key>ngrams/books/20200217/eng/2-00000-of-00003.gz</Key><Size>1500</Size></Contents>
                 <Contents><Key>ngrams/books/20200217/eng/2-00001-of-00003.gz</Key><Size>250</Size></Contents>
                 </ListBucketResult>""",
        "ngrams/books/20200217/eng/2-00001-of-00003.gz":
            """<ListBucketResult><IsTruncated>false</IsTruncated>
               <Contents><Key>ngrams/books/20200217/eng/2-00002-of-00003.gz</Key><Size>oops</Size></Contents>
               <Contents><Key>ngrams/books/20200217/eng/totalcounts-2</Key><Size>99</Size></Contents>
               </ListBucketResult>""",
    }

    class Resp:
        encoding = "utf-8"
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass

    class Session:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, timeout=None):
            from urllib.parse import parse_qs, urlparse
            marker = parse_qs(urlparse(url).query).get("marker", [None])[0]
            return Resp(pages[marker])

    monkeypatch.setattr(fetch.requests, "Session", Session)
    listing = fetch.fetch_file_urls("https://books.storage.googleapis.com/?prefix=ngrams/books/20200217/eng/2-",
                                    re.compile(r"^2-\d{5}-of-\d{5}\.gz$"))
    assert [u.rsplit("/", 1)[-1] for u in listing] == ["2-00000-of-00003.gz", "2-00001-of-00003.gz", "2-00002-of-00003.gz"]
    assert isinstance(listing, list)
    assert listing.sizes == {"2-00000-of-00003.gz": 1500, "2-00001-of-00003.gz": 250}     # an unreadable size is left out

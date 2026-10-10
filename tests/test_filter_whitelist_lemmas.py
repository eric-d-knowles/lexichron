"""Whitelist mode must look tokens up by lemma when a lemmatizer is given
(option 1): the unigram whitelist contains lemmas, so inflected forms in
multigrams have to be lemmatized before the membership test."""
import pytest

core = pytest.importorskip("ngramprep.ngram_filter.filters.core_cy")

from ngramprep.ngram_filter.config import FilterConfig  # noqa: E402
from ngramprep.ngram_filter.filters.processor_factory import build_processor  # noqa: E402


class FakeLemmatizer:
    table = {"women": "woman", "running": "run", "children": "child", "fathers": "father"}

    def lemmatize(self, word, pos="NOUN"):
        return self.table.get(word, word)


WHITELIST = {b"woman", b"run", b"child", b"the", b"of"}


def test_whitelist_lookup_uses_lemmas_when_lemmatizer_given():
    proc = build_processor(FilterConfig(whitelist=WHITELIST, lemma_gen=FakeLemmatizer(),
                                        min_context_tokens=1))
    assert proc(b"the women of the") == b"the woman of the"
    assert proc(b"The_DET children_NOUN running_VERB") == b"the child run"
    # a lemma not in the whitelist is still <UNK>
    assert proc(b"the fathers of") == b"the <UNK> of"


def test_whitelist_lookup_without_lemmatizer_matches_surface_forms():
    proc = build_processor(FilterConfig(whitelist=WHITELIST, lemma_gen=None, min_context_tokens=1))
    assert proc(b"the women of the") == b"the <UNK> of the"      # the mismatch option 1 fixes


def test_always_include_survives_in_whitelist_mode():
    proc = build_processor(FilterConfig(whitelist=WHITELIST, lemma_gen=FakeLemmatizer(),
                                        always_include={b"he", b"working-class"}, min_context_tokens=1))
    assert proc(b"he is working-class") == b"he <UNK> working-class"

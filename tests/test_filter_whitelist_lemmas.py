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


class CaseChangingLemmatizer:
    """spaCy's English lookup table has four entries whose lemma is not
    lowercase: truer/truest -> TRUE, falser/falsest -> FALSE."""
    table = {"truer": "TRUE", "truest": "TRUE", "falser": "FALSE", "falsest": "FALSE"}

    def lemmatize(self, word, pos="NOUN"):
        return self.table.get(word, word)


def test_lemma_is_lowercased_too_when_lowercasing():
    # rule filters (the unigram path): the lemma comes out as one lowercase type
    proc = build_processor(FilterConfig(lemma_gen=CaseChangingLemmatizer(), lowercase=True, filter_stops=False, filter_short=False,
                                        min_context_tokens=1))
    assert proc(b"truer_ADJ") == b"true" and proc(b"Truest_ADJ") == b"true" and proc(b"falsest") == b"false"
    # whitelist path: the lowercased lemma is what gets looked up
    proc = build_processor(FilterConfig(whitelist={b"true", b"story"}, lemma_gen=CaseChangingLemmatizer(),
                                        lowercase=True, filter_stops=False, filter_short=False, min_context_tokens=1))
    assert proc(b"truer story") == b"true story"
    # without lowercasing the lemma is taken as the table gives it
    proc = build_processor(FilterConfig(lemma_gen=CaseChangingLemmatizer(), lowercase=False, filter_stops=False, filter_short=False,
                                        min_context_tokens=1))
    assert proc(b"truer") == b"TRUE"


def test_whitelist_lookup_without_lemmatizer_matches_surface_forms():
    proc = build_processor(FilterConfig(whitelist=WHITELIST, lemma_gen=None, min_context_tokens=1))
    assert proc(b"the women of the") == b"the <UNK> of the"      # the mismatch option 1 fixes


def test_always_keep_matches_the_spelling_in_the_data_and_skips_every_rule():
    """An always-keep entry is a spelling: it protects the token whose
    surface form matches, case-folded by default, exactly from the case
    list, and the protected token is then normalized like the rest."""
    proc = build_processor(FilterConfig(whitelist=WHITELIST, lemma_gen=FakeLemmatizer(), lowercase=True,
                                        always_include={b"he", b"working-class", b"women"}, min_context_tokens=1))
    assert proc(b"he is working-class") == b"he <UNK> working-class"
    assert proc(b"He_PRON IS Working-Class_NOUN") == b"he <UNK> working-class"    # any case; then lowercased
    assert proc(b"women fathers") == b"woman <UNK>"               # kept by spelling, then lemmatized; not via lemma
    assert proc(b"woman") == b"woman"                             # in the whitelist anyway
    # the normal path: the rules that would drop it are skipped too
    proc = build_processor(FilterConfig(lowercase=True, alpha_only=True, filter_short=True, min_len=3,
                                        filter_stops=True, stop_set={"the", "is"}, lemma_gen=None,
                                        always_include={"A.I.", "the", "he"}, min_context_tokens=1))
    assert proc(b"The A.I. is here") == b"the a.i. <UNK> here"    # A.I. passes alpha; 'the' passes stopwords; 'is' does not
    assert proc(b"he saw a.i.") == b"he saw a.i."                 # 'he' passes the length rule; a.i. matches case-folded
    # the case list protects one spelling only
    proc = build_processor(FilterConfig(lowercase=True, alpha_only=True, filter_short=True, min_len=3,
                                        always_include_case={"AI"}, min_context_tokens=1))
    assert proc(b"AI ai Ai") == b"ai <UNK> <UNK>"                 # AI kept (then lowercased); ai and Ai fall to min_len
    proc = build_processor(FilterConfig(lowercase=False, alpha_only=True, filter_short=True, min_len=3,
                                        always_include_case={"AI"}, always_include={"he"}, min_context_tokens=1))
    assert proc(b"AI He") == b"AI He"                             # no lowercasing: kept as they were

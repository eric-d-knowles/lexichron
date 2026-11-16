"""The lemmatizer: a token's part of speech is used where spaCy's tables can
vouch for the result, and the lookup table decides everywhere else.

The rule is tested on small hand-made tables (no model needed), then against
the English model's own tables where it is installed."""
import pytest

from ngramprep.ngram_filter.lemmatizer import CachedSpacyLemmatizer, SpacyLemmatizer


def _made(lookup, tables, use_tags=True):
    """A lemmatizer over hand-made tables, without loading a model."""
    lem = object.__new__(SpacyLemmatizer)
    lem._lookup_table = lookup
    lem.use_tags = use_tags
    lem._tag_tables = tables if use_tags else {}
    return lem


LOOKUP = {"building": "build", "buildings": "building", "leaves": "leave", "left": "leave", "saw": "see",
          "better": "well", "years": "year", "glasses": "glass", "women": "woman", "pass": "pass",
          "truer": "TRUE"}
TABLES = {
    # (irregular forms, base forms, suffix rules)
    "noun": ({"leaves": "leaf", "women": "woman"},
             {"building", "leaf", "leave", "saw", "year", "years", "glass", "glasses", "woman", "business", "pass", "pas"},
             [("s", ""), ("ses", "s"), ("es", "")]),
    "verb": ({"saw": "see", "left": "leave"}, {"build", "leave", "see", "saw", "pass"}, [("s", ""), ("ing", "")]),
    "adj": ({"better": "good"}, {"left", "good", "well"}, [("er", "")]),
}


def test_an_irregular_form_listed_for_the_part_of_speech_wins():
    lem = _made(LOOKUP, TABLES)
    assert lem.lemmatize("leaves", "NOUN") == "leaf"          # the lookup says leave
    assert lem.lemmatize("saw", "VERB") == "see"              # although "to saw" is a verb too
    assert lem.lemmatize("left", "VERB") == "leave"
    assert lem.lemmatize("better", "ADJ") == "good"           # the lookup says well


def test_the_lookup_stands_when_it_is_a_lemma_this_part_of_speech_can_have():
    lem = _made(LOOKUP, TABLES)
    # "years" and "glasses" are listed as base forms in their own right (old age; spectacles),
    # but the plural reading is legitimate and is what the lookup gives: no change
    assert lem.lemmatize("years", "NOUN") == "year"
    assert lem.lemmatize("glasses", "NOUN") == "glass"
    assert lem.lemmatize("buildings", "NOUN") == "building"
    assert lem.lemmatize("leaves", "VERB") == "leave"
    # a rule that happens to yield another word ("pas") does not override a lookup that keeps the word
    assert lem.lemmatize("pass", "NOUN") == "pass"


def test_the_tag_decides_when_the_lookup_is_not_a_lemma_of_that_part_of_speech():
    lem = _made(LOOKUP, TABLES)
    assert lem.lemmatize("building", "NOUN") == "building"    # a noun in its own right; "build" is not a noun here
    assert lem.lemmatize("building", "VERB") == "build"
    assert lem.lemmatize("saw", "NOUN") == "saw"
    assert lem.lemmatize("left", "ADJ") == "left"


def test_nothing_is_stripped_from_a_word_the_tables_do_not_vouch_for():
    """What spaCy's rule lemmatizer does with only a coarse tag: business -> busines."""
    lem = _made(LOOKUP, TABLES)
    assert lem.lemmatize("business", "NOUN") == "business"    # a known base form: left alone
    assert lem.lemmatize("blorfs", "NOUN") == "blorfs"        # unknown: no rule result is a known word
    assert lem.tag_lemmas("blorfs", "NOUN") == (None, [])
    assert lem.lemmatize("zxqing", "VERB") == "zxqing"


def test_without_a_usable_tag_the_lookup_decides():
    lem = _made(LOOKUP, TABLES)
    for pos in (None, "", "PROPN", "DET", "PRON", "X", "ADV"):       # no tables for these here
        assert lem.lemmatize("building", pos) == "build", pos
        assert lem.lemmatize("leaves", pos) == "leave", pos
    # a cased token is not looked up in the (lowercase) tables
    assert lem.lemmatize("Leaves", "NOUN") == "Leaves" and lem.tag_lemmas("Leaves", "NOUN") == (None, [])
    assert lem.lemmatize("truer", "ADJ") == "TRUE"            # the lookup's answer as it is; the kernel lowercases
    # and with tags turned off it is the lookup alone
    off = _made(LOOKUP, TABLES, use_tags=False)
    assert off.lemmatize("building", "NOUN") == "build" and off.lemmatize("leaves", "NOUN") == "leave"


# ---- against the English model ------------------------------------------------

@pytest.fixture(scope="module")
def english():
    spacy = pytest.importorskip("spacy")
    try:
        return CachedSpacyLemmatizer(language="en") if spacy.util.is_package("en_core_web_sm") else pytest.skip("no English model")
    except OSError:
        pytest.skip("no English model")


def test_english_tags_separate_what_the_lookup_merges(english):
    cases = {  # (word, tag): (lookup, with the tag)
        ("building", "NOUN"): ("build", "building"), ("building", "VERB"): ("build", "build"),
        ("meeting", "NOUN"): ("meet", "meeting"), ("meeting", "VERB"): ("meet", "meet"),
        ("leaves", "NOUN"): ("leave", "leaf"), ("leaves", "VERB"): ("leave", "leave"),
        ("left", "ADJ"): ("leave", "left"), ("left", "VERB"): ("leave", "leave"),
        ("saw", "NOUN"): ("see", "saw"), ("saw", "VERB"): ("see", "see"),
        ("better", "ADJ"): ("well", "good"), ("better", "ADV"): ("well", "well"),
    }
    for (word, tag), (looked_up, tagged) in cases.items():
        assert english.lookup(word) == looked_up, word
        assert english.lemmatize(word, tag) == tagged, (word, tag)
        assert english.lemmatize(word) == looked_up, word


def test_english_leaves_ordinary_words_as_the_lookup_has_them(english):
    for word, tag in (("years", "NOUN"), ("days", "NOUN"), ("times", "NOUN"), ("arms", "NOUN"), ("glasses", "NOUN"),
                      ("women", "NOUN"), ("children", "NOUN"), ("was", "VERB"), ("has", "VERB"), ("running", "VERB"),
                      ("pass", "NOUN"), ("boss", "NOUN"), ("gas", "NOUN"), ("news", "NOUN"), ("data", "NOUN")):
        assert english.lemmatize(word, tag) == english.lookup(word), (word, tag)
    # the words spaCy's rule lemmatizer truncates when given only a coarse tag
    for word in ("business", "success", "glass", "analysis", "blorfs"):
        assert english.lemmatize(word, "NOUN") == word
    # tags without tables, and hyphenated tokens the tables do not list
    assert english.lemmatize("parks", "PROPN") == english.lookup("parks")
    assert english.lemmatize("human-beings", "NOUN") == "human-beings"


def test_cached_lemmatizer_survives_pickling_with_its_setting():
    import pickle
    lem = pickle.loads(pickle.dumps(CachedSpacyLemmatizer(language="en", use_tags=False)))
    assert lem._use_tags is False and lem._initialized is False
    assert pickle.loads(pickle.dumps(CachedSpacyLemmatizer(language="en")))._use_tags is True

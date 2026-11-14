"""Tests for the acquisition parser and the combined-multigram merge behavior."""
import struct

import pytest

from ngramprep.ngram_acquire.io.parse import parse_line, prepare_multigrams, _combine_multigrams_in_ngram
from ngramprep.ngram_acquire.worker import merge_packed_records, _pack_record
from ngramprep.ngram_acquire.utils.filters import make_ngram_type_predicate


def unpack(blob: bytes):
    n = len(blob) // 8
    vals = struct.unpack(f"<{n}Q", blob)
    return [tuple(vals[i:i + 3]) for i in range(0, n, 3)]


# --- parse_line -------------------------------------------------------------

def test_parse_line_basic():
    key, rec = parse_line("hello\t2000,100,50\t2001,150,60")
    assert key == "hello"
    assert rec == {"frequencies": [
        {"year": 2000, "frequency": 100, "document_count": 50},
        {"year": 2001, "frequency": 150, "document_count": 60},
    ]}


@pytest.mark.parametrize("line", ["", "   ", "no-tab-here", "x\tbad,tuple", "x\t1,2"])
def test_parse_line_rejects_malformed(line):
    assert parse_line(line) == (None, None)


def test_parse_line_applies_filter_after_combining():
    tagged = make_ngram_type_predicate("tagged")
    key, _ = parse_line(
        "working_NOUN class_NOUN in_ADP\t1990,1,1",
        filter_pred=tagged,
        combined_multigrams={"working class"},
    )
    assert key == "working-class_NOUN in_ADP"


# --- combining --------------------------------------------------------------

def test_combine_untagged():
    assert _combine_multigrams_in_ngram("the working class in", {"working class"}) == "the working-class in"


def test_combine_keeps_first_tag():
    out = _combine_multigrams_in_ngram("working_NOUN class_VERB in_ADP", {"working class"})
    assert out == "working-class_NOUN in_ADP"


def test_tagged_variants_collapse_to_same_key():
    """The second token's tag is dropped, so these two lines share a key."""
    s = {"working class"}
    k1, _ = parse_line("working_NOUN class_NOUN in_ADP\t1990,500,80", combined_multigrams=s)
    k2, _ = parse_line("working_NOUN class_VERB in_ADP\t1990,20,5", combined_multigrams=s)
    assert k1 == k2


# --- merging ----------------------------------------------------------------

def test_merge_packed_records_sums_overlapping_years():
    a = struct.pack("<6Q", 1990, 500, 80, 2000, 600, 90)
    b = struct.pack("<6Q", 1990, 20, 5, 2000, 30, 8)
    assert unpack(merge_packed_records(a, b)) == [(1990, 520, 85), (2000, 630, 98)]


def test_merge_packed_records_unions_disjoint_years_in_order():
    a = struct.pack("<3Q", 2000, 1, 1)
    b = struct.pack("<3Q", 1990, 2, 2)
    assert unpack(merge_packed_records(a, b)) == [(1990, 2, 2), (2000, 1, 1)]


def test_merge_is_commutative():
    a = struct.pack("<6Q", 1990, 500, 80, 2000, 600, 90)
    b = struct.pack("<3Q", 1990, 20, 5)
    assert merge_packed_records(a, b) == merge_packed_records(b, a)


def test_pack_then_merge_matches_hand_sum():
    _, r1 = parse_line("k\t1990,500,80\t2000,600,90")
    _, r2 = parse_line("k\t1990,20,5")
    merged = merge_packed_records(_pack_record(r1), _pack_record(r2))
    assert unpack(merged) == [(1990, 520, 85), (2000, 600, 90)]


# --- case ------------------------------------------------------------------

def test_pairs_match_whatever_their_case_and_keep_it():
    s = prepare_multigrams({"working class"})
    assert _combine_multigrams_in_ngram("The Working Class in", s) == "The Working-Class in"
    assert _combine_multigrams_in_ngram("WORKING_NOUN CLASS_NOUN", s) == "WORKING-CLASS_NOUN"
    assert _combine_multigrams_in_ngram("Working class", s) == "Working-class"
    # the list's own case does not matter either
    assert dict(prepare_multigrams({"Working  Class"})) == {"working class": None}
    k, _ = parse_line("Working_NOUN Class_NOUN in_ADP\t1990,1,1", combined_multigrams=s)
    assert k == "Working-Class_NOUN in_ADP"


def test_exact_case_sequences_match_as_written_beside_folded_ones():
    s = prepare_multigrams({"Working Class"}, case_sensitive=True)             # the older spelling of the call
    assert dict(s) == {} and s.exact == {"Working Class": None}
    assert _combine_multigrams_in_ngram("The Working Class in", s) == "The Working-Class in"
    assert _combine_multigrams_in_ngram("the working class in", s) == "the working class in"
    k, _ = parse_line("working class\t1990,1,1", combined_multigrams=s)
    assert k == "working class"
    # both kinds in one list: exact wins at a length; folded entries still match any case
    s = prepare_multigrams(["human being"], exact=["AI Lab => AI-Lab", "US Army"])
    assert dict(s) == {"human being": None} and s.exact == {"AI Lab": "AI-Lab", "US Army": None}
    assert _combine_multigrams_in_ngram("the AI Lab and the Human Being", s) == "the AI-Lab and the Human-Being"
    assert _combine_multigrams_in_ngram("the ai lab", s) == "the ai lab"
    assert _combine_multigrams_in_ngram("US Army us army", s) == "US-Army us army"
    assert bool(prepare_multigrams([], exact=["a b"])) and prepare_multigrams([], exact=[]) is None


def test_prepare_refuses_a_non_pair_and_combining_leaves_other_spacing_alone():
    with pytest.raises(ValueError, match="two or more words"):
        prepare_multigrams({"working-class"})
    assert prepare_multigrams(None) is None and prepare_multigrams(set()) is None
    s = prepare_multigrams(["working class"])
    assert _combine_multigrams_in_ngram("a  b", s) == "a  b"          # no match: the text stands, double space and all
    assert _combine_multigrams_in_ngram("x", s) == "x"


# --- longer sequences --------------------------------------------------------

def test_sequences_of_any_length_combine_and_the_longest_wins():
    s = prepare_multigrams({"new york", "new york city", "working class", "a b c d"})
    assert s.max_words == 4 and "new york city" in s
    assert _combine_multigrams_in_ngram("in New York City today", s) == "in New-York-City today"
    assert _combine_multigrams_in_ngram("in New York today", s) == "in New-York today"
    assert _combine_multigrams_in_ngram("New York City working class", s) == "New-York-City working-class"
    assert _combine_multigrams_in_ngram("a b c d", s) == "a-b-c-d"
    assert _combine_multigrams_in_ngram("a b c", s) == "a b c"                  # no shorter entry: nothing
    assert _combine_multigrams_in_ngram("York City New York", s) == "York City New-York"
    # tags: the first token's survives, the rest are dropped
    assert _combine_multigrams_in_ngram("in_ADP New_PROPN York_PROPN City_PROPN", s) == "in_ADP New-York-City_PROPN"
    k, _ = parse_line("the_DET New_PROPN York_PROPN City_PROPN is_VERB\t1990,1,1", combined_multigrams=s)
    assert k == "the_DET New-York-City_PROPN is_VERB"
    # a plain set works too (its longest entry is measured)
    assert _combine_multigrams_in_ngram("new york city", {"new york city"}) == "new-york-city"


# --- targets ------------------------------------------------------------------

def test_an_entry_can_name_the_token_it_becomes():
    from ngramprep.ngram_acquire.io.parse import parse_multigram_entry
    assert parse_multigram_entry("a . i . => A.I.") == (["a", ".", "i", "."], "A.I.")
    assert parse_multigram_entry("new york city => New York City") == (["new", "york", "city"], "New-York-City")
    assert parse_multigram_entry("human being") == (["human", "being"], None)
    assert parse_multigram_entry("human being =>") == (["human", "being"], None)        # blank: the default
    assert parse_multigram_entry((["a", "b"], "ab")) == (["a", "b"], "ab")
    s = prepare_multigrams(["a . i . => A.I.", "New York City => New York City", "working class"])
    assert dict(s) == {"a . i .": "A.I.", "new york city": "New-York-City", "working class": None}
    assert s.max_words == 4
    assert _combine_multigrams_in_ngram("A_DET ._. I_PRON ._. is_VERB", s) == "A.I._DET is_VERB"
    assert _combine_multigrams_in_ngram("a . i .", s) == "A.I."                      # the target as given, whatever the case
    assert _combine_multigrams_in_ngram("in new york city", s) == "in New-York-City"
    assert _combine_multigrams_in_ngram("the Working Class", s) == "the Working-Class"   # no target: the data's case
    k, _ = parse_line("the_DET A_DET ._. I_PRON ._.\t1990,1,1", combined_multigrams=s)
    assert k == "the_DET A.I._DET"
    assert prepare_multigrams(s) is s                                                 # already prepared

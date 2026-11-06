"""Tests for the acquisition parser and the combined-bigram merge behavior."""
import struct

import pytest

from ngramprep.ngram_acquire.io.parse import parse_line, prepare_bigrams, _combine_bigrams_in_ngram
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
        combined_bigrams={"working class"},
    )
    assert key == "working-class_NOUN in_ADP"


# --- combining --------------------------------------------------------------

def test_combine_untagged():
    assert _combine_bigrams_in_ngram("the working class in", {"working class"}) == "the working-class in"


def test_combine_keeps_first_tag():
    out = _combine_bigrams_in_ngram("working_NOUN class_VERB in_ADP", {"working class"})
    assert out == "working-class_NOUN in_ADP"


def test_tagged_variants_collapse_to_same_key():
    """The second token's tag is dropped, so these two lines share a key."""
    s = {"working class"}
    k1, _ = parse_line("working_NOUN class_NOUN in_ADP\t1990,500,80", combined_bigrams=s)
    k2, _ = parse_line("working_NOUN class_VERB in_ADP\t1990,20,5", combined_bigrams=s)
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
    s = prepare_bigrams({"working class"})
    assert _combine_bigrams_in_ngram("The Working Class in", s) == "The Working-Class in"
    assert _combine_bigrams_in_ngram("WORKING_NOUN CLASS_NOUN", s) == "WORKING-CLASS_NOUN"
    assert _combine_bigrams_in_ngram("Working class", s) == "Working-class"
    # the list's own case does not matter either
    assert prepare_bigrams({"Working  Class"}) == frozenset({"working class"})
    k, _ = parse_line("Working_NOUN Class_NOUN in_ADP\t1990,1,1", combined_bigrams=s)
    assert k == "Working-Class_NOUN in_ADP"


def test_case_sensitive_matching_is_a_switch():
    s = prepare_bigrams({"Working Class"}, case_sensitive=True)
    assert s == frozenset({"Working Class"})
    assert _combine_bigrams_in_ngram("The Working Class in", s, case_sensitive=True) == "The Working-Class in"
    assert _combine_bigrams_in_ngram("the working class in", s, case_sensitive=True) == "the working class in"
    k, _ = parse_line("working class\t1990,1,1", combined_bigrams=s, bigram_case_sensitive=True)
    assert k == "working class"


def test_prepare_refuses_a_non_pair_and_combining_leaves_other_spacing_alone():
    with pytest.raises(ValueError, match="two words"):
        prepare_bigrams({"working-class"})
    with pytest.raises(ValueError, match="two words"):
        prepare_bigrams({"the working class"})
    assert prepare_bigrams(None) is None and prepare_bigrams(set()) is None
    s = prepare_bigrams(["working class"])
    assert _combine_bigrams_in_ngram("a  b", s) == "a  b"          # no match: the text stands, double space and all
    assert _combine_bigrams_in_ngram("x", s) == "x"

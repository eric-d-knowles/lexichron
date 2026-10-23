"""lexichron.inspection: a few decoded records from any database or vocabulary file."""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_rocks                                      # noqa: E402
from ngramprep.common_db import api as dbapi            # noqa: E402
from lexichron import inspection                        # noqa: E402


@pytest.fixture
def rocks(monkeypatch):
    fake_rocks.reset()
    monkeypatch.setattr(dbapi, "rs", fake_rocks)
    return fake_rocks


def _packed(*triples):
    return b"".join(struct.pack("<QQQ", *t) for t in triples)


def _ngram_db(path):
    db = fake_rocks.open(str(path), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"__processed__/1-00000-of-00014.gz", b"marker")
        wb.put(b"apple", _packed((1990, 5, 2), (2000, 7, 3)))
        wb.put(b"apple pie", _packed((2000, 1, 1)))
        wb.put(b"zebra", _packed((1950, 2, 2)))
    db.compact_all()
    return db


def _pivoted_db(path):
    db = fake_rocks.open(str(path), mode="rw", create_if_missing=True)
    with db.write_batch() as wb:
        wb.put(b"__rangemap__/partition", b"[]")
        for year, ng, count in ((1990, b"cat run", 5), (2011, b"nuclear <UNK>", 9), (2011, b"nuclear power", 4)):
            wb.put(struct.pack(">I", year) + ng, struct.pack("<QQQ", year, count, count))
    return db


def test_ngram_database_is_sniffed_and_decoded(rocks, tmp_path):
    _ngram_db(tmp_path / "1grams_processed.db")
    info = inspection.describe(str(tmp_path / "1grams_processed.db"))
    assert info["kind"] == "rocksdb" and info["layout"] == "ngram" and info["compacted"]
    assert info["records"] == 3 and info["bookkeeping"] == 1        # the marker is not a record
    layout, found = inspection.rows(str(tmp_path / "1grams_processed.db"), "head", n=10)
    assert layout == "ngram"
    assert found == [("apple", "12 in 5 volumes (1990-2000, 2 years)"),
                     ("apple pie", "1 in 1 volumes (2000)"),
                     ("zebra", "2 in 2 volumes (1950)")]           # bookkeeping key skipped
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "prefix", "apple", 10)[1] == found[:2]
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "key", "b", 10)[1] == found[2:]
    # the raw layout shows everything, bookkeeping included
    assert inspection.rows(str(tmp_path / "1grams_processed.db"), "head", n=1, layout="raw")[1][0][0].startswith("__processed__/")


def test_pivoted_database_keys_are_typed_with_the_year(rocks, tmp_path):
    db = str(tmp_path / "5grams_pivoted.db")
    _pivoted_db(db)
    info = inspection.describe(db)
    assert info["layout"] == "pivoted" and not info["compacted"] and info["records"] == 3
    assert "not compacted" in inspection.format_summary(info)
    _, found = inspection.rows(db, "prefix", "[2011] nuclear", 10)
    assert found == [("[2011] nuclear <UNK>", "9 in 9 volumes"), ("[2011] nuclear power", "4 in 4 volumes")]
    assert inspection.rows(db, "key", "[2011]", 1)[1] == [("[2011] nuclear <UNK>", "9 in 9 volumes")]
    with pytest.raises(ValueError):
        inspection.rows(db, "key", "nuclear", 10)             # no year: not a pivoted key
    with pytest.raises(ValueError):
        inspection.rows(db, "key", "[soon] nuclear", 10)


def test_vocabulary_file_by_rank_token_and_prefix(tmp_path):
    wl = tmp_path / "whitelist.txt"
    wl.write_text("the\nof\napple\napricot\nzebra\n")
    info = inspection.describe(str(wl))
    assert info["kind"] == "text" and info["lines"] == 5
    assert inspection.rows(str(wl), "head", n=2) == ("text", [("1", "the"), ("2", "of")])
    assert inspection.rows(str(wl), "prefix", "ap", 10)[1] == [("3", "apple"), ("4", "apricot")]
    assert inspection.rows(str(wl), "key", "apricot", 10)[1] == [("4", "apricot"), ("5", "zebra")]


def test_missing_and_unopenable_paths_are_reported(rocks, tmp_path):
    assert inspection.describe(str(tmp_path / "nope.db"))["kind"] == "missing"
    (tmp_path / "notadb").mkdir()
    info = inspection.describe(str(tmp_path / "notadb"))
    assert info["kind"] == "rocksdb" and "cannot open" in info["error"]


def test_candidates_list_the_corpus_directory(rocks, tmp_path):
    _ngram_db(tmp_path / "1grams_processed.db")
    (tmp_path / "1grams_processed.db" / "CURRENT").write_text("MANIFEST-000001\n")
    (tmp_path / "1grams_processed.db" / "whitelist.txt").write_text("the\n")
    (tmp_path / "scratch.db").mkdir()                                  # no CURRENT: not a database
    assert inspection.candidates(str(tmp_path)) == [
        ("1grams_processed.db", str(tmp_path / "1grams_processed.db")),
        ("1grams_processed.db/whitelist.txt", str(tmp_path / "1grams_processed.db" / "whitelist.txt"))]
    assert inspection.candidates(None) == [] and inspection.candidates(str(tmp_path / "missing")) == []


def test_cli_inspect_prints_records(rocks, tmp_path):
    import contextlib
    import io
    from lexichron.cli import main
    _ngram_db(tmp_path / "1grams_processed.db")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["inspect", str(tmp_path / "1grams_processed.db"), "--prefix", "apple", "--rows", "5"]) == 0
    out = buf.getvalue()
    assert "3 records" in out and "apple pie" in out and "zebra" not in out
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["inspect", str(tmp_path / "none.db")]) == 1


def test_tables_are_read_with_their_columns(tmp_path):
    search = tmp_path / "tuning" / "s1"
    search.mkdir(parents=True)
    (search / "summary.tsv").write_text("cell\tvector_size\tseed_overlap\nvs100_w2\t100\t0.8100\nvs300_w2\t300\t0.7500\n")
    (search / "summary.json").write_text('{"best_cell": "vs100_w2", "best_rule": "most stable", "cells": [1, 2]}')
    info = inspection.describe(str(search / "summary.tsv"))
    assert info["kind"] == "table" and info["lines"] == 2 and info["columns"] == ["cell", "vector_size", "seed_overlap"]
    assert "suggested cell vs100_w2" in inspection.format_summary(info)
    layout, rows = inspection.rows(str(search / "summary.tsv"), "head", n=10)
    assert layout == "table" and rows[0] == ("cell\tvector_size\tseed_overlap", "") and rows[1] == ("vs100_w2", "100\t0.8100")
    assert inspection.rows(str(search / "summary.tsv"), "prefix", "vs300", 10)[1][1:] == [("vs300_w2", "300\t0.7500")]
    assert inspection.candidates(str(tmp_path)) == [("tuning/s1/summary.tsv", str(search / "summary.tsv"))]


def test_search_results_are_read_formatted_and_printed(tmp_path, monkeypatch):
    import json
    search = tmp_path / "tuning" / "s2"
    search.mkdir(parents=True)
    (search / "search.json").write_text(json.dumps({"grid": {"vector_size": [50, 100]}, "sample_years": [1990, 1991],
                                                     "seeds": 2, "split_half": False}))
    cells = [{"cell": "vs50_w2_mc5_e5_sg1_wbfreq", "vector_size": 50, "window": 2, "min_count": 5, "epochs": 5,
              "approach": "skip-gram", "weight_by": "freq", "years": 1, "vocab_size": 1200, "coverage": 0.8,
              "simlex_rho": 0.2, "seed_overlap": 0.5, "distant_overlap": None},
             {"cell": "vs100_w2_mc5_e5_sg1_wbfreq", "vector_size": 100, "window": 2, "min_count": 5, "epochs": 5,
              "approach": "skip-gram", "weight_by": "freq", "years": 1, "vocab_size": 1200, "coverage": 0.8,
              "simlex_rho": 0.25, "seed_overlap": 0.6, "seed_overlap_sd": 0.0123, "change_ratio": 1.8,
              "change_ratio_sd": None, "distant_overlap": None}]
    (search / "summary.json").write_text(json.dumps({"cells": cells, "best_cell": cells[1]["cell"], "best_rule": "rule"}))
    (search / "results.tsv").write_text("cell\tyear\tvocab_size\tcoverage\tseed_overlap\n"
                                        "vs50_w2_mc5_e5_sg1_wbfreq\t1990\t1200\t0.8000\t0.5000\n"
                                        "vs100_w2_mc5_e5_sg1_wbfreq\t1990\t1200\t0.8000\t\n")
    (search / "models").mkdir()
    (search / "models" / "y1990_vs50_w2_mc5_e5_sg1_wbfreq_s0.npz").write_bytes(b"")
    assert inspection.searches(str(tmp_path)) == [("s2", str(search))]
    assert inspection.searches(str(tmp_path / "nowhere")) == []
    s = inspection.read_search(str(search))
    assert s["years_sampled"] == [1990, 1991] and s["years_scored"] == [1990] and s["models_on_disk"] == 1
    assert s["best_cell"] == "vs100_w2_mc5_e5_sg1_wbfreq" and len(s["results"]) == 2 and s["pairs"] == []
    blank = next(r for r in s["results"] if r["cell"].startswith("vs100"))
    assert blank["seed_overlap"] != blank["seed_overlap"]                             # blank -> NaN
    o = inspection.search_overview(s)
    assert o["tiles"][0] == ("Cells", "2", "in the table; from this grid")
    assert o["tiles"][1] == ("Years", "1/2", "complete for this grid")          # both cells have 1990
    assert o["tiles"][2] == ("Models", "1/8", "for this grid; 1 on disk in all")
    assert o["tiles"][3] == ("Suggested cell", "vs100_w2_mc5_e5_sg1_wbfreq", "by the neighbourhoods rule")
    # 1991 scored for one cell but not the other: the year is under way, not complete
    rows2 = s["results"] + [dict(s["results"][0], year=1991)]
    s2 = dict(s, results=rows2, years_scored=[1990, 1991])
    assert inspection.search_overview(s2)["tiles"][1] == ("Years", "1/2", "complete for this grid; 1991 under way")
    s3 = dict(s2, results=rows2 + [dict(s["results"][1], year=1991)])
    assert inspection.search_overview(s3)["tiles"][1] == ("Years", "2/2", "complete for this grid; finished")
    # cells outside the current grid fill the table but do not count towards its progress
    s4 = dict(s3, settings=dict(s3["settings"], grid={"vector_size": [50, 100], "weight_by": ["none"]}))
    o4 = inspection.search_overview(s4)
    assert o4["tiles"][0] == ("Cells", "2", "in the table; 0 from this grid, 2 from earlier grids")
    assert o4["tiles"][1] == ("Years", "0/2", "complete for this grid; nothing scored yet")
    assert o["grid"].startswith("Grid: vector size 50/100  ·  2 seed(s)  ·  years 1990–1991, 2 sampled")
    assert "provisional: 1990 scored so far" in o["grid"]
    lines = inspection.format_search(s)
    assert lines[0].startswith("Search 's2': Cells 2 (in the table; from this grid); Years 1/2 (complete for this grid)")
    assert any(l.startswith("Suggested (neighbourhoods): vs100_w2_mc5_e5_sg1_wbfreq") for l in lines)
    assert set(s["suggestions"]) == {"benchmarks", "neighbourhoods"}        # computed for an older summary.json
    headings, rows, stems = inspection.search_table(s)
    assert headings[:3] == ["", "Dims", "Years"]            # only the axis the grid varies, then the scores
    assert rows[0][0] == "" and rows[1][0] == "★"
    assert rows[0][headings.index("Vocab")] == "1,200" and rows[0][headings.index("Seed")] == "0.500"
    assert "Period" not in headings and stems == [c["cell"] for c in cells]      # never scored: no column
    assert rows[0][headings.index("Change ratio")] == "–"                          # scored for one cell: "–" for the other
    assert rows[1][headings.index("Seed")] == "0.600 ±.012"                   # a score with its spread
    assert rows[1][headings.index("Change ratio")] == "1.800"                 # no spread recorded: plain
    assert inspection.best_in_columns(s, headings) == {"SimLex ρ": "0.250", "Seed": "0.600", "Change ratio": "1.800"}   # coverage ties
    yh, yrows = inspection.cell_years(s, "vs100_w2_mc5_e5_sg1_wbfreq")
    assert yh == ["Year", "Vocab", "Coverage", "Seed"] and yrows == [["1990", "1,200", "0.800", "–"]]
    assert inspection.cell_pairs(s, "vs100_w2_mc5_e5_sg1_wbfreq") == ""
    detail = inspection.cell_detail(s, "vs100_w2_mc5_e5_sg1_wbfreq")
    assert detail[0].endswith(": per year") and "1990: Vocab 1,200" in detail[1] and "Seed –" in detail[1]
    assert inspection.cell_detail(s, "nope") == ["nope: no rows yet"]
    # describe() and the CLI treat the directory as a search
    info = inspection.describe(str(search))
    assert info["kind"] == "search" and "Suggested (neighbourhoods)" in inspection.format_summary(info)
    from lexichron import cli
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["inspect", str(search), "--key", "vs50_w2_mc5_e5_sg1_wbfreq"]) == 0
    out = buf.getvalue()
    assert "Dims" in out and "★  100" in out and "vs50_w2_mc5_e5_sg1_wbfreq: per year" in out


def test_search_table_is_rebuilt_from_the_rows_when_the_summary_lags(tmp_path):
    """A summary.json written by an older stage (or a narrowed grid) lists
    fewer cells than results.tsv has: the rows win."""
    import json
    search = tmp_path / "tuning" / "s3"
    search.mkdir(parents=True)
    (search / "search.json").write_text(json.dumps({"grid": {"vector_size": [8]}, "sample_years": [1990, 1991],
                                                     "seeds": 2, "split_half": False}))
    (search / "summary.json").write_text(json.dumps({"cells": [{"cell": "vs8_w2_mc2_e1_sg1_wbfreq", "vector_size": 8,
                                                                "years": 1, "coverage": 0.9, "simlex_rho": 0.3,
                                                                "seed_overlap": 0.5}],
                                                      "best_cell": "vs8_w2_mc2_e1_sg1_wbfreq", "best_rule": "old"}))
    (search / "results.tsv").write_text("cell\tyear\tvector_size\twindow\tmin_count\tepochs\tapproach\tweight_by\t"
                                        "vocab_size\tcoverage\tsimlex_rho\tseed_overlap\n"
                                        "vs8_w2_mc2_e1_sg1_wbfreq\t1990\t8\t2\t2\t1\tskip-gram\tfreq\t100\t0.9\t0.3\t0.5\n"
                                        "vs16_w2_mc2_e1_sg1_wbnone\t1990\t16\t2\t2\t1\tskip-gram\tnone\t100\t0.9\t0.3\t0.6\n"
                                        "vs16_w2_mc2_e1_sg1_wbnone\t1991\t16\t2\t2\t1\tskip-gram\tnone\t100\t0.9\t0.3\t0.6\n")
    s = inspection.read_search(str(search))
    assert [c["cell"] for c in s["cells"]] == ["vs16_w2_mc2_e1_sg1_wbnone", "vs8_w2_mc2_e1_sg1_wbfreq"]
    assert s["best_cell"] == "vs16_w2_mc2_e1_sg1_wbnone" and "scored on 2 year(s)" in s["best_rule"]
    headings, rows, stems = inspection.search_table(s)
    assert "Weighting" in headings and "Dims" in headings          # the rows differ on them


def test_search_table_views_and_spreads():
    cells = [{"cell": "a", "vector_size": 50, "window": 2, "min_count": 5, "epochs": 5, "approach": "skip-gram",
              "weight_by": "freq", "years": 2, "vocab_size": 1000, "coverage": 0.9, "simlex_rho": 0.3,
              "seed_overlap": 0.5, "seed_overlap_sd": 0.02, "change_ratio": 1.4, "axis_rho": 0.6, "axis_seed": 0.9,
              "axis_change_ratio": 1.1}]
    s = {"settings": {"grid": {"vector_size": [50]}}, "cells": cells, "best_cell": "a"}
    h_all, rows, _ = inspection.search_table(s)
    assert "Seed" in h_all and "Axis ρ" in h_all and "Axis change" in h_all
    assert "Split-half" not in h_all and "Period" not in h_all                 # never scored: left out
    assert rows[0][h_all.index("Seed")] == "0.500 ±.020"
    h, rows, _ = inspection.search_table(s, view="axis", spreads=False)
    assert h == ["", "Years", "Axis ρ"] and rows[0][2:] == ["0.600"]         # seed and change: the "all" view only
    h, rows, _ = inspection.search_table(s, view="neighbourhoods")
    assert h == ["", "Years", "Seed", "Change ratio"] and rows[0][2] == "0.500 ±.020"
    h, _, _ = inspection.search_table(s, view="benchmarks")
    assert h == ["", "Years", "Vocab", "Coverage", "SimLex ρ", "Axis ρ"]
    # a pair line shows the axis agreement when it was scored
    s["pairs"] = [{"cell": "a", "year_a": 1990, "year_b": 1991, "kind": "adjacent", "overlap": 0.7, "cosine": 0.9,
                   "axis": 0.95}]
    assert "axis agreement" in inspection.cell_pairs(s, "a") and "0.700 / 0.900 / 0.950" in inspection.cell_pairs(s, "a")


def test_each_view_stars_its_own_family_pick():
    cells = [{"cell": "a", "vector_size": 100, "years": 2, "coverage": 0.9, "simlex_rho": 0.3, "seed_overlap": 0.5},
             {"cell": "b", "vector_size": 300, "years": 2, "coverage": 0.9, "simlex_rho": 0.4, "seed_overlap": 0.4}]
    s = {"settings": {"grid": {"vector_size": [100, 300]}}, "cells": cells, "best_cell": "a", "best_rule": "overall",
         "suggestions": {"benchmarks": {"cell": "b", "rule": "best benchmarks"},
                         "neighbourhoods": {"cell": "a", "rule": "most stable"}}}
    assert inspection.suggested_for(s, "benchmarks") == ("b", "best benchmarks")
    assert inspection.suggested_for(s, "all") == ("a", "overall")
    assert inspection.suggested_for(s, "axis") == ("a", "overall")           # no axis pick: the overall one
    _, rows, _ = inspection.search_table(s, view="benchmarks")
    assert [r[0] for r in rows] == ["", "★"]
    _, rows, _ = inspection.search_table(s, view="neighbourhoods")
    assert [r[0] for r in rows] == ["★", ""]
    assert inspection.suggestions_line(s) == "benchmarks: b  ·  neighbourhoods: a"

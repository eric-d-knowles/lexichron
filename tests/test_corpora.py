"""The corpora under a directory and their state (lexichron.corpora)."""
from pathlib import Path

from lexichron.corpora import corpus_info, scan_corpora, stage_prerequisites


def _corpus(root: Path, lang: str, n: int, *, db=False, pivoted=False, settings=False, searches=0, ensembles=0):
    d = root / "20200217" / lang / f"{n}gram_files"
    d.mkdir(parents=True, exist_ok=True)
    if db:
        (d / f"{n}grams.db").mkdir(); (d / f"{n}grams.db" / "CURRENT").write_text("x")
    if pivoted:
        (d / f"{n}grams_pivoted.db").mkdir(); (d / f"{n}grams_pivoted.db" / "CURRENT").write_text("x")
    if settings:
        (d / "lexichron.yaml").write_text("corpus: {}\n")
    for i in range(searches):
        (d / "tuning" / f"s{i}").mkdir(parents=True)
    for i in range(ensembles):
        (d / "models" / f"e{i}").mkdir(parents=True); (d / "models" / f"e{i}" / "train.json").write_text("{}")
    return d


def test_scan_finds_corpora_and_their_state(tmp_path):
    root = tmp_path / "corpora"
    a = _corpus(root, "eng-us", 5, db=True, pivoted=True, settings=True, searches=2, ensembles=1)
    b = _corpus(root, "eng", 1, db=True)
    _corpus(root, "fre", 2)                                   # an empty directory is not a corpus
    (root / "notes").mkdir(); (root / "20200217" / "eng" / "junk").mkdir()
    found = scan_corpora(root)
    assert [i.label for i in found] == ["eng · 20200217 · 1-grams", "eng-us · 20200217 · 5-grams"]
    us = found[1]
    assert us.dir == a and us.settings == a / "lexichron.yaml" and us.database and us.pivoted
    assert us.searches == 2 and us.ensembles == 1
    assert us.state == "acquired · filtered · 2 searches · 1 ensemble"
    one = found[0]
    assert one.dir == b and one.settings is None and one.database and not one.pivoted
    assert one.state == "acquired · not filtered"
    assert scan_corpora(tmp_path / "nowhere") == []


def test_prerequisites_follow_the_state(tmp_path):
    root = tmp_path / "corpora"
    fresh = corpus_info(root, "20200217", "ger", 3)
    assert not fresh.dir.exists() and fresh.state == "new: nothing acquired yet"
    pre = stage_prerequisites(fresh)
    assert pre["acquire"] is None and "Acquire first" in pre["filter"] and "Filter first" in pre["tune"] == pre["train"]
    _corpus(root, "ger", 3, db=True)
    pre = stage_prerequisites(corpus_info(root, "20200217", "ger", 3))
    assert pre["filter"] is None and pre["tune"] is not None
    _corpus(root, "ger", 3, pivoted=True)
    assert all(v is None for v in stage_prerequisites(corpus_info(root, "20200217", "ger", 3)).values())
    assert all(v is None for v in stage_prerequisites(None).values())

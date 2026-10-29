"""The corpora under a directory, and what state each is in.

A corpus is a Google Books n-gram set identified by release, language and
n-gram size, kept at ``<root>/<release>/<language>/<N>gram_files/`` with
its database, its pivoted database, its tuning searches, its model
ensembles and its settings file. The UI's corpus bar lists what a root
holds and says how far each has got; the stage buttons follow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["CorpusInfo", "corpus_info", "scan_corpora", "stage_prerequisites", "SETTINGS_FILENAME"]

SETTINGS_FILENAME = "lexichron.yaml"
_RELEASE = re.compile(r"^\d{8}$")
_FILES_DIR = re.compile(r"^(\d)gram_files$")


@dataclass
class CorpusInfo:
    root: Path
    release: str
    language: str
    ngram_size: int
    dir: Path                       # <root>/<release>/<language>/<N>gram_files
    settings: Optional[Path]        # its lexichron.yaml, when there is one
    database: bool                  # <N>grams.db exists: acquired
    pivoted: bool                   # <N>grams_pivoted.db exists: filtered and pivoted
    searches: int                   # tuning/<name> directories
    ensembles: int                  # models/<name> directories with a train.json

    @property
    def label(self) -> str:
        return f"{self.language} · {self.release} · {self.ngram_size}-grams"

    @property
    def state(self) -> str:
        """One line: how far the corpus has got."""
        if not self.database and not self.pivoted:
            return "new: nothing acquired yet"
        parts = ["acquired" if self.database else "no database"]
        parts.append("filtered" if self.pivoted else "not filtered")
        if self.searches:
            parts.append(f"{self.searches} search{'es' if self.searches != 1 else ''}")
        if self.ensembles:
            parts.append(f"{self.ensembles} ensemble{'s' if self.ensembles != 1 else ''}")
        return " · ".join(parts)


def _is_db(p: Path) -> bool:
    return p.is_dir() and (p / "CURRENT").exists()


def corpus_info(root: str | Path, release: str, language: str, ngram_size: int) -> CorpusInfo:
    """What the directory of one corpus holds (it need not exist yet)."""
    from ngramprep.ngram_acquire.db.build_path import build_db_path
    root = Path(root).expanduser()
    n = int(ngram_size)
    db = Path(build_db_path(str(root), n, str(release), str(language)))
    d = db.parent
    settings = d / SETTINGS_FILENAME
    tuning, models = d / "tuning", d / "models"
    searches = sum(1 for s in tuning.iterdir() if s.is_dir()) if tuning.is_dir() else 0
    ensembles = sum(1 for s in models.iterdir() if s.is_dir() and (s / "train.json").is_file()) \
        if models.is_dir() else 0
    return CorpusInfo(root=root, release=str(release), language=str(language), ngram_size=n, dir=d,
                      settings=settings if settings.is_file() else None,
                      database=_is_db(db), pivoted=_is_db(d / f"{n}grams_pivoted.db"),
                      searches=searches, ensembles=ensembles)


def scan_corpora(root: str | Path) -> List[CorpusInfo]:
    """The corpora under ``root``: every ``<release>/<language>/<N>gram_files``
    directory that holds a database, a settings file, or anything else
    lexichron writes. Sorted by language, release, size."""
    root = Path(root).expanduser()
    out: List[CorpusInfo] = []
    if not root.is_dir():
        return out
    try:
        releases = [p for p in root.iterdir() if p.is_dir() and _RELEASE.match(p.name)]
    except OSError:
        return out
    for rel in releases:
        try:
            langs = [p for p in rel.iterdir() if p.is_dir()]
        except OSError:
            continue
        for lang in langs:
            try:
                dirs = [p for p in lang.iterdir() if p.is_dir() and _FILES_DIR.match(p.name)]
            except OSError:
                continue
            for d in dirs:
                n = int(_FILES_DIR.match(d.name).group(1))
                info = corpus_info(root, rel.name, lang.name, n)
                if info.settings or info.database or info.pivoted or info.searches or info.ensembles:
                    out.append(info)
    out.sort(key=lambda i: (i.language, i.release, i.ngram_size))
    return out


def stage_prerequisites(info: Optional[CorpusInfo]) -> Dict[str, Optional[str]]:
    """For each stage, why it cannot run on this corpus yet (None: it can).
    Acquire creates the corpus, so it always can."""
    if info is None:
        return {"acquire": None, "filter": None, "tune": None, "train": None, "measure": None}
    out: Dict[str, Optional[str]] = {"acquire": None}
    out["filter"] = None if info.database else f"needs the database {info.ngram_size}grams.db: run Acquire first"
    pivot = None if info.pivoted else f"needs the pivoted database {info.ngram_size}grams_pivoted.db: run Filter first"
    out["tune"] = pivot
    out["train"] = pivot
    out["measure"] = None if info.ensembles else "needs a model ensemble under models/: run Train first"
    return out

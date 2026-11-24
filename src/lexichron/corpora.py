"""The corpora under a directory, and what state each is in.

A corpus is a Google Books n-gram set identified by release, name and
n-gram size, kept at ``<root>/<release>/<name>/<N>gram_files/`` with
its database, its pivoted database, its tuning searches, its model
ensembles and its settings file. The UI's corpus bar lists what a root
holds and says how far each has got; the stage buttons follow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["CorpusInfo", "CorpusLayout", "corpus_info", "corpus_layout", "training_layout", "scan_corpora",
           "stage_prerequisites",
           "SETTINGS_FILENAME", "SOURCES", "UNITS"]

SETTINGS_FILENAME = "lexichron.yaml"
_RELEASE = re.compile(r"^\d{8}$")
_FILES_DIR = re.compile(r"^(\d)gram_files$")


@dataclass(frozen=True)
class CorpusLayout:
    """Where a corpus keeps what the stages read and write. Every stage,
    the safeguards and the UI get their paths here, so a corpus of another
    kind needs one more function that returns this, and nothing else has to
    know how its directory is named."""
    dir: Path                       # the corpus directory: databases, tuning/, models/, lexichron.yaml
    raw_db: Path                    # what acquire writes
    processed_db: Path              # filter's output keyed like the raw database (the unigram vocabulary's source)
    training_db: Path               # filter's year-keyed output, which tune and train read
    vocabulary: Path                # the vocabulary this corpus's filter writes
    shared_vocabulary: Path         # the vocabulary its n-grams are filtered against by default (the unigrams')
    tmp_dir: Path                   # the filter's working directory

    @property
    def tuning(self) -> Path:
        return self.dir / "tuning"

    @property
    def models(self) -> Path:
        return self.dir / "models"


SOURCES = ("ngrams", "davies")
UNITS = ("sentences", "tokens")


def corpus_layout(db_path_stub: str | Path, ngram_size: Optional[int], repo_release_id: Optional[str],
                  repo_corpus_id: str, source: str = "ngrams", unit: str = "sentences") -> CorpusLayout:
    """Where a corpus keeps its files. Environment variables and ``~`` in
    the stub are expanded; one that is not set raises ``ValueError``.

    A Google Books n-gram corpus (``source="ngrams"``) is
    ``<stub>/<release>/<name>/<N>gram_files/``. A Davies corpus
    (``source="davies"``) has no release or n-gram size; it is read twice,
    as Google's files come twice: ``unit="tokens"`` is its single tokens
    (``<stub>/<name>/token_files/``, the counterpart of the 1-grams, from
    which the vocabulary is built) and ``unit="sentences"`` its running
    text (``<stub>/<name>/sentence_files/``, the counterpart of the
    5-grams, which is filtered against that vocabulary and trained on)."""
    if source not in SOURCES:
        raise ValueError(f"corpus source must be one of {SOURCES}, got {source!r}")
    if source == "davies":
        from ngramprep.utilities.paths import expand_path
        if unit not in UNITS:
            raise ValueError(f"corpus unit must be one of {UNITS}, got {unit!r}")
        root = Path(expand_path(str(db_path_stub))) / str(repo_corpus_id).lower()
        d = root / f"{unit[:-1]}_files"
        return CorpusLayout(dir=d, raw_db=d / f"{unit}.db", processed_db=d / f"{unit}_processed.db",
                            training_db=d / f"{unit}_pivoted.db",
                            vocabulary=d / f"{unit}_processed.db" / "whitelist.txt",
                            shared_vocabulary=root / "token_files" / "tokens_processed.db" / "whitelist.txt",
                            tmp_dir=d / "processing_tmp")
    from ngramprep.ngram_filter.pipeline.orchestrator import derive_paths
    n = int(ngram_size)
    paths = derive_paths(str(db_path_stub), n, str(repo_release_id), str(repo_corpus_id))
    unigram = paths if n == 1 else derive_paths(str(db_path_stub), 1, str(repo_release_id), str(repo_corpus_id))
    raw = Path(paths["src_db"])
    return CorpusLayout(dir=raw.parent, raw_db=raw, processed_db=Path(paths["dst_db"]),
                        training_db=Path(paths["pivoted_db"]), vocabulary=Path(paths["whitelist"]),
                        shared_vocabulary=Path(unigram["whitelist"]), tmp_dir=Path(paths["tmp_dir"]))


def training_layout(db_path_stub: str | Path, ngram_size: Optional[int], repo_release_id: Optional[str],
                    repo_corpus_id: str, source: str = "ngrams", unit: str = "sentences") -> CorpusLayout:
    """The layout of the corpus a stage trains or measures models on (tune,
    train, measure). A Davies corpus is trained on its sentences; its tokens
    exist to give the vocabulary, and asking for them here is refused."""
    if source == "davies" and unit != "sentences":
        raise ValueError(f"models are trained on a Davies corpus's sentences, and corpus.unit is {unit!r}: "
                         f"use the settings file in the corpus's sentence_files directory (corpus.unit: sentences)")
    return corpus_layout(db_path_stub, ngram_size, repo_release_id, repo_corpus_id, source=source, unit=unit)


@dataclass
class CorpusInfo:
    root: Path
    release: str
    name: str                       # the corpus's identifier: eng, eng-us, eng-fiction, ...
    ngram_size: int
    dir: Path                       # <root>/<release>/<name>/<N>gram_files
    settings: Optional[Path]        # its lexichron.yaml, when there is one
    database: bool                  # <N>grams.db exists: acquired
    pivoted: bool                   # <N>grams_pivoted.db exists: filtered and pivoted
    searches: int                   # tuning/<name> directories
    ensembles: int                  # models/<name> directories with a train.json

    @property
    def label(self) -> str:
        return f"{self.name} · {self.release} · {self.ngram_size}-grams"

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


def corpus_info(root: str | Path, release: str, name: str, ngram_size: int) -> CorpusInfo:
    """What the directory of one corpus holds (it need not exist yet)."""
    from ngramprep.utilities.paths import expand_path
    root = Path(expand_path(root))
    n = int(ngram_size)
    layout = corpus_layout(root, n, release, name)
    db, d = layout.raw_db, layout.dir
    settings = d / SETTINGS_FILENAME
    tuning, models = layout.tuning, layout.models
    searches = sum(1 for s in tuning.iterdir() if s.is_dir()) if tuning.is_dir() else 0
    ensembles = sum(1 for s in models.iterdir() if s.is_dir() and (s / "train.json").is_file()) \
        if models.is_dir() else 0
    return CorpusInfo(root=root, release=str(release), name=str(name), ngram_size=n, dir=d,
                      settings=settings if settings.is_file() else None,
                      database=_is_db(db), pivoted=_is_db(layout.training_db),
                      searches=searches, ensembles=ensembles)


def scan_corpora(root: str | Path) -> List[CorpusInfo]:
    """The corpora under ``root``: every ``<release>/<name>/<N>gram_files``
    directory that holds a database, a settings file, or anything else
    lexichron writes. Sorted by name, release, size. A directory that
    names an environment variable which is not set holds none."""
    from ngramprep.utilities.paths import UnsetVariable, expand_path
    try:
        root = Path(expand_path(root))
    except UnsetVariable:
        return []
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
    out.sort(key=lambda i: (i.name, i.release, i.ngram_size))
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

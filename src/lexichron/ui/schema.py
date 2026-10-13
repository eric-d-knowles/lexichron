"""Describe a stage's settings as form fields, derived from its Python signature.

The form in ``lexichron ui`` is generated from this, so a new keyword argument
on a stage function appears in the UI without any UI code. Settings keep their
Python argument names in the settings file (so the file documents the call and
``--set`` works), but the form shows plain-language labels and help from
``LABELS`` and ``HELP`` below, falling back to the function's docstring.
"""
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lexichron.config import CORPUS_KEY_MAP

__all__ = ["Field", "Section", "stage_sections", "SLURM_FIELDS", "parse_args_doc",
           "LABELS", "HELP", "ADVANCED"]

# Settings that are not meaningful in a form (set by the CLI / runtime).
HIDDEN = {"progress_path", "write_batch_size", "open_type"}   # open_type: write:packed24 is the only sensible value

# Settings most users never touch; shown under a collapsed "Advanced" heading.
ADVANCED = {"archive_path_stub", "spool_dir", "chunk_entries", "random_seed",
            "compact_after_ingest",
            # filter
            "text_language",
            "vocabulary_top_n", "output_vocabulary_spell_check", "progress_path"}

# Known choice lists for arguments whose type is a plain str/int. A list with a
# single entry is shown as fixed text rather than a menu.
CHOICES: Dict[str, List[str]] = {
    "release": ["20200217"],
    "ngram_size": ["1", "2", "3", "4", "5"],
    "ngram_type": ["tagged", "untagged", "all"],
    "open_type": ["write:packed24", "write", "read:packed24", "read"],
    "language": ["eng", "eng-us", "eng-gb", "eng-fiction", "chi-sim", "fre", "ger", "heb", "ita", "rus", "spa"],
    "mode": ["resume", "restart"],
    "text_language": ["en", "fr", "de", "it", "es", "ru", "he", "zh"],
}

# Preferred order within a section (what to get, then how); anything not listed
# follows in signature order.
ORDER = ["release", "language", "ngram_size", "db_path_stub",
         "ngram_type", "combined_bigrams", "overwrite_db", "file_range",
         # filter
         "lowercase", "lemmatize", "drop_stopwords", "alpha_only", "ascii_only", "min_len",
         "min_context_tokens", "bin_size", "vocabulary", "always_include",
         "output_vocabulary", "output_vocabulary_top_n", "output_vocabulary_year_range", "mode",
         "workers"]

# Plain-language labels for the form (settings-file keys stay as they are).
LABELS: Dict[str, str] = {
    "release": "Release",
    "language": "Language",
    "db_path_stub": "Corpus directory",
    "archive_path_stub": "Archive directory",
    "ngram_size": "N-gram size",
    "ngram_type": "Token type",
    "file_range": "File range",
    "workers": "Workers",
    "overwrite_db": "Start fresh",
    "combined_bigrams": "Combine bigrams",
    "random_seed": "Shuffle seed",
    "open_type": "Database profile",
    "compact_after_ingest": "Compact when done",
    "chunk_entries": "Chunk size",
    "spool_dir": "Spool directory",
    # filter
    "text_language": "Text language",
    "lowercase": "Lowercase",
    "lemmatize": "Lemmatize",
    "drop_stopwords": "Drop stopwords",
    "alpha_only": "Letters only",
    "ascii_only": "ASCII letters only",
    "min_len": "Min token length",
    "min_context_tokens": "Min real tokens",
    "bin_size": "Year bin size",
    "vocabulary": "Vocabulary file",
    "vocabulary_top_n": "Vocabulary top N",
    "always_include": "Always keep",
    "output_vocabulary": "Write vocabulary",
    "output_vocabulary_top_n": "Vocabulary size",
    "output_vocabulary_year_range": "Vocabulary years",
    "output_vocabulary_spell_check": "Spell-check vocabulary",
    "mode": "Mode",
}

# User-facing help, where the docstring is too terse or too technical.
HELP: Dict[str, str] = {
    "release": "Google Books n-gram release. Only the February 2020 release is supported.",
    "language": "Which Google Books corpus (eng = all English, eng-us, eng-gb, eng-fiction, ...).",
    "db_path_stub": ("Where Google Books corpora are kept, e.g. /scratch/<you>/NLP_corpora/Google_Books. "
                     "Each corpus goes in <dir>/<release>/<language>/<n>gram_files/, with its database, "
                     "this settings file and its run logs. Created if missing."),
    "ngram_size": "1 = single words, 2 = word pairs, up to 5. Larger n means far more data.",
    "ngram_type": ("tagged = only tokens with a part-of-speech tag (run_VERB); untagged = only plain "
                   "tokens; all = both."),
    "file_range": ("First and last file to download (0-based, inclusive). Blank = every file. "
                   "'0' to '0' downloads a single file for a quick test."),
    "workers": ("Parallel download/parse workers. Blank = one per CPU, minus one. Slurm jobs use "
                "the job's CPU count automatically."),
    "overwrite_db": "Delete an existing database first. Off = resume it, skipping files already done.",
    "combined_bigrams": ("Word pairs to treat as one hyphenated token, comma-separated: "
                         "working class, middle class  ->  working-class, middle-class."),
    "random_seed": "Seed for shuffling the download order (blank = no shuffle).",
    "open_type": "RocksDB profile. Leave at write:packed24 unless you know why.",
    "compact_after_ingest": "Compact the database when done (recommended).",
    "chunk_entries": "Entries per spooled chunk. Lower on small allocations, raise on large nodes.",
    "spool_dir": "Where workers write chunk files (blank = $TMPDIR; node-local disk is ideal).",
    "archive_path_stub": "If set, compress the finished database into this directory (.tar.zst).",
    # filter
    "text_language": "Language for stopwords and lemmatization (spaCy code). Default: from the corpus.",
    "lowercase": "Lowercase every token.",
    "lemmatize": "Reduce tokens to their lemma (women -> woman). Also applied before the vocabulary lookup.",
    "drop_stopwords": "Replace stopwords with <UNK>.",
    "alpha_only": "Replace tokens that are not purely alphabetic (numbers, punctuation) with <UNK>.",
    "ascii_only": "With 'Letters only', also reject letters outside A-Z. Default: on for English only.",
    "min_len": "Replace tokens shorter than this with <UNK>.",
    "min_context_tokens": "Drop n-grams with fewer than this many non-<UNK> tokens. Blank: 1 for unigrams, 2 for longer n-grams.",
    "bin_size": "Aggregate years into bins: 1 = annual, 5 = five-year bins.",
    "vocabulary": ("Vocabulary file (whitelist.txt): tokens not in it become <UNK> and the other filters are "
                   "skipped. Blank: none for unigrams; for longer n-grams, this corpus's own unigram vocabulary, "
                   "which the unigram filter writes."),
    "vocabulary_top_n": "Use only the first N tokens of the vocabulary file.",
    "always_include": "Comma-separated tokens kept even if not in the vocabulary (e.g. he, she, his, her).",
    "output_vocabulary": "Write the vocabulary (whitelist.txt) from the result. Blank: on for unigrams, off for longer n-grams.",
    "output_vocabulary_top_n": "How many of the most frequent tokens the vocabulary keeps.",
    "output_vocabulary_year_range": "Only tokens present in every year of this range, e.g. 1968 to 2019.",
    "output_vocabulary_spell_check": "Only correctly spelled tokens (enchant dictionary).",
    "mode": "resume = continue from what the destination already has; restart = delete it first.",
    "filter.workers": "Parallel filtering workers. Blank = one per CPU, minus one. Slurm jobs use the job's CPU count automatically.",
}


@dataclass
class Field:
    name: str                 # key in the settings file
    kind: str                 # "text" | "int" | "float" | "bool" | "optbool" | "choice" | "list" | "path" | "range"
    default: Any = None
    required: bool = False
    choices: Optional[List[str]] = None
    help: str = ""
    api_name: Optional[str] = None  # python argument name if different from `name`
    label: Optional[str] = None     # plain-language label (defaults to name)
    advanced: bool = False

    @property
    def title(self) -> str:
        return self.label or self.name


@dataclass
class Section:
    name: str
    fields: List[Field] = field(default_factory=list)


def parse_args_doc(doc: Optional[str]) -> Dict[str, str]:
    """Return {arg_name: help} from a Google-style ``Args:`` docstring section."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    out: Dict[str, str] = {}
    in_args = False
    current = None
    for line in lines:
        if re.match(r"^\s*Args:\s*$", line):
            in_args = True
            continue
        if in_args and re.match(r"^\S", line):      # next section header
            break
        if not in_args:
            continue
        m = re.match(r"^\s{1,8}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if m:
            current = m.group(1)
            out[current] = m.group(2).strip()
        elif current and line.strip():
            out[current] += " " + line.strip()
    return out


def _kind_for(annotation: Any, default: Any, name: str) -> str:
    if name in CHOICES:
        return "choice"
    if name.endswith("_range"):
        return "range"
    if name == "vocabulary":
        return "path"
    if name.endswith(("_stub", "_dir", "_path")):
        return "path"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    optional = False
    if origin is typing.Union and type(None) in args:   # Optional[X]
        optional = True
        inner = [a for a in args if a is not type(None)]
        annotation = inner[0] if len(inner) == 1 else annotation
        origin = typing.get_origin(annotation); args = typing.get_args(annotation)
    text = str(annotation)
    if annotation is bool and optional and default is None:
        return "optbool"                                  # yes / no / leave to the default
    if annotation is bool or isinstance(default, bool):
        return "bool"
    if annotation is int or isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if annotation is float or isinstance(default, float):
        return "float"
    if origin in (tuple, list, set) or any(w in text for w in ("Tuple", "List", "Set", "tuple[", "list[", "set[")) or annotation is set:
        return "list"
    return "text"


def stage_sections(func: Callable[..., Any], stage: str) -> List[Section]:
    """Build the ``corpus`` and ``<stage>`` sections for a stage function."""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    docs = parse_args_doc(func.__doc__)
    api_to_corpus = {v: k for k, v in CORPUS_KEY_MAP.items()}

    corpus = Section("corpus")
    stage_sec = Section(stage)
    for pname, p in sig.parameters.items():
        if pname in HIDDEN:
            continue
        ann = hints.get(pname, p.annotation)
        default = None if p.default is inspect.Parameter.empty else p.default
        required = p.default is inspect.Parameter.empty
        key = api_to_corpus.get(pname, pname)
        f = Field(key, _kind_for(ann, default, key), default, required,
                  CHOICES.get(key), HELP.get(f"{stage}.{key}") or HELP.get(key) or docs.get(pname, ""),
                  api_name=pname if key != pname else None,
                  label=LABELS.get(key), advanced=key in ADVANCED)
        (corpus if pname in api_to_corpus else stage_sec).fields.append(f)
    for sec in (corpus, stage_sec):
        sec.fields.sort(key=lambda f: ORDER.index(f.name) if f.name in ORDER else len(ORDER))
    return [corpus, stage_sec]


# Slurm resources live in the settings file too, under `slurm:`.
SLURM_FIELDS = Section("slurm", [
    Field("account", "text", "", True, help="Slurm account, e.g. torch_pr_600_general", label="Account"),
    Field("partition", "text", "", False, help="Partition (leave empty for the default)", label="Partition"),
    # Acquisition throughput flattens out around 40 workers (the single merging
    # parent becomes the limit); 28G covers 40 workers on 5-grams.
    Field("cpus", "int", 40, True, help="CPUs for the job (also used as the worker count unless set). "
          "Acquisition gains little beyond 40.", label="CPUs"),
    Field("mem", "text", "28G", True, help="Memory, e.g. 28G. Acquisition needs about 0.7G per worker.", label="Memory"),
    Field("time", "text", "12:00:00", True, help="Time limit, HH:MM:SS", label="Time limit"),
    Field("job_name", "text", "lexichron", False, help="Job name shown in squeue", label="Job name"),
])

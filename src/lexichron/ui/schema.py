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
           "LABELS", "HELP", "ADVANCED", "GRID_AXES"]

# Settings that are not meaningful in a form (set by the CLI / runtime).
HIDDEN = {"progress_path", "write_batch_size", "open_type",   # open_type: write:packed24 is the only sensible value
          "train_fn",                                            # tune, train: a test hook
          "array_task", "array_count"}                           # set by the batch script (Submit: Jobs)

# Settings most users never touch; shown under a collapsed "Advanced" heading.
ADVANCED = {"archive_path_stub", "spool_dir", "chunk_entries", "random_seed",
            "compact_after_ingest",
            # filter
            "text_language",
            "vocabulary_top_n", "output_vocabulary_spell_check", "progress_path",
            # tune
            "neighbours", "probes_per_decile", "similarity_dataset", "analogy_dataset", "axis_lexicon",
            "axis_poles", "temp_dir",
            "keep_models",
            # train
            "noise_seed", "corpus_writers"}

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
    "unk_mode": ["strip", "reject", "retain"],
    # train: the hyperparameters take one value each (tune's grid rows take several)
    "approach": ["skip-gram", "cbow"],
    "weight_by": ["freq", "doc_freq", "none"],
    "resampling": ["poisson", "none"],
}

# Hyperparameter axes of the tune stage: shown as rows of toggles.
GRID_AXES = ("vector_size", "window", "min_count", "epochs", "approach", "weight_by")

# Preferred order within a section (what to get, then how); anything not listed
# follows in signature order.
ORDER = ["release", "language", "ngram_size", "db_path_stub",
         "ngram_type", "combined_bigrams", "overwrite_db", "file_range",
         # filter
         "lowercase", "lemmatize", "drop_stopwords", "alpha_only", "ascii_only", "min_len",
         "min_context_tokens", "bin_size", "vocabulary", "no_vocabulary", "always_include",
         "pivot", "output_vocabulary", "output_vocabulary_top_n", "output_vocabulary_year_range", "mode",
         "workers"]
ORDER_BY_STAGE = {
    "tune": ["release", "language", "ngram_size", "db_path_stub",
             "years", "sample_years", "year_pairs", "vector_size", "window", "min_count", "epochs", "approach",
             "weight_by", "seeds", "split_half", "unk_mode", "name", "mode", "workers", "workers_per_model"],
    "train": ["release", "language", "ngram_size", "db_path_stub",
              "years", "vector_size", "window", "min_count", "epochs", "approach", "weight_by",
              "corpus_replicates", "seeds_per_corpus", "resampling", "unk_mode", "name", "mode", "workers",
              "workers_per_model"],
}

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
    "no_vocabulary": "No vocabulary",
    "pivot": "Pivot by year",
    "vocabulary_top_n": "Vocabulary top N",
    "always_include": "Always keep",
    "output_vocabulary": "Write vocabulary",
    "output_vocabulary_top_n": "Vocabulary size",
    "output_vocabulary_year_range": "Vocabulary years",
    "output_vocabulary_spell_check": "Spell-check vocabulary",
    "mode": "Mode",
    # tune
    "years": "Study period",
    "sample_years": "Years to sample",
    "year_pairs": "Year pairs",
    "vector_size": "Vector size",
    "window": "Window",
    "min_count": "Min count",
    "epochs": "Epochs",
    "approach": "Approach",
    "weight_by": "Weighting",
    "seeds": "Seeds per cell",
    "split_half": "Split halves",
    "unk_mode": "<UNK> tokens",
    "name": "Search name",
    "workers_per_model": "Threads per model",
    "neighbours": "Neighbours compared",
    "probes_per_decile": "Probe words per decile",
    "similarity_dataset": "Similarity dataset",
    "analogy_dataset": "Analogy dataset",
    "axis_lexicon": "Valence lexicon",
    "axis_poles": "Axis pole words",
    "temp_dir": "Corpus file directory",
    "keep_models": "Keep models",
    # train
    "corpus_replicates": "Corpus replicates",
    "seeds_per_corpus": "Seeds per corpus",
    "resampling": "Resampling",
    "noise_seed": "Ensemble seed",
    "corpus_writers": "Corpus writers",
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
    "no_vocabulary": "Filter 2-5-grams with the rule filters alone, without a vocabulary. Rarely wanted: the result is far larger.",
    "pivot": "Write the database keyed by year then n-gram (what training reads) instead of by n-gram (what the vocabulary is built from). Blank: off for unigrams, on for longer n-grams.",
    "vocabulary_top_n": "Use only the first N tokens of the vocabulary file.",
    "always_include": "Comma-separated tokens kept even if not in the vocabulary (e.g. he, she, his, her).",
    "output_vocabulary": "Write the vocabulary (whitelist.txt) from the result. Blank: on for unigrams, off for longer n-grams.",
    "output_vocabulary_top_n": "How many of the most frequent tokens the vocabulary keeps.",
    "output_vocabulary_year_range": "Only tokens present in every year of this range, e.g. 1968 to 2019.",
    "output_vocabulary_spell_check": "Only correctly spelled tokens (enchant dictionary).",
    "mode": "resume = continue from what the destination already has; restart = delete it first.",
    "filter.workers": "Parallel filtering workers. Blank = one per CPU, minus one. Slurm jobs use the job's CPU count automatically.",
    "tune.years": "First and last year of the study; the sampled years are spread over it.",
    "tune.sample_years": "The years to train on, if you want to choose them (comma-separated). Blank: adjacent pairs spread over the study period. Adjacent years let year-to-year stability be measured.",
    "tune.year_pairs": "Without chosen years: how many adjacent pairs to spread over the period (4 pairs = 8 years).",
    "tune.vector_size": "Dimensions of each word vector. Light several values to search over them.",
    "tune.window": "Context words on each side of the target.",
    "tune.min_count": "Words seen fewer times than this in a year's corpus are left out of that year's model.",
    "tune.epochs": "Passes over the corpus.",
    "tune.approach": "skip-gram predicts context from the word (slower, better for rarer words); cbow the reverse.",
    "tune.weight_by": "How often an n-gram is repeated in the corpus: log5 of its match count (freq), of its volume count (doc_freq), or once (none).",
    "tune.seeds": "Models per cell and year with different random seeds; 2 or more measures seed stability.",
    "tune.split_half": "Also train on each random half of a year's corpus and score their agreement (split-half reliability).",
    "tune.unk_mode": "strip: drop <UNK> tokens, keep the n-gram if two real tokens remain. reject: drop any n-gram with one. retain: keep <UNK> as a token.",
    "tune.name": "Name of this search; results go to <corpus>/tuning/<name>/. Blank: a timestamp.",
    "tune.mode": "resume keeps models already trained (a grown grid trains only its new cells); restart deletes the search first.",
    "tune.workers": "CPUs to use in total. Blank = all of the job's.",
    "tune.workers_per_model": "Threads per model, at least; CPUs / threads models train at once, across years. When fewer models remain than that, each gets more threads.",
    "tune.neighbours": "Nearest neighbours per probe word compared between two models.",
    "tune.probes_per_decile": "Probe words sampled from each frequency decile of the vocabulary.",
    "tune.similarity_dataset": "A word-pair similarity file (word, word, score per line). Blank: WordSim-353 and SimLex-999.",
    "tune.analogy_dataset": "An analogy questions file (a b c d per line). Blank: Google's questions-words.",
    "tune.axis_lexicon": "Human valence ratings (word, rating per line) for the axis scores. Blank: the VADER lexicon the image carries; none: skip them.",
    "tune.axis_poles": "Two lines of pole words, pleasant then unpleasant, for the axis. Blank: the implicit-association lists.",
    "tune.temp_dir": "Where the years' corpus files are written. Blank: inside the search directory. A job's node-local $TMPDIR is faster: write it as $TMPDIR (expanded in the job).",
    "tune.keep_models": "Keep the trained models after scoring (they are small).",
    "train.years": "First and last year of the study; every year between them with data gets an ensemble.",
    "train.vector_size": "Dimensions of each word vector. The Inspect tab's search table shows what tuning suggested.",
    "train.window": "Context words on each side of the target.",
    "train.min_count": "Words seen fewer times than this in a year's corpus are left out of that year's model.",
    "train.epochs": "Passes over the corpus.",
    "train.approach": "skip-gram predicts context from the word (slower, better for rarer words); cbow the reverse.",
    "train.weight_by": "How often an n-gram is repeated in the corpus: log5 of its match count (freq), of its volume count (doc_freq), or once (none).",
    "train.corpus_replicates": "Resamples of each year's corpus, each trained separately: the spread across them is the text's share of a measurement's error. 20 is a working default; 40 for a tight error estimate; 1 with resampling off.",
    "train.seeds_per_corpus": "Models trained on each resample with different seeds: their spread is the fit's share. 2 separates it from the text's.",
    "train.resampling": "poisson: every n-gram's repeat count is redrawn from a Poisson with that mean, another same-sized sample of books. none: the corpus as it is; only the seeds vary.",
    "train.noise_seed": "The number every replicate's seeds derive from; the same number reproduces the ensemble.",
    "train.unk_mode": "strip: drop <UNK> tokens, keep the n-gram if two real tokens remain. reject: drop any n-gram with one. retain: keep <UNK> as a token.",
    "train.axis_lexicon": "Human valence ratings (word, rating per line) for the noise profile measured across each year's replicates. Blank: the VADER lexicon the image carries; none: skip it.",
    "train.axis_poles": "Two lines of pole words, pleasant then unpleasant, for that axis. Blank: the implicit-association lists.",
    "train.name": "Name of this ensemble; the models go to <corpus>/models/<name>/. Blank: a timestamp.",
    "train.mode": "resume keeps models already trained; restart deletes the ensemble first.",
    "train.workers": "CPUs to use in total. Blank = all of the job's.",
    "train.workers_per_model": "Threads per model, at least; CPUs / threads models train at once, across years. When fewer models remain than that, each gets more threads.",
    "train.corpus_writers": "Processes writing corpus resamples ahead of training (each streams the pivoted database).",
    "train.temp_dir": "Where the corpus resamples are written. Blank: inside the ensemble directory. A job's node-local $TMPDIR is faster: write it as $TMPDIR (expanded in the job).",
}


@dataclass
class Field:
    name: str                 # key in the settings file
    kind: str                 # "text" | "int" | "float" | "bool" | "optbool" | "choice" | "list" | "path" | "range" | "grid"
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


def _kind_for(annotation: Any, default: Any, name: str, stage: str = "") -> str:
    if name in GRID_AXES and stage == "tune":
        return "grid"
    if name in CHOICES:
        return "choice"
    if name.endswith("_range") or name == "years":
        return "range"
    if name.endswith("_dataset"):
        return "path"
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
        kind = _kind_for(ann, default, key, stage)
        choices = CHOICES.get(key)
        if kind == "grid":
            from lexichron.training.grid import AXIS_BY_NAME
            axis = AXIS_BY_NAME[key]
            choices = [str(v) for v in axis.presets]
            default = list(axis.default)
        f = Field(key, kind, default, required,
                  choices, HELP.get(f"{stage}.{key}") or HELP.get(key) or docs.get(pname, ""),
                  api_name=pname if key != pname else None,
                  label=LABELS.get(key), advanced=key in ADVANCED)
        (corpus if pname in api_to_corpus else stage_sec).fields.append(f)
    order = ORDER_BY_STAGE.get(stage, ORDER)
    for sec in (corpus, stage_sec):
        sec.fields.sort(key=lambda f: order.index(f.name) if f.name in order else len(order))
    return [corpus, stage_sec]


# Slurm resources live in the settings file too, under `slurm:`.
SLURM_FIELDS = Section("slurm", [
    Field("account", "text", "", True, help="Slurm account, e.g. torch_pr_600_general", label="Account"),
    Field("partition", "text", "", False, help="Partition (leave empty for the default)", label="Partition"),
    # Acquisition throughput flattens out around 40 workers (the single merging
    # parent becomes the limit); 28G covers 40 workers on 5-grams.
    Field("cpus", "int", 40, True, help="CPUs for the job (also used as the worker count unless set). "
          "Acquisition gains little beyond 40.", label="CPUs"),
    Field("mem", "text", "28G", True, help="Memory, e.g. 28G. Acquisition needs about 0.6G per worker plus 3G; the Submit tab shows the estimate.", label="Memory"),
    Field("time", "text", "12:00:00", True, help="Time limit, HH:MM:SS", label="Time limit"),
    Field("job_name", "text", "lexichron", False, help="Job name shown in squeue", label="Job name"),
    Field("jobs", "int", 1, False, help=("Split the run over this many jobs (a Slurm job array), each taking "
          "every N-th year and writing into the same output. For the tune and train stages; the "
          "others run as one job. Each job gets the CPUs, memory and time above."), label="Jobs"),
])

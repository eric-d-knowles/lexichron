# lexichron

Tools for studying semantic change with word embeddings trained on large
historical corpora. lexichron acquires Google Books Ngrams (1- to 5-grams) or
one of Mark Davies' corpora (COHA so far), filters and normalizes the text
into RocksDB databases, searches word2vec hyperparameters for stable models,
trains each year as an ensemble of replicate models, and measures your words'
time series with error bars from those ensembles.

The pipeline is built for corpora of millions to billions of records and is
meant to run on an HPC cluster under Slurm. It works on smaller machines at
the cost of speed.

## Contents

- [Citation](#citation)
- [The pipeline](#the-pipeline)
- [System requirements](#system-requirements)
- [Installation](#installation)
  - [Container installation](#container-installation)
  - [Development installation (conda)](#development-installation-conda)
- [Command line](#command-line)
  - [The acquire stage](#the-acquire-stage)
  - [The filter stage](#the-filter-stage)
- [Davies corpora](#davies-corpora)
- [Terminal user interface](#terminal-user-interface)
- [Hyperparameter search (`tune`)](#hyperparameter-search-tune)
- [Final models (`train`)](#final-models-train)
- [Your words' time series (`measure`)](#your-words-time-series-measure)
- [Example notebooks](#example-notebooks)
- [Output files](#output-files)
- [Provenance](#provenance)
- [Upgrading from older versions](#upgrading-from-older-versions)
- [Support and maintenance](#support-and-maintenance)

## Citation

If you use lexichron in your research, please cite it:

```bibtex
@software{knowles2026lexichron,
  author = {Knowles, Eric D.},
  title = {lexichron: Scalable tools for temporal linguistic analysis},
  year = {2026},
  url = {https://github.com/eric-d-knowles/lexichron},
  version = {0.14.9}
}
```

Or click "Cite this repository" in the GitHub sidebar for other formats.

## The pipeline

Five stages, each run from one settings file (`lexichron <stage>
lexichron.yaml`), the terminal UI, or Python:

| Stage | What it does |
|---|---|
| `acquire` | Downloads a Google Books release (or reads a Davies corpus you hold) into a RocksDB database of per-year counts. Can join word sequences into single tokens as it reads ("working class" → `working-class`). |
| `filter` | Lowercases, lemmatizes (using the part-of-speech tag where there is one) and filters tokens (stopwords, non-alphabetic, short words, a vocabulary); dropped tokens become `<UNK>`. Run on unigrams it writes a vocabulary; run on 2-5-grams it writes them keyed by year, which is what training reads. |
| `tune` | Trains a grid of word2vec settings on a few sampled years and scores each cell's stability and quality; suggests a cell, and leaves the choice to you. |
| `train` | Trains the chosen cell for every year as an ensemble of replicates (corpus resamples × seeds) and reports each year's noise. |
| `measure` | Computes a scalar for your words in every replicate (a projection onto an axis, a similarity, a frequency) and writes each word's series with standard errors. |

N-gram corpora can be English, Chinese (simplified), French, German, Hebrew,
Italian, Russian or Spanish. A Davies corpus goes through the same stages;
only acquisition differs (see [Davies corpora](#davies-corpora)). Every
stage resumes after an interruption, and acquire and filter check that their
output holds exactly the counts they read.

## System requirements

- A cluster node or workstation with many cores (30+ recommended).
- Memory: acquisition needs about 3 GB plus up to 1 GB per worker; filtering
  about 0.9 GB per worker plus 4 GB; 80+ GB is comfortable for 5-gram work.
- Fast storage (NVMe recommended), including a few GB of node-local temp
  space (`$TMPDIR`).
- Several TB of disk for the largest corpora.

## Installation

- **Container (for users):** a prebuilt Apptainer image with lexichron and
  every dependency. Nothing to compile.
- **Conda (for developers):** an editable install from source, for changing
  the code.

### Container installation

**1. Install.** On a cluster with Apptainer or Singularity:

```bash
curl -fsSL https://raw.githubusercontent.com/eric-d-knowles/lexichron/main/install.sh | bash
```

(Without `| bash`, `curl` only prints the script.) This pulls the current
release image (about 4 GB) into `$SCRATCH/lexichron/` (or `~/.lexichron/`)
and puts two commands in `~/.local/bin`:

- `lexichron-ui`: the terminal interface (see below), the easiest way to
  download and process a corpus.
- `lexichron`: the command line, e.g. `lexichron acquire lexichron.yaml`.

Both run inside the image, find Apptainer or Singularity themselves (loading
its module if needed), and re-pull the image if a scratch purge removes it.
The commands below say `apptainer`; on a Singularity cluster type
`singularity` or `alias apptainer=singularity`.

- A specific version: `LEXICHRON_VERSION=0.14.9 bash install.sh`.
- A broken download is resumed with curl and checked against the
  registry's SHA-256; rerun the installer if it is interrupted.
  `LEXICHRON_FETCH=curl` uses curl from the start, and
  `LEXICHRON_SIF=/path/to/lexichron-X.Y.Z.sif` installs from a copied image
  file. If the login nodes' network is slow, run the installer on a compute
  node.

That is all you need to download and process corpora. Steps 2 and 3 are for
working with lexichron from your own notebooks and scripts.

**2. Create a project environment.** Choose a directory for the project (empty,
or an existing one such as a fresh clone) and run:

```bash
apptainer run --app new-project $SCRATCH/lexichron/images/lexichron-0.14.9.sif /scratch/$USER/projects/gender-semantics
```

This creates `gender-semantics/.venv`, a Python environment that inherits
lexichron and its dependencies from the image, installs the directory's
`requirements.txt` if there is one, and registers a Jupyter kernel. Your home
directory and the usual data roots (`/scratch`, `/vast`, `/gpfs`, `/work`,
`/project`, `/projects`, `/data`) are visible inside the container; add
others with `--bind /path`, and set environment variables for the project
with `--env NAME=VALUE`.

**3. Open a notebook** in JupyterLab, Positron or VS Code and pick
*Python (gender-semantics | lexichron 0.14.9)* from the kernel menu (reload
the window if it is not listed). The repository's
[`notebooks/`](https://github.com/eric-d-knowles/lexichron/tree/main/notebooks)
folder has examples to copy.

#### Adding packages

In a notebook on the project's kernel, `%pip install rpy2 pymc`; from a shell,
`.venv/host-python -m pip install rpy2 pymc`. Packages go into the project's
`.venv` and persist; `.venv/host-python -m pip freeze --local` lists them.
Install on a login node if compute nodes have no internet access.

#### Updating a project to a new release

Install the new release (step 1), then run `new-project` again on the same
directory with the new image:

```bash
apptainer run --app new-project $SCRATCH/lexichron/images/lexichron-<version>.sif /path/to/project
```

It keeps `.venv` and what you installed, keeps the launcher's `--env` and
`--bind` settings (passing one again replaces it; `--reset` drops them all),
points the launchers at the new image, and registers a kernel named for the
new version. Switch the notebook to that kernel and restart it. The project's
kernels from earlier releases are listed; `--remove-old-kernels` removes them
(only kernels that run this project's `.venv`).
Data, settings files and notebooks are untouched, and running jobs keep the
image they started with. If a future image moves to a newer Python than
3.11, the command refuses and says how to recreate the environment.

#### Scripts, batch jobs and Slurm from notebooks

`.venv/host-python` is a launcher on the host that runs the project's Python
inside the image, so it works anywhere a Python path does, including Slurm
scripts:

```bash
/path/to/project/.venv/host-python my_script.py
/path/to/project/.venv/host-python -m jupyter nbconvert --to notebook --execute --output run.ipynb analysis.ipynb
```

The project's kernel runs `.venv/host-kernel`, which also starts a small
helper on the host so that code in the container can reach `sbatch`, `squeue`
and the other Slurm commands (`lexichron.bridge.slurm_backend()`); the helper
stops with the kernel. `new-project` rewrites `host-kernel` on every run
unless you have edited it and deleted its `Generated by lexichron
new-project` line.

#### GPUs and JAX

The image carries [JAX](https://docs.jax.dev) with NVIDIA GPU support,
[NumPyro](https://num.pyro.ai) and
[BlackJAX](https://blackjax-devs.github.io/blackjax/), for statistical models
fitted downstream of lexichron; the stages do not use them. The host needs
only an NVIDIA driver (525 or later). The launchers pass the GPUs into the
container when the job has one and point JAX at the CPU otherwise
(`LEXICHRON_NV=1` or `0` forces it; `JAX_PLATFORMS` chooses yourself):

```bash
srun --gres=gpu:1 --time=00:10:00 /path/to/project/.venv/host-python -c "import jax; print(jax.devices())"
```

JAX takes GPU memory as it needs it (`XLA_PYTHON_CLIENT_PREALLOCATE=false`),
so processes can share a GPU. For double-precision work, data-center GPUs
(A100, H100) are far faster than workstation ones (L40S).

The image is read-only: every output path must be on a host directory such
as `/scratch`. `apptainer run-help lexichron-0.14.9.sif` prints a usage
summary; `container/README.md` describes how the image is built.

### Development installation (conda)

For working on lexichron itself, installed in editable mode. Linux x86_64
only (`rocks-shim` ships a wheel for that platform alone).

**1. Clone the repository.**

```bash
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron
```

**2. Make an environment, or use one you have.** The Cython extensions need
a C++ compiler. A new environment, from `environment.yml` (Python 3.11, the
compiler, Cython, Enchant, and lexichron itself with the test tools):

```bash
conda env create -f environment.yml
conda activate lexichron
```

Or an existing one:

```bash
conda activate your-environment
conda install -c conda-forge cxx-compiler
pip install -e '.[dev]'
```

With the environment active, `which lexichron` should name its command, not
`~/.local/bin/lexichron` (which runs the image).

**3. Install the Enchant dictionaries and NLTK data**, once per environment:

```bash
bash scripts/setup_nlp_resources.sh
```

It installs Enchant if missing, the Hunspell dictionaries for every supported
language, and the NLTK Swadesh list, with activation hooks so pyenchant finds
them. spaCy's models download on first import.

**4. Rebuild gensim's corpus reader without the GIL**, once per environment.
The stock reader re-takes the GIL for every line, which on five-token n-gram
lines caps a model at three or four cores. The image does this at build time:

```bash
pip install --target /tmp/cython-for-gensim "Cython==3.1.3"
PYTHONPATH=/tmp/cython-for-gensim python scripts/patch_gensim.py
rm -rf /tmp/cython-for-gensim
```

`pip install --force-reinstall gensim` undoes it. When you run a stage from
the environment yourself, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
first (the batch scripts and the image do), or OpenBLAS's threads halve
training speed.

**5. Register a Jupyter kernel** if you want one:

```bash
python -m ipykernel install --user --name=lexichron --display-name="Python (lexichron)"
```

`docs/architecture.md` describes how acquire and filter work inside.

## Command line

A settings file holds the stage settings, named exactly as in each stage's
Python function; `examples/lexichron.yaml` is a commented template. One file
describes one corpus: a `corpus` section (release, name such as `eng-us`,
n-gram size, database directory; for a Davies corpus see
[Davies corpora](#davies-corpora)) and a section per stage.

```bash
lexichron acquire lexichron.yaml                       # download and ingest
lexichron filter lexichron.yaml                        # filter the ingested database
lexichron acquire lexichron.yaml --set acquire.workers=20   # override a setting
lexichron filter lexichron.yaml --dry-run              # show the call, don't run it
lexichron tune lexichron.yaml                          # hyperparameter search
lexichron train lexichron.yaml                         # the final models, as replicate ensembles
lexichron measure lexichron.yaml                       # your words' series, with error bars, from an ensemble
lexichron inspect 5grams_pivoted.db --prefix "[2011] nuclear"   # look at a database
lexichron inspect models/eng-us-final                  # an ensemble's years and noise profile
lexichron inspect models/eng-us-final/measures/gender  # a measure's years x words
```

`lexichron inspect` opens a database read-only (safe on any node) and prints
decoded records: the first ones, those from `--key` on, or those with a
`--prefix`. `--series` adds each n-gram's year counts, and `--info` prints the
database's provenance record. The key layout is detected (`--layout`
overrides it): `[2011] nuclear` in a pivoted database, the n-gram itself in
an n-gram-keyed one, `[fic][2000] text` in a Davies corpus. A vocabulary file
is shown by rank.

Misspelled or missing settings are reported before anything runs. When the
settings would delete earlier output (`acquire.overwrite_db`, or `mode:
restart` for filter, a named search or a named ensemble), the command says
what and how much and asks first. A batch job cannot answer, so it needs
`--yes` and otherwise stops without deleting anything; the UI adds `--yes`
once you have confirmed in its dialog. Inside the container, use
`apptainer exec lexichron.sif lexichron acquire lexichron.yaml`, or a
project's `.venv/host-python -m lexichron.cli acquire lexichron.yaml`.

### The acquire stage

`lexichron acquire` downloads one Google Books release and writes
`<n>grams.db`, keyed by n-gram, with each key's per-year counts. Worker
processes download and parse the files in parallel (`file_range` limits
which; `random_seed` shuffles the order so a partial run samples the alphabet
evenly). The database is then compacted, its counts checked, and, if
`archive_path_stub` is set, archived as a `.tar.zst`.

- **Resuming.** Each file's records go into the database together with its
  "processed" marker, so an interrupted run loses only the files in flight,
  and the next run (`overwrite_db: false`) does the rest. A file that fails
  three attempts does not stop the others; the run reports it and exits with
  an error, and a rerun fetches only the missing files.
- **Summed counts.** With `combined_multigrams`, several raw n-grams can
  become one key (`working_NOUN class_NOUN` and `working_NOUN class_VERB`
  both become `working-class_NOUN`), and their counts are added.
- **Checked counts** (`verify_counts`, on by default). Each file's marker
  records the totals its worker kept, and after compaction the database must
  hold exactly their sum, so a file taken in twice or a record lost fails
  the run. Years must arrive in order. A line whose counts cannot be read in
  full is logged and counted, and fails the run unless
  `allow_unparsed_lines: true`. The outcome goes into the database's
  provenance and the UI.
- **Memory.** The log reports the job's memory use after the files and
  around the count check, so a job killed for memory can be explained.

How this works inside: [docs/architecture.md](docs/architecture.md).

### The filter stage

`lexichron filter` turns the acquired database into a normalized one. Each
token is lowercased and lemmatized (*women* → *woman*, with spaCy's tables
for the corpus's language, and for English with the token's tag where there
is one: see *Lemmas and part of speech* below). It is then either checked
against a vocabulary or put through the rule filters: stopwords, tokens that
are not purely alphabetic (ASCII letters only for English), and tokens
shorter than `min_len` become `<UNK>`. N-grams with fewer than
`min_context_tokens` real tokens are dropped, and years can be binned
(`bin_size`). Raw n-grams that become the same processed key have their
counts summed.

The stage runs twice per corpus:

1. **Unigrams**, with no vocabulary, write `1grams_processed.db` and
   `1grams_processed.db/whitelist.txt`: the `output_vocabulary_top_n` most
   frequent lemmas, optionally only those present in every year of
   `output_vocabulary_year_range` (only years the corpus has text for are
   required), and optionally spell-checked (which, with lowercasing, also
   drops proper nouns).
2. **Longer n-grams** are filtered against that vocabulary (a token is kept
   if its lemma is in it) and written as `<n>grams_pivoted.db`, keyed by
   year then n-gram, which is what training reads. (*Pivot by year* off
   writes an n-gram-keyed database instead.)

**Always keep** (`always_include`, and `always_include_case` for spellings
whose capitalization must match) names tokens kept whatever the other rules
say. An entry is matched against the token as it appears in the data,
case-folded unless it is on the case list: `he` keeps *He* and *HE*; `AI` on
the case list keeps *AI* but not *ai*. A kept token is still lowercased and
lemmatized, so list every spelling you want kept (`woman` does not keep
*women*). A unigram run adds the kept words to the vocabulary whatever their
frequency, so listing a word there keeps it in the longer n-grams too. Tokens
made at acquisition (`combined_multigrams`) are kept without being listed and
are never lemmatized: to fold a plural, say so where the token is made
(`human beings => human-being`).

**Lemmas and part of speech** (`lemmatize_by_tag`, on by default; English).
spaCy's lookup table gives one lemma per spelling: *building* becomes *build*
whether it is a noun or a verb. Google's tagged n-grams say which it is, and
the filter uses the tag where spaCy's tables vouch for the answer. For a
noun, verb, adjective or adverb:

1. an irregular form listed for that part of speech wins (*leaves* the noun
   is *leaf*; *better* the adjective is *good*);
2. otherwise the lookup's lemma stands if that part of speech can have it
   (*years* → *year*);
3. otherwise the word keeps its own form: *building* the noun stays
   *building*, *saw* the noun stays *saw*;
4. a word the tables do not know, or no tag: the lookup decides.

Nothing is ever stripped from an unknown word (spaCy's rule lemmatizer, given
only a coarse tag, turns *business* into *busines*). A vocabulary must be
lemmatized the same way as the n-grams filtered against it: each run records
how, and the stage refuses a mismatch, so after changing the setting run the
unigram filter again. Untagged n-grams always get the lookup alone.

- **Resuming.** The work is split into key ranges balanced by data volume;
  each range's output and its marker go in together, so an interrupted run
  resumes from the ranges it finished (`mode: resume`). A resume must be
  given what the first run was given (the vocabulary's content, how lemmas
  are made, the keep lists, the rules) and is refused, naming the
  difference, if anything changed; a rebuilt vocabulary needs `mode:
  restart`. A source whose acquisition never finished compacting is refused.
- **Checked counts** (`verify_counts`, on by default). The run checks that
  it read exactly what the source holds, that the destination holds exactly
  what it wrote, and that what it read equals what it kept plus what it
  dropped. A difference fails the run and no vocabulary is written. The
  outcome goes into the provenance and the Progress tab.
- **Memory.** About 0.5 to 0.9 GB per worker on 5-grams (the Submit tab
  allows 0.9 GB plus 4 GB; 0.4 GB per worker for unigrams). Slurm's
  `MaxRSS` reads higher, because it counts shared pages once per worker.

When spellings merge under one lemma their volume counts are added, so a
filtered record's volume count is an upper bound on the books it appears in.
Occurrence counts are exact.

On the Progress tab, a filter run keeps four counts apart: *scanned* (source
records read), *kept* (source records that passed), *year records* (pivoted
runs: one per n-gram and year), and *in database* (after compaction, fewer,
since records with the same key merge). Kept and in database depend only on
the corpus and the rules.

## Davies corpora

Mark Davies' corpora (english-corpora.org) are licensed, so lexichron never
downloads them: you point it at the files you hold. COHA, the Corpus of
Historical American English, is supported, in the layout of its 2015 files
(twenty archives per edition, one per decade, 1810s to 2000s). The other
corpora differ in how their files hold documents, years and genres and are
not read yet.

A Davies corpus is described by `source: davies` and a `name`. It has no
release and no n-gram size; a settings file that gives them is refused.

```yaml
corpus:
  source: davies
  name: coha
  unit: tokens                 # tokens first, then sentences (the default)
  db_path_stub: /scratch/$USER/NLP_corpora/Davies

acquire:
  source_dir: /scratch/$USER/NLP_corpora/english-corpora/COHA   # holds wlp/ and text/; only read
  ngram_type: tagged           # tagged reads wlp/, untagged reads text/
  # genres: [fic, mag]         # default: all four (fic, mag, news, nf)
```

**Two databases, as with n-grams.** `unit: tokens` makes a database of the
corpus's single tokens, the counterpart of Google's 1-grams: filtering it
builds the vocabulary, with the same settings (top N, year range, spell
check) as the unigram filter. `unit: sentences` makes a database of running
text, which is filtered against that vocabulary, pivoted by year, and is
what `tune` and `train` read. Each has its own directory and settings file:

```
{db_path_stub}/coha/
    token_files/       tokens.db, tokens_processed.db (with whitelist.txt), lexichron.yaml
    sentence_files/    sentences.db, sentences_pivoted.db, tuning/, models/, lexichron.yaml
```

The order is the n-gram order: acquire and filter the tokens, then acquire
and filter the sentences, then tune, train and measure on the sentences. The
sentences can also be filtered before the tokens are, with `no_vocabulary:
true` (the rule filters alone) or a `vocabulary` file of your own.

**What a record is.** A document is cut into units of text whose words
really were next to each other: at sentence-final punctuation (`.`, `?`, `!`
as tokens of their own), at paragraph and headline marks, and wherever the
corpus removed words for copyright (COHA leaves ten `@` for every two
hundred words). None of those tokens is kept. A unit is stored under its
document's year with how often it occurred and in how many documents, the
shape of an n-gram record, so every later stage treats it as a long n-gram.
A stretch with no such break for `max_unit_tokens` (500) tokens is cut
there; in COHA none is that long.

**Tagged or untagged** (`ngram_type`), as for Google's files. `tagged` reads
the word/lemma/part-of-speech edition (`wlp/`) and stores each token with a
coarse tag, `building_NOUN`, mapped from the corpus's CLAWS7 tag (the first
alternative of an ambiguous tag; proper nouns are `NOUN`, as in Google's
files). The filter then makes lemmas by tag exactly as it does for tagged
n-grams, which is what makes a COHA series comparable with a Google Books
one. `untagged` reads the plain-text edition (`text/`) and stores bare
words, which get the lookup lemma. The two editions hold the same text and
are cut into the same units. Use the same `ngram_type` and `genres` for the
tokens and the sentences; a vocabulary lemmatized the other way is refused
at filter time, but nothing checks the genres.

**Genres** are chosen when the corpus is acquired, not later: the database
has no place for them. Another selection is another `db_path_stub`.

**Time.** A document's own year is stored. One year of COHA is little text
(a tuning search on 2004 and 2005 trained usable models, but two halves of a
year's text agreed on almost none of a word's nearest neighbours), so set
the filter's `bin_size` (5 or 10) unless you have a reason not to. Set the
same `bin_size` on the tokens, so the vocabulary's year range asks a word to
occur in every bin rather than every year.

A binned database holds each bin under its first year (1965, 1970, ... for
five-year bins). Tune and train read the bin size from the filter's record
in the database: train makes an ensemble for every bin the study period
touches (1968-2009 gives 1965 ... 2005, the first bin holding 1965-1969),
tune samples bins and scores two bins in a row as adjacent, and chosen
`sample_years` must be bins' first years. A rare word can be present in a
bin and still too thin to measure: check each word's count per bin before
reading its estimates.

**Checks.** Acquisition is the n-gram acquisition's own machinery: one
sorted file per archive ingested with its marker, a mandatory compaction,
and a count check of the finished database against what the archives
contributed. A file that is not laid out as expected fails its archive by
name; a row of the tagged edition that is not word, lemma, tag is counted
and fails the run unless `allow_unparsed_lines` is set.

On the 2015 files with 24 CPUs, each acquisition takes about 80 seconds
(116,614 documents, 472.6 million tokens read; 2,629,014 distinct tokens and
22,473,811 distinct units), filtering the tokens half a minute and filtering
the sentences two minutes.

The notebook-era `daviesprep` modules and the Davies notebooks below are the
older route. They also read COCA and the Movies corpus, without the checks
above; they are kept until those corpora are read the new way.

## Terminal user interface

`lexichron-ui` is a terminal interface for people who would rather not edit
YAML or write Slurm scripts. It works over plain SSH and needs no setup: fill
in where the corpus should go and what to download, and submit. It starts a
small helper on the host beside it so jobs can be submitted and watched from
inside the container. `lexichron-ui /path/to/lexichron.yaml` opens a settings
file; a project made with `new-project` has its own `.venv/lexichron-ui`.
Started straight from the image (`apptainer run --app ui
lexichron-0.14.9.sif`), everything works except Submit and the job table.

Two bars sit above the tabs:

- **Corpus**: the corpus everything acts on. Type the directory that holds
  your corpora and pick one; the line beneath says how far it has got
  (acquired, filtered, searches, ensembles). Choosing a corpus opens its
  settings file. *New corpus* sets up one that does not exist yet and takes
  you to Acquire.
- **Stage**: Acquire, Filter, Tune, Train or Measure. A stage whose input is
  missing is greyed out, with the reason as its tooltip.

The tabs:

- **Settings**: the stage's form, with rarely used settings under
  *Advanced*; hover a `?` or press F1 for help. Problems are reported as you
  type. *Save* writes `lexichron.yaml` beside the database it describes (and
  asks before replacing a file that was not the one opened); *Resolved call*
  shows the exact Python call. Plain `lexichron-ui` reopens the file you
  saved last.
- **Run**: runs the stage here, for small tests or inside an interactive
  allocation.
- **Submit**: account, partition, CPUs, memory, time and, for tune, train
  and measure, the number of jobs. Defaults come from lexichron (per stage,
  and smaller for a Davies corpus), then your account and partition in
  `~/.lexichron/slurm.yaml`, then the corpus file's `slurm:` section; a line
  says where each value came from, and a note says how to size the job.
  *Write batch script* writes `lexichron.<stage>.sbatch`; *Submit* also
  submits it. *Jobs* above 1 makes a job array whose tasks each take every
  N-th year; it needs a named search or ensemble. The tab estimates the
  job's memory and asks before submitting one that looks too small, and it
  asks before anything that would delete earlier output. A script is never
  written without an account.
- **Progress**: recent runs, newest first, whether run here or by Slurm,
  with a progress bar, what is being worked on, problems and what they will
  lead to, meters, and the log's tail. An acquisition's bar measures
  progress by file size, not file count, and its meters keep records *read*,
  *kept* and *held* by the database apart; a filter run shows what each rule
  removed and the count check as it goes. Your Slurm jobs are listed below
  with a *Cancel* button. The UI reads each run's
  `.lexichron/runs/<time>_<stage>/progress.json`, so it can be closed and
  reopened while a job runs.
- **Inspect**: look inside a database or vocabulary file, as `lexichron
  inspect` does (read-only, fine on a login node), and see how it was made.
  With Tune, Train or Measure selected, it shows a search's scores, an
  ensemble's years and noise profile, or a measure's years × words.

## Hyperparameter search (`tune`)

Word2vec is a random fit: two trainings on the same corpus give different
vectors, and in a diachronic design that run-to-run variation is what
masquerades as change over time. `lexichron tune` measures it for a grid of
settings before any final model is trained.

- **The grid.** Each hyperparameter (vector size, window, min count, epochs,
  approach, weighting) takes one or more values; the UI shows them as rows
  of toggles. Every combination is a *cell*.
- **The years.** Not the whole study period: adjacent pairs spread over it
  (four pairs by default: 1968-69, 1985-86, 2001-02, 2018-19 for 1968-2019),
  or years you choose. Adjacent years let year-to-year agreement be measured.
- **The models.** Per cell and year: one model per seed (two by default) on
  the year's corpus, and one on each random half of it. A year's corpus is
  written once per weighting (``<UNK>`` tokens stripped by default; an
  n-gram is repeated log5 of its count times) and deleted afterwards.
- **The scores.** Agreement between two models is the mean overlap of the
  ten nearest neighbours of probe words sampled from every frequency decile,
  within the vocabulary the two share (and a Procrustes-aligned cosine as a
  second view). Three noise contrasts: *seed* (same corpus, another seed),
  *split-half* (the two halves of a year), *adjacent year*; and one change
  contrast, *period* (the first year of each sampled pair against the
  others, six comparisons for four pairs). The *change ratio* is period
  dissimilarity over adjacent-year dissimilarity — the between/within ratio
  of a one-way analysis of the neighbourhood differences — and says how
  far periods differ beyond year-to-year noise without assuming the change
  runs one way over the span (a first-vs-last *distant* pair is kept in
  `pairs.tsv` for reference). Every per-year overlap carries its standard
  error across probe words (`_se` in `results.tsv`), and every summary
  score its spread across years or pairs (`_sd`), shown as `0.590 ±.012` so
  that a difference between cells can be read against it. Quality:
  Spearman correlation on SimLex-999 and WordSim-353, analogy accuracy, and
  coverage of the vocabulary list. The image carries the three datasets
  under `/opt/lexichron/data` (fetched at build from gensim's repository,
  shape-checked, checksums in `SOURCES.md` there); a run records which
  file scored it. Elsewhere, set `similarity_dataset` / `analogy_dataset`
  or let gensim's own copies be found.
- **The axis yardstick.** Much diachronic work measures words by projecting
  them onto an axis between two sets of pole words. How well a model
  supports that is checked with the one axis every space has and every
  rater agrees on, valence: the pleasant/unpleasant lists of the
  implicit-association literature (Greenwald et al. 1998, as in Caliskan
  et al. 2017) define the axis in each model, and a lexicon of human
  valence ratings — the VADER lexicon (Hutto & Gilbert 2014, MIT licence),
  fetched into the image with the other datasets — is projected onto it.
  *Axis ρ* is the Spearman correlation between the projections and the
  ratings (validity); *axis seed*, *split* and *adjacent* the correlation
  of the projections between two models, each on its own axis
  (reliability of that kind of measurement, which is closer to what an
  axis analysis needs than neighbourhood overlap); *axis period* and *axis
  change* the period contrast and change ratio for projections. Any
  `word <tab> rating` file serves as the lexicon (`axis_lexicon`: Warriner
  et al.'s norms, say) and any two-line file as the poles (`axis_poles`),
  for a different axis; nothing in the default assumes a research question.
- **The tables.** `tuning/<name>/results.tsv` (cell x year), `pairs.tsv`
  (year pairs), `summary.tsv` and `summary.json` (per cell, with the
  change ratios). The scores themselves live one per file under
  `scores/` (a cell-year each) and `pairs/` (a year pair each); the tables
  are exports rebuilt from the files, and `lexichron inspect` and the
  Inspect tab read the files, so they are current whoever is still writing. The Inspect tab shows the columns in families
  (benchmarks, neighbourhoods, valence axis, or all) with the ± spreads on
  or off, so the table fits an ordinary window.
- **The suggestions.** Each family stars its own cell by its own rule, and
  says so: *benchmarks* — the highest mean of SimLex, WordSim, analogy and
  axis validity among cells within 0.02 of the best coverage (the best
  general-purpose space); *neighbourhoods* — the most stable by the mean of
  seed, split-half and adjacent-year overlap, within 0.05 of the best
  coverage and SimLex (for analyses built on nearest neighbours); *valence
  axis* — the most reliable axis measurement by the mean of split-half
  agreement (Spearman–Brown corrected to the full corpus) and adjacent-year
  agreement, within 0.02 of the best coverage and 0.01 of the best axis
  validity, and of the cells within one standard error of the best, the
  cheapest to train (for analyses that project words onto axes). Only
  cells scored on the most years are compared. The overall suggested cell
  is the axis family's when axis scores exist, the neighbourhoods' otherwise;
  the choice stays yours, and the columns are there to disagree with.
  With the Tune stage selected, the Inspect tab shows a search's scores:
  one row per cell (sortable by any column, the suggested cell starred),
  how far the search has got, and on selecting a cell its per-year rows
  and year pairs. `lexichron inspect <corpus>/tuning/<name>` prints the
  same table (`--key <cell>` adds that cell's years).
- **While it runs.** A cell is scored as soon as its models for a year are
  on disk, and its year pairs as soon as the later year's model exists, so
  the tables and the Inspect tab show a provisional picture at any point.
  Only one cell's models are in memory at a time: a few hundred MB per
  training process (`workers / workers_per_model` of them) plus the corpus
  writer, whatever the grid's size. The models of all years form one queue,
  costliest first, with corpora written a year ahead (how the queue is
  scheduled: [docs/architecture.md](docs/architecture.md)).
- **Growing a grid.** Models are kept under `tuning/<name>/models/`; in
  resume mode a rerun with more values trains only the new cells, and
  scores already in the tables are kept rather than recomputed.
- **Splitting a search over jobs.** With *Jobs* above 1 on the Submit tab
  (or `--set tune.array_task=<i> --set tune.array_count=<N>` on a job you
  write yourself; a hand-written `#SBATCH --array` script may leave them
  to Slurm's variables), each task trains and scores every N-th sampled
  year, counting from its own, so every task gets an even mix of early
  and late years and they finish together. A year pair is scored by
  whichever task finds both years' models on disk, so the pairs are
  complete when the last task is; a task that finishes first reports its
  best cell as "so far". A task cannot restart a search (`mode: restart`
  is refused under an array): delete the search first, as the UI does.

## Final models (`train`)

`lexichron train` fits the cell you chose (the search's suggestion, copied
into the form by *Use its suggested cell*, or any other) for every year of
the study, as an ensemble per year rather than one model: `corpus_replicates`
resamples of the year's corpus, each fitted with `seeds_per_corpus` seeds.

- **Why replicates.** Anything measured on a word2vec model — a word's
  projection on an axis, say — carries two kinds of noise: the sample of
  text the model saw, and the fit (random initialisation and negative
  sampling). One model per year cannot tell either from change over time.
  A corpus replicate redraws every n-gram's repeat count from a Poisson
  with that count as its mean (`resampling: poisson`), which is another
  same-sized sample of books: with the `none` weighting a count-1 n-gram
  survives with probability 1 − 1/e. Two seeds on the same replicate
  isolate the fit's share. The design is nested (seeds within resamples: a
  seed index is a different fit on every resample), a two-level
  random-effects model whose arithmetic gives the error of any measurement
  made the same way on these models. The per-replicate models are the product and
  nothing is aligned: a scalar computed inside one model is unaffected by
  the arbitrary rotation between fits, so replicates are averaged, and
  their spread used, as they are.
- **What is written.** `models/<name>/train.json` records the cell, the
  years with data, the design and the ensemble seed (every replicate's
  resampling and training seeds derive from it and its coordinates, so the
  ensemble is reproducible from the manifest alone). Each year's models land
  in `y<year>/` as `c<corpus>_s<seed>.npz` (words, vectors and counts,
  readable with numpy), with a `year.json` of the corpus and model facts.
- **The noise profile.** `year.json` also carries a general-purpose
  measurement of the year's noise, with no research question assumed, in
  the three families the search scores. Each measures a scalar inside
  every replicate and decomposes its spread across the nested design into
  *corpus variance* (across resamples), *seed variance* (across seeds
  within a resample), the *standard error of the ensemble mean*
  (√(corpus/C + seed/(C·S))) and, where it applies, the *noise share*, that
  error's square against the spread across words. *Valence axis*: the
  valence lexicon's words projected onto each model's own
  pleasant–unpleasant axis (the `tune` yardstick), per word.
  *Neighbourhoods*: the overlap of probe words' nearest neighbours between
  the two seeds of one resample (the fit alone) and between consecutive
  resamples (text and fit together), within the year's common vocabulary,
  on the same footing as the search's seed and split-half overlap.
  *Benchmarks*: SimLex-999 and WordSim-353 correlations per replicate, as
  an ensemble mean with its standard error. The Inspect tab shows the
  years in a table with a *Columns* selector per family, and above it the
  profile of the years complete so far, averaged with the latest year's
  own figure, which says after the first year rather than the last whether
  the design is tight enough; it follows a running job. `lexichron inspect
  models/<name>` prints the same. The Progress tab shows how many of the
  current year's replicates exist. An ensemble trained before a family
  existed gains it from a run in resume mode, which rewrites the records
  from the models on disk without retraining. The same arithmetic on your
  own axis and words is the measurement error of your series: that is the
  `measure` stage, next.
- **Scale.** The default 20 × 2 is 40 models a year; 40 × 2 for a tight
  error estimate. Fifty years of 80 eight-minute models is about 140
  node-hours, so a run spans jobs: `mode: resume` trains only the models
  not on disk, and *Jobs* on the Submit tab splits the years over a job
  array exactly as for `tune` (round-robin, each task its own years, the
  per-year records written by the task that trained the year). Corpus
  resamples are written ahead of training by `corpus_writers` processes
  and deleted as their models finish; `temp_dir: $TMPDIR` keeps them on
  node-local disk.

## Your words' time series (`measure`)

`lexichron measure` is where a research question meets the models. It takes
one trained ensemble, a list of words, and one thing to measure, computes
that scalar inside every model of every year for every word, and writes each
word's series as the ensemble mean per year with its standard error and the
two parts of its variance. The settings file's `measure:` section names the
ensemble, the measure and its words; the Measure stage's form in the UI is
the same, its *Ensemble* setting a list of the corpus's ensembles, newest
first.

- **What can be measured.** Three kinds, all computed inside one model:
  `projection`, each word's cosine with the axis from the *negative* pole
  words' centroid to the *positive* pole words' (the poles as two lists, or
  a `poles_file` of two lines or of one `positive,negative` pair per line;
  pairs and lists give the same axis, since the mean of the pairwise
  differences is the difference of the means); `similarity`, each word's
  cosine with the centroid of the `target` words (one word: the plain
  cosine); and `frequency`, the word's share of the model's tokens, whose
  error is the text's alone. At least three pole words per side must be in
  a model for its axis to be built; the per-year record says how many were.
  A word may be a group, `surgeon|physician|doctor`: one series whose value
  in each model is the mean of the members' values, so its error across the
  ensemble is exact where averaging the members' series afterwards would
  have to take their errors as independent.
- **Why nothing is aligned.** Each of these is a cosine or a count computed
  in one model's own space, so the arbitrary rotation between fits does not
  touch it, and a year's replicates are averaged, and their spread used, as
  they are. That is the border this stage keeps to: a lexeme's *vector*
  over time, or a construct's centroid tracked across years, compares
  vectors between models and needs every replicate rotated into a common
  frame first (one Procrustes rotation per replicate), a different
  operation that lexichron does not yet provide. Most questions of that
  kind have a within-model form, a word's cosine with a fixed set of
  reference words, say, which this stage measures with no alignment and a
  clean error.
- **The error.** Per word and year: the *mean* over the year's replicates,
  the *standard error* of that mean, √(corpus var / C + seed var / n), and
  the two variances, from the same nested arithmetic as the `train` noise
  profile; and *n*, the replicates that had the word, since a rare word can
  miss a resample's min count (it is then averaged over the replicates that
  have it, and a word found in one resample only gets the error of its
  seeds alone, flagged by its n). Per year, the summary gives the *noise
  share*: the squared standard error against the variance of the means
  across your words, how much of the spread between them is noise.
- **What is written**, under `models/<ensemble>/measures/<name>/` (or
  `out_dir`): `measure.json` (the specification and the ensemble it ran on);
  per year, `y<year>.json` (the per-word statistics, pole or target
  coverage, the summary) and `y<year>.npz` (the raw corpus × seed × word
  values, for your own statistics); `series.tsv`, long, one row per year and
  word (`year, word, mean, se, n, var_seed, var_corpus`), and `summary.tsv`,
  one row per year, both rebuilt from the year files after every year. One
  name holds one specification: a run under an existing name with a
  different kind, word list or poles is refused (give it another name, or
  `mode: restart`). A year whose models are not all trained yet is left
  pending and filled in by a later run in resume mode, so a measure can be
  started while `train` is still running. The Inspect tab shows a measure's
  years × words, with *Show* switching between means, standard errors, both
  and n, and the error summary above it; `lexichron inspect
  models/<ensemble>/measures/<name>` prints the same.
- **From Python.** The stage is a plain function and the files are plain:

  ```python
  from lexichron.stages.measure import measure_series
  out = measure_series(5, "20200217", "eng-us", "/scratch/me/NLP_corpora/Google_Books",
                       ensemble="eng-us-final", name="gender", kind="projection",
                       words=["doctor", "nurse", "engineer"],
                       positive=["he", "him", "man", "men"], negative=["she", "her", "woman", "women"])
  import pandas as pd
  series = pd.read_csv(f"{out}/series.tsv", sep="\t")      # year, word, mean, se, n, ...
  ```

  `lexichron.training.measures` has the kernels themselves (`projection`,
  `similarity`, `frequency`, each `(Vectors, Spec) -> values`) for measuring
  one model, and `lexichron.training.ensemble.word_components` turns a
  (corpus, seed, word) cube of any within-model scalar into the same
  per-word statistics.
- **Scale.** Reading the models is the cost: a year of 40 models of 300
  dimensions is a few gigabytes decompressed, read `workers` models at a
  time (8 by default; each holds a whole model in memory). The default
  Slurm settings ask for 8 CPUs and 16 GB; *Jobs* splits the years over a
  job array as for `tune` and `train`.

## Example notebooks

The `notebooks/` folder has one notebook per workflow; copy the ones you need
into your project and edit the settings at the top.

| Notebook | What it does |
|---|---|
| `unigrams_workflow.ipynb` | Acquire and filter 1-grams and write the vocabulary |
| `multigrams_workflow.ipynb` | Acquire 2–5-grams and filter them against the vocabulary into the year-keyed database |

The other notebooks predate the `tune`, `train` and `measure` stages and use
the notebook-era `train`, `analyze` and `daviesprep` modules: single aligned
models per year rather than replicate ensembles. They are kept for COCA and
the Movies corpus, which the stages do not read yet, and as references:
`training_workflow.ipynb`, `ngram_change_analysis_workflow_eng.ipynb` and
`_rus.ipynb` (Google Books); `davies_acquisition_workflow.ipynb`,
`coha_training_workflow.ipynb`, `movies_training_workflow.ipynb` and
`coha_change_analysis_workflow.ipynb` (Davies).

## Output files

The n-gram stages lay their outputs out under the `db_path_stub` you give
them. It may name environment variables and start with `~`
(`/scratch/$USER/NLP_corpora/Google_Books`): they are expanded where the
stage runs, the settings file keeps them as written, and one file then
serves everyone on a cluster. A variable that is not set is an error, not a
directory named `$USER`.

```
{db_path_stub}/{release}/{name}/{N}gram_files/
    {N}grams.db                 raw n-grams as downloaded (acquire)
    {N}grams.db/lexichron.json  how the database was made (every database
                                a stage writes carries one; see below)
    1grams_processed.db         filtered unigrams, keyed by lemma (filter)
    1grams_processed.db/whitelist.txt   the vocabulary built from them
    {N}grams_pivoted.db         filtered 2-5-grams keyed year-first, one record
                                per n-gram and year: what training reads (filter)
    {N}grams_processed.db       2-5-grams keyed by n-gram, only when
                                "Pivot by year" is turned off
    logs/                       one log per filter run
    tuning/<name>/              a hyperparameter search (tune): models, scores, tables
    models/<name>/              an ensemble of final models (train): train.json,
                                y<year>/c<corpus>_s<seed>.npz, y<year>/year.json
    lexichron.yaml              the settings the UI saved
    .lexichron/runs/            progress records of every run
```

A Davies corpus is laid out as `{db_path_stub}/{name}/token_files/` and
`sentence_files/`, with `tokens*` and `sentences*` where an n-gram corpus
has `{N}grams*` (see [Davies corpora](#davies-corpora)).

Nothing temporary is left behind: workers spool their files under the
job's scratch directory, and each is moved into the database (or deleted)
as it is ingested.

## Provenance

Every database a stage finishes carries a `lexichron.json` in its directory:
which stage made it, with which lexichron and rocks-shim versions, when,
where and under which Slurm job; the resolved settings, every option
explicit; what it was made from (for a filter run, the source database with
a fingerprint of *its* record, the vocabulary file with its length and
checksum, the always-kept tokens); the run's figures (lines read, bytes
parsed, records scanned, kept, year records, inserted, dropped per rule);
and the compacted database's record count and size. The record travels
with the database when it is copied or archived, so the numbers for a
methods section come from the artifact itself: `lexichron inspect
<database> --info` prints it, and the Inspect tab shows it under *How it
was made*. A pivoted database's record thus describes the whole chain back
to Google's files. (Runs made by notebooks calling the stage functions
directly get a record only when they pass `progress_path`; the command
line and the UI always do.)

## Upgrading from older versions

Running `new-project` with the new image brings a project up to date (see
[Updating a project](#updating-a-project-to-a-new-release)). Beyond that:

- **Before 0.14.6**, a project's kernel ran `.venv/host-python` and could not
  submit Slurm jobs; `new-project` now registers one that runs
  `.venv/host-kernel`. Before 0.14.9, `new-project` dropped `--env` and
  `--bind` settings not given again.
- **Before 0.14**, the corpus identifier was `corpus.language`; such files
  and `--set corpus.language=` load as `corpus.name`.
- **Before 0.12**, the filter made lemmas without tags. Run the unigram
  filter again before filtering longer n-grams against its vocabulary; the
  stage refuses a mismatch.
- **Before 0.10**, project launchers had no GPU support.
- **Before 0.9**, a tuning search kept only its tables; its next run or read
  imports them into per-cell score files.

## Support and maintenance

This is research software, provided as-is. Issues and pull requests are welcome, but there is no guarantee of a response or of ongoing maintenance. If you depend on it, consider maintaining your own fork.

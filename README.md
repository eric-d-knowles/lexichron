# lexichron

Tools for studying semantic change over time with word embeddings trained on
large historical corpora. lexichron covers the whole pipeline: acquiring
Google Books Ngrams (1- to 5-grams) or Mark Davies' corpora (COHA, COCA and
others), filtering and normalizing the text, storing it in a queryable RocksDB
database, training per-year `word2vec` models, aligning them, and analyzing how
word meanings move.

The pipelines are built for corpora of millions to billions of records and are
meant to run on an HPC cluster or similar hardware. They can be tuned for
smaller machines at the cost of speed.

## Contents

- [Citation](#citation)
- [Capabilities](#capabilities)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Container installation](#container-installation)
  - [Development installation (conda)](#development-installation-conda)
- [Command line](#command-line)
- [Terminal user interface](#terminal-user-interface)
- [Example notebooks](#example-notebooks)
- [Output files](#output-files)
- [Reading the progress display](#reading-the-progress-display)
- [Support and Maintenance](#support-and-maintenance)

## Citation

If you use lexichron in your research, please cite it:

```bibtex
@software{knowles2026lexichron,
  author = {Knowles, Eric D.},
  title = {lexichron: Scalable tools for temporal linguistic analysis},
  year = {2026},
  url = {https://github.com/eric-d-knowles/lexichron},
  version = {0.6.0}
}
```

Alternatively, you can click "Cite this repository" in the GitHub sidebar for additional citation formats.

## Capabilities

### Data Preparation

- **Data acquisition:** Download n-gram datasets (1- through 5-grams) from the 2020 release of Google Books Ngrams, or read Davies corpora (which must be licensed and downloaded by the user), and ingest them into a RocksDB database.
- **Language support:** N-gram pipelines support English, Chinese (simplified), French, German, Hebrew, Italian, Russian, and Spanish.
- **Filtering:** Any combination of case normalization, stopword removal, short-word removal, non-alphabetic token removal, and lemmatization. Discarded tokens are replaced with `<UNK>`.
- **Whitelists:** Write the top-N most frequent unigrams, optionally spell-checked, and use the list to filter longer n-grams. With case normalization, spell-checking also drops proper nouns ("Jackson", "Einstein"). A year range restricts the list to tokens present in every year of the range.
- **Bigram hyphenation:** Convert chosen bigrams into hyphenated unigrams ("working class" → "working-class") so that multiword concepts survive as single tokens.
- **Token immunity:** Name tokens that are always kept regardless of the filtering rules, such as domain terms, proper nouns, or specific multiword expressions.
- **Pivoting:** Reorganize n-gram data from one record per n-gram to one record per n-gram-year:
  - BEFORE: `n-gram → (year1, count1, volumes1) (year2, count2, volumes2) ... (yearn, countn, volumesn)`
  - AFTER:
    - `[year1] n-gram → (count1, volumes1)`
    - `[year2] n-gram → (count2, volumes2)`
    - `...`
    - `[yearn] n-gram → (countn, volumesn)`
- **Parallel processing** with load balancing, progress reporting, and resumption after interruption.
- **Storage** in RocksDB, a key-value database that stays fast at corpus scale.

### Model Training and Evaluation

- **Word embeddings:** Train `word2vec` models on the processed n-grams with `gensim`. `corpus_file` mode gives multithreaded training and can train several years at once. Hyperparameters:
  - `approach`: use skip-gram or continuous bag-of-words (CBOW) architectures
  - `vector_size`: the number of vector dimensions (features) to extract
  - `window_size`: the width of the context window
  - `min_count`: the minimum frequency of words to include in the model
  - `weight_by`: downweight common ngrams by frequency or document count
- **Evaluation:** Score models on standard similarity and analogy benchmarks, plot the results, and use mixed-model regression to estimate how each hyperparameter affects performance across years.

## Workflow

There are two parallel pipelines, one per data source:

### Google Ngrams Pipeline

1. **`ngram_acquire`**: Fetch raw n-gram files (1-5 grams) from the Google Books repository and store in a RocksDB database for fast querying.
2. **`ngram_filter`**: Apply linguistic transformations (case normalization, lemmatization, stopword removal, a vocabulary) to prepare data. Unigrams yield a vocabulary whitelist; 2-5-grams are written, in the same pass, in the year-first ("pivoted") layout that time-series analysis and model training read.

### Davies Corpora Pipeline

1. **`davies_acquire`**: Ingest Davies corpus files (COHA, COCA, etc.) with genre and year information into RocksDB.
2. **`davies_filter`**: Apply the same filtering and preprocessing transformations as `ngram_filter` for consistency.

### Analysis Tools

**`analyze`**: Track semantic drift, similarity change, and projections onto semantic dimensions across years, using the trained embeddings. Works with models from either pipeline.

### Model Training

**`train/word2vec`**: Train per-year `word2vec` models, evaluate them, align them across years, and analyze hyperparameter effects.

## System Requirements

- HPC cluster or workstation with multiple CPU cores (30+ cores recommended)
- RAM: acquisition needs roughly 3 GB plus a few hundred MB per worker
  (parsed data is streamed through on-disk chunks); filtering and training
  benefit from more, and 80+ GB is comfortable for 5-gram work
- Fast local storage (NVMe SSD recommended), including a few GB of node-local
  temp space (`$TMPDIR`) for the acquisition spool
- Several TB of disk space for processing and storing very large corpora
- Settings can be tuned for smaller machines at the cost of speed

## Installation

There are two ways to install lexichron:

- **Container (recommended for users):** pull a prebuilt Apptainer image that
  contains lexichron and every dependency, then run the notebooks against it.
  Nothing to compile, no conda environment, no setup script.
- **Conda (for developers):** an editable install from source into a conda
  environment, for modifying the code.

### Container installation

One command installs lexichron on a cluster that has Apptainer (most do; if
`apptainer` is not on your PATH, `module load apptainer` first):

```bash
curl -fsSL https://raw.githubusercontent.com/eric-d-knowles/lexichron/main/install.sh | bash
```

This pulls the current release image into `$SCRATCH/lexichron/` (or
`~/.lexichron/`) and installs two commands in `~/.local/bin`:

- `lexichron-ui` — the terminal interface (see below), the easiest way to
  download and process a corpus.
- `lexichron` — the command line, e.g. `lexichron acquire lexichron.yaml`.

Both run inside the image without you having to type any container commands,
and both re-pull the image automatically if a scratch purge removes it. To
install a specific version: `LEXICHRON_VERSION=0.6.0 bash install.sh`.

That is all that is needed to download and process corpora. Steps 2 and 3
below are for people who want to work with lexichron from their own notebooks
and scripts.

**2. Create a project environment** (for notebooks and your own code). Choose
a directory for your project (your notebooks and outputs will live there). It
can be empty or an existing project, such as a fresh clone of a repository.
Run:

```bash
apptainer run --app new-project $SCRATCH/lexichron/images/lexichron-0.6.0.sif /scratch/$USER/projects/gender-semantics
```

This creates a Python environment in `gender-semantics/.venv` that inherits
lexichron and all of its dependencies from the image, and registers a Jupyter
kernel for it. If the directory contains a `requirements.txt`, those packages
are installed too, so an existing project is set up in the same single step.
Re-running the command on a directory that already has a `.venv` keeps the
environment and refreshes the kernel and launcher. Your home directory and the
usual cluster data roots (`/scratch`, `/vast`, `/gpfs`, `/work`, `/project`,
`/projects`, `/data`) are made visible inside the container automatically when
they exist; add other directories with `--bind /path`, and set environment
variables for the project with `--env NAME=VALUE`.

Example notebooks for each workflow are in the
[`notebooks/`](https://github.com/eric-d-knowles/lexichron/tree/main/notebooks)
folder of this repository; put the ones you want in your project directory and
adapt them. They track the latest release, so use them with the latest image.

**3. Open a notebook** in JupyterLab, Positron or VS Code and select
*Python (gender-semantics | lexichron 0.6.0)* from the kernel menu. If the
kernel is not listed, reload the editor window.

#### Adding packages to a project

Open a notebook on the project's kernel and install packages the usual way:

```python
%pip install rpy2 pymc
```

or from a shell, using the project's launcher:

```bash
/scratch/$USER/projects/gender-semantics/.venv/host-python -m pip install rpy2 pymc
```

Packages land in the project's `.venv` and persist. Each project gets its own
environment; `.venv/host-python -m pip freeze --local` lists what you added.
Install on a login node if compute nodes have no internet access. The
environment is tied to the image's Python version (3.11): if a future image
moves to a newer Python, run `new-project` again and reinstall from your frozen
list; the command tells you when this is needed.

#### Running scripts and batch jobs

`new-project` also writes `<project>/.venv/host-python`, a launcher that runs
the project's Python inside the image with the right directories and
environment. It is an ordinary executable on the host, so it can be used
anywhere a Python path is expected, including Slurm scripts:

```bash
/scratch/$USER/projects/gender-semantics/.venv/host-python my_script.py
```

To execute a notebook unattended inside a Slurm job:

```bash
/scratch/$USER/projects/gender-semantics/.venv/host-python -m jupyter nbconvert \
    --to notebook --execute --output run.ipynb unigrams_workflow.ipynb
```

Notes on the container route:

- The image is read-only. All output paths (`db_path_stub`, model directories,
  logs) must point at bind-mounted host directories such as `/scratch`.
- Each image is pinned to one lexichron version. To use a different version,
  pull its image and run `new-project` with it (the project's environment is
  tied to the image's Python version, so it may need to be recreated; the
  command tells you if so).
- `apptainer run-help lexichron-0.6.0.sif` prints a short usage summary.
- How the image is built and released is described in `container/README.md`.

### Development installation (conda)

For working on lexichron itself. Clone the repository, activate a conda
environment, and install in editable mode so that changes to the source take
effect without reinstalling:

```bash
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron
conda activate your-environment
pip install -e .
```

`docs/architecture.md` describes how the filter and pivot pipelines are
structured.

#### Additional setup: Enchant library and Hunspell dictionaries

(The container image includes all of this; the steps below are for the conda
route only.)

Spell-checking relies on the **Enchant C library** and **Hunspell dictionaries**, and
model alignment requires the NLTK package's **Swadesh list of stable words**. These
components cannot be installed automatically via `pip`, so one additional
setup step is required after installing `lexichron`.

Activate the environment where `lexichron` is installed, then run the setup script:

```bash
bash scripts/setup_nlp_resources.sh
```

The script will:
- Ensure the **Enchant C library** is installed in the active Conda environment
- Download Hunspell dictionaries for all supported languages
- Install them inside the environment
- Configure Conda activation hooks so pyenchant can locate the dictionaries and the Enchant shared library

You only need to run this script **once per environment**.

#### Don't have an environment yet?

A reference `environment.yml` is provided with all dependencies pre-configured. To
create a dedicated conda environment from it:

```bash
conda env create -f environment.yml
conda activate lexichron
```

Then follow the standard installation steps above.

#### Registering a Jupyter kernel (if needed)

If you don't already have a Jupyter kernel registered for your project environment, you
can register one now:

```bash
python -m ipykernel install --user --name=lexichron --display-name="Python (lexichron)"
```

`--name` sets the internal kernel identifier and `--display-name` sets what appears in
Jupyter's kernel menu. Replace both with something meaningful to your project — for
example, `--name=gender_semantics --display-name="Python (gender semantics)"`.

#### Notes on the conda route

- **C++ compiler**: Building lexichron requires a C++ compiler (`g++`). This is included automatically if you create the environment from `environment.yml`. If installing into an existing environment, ensure one is available with `conda install -c conda-forge gxx`.
- **spaCy models** are downloaded automatically on first import.
- **Hunspell dictionaries** are handled by the setup script above and are not downloaded automatically.
- **`rocks-shim`** (a dependency of lexichron) is distributed as a pre-built Linux x86_64 wheel. If you are on macOS or Windows, installation will fail at this step. HPC cluster users on Linux are unaffected.

## Command line

Pipeline stages can also be run from a settings file instead of a notebook.
The file holds the settings a notebook's setup cells would hold, named exactly
as in the stage's Python function; `examples/lexichron.yaml` is a commented
template. One file describes one corpus: a shared `corpus` section (release,
language, n-gram size, database directory) and a section per stage.

```bash
lexichron acquire lexichron.yaml                       # download and ingest
lexichron filter lexichron.yaml                        # filter the ingested database
lexichron acquire lexichron.yaml --set acquire.workers=20   # override a setting
lexichron filter lexichron.yaml --dry-run              # show the call, don't run it
lexichron inspect 5grams_pivoted.db --prefix "[2011] nuclear"   # look at a database
```

`lexichron inspect` opens a database read-only (any size, any node) and
prints a few decoded records: the first ones, those from `--key` onwards, or
those with a `--prefix` (`--series` adds every year's counts under each
n-gram-keyed record; `--info` prints the database's provenance record instead). The key layout is sniffed from the first records
(`--layout` overrides it): a pivoted database's keys are typed with the year,
`[2011] nuclear`; an n-gram-keyed database's as the n-gram; a Davies
corpus's as `[fic][2000] text`. A vocabulary file is shown by rank.

Misspelled or missing settings are reported before anything runs. Inside the
container: `apptainer exec lexichron.sif lexichron acquire lexichron.yaml` (or a
project's `.venv/host-python -m lexichron.cli acquire lexichron.yaml`).

### The filter stage

`lexichron filter` turns the acquired database (`<n>grams.db`, every raw
n-gram as Google wrote it) into `<n>grams_processed.db`, where the text is
normalized and the noise removed. Each token is lowercased, lemmatized
(*women* → *woman*, with spaCy's tables for the corpus's language) and then
either checked against a vocabulary or put through the rule filters:
stopwords, tokens that are not purely alphabetic (ASCII letters only for
English), and tokens shorter than `min_len` become `<UNK>`. N-grams with
fewer than `min_context_tokens` real tokens are dropped, and years can be
binned. Raw n-grams that collapse onto the same processed key have their
year counts summed.

The stage is run twice per corpus, and writes a different kind of database
each time:

1. **Unigrams**, with no vocabulary, write `1grams_processed.db` (keyed by
   n-gram, a packed list of per-year counts each) and, from it,
   `1grams_processed.db/whitelist.txt`: the `output_vocabulary_top_n` most
   frequent lemmas, optionally only those present in every year of
   `output_vocabulary_year_range` and optionally spell-checked.
2. **Longer n-grams** use that whitelist (by default this corpus's own; a
   token is kept only if its lemma is in it, and `always_include` keeps
   focal tokens the unigram stage would have dropped as stopwords: *he*,
   *she*, ...) and write `<n>grams_pivoted.db` directly: keyed by year then
   n-gram, one record per (year, n-gram), which is what training reads.
   Filtering and pivoting are one pass over the raw database; raw spellings
   that collapse onto one n-gram are summed by the database's merge
   operator. (*Pivot by year* off writes an n-gram-keyed database instead.)

The work is split into key ranges balanced by data volume (from the source
database's own file metadata, so there is no pre-scan), worker processes
scan their ranges and spool results to node-local disk, and one process
merges everything into the destination. A worker holds the source open with
rocks-shim's `scan` profile (only the files its range touches, nothing
pinned, a small cache), so it needs about 200 MB however large the
database is, and all the CPUs of an allocation can be used: 40 workers fit
in the default 28G. A run that is interrupted resumes from the ranges
it had finished (`mode: resume`); `mode: restart` starts over.

## Terminal user interface

`lexichron-ui` is a terminal interface for people who would rather not edit
YAML or write Slurm scripts by hand. It works over a plain SSH session and
needs nothing set up in advance: fill in where the corpus should go and what
to download, and submit. The installer above puts it on your PATH; it starts
the interface together with a small helper on the host, so that jobs can be
submitted and watched from inside the container. (`lexichron-ui
/path/to/lexichron.yaml` opens an existing settings file; a project
environment created with `new-project` also has its own `.venv/lexichron-ui`
that uses that environment.)

Tabs:

- **Settings** — a *Stage* selector (`acquire` or `filter`) and a form for
  that stage's settings: the corpus section shared by every stage (directory,
  release, language, n-gram size) and the stage's own, with rarely-used
  settings under *Advanced*. Hover a `?` (or press F1) for help on a setting. Missing or
  invalid settings are reported at the top as you type. *Save* writes the
  settings file beside the database it describes,
  `<corpus directory>/<release>/<language>/<n>gram_files/lexichron.yaml`, so
  each corpus carries its own settings; that corpus's runs, logs and
  progress go in `.lexichron/` next to it. The exact Python call
  that will run is under *Resolved call* for anyone who wants to check it.
  Plain `lexichron-ui` reopens the file you saved last time, so closing the
  UI while a job runs and reopening it later brings back the settings and
  the job's progress.
- **Run** — runs the stage here, streaming its output. For small tests on a
  login node (set a small file range) or inside an interactive allocation.
- **Submit** — Slurm account, partition, CPUs, memory and time. The defaults
  (40 CPUs, 28G) are where acquisition stops gaining from more workers: the
  single process that merges their output into RocksDB becomes the limit.
  *Write batch script* writes `lexichron.<stage>.sbatch` next to the settings
  file; *Submit* also submits it and shows the job id.
- **Progress** — the latest runs, newest first, whether started with *Run
  here* or by a Slurm job: a progress bar, what is being worked on, the
  run's inputs, a grid of meters, and the tail of the run's log (collapsed;
  click *Log* to open it), refreshed every few seconds. The meters keep
  three record counts apart: what was *read* (raw data lines for
  acquisition, source records for filtering), what was *inserted* into the
  database (fewer: records with the same key merge before insertion) and
  what the final database *holds* (fewer again, exact once compacted). An
  acquisition run shows this session's figures and, since the database
  accumulates over acquisition jobs, the totals for the entire corpus with
  the database's count among them. A filter run also shows what each rule removed:
  tokens that were not alphabetic, too short, stopwords or not in the
  vocabulary, and n-grams dropped for having too few real tokens. The
  Runs table's *Records* column is the records kept so far while a run is
  going and the compacted database's count (marked `*`) once it is done;
  *Clear finished runs* removes the records of runs that are no longer
  running. (Every run writes
  `.lexichron/runs/<time>_<stage>/progress.json`; the UI reads it, so it can
  be closed and reopened while a job runs.) Your queued and running Slurm
  jobs are listed underneath, with a *Cancel* button.
- **Inspect** — look inside a database: pick one of the corpus's databases
  or vocabulary files (or type any path), and show the first rows, the rows
  from a key, or the rows with a prefix, decoded for the database's layout
  (`lexichron inspect` on the command line does the same); selecting a row
  of an n-gram-keyed database shows its full year series, and *How it was
  made* shows the database's provenance record. Databases are opened
  read-only with a small footprint, so this is fine on a login node.

Run straight from the image instead (`apptainer run --app ui lexichron-0.6.0.sif`)
and everything works except *Submit* and the job table, since Slurm commands
are not visible from inside the container.

## Example notebooks

The `notebooks/` folder contains one notebook per workflow. Copy the ones you
need into your project directory and edit the paths and settings at the top.

**Google Ngrams**

| Notebook | What it does |
|---|---|
| `unigrams_workflow.ipynb` | Download and ingest 1-grams, filter and lemmatize them, and write a vocabulary whitelist |
| `multigrams_workflow.ipynb` | Download and ingest 2–5-grams, filter them against the whitelist, and pivot to per-year format |
| `training_workflow.ipynb` | Train per-year `word2vec` models on processed n-grams, then normalize and align them |
| `ngram_change_analysis_workflow_eng.ipynb` | Track semantic change over time in the English models (similarity, relatedness, dimension projections) |
| `ngram_change_analysis_workflow_rus.ipynb` | The same analyses for Russian |

**Davies corpora** (COHA, COCA, Movies, …; the corpus files must be licensed and downloaded by you)

| Notebook | What it does |
|---|---|
| `davies_acquisition_workflow.ipynb` | Ingest Davies corpus files with year and genre metadata, filter them, and write a whitelist |
| `coha_training_workflow.ipynb` | Train and align per-year `word2vec` models on COHA |
| `movies_training_workflow.ipynb` | The same for the Movies corpus |
| `coha_change_analysis_workflow.ipynb` | Track semantic change in the COHA models |

For the full set of options, see the docstrings in `lexichron.stages.filter`,
`ngramprep.ngram_filter.config` and `daviesprep.davies_filter.config`.

## Output files

The n-gram stages lay their outputs out under the `db_path_stub` you give
them:

```
{db_path_stub}/{release}/{language}/{N}gram_files/
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
    lexichron.yaml              the settings the UI saved
    .lexichron/runs/            progress records of every run
```

Nothing temporary is left behind: workers spool their chunks under the
job's scratch directory and the chunks are deleted as they are ingested.

Model training writes one model per year (or per year bin) into the model
directory you specify. Use `common_db.compress_db()` to archive a database for
long-term storage or transfer.

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

## Reading a filter run's figures

The Progress tab (and the run's log) keep four record counts apart:

- **Scanned**: records read from the source database.
- **Kept**: source records that passed the filter. For 2-5-grams this is
  the number of n-grams kept, each with its whole year series.
- **Year records** (pivoted runs only): records made from them, one per
  n-gram and year.
- **Inserted**: records handed to the database, fewer than the above because
  records that normalize to the same key merge inside a worker's chunk
  before insertion.
- **In database**: what the compacted database holds: fewer again, since the
  same key arrives from several chunks and work units and those merge only
  at compaction.

Kept and In database are properties of the corpus and the rules; Inserted
and Year records depend on partitioning and chunk size.

## Support and Maintenance

This is research software, provided as-is. Issues and pull requests are welcome, but there is no guarantee of a response or of ongoing maintenance. If you depend on it, consider maintaining your own fork.

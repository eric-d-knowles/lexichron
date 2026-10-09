# lexichron

**Scalable tools for temporal linguistic analysis using Google Books Ngrams and Davies Corpora**

A comprehensive platform for semantic change research combining data preparation, text processing, and word embedding analysis. Supports both Google Ngrams (1-5 grams) and Mark Davies' corpora (COHA, COCA, etc.). Ideal for large datasets consisting of millions or billions of text examples. Provides efficient pipelines for acquiring, filtering, transforming, and organizing raw data—and for training and evaluating `word2vec` models to track semantic change over time.

While `lexichron` can be tuned to run on systems with fewer CPUs and less RAM, the package truly shines on High Performance Computing (HPC) or cloud infrastructures. Processing pipelines that might take days or weeks on a laptop can be completed in hours on a cluster or cloud platform.

## Contents

- [Citation](#citation)
- [Capabilities](#capabilities)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Container installation](#container-installation)
  - [Standard installation (conda)](#standard-installation-conda)
- [Quick Start](#quick-start)
- [Output Files](#output-files)
- [Advanced: Monitoring and Architecture](#advanced-monitoring-and-architecture)
- [Support and Maintenance](#support-and-maintenance)

## Citation

If you use lexichron in your research, please cite it:

```bibtex
@software{knowles2026lexichron,
  author = {Knowles, Eric D.},
  title = {lexichron: Scalable tools for temporal linguistic analysis},
  year = {2026},
  url = {https://github.com/eric-d-knowles/lexichron},
  version = {0.1.0}
}
```

Alternatively, you can click "Cite this repository" in the GitHub sidebar for additional citation formats.

## Capabilities

### Data Preparation

- **Data acquisition:** Download n-gram datasets (1- through 5-grams) or access Davies corpora. (Davies datasets must be licensed and downloaded by the user.) Immediately ingest data into a queryable RocksDB database.
- **Language support:** N-gram pipelines support English, Chinese (simplified), French, German, Hebrew, Italian, Russian, and Spanish.
- **Configurable processing:** Apply any or all of the following transformations: case normalization, stopword removal, short word removal, non-alphabetic token removal, and lemmatization. Discarded tokens are replaced in the corpus with `<UNK>`.
- **Whitelist creation:** Output the top-N most frequent unigrams, applying optional spell-checking, then use this whitelist to efficiently filter text examples. Spell-checking discards proper nouns when used in conjunction with case normalization (e.g., "Jackson" and "Einstein" would be discarded). A year range can be defined to ensure that the whitelist contains only tokens found in all specified years.
- **Bigram hyphenation:** Automatically convert semantically interesting bigrams into hyphenated unigrams (e.g., "working class" → "working-class", "nuclear family" → "nuclear-family"), preserving multiword concepts as single tokens for downstream analysis.
- **Token immunity:** Define tokens that should always be preserved during filtering, immune to exclusion rules. Useful for domain-specific terms, names or proper nouns, historical keywords, or particular multiword expressions that you want to ensure remain in your corpus regardless of other filtering criteria.
- **Temporal analysis support:** Reorganize n-gram data into a format suitable for time-series analyses:
  - BEFORE: `n-gram → (year1, count1, volumes1) (year2, count2, volumes2) ... (yearn, countn, volumesn)`
  - AFTER:
    - `[year1] n-gram → (count1, volumes1)`
    - `[year2] n-gram → (count2, volumes2)`
    - `...`
    - `[yearn] n-gram → (countn, volumesn)`
- **High-throughput architecture:** Parallel processing with load balancing, progress tracking, and resume capability in the event of interruption.
- **Research-friendly storage:** Fast key-value database (RocksDB) quickly queries even enormous datasets.

### Model Training and Evaluation

- **Word embeddings:** Train `word2vec` models on the processed n-grams using `gensim`'s implementation. Optionally use `corpus_file` mode to enable fast, multithreaded training and training multiple years at once. Easily adjust model hyperparameters:
  - `approach`: use skip-gram or continuous bag-of-words (CBOW) architectures
  - `vector_size`: the number of vector dimensions (features) to extract
  - `window_size`: the width of the context window
  - `min_count`: the minimum frequency of words to include in the model
  - `weight_by`: downweight common ngrams by frequency or document count
- **Evaluation:** Evaluate the performance of the trained model using standard intrinsic tests of similarity and analogy performance. Plot the results for visual comparison of model quality. Use mixed-model regression to quantify the impact of different hyperparameters on model performance across years.

## Workflow

The toolkit provides two parallel pipelines for different data sources:

### Google Ngrams Pipeline

1. **`ngram_acquire`**: Fetch raw n-gram files (1-5 grams) from the Google Books repository and store in a RocksDB database for fast querying.
2. **`ngram_filter`**: Apply linguistic transformations (case normalization, lemmatization, stopword removal, spell-checking, bigram hyphenation) to prepare data. Optionally generate vocabulary whitelists.
3. **`ngram_pivot`**: Reorganize data from "wide" (per-ngram) to "long" (per-year) format for time-series analysis.

### Davies Corpora Pipeline

1. **`davies_acquire`**: Ingest Davies corpus files (COHA, COCA, etc.) with genre and year information into RocksDB.
2. **`davies_filter`**: Apply the same filtering and preprocessing transformations as `ngram_filter` for consistency.

### Analysis Tools

**`analyze`**: General-purpose analysis tools for tracking semantic drift and similarity changes across time using trained word embeddings. Works with both ngram and Davies corpus data.

### Model Training

**`train/word2vec`**: Train per-year word2vec models, evaluate across intrinsic benchmarks, align models across years, and analyze hyperparameter impact via regression.

## System Requirements

- HPC cluster or workstation with multiple CPU cores (30+ cores recommended)
- Large amount of RAM (80+ GB recommended)
- Fast local storage (NVMe SSD recommended)
- Several TB of disk space for processing and storing very large corpora
- Settings can be tuned for fewer resources, but at the cost of processing speed

## Installation

There are two ways to install lexichron:

- **Container (recommended for users):** pull a prebuilt Apptainer image that
  contains lexichron and every dependency, then run the notebooks against it.
  Nothing to compile, no conda environment, no setup script.
- **Conda (for developers):** install from source into a conda environment.
  Use this if you plan to modify the code.

### Container installation

Each release of lexichron is published as an Apptainer image at
`ghcr.io/eric-d-knowles/lexichron`. The image version is the lexichron version.

**1. Pull the image** (once; on many clusters you first need `module load apptainer`):

```bash
mkdir -p /scratch/$USER/containers && cd /scratch/$USER/containers
apptainer pull lexichron-0.1.0.sif oras://ghcr.io/eric-d-knowles/lexichron:0.1.0
```

Replace `0.1.0` with the version you want (`latest` is the newest release).
Keep the image on a large filesystem (it is roughly 700 MB), not in your home
directory.

**2. Create a project environment.** Choose a directory for your project (your
notebooks and outputs will live there) and run:

```bash
apptainer run --app new-project /scratch/$USER/containers/lexichron-0.1.0.sif /scratch/$USER/projects/gender-semantics
```

This creates a Python environment in `gender-semantics/.venv` that inherits
lexichron and all of its dependencies from the image, and registers a Jupyter
kernel for it. Your home directory and the usual cluster data roots
(`/scratch`, `/vast`, `/gpfs`, `/work`, `/project`, `/projects`, `/data`) are
made visible inside the container automatically when they exist; add other
directories with `--bind /path`.

*Optional:* to start from the example notebooks, check out the copy that
matches your image version into the project directory:

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/eric-d-knowles/lexichron.git /tmp/lexichron-v0.1.0
cp /tmp/lexichron-v0.1.0/notebooks/*.ipynb /scratch/$USER/projects/gender-semantics/
```

(or download them from the `notebooks/` folder of the `v0.1.0` tag on GitHub).

**3. Open a notebook** in JupyterLab, Positron or VS Code and select
*Python (gender-semantics | lexichron 0.1.0)* from the kernel menu. If the
kernel is not listed, reload the editor window.

#### Adding packages to a project

Open a notebook on the project's kernel and install packages the usual way:

```python
%pip install rpy2 pymc
```

or from a shell:

```bash
apptainer exec lexichron-0.1.0.sif /scratch/$USER/projects/gender-semantics/.venv/bin/pip install rpy2 pymc
```

Packages land in the project's `.venv` and persist. Each project gets its own
environment; `.venv/bin/pip freeze --local` lists what you added. Install on a
login node if compute nodes have no internet access. The environment is tied to
the image's Python version (3.11): if a future image moves to a newer Python,
run `new-project` again and reinstall from your frozen list; the script tells
you when this is needed.

To run Python directly instead of through a notebook:

```bash
apptainer exec --bind /scratch lexichron-0.1.0.sif python my_script.py
```

and to execute a notebook unattended inside a Slurm job:

```bash
apptainer exec --bind /scratch lexichron-0.1.0.sif \
    jupyter nbconvert --to notebook --execute --output run.ipynb notebooks/unigrams_workflow.ipynb
```

Notes on the container route:

- The image is read-only. All output paths (`db_path_stub`, model directories,
  logs) must point at bind-mounted host directories such as `/scratch`.
- Each image is pinned to one lexichron version. To use a different version,
  pull its image and run `new-project` with it (the project's environment is
  tied to the image's Python version, so it may need to be recreated; the
  command tells you if so).
- `apptainer run-help lexichron-0.1.0.sif` prints a short usage summary.
- How the image is built and released is described in `container/README.md`.

### Standard installation (conda)

Clone the repository, navigate into it, activate your project's conda environment, and
install:

```bash
git clone https://github.com/eric-d-knowles/lexichron.git
cd lexichron
conda activate your-environment
pip install .
```

If you plan to modify the source code, install in editable mode instead:

```bash
pip install -e .
```

Editable mode links the installation directly to your cloned repository, so any changes
you make to the source are immediately reflected without reinstalling.

### Additional setup: Enchant library and Hunspell dictionaries

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

### Don't have an environment yet?

A reference `environment.yml` is provided with all dependencies pre-configured. To
create a dedicated conda environment from it:

```bash
conda env create -f environment.yml
conda activate lexichron
```

Then follow the standard installation steps above.

### Registering a Jupyter kernel (if needed)

If you don't already have a Jupyter kernel registered for your project environment, you
can register one now:

```bash
python -m ipykernel install --user --name=lexichron --display-name="Python (lexichron)"
```

`--name` sets the internal kernel identifier and `--display-name` sets what appears in
Jupyter's kernel menu. Replace both with something meaningful to your project — for
example, `--name=gender_semantics --display-name="Python (gender semantics)"`.

### Notes

- **C++ compiler**: Building lexichron requires a C++ compiler (`g++`). This is included automatically if you create the environment from `environment.yml`. If installing into an existing environment, ensure one is available with `conda install -c conda-forge gxx`.
- **spaCy models** are downloaded automatically on first import.
- **Hunspell dictionaries** are handled by the setup script above and are not downloaded automatically.
- **`rocks-shim`** (a dependency of lexichron) is distributed as a pre-built Linux x86_64 wheel. If you are on macOS or Windows, installation will fail at this step. HPC cluster users on Linux are unaffected.

## Quick Start

See the `notebooks/` directory for complete workflow examples:

### Google Ngrams Workflows

- **`eng_unigrams_workflow.ipynb`** — Download and ingest 1-grams, apply filtering and preprocessing, generate vocabulary whitelist (English)
- **`eng_multigrams_workflow.ipynb`** — Download and filter 2-5 grams using whitelist (English)
- **`rus_unigrams_workflow.ipynb`** — Same as English unigrams but for Russian
- **`rus_multigrams_workflow.ipynb`** — Same as English multigrams but for Russian
- **`ngrams_change_analysis_workflow.ipynb`** — Analyze semantic drift and track meaning changes over time

### Davies Corpora Workflows

- **`davies_acquisition_workflow.ipynb`** — Ingest Davies corpus files with genre and year information
- **`coha_training_workflow.ipynb`** — Train word2vec models on COHA corpus data
- **`coha_change_analysis_workflow.ipynb`** — Analyze semantic change in historical English (COHA)

### Model Training & Evaluation

- **`training_workflow.ipynb`** — Train word embeddings on processed n-grams
- **`ngram_training_workflow.ipynb`** — End-to-end word2vec training pipeline for n-grams

### Basic Usage Example

```python
from pathlib import Path
from ngramprep.ngram_acquire import download_and_ingest_to_rocksdb
from ngramprep.ngram_filter import PipelineConfig, FilterConfig, build_processed_db
from ngramprep.ngram_pivot import run_pivot_pipeline
from ngramprep.ngram_pivot.config import PipelineConfig as PivotConfig

# Step 1: Download and ingest n-grams
download_and_ingest_to_rocksdb(
    ngram_size=1,
    repo_release_id="20200217",
    repo_corpus_id="eng",
    db_path_stub="/data/ngrams",
    workers=30
)

# Step 2: Filter and clean
pipeline_config = PipelineConfig(
    src_db=Path("/data/ngrams/1grams.db"),
    dst_db=Path("/data/ngrams/1grams_processed.db"),
    tmp_dir=Path("/data/ngrams/tmp"),
    num_workers=40,
    mode="restart"
)

filter_config = FilterConfig(
    lowercase=True,
    filter_short=True,
    alpha_only=True
)

build_processed_db(pipeline_config, filter_config)

# Step 3: Pivot for time-series analysis (optional)
pivot_config = PivotConfig(
    src_db=Path("/data/ngrams/1grams_processed.db"),
    dst_db=Path("/data/ngrams/1grams_pivoted.db"),
    tmp_dir=Path("/data/ngrams/pivot_tmp"),
    num_workers=30,
    mode="restart"
)

run_pivot_pipeline(pivot_config)
```

For detailed configuration options, see the docstrings in `ngramprep.ngram_filter.config`, `ngramprep.ngram_pivot.config`, and `daviesprep.davies_filter.config`, or refer to the example notebooks.

## Output Files

After running the pipelines, you'll have:

- **Final database** (`dst_db`): Query-ready RocksDB containing your processed n-grams
- **Frequency whitelist** (optional): Text file listing retained n-grams with occurrence counts (useful for documenting your corpus)
- **Compressed archive** (optional): Use `common_db.compress_db()` for efficient long-term storage and transfer

**Temporary files** (can be deleted after completion):
- `tmp_dir/worker_outputs/`: Intermediate processing shards
- `tmp_dir/work_tracker.db`: Progress tracking database (useful for debugging interrupted jobs)

## Advanced: Monitoring and Architecture

### Real-time Progress Display

The `ngram_filter` and `ngram_pivot` pipelines print periodic updates showing:

```
      items         kept%         workers         units          rate          elapsed
──────────────────────────────────────────────────────────────────────────────────────────
    128.56M         85.4%          8/40          10·24·1237     214.2k/s        10m00s
```

Column meanings:

- **items**: Total records processed so far
- **kept%**: Percentage of n-grams retained after filtering (100% for pivot)
- **workers**: Active workers / total workers (shows load distribution)
- **units**: Work distribution status as `pending·processing·completed` (shows load balancing)
- **rate**: Processing throughput (records per second)
- **elapsed**: Total time since pipeline started

### Two-Stage Pipeline Architecture

The `ngram_filter` and `ngram_pivot` pipelines use a two-phase design for memory efficiency and fault tolerance:

1. **Processing stage**: Workers divide the input data into chunks, process them in parallel, and write results to temporary files (`tmp_dir/worker_outputs/`)
2. **Ingestion stage**: Temporary files are merged into the final database using parallel streaming

This design enables:

- **Resume capability**: Interrupted jobs pick up where they left off
- **Load balancing**: Work units are pre-balanced via density-based sampling; workers steal remaining units as they finish
- **Balanced work units**: Density-based sampling scans the corpus to estimate token frequency distributions, then partitions work so each unit has similar total token mass, reducing straggler workers and keeping throughput consistent
- **Memory efficiency**: Large datasets don't need to fit in RAM
- **Predictable resource usage**: Memory consumption is bounded regardless of corpus size

*Note: Davies acquisition pipelines use simpler direct ingestion and do not employ the two-stage architecture.*

## Support and Maintenance

This project is provided as-is for research and development purposes. While issues and pull requests are welcome, there is no guarantee of response time or ongoing maintenance. The code is shared in the spirit of open science, but support is provided on a best-effort basis only.

For critical production use, consider forking and maintaining your own version.

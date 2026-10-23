# lexichron — notes for Claude Code sessions

lexichron turns Google Books n-gram data (and Davies corpora) into per-year
word2vec models for research on semantic change. It is a Python package with a
command line (`lexichron <stage> project.yaml`), a Textual terminal UI
(`lexichron ui`), and a container image; researchers run it on HPC clusters
under Slurm. The README is the user manual and is kept current; read the
section for whatever you are touching before changing it.

## Layout

- `src/lexichron/` — the application: `cli.py` (stage registry `STAGES`,
  `inspect`), `config.py` (project YAML → stage call), `stages/` (one plain-
  value entry point per stage: `filter`, `tune`, `train`; `acquire` lives in
  `ngramprep`), `training/` (the shared word2vec core: `corpus` writer,
  `trainer`, `grid`, `scheduler`, `scores`, `ensemble`, `axis`, `stability`,
  `benchmarks`), `ui/` (`app.py`, `schema.py` which derives each stage's form
  from its function signature, `slurm.py` which writes batch scripts),
  `arrays.py` (Slurm job-array slicing), `safeguards.py`, `resources.py`,
  `provenance.py`, `inspection.py`.
- `src/ngramprep/` — acquisition, filtering and pivoting of n-gram databases
  (RocksDB through `rocks-shim`), with Cython kernels.
- `src/train/`, `src/analyze/` — the older notebook-era code; the stages above
  supersede the training parts. Do not build new features on them.
- `tests/` — pytest; `fake_rocks.py` stands in for RocksDB and `test_tune_stage.py`
  has a deterministic `fake_train` so stages run without gensim.
- `container/`, `install.sh`, `.github/workflows/container.yml` — the image,
  the one-line installer, CI.

## Working conventions

- The UI form is generated: a stage's keyword arguments become fields. When you
  add a setting, add its label and help to `ui/schema.py` (`LABELS`, `HELP`,
  `ORDER_BY_STAGE`, and `ADVANCED` or `HIDDEN` as appropriate) and document it
  in the function's `Args:` docstring, which is the fallback help.
- Everything a stage writes is resumable and written atomically (`.part` then
  rename). Stages never prompt; the CLI/UI ask about deletions through
  `safeguards.overwrite_risk` before calling them.
- Keep stage functions callable from Python with plain values (strings,
  numbers, lists): the same function serves CLI, UI and notebooks.
- Prose style in code, docs and commits: plain sentences, say why, no
  marketing. Commit messages describe the change and the reason.

## Tests

```bash
pip install -e .          # builds the Cython extensions the tests import
python -m pytest -q tests
```

The UI tests need `textual` (`pip install -e '.[ui]'`). CI runs the whole
suite inside the freshly built image on every version tag and on manual
builds (`gh workflow run container.yml --ref <branch> -f tag=<name>`), which
is the reliable way to run the suite when a local environment is incomplete.
The bundled datasets the tune stage expects (SimLex, WordSim, questions-words,
the VADER lexicon) come from `$LEXICHRON_DATA_DIR` or `/opt/lexichron/data`
in the image; tests create their own.

## Releases

1. Bump the version in `pyproject.toml`, `src/lexichron/__init__.py`,
   `CITATION.cff` (and `date-released`), and the version strings in the README.
2. Commit to `main`, tag `vX.Y.Z`, push the tag.
3. CI builds the image, runs the tests inside it, and publishes
   `ghcr.io/eric-d-knowles/lexichron:X.Y.Z` and `:latest`. A manual
   workflow run publishes an image under the tag you give it, never `latest`.

## Design decisions that are settled

- **Tuning vs training are separate stages.** `tune` runs a grid on a few
  sampled years (adjacent pairs spread over the study period) and scores
  stability (seed, split-half, adjacent-year) and quality (benchmarks,
  coverage) plus the valence-axis family; it suggests a cell per metric family
  and never applies a choice itself. `train` fits one cell for every year as a
  replicate ensemble (Poisson corpus resamples × seeds) and reports a per-year
  noise profile. Tuning metrics must not assume a user's research question; the
  valence axis is the general-purpose yardstick for axis-projection work.
- **Per-replicate models are the product.** Nothing is aligned or averaged in
  `train`; scalar measurements made inside each replicate are rotation-
  invariant. Projecting a user's own words onto their own axis is a later
  stage, not part of `train`.
- **Scores are files.** A tuning search keeps one JSON file per scored
  cell-year (`scores/`) and per year pair (`pairs/`); `results.tsv`,
  `pairs.tsv`, `summary.*` are exports. Readers derive from the files.
- **Job arrays.** `slurm.jobs > 1` splits `tune` or `train` over the tasks of
  a Slurm job array; a task takes every N-th year (round-robin), runs in
  resume mode, and writes only its years. A year pair is scored by whichever
  task finds both models on disk. `mode: restart` is refused under an array:
  the UI deletes first, then submits.
- **The cross-year scheduler** (`training/scheduler.py`) trains one queue of
  models in year order with corpora written ahead, keyed by a task's
  `corpus_key` (the year for tune, year × replicate for train).
- **`<UNK>` default is `strip`** in training (drop the token, keep the n-gram
  if two real tokens remain).
- The image rebuilds gensim's `corpus_file` reader without the GIL
  (`scripts/patch_gensim.py`); without it a model is capped at a few cores.

## Running on NYU Torch (Eric's setup)

- Never run a stage on a login node. Tests and the UI are fine there. For a
  smoke test of a stage, run it on a compute node:
  `srun --account=torch_pr_600_general --cpus-per-task=16 --mem=48G --time=01:00:00 lexichron <stage> lexichron.yaml --set ...`
  For anything longer, write a batch script (`write_sbatch`, or the UI's Submit
  tab) and `sbatch` it; inside a job `$SLURM_CPUS_PER_TASK` sets the workers.
- Corpora live under `/scratch/edk202/NLP_corpora/Google_Books/`; the working
  corpus is `eng-us`, release `20200217`, 5-grams, tagged tokens only. The
  chosen tuning cell for it is 100 dims, window 2, min count 5, 10 epochs,
  skip-gram, `none` weighting.
- Put corpus files on node-local disk in jobs (`temp_dir: $TMPDIR`).
- Nothing on a login node should hold large data in memory; the pivoted
  database is hundreds of GB and is read in streaming scans only.

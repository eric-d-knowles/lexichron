# Container build and release (maintainer notes)

End users only need the "Container installation" section of the main README.
This file covers how the image is produced.

## Files

- `lexichron.def` — Apptainer definition. Builds lexichron from the repository
  checkout and bakes in Enchant, Hunspell dictionaries, spaCy models and the
  NLTK Swadesh list. Defines two apps: `new-project` (the documented user
  path) and `register-kernel`, which `new-project` calls to write the
  kernelspec and which can also be run on its own to get a kernel for the
  bare image (`apptainer run --app register-kernel lexichron.sif`).
- `register_kernel.sh`, `new_project.sh` — the scripts behind those apps. They
  are copied into the image at `/opt/lexichron/` and are meant to run *inside*
  it; they are not run from the repository.
- `../.github/workflows/container.yml` — builds and publishes the image.

## Releasing a version

1. Set the version in `pyproject.toml` (and `CITATION.cff`, `src/ngramprep/__init__.py`).
2. Commit, then tag and push:

       git tag v0.2.0
       git push origin main --tags

3. The workflow builds the image, runs its self-tests, and publishes
   `ghcr.io/eric-d-knowles/lexichron:0.2.0` and `:latest`. It fails early if the
   tag does not match `pyproject.toml`. Each run's Summary page lists the image
   size, apps, versions and sha256.

Each image is pinned to one lexichron version; a fix means a new release.

## Testing unreleased code

Actions → "Build container image" → Run workflow (any branch). This publishes
`ghcr.io/eric-d-knowles/lexichron:dev` without touching `latest`. The `.sif`
is also attached to the run as an artifact for seven days.

`dev` and `latest` are moving tags, and Apptainer caches pulls by tag. To
re-pull one of them after a new build:

    apptainer cache clean -f
    apptainer pull --force lexichron-dev.sif oras://ghcr.io/eric-d-knowles/lexichron:dev
    apptainer inspect --list-apps lexichron-dev.sif   # should list both apps

## Building locally

From the repository root, on a machine with Apptainer and root or fakeroot:

    apptainer build --fakeroot lexichron.sif container/lexichron.def

## Design notes

- The image is read-only. Everything the pipelines write goes to bind-mounted
  host paths; `register_kernel.sh` writes a launcher into the kernelspec that
  binds the standard data roots that exist on the host, skipping those the
  site already binds system-wide (detected from `/proc/mounts` at
  registration time).
- Project environments are `python -m venv --system-site-packages` created
  inside the container, so they inherit the image's packages and store only
  the user's additions. `register_kernel.sh` writes one host-side launcher
  script (`<project>/.venv/host-python`) that applies the binds and `--env`
  settings; the kernelspec calls it, and so can Slurm scripts. The compiler is kept in the image so that source
  builds in those environments work. A venv is tied to the image's Python
  minor version; `new_project.sh` refuses to reuse one built for a different
  version.
- Compiled extensions (`.so`) are not tracked in git; the build compiles them.

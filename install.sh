#!/usr/bin/env bash
# install.sh — install lexichron on an HPC cluster in one step.
#
#   curl -fsSL https://raw.githubusercontent.com/eric-d-knowles/lexichron/main/install.sh | bash
#
# What it does:
#   1. checks that Apptainer (or Singularity) is available, loading its module if needed
#   2. pulls the lexichron image into $SCRATCH/lexichron (or ~/.lexichron)
#   3. installs the `lexichron` and `lexichron-ui` commands into ~/.local/bin
#   4. makes sure ~/.local/bin is on your PATH
#
# Options (environment variables):
#   LEXICHRON_VERSION   version to install (default: latest release)
#   LEXICHRON_HOME      where to keep the image (default: $SCRATCH/lexichron,
#                       else ~/.lexichron)
#   LEXICHRON_BIN       where to put the commands (default ~/.local/bin)
set -euo pipefail

REPO="eric-d-knowles/lexichron"
VERSION="${LEXICHRON_VERSION:-latest}"
BIN_DIR="${LEXICHRON_BIN:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'lexichron install: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Container runtime: Apptainer, or Singularity (its predecessor; the
#    commands are the same, so either serves)
# ---------------------------------------------------------------------------
RUNTIME=""
find_runtime() {
    for c in apptainer singularity; do
        if command -v "$c" >/dev/null 2>&1; then RUNTIME="$c"; return 0; fi
    done
    return 1
}
if ! find_runtime && command -v module >/dev/null 2>&1; then
    for m in apptainer singularity; do
        if module avail "$m" 2>&1 | grep -qi "$m"; then
            say "Loading the $m module..."
            module load "$m" 2>/dev/null || true
            find_runtime && break
        fi
    done
fi
[ -n "$RUNTIME" ] || die "neither 'apptainer' nor 'singularity' is available. On most clusters: module load apptainer (or module load singularity), then re-run."
say "Container runtime: $RUNTIME"

# ---------------------------------------------------------------------------
# 2. Where to keep the image
# ---------------------------------------------------------------------------
if [ -n "${LEXICHRON_HOME:-}" ]; then
    HOME_DIR="$LEXICHRON_HOME"
elif [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ]; then
    HOME_DIR="$SCRATCH/lexichron"
elif [ -d "/scratch/$USER" ]; then
    HOME_DIR="/scratch/$USER/lexichron"
else
    HOME_DIR="$HOME/.lexichron"
fi
IMAGE_DIR="$HOME_DIR/images"
export APPTAINER_TMPDIR="$HOME_DIR/apptainer-tmp" SINGULARITY_TMPDIR="$HOME_DIR/apptainer-tmp"
export APPTAINER_CACHEDIR="$HOME_DIR/apptainer-cache" SINGULARITY_CACHEDIR="$HOME_DIR/apptainer-cache"
mkdir -p "$IMAGE_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---------------------------------------------------------------------------
# 3. Pull the image (into a temporary name, then rename to its real version)
# ---------------------------------------------------------------------------
REF="ghcr.io/$REPO:$VERSION"
TMP_SIF="$IMAGE_DIR/.lexichron-download-$$.sif"
say "Pulling oras://$REF ..."
"$RUNTIME" pull "$TMP_SIF" "oras://$REF" || die "pull failed (is the cluster able to reach ghcr.io?)"

ACTUAL="$("$RUNTIME" exec "$TMP_SIF" lexichron --version 2>/dev/null | awk '{print $2}')"
[ -n "$ACTUAL" ] || die "the downloaded image does not report a lexichron version"
SIF="$IMAGE_DIR/lexichron-$ACTUAL.sif"
mv -f "$TMP_SIF" "$SIF"
say "Image: $SIF (lexichron $ACTUAL)"

# ---------------------------------------------------------------------------
# 4. Host commands
# ---------------------------------------------------------------------------
# Re-pulls use the tag that was pulled: a release image reports its version,
# which is also its tag; a manual build reports 0.2.0+dev.<sha>, which is not.
case "$ACTUAL" in *+*) PULL_TAG="$VERSION" ;; *) PULL_TAG="$ACTUAL" ;; esac
# The runtime travels as a variable (images before 0.8 do not know a
# --runtime option). BASH_ENV is dropped: environment-modules sets it to its
# own bash init, which the image's bash would otherwise try to source.
export LEXICHRON_RUNTIME="$RUNTIME"
env -u BASH_ENV -u ENV "$RUNTIME" run --app install "$SIF" --bin "$BIN_DIR" --ref "ghcr.io/$REPO:$PULL_TAG" \
    --tmpdir "$APPTAINER_TMPDIR" --cachedir "$APPTAINER_CACHEDIR" >/dev/null \
    || die "writing the lexichron commands failed (run without >/dev/null to see why: $RUNTIME run --app install $SIF)"
# Images before 0.8 write commands that call apptainer by name and source
# BASH_ENV: point them at the runtime found here.
for f in "$BIN_DIR/lexichron" "$BIN_DIR/lexichron-ui"; do
    [ -f "$f" ] || continue
    if grep -q 'command -v apptainer' "$f" 2>/dev/null; then
        sed -i -e "s/\bapptainer \(pull\|exec\)\b/$RUNTIME \1/g" -e "s/command -v apptainer/command -v $RUNTIME/" \
               -e "s/'apptainer' not found; try: module load apptainer/'$RUNTIME' not found; try: module load $RUNTIME/" \
               -e "0,/^set -u\$/s//set -u\nunset BASH_ENV ENV/" "$f"
    fi
done

# ---------------------------------------------------------------------------
# 5. PATH
# ---------------------------------------------------------------------------
# The line only prepends when the directory is not on PATH yet. An
# unconditional prepend moves ~/.local/bin ahead of an activated conda
# environment every time a new shell reads ~/.bashrc (a job shell started
# with srun from an activated login shell, say), and then `lexichron`
# silently means the image launcher rather than the environment's command.
PATH_LINE="case \":\$PATH:\" in *\":$BIN_DIR:\"*) ;; *) export PATH=\"$BIN_DIR:\$PATH\" ;; esac"
OLD_PATH_LINE="export PATH=\"$BIN_DIR:\$PATH\""
case ":$PATH:" in
    *":$BIN_DIR:"*) on_path=1 ;;
    *) on_path=0 ;;
esac
if [ -w "$HOME/.bashrc" ] && grep -qxF "$OLD_PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    # an earlier install wrote the unconditional form: replace it in place
    awk -v old="$OLD_PATH_LINE" -v new="$PATH_LINE" '$0 == old { print new; next } { print }' \
        "$HOME/.bashrc" > "$HOME/.bashrc.lexichron-tmp" && mv "$HOME/.bashrc.lexichron-tmp" "$HOME/.bashrc"
    say "Updated the PATH line in ~/.bashrc so an activated environment stays ahead of $BIN_DIR"
elif [ "$on_path" = 0 ] && [ -w "$HOME/.bashrc" ] && ! grep -qF "$BIN_DIR" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# added by lexichron install\n%s\n' "$PATH_LINE" >> "$HOME/.bashrc"
    say "Added $BIN_DIR to your PATH in ~/.bashrc"
fi

say ""
say "lexichron $ACTUAL is installed."
say "  lexichron-ui              open the terminal interface"
say "  lexichron --help          command line"
if [ "$on_path" = 0 ]; then
    say ""
    say "Open a new shell (or run:  $PATH_LINE ) for the commands to be found."
fi

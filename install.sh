#!/usr/bin/env bash
# install.sh — install lexichron on an HPC cluster in one step.
#
#   curl -fsSL https://raw.githubusercontent.com/eric-d-knowles/lexichron/main/install.sh | bash
#
# What it does:
#   1. checks that Apptainer is available (and says how to load it if not)
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
# 1. Apptainer
# ---------------------------------------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1 && module avail apptainer 2>&1 | grep -qi apptainer; then
        say "Loading the apptainer module..."
        module load apptainer
    fi
fi
command -v apptainer >/dev/null 2>&1 || die "'apptainer' is not available. On most clusters: module load apptainer, then re-run."

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
export APPTAINER_TMPDIR="$HOME_DIR/apptainer-tmp"
export APPTAINER_CACHEDIR="$HOME_DIR/apptainer-cache"
mkdir -p "$IMAGE_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---------------------------------------------------------------------------
# 3. Pull the image (into a temporary name, then rename to its real version)
# ---------------------------------------------------------------------------
REF="ghcr.io/$REPO:$VERSION"
TMP_SIF="$IMAGE_DIR/.lexichron-download-$$.sif"
say "Pulling oras://$REF ..."
apptainer pull "$TMP_SIF" "oras://$REF" || die "pull failed (is the cluster able to reach ghcr.io?)"

ACTUAL="$(apptainer exec "$TMP_SIF" lexichron --version 2>/dev/null | awk '{print $2}')"
[ -n "$ACTUAL" ] || die "the downloaded image does not report a lexichron version"
SIF="$IMAGE_DIR/lexichron-$ACTUAL.sif"
mv -f "$TMP_SIF" "$SIF"
say "Image: $SIF (lexichron $ACTUAL)"

# ---------------------------------------------------------------------------
# 4. Host commands
# ---------------------------------------------------------------------------
apptainer run --app install "$SIF" --bin "$BIN_DIR" --ref "ghcr.io/$REPO:$ACTUAL" \
    --tmpdir "$APPTAINER_TMPDIR" --cachedir "$APPTAINER_CACHEDIR" >/dev/null

# ---------------------------------------------------------------------------
# 5. PATH
# ---------------------------------------------------------------------------
PATH_LINE="export PATH=\"$BIN_DIR:\$PATH\""
case ":$PATH:" in
    *":$BIN_DIR:"*) on_path=1 ;;
    *) on_path=0 ;;
esac
if [ "$on_path" = 0 ] && [ -w "$HOME/.bashrc" ] && ! grep -qF "$BIN_DIR" "$HOME/.bashrc" 2>/dev/null; then
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

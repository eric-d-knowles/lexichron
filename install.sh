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
#   LEXICHRON_SIF       install from this image file instead of downloading
#                       (one copied from a machine that could download it)
#   LEXICHRON_FETCH     "curl" downloads the image with curl from the start
#                       (resumable; what the installer falls back to when the
#                       container runtime's own download is cut off)
#
# The image is one file of about 4 GB. On a slow or unreliable connection the
# runtime's download can be cut off partway, and it starts over each time.
# The installer then fetches the same file with curl, which resumes where a
# broken connection left off, also across runs of this script, and checks
# the file's SHA-256 against the registry's before using it.
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
# 3. Get the image (into a temporary name, then rename to its real version)
# ---------------------------------------------------------------------------
REF="ghcr.io/$REPO:$VERSION"
TMP_SIF="$IMAGE_DIR/.lexichron-download-$$.sif"

# The image straight from the registry with curl, resuming a partial file.
# An ORAS artifact is a manifest naming one blob (the .sif); the blob is
# served with HTTP range support, so a download cut off at 10% continues from
# 10%. The partial file is named by the blob's digest and kept between runs.
fetch_with_curl() {
    command -v curl >/dev/null 2>&1 || { say "curl is not available for a resumable download"; return 1; }
    local api="https://ghcr.io/v2/$REPO" token manifest digest size part have tries=0 last=-1 stalls=0
    token="$(curl -fsSL "https://ghcr.io/token?scope=repository:$REPO:pull" | sed -E 's/.*"token":"([^"]+)".*/\1/')" \
        || { say "could not get a download token from ghcr.io"; return 1; }
    manifest="$(curl -fsSL -H "Authorization: Bearer $token" -H "Accept: application/vnd.oci.image.manifest.v1+json" \
        "$api/manifests/$VERSION")" || { say "could not read the image's manifest ($REF)"; return 1; }
    digest="$(printf '%s' "$manifest" | grep -oE '"layers":\[.*' | grep -oE 'sha256:[0-9a-f]{64}' | head -n 1)"
    size="$(printf '%s' "$manifest" | grep -oE '"layers":\[.*' | grep -oE '"size":[0-9]+' | head -n 1 | grep -oE '[0-9]+')"
    [ -n "$digest" ] && [ -n "$size" ] || { say "the image's manifest is not in the expected form"; return 1; }
    part="$IMAGE_DIR/.lexichron-${digest#sha256:}.part"
    have=0; [ -f "$part" ] && have="$(wc -c < "$part" | tr -d ' ')"
    if [ "$have" -gt 0 ] && [ "$have" -lt "$size" ]; then
        say "Resuming an earlier download at $((have / 1048576)) of $((size / 1048576)) MiB"
    fi
    while [ "$have" -lt "$size" ]; do
        tries=$((tries + 1))
        # a fresh token each time: one lasts minutes, a slow download hours
        token="$(curl -fsSL "https://ghcr.io/token?scope=repository:$REPO:pull" | sed -E 's/.*"token":"([^"]+)".*/\1/')" || true
        # HTTP/1.1: the cut-offs seen in the field were HTTP/2 stream resets.
        # A connection that stalls for a minute is dropped and resumed.
        curl -fL --http1.1 -C - --speed-limit 1024 --speed-time 60 --progress-bar \
             -H "Authorization: Bearer $token" -o "$part" "$api/blobs/$digest" || true
        have=0; [ -f "$part" ] && have="$(wc -c < "$part" | tr -d ' ')"
        [ "$have" -ge "$size" ] && break
        if [ "$have" -le "$last" ]; then stalls=$((stalls + 1)); else stalls=0; fi
        last="$have"
        if [ "$stalls" -ge 5 ]; then
            say "No progress in 5 attempts; $((have / 1048576)) of $((size / 1048576)) MiB are kept in $part."
            say "Run the installer again to continue from there."
            return 1
        fi
        say "Connection lost at $((have / 1048576)) of $((size / 1048576)) MiB; resuming (attempt $((tries + 1)))..."
        sleep 5
    done
    if [ "$have" -ne "$size" ]; then
        say "the downloaded file is $have bytes, not the $size the registry lists; removing it"
        rm -f "$part"; return 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        say "Checking the download against the registry's SHA-256..."
        if [ "sha256:$(sha256sum "$part" | awk '{print $1}')" != "$digest" ]; then
            say "the downloaded file does not match the registry's SHA-256; removing it"
            rm -f "$part"; return 1
        fi
    fi
    mv -f "$part" "$TMP_SIF"
}

if [ -n "${LEXICHRON_SIF:-}" ]; then
    [ -f "$LEXICHRON_SIF" ] || die "LEXICHRON_SIF=$LEXICHRON_SIF: no such file"
    say "Installing from $LEXICHRON_SIF"
    cp -f "$LEXICHRON_SIF" "$TMP_SIF" || die "could not copy $LEXICHRON_SIF into $IMAGE_DIR"
elif [ "${LEXICHRON_FETCH:-}" = "curl" ] || ls "$IMAGE_DIR"/.lexichron-*.part >/dev/null 2>&1; then
    # asked for, or an earlier run left a partial download to continue
    say "Downloading $REF with curl (resumable)..."
    fetch_with_curl || die "the download did not finish. Run the installer again to continue it, or copy an image file from another machine and install it with LEXICHRON_SIF=/path/to/lexichron-X.Y.Z.sif"
else
    say "Pulling oras://$REF ..."
    if ! "$RUNTIME" pull "$TMP_SIF" "oras://$REF"; then
        rm -f "$TMP_SIF"
        say ""
        say "The $RUNTIME download was cut off. Fetching the same image with curl, which resumes"
        say "where a broken connection leaves off..."
        fetch_with_curl || die "the download did not finish. Run the installer again to continue it, or copy an image file from another machine and install it with LEXICHRON_SIF=/path/to/lexichron-X.Y.Z.sif"
    fi
fi

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

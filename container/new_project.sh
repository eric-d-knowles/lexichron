#!/usr/bin/env bash
# new_project.sh
#
# Create a project environment on top of this lexichron image: a Python
# virtual environment in the project directory that inherits lexichron and all
# of its dependencies from the image, and into which you can install your own
# packages. A Jupyter kernel for the project is registered at the same time.
#
# This script is shipped INSIDE the image and is run through Apptainer:
#
#   apptainer run --app new-project lexichron-0.2.0.sif /path/to/project [options]
#
# Options:
#   --name NAME    Kernel identifier (default: "<project>-lexichron-0.2.0").
#   --display NAME Display name in the kernel menu
#                  (default: "Python (<project> | lexichron 0.2.0)").
#   --bind DIR     Additional host directory to make visible inside the
#                  container when the kernel runs. May be repeated.
#
# Afterwards:
#   - In a notebook using the project's kernel, install packages with
#         %pip install <package>
#   - From a shell:
#         apptainer exec lexichron-0.2.0.sif /path/to/project/.venv/bin/pip install <package>
#   - Record the environment with
#         apptainer exec lexichron-0.2.0.sif /path/to/project/.venv/bin/pip freeze --local
#
# The environment lives in <project>/.venv and is tied to this image's Python
# version. If a future image moves to a new Python minor version, run this
# script again and reinstall your packages from the frozen list.

set -euo pipefail

usage() { sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[ $# -ge 1 ] || usage

PROJECT_DIR="$1"; shift
PASSTHROUGH=()
KERNEL_NAME=""
DISPLAY_NAME=""

while [ $# -gt 0 ]; do
    case "$1" in
        --name)    KERNEL_NAME="$2"; shift 2 ;;
        --display) DISPLAY_NAME="$2"; shift 2 ;;
        --bind)    PASSTHROUGH+=("--bind" "$2"); shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

IMAGE="${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}"
if [ -z "$IMAGE" ]; then
    echo "Error: this script must be run inside the lexichron image:"
    echo "  apptainer run --app new-project lexichron-<version>.sif /path/to/project"
    exit 1
fi

# ---------------------------------------------------------------------------
# Project directory: must be on a host filesystem visible in the container.
# ---------------------------------------------------------------------------
mkdir -p "$PROJECT_DIR"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd -P)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV="$PROJECT_DIR/.venv"

BASENAME="$(basename "$IMAGE" .sif)"
VERSION="${BASENAME#lexichron-}"
[ -n "$KERNEL_NAME" ]  || KERNEL_NAME="${PROJECT_NAME}-${BASENAME}"
[ -n "$DISPLAY_NAME" ] || DISPLAY_NAME="Python (${PROJECT_NAME} | lexichron ${VERSION})"

# ---------------------------------------------------------------------------
# Virtual environment, inheriting the image's site-packages.
# ---------------------------------------------------------------------------
IMAGE_PY="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [ -f "$VENV/pyvenv.cfg" ]; then
    VENV_PY="$(sed -n 's/^version *= *\([0-9]*\.[0-9]*\).*/\1/p' "$VENV/pyvenv.cfg")"
    if [ "$VENV_PY" != "$IMAGE_PY" ]; then
        echo "Error: $VENV was created with Python $VENV_PY; this image has Python $IMAGE_PY."
        echo "Move it aside (mv \"$VENV\" \"$VENV.py$VENV_PY\"), run this again, and reinstall"
        echo "your packages (a 'pip freeze' list from the old environment helps)."
        exit 1
    fi
    echo "Using existing environment: $VENV"
else
    echo "Creating environment: $VENV"
    python -m venv --system-site-packages "$VENV"
fi

# Record which image created it, for humans reading the directory later.
cat > "$VENV/LEXICHRON_IMAGE" <<EOF
image=$IMAGE
python=$(python -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# Sanity: the venv sees lexichron through the image.
"$VENV/bin/python" -c "import ngramprep" || {
    echo "Error: the new environment cannot import lexichron. This should not happen."
    exit 1
}

# ---------------------------------------------------------------------------
# Kernel for this project. Make sure the project directory itself is visible
# at kernel start (register_kernel.sh skips it if a site bind already covers it).
# ---------------------------------------------------------------------------
bash /opt/lexichron/register_kernel.sh \
    --python "$VENV/bin/python" \
    --name "$KERNEL_NAME" \
    --display "$DISPLAY_NAME" \
    --bind "$PROJECT_DIR" \
    "${PASSTHROUGH[@]}"

echo ""
echo "Project environment ready."
echo "  Install packages from a notebook on this kernel:  %pip install <package>"
echo "  ...or from a shell:  apptainer exec $(basename "$IMAGE") $VENV/bin/pip install <package>"

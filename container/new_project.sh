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
#   --env K=V      Environment variable set inside the container for this
#                  project (e.g. --env CMDSTAN=/path/to/cmdstan). May be repeated.
#   --no-install   Do not install <project>/requirements.txt even if present.
#
# Creates:
#   <project>/.venv/              the environment (inherits the image's packages)
#   <project>/.venv/host-python   a launcher that runs the project's Python inside
#                                 the image; callable by path from the host, e.g.
#                                 in Slurm scripts:  .venv/host-python script.py
#
# Afterwards:
#   - In a notebook using the project's kernel, install packages with
#         %pip install <package>
#   - From a shell:
#         /path/to/project/.venv/host-python -m pip install <package>
#   - Record the environment with
#         /path/to/project/.venv/host-python -m pip freeze --local
#   If <project>/requirements.txt exists it is installed automatically.
#
# The environment lives in <project>/.venv and is tied to this image's Python
# version. If a future image moves to a new Python minor version, run this
# script again and reinstall your packages from the frozen list.

set -euo pipefail

usage() { sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[ $# -ge 1 ] || usage

PROJECT_DIR="$1"; shift
PASSTHROUGH=()
KERNEL_NAME=""
DISPLAY_NAME=""
INSTALL_REQS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --name)    KERNEL_NAME="$2"; shift 2 ;;
        --display) DISPLAY_NAME="$2"; shift 2 ;;
        --bind)    PASSTHROUGH+=("--bind" "$2"); shift 2 ;;
        --env)     PASSTHROUGH+=("--env" "$2"); shift 2 ;;
        --no-install) INSTALL_REQS=0; shift ;;
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
    --launcher "$VENV/host-python" \
    --name "$KERNEL_NAME" \
    --display "$DISPLAY_NAME" \
    --bind "$PROJECT_DIR" \
    "${PASSTHROUGH[@]}"

# ---------------------------------------------------------------------------
# Project requirements, if any. pip runs inside this container; the venv is on
# the host filesystem, so the packages persist.
# ---------------------------------------------------------------------------
if [ "$INSTALL_REQS" = 1 ] && [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo ""
    echo "Installing $PROJECT_DIR/requirements.txt into the project environment..."
    if ! "$VENV/bin/python" -m pip install --quiet -r "$PROJECT_DIR/requirements.txt"; then
        echo "Warning: requirements install failed (no network on this node?)."
        echo "Re-run from a node with internet access:"
        echo "  $VENV/host-python -m pip install -r $PROJECT_DIR/requirements.txt"
    fi
fi

# ---------------------------------------------------------------------------
# UI launcher (host side): runs the Slurm host bridge alongside the UI so that
# the UI can submit and monitor jobs; stops the bridge when the UI exits.
# ---------------------------------------------------------------------------
UI_LAUNCHER="$VENV/lexichron-ui"
cat > "$UI_LAUNCHER" <<EOF
#!/usr/bin/env bash
# Generated by lexichron new-project. Starts the host bridge (so the UI can run
# sbatch/squeue on this machine) and the lexichron UI inside the image.
#   $UI_LAUNCHER [project.yaml] [--stage acquire]
set -u
PROJECT_DIR=$(printf '%q' "$PROJECT_DIR")
BRIDGE_DIR="\$PROJECT_DIR/.lexichron/bridge"
mkdir -p "\$BRIDGE_DIR"
# The bridge script is inside the image; copy it out once so it can run on the host.
BRIDGE_SH="\$PROJECT_DIR/.lexichron/host_bridge.sh"
[ -x "\$BRIDGE_SH" ] || "$VENV/host-python" -c "import shutil; shutil.copy('/opt/lexichron/host_bridge.sh', '\$BRIDGE_SH')" && chmod 755 "\$BRIDGE_SH"
bash "\$BRIDGE_SH" "\$BRIDGE_DIR" &
BRIDGE_PID=\$!
trap 'touch "\$BRIDGE_DIR/stop"; kill \$BRIDGE_PID 2>/dev/null' EXIT
args=("\$@"); [ \${#args[@]} -gt 0 ] || args=("\$PROJECT_DIR/project.yaml")
LEXICHRON_BRIDGE_DIR="\$BRIDGE_DIR" "$VENV/host-python" -m lexichron.cli ui "\${args[@]}"
EOF
chmod 755 "$UI_LAUNCHER"

echo ""
echo "Project environment ready."
echo "  Install packages from a notebook on this kernel:  %pip install <package>"
echo "  ...or from a shell:  $VENV/host-python -m pip install <package>"
echo "  Run a script inside the environment:  $VENV/host-python script.py"
echo "  Open the terminal UI (with Slurm submit): $UI_LAUNCHER"

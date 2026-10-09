#!/usr/bin/env bash
# register_kernel.sh
#
# Register this lexichron container image as a Jupyter kernel, so the
# notebooks in notebooks/ can be run against the image from JupyterLab,
# Positron or VS Code without installing anything in a conda environment.
#
# This script is shipped INSIDE the image and is run through Apptainer:
#
#   apptainer run --app register-kernel lexichron-0.2.0.sif [options]
#
# Options:
#   --bind DIR     Additional host directory to make visible inside the
#                  container. May be repeated. Common cluster data roots
#                  (/scratch, /vast, /gpfs, /work, /project, /projects, /data)
#                  are bound automatically when they exist on the host and
#                  are not already bound by the site configuration.
#   --name NAME    Kernel identifier (default: derived from the image file
#                  name, e.g. "lexichron-0.2.0").
#   --display NAME Display name in the kernel menu
#                  (default: "Python (lexichron 0.2.0)").
#   --python PATH  Python interpreter to run inside the container (default:
#                  the image's own). Used by the new-project app to point a
#                  kernel at a project's virtual environment.
#
# Afterwards, restart Jupyter (or reload the VS Code / Positron window) and
# pick the kernel from the kernel menu.

set -euo pipefail

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

EXTRA_BINDS=()
KERNEL_NAME=""
DISPLAY_NAME=""
PYTHON="python"

while [ $# -gt 0 ]; do
    case "$1" in
        --bind)    EXTRA_BINDS+=("$2"); shift 2 ;;
        --name)    KERNEL_NAME="$2"; shift 2 ;;
        --display) DISPLAY_NAME="$2"; shift 2 ;;
        --python)  PYTHON="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Which image are we running from? Apptainer sets this for every container.
# ---------------------------------------------------------------------------
IMAGE="${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}"
if [ -z "$IMAGE" ]; then
    echo "Error: this script must be run inside the lexichron image:"
    echo "  apptainer run --app register-kernel lexichron-<version>.sif"
    exit 1
fi

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
BASENAME="$(basename "$IMAGE" .sif)"                 # lexichron-0.2.0
VERSION="${BASENAME#lexichron-}"                     # 0.2.0 (or the basename if no prefix)
[ -n "$KERNEL_NAME" ]  || KERNEL_NAME="$BASENAME"
[ -n "$DISPLAY_NAME" ] || DISPLAY_NAME="Python (lexichron ${VERSION})"

# ---------------------------------------------------------------------------
# Bind mounts. Paths the site binds system-wide are visible in /proc/mounts
# right now; re-binding those only produces a warning at every kernel start,
# so they are recorded and skipped. Everything else is decided on the host at
# kernel start, by the small wrapper written into the kernelspec, so a data
# root that exists on the host is bound and one that does not is ignored.
# ---------------------------------------------------------------------------
SYSTEM_BOUND=""
MOUNTS="$(awk '{print $2}' /proc/mounts)"
for d in /scratch /vast /gpfs /work /project /projects /data "${EXTRA_BINDS[@]}"; do
    # Already visible if it, or a parent of it, is a mount point in here.
    p="$d"
    while [ "$p" != "/" ]; do
        if printf '%s\n' "$MOUNTS" | grep -qx "$p"; then
            SYSTEM_BOUND="$SYSTEM_BOUND $d"
            break
        fi
        p="$(dirname "$p")"
    done
done

CANDIDATES=""
for d in /scratch /vast /gpfs /work /project /projects /data "${EXTRA_BINDS[@]}"; do
    case " $CANDIDATES " in *" $d "*) continue ;; esac
    CANDIDATES="$CANDIDATES $d"
done
CANDIDATES="${CANDIDATES# }"

# ---------------------------------------------------------------------------
# Write the kernelspec
# ---------------------------------------------------------------------------
KERNEL_DIR="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/${KERNEL_NAME}"
mkdir -p "$KERNEL_DIR"

python3 - "$KERNEL_DIR/kernel.json" "$IMAGE" "$DISPLAY_NAME" "$CANDIDATES" "$SYSTEM_BOUND" "$PYTHON" <<'EOF'
import json, sys
out, image, display, candidates, system_bound, python = sys.argv[1:]

# Runs on the HOST each time the kernel starts.
launcher = f"""
img={json.dumps(image)}
py={json.dumps(python)}
binds=""
for d in {candidates}; do
  case " {system_bound} " in *" $d "*) continue ;; esac
  [ -d "$d" ] && binds="$binds --bind $d"
done
exec apptainer exec $binds "$img" "$py" -m ipykernel_launcher -f "$1"
""".strip()

spec = {
    "argv": ["bash", "-c", launcher, "lexichron-kernel", "{connection_file}"],
    "display_name": display,
    "language": "python",
    "metadata": {"debugger": True},
}
with open(out, "w") as fh:
    json.dump(spec, fh, indent=2)
    fh.write("\n")
EOF

echo "Registered kernel '${KERNEL_NAME}' -> ${KERNEL_DIR}/kernel.json"
echo "  image:          ${IMAGE}"
echo "  python:         ${PYTHON}"
echo "  display name:   ${DISPLAY_NAME}"
echo "  bound by site:  ${SYSTEM_BOUND:-(none)}"
echo "  bound on start: ${CANDIDATES} (whichever exist on this machine)"
echo ""
echo "Restart Jupyter (or reload the window in Positron/VS Code) and select"
echo "'${DISPLAY_NAME}' from the kernel menu."
echo ""
echo "To remove it later: jupyter kernelspec remove ${KERNEL_NAME}"

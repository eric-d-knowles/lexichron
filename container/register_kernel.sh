#!/usr/bin/env bash
# register_kernel.sh
#
# Register a lexichron container image as a Jupyter kernel, so the notebooks
# in notebooks/ can be run against the image from JupyterLab, Positron or
# VS Code without installing anything in a conda environment.
#
# Usage:
#   bash register_kernel.sh /path/to/lexichron-0.2.0.sif [--bind /dir ...] [--name NAME]
#
# Options:
#   --bind DIR     Additional host directory to make visible inside the
#                  container. May be repeated. Common cluster data roots
#                  (/scratch, /vast, /gpfs, /work, /project, /data) are bound
#                  automatically if they exist on this machine.
#   --name NAME    Kernel identifier (default: derived from the image file
#                  name, e.g. "lexichron-0.2.0").
#   --display NAME Display name in the kernel menu
#                  (default: "Python (lexichron 0.2.0)").
#
# Afterwards, restart Jupyter and pick the kernel from the kernel menu.

set -euo pipefail

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

[ $# -ge 1 ] || usage

IMAGE="$1"; shift
EXTRA_BINDS=()
KERNEL_NAME=""
DISPLAY_NAME=""

while [ $# -gt 0 ]; do
    case "$1" in
        --bind)    EXTRA_BINDS+=("$2"); shift 2 ;;
        --name)    KERNEL_NAME="$2"; shift 2 ;;
        --display) DISPLAY_NAME="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
if ! command -v apptainer >/dev/null 2>&1; then
    echo "Error: 'apptainer' not found on PATH. On many clusters: module load apptainer"
    exit 1
fi

IMAGE="$(readlink -f "$IMAGE")"
if [ ! -f "$IMAGE" ]; then
    echo "Error: image not found: $IMAGE"
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
# Bind mounts: $HOME is bound by Apptainer automatically; add data roots.
# ---------------------------------------------------------------------------
# Paths the site already binds system-wide (apptainer.conf "bind path") are
# skipped, since re-binding them only produces a warning on every start.
SYSTEM_BINDS="$(apptainer exec --no-home --contain "$IMAGE" cat /proc/mounts 2>/dev/null | awk '{print $2}' || true)"

BINDS=()
for d in /scratch /vast /gpfs /work /project /projects /data "${EXTRA_BINDS[@]}"; do
    [ -d "$d" ] || continue
    case " ${BINDS[*]-} " in *" $d "*) continue ;; esac
    if printf '%s\n' "$SYSTEM_BINDS" | grep -qx "$d"; then
        echo "  (skipping $d: already bound by the site configuration)"
        continue
    fi
    BINDS+=("$d")
done

BIND_ARGS=()
for d in "${BINDS[@]}"; do BIND_ARGS+=("--bind" "$d"); done

# ---------------------------------------------------------------------------
# Write the kernelspec
# ---------------------------------------------------------------------------
KERNEL_DIR="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/${KERNEL_NAME}"
mkdir -p "$KERNEL_DIR"

python3 - "$KERNEL_DIR/kernel.json" "$IMAGE" "$DISPLAY_NAME" "${BIND_ARGS[@]}" <<'EOF'
import json, sys
out, image, display, *bind_args = sys.argv[1:]
argv = ["apptainer", "exec", *bind_args, image,
        "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
spec = {
    "argv": argv,
    "display_name": display,
    "language": "python",
    "metadata": {"debugger": True},
}
with open(out, "w") as fh:
    json.dump(spec, fh, indent=2)
    fh.write("\n")
EOF

echo "Registered kernel '${KERNEL_NAME}' -> ${KERNEL_DIR}/kernel.json"
echo "  image:        ${IMAGE}"
echo "  display name: ${DISPLAY_NAME}"
echo "  bind mounts:  ${BINDS[*]:-(none beyond \$HOME)}"
echo ""
echo "Restart Jupyter (or reload the window in Positron/VS Code) and select"
echo "'${DISPLAY_NAME}' from the kernel menu."
echo ""
echo "To remove it later: jupyter kernelspec remove ${KERNEL_NAME}"

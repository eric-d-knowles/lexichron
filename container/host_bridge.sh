#!/usr/bin/env bash
# host_bridge.sh — run a few Slurm commands on behalf of a process inside the
# container, which cannot see the host's sbatch/squeue.
#
#   host_bridge.sh BRIDGE_DIR
#
# Protocol (file based, so it needs nothing but a shared directory):
#   request:  BRIDGE_DIR/req/<id>.json   {"cmd": "sbatch", "args": ["path.sbatch"]}
#   response: BRIDGE_DIR/res/<id>.json   {"rc": 0, "stdout": "...", "stderr": "..."}
# Only the commands listed in ALLOWED are run. The loop exits when
# BRIDGE_DIR/stop exists or its parent process goes away.
set -u

BRIDGE_DIR="${1:?usage: host_bridge.sh BRIDGE_DIR}"
ALLOWED="sbatch squeue sacct scancel sinfo"
PARENT=$PPID

mkdir -p "$BRIDGE_DIR/req" "$BRIDGE_DIR/res"
echo $$ > "$BRIDGE_DIR/bridge.pid"
rm -f "$BRIDGE_DIR/stop"
trap 'rm -f "$BRIDGE_DIR/bridge.pid"; exit 0' TERM INT HUP

json_escape() {  # stdin -> JSON string literal
    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
      || { printf '"'; sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | awk '{printf "%s\\n", $0}'; printf '"'; }
}

while :; do
    [ -e "$BRIDGE_DIR/stop" ] && break
    kill -0 "$PARENT" 2>/dev/null || break

    for req in "$BRIDGE_DIR"/req/*.json; do
        [ -e "$req" ] || continue
        id="$(basename "$req" .json)"
        # Parse with python (present on every HPC login node); fall back to refusing.
        mapfile -t parsed < <(python3 - "$req" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("cmd", ""))
for a in d.get("args", []):
    print(a)
PY
) || parsed=()
        cmd="${parsed[0]:-}"
        args=("${parsed[@]:1}")
        rm -f "$req"

        out="$BRIDGE_DIR/res/$id.json"
        tmp="$out.tmp"
        case " $ALLOWED " in
            *" $cmd "*)
                stdout="$("$cmd" "${args[@]}" 2>"$BRIDGE_DIR/res/$id.err")"; rc=$?
                stderr="$(cat "$BRIDGE_DIR/res/$id.err")"; rm -f "$BRIDGE_DIR/res/$id.err"
                ;;
            *)
                rc=126; stdout=""; stderr="command not allowed by bridge: '$cmd' (allowed: $ALLOWED)"
                ;;
        esac
        {
            printf '{"rc": %d, "stdout": ' "$rc"; printf '%s' "$stdout" | json_escape
            printf ', "stderr": ';                 printf '%s' "$stderr" | json_escape
            printf '}\n'
        } > "$tmp"
        mv -f "$tmp" "$out"
    done
    sleep 0.3
done
rm -f "$BRIDGE_DIR/bridge.pid"

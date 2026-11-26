"""The --env and --bind settings of a launcher register_kernel.sh wrote, less
those a new-project run is given again; new_project.sh keeps the rest.

    python3 launcher_settings.py LAUNCHER PROJECT_DIR [env=K=V | bind=DIR ...]

Prints one ``env=K=V`` or ``bind=DIR`` per line, in the launcher's order. An
env is dropped when the arguments name the same variable (the new value
wins), a bind when the arguments name the same directory. The standard data
roots and the project directory are bound by every launcher, so they are not
printed. Reads the launcher's two generated lines:

    for d in /scratch /vast ... EXTRA; do
    exec "$runtime" exec$nv $binds --env K=V ... "$img" "$py" "$@"
"""
import re
import shlex
import sys

STANDARD_ROOTS = {"/scratch", "/vast", "/gpfs", "/work", "/project", "/projects", "/data"}


def settings(text, project_dir):
    """([K=V, ...], [DIR, ...]) of the launcher ``text``."""
    envs, binds = [], []
    m = re.search(r"^exec \"\$runtime\" exec\$nv \$binds(.*?) \"\$img\" \"\$py\" \"\$@\"$", text, re.M)
    if m:
        words = shlex.split(m.group(1))
        envs = [words[i + 1] for i in range(len(words) - 1) if words[i] == "--env"]
    m = re.search(r"^for d in (.*); do$", text, re.M)
    if m:
        binds = [d for d in shlex.split(m.group(1)) if d not in STANDARD_ROOTS and d != project_dir]
    return envs, binds


def kept(text, project_dir, given):
    envs, binds = settings(text, project_dir)
    given_names = {g[4:].split("=", 1)[0] for g in given if g.startswith("env=")}
    given_dirs = {g[5:].rstrip("/") for g in given if g.startswith("bind=")}
    return ([f"env={kv}" for kv in envs if kv.split("=", 1)[0] not in given_names]
            + [f"bind={d}" for d in binds if d.rstrip("/") not in given_dirs])


if __name__ == "__main__":
    launcher, project_dir, *given = sys.argv[1:]
    with open(launcher) as fh:
        print("\n".join(kept(fh.read(), project_dir, given)))

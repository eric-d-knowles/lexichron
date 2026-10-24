"""The host-side installers, exercised with a fake `apptainer` on PATH."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_UI = ROOT / "container" / "install_ui.sh"
INSTALL_SH = ROOT / "install.sh"
BRIDGE = ROOT / "container" / "host_bridge.sh"
if not INSTALL_UI.exists():                      # running inside the image
    INSTALL_UI = Path("/opt/lexichron/install_ui.sh")
    INSTALL_SH = Path("/opt/lexichron/install.sh")
    BRIDGE = Path("/opt/lexichron/host_bridge.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fake_apptainer(bin_dir: Path, log: Path, install_ui_copy: Path) -> None:
    """A stand-in that records calls, fakes pull/exec/run just enough."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = f"""#!/usr/bin/env bash
echo "$@" >> "{log}"
case "$1" in
  pull)  touch "$2"; exit 0 ;;                              # pull DEST oras://REF
  exec)  shift; img="$1"; shift
         if [ "$1 $2" = "lexichron --version" ]; then echo "lexichron 0.1.0"; exit 0; fi
         echo "exec: $*"; exit 0 ;;
  run)   # run --app install IMG args...  -> run install_ui.sh "inside" the image
         shift; [ "$1" = "--app" ] && shift && app="$1" && shift
         img="$1"; shift
         [ -z "${{BASH_ENV:-}}" ] || {{ echo "BASH_ENV leaked into the image" >&2; exit 3; }}
         if [ -n "${{FAKE_OLD_IMAGE:-}}" ]; then        # a 0.7 image: launchers that call apptainer by name
           bin="$HOME/.local/bin"; mkdir -p "$bin"
           printf '%s\n' '#!/usr/bin/env bash' 'set -u' 'img=x' 'binds=""' 'if ! command -v apptainer >/dev/null 2>&1; then' \
             '  echo "lexichron: '"'"'apptainer'"'"' not found; try: module load apptainer" >&2; exit 127' 'fi' \
             '[ -f "$img" ] || apptainer pull "$img" "oras://ref"' 'exec apptainer exec $binds "$img" lexichron "$@"' > "$bin/lexichron"
           printf '%s\n' '#!/usr/bin/env bash' 'set -u' 'if ! command -v apptainer >/dev/null 2>&1; then exit 127; fi' \
             'apptainer exec $binds --env X=1 "$img" lexichron ui "$@"' > "$bin/lexichron-ui"
           chmod 755 "$bin/lexichron" "$bin/lexichron-ui"; exit 0
         fi
         env -u LEXICHRON_RUNTIME_SEEN APPTAINER_CONTAINER="$img" bash "{install_ui_copy}" "$@"; exit $? ;;
esac
exit 1
"""
    (bin_dir / "apptainer").write_text(script)
    (bin_dir / "apptainer").chmod(0o755)
    # the same stand-in under Singularity's name, kept aside until a test wants it
    (bin_dir.parent / "singularity-only" / "singularity").parent.mkdir(parents=True, exist_ok=True)
    (bin_dir.parent / "singularity-only" / "singularity").write_text(script.replace("APPTAINER_CONTAINER=", "SINGULARITY_CONTAINER="))
    (bin_dir.parent / "singularity-only" / "singularity").chmod(0o755)


def _path_without_runtimes() -> str:
    """This machine's PATH minus any directory that holds apptainer or
    singularity, so the fake runtime is the only one the scripts can find
    (a cluster with a system-wide apptainer would otherwise shadow it)."""
    keep = []
    for d in os.environ["PATH"].split(os.pathsep):
        if d and not any((Path(d) / r).exists() for r in ("apptainer", "singularity")):
            keep.append(d)
    return os.pathsep.join(keep)


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"; home.mkdir()
    (home / ".bashrc").write_text("# rc\n")
    # install_ui.sh copies /opt/lexichron/host_bridge.sh; point a copy at a temp path
    ui_copy = tmp_path / "install_ui.sh"
    ui_copy.write_text(INSTALL_UI.read_text().replace("/opt/lexichron/host_bridge.sh", str(BRIDGE)))
    fake_bin = tmp_path / "fakebin"
    log = tmp_path / "calls.log"
    _fake_apptainer(fake_bin, log, ui_copy)
    base_path = _path_without_runtimes()
    e = dict(os.environ, HOME=str(home), PATH=f"{fake_bin}:{base_path}",
             SCRATCH=str(tmp_path / "scratch"), USER="tester")
    e.pop("APPTAINER_CONTAINER", None); e.pop("SINGULARITY_CONTAINER", None)
    # A cluster may set BASH_ENV to a profile script that every non-interactive
    # bash sources before running; here it would put the real apptainer back
    # on the PATH ahead of the fake one. The launchers unset it for the image
    # themselves; the tests must not inherit it either.
    e.pop("BASH_ENV", None); e.pop("ENV", None)
    (tmp_path / "scratch").mkdir()
    return {"home": home, "env": e, "log": log, "tmp": tmp_path, "ui_copy": ui_copy, "base_path": base_path}


def test_install_ui_writes_both_commands(env):
    img = env["tmp"] / "images" / "lexichron-0.1.0.sif"; img.parent.mkdir(); img.touch()
    e = dict(env["env"], APPTAINER_CONTAINER=str(img))
    r = subprocess.run(["bash", str(env["ui_copy"]), "--bind", "/opt"], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    bin_dir = env["home"] / ".local" / "bin"
    for name in ("lexichron", "lexichron-ui"):
        p = bin_dir / name
        assert p.exists() and p.stat().st_mode & stat.S_IXUSR
        text = p.read_text()
        assert str(img) in text and "ghcr.io/eric-d-knowles/lexichron:0.1.0" in text
        assert '"$runtime" pull' in text                    # re-pull when purged
        assert "/opt" in text                                # extra bind carried over
    assert (env["home"] / ".lexichron" / "host_bridge.sh").exists()
    assert 'exec "$runtime" exec' in (bin_dir / "lexichron").read_text()
    ui = (bin_dir / "lexichron-ui").read_text()
    assert "exec apptainer" not in ui                                       # trap must run
    # one bridge directory per session, named by host and launcher pid
    assert 'BRIDGE_DIR="$HOME/.lexichron/bridge/$host-$$"' in ui and "kill -0" in ui
    assert "rm -rf \"$BRIDGE_DIR\"" in ui                                  # removed on exit


def test_lexichron_wrapper_runs_in_image_and_repulls_when_missing(env):
    img = env["tmp"] / "images" / "lexichron-0.1.0.sif"; img.parent.mkdir(); img.touch()
    e = dict(env["env"], APPTAINER_CONTAINER=str(img))
    subprocess.run(["bash", str(env["ui_copy"])], env=e, check=True, capture_output=True)
    wrapper = env["home"] / ".local" / "bin" / "lexichron"

    r = subprocess.run(["bash", str(wrapper), "acquire", "x.yaml"], env=env["env"], capture_output=True, text=True)
    assert r.returncode == 0 and "exec: lexichron acquire x.yaml" in r.stdout

    img.unlink()                                             # simulate a purge
    r = subprocess.run(["bash", str(wrapper), "--version"], env=env["env"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "pulling oras://ghcr.io/eric-d-knowles/lexichron:0.1.0" in r.stderr
    assert img.exists()                                      # fake pull recreated it


def test_install_sh_end_to_end(env):
    r = subprocess.run(["bash", str(INSTALL_SH)], env=env["env"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    scratch = env["tmp"] / "scratch" / "lexichron"
    assert (scratch / "images" / "lexichron-0.1.0.sif").exists()      # renamed to real version
    assert not list((scratch / "images").glob(".lexichron-download-*"))
    calls = env["log"].read_text()
    assert "pull" in calls and "oras://ghcr.io/eric-d-knowles/lexichron:latest" in calls
    assert "--app install" in calls and "--ref ghcr.io/eric-d-knowles/lexichron:0.1.0" in calls
    bin_dir = env["home"] / ".local" / "bin"
    assert (bin_dir / "lexichron").exists() and (bin_dir / "lexichron-ui").exists()
    rc = (env["home"] / ".bashrc").read_text()
    assert str(bin_dir) in rc and rc.count("export PATH=") == 1        # PATH added
    assert rc.strip().endswith("esac")                                  # the guarded form
    assert "lexichron 0.1.0 is installed" in r.stdout


def test_install_sh_path_line_does_not_shadow_an_activated_environment(env):
    """Reading ~/.bashrc again (a job shell started from an activated login
    shell) must not move ~/.local/bin ahead of the environment; and an
    earlier install's unconditional line is replaced, not duplicated."""
    bin_dir = env["home"] / ".local" / "bin"
    rc = env["home"] / ".bashrc"
    rc.write_text(f'# rc\nexport PATH="{bin_dir}:$PATH"\n')
    r = subprocess.run(["bash", str(INSTALL_SH)], env=env["env"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    text = rc.read_text()
    assert text.count("export PATH=") == 1 and text.count("esac") == 1
    assert "Updated the PATH line" in r.stdout
    envdir = env["tmp"] / "env" / "bin"; envdir.mkdir(parents=True)
    probe = subprocess.run(
        ["bash", "-c", f'export PATH="{envdir}:$PATH"; . "{rc}"; . "{rc}"; echo "$PATH"'],
        env={**env["env"], "PATH": f"{bin_dir}:{env['base_path']}"}, capture_output=True, text=True)
    assert probe.stdout.strip().split(":")[0] == str(envdir), probe.stdout


def test_install_sh_works_with_singularity_instead_of_apptainer(env):
    """A cluster that only has `singularity` (NYUAD's Jubail, say): the
    installer uses it, and the launchers it writes run the image with it."""
    e = dict(env["env"], PATH=f"{env['tmp'] / 'singularity-only'}:{env['base_path']}")   # no fake apptainer
    assert shutil.which("apptainer", path=e["PATH"]) is None
    r = subprocess.run(["bash", str(INSTALL_SH)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "Container runtime: singularity" in r.stdout
    calls = env["log"].read_text()
    assert "pull" in calls and "--runtime" not in calls                 # the runtime travels as a variable
    bin_dir = env["home"] / ".local" / "bin"
    launcher = (bin_dir / "lexichron").read_text()
    assert "runtime=singularity" in launcher and "SINGULARITY_CACHEDIR=" in launcher
    assert "unset BASH_ENV ENV" in launcher                             # modules' init must not reach the image
    env["log"].write_text("")
    r = subprocess.run([str(bin_dir / "lexichron"), "--help"], env=e, capture_output=True, text=True)
    assert r.returncode == 0 and "exec: lexichron --help" in r.stdout
    # a launcher written for apptainer still runs where only singularity exists
    swapped = launcher.replace("runtime=singularity", "runtime=apptainer")
    (bin_dir / "lexichron").write_text(swapped)
    r = subprocess.run([str(bin_dir / "lexichron"), "--help"], env=e, capture_output=True, text=True)
    assert r.returncode == 0 and "exec: lexichron --help" in r.stdout
    # and neither at all is a clear message, not a stack of errors
    bare = dict(e, PATH=env["base_path"])
    r = subprocess.run([str(bin_dir / "lexichron"), "--help"], env=bare, capture_output=True, text=True)
    assert r.returncode == 127 and "neither 'apptainer' nor 'singularity'" in r.stderr


def test_install_sh_copes_with_modules_bash_env_and_older_images(env):
    """environment-modules exports BASH_ENV (its init needs the host's tclsh,
    which the image lacks): the installer must not pass it in. And an image
    from before 0.8 writes launchers that call apptainer by name: on a
    singularity-only cluster the installer points them at singularity."""
    e = dict(env["env"], PATH=f"{env['tmp'] / 'singularity-only'}:{env['base_path']}",
             BASH_ENV="/share/apps/Modules/init/bash", FAKE_OLD_IMAGE="1")
    r = subprocess.run(["bash", str(INSTALL_SH)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    bin_dir = env["home"] / ".local" / "bin"
    launcher = (bin_dir / "lexichron").read_text()
    assert "apptainer pull" not in launcher and "exec singularity exec" in launcher
    assert "command -v singularity" in launcher and "'singularity' not found; try: module load singularity" in launcher
    assert launcher.count("unset BASH_ENV ENV") == 1 and launcher.index("set -u") < launcher.index("unset BASH_ENV")
    ui = (bin_dir / "lexichron-ui").read_text()
    assert "singularity exec $binds --env X=1" in ui
    # the fixed launcher runs with singularity, and keeps BASH_ENV from the image
    env["log"].write_text("")
    r = subprocess.run([str(bin_dir / "lexichron"), "--help"], env=dict(e, FAKE_OLD_IMAGE=""), capture_output=True, text=True)
    assert r.returncode == 0 and "exec: lexichron --help" in r.stdout, r.stderr

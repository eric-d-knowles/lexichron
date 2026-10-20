"""Before a stage deletes earlier output, the CLI and the UI say what and how
much and ask; a batch job needs --yes."""
import io
import sys
from pathlib import Path

import pytest

from lexichron import cli
from lexichron.safeguards import dir_size, format_bytes, overwrite_risk
from ngramprep.ngram_acquire.db.build_path import build_db_path

CORPUS = dict(ngram_size=5, repo_release_id="20200217", repo_corpus_id="eng-us")


def _corpus_dir(stub: Path) -> Path:
    return Path(build_db_path(str(stub), 5, "20200217", "eng-us")).parent


def _fill(d: Path, n: int = 3, size: int = 1000) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{i:06d}.sst").write_bytes(b"x" * size)
    (d / "CURRENT").write_text("MANIFEST-000001\n")


def test_dir_size_and_format(tmp_path):
    _fill(tmp_path / "db", n=4, size=250)
    (tmp_path / "db" / "sub").mkdir(); (tmp_path / "db" / "sub" / "a").write_bytes(b"y" * 10)
    nbytes, nfiles = dir_size(tmp_path / "db")
    assert nbytes == 4 * 250 + len("MANIFEST-000001\n") + 10 and nfiles == 6
    assert format_bytes(512) == "512 B" and format_bytes(1_500) == "1.5 KB"
    assert format_bytes(412_300_000_000) == "412 GB" and format_bytes(2_000_000_000_000) == "2.0 TB"


def test_no_risk_unless_the_settings_delete_and_the_target_exists(tmp_path):
    stub = tmp_path / "data"
    base = _corpus_dir(stub)
    # nothing on disk: nothing at stake whatever the mode
    assert overwrite_risk("filter", dict(CORPUS, db_path_stub=str(stub), mode="restart")) is None
    assert overwrite_risk("acquire", dict(CORPUS, db_path_stub=str(stub), overwrite_db=True)) is None
    _fill(base / "5grams_pivoted.db")
    _fill(base / "5grams_processed.db")
    _fill(Path(build_db_path(str(stub), 5, "20200217", "eng-us")))
    _fill(base / "tuning" / "run1" / "models")
    # resume / overwrite_db off: nothing deleted
    assert overwrite_risk("filter", dict(CORPUS, db_path_stub=str(stub), mode="resume")) is None
    assert overwrite_risk("filter", dict(CORPUS, db_path_stub=str(stub))) is None
    assert overwrite_risk("acquire", dict(CORPUS, db_path_stub=str(stub))) is None
    assert overwrite_risk("tune", dict(CORPUS, db_path_stub=str(stub), mode="resume", name="run1")) is None
    # an unnamed search gets a fresh timestamped directory, so restart deletes nothing
    assert overwrite_risk("tune", dict(CORPUS, db_path_stub=str(stub), mode="restart")) is None
    assert overwrite_risk("tune", dict(CORPUS, db_path_stub=str(stub), mode="restart", name="other")) is None


def test_risk_names_the_right_target(tmp_path):
    stub = tmp_path / "data"
    base = _corpus_dir(stub)
    _fill(base / "5grams_pivoted.db", n=5)
    _fill(base / "5grams_processed.db", n=2)
    raw = Path(build_db_path(str(stub), 5, "20200217", "eng-us")); _fill(raw, n=7)
    _fill(base / "tuning" / "run1" / "models", n=3)

    r = overwrite_risk("filter", dict(CORPUS, db_path_stub=str(stub), mode="restart"))      # 5-grams pivot by default
    assert r.path == base / "5grams_pivoted.db" and r.nfiles == 6 and "pivoted" in r.what
    assert r.setting == "filter.mode = restart" and "5grams_pivoted.db" in r.describe()
    r = overwrite_risk("filter", dict(CORPUS, db_path_stub=str(stub), mode="restart", pivot=False))
    assert r.path == base / "5grams_processed.db" and r.nfiles == 3
    r = overwrite_risk("acquire", dict(CORPUS, db_path_stub=str(stub), overwrite_db=True))
    assert r.path == raw and r.nfiles == 8 and r.setting == "acquire.overwrite_db = true"
    r = overwrite_risk("tune", dict(CORPUS, db_path_stub=str(stub), mode="restart", name="run1"))
    assert r.path == base / "tuning" / "run1" and r.nfiles == 4 and "run1" in r.what


class _Stage:
    """Stands in for a stage: records whether it was called."""
    calls = 0

    @staticmethod
    def run(ngram_size, repo_release_id, repo_corpus_id, db_path_stub, mode="resume", progress_path=None):
        _Stage.calls += 1


@pytest.fixture
def project(tmp_path, monkeypatch):
    stub = tmp_path / "data"
    _fill(_corpus_dir(stub) / "5grams_pivoted.db", n=3)
    proj = tmp_path / "project.yaml"
    proj.write_text(f"corpus:\n  release: '20200217'\n  language: eng-us\n  ngram_size: 5\n  db_path_stub: {stub}\n"
                    "filter:\n  mode: restart\n")
    monkeypatch.setattr(cli, "_resolve", lambda dotted: _Stage.run)
    import lexichron.provenance as provenance
    monkeypatch.setattr(provenance, "record_run", lambda *a, **k: None)
    _Stage.calls = 0
    return proj


def test_cli_batch_job_stops_without_yes(project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))             # no terminal
    assert cli.main(["filter", str(project)]) == 2
    err = capsys.readouterr().err
    assert "filter.mode = restart deletes the pivoted 5-gram database" in err
    assert "5grams_pivoted.db" in err and "--yes" in err
    assert _Stage.calls == 0
    # a resumed run is not asked anything
    assert cli.main(["filter", str(project), "--set", "filter.mode=resume"]) == 0
    assert _Stage.calls == 1
    # --yes goes ahead, saying so
    assert cli.main(["filter", str(project), "--yes"]) == 0
    assert "Going ahead (--yes)" in capsys.readouterr().err and _Stage.calls == 2


def test_cli_terminal_asks(project, monkeypatch, capsys):
    class Tty(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr(sys, "stdin", Tty())
    import builtins
    answers = iter(["n", "y"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert cli.main(["filter", str(project)]) == 2 and _Stage.calls == 0
    assert "nothing deleted, nothing run" in capsys.readouterr().err
    assert cli.main(["filter", str(project)]) == 0 and _Stage.calls == 1
    # dry runs never ask
    monkeypatch.setattr(builtins, "input", lambda prompt="": pytest.fail("asked on a dry run"))
    assert cli.main(["filter", str(project), "--dry-run"]) == 0


def test_sbatch_carries_yes_only_when_confirmed(tmp_path, monkeypatch):
    from lexichron.ui.slurm import write_sbatch
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    proj = tmp_path / "project.yaml"; proj.write_text("corpus: {}\n")
    assert "--yes" not in write_sbatch(proj, "filter", {}).read_text()
    text = write_sbatch(proj, "filter", {}, assume_yes=True).read_text()
    assert text.rstrip().splitlines()[-1].endswith("--yes")

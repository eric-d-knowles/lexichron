import os
import stat

from lexichron.ui.schema import stage_sections, parse_args_doc, SLURM_FIELDS
from lexichron.ui.slurm import write_sbatch, stage_command
from ngramprep.ngram_acquire import download_and_ingest_to_rocksdb as acquire


def test_sections_cover_every_visible_argument():
    corpus, stage = stage_sections(acquire, "acquire")
    assert [f.name for f in corpus.fields] == ["release", "language", "db_path_stub", "archive_path_stub"]
    names = {f.name for f in stage.fields}
    assert {"ngram_size", "ngram_type", "workers", "overwrite_db", "combined_bigrams", "chunk_entries"} <= names
    assert "progress_path" not in names and "write_batch_size" not in names
    kinds = {f.name: f.kind for f in stage.fields}
    assert kinds["overwrite_db"] == "bool" and kinds["workers"] == "int"
    assert kinds["ngram_size"] == "choice" and kinds["combined_bigrams"] == "list"
    assert kinds["spool_dir"] == "path"
    assert next(f for f in stage.fields if f.name == "ngram_size").required
    assert "N-gram size" in next(f for f in stage.fields if f.name == "ngram_size").help


def test_parse_args_doc_handles_multiline():
    doc = """
    Do things.

    Args:
        alpha: First line
            continues here.
        beta (int): Second.

    Returns:
        nothing
    """
    d = parse_args_doc(doc)
    assert d == {"alpha": "First line continues here.", "beta": "Second."}


def test_write_sbatch_uses_launcher_when_present(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; (proj / ".venv").mkdir(parents=True)
    launcher = proj / ".venv" / "host-python"; launcher.write_text("#!/bin/sh\n"); launcher.chmod(0o755)
    yaml_path = proj / "project.yaml"; yaml_path.write_text("corpus: {}\n")
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)

    out = write_sbatch(yaml_path, "acquire", {"account": "acct", "cpus": 8, "mem": "32G", "time": "02:00:00"})
    text = out.read_text()
    assert out.name == "project.acquire.sbatch" and out.stat().st_mode & stat.S_IXUSR
    assert "#SBATCH --account=acct" in text and "#SBATCH --cpus-per-task=8" in text
    assert "--partition" not in text
    assert str(launcher) in text and "-m lexichron.cli acquire" in text
    assert "--set acquire.workers=$SLURM_CPUS_PER_TASK" in text
    assert (proj / ".lexichron" / "slurm").is_dir()


def test_stage_command_falls_back_to_image(tmp_path, monkeypatch):
    yaml_path = tmp_path / "project.yaml"; yaml_path.write_text("")
    monkeypatch.setenv("APPTAINER_CONTAINER", "/img/lexichron.sif")
    assert stage_command(yaml_path, "acquire")[:3] == ["apptainer", "exec", "/img/lexichron.sif"]
    monkeypatch.delenv("APPTAINER_CONTAINER")
    assert stage_command(yaml_path, "acquire") == ["lexichron", "acquire", str(yaml_path)]


def test_slurm_fields_have_required_account():
    assert next(f for f in SLURM_FIELDS.fields if f.name == "account").required

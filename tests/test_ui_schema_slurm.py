import os
import stat

from lexichron.ui.schema import stage_sections, parse_args_doc, SLURM_FIELDS
from lexichron.ui.slurm import write_sbatch, stage_command
from ngramprep.ngram_acquire import download_and_ingest_to_rocksdb as acquire


def test_sections_cover_every_visible_argument():
    corpus, stage = stage_sections(acquire, "acquire")
    assert [f.name for f in corpus.fields] == ["release", "language", "ngram_size", "db_path_stub", "archive_path_stub"]
    names = {f.name for f in stage.fields}
    assert {"ngram_type", "workers", "overwrite_db", "combined_bigrams", "chunk_entries"} <= names
    assert "ngram_size" not in names
    assert "progress_path" not in names and "write_batch_size" not in names
    kinds = {f.name: f.kind for f in stage.fields}
    assert kinds["overwrite_db"] == "bool" and kinds["workers"] == "int"
    assert kinds["combined_bigrams"] == "list"
    assert kinds["spool_dir"] == "path"
    size = next(f for f in corpus.fields if f.name == "ngram_size")
    assert size.required and size.kind == "choice"
    assert size.title == "N-gram size" and "single words" in size.help
    # docstring help is the fallback for arguments without a HELP entry
    assert kinds["file_range"] == "range"
    assert {f.name for f in stage.fields if f.advanced} == {
        "spool_dir", "chunk_entries", "random_seed", "compact_after_ingest"}   # open_type is hidden
    assert next(f for f in corpus.fields if f.name == "archive_path_stub").advanced


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
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)

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
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    monkeypatch.setenv("APPTAINER_CONTAINER", "/img/lexichron.sif")
    assert stage_command(yaml_path, "acquire")[:3] == ["apptainer", "exec", "/img/lexichron.sif"]
    monkeypatch.delenv("APPTAINER_CONTAINER")
    assert stage_command(yaml_path, "acquire") == ["lexichron", "acquire", str(yaml_path)]


def test_slurm_fields_have_required_account():
    assert next(f for f in SLURM_FIELDS.fields if f.name == "account").required


def test_write_sbatch_job_array_splits_a_tune_search(tmp_path, monkeypatch):
    """slurm.jobs above 1 makes the script a job array whose tasks run the
    stage in resume mode on their share of the years; the other stages
    ignore it."""
    yaml_path = tmp_path / "project.yaml"; yaml_path.write_text("corpus: {}\n")
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    text = write_sbatch(yaml_path, "tune", {"account": "a", "cpus": 16, "mem": "32G", "time": "08:00:00",
                                            "jobs": 4}, extra_sets=["tune.mode=restart"]).read_text()
    assert "#SBATCH --array=0-3" in text and "%A_%a.out" in text and "%j" not in text
    assert "--set tune.array_task=$SLURM_ARRAY_TASK_ID --set tune.array_count=$SLURM_ARRAY_TASK_COUNT" in text
    assert "--set tune.mode=resume" in text and "tune.mode=restart" not in text   # the caller deleted first
    assert "--set tune.workers=$SLURM_CPUS_PER_TASK" in text
    plain = write_sbatch(yaml_path, "tune", {"account": "a", "cpus": 16, "mem": "32G", "time": "08:00:00",
                                             "jobs": 1}).read_text()
    assert "--array" not in plain and "array_task" not in plain and "%j.out" in plain
    other = write_sbatch(yaml_path, "filter", {"account": "a", "cpus": 16, "mem": "32G", "time": "08:00:00",
                                               "jobs": 4}).read_text()
    assert "--array" not in other and "array_task" not in other


def test_array_settings_are_hidden_from_the_form():
    from lexichron.stages.tune import tune_models
    _, stage = stage_sections(tune_models, "tune")
    names = {f.name for f in stage.fields}
    assert "array_task" not in names and "array_count" not in names
    assert next(f for f in SLURM_FIELDS.fields if f.name == "jobs").default == 1


def test_labels_can_differ_by_stage():
    # tune and train both take ``name``, but one names a search and the other an ensemble
    from lexichron.stages.tune import tune_models
    from lexichron.stages.train import train_models
    _, tune_stage = stage_sections(tune_models, "tune")
    _, train_stage = stage_sections(train_models, "train")
    assert next(f for f in tune_stage.fields if f.name == "name").title == "Search name"
    train_name = next(f for f in train_stage.fields if f.name == "name")
    assert train_name.title == "Ensemble name"
    assert "models/<name>" in train_name.help

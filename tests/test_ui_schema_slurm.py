import os

import pytest
import stat

from lexichron.ui.schema import stage_sections, parse_args_doc, SLURM_FIELDS
from lexichron.ui.slurm import write_sbatch, stage_command
from ngramprep.ngram_acquire import download_and_ingest_to_rocksdb as acquire


def test_sections_cover_every_visible_argument():
    corpus, stage = stage_sections(acquire, "acquire")
    assert [f.name for f in corpus.fields] == ["release", "name", "ngram_size", "db_path_stub", "archive_path_stub"]
    names = {f.name for f in stage.fields}
    assert {"ngram_type", "workers", "overwrite_db", "combined_multigrams", "chunk_entries"} <= names
    assert "ngram_size" not in names
    assert "progress_path" not in names and "write_batch_size" not in names
    kinds = {f.name: f.kind for f in stage.fields}
    assert kinds["overwrite_db"] == "bool" and kinds["workers"] == "int"
    assert kinds["combined_multigrams"] == "multigrams"
    assert kinds["spool_dir"] == "path"
    size = next(f for f in corpus.fields if f.name == "ngram_size")
    assert size.required and size.kind == "choice"
    assert size.title == "N-gram size" and "single words" in size.help
    # docstring help is the fallback for arguments without a HELP entry
    assert kinds["file_range"] == "range"
    assert {f.name for f in stage.fields if f.advanced} == {
            "spool_dir", "chunk_entries", "random_seed", "verify_counts", "allow_unparsed_lines"}
    # hidden: open_type; spool_format (the reference path, for tests); and acquire's
    # compact_after_ingest, which still loads from a settings file but is ignored
    assert not {"open_type", "spool_format", "compact_after_ingest"} & set(kinds)
    assert kinds["verify_counts"] == "bool" and kinds["allow_unparsed_lines"] == "bool"
    assert "combined_multigrams_case" not in kinds                            # edited with combined_multigrams
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
    # Singularity sets only its own variable: the command names it, not apptainer
    monkeypatch.setenv("SINGULARITY_CONTAINER", "/img/lexichron.sif")
    assert stage_command(yaml_path, "acquire")[:3] == ["singularity", "exec", "/img/lexichron.sif"]


def test_write_sbatch_lets_the_job_pick_its_container_runtime(tmp_path, monkeypatch):
    """A UI started by singularity must not write a script that calls
    apptainer (a cluster with one rarely has the other, and the job's node
    need not have the submitting node's PATH): the script prefers the
    runtime the UI ran under and falls back to whichever the job finds."""
    yaml_path = tmp_path / "project.yaml"; yaml_path.write_text("corpus: {}\n")
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.setenv("SINGULARITY_CONTAINER", "/img/lexichron.sif")
    text = write_sbatch(yaml_path, "tune", {"account": "a", "cpus": 8, "mem": "8G", "time": "01:00:00"}).read_text()
    assert "runtime=singularity\n" in text and 'for c in apptainer singularity; do' in text
    assert "module load" in text and "exit 127" in text
    assert '"$runtime" exec /img/lexichron.sif lexichron tune' in text
    assert "apptainer exec" not in text and "singularity exec" not in text
    # the script resolves the runtime at run time: with only a fake singularity on PATH it runs it
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "singularity").write_text('#!/bin/sh\necho "singularity $*"\n'); (fake / "singularity").chmod(0o755)
    import subprocess
    body = text.replace("set -euo pipefail", "set -eo pipefail")     # the test's env lacks the Slurm variables
    out = subprocess.run(["bash", "-c", body], env={"PATH": f"{fake}:/usr/bin:/bin", "SLURM_CPUS_PER_TASK": "8"},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith("singularity exec /img/lexichron.sif lexichron tune")
    monkeypatch.delenv("SINGULARITY_CONTAINER")
    monkeypatch.setenv("APPTAINER_CONTAINER", "/img/lexichron.sif")
    text = write_sbatch(yaml_path, "tune", {"account": "a", "cpus": 8, "mem": "8G", "time": "01:00:00"}).read_text()
    assert "runtime=apptainer\n" in text


def test_slurm_fields_have_required_account():
    assert next(f for f in SLURM_FIELDS.fields if f.name == "account").required


def test_write_sbatch_job_array_splits_a_tune_search(tmp_path, monkeypatch):
    """slurm.jobs above 1 makes the script a job array whose tasks run the
    stage in resume mode on their share of the years; the other stages
    ignore it."""
    yaml_path = tmp_path / "project.yaml"; yaml_path.write_text("corpus: {}\ntune:\n  name: grid1\n")
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


def test_a_worker_count_saved_in_the_settings_is_not_overridden_by_the_cpus(tmp_path, monkeypatch):
    """The Submit tab works its memory estimate out from the stage's saved
    worker count, and tells a user short of memory to set one. The script
    used to end with --set <stage>.workers=$SLURM_CPUS_PER_TASK regardless, so
    the job ran one worker per CPU whatever the settings file said."""
    from lexichron.resources import memory_estimate
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    slurm = {"account": "a", "cpus": 80, "mem": "60G", "time": "01:00:00"}

    pinned = tmp_path / "pinned.yaml"
    pinned.write_text("corpus: {}\nacquire:\n  overwrite_db: true\n  workers: 20\nfilter: {}\n")
    text = write_sbatch(pinned, "acquire", slurm).read_text()
    assert "acquire.workers=$SLURM_CPUS_PER_TASK" not in text                 # the saved 20 stands
    assert "# acquire.workers is set to 20 in the settings file" in text
    assert memory_estimate("acquire", {"workers": 20}, slurm).processes == 20   # and is what the estimate describes
    # another stage of the same file has no count of its own: its job follows the CPUs
    assert "--set filter.workers=$SLURM_CPUS_PER_TASK" in write_sbatch(pinned, "filter", slurm).read_text()

    # no count, a blank one, or a file that cannot be read: the CPUs decide, as the estimate assumes
    for body in ("corpus: {}\n", "corpus: {}\nacquire:\n  workers:\n", "corpus: {}\nacquire:\n", ": not yaml ["):
        free = tmp_path / "free.yaml"; free.write_text(body)
        assert "--set acquire.workers=$SLURM_CPUS_PER_TASK" in write_sbatch(free, "acquire", slurm).read_text(), body
    assert memory_estimate("acquire", {}, slurm).processes == 80
    # an override given at submit time still wins over both
    text = write_sbatch(pinned, "acquire", slurm, extra_sets=["acquire.workers=7"]).read_text()
    assert "--set acquire.workers=7" in text and "SLURM_CPUS_PER_TASK" not in text.split("lexichron")[-1]


def test_a_job_array_is_refused_without_a_name_for_the_search_or_ensemble(tmp_path, monkeypatch):
    """An unnamed search or ensemble is named by the time it starts. The
    tasks of an array start at different seconds, so each would make its
    own directory and none would see the others' models: no year pair could
    be scored. (A job name on the Submit tab is not the search's name.)"""
    from lexichron.config import ConfigError
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    slurm = {"account": "a", "cpus": 16, "mem": "32G", "time": "08:00:00", "jobs": 4, "job_name": "occfem_tuning1"}
    bare = tmp_path / "project.yaml"; bare.write_text("corpus: {}\ntune:\n  name: ''\n")
    for stage, label in (("tune", "Search name"), ("train", "Ensemble name")):
        with pytest.raises(ConfigError, match=f"needs a name.*'{label}'"):
            write_sbatch(bare, stage, slurm)
        # a name given as an override, or one job, is fine
        assert "--array=0-3" in write_sbatch(bare, stage, slurm, extra_sets=[f"{stage}.name=run1"]).read_text()
        assert "--array" not in write_sbatch(bare, stage, dict(slurm, jobs=1)).read_text()
    named = tmp_path / "named.yaml"; named.write_text("corpus: {}\ntrain:\n  name: ens1\n")
    assert "--array=0-3" in write_sbatch(named, "train", slurm).read_text()

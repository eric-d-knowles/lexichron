"""Where a job's Slurm settings come from: stage defaults, the user's
account, the corpus's file (lexichron.slurm_settings)."""
import pytest

from lexichron.config import ConfigError
from lexichron.slurm_settings import (describe_sources, read_user_defaults, resolve, seed_user_defaults,
                                      stage_override, user_defaults_path, write_user_defaults)
from lexichron.ui.slurm import write_sbatch


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXICHRON_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_resolution_layers_stage_defaults_user_defaults_and_the_file(state):
    values, sources = resolve("train", None, {})
    assert values == {"cpus": 64, "mem": "64G", "time": "1-00:00:00", "jobs": 13}
    assert sources["cpus"] == "lexichron's default for train" and "account" not in values
    write_user_defaults({"account": "proj_a", "partition": ""})
    assert read_user_defaults() == {"account": "proj_a"}
    values, sources = resolve("acquire", {"mem": "40G"}, None)
    assert values == {"cpus": 40, "mem": "40G", "time": "12:00:00", "account": "proj_a"}   # no jobs for acquire
    assert sources["account"] == "your defaults" and sources["mem"] == "this corpus"
    section = {"account": "proj_b", "cpus": 32, "train": {"cpus": 96, "time": "2-00:00:00"}}
    values, sources = resolve("train", section)
    assert values["account"] == "proj_b" and sources["account"] == "this corpus"
    assert values["cpus"] == 96 and sources["cpus"] == "this corpus, train"
    assert values["jobs"] == 13 and sources["jobs"] == "lexichron's default for train"
    assert resolve("tune", section)[0]["cpus"] == 32                      # the top-level key, for other stages
    line = describe_sources(*resolve("train", section))
    assert "account proj_b (this corpus)" in line and "CPUs 96, time 2-00:00:00 (this corpus, train)" in line
    assert "partition: the cluster's default" in line
    assert describe_sources(*resolve("train", None, {})).startswith("no account:")


def test_save_writes_the_stage_block_and_keeps_the_account_out_while_it_is_the_default(state):
    write_user_defaults({"account": "proj_a"})
    form = {"account": "proj_a", "cpus": 64, "mem": "64G", "time": "1-00:00:00", "jobs": 13, "job_name": "lexichron"}
    section = stage_override({"cpus": 32, "tune": {"cpus": 48}}, "train", form)
    assert section == {"cpus": 32, "tune": {"cpus": 48},
                       "train": {"cpus": 64, "mem": "64G", "time": "1-00:00:00", "jobs": 13, "job_name": "lexichron"}}
    # a different account for this corpus is written; back to the default, it is removed again
    other = stage_override(section, "train", dict(form, account="proj_b"))
    assert other["account"] == "proj_b"
    assert "account" not in stage_override(other, "train", form)
    # jobs mean nothing to acquire and are not written for it
    assert "jobs" not in stage_override({}, "acquire", dict(form, cpus=40))["acquire"]


def test_first_saved_account_becomes_the_default_once(state):
    assert not seed_user_defaults({"cpus": 8})                               # nothing to seed
    assert seed_user_defaults({"account": "proj_a", "partition": "cs"})
    assert read_user_defaults() == {"account": "proj_a", "partition": "cs"}
    assert not seed_user_defaults({"account": "proj_b"})                     # later changes are on request
    assert read_user_defaults()["account"] == "proj_a"
    assert user_defaults_path() == state / "slurm.yaml"


def test_no_script_without_an_account(tmp_path, monkeypatch):
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    yaml_path = tmp_path / "project.yaml"; yaml_path.write_text("corpus: {}\n")
    with pytest.raises(ConfigError, match="no Slurm account"):
        write_sbatch(yaml_path, "train", {"cpus": 64, "mem": "64G", "time": "1-00:00:00"})
    assert not yaml_path.with_suffix(".train.sbatch").exists()

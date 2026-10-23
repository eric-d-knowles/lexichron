"""Splitting a stage's years over the tasks of a job array."""
import pytest

from lexichron.arrays import ArrayTask, array_task, task_share


def test_task_share_is_round_robin():
    years = list(range(1968, 1976))                       # 8 years
    assert task_share(years, None) == years
    assert task_share(years, ArrayTask(0, 3)) == [1968, 1971, 1974]
    assert task_share(years, ArrayTask(1, 3)) == [1969, 1972, 1975]
    assert task_share(years, ArrayTask(2, 3)) == [1970, 1973]
    assert task_share([1990, 1991], ArrayTask(3, 4)) == []     # more tasks than years: nothing to do
    # every year lands in exactly one task
    assert sorted(y for i in range(3) for y in task_share(years, ArrayTask(i, 3))) == years


def test_array_task_from_settings_or_environment(monkeypatch):
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
    assert array_task(None, None) is None
    assert array_task(0, 1) is None                       # one task is not an array
    assert array_task(2, 5) == ArrayTask(2, 5)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "8")
    assert array_task(None, None) == ArrayTask(3, 8)      # a hand-written #SBATCH --array script
    assert array_task(1, 2) == ArrayTask(1, 2)            # the settings win
    with pytest.raises(ValueError):
        array_task(5, 5)
    with pytest.raises(ValueError):
        array_task(1, None)
    t = ArrayTask(3, 8)
    assert t.label == "task 4 of 8" and t.suffix == "t3"


def test_job_id_is_the_queue_name(monkeypatch):
    from ngramprep.ngram_acquire.progress import slurm_job_id
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    assert slurm_job_id() == "99"
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    assert slurm_job_id() == "12345_3"

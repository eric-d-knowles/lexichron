"""Splitting a stage's years over the tasks of a Slurm job array.

The stages that train models (``tune``, ``train``) work year by year: a
year's corpus is written once and every model that needs it is trained
from it. A run too long for one job is split by year over the tasks of a
job array: task *i* of *N* takes every N-th year, counting from the i-th
(round-robin), so each task gets an even mix of the early, small corpora
and the late, large ones and the tasks finish together. Each task is an
ordinary run of the stage on its years, writing into the same output
directory; what belongs to a year is written by the one task that has it,
and what joins two years (a year-pair score) is written by whichever task
finds both years' models on disk.

A task learns its index and the count from the stage's ``array_task`` and
``array_count`` settings, which the batch script sets from Slurm's
``$SLURM_ARRAY_TASK_ID`` and ``$SLURM_ARRAY_TASK_COUNT``; a script written
by hand can leave them to the environment. A run with neither is the whole
range in one job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, TypeVar

__all__ = ["ArrayTask", "array_task", "task_share"]

T = TypeVar("T")


@dataclass(frozen=True)
class ArrayTask:
    index: int        # 0-based
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"array_count must be at least 1, not {self.count}")
        if not 0 <= self.index < self.count:
            raise ValueError(f"array_task {self.index} is not one of the {self.count} tasks (0 to {self.count - 1})")

    @property
    def label(self) -> str:
        """``task 3 of 8`` (1-based for people)."""
        return f"task {self.index + 1} of {self.count}"

    @property
    def suffix(self) -> str:
        """``t3`` for file and directory names."""
        return f"t{self.index}"


def array_task(task: Optional[int] = None, count: Optional[int] = None) -> Optional[ArrayTask]:
    """The task this run is, from the settings when given, else from Slurm's
    environment, else None (not an array: the whole range). A count of 1
    is not an array either."""
    if task is None and count is None:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if not (env_task and env_count and env_task.lstrip("-").isdigit() and env_count.isdigit()):
            return None
        task, count = int(env_task), int(env_count)
    if task is None or count is None:
        raise ValueError("array_task and array_count go together: set both or neither")
    task, count = int(task), int(count)
    if count == 1 and task == 0:
        return None
    return ArrayTask(task, count)


def task_share(items: Sequence[T], task: Optional[ArrayTask]) -> List[T]:
    """The items this task works on: every ``count``-th one from ``index``
    (round-robin), or all of them when there is no array. A task past the
    end of a short list gets nothing, and should say so and stop."""
    if task is None:
        return list(items)
    return list(items)[task.index::task.count]


"""Self-contained task definitions for container builds."""

from .task_bank import TaskSpec, get_task, list_task_descriptors, list_task_summaries, task_ids

__all__ = [
    "TaskSpec",
    "get_task",
    "list_task_descriptors",
    "list_task_summaries",
    "task_ids",
]


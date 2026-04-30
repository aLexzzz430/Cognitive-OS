from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Task:
    title: str
    due_date: date
    completed: bool = False


def next_due(tasks: list[Task], today: date) -> Task | None:
    actionable = [
        task
        for task in tasks
        if not task.completed and task.due_date > today
    ]
    if not actionable:
        return None
    return min(actionable, key=lambda task: task.due_date)

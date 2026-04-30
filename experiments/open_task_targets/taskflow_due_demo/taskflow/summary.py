from __future__ import annotations

from datetime import date

from .scheduler import Task


def overdue_count(tasks: list[Task], today: date) -> int:
    return sum(1 for task in tasks if not task.completed and task.due_date < today)

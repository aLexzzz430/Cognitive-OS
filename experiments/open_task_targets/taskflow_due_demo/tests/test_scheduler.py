from datetime import date

from taskflow import Task, next_due


def test_next_due_includes_task_due_today() -> None:
    today = date(2026, 4, 29)
    tasks = [
        Task("tomorrow", date(2026, 4, 30)),
        Task("today", today),
    ]

    assert next_due(tasks, today) == tasks[1]


def test_next_due_skips_completed_tasks() -> None:
    today = date(2026, 4, 29)
    tasks = [
        Task("done today", today, completed=True),
        Task("tomorrow", date(2026, 4, 30)),
    ]

    assert next_due(tasks, today) == tasks[1]

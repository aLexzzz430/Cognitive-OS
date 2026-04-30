from datetime import date

from taskflow import Task, overdue_count


def test_overdue_count_ignores_today() -> None:
    today = date(2026, 4, 29)
    tasks = [
        Task("yesterday", date(2026, 4, 28)),
        Task("today", today),
        Task("done old", date(2026, 4, 27), completed=True),
    ]

    assert overdue_count(tasks, today) == 1

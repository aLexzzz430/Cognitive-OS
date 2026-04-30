# Taskflow Due Demo

Taskflow is a tiny scheduling helper used by a team dashboard.

Rules:
- Completed tasks are not actionable.
- A task due today is actionable today.
- The next due task is the incomplete task with the earliest actionable due date.

Run tests with:

```bash
python -m pytest -q
```

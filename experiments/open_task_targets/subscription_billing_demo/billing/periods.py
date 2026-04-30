from __future__ import annotations

from calendar import monthrange
from datetime import date


def days_in_month(day: date) -> int:
    return monthrange(day.year, day.month)[1]


def active_days(start: date, end: date) -> int:
    """Return the number of billable days in an inclusive date window."""

    if end < start:
        return 0
    return (end - start).days


def same_month(start: date, end: date) -> bool:
    return start.year == end.year and start.month == end.month


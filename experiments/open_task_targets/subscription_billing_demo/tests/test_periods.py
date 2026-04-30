from datetime import date

from billing.periods import active_days, days_in_month, same_month


def test_active_days_counts_same_day_window() -> None:
    assert active_days(date(2026, 4, 15), date(2026, 4, 15)) == 1


def test_active_days_counts_inclusive_window() -> None:
    assert active_days(date(2026, 4, 1), date(2026, 4, 15)) == 15


def test_month_helpers() -> None:
    assert days_in_month(date(2026, 4, 1)) == 30
    assert same_month(date(2026, 4, 1), date(2026, 4, 30))
    assert not same_month(date(2026, 4, 30), date(2026, 5, 1))


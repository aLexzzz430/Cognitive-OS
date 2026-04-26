from __future__ import annotations

from sales_core.report import department_totals


def test_department_totals_group_by_department_not_region() -> None:
    rows = [
        {"department": "hardware", "amount_cents": 1200},
        {"department": "software", "amount_cents": 500},
        {"department": "hardware", "amount_cents": 300},
    ]

    assert department_totals(rows) == {"hardware": 1500, "software": 500}

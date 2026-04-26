from __future__ import annotations

from sales_core.grouping import group_rows


def test_group_rows_uses_requested_key() -> None:
    rows = [
        {"department": "hardware", "amount_cents": 1200},
        {"department": "software", "amount_cents": 500},
    ]

    grouped = group_rows(rows, "department")

    assert sorted(grouped) == ["hardware", "software"]

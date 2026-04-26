from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .grouping import group_rows


def department_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    grouped = group_rows(rows, "department")
    return {
        str(department): sum(int(row["amount_cents"]) for row in department_rows)
        for department, department_rows in grouped.items()
    }

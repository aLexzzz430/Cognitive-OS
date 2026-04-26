from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Mapping

from .amounts import parse_amount


def invoice_total(rows: Iterable[Mapping[str, object]]) -> Decimal:
    total = Decimal("0")
    for row in rows:
        total += parse_amount(row.get("amount", "0"))
    return total


def is_large_invoice(rows: Iterable[Mapping[str, object]], *, threshold: str = "$1,000") -> bool:
    return invoice_total(rows) >= parse_amount(threshold)

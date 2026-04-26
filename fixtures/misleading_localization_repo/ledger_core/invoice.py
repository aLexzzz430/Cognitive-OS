from __future__ import annotations

from collections.abc import Iterable, Mapping

from .currency import normalize_currency


def invoice_total(lines: Iterable[Mapping[str, str]]) -> int:
    total = 0
    for line in lines:
        total += normalize_currency(line["amount"])
    return total


def invoice_balance_due(lines: Iterable[Mapping[str, str]], credit: str = "$0.00") -> int:
    return invoice_total(lines) - normalize_currency(credit)

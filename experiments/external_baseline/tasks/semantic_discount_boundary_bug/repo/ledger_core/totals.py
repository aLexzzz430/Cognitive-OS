from __future__ import annotations

from collections.abc import Iterable, Mapping


def subtotal_cents(lines: Iterable[Mapping[str, int]]) -> int:
    total = 0
    for line in lines:
        quantity = int(line.get("quantity", 1))
        total += int(line["price_cents"]) * quantity
    return total

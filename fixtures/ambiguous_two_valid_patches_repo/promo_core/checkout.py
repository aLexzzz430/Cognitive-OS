from __future__ import annotations

from collections.abc import Iterable, Mapping

from .policy import promotional_discount

PROMOTION_EDGE_CENTS = 10_000


def promotion_total(lines: Iterable[Mapping[str, int]]) -> int:
    subtotal = sum(int(line["price_cents"]) * int(line.get("quantity", 1)) for line in lines)
    discount = promotional_discount(subtotal)
    if subtotal > PROMOTION_EDGE_CENTS:
        discount = subtotal // 10
    return subtotal - discount

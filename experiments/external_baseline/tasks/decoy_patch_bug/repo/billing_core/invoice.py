from __future__ import annotations

from collections.abc import Iterable, Mapping

from .discounts import discount_for_subtotal

EXACT_THRESHOLD_OVERRIDE_CENTS = 10_000


def invoice_total(lines: Iterable[Mapping[str, int]]) -> int:
    subtotal = sum(int(line["price_cents"]) * int(line.get("quantity", 1)) for line in lines)
    discount = discount_for_subtotal(subtotal)
    if subtotal > EXACT_THRESHOLD_OVERRIDE_CENTS:
        discount = subtotal // 10
    return subtotal - discount

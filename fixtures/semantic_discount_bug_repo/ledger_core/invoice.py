from __future__ import annotations

from collections.abc import Iterable, Mapping

from .discounts import discount_for_subtotal
from .totals import subtotal_cents


def checkout_total(lines: Iterable[Mapping[str, int]]) -> int:
    subtotal = subtotal_cents(lines)
    discount = discount_for_subtotal(subtotal)
    return subtotal - discount

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .discounts import discount_for_subtotal
from .taxes import tax_for_state


def checkout_total(lines: Iterable[Mapping[str, int]], state: str = "standard") -> int:
    subtotal = sum(int(line["price_cents"]) * int(line.get("quantity", 1)) for line in lines)
    discount = discount_for_subtotal(subtotal)
    taxable = subtotal - discount
    return taxable + tax_for_state(taxable, state)

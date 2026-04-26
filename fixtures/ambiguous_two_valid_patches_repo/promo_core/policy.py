from __future__ import annotations

PROMOTION_EDGE_CENTS = 10_000


def promotional_discount(subtotal_cents: int) -> int:
    if subtotal_cents > PROMOTION_EDGE_CENTS:
        return subtotal_cents // 10
    return 0

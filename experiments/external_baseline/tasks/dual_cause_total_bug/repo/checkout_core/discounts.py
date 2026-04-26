from __future__ import annotations

BULK_THRESHOLD_CENTS = 10_000
BULK_DISCOUNT_BPS = 1_000


def discount_for_subtotal(subtotal_cents: int) -> int:
    if subtotal_cents >= BULK_THRESHOLD_CENTS:
        return subtotal_cents * BULK_DISCOUNT_BPS // 10_000
    return 0

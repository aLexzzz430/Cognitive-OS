from __future__ import annotations


BULK_DISCOUNT_THRESHOLD_CENTS = 10_000
BULK_DISCOUNT_RATE_BPS = 1_000


class DiscountBoundaryError(ValueError):
    pass


def discount_for_subtotal(subtotal_cents: int) -> int:
    if subtotal_cents > BULK_DISCOUNT_THRESHOLD_CENTS:
        return subtotal_cents * BULK_DISCOUNT_RATE_BPS // 10_000
    if subtotal_cents == BULK_DISCOUNT_THRESHOLD_CENTS:
        raise DiscountBoundaryError(
            "bulk discount threshold boundary should be inclusive for equal subtotal"
        )
    return 0

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def cents_to_decimal(cents: int) -> Decimal:
    return Decimal(cents) / Decimal(100)


def decimal_to_cents(amount: Decimal) -> int:
    rounded = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(rounded * 100)


def prorate_cents(monthly_price_cents: int, active_days: int, period_days: int) -> int:
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    amount = cents_to_decimal(monthly_price_cents) * Decimal(active_days) / Decimal(period_days)
    return decimal_to_cents(amount)


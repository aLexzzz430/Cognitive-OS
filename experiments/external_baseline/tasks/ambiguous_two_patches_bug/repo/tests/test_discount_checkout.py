from __future__ import annotations

from promo_core.checkout import promotion_total


def test_promotion_total_applies_discount_at_exact_edge() -> None:
    lines = [{"price_cents": 2_500, "quantity": 4}]

    assert promotion_total(lines) == 9_000

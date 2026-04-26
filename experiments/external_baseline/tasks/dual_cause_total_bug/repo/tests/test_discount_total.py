from __future__ import annotations

from checkout_core.invoice import checkout_total


def test_checkout_total_applies_discount_at_exact_threshold_before_tax() -> None:
    lines = [{"price_cents": 2_500, "quantity": 4}]

    assert checkout_total(lines, state="standard") == 9_450

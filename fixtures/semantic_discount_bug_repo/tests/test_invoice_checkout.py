from __future__ import annotations

from ledger_core.invoice import checkout_total


def test_checkout_applies_bulk_discount_at_exact_threshold() -> None:
    total = checkout_total([{"price_cents": 2_500, "quantity": 4}])
    assert total == 9_000


def test_checkout_below_threshold_has_no_discount() -> None:
    total = checkout_total([{"price_cents": 2_499, "quantity": 4}])
    assert total == 9_996

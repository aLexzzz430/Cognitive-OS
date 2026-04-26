from __future__ import annotations

from billing_core.invoice import invoice_total


def test_invoice_total_applies_discount_at_exact_threshold() -> None:
    lines = [{"price_cents": 2_500, "quantity": 4}]

    assert invoice_total(lines) == 9_000

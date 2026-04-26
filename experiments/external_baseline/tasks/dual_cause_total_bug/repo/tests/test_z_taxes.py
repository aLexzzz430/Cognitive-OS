from __future__ import annotations

from checkout_core.taxes import tax_for_state


def test_standard_tax_rate() -> None:
    assert tax_for_state(9_000, "standard") == 450


def test_reduced_tax_rate() -> None:
    assert tax_for_state(9_000, "reduced") == 225

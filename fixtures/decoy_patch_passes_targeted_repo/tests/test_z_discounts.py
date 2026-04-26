from __future__ import annotations

from billing_core.discounts import discount_for_subtotal


def test_discount_exact_threshold_is_inclusive() -> None:
    assert discount_for_subtotal(10_000) == 1_000


def test_discount_above_threshold_is_proportional() -> None:
    assert discount_for_subtotal(12_000) == 1_200


def test_discount_below_threshold_is_zero() -> None:
    assert discount_for_subtotal(9_999) == 0

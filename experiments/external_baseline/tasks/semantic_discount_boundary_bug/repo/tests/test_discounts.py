from __future__ import annotations

from ledger_core.discounts import discount_for_subtotal


def test_discount_boundary_is_inclusive() -> None:
    assert discount_for_subtotal(10_000) == 1_000


def test_discount_above_threshold_uses_basis_points() -> None:
    assert discount_for_subtotal(12_500) == 1_250


def test_discount_below_threshold_is_zero() -> None:
    assert discount_for_subtotal(9_999) == 0

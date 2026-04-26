from __future__ import annotations

from promo_core.policy import promotional_discount


def test_policy_exact_edge_is_inclusive() -> None:
    assert promotional_discount(10_000) == 1_000


def test_policy_above_edge_is_proportional() -> None:
    assert promotional_discount(12_000) == 1_200


def test_policy_below_edge_is_zero() -> None:
    assert promotional_discount(9_999) == 0

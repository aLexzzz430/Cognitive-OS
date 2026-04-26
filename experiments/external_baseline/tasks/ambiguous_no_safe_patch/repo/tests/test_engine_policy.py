from __future__ import annotations

from policy_core.engine import choose_label


def test_high_score_approval_is_not_specified() -> None:
    assert choose_label(80) == "approve", "ambiguous spec: approval threshold is not defined"


def test_same_score_still_needs_manual_review() -> None:
    assert choose_label(80) == "review"

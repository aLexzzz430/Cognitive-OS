"""
planner/planning_policy.py

Planner policy definitions for ObjectiveDecomposer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PlanningPolicy:
    """Injectable planning policy."""

    name: str
    exploration_base_order: Sequence[str]
    exploration_safe_order: Sequence[str]
    canonical_chain: Sequence[str]
    canonical_chain_intents: Dict[str, str]
    canonical_chain_fallback_map: Dict[str, Sequence[str]]
    confirmation_low_risk_functions: Sequence[str]


DEFAULT_POLICY = PlanningPolicy(
    name="default",
    exploration_base_order=(
        "join_tables",
        "aggregate_group",
        "filter_by_predicate",
        "array_transform",
        "compute_stats",
    ),
    exploration_safe_order=(
        "compute_stats",
        "filter_by_predicate",
        "array_transform",
        "aggregate_group",
        "join_tables",
    ),
    canonical_chain=("scan", "calibrate", "route", "commit"),
    canonical_chain_intents={
        "scan": "explore",
        "calibrate": "compute",
        "route": "exploit",
        "commit": "exploit",
    },
    canonical_chain_fallback_map={
        "scan": ("calibrate",),
        "calibrate": ("route",),
        "route": ("commit",),
        "commit": ("route",),
    },
    confirmation_low_risk_functions=(
        "compute_stats",
        "filter_by_predicate",
        "array_transform",
    ),
)


HARD_PARTIAL_OBSERVABLE_POLICY = PlanningPolicy(
    name="hard_partial_observable",
    exploration_base_order=DEFAULT_POLICY.exploration_base_order,
    exploration_safe_order=DEFAULT_POLICY.exploration_safe_order,
    canonical_chain=DEFAULT_POLICY.canonical_chain,
    canonical_chain_intents=DEFAULT_POLICY.canonical_chain_intents,
    canonical_chain_fallback_map=DEFAULT_POLICY.canonical_chain_fallback_map,
    confirmation_low_risk_functions=DEFAULT_POLICY.confirmation_low_risk_functions,
)


TAGGED_POLICIES: Dict[str, PlanningPolicy] = {
    "hard_partial_observable": HARD_PARTIAL_OBSERVABLE_POLICY,
    "novel_api": DEFAULT_POLICY,
}


def resolve_planning_policy(
    tags: Sequence[str],
    injected_policy: Optional[PlanningPolicy] = None,
) -> PlanningPolicy:
    """
    Resolve planning policy by tags.

    Priority:
    1) injected policy
    2) tagged policy
    3) default compatibility policy
    """
    if injected_policy is not None:
        return injected_policy

    for tag in tags:
        normalized = str(tag or "").strip().lower()
        if normalized in TAGGED_POLICIES:
            return TAGGED_POLICIES[normalized]
    return DEFAULT_POLICY

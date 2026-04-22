from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from core.main_loop_components import CAPABILITY_ADVISORY, CAPABILITY_CONSTRAINED_CONTROL, CAPABILITY_PRIMARY_CONTROL


@dataclass(frozen=True)
class GovernanceState:
    """Immutable governance capability snapshot used by governance runtime."""

    organ_failure_streaks: Mapping[str, int]
    organ_capability_flags: Mapping[str, str]
    organ_failure_threshold: int


@dataclass(frozen=True)
class GovernanceStatePatch:
    """Patch emitted by governance runtime to update governance-related loop state."""

    organ_failure_streaks: Dict[str, int]
    organ_capability_flags: Dict[str, str]


def capability_for_organ(state: GovernanceState, organ: str) -> str:
    if organ not in state.organ_capability_flags:
        return CAPABILITY_ADVISORY
    raw = str(state.organ_capability_flags.get(organ, CAPABILITY_ADVISORY) or CAPABILITY_ADVISORY)
    if raw not in {CAPABILITY_ADVISORY, CAPABILITY_CONSTRAINED_CONTROL, CAPABILITY_PRIMARY_CONTROL}:
        return CAPABILITY_ADVISORY
    if int(state.organ_failure_streaks.get(organ, 0) or 0) >= int(state.organ_failure_threshold or 0):
        return CAPABILITY_ADVISORY
    return raw

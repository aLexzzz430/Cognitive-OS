from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


STATUS_PROPOSED = "proposed"
STATUS_QUALIFIED = "qualified"
STATUS_PROMOTED = "promoted"
STATUS_GRADUATED = "graduated"
STATUS_WEAKENED = "weakened"
STATUS_RETIRED = "retired"
STATUS_REOPENED = "reopened"
STATUS_DOWNGRADED = "downgraded"
STATUS_INVALIDATED = "invalidated"

ALL_LIFECYCLE_STATUSES = (
    STATUS_PROPOSED,
    STATUS_QUALIFIED,
    STATUS_PROMOTED,
    STATUS_GRADUATED,
    STATUS_WEAKENED,
    STATUS_RETIRED,
    STATUS_REOPENED,
    STATUS_DOWNGRADED,
    STATUS_INVALIDATED,
    "active",
)

TRANSITION_PROPOSE = "propose"
TRANSITION_PROMOTE = "promote"
TRANSITION_WEAKEN = "weaken"
TRANSITION_RETIRE = "retire"
TRANSITION_REOPEN = "reopen"
TRANSITION_MERGE = "merge"
TRANSITION_SPLIT = "split"
TRANSITION_QUALIFY = "qualify"
TRANSITION_GRADUATE = "graduate"
TRANSITION_DOWNGRADE = "downgrade"
TRANSITION_INVALIDATE = "invalidate"

ALL_LIFECYCLE_TRANSITIONS = (
    TRANSITION_PROPOSE,
    TRANSITION_PROMOTE,
    TRANSITION_WEAKEN,
    TRANSITION_RETIRE,
    TRANSITION_REOPEN,
    TRANSITION_MERGE,
    TRANSITION_SPLIT,
    TRANSITION_QUALIFY,
    TRANSITION_GRADUATE,
    TRANSITION_DOWNGRADE,
    TRANSITION_INVALIDATE,
)

_TARGET_STATUS = {
    TRANSITION_PROPOSE: STATUS_PROPOSED,
    TRANSITION_QUALIFY: STATUS_QUALIFIED,
    TRANSITION_PROMOTE: STATUS_PROMOTED,
    TRANSITION_GRADUATE: STATUS_GRADUATED,
    TRANSITION_WEAKEN: STATUS_WEAKENED,
    TRANSITION_RETIRE: STATUS_RETIRED,
    TRANSITION_REOPEN: STATUS_REOPENED,
    TRANSITION_DOWNGRADE: STATUS_DOWNGRADED,
    TRANSITION_INVALIDATE: STATUS_INVALIDATED,
    TRANSITION_MERGE: None,
    TRANSITION_SPLIT: None,
}

_ALLOWED_TRANSITIONS = {
    None: {TRANSITION_PROPOSE},
    "": {TRANSITION_PROPOSE},
    "active": {
        TRANSITION_PROMOTE,
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_REOPEN,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_GRADUATE,
        TRANSITION_DOWNGRADE,
        TRANSITION_INVALIDATE,
    },
    STATUS_PROPOSED: {TRANSITION_QUALIFY, TRANSITION_RETIRE, TRANSITION_INVALIDATE},
    STATUS_QUALIFIED: {
        TRANSITION_PROMOTE,
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_REOPEN,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_GRADUATE,
        TRANSITION_DOWNGRADE,
        TRANSITION_INVALIDATE,
    },
    STATUS_PROMOTED: {
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_REOPEN,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_GRADUATE,
        TRANSITION_DOWNGRADE,
        TRANSITION_INVALIDATE,
    },
    STATUS_GRADUATED: {
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_REOPEN,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_DOWNGRADE,
        TRANSITION_INVALIDATE,
    },
    STATUS_WEAKENED: {
        TRANSITION_REOPEN,
        TRANSITION_RETIRE,
        TRANSITION_PROMOTE,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_DOWNGRADE,
        TRANSITION_INVALIDATE,
    },
    STATUS_REOPENED: {
        TRANSITION_QUALIFY,
        TRANSITION_PROMOTE,
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_GRADUATE,
        TRANSITION_INVALIDATE,
    },
    STATUS_DOWNGRADED: {
        TRANSITION_QUALIFY,
        TRANSITION_PROMOTE,
        TRANSITION_WEAKEN,
        TRANSITION_RETIRE,
        TRANSITION_MERGE,
        TRANSITION_SPLIT,
        TRANSITION_INVALIDATE,
    },
    STATUS_RETIRED: {TRANSITION_REOPEN, TRANSITION_INVALIDATE},
    STATUS_INVALIDATED: set(),
}

_KNOWN_LIFECYCLE_STATUS_SET = set(ALL_LIFECYCLE_STATUSES)


@dataclass
class LifecycleEvent:
    transition: str
    from_status: str
    to_status: str
    timestamp: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def allowed_transitions(status: Optional[str]) -> List[str]:
    allowed = _ALLOWED_TRANSITIONS.get(status, set())
    return sorted(allowed)


def is_lifecycle_status(status: Optional[str]) -> bool:
    return str(status or "").strip().lower() in _KNOWN_LIFECYCLE_STATUS_SET


def normalize_lifecycle_status(
    status: Optional[str],
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    default: str = STATUS_QUALIFIED,
) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in _KNOWN_LIFECYCLE_STATUS_SET:
        return normalized

    for event in reversed(list(history or [])):
        if not isinstance(event, dict):
            continue
        candidate = str(event.get("to_status") or event.get("status") or "").strip().lower()
        if candidate in _KNOWN_LIFECYCLE_STATUS_SET:
            return candidate
    return str(default or STATUS_QUALIFIED)


def apply_lifecycle_transition(
    *,
    status: Optional[str],
    transition: str,
    history: Optional[List[Dict[str, Any]]] = None,
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_transition = str(transition or "").strip().lower()
    if normalized_transition not in ALL_LIFECYCLE_TRANSITIONS:
        raise ValueError(f"Unknown lifecycle transition: {transition}")

    allowed = _ALLOWED_TRANSITIONS.get(status, set())
    if normalized_transition not in allowed:
        raise ValueError(
            f"Transition '{normalized_transition}' not allowed from status '{status}'"
        )

    now = timestamp or datetime.utcnow().isoformat()
    to_status = _TARGET_STATUS[normalized_transition] or str(status or "")
    event = LifecycleEvent(
        transition=normalized_transition,
        from_status=str(status or ""),
        to_status=to_status,
        timestamp=now,
        reason=str(reason or ""),
        metadata=dict(metadata or {}),
    )
    updated_history = list(history or [])
    updated_history.append(event.to_dict())
    return {
        "status": to_status,
        "event": event.to_dict(),
        "history": updated_history,
    }


def bootstrap_lifecycle(
    *,
    reason: str = "formal_commit",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    first = apply_lifecycle_transition(
        status=None,
        transition=TRANSITION_PROPOSE,
        history=[],
        reason=reason,
        timestamp=timestamp,
    )
    return apply_lifecycle_transition(
        status=first["status"],
        transition=TRANSITION_QUALIFY,
        history=first["history"],
        reason=reason,
        timestamp=timestamp,
    )

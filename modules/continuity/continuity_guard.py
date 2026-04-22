"""Guardrails for continuity snapshot inheritance during resume/persist."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


_ALLOWED_TOP_LEVEL_KEYS: Set[str] = {
    'agent_id',
    'identity',
    'goals',
    'agenda',
    'approved_experiments',
    'autobiographical_summary',
    'semantic_memory',
    'procedural_memory',
    'transfer_memory',
    'durable_identity',
    'active_commitments',
    'long_horizon_agenda',
}
_ALLOWED_IDENTITY_FIELDS: Set[str] = {
    'name',
    'traits',
    'values',
    'capabilities',
    'limitations',
}
_SET_IDENTITY_FIELDS: Set[str] = {'traits', 'values', 'capabilities', 'limitations'}


class IllegalInheritancePolicy(str, Enum):
    """How resume should react once illegal inheritance is detected."""

    REJECT_RESTORE = 'reject_restore'
    DEGRADED_RESTORE = 'degraded_restore'
    REQUIRE_MANUAL_CONFIRMATION = 'require_manual_confirmation'


@dataclass
class GuardVerdict:
    """Structured continuity guard verdict."""

    accepted: bool
    policy: IllegalInheritancePolicy
    reasons: List[str] = field(default_factory=list)
    identity_drift: List[str] = field(default_factory=list)
    goal_drift: List[str] = field(default_factory=list)
    illegal_state: List[str] = field(default_factory=list)
    sanitized_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accepted': bool(self.accepted),
            'policy': self.policy.value,
            'reasons': list(self.reasons),
            'identity_drift': list(self.identity_drift),
            'goal_drift': list(self.goal_drift),
            'illegal_state': list(self.illegal_state),
        }


def _to_set(v: Any) -> Set[str]:
    if isinstance(v, set):
        return {str(x) for x in v}
    if isinstance(v, list):
        return {str(x) for x in v}
    if isinstance(v, tuple):
        return {str(x) for x in v}
    if v is None:
        return set()
    return {str(v)}


def detect_identity_drift(snapshot_before: Dict[str, Any], snapshot_after: Dict[str, Any]) -> List[str]:
    """Detect suspicious shifts in identity fields during resume."""
    drifts: List[str] = []
    before_identity = (snapshot_before or {}).get('identity', {}) if isinstance(snapshot_before, dict) else {}
    after_identity = (snapshot_after or {}).get('identity', {}) if isinstance(snapshot_after, dict) else {}

    before_fields = before_identity.get('fields', {}) if isinstance(before_identity, dict) else {}
    after_fields = after_identity.get('fields', {}) if isinstance(after_identity, dict) else {}

    before_name = before_fields.get('name')
    after_name = after_fields.get('name')
    if before_name and after_name and str(before_name) != str(after_name):
        drifts.append(f'identity.name mismatch: {before_name} -> {after_name}')

    for field in sorted(_SET_IDENTITY_FIELDS):
        before_set = _to_set(before_fields.get(field))
        after_set = _to_set(after_fields.get(field))
        if before_set and after_set and before_set.isdisjoint(after_set):
            drifts.append(f'identity.{field} disjoint with prior state')

    return drifts


def detect_goal_drift(snapshot_before: Dict[str, Any], snapshot_after: Dict[str, Any]) -> List[str]:
    """Detect goal-level continuity breaks where active goals are replaced wholesale."""
    drifts: List[str] = []
    before_goals = snapshot_before.get('goals', []) if isinstance(snapshot_before, dict) else []
    after_goals = snapshot_after.get('goals', []) if isinstance(snapshot_after, dict) else []

    before_active = {
        str(goal.get('goal_id'))
        for goal in before_goals
        if isinstance(goal, dict) and str(goal.get('status', 'active')) == 'active' and goal.get('goal_id')
    }
    after_active = {
        str(goal.get('goal_id'))
        for goal in after_goals
        if isinstance(goal, dict) and str(goal.get('status', 'active')) == 'active' and goal.get('goal_id')
    }

    if before_active and after_active and before_active.isdisjoint(after_active):
        drifts.append('all active goals replaced without overlap')

    return drifts


def _sanitize_snapshot(snapshot_after: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    sanitized: Dict[str, Any] = {}
    illegal: List[str] = []
    if not isinstance(snapshot_after, dict):
        return {}, ['snapshot is not dict']

    for key, value in snapshot_after.items():
        if key == 'experiments':
            experiments = value if isinstance(value, list) else []
            approved: List[Dict[str, Any]] = []
            for item in experiments:
                if not isinstance(item, dict):
                    illegal.append('approved_experiment_type_pollution')
                    continue
                status = str(item.get('status', '') or '').strip().lower()
                metadata = item.get('metadata', {}) if isinstance(item.get('metadata', {}), dict) else {}
                if bool(metadata.get('approved', False)) or status == 'completed':
                    approved.append(dict(item))
                else:
                    illegal.append(f'unapproved_experiment:{item.get("exp_id", "unknown")}')
            sanitized['approved_experiments'] = approved
            continue
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            illegal.append(f'top_level_field_pollution:{key}')
            continue
        if key in {'semantic_memory', 'procedural_memory', 'transfer_memory', 'durable_identity', 'autobiographical_summary'} and not isinstance(value, dict):
            illegal.append(f'{key}_type_pollution')
            sanitized[key] = {}
            continue
        if key in {'active_commitments', 'long_horizon_agenda', 'approved_experiments'} and not isinstance(value, list):
            illegal.append(f'{key}_type_pollution')
            sanitized[key] = []
            continue
        sanitized[key] = value

    identity = sanitized.get('identity', {})
    if isinstance(identity, dict):
        fields = identity.get('fields', {})
        if isinstance(fields, dict):
            clean_fields: Dict[str, Any] = {}
            for field_name, field_value in fields.items():
                if field_name not in _ALLOWED_IDENTITY_FIELDS:
                    illegal.append(f'identity_field_pollution:{field_name}')
                    continue
                if field_name in _SET_IDENTITY_FIELDS and not isinstance(field_value, (list, set, tuple)):
                    illegal.append(f'identity_type_pollution:{field_name}')
                    clean_fields[field_name] = []
                    continue
                clean_fields[field_name] = field_value
            identity['fields'] = clean_fields
            sanitized['identity'] = identity

    return sanitized, illegal


def detect_illegal_state_inheritance(snapshot_before: Dict[str, Any], snapshot_after: Dict[str, Any]) -> List[str]:
    """Detect field pollution/type pollution that should not be inherited."""
    _ = snapshot_before
    _, illegal = _sanitize_snapshot(snapshot_after)
    return illegal


def validate_resume(
    snapshot_before: Dict[str, Any],
    snapshot_after: Dict[str, Any],
    *,
    illegal_policy: IllegalInheritancePolicy = IllegalInheritancePolicy.DEGRADED_RESTORE,
) -> GuardVerdict:
    """Validate whether resume snapshot can be inherited safely."""
    sanitized, illegal = _sanitize_snapshot(snapshot_after)
    identity_drift = detect_identity_drift(snapshot_before, sanitized)
    goal_drift = detect_goal_drift(snapshot_before, sanitized)

    reasons = list(identity_drift) + list(goal_drift) + list(illegal)

    if illegal and illegal_policy == IllegalInheritancePolicy.REJECT_RESTORE:
        return GuardVerdict(
            accepted=False,
            policy=illegal_policy,
            reasons=reasons,
            identity_drift=identity_drift,
            goal_drift=goal_drift,
            illegal_state=illegal,
            sanitized_snapshot=sanitized,
        )

    if illegal and illegal_policy == IllegalInheritancePolicy.REQUIRE_MANUAL_CONFIRMATION:
        return GuardVerdict(
            accepted=False,
            policy=illegal_policy,
            reasons=reasons,
            identity_drift=identity_drift,
            goal_drift=goal_drift,
            illegal_state=illegal,
            sanitized_snapshot=sanitized,
        )

    return GuardVerdict(
        accepted=True,
        policy=illegal_policy,
        reasons=reasons,
        identity_drift=identity_drift,
        goal_drift=goal_drift,
        illegal_state=illegal,
        sanitized_snapshot=sanitized,
    )

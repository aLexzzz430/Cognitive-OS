from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from core.cognition.outcome_model_update import OutcomeModelUpdateResult


GOAL_PRESSURE_VERSION = "goal_pressure.v0.1"


@dataclass(frozen=True)
class GoalPressureUpdateResult:
    schema_version: str
    created_or_updated: bool
    goal_id: str
    pressure_type: str
    priority: float
    goal_patch: Dict[str, Any]
    audit_event: Dict[str, Any]
    reason: str

    def to_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_or_updated": self.created_or_updated,
            "goal_id": self.goal_id,
            "pressure_type": self.pressure_type,
            "priority": self.priority,
            "goal_patch_keys": sorted(self.goal_patch.keys()),
            "audit_event": dict(self.audit_event),
            "reason": self.reason,
        }


def build_goal_pressure_update(
    *,
    outcome_update: OutcomeModelUpdateResult,
    existing_state: Optional[Mapping[str, Any]] = None,
    episode: int = 0,
    tick: int = 0,
) -> GoalPressureUpdateResult:
    """Convert self-model learning pressure into goal_stack state patches."""
    state = existing_state if isinstance(existing_state, Mapping) else {}
    goal_stack = _mapping(state.get("goal_stack"))
    action_name = str(outcome_update.action_name or "unknown_action").strip() or "unknown_action"
    capability = _mapping(outcome_update.self_patch.get("self_summary.capability_estimate")).get(action_name, {})
    capability = _mapping(capability)
    pressure_type = _pressure_type(outcome_update=outcome_update, capability=capability)
    if not pressure_type:
        return GoalPressureUpdateResult(
            schema_version=GOAL_PRESSURE_VERSION,
            created_or_updated=False,
            goal_id="",
            pressure_type="none",
            priority=0.0,
            goal_patch={},
            audit_event={},
            reason="no_goal_pressure_threshold_met",
        )

    if pressure_type == "capability_repair":
        goal_id = _stable_goal_id("capability_repair", action_name)
        priority = _capability_repair_priority(capability=capability)
        title = f"Improve unreliable action: {action_name}"
        objective = (
            f"Investigate repeated failures for action '{action_name}', isolate the failure boundary, "
            "and define a verifier-gated improvement or refusal rule."
        )
        success_condition = "Failure mode is explained with evidence and converted into a bounded guard, regression, or approved fix."
        allowed_actions = ["read_logs", "read_reports", "run_readonly_analysis", "run_eval", "write_report"]
        forbidden_actions = ["modify_core_runtime_without_approval", "promote_failed_skill_without_verifier"]
    else:
        goal_id = _stable_goal_id("skill_candidate", action_name)
        priority = _skill_candidate_priority(capability=capability)
        title = f"Evaluate reusable skill: {action_name}"
        objective = (
            f"Compile repeated verified successes for action '{action_name}' into a candidate skill card "
            "with applicability, negative examples, and verifier requirements."
        )
        success_condition = "Skill candidate exists only if repeated verified success and explicit failure boundaries are documented."
        allowed_actions = ["read_logs", "read_reports", "run_readonly_analysis", "write_report", "propose_skill_candidate"]
        forbidden_actions = ["install_skill_without_review", "treat_single_success_as_skill"]

    subgoals = _upsert_subgoal(
        goal_stack.get("subgoals"),
        {
            "schema_version": GOAL_PRESSURE_VERSION,
            "goal_id": goal_id,
            "source": "self_model_learning_pressure",
            "pressure_type": pressure_type,
            "title": title,
            "objective": objective,
            "status": "active",
            "priority": priority,
            "permission_level": "L1",
            "success_condition": success_condition,
            "allowed_actions": allowed_actions,
            "forbidden_actions": forbidden_actions,
            "evidence_refs": _evidence_refs(outcome_update),
            "created_from": {
                "action": action_name,
                "outcome": outcome_update.outcome,
                "verified": outcome_update.verified,
                "episode": int(episode or 0),
                "tick": int(tick or 0),
            },
            "metadata": {
                "capability": capability,
                "risk_delta_seen": _patch_delta(outcome_update.learning_patch, "risk_delta"),
                "confidence_delta_seen": _patch_delta(outcome_update.learning_patch, "confidence_delta"),
            },
        },
    )
    status = _mapping(goal_stack.get("goal_status"))
    status[goal_id] = {
        "status": "active",
        "source": "goal_pressure",
        "pressure_type": pressure_type,
        "last_action": action_name,
        "last_outcome": outcome_update.outcome,
        "last_verified": outcome_update.verified,
        "last_episode": int(episode or 0),
        "last_tick": int(tick or 0),
    }
    priority_map = _mapping(goal_stack.get("goal_priority"))
    priority_map[goal_id] = priority
    history = _rolling_history(
        goal_stack.get("goal_history"),
        {
            "schema_version": GOAL_PRESSURE_VERSION,
            "event_type": "goal_pressure_update",
            "goal_id": goal_id,
            "pressure_type": pressure_type,
            "priority": priority,
            "action": action_name,
            "outcome": outcome_update.outcome,
            "verified": outcome_update.verified,
            "episode": int(episode or 0),
            "tick": int(tick or 0),
        },
    )
    patch = {
        "goal_stack.subgoals": subgoals,
        "goal_stack.goal_status": status,
        "goal_stack.goal_priority": priority_map,
        "goal_stack.goal_history": history,
    }
    audit_event = {
        "event_type": "goal_pressure_update",
        "schema_version": GOAL_PRESSURE_VERSION,
        "episode": int(episode or 0),
        "tick": int(tick or 0),
        "data": {
            "goal_id": goal_id,
            "pressure_type": pressure_type,
            "priority": priority,
            "action": action_name,
            "outcome": outcome_update.outcome,
        },
        "source_module": "core.cognition",
        "source_stage": "evidence_commit",
    }
    return GoalPressureUpdateResult(
        schema_version=GOAL_PRESSURE_VERSION,
        created_or_updated=True,
        goal_id=goal_id,
        pressure_type=pressure_type,
        priority=priority,
        goal_patch=patch,
        audit_event=audit_event,
        reason="self_model_learning_pressure",
    )


def _pressure_type(*, outcome_update: OutcomeModelUpdateResult, capability: Mapping[str, Any]) -> str:
    if outcome_update.outcome == "failure":
        return "capability_repair"
    verified_successes = _safe_int(capability.get("verified_successes"))
    if outcome_update.verified and verified_successes >= 2:
        return "skill_candidate"
    return ""


def _capability_repair_priority(*, capability: Mapping[str, Any]) -> float:
    attempts = max(1, _safe_int(capability.get("attempts")))
    failures = _safe_int(capability.get("failures"))
    reliability = _safe_float(capability.get("reliability"), default=0.0)
    pressure = 0.72 + min(0.18, failures / attempts * 0.18) + min(0.08, (1.0 - reliability) * 0.08)
    return round(max(0.0, min(1.0, pressure)), 4)


def _skill_candidate_priority(*, capability: Mapping[str, Any]) -> float:
    verified_successes = _safe_int(capability.get("verified_successes"))
    reliability = _safe_float(capability.get("reliability"), default=0.5)
    pressure = 0.66 + min(0.18, verified_successes * 0.045) + min(0.08, reliability * 0.08)
    return round(max(0.0, min(1.0, pressure)), 4)


def _stable_goal_id(kind: str, action_name: str) -> str:
    safe_action = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in action_name.lower())
    return f"goal:{kind}:{safe_action or 'unknown_action'}"


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _upsert_subgoal(value: Any, new_goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    goal_id = str(new_goal.get("goal_id") or "")
    out: List[Dict[str, Any]] = []
    replaced = False
    for row in rows:
        if str(row.get("goal_id") or "") == goal_id:
            merged = dict(row)
            merged.update(new_goal)
            out.append(merged)
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(dict(new_goal))
    return out[-80:]


def _rolling_history(value: Any, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    rows.append(dict(event))
    return rows[-80:]


def _evidence_refs(outcome_update: OutcomeModelUpdateResult) -> List[str]:
    refs: List[str] = []
    audit = outcome_update.audit_event.get("data") if isinstance(outcome_update.audit_event, Mapping) else {}
    if isinstance(audit, Mapping):
        for item in audit.get("evidence_refs", []) or []:
            if isinstance(item, str):
                refs.append(item)
    for item in outcome_update.learning_patch.get("learning_context.belief_updates", []) or []:
        if not isinstance(item, Mapping):
            continue
        for ref in item.get("evidence_refs", []) or []:
            if isinstance(ref, Mapping) and ref.get("evidence_id"):
                refs.append(f"evidence:{ref.get('evidence_id')}")
    deduped: List[str] = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return deduped[-12:]


def _patch_delta(learning_patch: Mapping[str, Any], key: str) -> float:
    prediction_error = learning_patch.get("learning_context.prediction_error")
    if not isinstance(prediction_error, Mapping):
        return 0.0
    return _safe_float(prediction_error.get(key), default=0.0)

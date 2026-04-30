from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from core.runtime.failure_learning import failure_objects_to_behavior_rules
from core.runtime.runtime_modes import RuntimeMode, mode_policy_for_mode
from core.task_discovery.models import string_list


RUNTIME_BEHAVIOR_POLICY_VERSION = "conos.runtime_behavior_policy/v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _dict_list(value: Any) -> list[Dict[str, Any]]:
    return [dict(item) for item in _as_list(value) if isinstance(item, Mapping)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _float(value, default)))


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


@dataclass(frozen=True)
class RuntimeBehaviorDecision:
    schema_version: str
    runtime_mode: str
    reason: str
    selected_task: Dict[str, Any] = field(default_factory=dict)
    task_priority_updates: list[Dict[str, Any]] = field(default_factory=list)
    model_selection: Dict[str, Any] = field(default_factory=dict)
    llm_budget: Dict[str, Any] = field(default_factory=dict)
    permission_policy: Dict[str, Any] = field(default_factory=dict)
    learning_behavior_rules: Dict[str, Any] = field(default_factory=dict)
    governance_constraints: list[Dict[str, Any]] = field(default_factory=list)
    regression_tests: list[Dict[str, Any]] = field(default_factory=list)
    retrieval_objects: list[Dict[str, Any]] = field(default_factory=list)
    audit_events: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_mode": self.runtime_mode,
            "reason": self.reason,
            "selected_task": dict(self.selected_task),
            "task_priority_updates": [dict(item) for item in self.task_priority_updates],
            "model_selection": dict(self.model_selection),
            "llm_budget": dict(self.llm_budget),
            "permission_policy": dict(self.permission_policy),
            "learning_behavior_rules": dict(self.learning_behavior_rules),
            "governance_constraints": [dict(item) for item in self.governance_constraints],
            "regression_tests": [dict(item) for item in self.regression_tests],
            "retrieval_objects": [dict(item) for item in self.retrieval_objects],
            "audit_events": [dict(item) for item in self.audit_events],
        }


def derive_runtime_behavior_policy(
    state: Mapping[str, Any] | None,
    *,
    task_candidates: Sequence[Mapping[str, Any]] | None = None,
    learning_context: Mapping[str, Any] | None = None,
    explicit_mode: str = "",
) -> RuntimeBehaviorDecision:
    """Derive the next behavior envelope from self/world state.

    This is the concrete bridge from model state to action governance. It keeps
    the decision deterministic and auditable: goal pressure selects work,
    world uncertainty decides whether to think or act, self-model state adjusts
    budgets, and failure learning contributes behavior rules.
    """

    payload = _as_dict(state)
    self_summary = _as_dict(payload.get("self_summary") or payload.get("self_model") or payload.get("self_model_state"))
    world_summary = _as_dict(payload.get("world_summary") or payload.get("world_model") or payload.get("world_model_state"))
    telemetry = _as_dict(payload.get("telemetry_summary") or payload.get("runtime_health"))
    learning = _as_dict(learning_context or payload.get("learning_context"))
    failure_objects = _dict_list(learning.get("failure_objects")) or _dict_list(self_summary.get("failure_learning_objects"))
    behavior_rules = _as_dict(learning.get("failure_behavior_rules") or self_summary.get("failure_learning_behavior_rules"))
    if not behavior_rules:
        behavior_rules = failure_objects_to_behavior_rules(failure_objects)

    candidates = _candidate_tasks(payload, task_candidates)
    scored = [
        _score_task_candidate(
            candidate,
            self_summary=self_summary,
            world_summary=world_summary,
            behavior_rules=behavior_rules,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda row: (row["adjusted_priority"], row["task_id"]), reverse=True)
    selected = dict(scored[0]["task"]) if scored else {}
    selected_score = dict(scored[0]) if scored else {}
    selected.pop("_behavior_score", None)

    mode, reason = _select_mode(
        selected,
        self_summary=self_summary,
        world_summary=world_summary,
        telemetry=telemetry,
        explicit_mode=explicit_mode,
    )
    mode_policy = mode_policy_for_mode(mode)
    llm_budget = _budget_for_state(mode_policy, self_summary=self_summary, world_summary=world_summary)
    model_selection = _model_selection_for_state(mode_policy, self_summary=self_summary, world_summary=world_summary)
    permission_policy = _permission_policy_for_task(mode_policy, selected)
    audit_events = [
        {
            "event_type": "runtime_behavior_policy_decision",
            "schema_version": RUNTIME_BEHAVIOR_POLICY_VERSION,
            "runtime_mode": mode.value,
            "reason": reason,
            "selected_task_id": str(selected.get("task_id") or selected.get("goal_id") or ""),
            "learning_rule_count": int(behavior_rules.get("rule_count", 0) or 0),
            "world_uncertainty": _clamp(world_summary.get("uncertainty_estimate") or world_summary.get("uncertainty")),
            "world_risk": _clamp(world_summary.get("risk_estimate") or world_summary.get("risk")),
            "self_confidence": _clamp(self_summary.get("confidence"), 0.5),
        }
    ]
    return RuntimeBehaviorDecision(
        schema_version=RUNTIME_BEHAVIOR_POLICY_VERSION,
        runtime_mode=mode.value,
        reason=reason,
        selected_task=selected,
        task_priority_updates=[
            {
                "task_id": row["task_id"],
                "base_priority": row["base_priority"],
                "adjusted_priority": row["adjusted_priority"],
                "reasons": row["reasons"],
            }
            for row in scored
        ],
        model_selection=model_selection,
        llm_budget=llm_budget,
        permission_policy=permission_policy,
        learning_behavior_rules=behavior_rules,
        governance_constraints=_dict_list(behavior_rules.get("governance_constraints")),
        regression_tests=_dict_list(behavior_rules.get("regression_tests")),
        retrieval_objects=_dict_list(behavior_rules.get("retrieval_objects")),
        audit_events=audit_events,
    )


def _candidate_tasks(
    state: Mapping[str, Any],
    task_candidates: Sequence[Mapping[str, Any]] | None,
) -> list[Dict[str, Any]]:
    if task_candidates is not None:
        return [dict(item) for item in task_candidates if isinstance(item, Mapping)]
    goal_stack = _as_dict(state.get("goal_stack"))
    subgoals = _dict_list(goal_stack.get("subgoals"))
    if subgoals:
        return subgoals
    return _dict_list(state.get("task_candidates"))


def _score_task_candidate(
    candidate: Mapping[str, Any],
    *,
    self_summary: Mapping[str, Any],
    world_summary: Mapping[str, Any],
    behavior_rules: Mapping[str, Any],
) -> Dict[str, Any]:
    task = dict(candidate)
    task_id = _text(task.get("task_id") or task.get("goal_id") or task.get("title") or task.get("objective"))
    base = _clamp(task.get("priority"), 0.5)
    pressure_type = _lower(task.get("pressure_type") or task.get("source") or "")
    objective = _lower(" ".join([_text(task.get("objective")), _text(task.get("title")), _text(task.get("gap"))]))
    allowed = set(string_list(task.get("allowed_actions")))
    reasons: list[str] = []
    bonus = 0.0
    penalty = 0.0

    if (_dict_list(self_summary.get("recent_failures")) or string_list(self_summary.get("error_flags"))) and (
        "self" in pressure_type or "failure" in pressure_type or "repair" in objective
    ):
        bonus += 0.16
        reasons.append("self_model_failure_pressure")
    if (
        _clamp(world_summary.get("uncertainty_estimate") or world_summary.get("uncertainty")) >= 0.65
        or _dict_list(world_summary.get("latent_hypotheses"))
    ) and ("world" in pressure_type or "hypothesis" in pressure_type or "uncertainty" in objective):
        bonus += 0.18
        reasons.append("world_model_uncertainty_pressure")
    preferred_actions = set(_as_dict(behavior_rules.get("preferred_actions")).keys())
    avoided_actions = set(_as_dict(behavior_rules.get("avoided_actions")).keys())
    blocked_actions = set(string_list(behavior_rules.get("blocked_actions")))
    if allowed & preferred_actions:
        bonus += 0.10
        reasons.append("failure_learning_prefers_allowed_action")
    if allowed & avoided_actions:
        penalty += 0.12
        reasons.append("failure_learning_avoids_allowed_action")
    if allowed & blocked_actions:
        penalty += 0.40
        reasons.append("failure_learning_blocks_allowed_action")
    penalty += 0.20 * _clamp(task.get("distraction_penalty"))
    penalty += 0.15 * _clamp(task.get("risk"))
    adjusted = _clamp(base + bonus - penalty, base)
    task["_behavior_score"] = {
        "base_priority": round(base, 6),
        "adjusted_priority": round(adjusted, 6),
        "reasons": reasons,
    }
    return {
        "task_id": task_id,
        "task": task,
        "base_priority": round(base, 6),
        "adjusted_priority": round(adjusted, 6),
        "reasons": reasons,
    }


def _select_mode(
    selected_task: Mapping[str, Any],
    *,
    self_summary: Mapping[str, Any],
    world_summary: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    explicit_mode: str,
) -> tuple[RuntimeMode, str]:
    explicit = _text(explicit_mode).upper()
    if explicit in RuntimeMode.__members__:
        return RuntimeMode[explicit], "explicit_runtime_mode"
    if string_list(telemetry.get("anomaly_flags")):
        return RuntimeMode.WAITING_HUMAN, "runtime_anomaly_requires_human_review"
    uncertainty = _clamp(world_summary.get("uncertainty_estimate") or world_summary.get("uncertainty"))
    risk = _clamp(world_summary.get("risk_estimate") or world_summary.get("risk"))
    if risk >= 0.85:
        return RuntimeMode.WAITING_HUMAN, "world_risk_above_autonomy_boundary"
    if uncertainty >= 0.65 or risk >= 0.72 or _dict_list(world_summary.get("latent_hypotheses")):
        return RuntimeMode.DEEP_THINK, "world_model_uncertainty_requires_deep_think"
    if not selected_task:
        return RuntimeMode.IDLE, "no_selected_goal_pressure"
    allowed = set(string_list(selected_task.get("allowed_actions")))
    if {"edit_in_mirror", "apply_patch", "execute", "run_tests"} & allowed:
        if bool(selected_task.get("requires_human_approval", False)):
            return RuntimeMode.WAITING_HUMAN, "selected_task_requires_human_approval"
        return RuntimeMode.ROUTINE_RUN, "selected_task_can_run_under_governed_routine"
    if _clamp(self_summary.get("adaptation_readiness"), 0.5) >= 0.72 and allowed & {"propose_patch", "write_report"}:
        return RuntimeMode.CREATING, "self_model_adaptation_pressure_can_create_candidates"
    return RuntimeMode.IDLE, "selected_task_is_read_only_background_work"


def _budget_for_state(
    mode_policy: Mapping[str, Any],
    *,
    self_summary: Mapping[str, Any],
    world_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    budget = dict(_as_dict(mode_policy.get("llm_budget")))
    tight = _lower(self_summary.get("resource_tightness")) in {"tight", "critical"} or bool(self_summary.get("budget_tight"))
    if tight:
        budget["max_llm_calls"] = min(int(budget.get("max_llm_calls", 0) or 0), 1)
        budget["max_completion_tokens"] = min(int(budget.get("max_completion_tokens", 0) or 0), 512)
        budget["escalation_allowed"] = False
        budget["budget_reason"] = "self_model_resource_tightness"
    elif _clamp(world_summary.get("uncertainty_estimate") or world_summary.get("uncertainty")) >= 0.75:
        budget["max_wall_clock_seconds"] = max(float(budget.get("max_wall_clock_seconds", 0) or 0), 180.0)
        budget["budget_reason"] = "world_model_high_uncertainty"
    return budget


def _model_selection_for_state(
    mode_policy: Mapping[str, Any],
    *,
    self_summary: Mapping[str, Any],
    world_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    selection = dict(_as_dict(mode_policy.get("model_selection")))
    if _lower(self_summary.get("resource_tightness")) in {"tight", "critical"} or bool(self_summary.get("budget_tight")):
        selection.update(
            {
                "model_tier": "local_small",
                "prefer_strongest_model": False,
                "allow_cloud_escalation": False,
                "selection_reason": "self_model_budget_tight",
            }
        )
    elif _clamp(world_summary.get("uncertainty_estimate") or world_summary.get("uncertainty")) >= 0.65:
        selection.update(
            {
                "model_tier": "strong",
                "thinking_mode": "on_or_budgeted",
                "prefer_strongest_model": True,
                "allow_cloud_escalation": True,
                "selection_reason": "world_model_uncertainty",
            }
        )
    return selection


def _permission_policy_for_task(mode_policy: Mapping[str, Any], selected_task: Mapping[str, Any]) -> Dict[str, Any]:
    policy = dict(_as_dict(mode_policy.get("permission_policy")))
    if not selected_task:
        return policy
    permission_level = _text(selected_task.get("permission_level") or "L1").upper()
    if permission_level in {"L0", "L1"}:
        policy["allowed_capability_layers"] = sorted(set(policy.get("allowed_capability_layers", [])) | {"read"})
        policy["side_effects_allowed"] = False
    elif permission_level == "L2":
        policy["allowed_capability_layers"] = sorted(set(policy.get("allowed_capability_layers", [])) | {"read", "propose_patch"})
        policy["approval_required_capability_layers"] = sorted(
            set(policy.get("approval_required_capability_layers", [])) | {"execute", "sync_back"}
        )
        policy["side_effects_allowed"] = False
    else:
        policy["approval_required_capability_layers"] = sorted(
            set(policy.get("approval_required_capability_layers", [])) | {"execute", "network", "credential", "sync_back"}
        )
        policy["side_effects_allowed"] = False
    if bool(selected_task.get("requires_human_approval", False)):
        policy["approval_required_capability_layers"] = sorted(
            set(policy.get("approval_required_capability_layers", [])) | set(policy.get("allowed_capability_layers", []))
        )
        policy["side_effects_allowed"] = False
    return policy

from __future__ import annotations

from typing import Any, Dict, List

from core.cognition.model_influence import ModelInfluenceInput, apply_cognitive_model_influence
from core.orchestration.runtime_stage_contracts import (
    Stage2PlanConstraintsInput,
    Stage2SelfModelSuppressionInput,
)
from planner.constraint_policy import ConstraintInput, apply_plan_constraints
from self_model.action_policy import SuppressionInput, apply_self_model_suppression


def run_stage2_plan_constraints(loop: Any, stage_input: Stage2PlanConstraintsInput) -> List[Dict[str, Any]]:
    constraint_result = apply_plan_constraints(
        ConstraintInput(
            candidate_actions=stage_input.candidate_actions,
            obs_before=stage_input.obs_before,
            has_plan=loop._plan_state.has_plan,
            current_step=loop._plan_state.current_step if loop._plan_state.has_plan else None,
            tick=loop._tick,
            episode=loop._episode,
            mechanism_control_summary=(
                loop._last_mechanism_runtime_view.get("mechanism_control_summary", {})
                if isinstance(loop._last_mechanism_runtime_view, dict)
                else {}
            ),
        )
    )
    if constraint_result.pending_replan_patch is not None:
        loop._pending_replan = constraint_result.pending_replan_patch
    if constraint_result.viability_entry:
        loop._candidate_viability_log.append(dict(constraint_result.viability_entry))
        loop._governance_log.append(dict(constraint_result.viability_entry))
    return constraint_result.filtered_candidates


def run_stage2_self_model_suppression(
    loop: Any,
    stage_input: Stage2SelfModelSuppressionInput,
) -> List[Dict[str, Any]]:
    candidate_actions = stage_input.candidate_actions
    continuity_snapshot = stage_input.continuity_snapshot
    obs_before = stage_input.obs_before if isinstance(stage_input.obs_before, dict) else {}
    enriched_snapshot = dict(continuity_snapshot or {})
    novel_api = obs_before.get("novel_api", {}) if isinstance(obs_before.get("novel_api", {}), dict) else {}
    for key in ("visible_functions", "discovered_functions", "available_functions"):
        if key not in enriched_snapshot and isinstance(novel_api.get(key, []), list):
            enriched_snapshot[key] = list(novel_api.get(key, []))
    if "perception" not in enriched_snapshot and isinstance(obs_before.get("perception", {}), dict):
        enriched_snapshot["perception"] = dict(obs_before.get("perception", {}))
    if "world_model_summary" not in enriched_snapshot and isinstance(obs_before.get("world_model", {}), dict):
        enriched_snapshot["world_model_summary"] = dict(obs_before.get("world_model", {}))
    recent_failures = sum(
        1
        for entry in loop._episode_trace[-5:]
        if float(entry.get("reward", 0.0) or 0.0) < 0.0
    )
    resource_state = getattr(loop, "_resource_state", None)
    suppression_result = apply_self_model_suppression(
        SuppressionInput(
            candidate_actions=candidate_actions,
            recent_failure_summary={"recent_failures": recent_failures},
            resource_state={
                "is_tight_budget": bool(getattr(resource_state, "is_tight_budget", lambda: False)())
                if resource_state is not None
                else False,
                "budget_band": resource_state.budget_band()
                if resource_state is not None and hasattr(resource_state, "budget_band")
                else "normal",
                "observation_mode": "unknown",
            },
            continuity_snapshot=enriched_snapshot,
            tick=loop._tick,
            episode=loop._episode,
        ),
        extract_action_function_name=loop._extract_action_function_name,
        infer_task_family=loop._infer_task_family,
        extract_phase_hint=loop._extract_phase_hint,
        reliability_tracker=getattr(loop, "_reliability_tracker", None),
    )
    loop._viability_audit_log.extend(suppression_result.audit_records)
    self_model_state: Dict[str, Any] = {}
    try:
        summary = loop._build_self_model_prediction_summary()
        if isinstance(summary, dict):
            self_model_state = dict(summary.get("self_model_state", {}) or summary)
            if "reliability_by_function" not in self_model_state:
                self_model_state["reliability_by_function"] = dict(summary.get("reliability_by_function", {}) or {})
            if "resource_tightness" not in self_model_state:
                self_model_state["resource_tightness"] = summary.get("resource_tightness", "normal")
            if "budget_tight" not in self_model_state:
                self_model_state["budget_tight"] = bool(summary.get("budget_tight", False))
            if "continuity_confidence" not in self_model_state:
                self_model_state["continuity_confidence"] = summary.get("continuity_confidence", 1.0)
    except Exception:
        self_model_state = {}

    local_mirror = obs_before.get("local_mirror", {}) if isinstance(obs_before.get("local_mirror", {}), dict) else {}
    learning_context = local_mirror.get("end_to_end_learning", {}) if isinstance(local_mirror.get("end_to_end_learning", {}), dict) else {}
    if learning_context:
        failure_objects = list(learning_context.get("failure_objects", []) or [])
        if failure_objects and "failure_learning_objects" not in self_model_state:
            self_model_state["failure_learning_objects"] = failure_objects
        behavior_rules = learning_context.get("failure_behavior_rules")
        if isinstance(behavior_rules, dict) and "failure_learning_behavior_rules" not in self_model_state:
            self_model_state["failure_learning_behavior_rules"] = dict(behavior_rules)

    world_model_state = {}
    if isinstance(enriched_snapshot.get("world_model_summary"), dict):
        world_model_state = dict(enriched_snapshot.get("world_model_summary", {}) or {})
    elif isinstance(enriched_snapshot.get("world_model"), dict):
        world_model_state = dict(enriched_snapshot.get("world_model", {}) or {})

    influence_result = apply_cognitive_model_influence(
        ModelInfluenceInput(
            candidate_actions=suppression_result.filtered_candidates,
            world_model_state=world_model_state,
            self_model_state=self_model_state,
            tick=int(getattr(loop, "_tick", 0) or 0),
            episode=int(getattr(loop, "_episode", 0) or 0),
        ),
        extract_action_name=lambda action: loop._extract_action_function_name(action, default="wait"),
    )
    loop._viability_audit_log.extend(influence_result.audit_records)
    return influence_result.candidate_actions


def materialize_stage2_prediction_fallback(loop: Any, candidate_actions: List[Dict[str, Any]]) -> None:
    """
    Ensure prediction signal exists before arbiter scoring.

    Heavy prediction may be unavailable; this fallback keeps the metadata shape
    stable and bounded so scoring can still reason over prediction fields.
    """
    if not bool(getattr(loop, "_prediction_enabled", False)):
        return
    for action in candidate_actions:
        if not isinstance(action, dict):
            continue
        meta = action.setdefault("_candidate_meta", {})
        if not isinstance(meta, dict):
            meta = {}
            action["_candidate_meta"] = meta
        if isinstance(meta.get("prediction"), dict):
            continue
        cf_delta = float(meta.get("counterfactual_delta", 0.0) or 0.0)
        cf_advantage = bool(meta.get("counterfactual_advantage", False))
        heuristic_success = max(0.0, min(1.0, 0.5 + (0.4 * cf_delta) + (0.08 if cf_advantage else 0.0)))
        heuristic_info_gain = max(0.0, min(1.0, abs(cf_delta)))
        meta["prediction"] = {
            "success": {"value": heuristic_success},
            "information_gain": {"value": heuristic_info_gain},
            "reward_sign": {"value": "positive" if heuristic_success >= 0.6 else "zero"},
            "risk_type": {"value": "execution_failure" if heuristic_success < 0.45 else "state_shift"},
            "overall_confidence": 0.42 if not cf_advantage else 0.58,
            "source": "counterfactual_fallback",
        }


def rank_counterfactual_candidates(candidate_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Promote counterfactual-advantaged candidates before final arbitration."""
    if not candidate_actions:
        return candidate_actions

    def _score(action: Dict[str, Any]) -> float:
        meta = action.get("_candidate_meta", {}) if isinstance(action, dict) else {}
        if not isinstance(meta, dict):
            return 0.0
        delta = float(meta.get("counterfactual_delta", 0.0) or 0.0)
        advantage = 0.2 if meta.get("counterfactual_advantage") else 0.0
        confidence = str(meta.get("counterfactual_confidence", "low")).lower()
        conf_bonus = {"high": 0.2, "medium": 0.1}.get(confidence, 0.0)
        return delta + advantage + conf_bonus

    ranked = sorted(candidate_actions, key=_score, reverse=True)
    for idx, action in enumerate(ranked):
        meta = action.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            meta["counterfactual_rank"] = idx
    return ranked

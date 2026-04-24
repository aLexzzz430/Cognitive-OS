from __future__ import annotations

from typing import Any, Dict, List

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
    return suppression_result.filtered_candidates


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

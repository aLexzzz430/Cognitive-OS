from __future__ import annotations

from typing import Any, Dict, Optional

from core.orchestration.stage_types import GovernanceStageInput, PlannerStageInput


def run_stage2_action_generation(
    loop: Any,
    obs_before: Dict[str, Any],
    surfaced: list,
    continuity_snapshot: Dict[str, Any],
    *,
    frame: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compatibility action-generation wrapper over the planner and governance stage modules."""
    frame = frame or loop._build_tick_context_frame(obs_before, continuity_snapshot)
    planner_out = loop._planner_stage.run(
        loop,
        PlannerStageInput(
            obs_before=obs_before,
            surfaced=surfaced,
            continuity_snapshot=continuity_snapshot,
            frame=frame,
        ),
    )
    action_to_use = planner_out.arm_action if planner_out.arm_meta.get("arm") != "base" else planner_out.base_action
    governance_out = loop._governance_stage.run(
        loop,
        GovernanceStageInput(
            action_to_use=action_to_use,
            planner_output=planner_out,
            continuity_snapshot=continuity_snapshot,
            obs_before=obs_before,
            surfaced=surfaced,
            frame=frame,
        ),
    )
    return {
        "raw_base_action": planner_out.raw_base_action,
        "base_action": planner_out.base_action,
        "arm_action": planner_out.arm_action,
        "arm_meta": planner_out.arm_meta,
        "plan_tick_meta": planner_out.plan_tick_meta,
        "deliberation_result": planner_out.deliberation_result,
        "candidate_actions": governance_out.candidate_actions,
        "decision_outcome": governance_out.decision_outcome,
        "action_to_use": governance_out.action_to_use,
        "governance_result": governance_out.governance_result,
        "governance_decision": governance_out.governance_result.get("selected_name"),
        "governance_reason": governance_out.governance_result.get("reason", ""),
        "policy_profile_object_id": loop._meta_control.policy_profile_object_id,
    }

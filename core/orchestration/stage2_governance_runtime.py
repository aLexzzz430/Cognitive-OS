from __future__ import annotations

import importlib
from typing import Any

from core.orchestration.governance_runtime import govern_action
from core.orchestration.governance_state import GovernanceState
from core.orchestration.runtime_stage_contracts import Stage2GovernanceInput
from core.orchestration.stage_types import GovernanceStageOutput
from core.orchestration.state_sync import StateSyncInput


def _append_local_machine_grounding_bridge_candidate(loop: Any, candidate_actions: Any, obs_before: Any) -> tuple[list[Any], dict[str, Any]]:
    candidates = list(candidate_actions or []) if isinstance(candidate_actions, list) else []
    if not isinstance(obs_before, dict):
        return candidates, {}
    try:
        action_grounding = importlib.import_module("integrations.local_machine.action_grounding")
        annotate_local_machine_patch_ranking = getattr(action_grounding, "annotate_local_machine_patch_ranking")
        build_local_machine_posterior_action_bridge_candidate = getattr(
            action_grounding,
            "build_local_machine_posterior_action_bridge_candidate",
        )
    except Exception:
        return candidates, {}
    candidates = annotate_local_machine_patch_ranking(
        candidates,
        obs_before,
        episode_trace=list(getattr(loop, "_episode_trace", []) or []),
    )
    bridge_candidate = build_local_machine_posterior_action_bridge_candidate(
        obs_before,
        episode_trace=list(getattr(loop, "_episode_trace", []) or []),
    )
    if not isinstance(bridge_candidate, dict):
        return candidates, {}

    def signature(action: Any) -> tuple[str, str]:
        if not isinstance(action, dict):
            return "", ""
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        fn = str(
            action.get("function_name")
            or action.get("action")
            or tool_args.get("function_name")
            or tool_args.get("action")
            or ""
        ).strip()
        kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else tool_args.get("kwargs", {})
        return fn, str(sorted(dict(kwargs or {}).items())) if isinstance(kwargs, dict) else ""

    bridge_sig = signature(bridge_candidate)
    if bridge_sig[0] and bridge_sig not in {signature(action) for action in candidates}:
        candidates.append(bridge_candidate)
    return candidates, bridge_candidate


def _materialize_selected_action_kwargs(loop: Any, action: Any, obs_before: Any) -> Any:
    if not isinstance(action, dict):
        return action
    synthesizer = getattr(loop, "_structured_answer_synthesizer", None)
    populate = getattr(synthesizer, "maybe_populate_action_kwargs", None)
    if not callable(populate):
        return action
    llm_client = None
    resolver = getattr(loop, "_resolve_structured_answer_llm_client", None)
    if callable(resolver):
        llm_client = resolver()
    populated = populate(
        action,
        obs_before if isinstance(obs_before, dict) else {},
        llm_client=llm_client,
    )
    return populated if isinstance(populated, dict) else action


def run_stage2_governance(loop: Any, stage_input: Stage2GovernanceInput) -> GovernanceStageOutput:
    """Run governance for the stage-2 candidate set and sync governance metadata."""
    action_to_use = stage_input.action_to_use
    candidate_actions = stage_input.candidate_actions
    arm_meta = stage_input.arm_meta
    continuity_snapshot = stage_input.continuity_snapshot
    obs_before = stage_input.obs_before
    decision_outcome = stage_input.decision_outcome
    frame = stage_input.frame
    decision_arbiter_selected = None
    if decision_outcome and decision_outcome.selected_candidate:
        decision_arbiter_selected = {
            "function_name": decision_outcome.selected_candidate.function_name,
            "action": loop._json_safe(decision_outcome.selected_candidate.action),
            "score": float(getattr(decision_outcome.selected_candidate, "score", 0.0) or 0.0),
            "reason": str(getattr(decision_outcome, "primary_reason", "") or ""),
        }

    fn_name = loop._extract_action_function_name(action_to_use, default="wait")
    candidate_actions, grounding_bridge_action = _append_local_machine_grounding_bridge_candidate(loop, candidate_actions, obs_before)
    terminal_completion_gate_active = False
    grounding_bridge_should_override = False
    if decision_outcome and decision_outcome.selected_candidate:
        selected_action = decision_outcome.selected_candidate.action
        selected_fn = decision_outcome.selected_candidate.function_name
        selected_source = (
            str(selected_action.get("_source", "") or "").strip().lower()
            if isinstance(selected_action, dict)
            else ""
        )
        selected_kind = (
            str(selected_action.get("kind", "") or "").strip().lower()
            if isinstance(selected_action, dict)
            else ""
        )
        visible_functions = (
            obs_before.get("novel_api", {}).get("visible_functions", [])
            if isinstance(obs_before.get("novel_api", {}), dict)
            else []
        )
        preserve_safe_baseline_action = (
            not visible_functions
            and fn_name == "inspect"
            and selected_source == "deliberation_probe"
            and (selected_kind == "probe" or "probe" in str(selected_fn or "").strip().lower())
        )
        if (
            selected_action
            and not preserve_safe_baseline_action
            and ((selected_fn and selected_fn != fn_name) or fn_name == "wait")
        ):
            action_to_use = selected_action
    if grounding_bridge_action:
        bridge_meta = grounding_bridge_action.get("_candidate_meta", {}) if isinstance(grounding_bridge_action.get("_candidate_meta", {}), dict) else {}
        terminal_completion_gate_active = bool(bridge_meta.get("terminal_completion_gate", False))
        grounding_bridge_should_override = bool(
            terminal_completion_gate_active
            or float(bridge_meta.get("fast_path_bonus", 0.0) or 0.0) > 0.0
            or float(bridge_meta.get("posterior_action_bonus", 0.0) or 0.0) > 0.0
            or float(bridge_meta.get("verify_after_patch_bonus", 0.0) or 0.0) > 0.0
            or float(bridge_meta.get("stalled_loop_recovery_bonus", 0.0) or 0.0) > 0.0
            or float(bridge_meta.get("progress_recovery_bonus", 0.0) or 0.0) > 0.0
        )
        if grounding_bridge_should_override:
            action_to_use = grounding_bridge_action

    governance_result = govern_action(
        loop=loop,
        action=action_to_use,
        candidate_actions=candidate_actions,
        continuity_snapshot=continuity_snapshot,
        frame=frame,
        reliability_port=loop._governance_ports,
        counterfactual_port=loop._governance_ports,
        governance_log_port=loop._governance_ports,
        organ_capability_port=loop._governance_ports,
        governance_state=GovernanceState(
            organ_failure_streaks=dict(loop._organ_failure_streaks),
            organ_capability_flags=dict(loop._organ_capability_flags),
            organ_failure_threshold=loop._organ_failure_threshold,
        ),
        meta_control_state={
            "arm_meta": arm_meta,
            "decision_outcome": decision_outcome,
            "obs_before": obs_before,
        },
    )
    loop._state_sync.sync(
        StateSyncInput(
            updates={
                "decision_context.governance_meta_control_snapshot_id": str(
                    governance_result.get("meta_control_snapshot_id", "") or ""
                ),
                "decision_context.governance_meta_control_inputs_hash": str(
                    governance_result.get("meta_control_inputs_hash", "") or ""
                ),
            },
            reason="governance_meta_control_snapshot_sync",
        )
    )
    action_to_use = governance_result.get("selected_action", action_to_use)
    selected_name = str(governance_result.get("selected_name") or "").strip()
    if grounding_bridge_should_override and grounding_bridge_action:
        action_to_use = grounding_bridge_action
        selected_name = str(
            grounding_bridge_action.get("function_name")
            or grounding_bridge_action.get("action")
            or selected_name
            or "wait"
        )
    action_to_use = loop._repair_action_function_name(action_to_use, selected_name)
    action_to_use = _materialize_selected_action_kwargs(loop, action_to_use, obs_before)
    governance_result["selected_action"] = action_to_use

    return GovernanceStageOutput(
        candidate_actions=candidate_actions,
        decision_outcome=decision_outcome,
        decision_arbiter_selected=decision_arbiter_selected,
        action_to_use=action_to_use,
        governance_result=governance_result,
    )

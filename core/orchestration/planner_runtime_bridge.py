from __future__ import annotations

from typing import Any, Dict, Optional

from core.orchestration.planner_runtime import PlannerPorts


def build_planner_ports(loop: Any) -> PlannerPorts:
    return PlannerPorts(
        plan_state=loop._plan_state,
        objective_decomposer=loop._objective_decomposer,
        plan_reviser=loop._plan_reviser,
        meta_control=loop._meta_control,
        build_tick_context_frame=lambda obs, continuity: loop._build_tick_context_frame(obs, continuity),
        extract_available_functions=loop._extract_available_functions,
        infer_task_family=loop._infer_task_family,
        ablation_flags_snapshot=loop._ablation_flags_snapshot,
        mark_continuity_task_completed=loop._mark_continuity_task_completed,
        mark_continuity_task_cancelled=loop._mark_continuity_task_cancelled,
        build_world_model_context=loop._build_world_model_context,
        build_world_model_transition_priors=loop._build_world_model_transition_priors,
        get_active_hypotheses=lambda: loop._hypotheses.get_active() if hasattr(loop, '_hypotheses') else [],
        get_reliability_tracker=lambda: loop._reliability_tracker if hasattr(loop, '_reliability_tracker') else None,
        get_episode=lambda: loop._episode,
        get_tick=lambda: loop._tick,
        get_max_ticks=lambda: loop.max_ticks,
        get_episode_reward=lambda: loop._episode_reward,
        get_episode_trace=lambda: loop._episode_trace,
        get_pending_replan=lambda: loop._pending_replan,
        get_world_provider_meta=lambda: loop._world_provider_meta,
        get_causal_ablation=lambda: loop._causal_ablation,
        get_learned_dynamics_predictor=lambda: getattr(loop, '_learned_dynamics_shadow_predictor', None),
        get_learned_dynamics_deployment_mode=lambda: getattr(loop, '_learned_dynamics_deployment_mode', 'shadow'),
        get_persistent_object_identity_tracker=lambda: getattr(loop, '_persistent_object_identity_tracker', None),
    )


def apply_planner_state_patch(loop: Any, patch: Dict[str, Any]) -> None:
    if not isinstance(patch, dict) or not patch:
        return
    update_context = patch.get('update_context')
    if isinstance(update_context, dict):
        loop._plan_state.update_context(
            tick=update_context.get('tick', loop._tick),
            reward=update_context.get('reward', loop._episode_reward),
            discovered_functions=update_context.get('discovered_functions', []),
        )
    step_transitions = patch.get('step_transitions')
    if isinstance(step_transitions, list) and step_transitions:
        loop._apply_step_transitions_with_feedback(step_transitions)
    else:
        if patch.get('advance_step'):
            loop._plan_state.advance_step()
        mark_failed_reason = patch.get('mark_failed_reason')
        if mark_failed_reason and loop._plan_state.current_step:
            loop._plan_state.fail_current_step(reason=str(mark_failed_reason))
    if patch.get('clear_plan'):
        loop._plan_state.clear_plan()
    if patch.get('set_plan') is not None:
        loop._plan_state.set_plan(patch['set_plan'])
    if 'pending_replan' in patch:
        loop._pending_replan = patch.get('pending_replan')


def consume_planner_runtime_result(
    loop: Any,
    runtime_out: Any,
    fallback_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state_patch = runtime_out.state_patch if isinstance(getattr(runtime_out, 'state_patch', None), dict) else {}
    decision_flags = runtime_out.decision_flags if isinstance(getattr(runtime_out, 'decision_flags', None), dict) else {}
    telemetry = runtime_out.telemetry if isinstance(getattr(runtime_out, 'telemetry', None), dict) else {}
    apply_planner_state_patch(loop, state_patch)
    selected_action = runtime_out.selected_action if isinstance(runtime_out.selected_action, dict) else (fallback_action or {})
    planner_payload = {
        'episode': int(loop._episode),
        'tick': int(loop._tick),
        'state_patch': dict(state_patch),
        'decision_flags': dict(decision_flags),
        'telemetry': dict(telemetry),
    }
    loop._last_planner_runtime_payload = planner_payload
    loop._planner_runtime_log.append(planner_payload)
    del loop._planner_runtime_log[:-60]
    return {
        'selected_action': selected_action,
        'state_patch': state_patch,
        'decision_flags': decision_flags,
        'telemetry': telemetry,
    }

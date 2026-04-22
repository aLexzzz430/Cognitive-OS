from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from core.main_loop_components import ARM_MODE_FULL
from core.orchestration.action_utils import extract_action_identity
from core.orchestration.governance_runtime import resolve_organ_control_outcome
from core.orchestration.governance_state import GovernanceState
from core.orchestration.self_model_stage import SelfModelRefreshInput, SelfModelStage
from core.orchestration.stage_types import StateSyncStageInput
from modules.world_model.events import EventType, WorldModelEvent
from trace_runtime import resolve_trace_runtime

_, _, DecisionSource = resolve_trace_runtime()


@dataclass(frozen=True)
class TickStartArtifacts:
    continuity_snapshot: Dict[str, Any]
    context: Dict[str, Any]
    tick_frame: Any
    tick_trace: Any


@dataclass(frozen=True)
class ExecutionOutcomeArtifacts:
    function_name: str
    terminal: bool
    step_success: bool


def begin_staged_tick(loop: Any, obs_before: Dict[str, Any]) -> TickStartArtifacts:
    loop._sync_llm_clients()
    loop._llm_calls_this_tick = 0
    loop._transfer_trace.next_decision_cycle()
    continuity_snapshot = loop._continuity.tick()
    loop._record_continuity_tick(continuity_snapshot)
    context = {'episode': loop._episode, 'tick': loop._tick, 'phase': 'active'}
    SelfModelStage.refresh(
        SelfModelRefreshInput(
            continuity_snapshot=continuity_snapshot,
            resource_state=getattr(loop, '_resource_state', None),
            self_model_facade=getattr(loop, '_self_model_facade', None),
            agent_id=str(getattr(loop, 'agent_id', 'agent')),
            arm_mode=str(getattr(loop, 'arm_mode', ARM_MODE_FULL)),
            teacher_present=bool(loop._teacher_allows_intervention()),
        )
    )
    tick_frame = loop._build_tick_context_frame(obs_before, continuity_snapshot)
    tick_trace = loop._causal_trace.new_trace(episode=loop._episode, tick=loop._tick)
    loop._causal_trace.set_observation_signature(
        tick_trace,
        loop._compute_observation_signature(obs_before),
    )
    loop._event_timeline.emit_stage_enter(loop._episode, loop._tick, 'retrieval')
    loop._event_timeline.emit_stage_enter(loop._episode, loop._tick, 'action_generation')
    loop._event_timeline.emit_stage_enter(loop._episode, loop._tick, 'execution')
    return TickStartArtifacts(
        continuity_snapshot=continuity_snapshot,
        context=context,
        tick_frame=tick_frame,
        tick_trace=tick_trace,
    )


def record_surfaced_candidates(loop: Any, tick_trace: Any, surfaced: List[Any]) -> None:
    for index, candidate in enumerate(surfaced):
        loop._causal_trace.add_candidate(
            trace=tick_trace,
            candidate_id=f"surfaced_{candidate.object_id}_{index}",
            source=DecisionSource.RETRIEVAL,
            function_name=getattr(candidate, 'function_name', ''),
            proposed_action={'object_id': candidate.object_id},
        )


def record_candidate_frontier(
    loop: Any,
    tick_trace: Any,
    candidate_actions: List[Dict[str, Any]],
    decision_outcome: Any,
    governance_result: Dict[str, Any],
) -> None:
    source_map = {
        'base_generation': DecisionSource.BASE_GENERATION,
        'skill_rewrite': DecisionSource.SKILL_REWRITE,
        'llm_rewrite': DecisionSource.LLM_REWRITE,
        'arm_evaluation': DecisionSource.ARM_EVALUATION,
        'recovery': DecisionSource.RECOVERY,
        'planner': DecisionSource.PLANNER,
        'wait_fallback': DecisionSource.WAIT_FALLBACK,
        'retrieval': DecisionSource.RETRIEVAL,
        'self_model': DecisionSource.SELF_MODEL,
        'history_reuse': DecisionSource.HISTORY_REUSE,
    }
    score_breakdown_by_signature: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if decision_outcome and isinstance(getattr(decision_outcome, 'score_breakdowns', None), list):
        for row in decision_outcome.score_breakdowns:
            if not isinstance(row, dict):
                continue
            signature = (
                str(row.get('source', 'base_generation')),
                str(row.get('function_name', 'wait')),
                str(row.get('kwargs_repr', '')),
            )
            score_breakdown_by_signature[signature] = row

    selected_signature = None
    if decision_outcome and decision_outcome.selected_candidate:
        selected_action = decision_outcome.selected_candidate.action
        selected_signature = (
            decision_outcome.selected_candidate.source.value,
            decision_outcome.selected_candidate.function_name,
            repr(
                (
                    selected_action.get('payload', {}).get('tool_args', {}).get('kwargs', {})
                    if isinstance(selected_action, dict)
                    else {}
                )
            ),
        )

    for index, candidate_action in enumerate(candidate_actions):
        payload = candidate_action.get('payload', {}) if isinstance(candidate_action, dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        function_name = tool_args.get('function_name', 'wait') if candidate_action.get('kind') != 'wait' else 'wait'
        source_name = candidate_action.get('_source', 'base_generation')
        candidate_id = f"generated_{loop._tick}_{index}_{source_name}_{function_name or 'wait'}"
        loop._causal_trace.add_candidate(
            trace=tick_trace,
            candidate_id=candidate_id,
            source=source_map.get(source_name, DecisionSource.BASE_GENERATION),
            function_name=function_name,
            proposed_action=candidate_action,
        )
        candidate_signature = (
            source_name,
            function_name,
            repr(tool_args.get('kwargs', {})),
        )
        breakdown_row = score_breakdown_by_signature.get(candidate_signature)
        if breakdown_row and isinstance(candidate_action, dict):
            meta = candidate_action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['decision_score_breakdown'] = breakdown_row
        if selected_signature and candidate_signature == selected_signature:
            loop._causal_trace.select_candidate(tick_trace, candidate_id)

    governance_payload = governance_result if isinstance(governance_result, dict) else {}
    governance_decision = str(
        governance_payload.get('selected_name')
        or governance_payload.get('selected_action')
        or 'unknown'
    )
    loop._causal_trace.set_governance(
        trace=tick_trace,
        decision=governance_decision,
        reason=str(governance_payload.get('reason', '') or ''),
        risk_assessment=governance_payload.get('risk'),
        opportunity_assessment=governance_payload.get('opportunity'),
        leak_gate_mode=governance_payload.get('leak_gate_mode'),
    )


def apply_execution_outcome(
    loop: Any,
    tick_trace: Any,
    *,
    action_to_use: Dict[str, Any],
    obs_before: Dict[str, Any],
    surfaced: List[Any],
    result: Dict[str, Any],
    reward: float,
) -> ExecutionOutcomeArtifacts:
    outcome = resolve_organ_control_outcome(
        reward=reward,
        result=result,
        governance_state=GovernanceState(
            organ_failure_streaks=dict(loop._organ_failure_streaks),
            organ_capability_flags=dict(loop._organ_capability_flags),
            organ_failure_threshold=loop._organ_failure_threshold,
        ),
        pending_organ_control_audit=loop._pending_organ_control_audit,
    )
    state_patch = outcome.get('state_patch')
    if state_patch:
        loop._organ_failure_streaks.update(dict(getattr(state_patch, 'organ_failure_streaks', {}) or {}))
        loop._organ_capability_flags.update(dict(getattr(state_patch, 'organ_capability_flags', {}) or {}))
    for event in outcome.get('governance_events', []):
        if isinstance(event, dict):
            loop._governance_ports.append_governance({
                'episode': loop._episode,
                'tick': loop._tick,
                **event,
            })
    for entry in outcome.get('audit_entries', []):
        if isinstance(entry, dict):
            loop._organ_control_audit_log.append(dict(entry))

    function_name = loop._extract_action_function_name(action_to_use, default='wait')
    terminal = bool(result.get('terminal', False) or result.get('done', False))
    step_success = bool(result.get('success', True))
    loop._causal_trace.set_execution(
        trace=tick_trace,
        success=step_success,
        terminal=terminal,
        reward=reward,
        error_type=result.get('error', {}).get('type') if isinstance(result.get('error'), dict) else None,
    )
    loop._causal_trace.set_final_action(tick_trace, action_to_use)
    loop._causal_trace.set_env_action(tick_trace, action_to_use)
    loop._event_timeline.emit_action_executed(
        episode=loop._episode,
        tick=loop._tick,
        function_name=function_name,
        success=step_success,
        terminal=terminal,
        reward=reward,
    )

    loop._resource_state.consume_tick()
    if function_name != 'wait':
        executable_functions = loop._collect_executable_function_names(obs_before)
        action_meta = action_to_use.get('_candidate_meta', {}) if isinstance(action_to_use.get('_candidate_meta', {}), dict) else {}
        action_identity = extract_action_identity(action_to_use, include_function_fallback=False)
        execution_context = {
            'task_family': loop._infer_task_family(obs_before),
            'phase': str(obs_before.get('phase', 'active')) if isinstance(obs_before, dict) else 'active',
            'observation_mode': str((obs_before.get('perception') or {}).get('coordinate_type', 'unknown')) if isinstance(obs_before, dict) else 'unknown',
            'resource_band': loop._resource_state.budget_band() if hasattr(loop._resource_state, 'budget_band') else 'normal',
            'available_functions': list(executable_functions),
            'discovered_functions': list(executable_functions),
            'visible_functions': list(executable_functions),
        }
        if action_identity:
            execution_context['action_identity'] = action_identity
        hidden_guidance = action_meta.get('hidden_state_guidance', {}) if isinstance(action_meta.get('hidden_state_guidance', {}), dict) else {}
        if hidden_guidance:
            execution_context['world_model_hidden_state'] = dict(hidden_guidance)
        if (not step_success or reward < 0.0) and hasattr(loop._reliability_tracker, 'record_failure_preference'):
            loop._reliability_tracker.record_failure_preference(
                action_identity or function_name,
                context=execution_context,
                action_meta=action_meta,
                reward=reward,
            )
        if loop._is_trackable_executable_function(function_name, executable_functions):
            loop._capability_profile.record_call(
                function_name,
                success=step_success,
                episode=loop._episode,
                allowed_functions=executable_functions,
                context=execution_context,
            )
            loop._reliability_tracker.record_action_type_outcome(function_name, step_success, context=execution_context)
        if hasattr(loop._reliability_tracker, 'record_teacher_dependence_event'):
            loop._reliability_tracker.record_teacher_dependence_event(
                teacher_present=bool(loop._teacher_allows_intervention()),
                success=bool(step_success),
            )
        else:
            loop._capability_profile.record_auxiliary_signal(
                channel='retrieval_provenance',
                identifier=function_name,
                success=step_success,
                episode=loop._episode,
                metadata={'reason': 'non_executable_action_type'},
            )
    loop._reliability_tracker.record_module_outcome('core_execution', success=step_success)

    if surfaced:
        for candidate in surfaced[:3]:
            hyp_id = candidate.object_id
            if hyp_id and loop._hypotheses.has_hypothesis(hyp_id):
                source = (
                    loop._grad_trigger_source.TEACHER
                    if loop._episode <= 2
                    else loop._grad_trigger_source.AGENT
                )
                loop._grad_tracker.on_object_consumed(hyp_id, source, loop._tick)
                was_beneficial = reward > 0
                loop._shared_store.record_consumption(hyp_id, loop._tick, loop._episode, was_beneficial=was_beneficial)
                loop._event_bus.emit(WorldModelEvent(
                    event_type=EventType.OBJECT_CONSUMED,
                    episode=loop._episode,
                    tick=loop._tick,
                    data={
                        'object_id': hyp_id,
                        'consumption_count': (loop._shared_store.get(hyp_id) or {}).get('consumption_count', 1),
                        'asset_status': loop._shared_store.get_asset_status(hyp_id),
                        'reward': reward,
                        'was_beneficial': was_beneficial,
                        'source_stage': 'retrieval',
                    },
                    source_stage='retrieval',
                ))
                if hyp_id.startswith('hyp_') or hyp_id.startswith('obj_'):
                    loop._capability_profile.record_auxiliary_signal(
                        channel='retrieval_provenance',
                        identifier=hyp_id,
                        success=was_beneficial,
                        episode=loop._episode,
                        metadata={'source_stage': 'retrieval'},
                    )

    return ExecutionOutcomeArtifacts(
        function_name=function_name,
        terminal=terminal,
        step_success=step_success,
    )


def finalize_staged_tick(
    loop: Any,
    tick_trace: Any,
    *,
    continuity_snapshot: Dict[str, Any],
    surfaced: List[Any],
    function_name: str,
    reward: float,
    terminal: bool,
) -> None:
    plan_summary = loop._plan_state.get_plan_summary()
    step_intent = loop._plan_state.get_intent_for_step()
    loop._causal_trace.set_context(
        trace=tick_trace,
        continuity_snapshot={'top_goal': getattr(continuity_snapshot.get('top_goal'), 'goal_id', None) if continuity_snapshot else None},
        retrieval_bundle_summary={
            'count': len(surfaced),
            'selected_ids': [
                str(getattr(candidate, 'object_id', '') or '')
                for candidate in surfaced[:5]
                if str(getattr(candidate, 'object_id', '') or '')
            ],
            'action_influence': (
                str(getattr(surfaced[0], 'action_influence', 'none') or 'none')
                if surfaced
                else 'none'
            ),
        },
    )
    loop._causal_trace.set_plan_context(tick_trace, plan_summary, step_intent)
    loop._event_timeline.emit_stage_exit(loop._episode, loop._tick, 'retrieval', {'surfaced': len(surfaced)})
    loop._event_timeline.emit_stage_exit(loop._episode, loop._tick, 'action_generation', {'action': function_name})
    loop._event_timeline.emit_stage_exit(loop._episode, loop._tick, 'execution', {'reward': reward, 'terminal': terminal})
    loop._record_llm_tick_summary()


def sync_tick_state(
    loop: Any,
    *,
    continuity_snapshot: Dict[str, Any],
    surfaced: List[Any],
    action_to_use: Dict[str, Any],
    result: Dict[str, Any],
    reward: float,
    terminal: bool,
) -> Dict[str, Any]:
    sync_out = loop._state_sync_stage.run(
        loop,
        StateSyncStageInput(
            continuity_snapshot=continuity_snapshot,
            surfaced=surfaced,
            action_to_use=action_to_use,
            result=result,
            reward=reward,
            terminal=terminal,
        ),
    )
    next_obs = sync_out.next_obs
    next_obs_terminal = bool(
        isinstance(next_obs, dict)
        and (next_obs.get('terminal', False) or next_obs.get('done', False))
    )
    final_result = result
    final_terminal = terminal
    if next_obs_terminal:
        final_terminal = True
        if isinstance(result, dict):
            final_result = dict(result)
            final_result['terminal'] = bool(final_result.get('terminal', False) or True)
            final_result['done'] = bool(final_result.get('done', False) or True)
            if not final_result.get('state') and isinstance(next_obs, dict):
                final_result['state'] = next_obs.get('state')
    return {
        'terminal': final_terminal,
        'reward': reward,
        'next_obs': next_obs,
        'result': final_result,
    }

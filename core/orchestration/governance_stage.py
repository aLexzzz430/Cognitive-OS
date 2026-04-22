from __future__ import annotations

from core.orchestration.action_utils import extract_action_click_identity
from core.orchestration.stage_types import GovernanceStageInput, GovernanceStageOutput
from modules.world_model.events import EventType, WorldModelEvent


def _candidate_function_name(loop, candidate) -> str:
    try:
        return str(loop._extract_action_function_name(candidate, default='') or '')
    except Exception:
        return ''


def _candidate_kind(candidate) -> str:
    if not isinstance(candidate, dict):
        return ''
    return str(candidate.get('kind', '') or '')


def _candidate_meta(candidate) -> dict:
    meta = candidate.get('_candidate_meta', {}) if isinstance(candidate, dict) else {}
    return meta if isinstance(meta, dict) else {}


def _recent_failed_click_search_active(loop) -> bool:
    episode_trace = list(getattr(loop, '_episode_trace', []) or [])
    recent_click_failures = 0
    for entry in reversed(episode_trace[-4:]):
        if not isinstance(entry, dict):
            continue
        action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
        fn_name = str(loop._extract_action_function_name(action, default='') or '')
        if fn_name != 'ACTION6':
            continue
        reward = float(entry.get('reward', 0.0) or 0.0)
        info_gain = float(entry.get('information_gain', 0.0) or 0.0)
        task_progress = entry.get('task_progress', {}) if isinstance(entry.get('task_progress', {}), dict) else {}
        positive = bool(task_progress.get('progressed', False) or task_progress.get('solved', False) or reward > 0.0)
        if not positive and reward <= 0.0 and info_gain <= 0.12:
            recent_click_failures += 1
    return recent_click_failures > 0


def _suppress_inspect_candidates_during_click_search(loop, candidates):
    if not isinstance(candidates, list) or not candidates:
        return candidates
    if not _recent_failed_click_search_active(loop):
        return candidates
    click_candidates = []
    inspect_candidates = []
    passthrough = []
    click_identities = set()
    for candidate in candidates:
        fn_name = _candidate_function_name(loop, candidate)
        kind = _candidate_kind(candidate)
        if fn_name == 'ACTION6':
            click_candidates.append(candidate)
            click_identity = extract_action_click_identity(candidate)
            if click_identity:
                click_identities.add(click_identity)
            continue
        if fn_name == 'inspect' or kind == 'inspect':
            inspect_candidates.append(candidate)
            continue
        passthrough.append(candidate)
    if len(click_candidates) < 2 or len(click_identities) < 2:
        return candidates
    filtered = list(click_candidates) + list(passthrough)
    for candidate in inspect_candidates:
        meta = _candidate_meta(candidate)
        meta['inspect_suppressed_for_click_search'] = True
        meta['inspect_suppression_reason'] = 'recent_click_failure_with_viable_click_frontier'
        candidate['_candidate_meta'] = meta
    return filtered if filtered else candidates


class GovernanceStage:
    """Stage-3 governance wrapper: normalization + arbiter + final gate selection."""

    def run(self, loop, stage_input: GovernanceStageInput) -> GovernanceStageOutput:
        loop._state_sync.sync(
            loop._state_sync_input_cls(
                updates={
                    'decision_context.policy_profile': loop._get_policy_profile(),
                    'decision_context.representation_profile': loop._get_representation_profile(),
                    'decision_context.policy_read_fallback_events': list(loop._meta_control.policy_read_fallback_events[-10:]),
                    'decision_context.planner_meta_control_snapshot_id': str(stage_input.planner_output.plan_tick_meta.get('meta_control_snapshot_id', '') or ''),
                    'decision_context.planner_meta_control_inputs_hash': str(stage_input.planner_output.plan_tick_meta.get('meta_control_inputs_hash', '') or ''),
                },
                reason='policy_profile_control_read',
            )
        )
        stage_input.planner_output.candidate_actions = loop._annotate_candidates_with_learning_updates(
            stage_input.planner_output.candidate_actions,
            stage_input.planner_output.decision_context,
        )
        apply_mechanism_control = getattr(loop, '_apply_mechanism_candidate_control', None)
        if callable(apply_mechanism_control):
            stage_input.planner_output.candidate_actions = apply_mechanism_control(
                candidate_actions=stage_input.planner_output.candidate_actions,
                decision_context=stage_input.planner_output.decision_context,
                obs_before=stage_input.obs_before,
            )
        if isinstance(stage_input.planner_output.decision_context, dict):
            decision_context = dict(stage_input.planner_output.decision_context)
            decision_context.setdefault('obs_before', stage_input.obs_before)
            stage_input.planner_output.decision_context = decision_context
        stage_input.planner_output.candidate_actions = _suppress_inspect_candidates_during_click_search(
            loop,
            stage_input.planner_output.candidate_actions,
        )
        decision_outcome = loop._decision_arbiter.decide(
            stage_input.planner_output.candidate_actions,
            stage_input.planner_output.decision_context,
        )
        governance_out = loop._stage2_governance_substage(
            action_to_use=stage_input.action_to_use,
            candidate_actions=stage_input.planner_output.candidate_actions,
            arm_meta=stage_input.planner_output.arm_meta,
            continuity_snapshot=stage_input.continuity_snapshot,
            obs_before=stage_input.obs_before,
            decision_outcome=decision_outcome,
            frame=stage_input.frame,
        )
        governance_result = governance_out.governance_result
        governance_candidates_before_norm = governance_result.get('governance_candidates_before_normalization', [])
        governance_candidates_after_norm = governance_result.get('governance_candidates_after_normalization', [])
        n_governance_candidates, n_non_wait_governance_candidates = loop._candidate_counts(
            [candidate.get('raw_action', {}) if isinstance(candidate, dict) else {} for candidate in governance_candidates_after_norm]
        )
        selected_is_wait = loop._extract_action_function_name(governance_out.action_to_use, default='wait') == 'wait'
        selected_valid_non_wait = not selected_is_wait and bool(
            loop._extract_action_function_name(governance_out.action_to_use, default='').strip()
        )
        metrics = dict(stage_input.planner_output.stage_metrics or {})
        loop._candidate_viability_log.append({
            'episode': loop._episode,
            'tick': loop._tick,
            'visible_functions': stage_input.planner_output.visible_functions,
            'discovered_functions': stage_input.planner_output.discovered_functions,
            'raw_base_action': loop._json_safe(stage_input.planner_output.raw_base_action),
            'candidate_generator_outputs': stage_input.planner_output.raw_candidates_snapshot,
            'after_plan_constraints': metrics.get('after_plan_constraints', []),
            'after_self_model_suppression': metrics.get('after_self_model', []),
            'after_counterfactual_rank': metrics.get('after_counterfactual_rank', []),
            'after_procedure_annotation': metrics.get('after_procedure_annotation', []),
            'decision_arbiter_selected': loop._json_safe(governance_out.decision_arbiter_selected),
            'governance_candidates_before_normalization': loop._json_safe(governance_candidates_before_norm),
            'governance_candidates_after_normalization': loop._json_safe(governance_candidates_after_norm),
            'governance_selected': loop._json_safe({
                'selected_name': governance_result.get('selected_name', 'wait'),
                'selected_action': governance_out.action_to_use,
                'reason': governance_result.get('reason', ''),
            }),
            'selected_is_wait': selected_is_wait,
            'selected_valid_non_wait': selected_valid_non_wait,
            'n_raw_candidates': metrics.get('n_raw_candidates', 0),
            'n_non_wait_raw_candidates': metrics.get('n_non_wait_raw_candidates', 0),
            'n_after_plan_constraints': metrics.get('n_after_plan_constraints', 0),
            'n_non_wait_after_plan_constraints': metrics.get('n_non_wait_after_plan_constraints', 0),
            'n_after_self_model': metrics.get('n_after_self_model', 0),
            'n_non_wait_after_self_model': metrics.get('n_non_wait_after_self_model', 0),
            'n_governance_candidates': n_governance_candidates,
            'n_non_wait_governance_candidates': n_non_wait_governance_candidates,
        })
        fn_name = loop._extract_action_function_name(governance_out.action_to_use, default='wait')
        loop._event_bus.emit(WorldModelEvent(
            event_type=EventType.ACTION_SELECTED,
            episode=loop._episode,
            tick=loop._tick,
            data={
                'function_name': fn_name,
                'arm': stage_input.planner_output.arm_meta.get('arm', 'base'),
                'governance_risk': governance_result.get('risk'),
                'governance_opportunity': governance_result.get('opportunity'),
                'governance_mode': governance_result.get('mode'),
                'governance_reason': governance_result.get('reason'),
                'surfaced_count': len(stage_input.surfaced or []),
                'decision_arbiter_decision': decision_outcome.primary_reason if decision_outcome else '',
            },
            source_stage='action_generation',
        ))
        return GovernanceStageOutput(
            candidate_actions=governance_out.candidate_actions,
            decision_outcome=decision_outcome,
            decision_arbiter_selected=governance_out.decision_arbiter_selected,
            action_to_use=governance_out.action_to_use,
            governance_result=governance_result,
        )

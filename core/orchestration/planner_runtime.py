from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from modules.world_model.protocol import WorldModelControlProtocol
from modules.world_model.learned_dynamics import (
    build_learned_dynamics_state_snapshot,
    _apply_prediction_to_snapshot,
)
from core.orchestration.action_utils import extract_action_function_name
from planner.constraint_policy import match_observation_pattern


@dataclass(frozen=True)
class PlannerPorts:
    """Explicit dependencies required by PlannerRuntime."""

    plan_state: Any
    objective_decomposer: Any
    plan_reviser: Any
    meta_control: Any
    build_tick_context_frame: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    extract_available_functions: Callable[[Dict[str, Any]], List[str]]
    infer_task_family: Callable[[], str]
    ablation_flags_snapshot: Callable[[], Dict[str, Any]]
    mark_continuity_task_completed: Callable[[Optional[str], str], None]
    mark_continuity_task_cancelled: Callable[[Optional[str], str], None]
    build_world_model_context: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]
    build_world_model_transition_priors: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]
    get_active_hypotheses: Callable[[], List[Any]]
    get_reliability_tracker: Callable[[], Optional[Any]]
    get_episode: Callable[[], int]
    get_tick: Callable[[], int]
    get_max_ticks: Callable[[], int]
    get_episode_reward: Callable[[], float]
    get_episode_trace: Callable[[], List[Any]]
    get_pending_replan: Callable[[], Optional[Dict[str, Any]]]
    get_world_provider_meta: Callable[[], Dict[str, Any]]
    get_causal_ablation: Callable[[], Any]
    get_learned_dynamics_predictor: Callable[[], Optional[Any]]
    get_learned_dynamics_deployment_mode: Callable[[], str]
    get_persistent_object_identity_tracker: Callable[[], Optional[Any]]


@dataclass(frozen=True)
class PlannerRuntimeResult:
    selected_action: Optional[Dict[str, Any]]
    state_patch: Dict[str, Any]
    decision_flags: Dict[str, Any]
    telemetry: Dict[str, Any]


class PlannerRuntime:
    """Planner runtime orchestration with explicit control/progress phases."""

    def __init__(self, ports: PlannerPorts) -> None:
        self._ports = ports
        self._last_lookahead_plan_value: Optional[float] = None

    def tick(
        self,
        *,
        phase: str,
        obs: Dict[str, Any],
        continuity_snapshot: Optional[Dict[str, Any]] = None,
        frame: Any = None,
        selected_action: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        reward: float = 0.0,
    ) -> PlannerRuntimeResult:
        if phase == 'control':
            return self._tick_control(obs=obs, continuity_snapshot=continuity_snapshot or {}, frame=frame)
        if phase == 'progress':
            return self._tick_progress(obs=obs, selected_action=selected_action or {}, result=result or {}, reward=reward)
        raise ValueError(f'Unsupported planner runtime phase: {phase}')

    def _finalize_result(
        self,
        *,
        selected_action: Optional[Dict[str, Any]],
        state_patch: Dict[str, Any],
        decision_flags: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> PlannerRuntimeResult:
        return PlannerRuntimeResult(
            selected_action=deepcopy(selected_action) if isinstance(selected_action, dict) else None,
            state_patch=deepcopy(state_patch),
            decision_flags=deepcopy(decision_flags),
            telemetry=deepcopy(telemetry),
        )

    def _tick_control(self, *, obs: Dict[str, Any], continuity_snapshot: Dict[str, Any], frame: Any) -> PlannerRuntimeResult:
        ports = self._ports
        episode = ports.get_episode()
        tick = ports.get_tick()
        plan_state = ports.plan_state

        frame = frame or ports.build_tick_context_frame(obs, continuity_snapshot or {})
        available_functions = ports.extract_available_functions(obs)
        episode_reward = ports.get_episode_reward()
        reward_trend = 'positive' if episode_reward > 0 else ('negative' if ports.get_episode_trace() else 'neutral')
        planner_controls = ports.meta_control.for_planner_replan(
            episode=episode,
            tick=tick,
            context={'gate': 'planner_replan', 'reward_trend': reward_trend},
        )
        planner_bias = float(planner_controls['planner_bias'])
        wm_summary = frame.world_model_summary
        wm_transition_priors = frame.world_model_transition_priors

        wm_control = WorldModelControlProtocol.from_context({
            'world_model_summary': wm_summary,
            'world_model_transition_priors': wm_transition_priors,
        })
        hidden_guidance = _extract_hidden_state_guidance(wm_summary, wm_control)
        unified_context = frame.unified_context
        ablation_cfg = ports.get_causal_ablation()
        unified_enabled = bool(getattr(ablation_cfg, 'enable_unified_context', True))
        unified_payload = unified_context.to_dict() if unified_enabled else {
            'current_goal': str(unified_context.current_goal or ''),
            'current_task': str(unified_context.current_task or ''),
            'self_model_summary': dict(unified_context.self_model_summary),
        }
        planner_control_profile = _apply_self_model_planner_modulation(
            _build_planner_control_profile(planner_controls),
            unified_payload.get('self_model_summary', {}) if isinstance(unified_payload, dict) else {},
        )
        task_family = ports.infer_task_family()
        world_provider_meta = ports.get_world_provider_meta() or {}
        domain = str(world_provider_meta.get('runtime_env', '') or task_family or 'unknown')
        ctx = {
            'episode': episode,
            'tick': tick,
            'max_ticks': ports.get_max_ticks(),
            'available_functions': available_functions,
            'visible_functions': available_functions,
            'discovered_functions': available_functions,
            'active_hypotheses': ports.get_active_hypotheses()[:3],
            'reward_trend': reward_trend,
            'planner_bias': planner_bias,
            'policy_profile': planner_controls['policy_profile'],
            'representation_profile': planner_controls['representation_profile'],
            'planner_control_profile': dict(planner_control_profile),
            'planner_branch_budget': int(planner_control_profile.get('branch_budget', 2) or 2),
            'planner_verification_budget': int(planner_control_profile.get('verification_budget', 0) or 0),
            'planner_strategy_mode': str(planner_control_profile.get('strategy_mode', 'balanced') or 'balanced'),
            'world_model_summary': wm_summary,
            'world_model_transition_priors': wm_transition_priors,
            'world_model_control': wm_control.to_dict(),
            'world_model_shift_risk': float(wm_control.state_shift_risk),
            'world_model_predicted_phase': str(wm_control.predicted_phase),
            'world_model_hidden_state': dict(hidden_guidance),
            'world_model_hidden_phase': str(hidden_guidance.get('phase', '') or ''),
            'world_model_hidden_state_depth': int(hidden_guidance.get('depth', 0) or 0),
            'world_model_hidden_drift_score': float(hidden_guidance.get('drift_score', 0.0) or 0.0),
            'world_model_hidden_uncertainty_score': float(hidden_guidance.get('uncertainty_score', 0.0) or 0.0),
            'world_model_constraints': list(wm_control.hard_constraints),
            'unified_cognitive_context': unified_payload,
            'deliberation_candidate_tests': list(unified_payload.get('candidate_tests', []) or []) if isinstance(unified_payload, dict) else [],
            'deliberation_candidate_programs': list(unified_payload.get('candidate_programs', []) or []) if isinstance(unified_payload, dict) else [],
            'deliberation_candidate_outputs': list(unified_payload.get('candidate_outputs', []) or []) if isinstance(unified_payload, dict) else [],
            'deliberation_budget': dict(unified_payload.get('deliberation_budget', {}) or {}) if isinstance(unified_payload, dict) else {},
            'deliberation_mode': str(unified_payload.get('deliberation_mode', '') or '') if isinstance(unified_payload, dict) else '',
            'ablation_flags': ports.ablation_flags_snapshot(),
            'task_family': task_family,
            'domain': domain,
            'environment_tags': [task_family, domain],
        }
        local_mirror = obs.get('local_mirror', {}) if isinstance(obs, dict) and isinstance(obs.get('local_mirror', {}), dict) else {}
        if local_mirror:
            ctx['local_mirror'] = dict(local_mirror)
            ctx['default_command_present'] = bool(local_mirror.get('default_command_present', False))
            ctx['allow_empty_exec'] = bool(local_mirror.get('allow_empty_exec', False))
            ctx['workspace_file_count'] = int(local_mirror.get('workspace_file_count', 0) or 0)
            ctx['terminal_after_plan'] = bool(local_mirror.get('terminal_after_plan', True))

        pending_replan = ports.get_pending_replan()
        patch: Dict[str, Any] = {'pending_replan': pending_replan}
        decision_flags: Dict[str, Any] = {
            'events': [],
            'policy_profile': planner_controls['policy_profile'],
            'representation_profile': planner_controls['representation_profile'],
            'planner_control_profile': dict(planner_control_profile),
            'meta_control_snapshot_id': planner_controls['meta_control_snapshot_id'],
            'meta_control_inputs_hash': planner_controls['meta_control_inputs_hash'],
        }

        top_goal = continuity_snapshot.get('top_goal')

        if top_goal and not plan_state.has_plan:
            patch['set_plan'] = ports.objective_decomposer.decompose(top_goal, ctx)
            decision_flags['events'].append('plan_bootstrap')

        if pending_replan and plan_state.has_plan:
            pending_replan_task_id = pending_replan.get('task_id') if isinstance(pending_replan, dict) else None
            replan_ctx = dict(ctx)
            if isinstance(pending_replan, dict):
                for key in ('world_model_control', 'world_model_transition_priors', 'world_model_summary'):
                    if key in pending_replan:
                        replan_ctx[key] = pending_replan[key]
                if 'visible_functions' in pending_replan:
                    replan_ctx['visible_functions'] = list(pending_replan.get('visible_functions', []) or [])
                    replan_ctx['available_functions'] = list(pending_replan.get('visible_functions', []) or [])
                    replan_ctx['discovered_functions'] = list(pending_replan.get('visible_functions', []) or [])
                if 'target_function' in pending_replan and 'block_reason' not in replan_ctx:
                    replan_ctx['blocked_target'] = pending_replan.get('target_function')
            raw_trigger = str(pending_replan.get('trigger', 'recovery') if isinstance(pending_replan, dict) else 'recovery')
            trigger = 'plan_blocked' if raw_trigger == 'planner_namespace_mismatch' else raw_trigger
            if raw_trigger == 'planner_namespace_mismatch':
                replan_ctx['block_reason'] = 'planner_namespace_mismatch'
            if ports.plan_reviser.should_revise(plan_state.current_plan, trigger, replan_ctx):
                revised = ports.plan_reviser.revise(plan_state.current_plan, trigger, replan_ctx)
                if revised:
                    patch['set_plan'] = revised
                    decision_flags['events'].append('plan_revised_blocked' if trigger == 'plan_blocked' else 'plan_revised_recovery')
                    ports.mark_continuity_task_completed(pending_replan_task_id, reason='replan_revised')
                else:
                    ports.mark_continuity_task_cancelled(pending_replan_task_id, reason='replan_no_revision')
            else:
                ports.mark_continuity_task_cancelled(pending_replan_task_id, reason='replan_gate_rejected')
            patch['pending_replan'] = None

        patch['update_context'] = {
            'tick': tick,
            'reward': episode_reward,
            'discovered_functions': available_functions,
        }
        telemetry = {
            'phase': 'control',
            'planner_bias': planner_bias,
            'reward_trend': reward_trend,
            'planner_control_profile': dict(planner_control_profile),
            'hidden_state': dict(hidden_guidance),
        }

        current_step = getattr(plan_state, 'current_step', None)
        current_step_attempts = int(getattr(current_step, 'execution_attempts', 0) or 0) if current_step is not None else 0
        pending_step_exit_guard = bool(
            plan_state.has_plan
            and current_step is not None
            and current_step_attempts <= 0
        )
        telemetry['pending_step_exit_guard'] = pending_step_exit_guard

        check_exit = plan_state.has_plan and not pending_step_exit_guard
        if check_exit and plan_state.check_exit():
            patch['clear_plan'] = True
            decision_flags['events'].append('plan_exit')
            replan_goal = top_goal or _synthetic_goal_from_plan(plan_state.current_plan)
            if replan_goal:
                patch['set_plan'] = ports.objective_decomposer.decompose(replan_goal, ctx)
                decision_flags['events'].append('plan_regenerated')
        elif plan_state.has_plan and self._should_trigger_policy_replan(
            planner_bias=planner_bias,
            tick=tick,
            reward_trend=reward_trend,
            world_shift_risk=float(wm_control.state_shift_risk),
            hidden_phase=str(hidden_guidance.get('phase', '') or ''),
            hidden_state_depth=int(hidden_guidance.get('depth', 0) or 0),
            hidden_drift_score=float(hidden_guidance.get('drift_score', 0.0) or 0.0),
            current_step=getattr(plan_state, 'current_step', None),
            planner_control_profile=planner_control_profile,
        ):
            lookahead = self._evaluate_plan_lookahead(
                current_plan=plan_state.current_plan,
                current_step=getattr(plan_state, 'current_step', None),
                obs=obs,
                wm_transition_priors=wm_transition_priors,
                wm_summary=wm_summary,
                world_shift_risk=float(wm_control.state_shift_risk),
                hidden_guidance=hidden_guidance,
                tick=tick,
                planner_control_profile=planner_control_profile,
            )
            if lookahead['events']:
                decision_flags['events'].extend(lookahead['events'])
            telemetry['plan_lookahead'] = lookahead['telemetry']
            revised = ports.plan_reviser.revise(plan_state.current_plan, 'plan_blocked', {
                'episode': episode,
                'tick': tick,
                'block_reason': 'self_model_high_planner_bias',
                'available_functions': available_functions,
                'discovered_functions': available_functions,
                'visible_functions': available_functions,
                'planner_control_profile': dict(planner_control_profile),
                'world_model_summary': wm_summary,
                'world_model_transition_priors': wm_transition_priors,
                'world_model_control': wm_control.to_dict(),
                'world_model_shift_risk': float(wm_control.state_shift_risk),
                'world_model_predicted_phase': str(wm_control.predicted_phase),
                'replan_trigger': 'policy_blockage_or_world_shift',
            })
            if revised:
                patch['set_plan'] = revised
                decision_flags['events'].append('policy_driven_replan')
        elif plan_state.has_plan:
            lookahead = self._evaluate_plan_lookahead(
                current_plan=plan_state.current_plan,
                current_step=getattr(plan_state, 'current_step', None),
                obs=obs,
                wm_transition_priors=wm_transition_priors,
                wm_summary=wm_summary,
                world_shift_risk=float(wm_control.state_shift_risk),
                hidden_guidance=hidden_guidance,
                tick=tick,
                planner_control_profile=planner_control_profile,
            )
            if lookahead['events']:
                decision_flags['events'].extend(lookahead['events'])
            telemetry['plan_lookahead'] = lookahead['telemetry']
            if lookahead['force_replan']:
                revised = ports.plan_reviser.revise(plan_state.current_plan, 'plan_blocked', {
                    'episode': episode,
                    'tick': tick,
                    'block_reason': 'world_model_plan_value_guard',
                    'available_functions': available_functions,
                    'discovered_functions': available_functions,
                    'visible_functions': available_functions,
                    'planner_control_profile': dict(planner_control_profile),
                    'world_model_summary': wm_summary,
                    'world_model_transition_priors': wm_transition_priors,
                    'world_model_control': wm_control.to_dict(),
                    'world_model_shift_risk': float(wm_control.state_shift_risk),
                    'world_model_predicted_phase': str(wm_control.predicted_phase),
                    'replan_trigger': 'world_model_lookahead_guard',
                    'lookahead_eval': lookahead['telemetry'],
                })
                if revised:
                    patch['set_plan'] = revised
                    decision_flags['events'].append('policy_driven_replan')

        return self._finalize_result(selected_action=None, state_patch=patch, decision_flags=decision_flags, telemetry=telemetry)

    def _tick_progress(self, *, obs: Dict[str, Any], selected_action: Dict[str, Any], result: Dict[str, Any], reward: float) -> PlannerRuntimeResult:
        ports = self._ports
        plan_state = ports.plan_state
        patch: Dict[str, Any] = {}
        decision_flags: Dict[str, Any] = {'events': []}

        if not plan_state.has_plan:
            return self._finalize_result(selected_action=selected_action, state_patch=patch, decision_flags=decision_flags, telemetry={'phase': 'progress'})

        current_step = plan_state.current_step
        if not current_step:
            return self._finalize_result(selected_action=selected_action, state_patch=patch, decision_flags=decision_flags, telemetry={'phase': 'progress'})

        fn_name = extract_action_function_name(selected_action, default='wait')
        step_id = str(getattr(current_step, 'step_id', '') or '')
        step_target = current_step.target_function or 'wait'
        action_matches = (step_target == 'combine') or (fn_name == step_target)

        constraints = current_step.constraints if isinstance(current_step.constraints, dict) else {}
        success_pattern = constraints.get('success_observation_pattern') if isinstance(constraints.get('success_observation_pattern'), dict) else None
        failure_pattern = constraints.get('failure_observation_pattern') if isinstance(constraints.get('failure_observation_pattern'), dict) else None
        min_reward_for_success = float(constraints.get('min_reward_for_success', 0.0) or 0.0)
        max_attempts = int(constraints.get('max_attempts', 3) or 3)
        step_intent = str(getattr(current_step, 'intent', '') or '').strip().lower()

        step_success = bool(result.get('success', True))
        explicit_failure = bool(result.get('failed', False)) or str(result.get('status', '')).lower() in {'failed', 'error'}
        success_by_result = step_success and (reward >= min_reward_for_success) and not explicit_failure
        fail_by_result = explicit_failure or (not step_success) or reward < 0
        success_by_pattern = match_observation_pattern(success_pattern, obs, result) if success_pattern else False
        fail_by_pattern = match_observation_pattern(failure_pattern, obs, result) if failure_pattern else False
        attempts = int(getattr(current_step, 'execution_attempts', 0) or 0)
        blocked_by_attempts = attempts >= max_attempts and action_matches
        observed_state_change = bool(
            result.get('state_changed', False)
            or result.get('observation_changed', False)
        )
        information_gain = float(result.get('information_gain', 0.0) or 0.0)
        informative_probe_success = (
            action_matches
            and step_intent in {'explore', 'test'}
            and not explicit_failure
            and not fail_by_pattern
            and (
                observed_state_change
                or information_gain >= 0.12
            )
        )

        if action_matches and (success_by_result or success_by_pattern or informative_probe_success) and not fail_by_pattern:
            patch['step_transitions'] = [
                {'event': 'start', 'step_id': step_id, 'action_function': fn_name},
                {'event': 'complete', 'step_id': step_id, 'result': f'action={fn_name}, reward={reward}'},
            ]
            patch['advance_step'] = True
            decision_flags['events'].append('step_advanced')
            return self._finalize_result(selected_action=selected_action, state_patch=patch, decision_flags=decision_flags, telemetry={'phase': 'progress'})

        if action_matches and (fail_by_result or fail_by_pattern or blocked_by_attempts):
            failure_reason = 'blocked_max_attempts' if blocked_by_attempts else 'negative_or_pattern_failure'
            failure_note = f'action={fn_name}, reward={reward}, reason={failure_reason}, attempts={attempts}'
            patch['step_transitions'] = [
                {'event': 'start', 'step_id': step_id, 'action_function': fn_name},
                {'event': 'fail', 'step_id': step_id, 'reason': failure_note, 'block_plan': blocked_by_attempts},
            ]
            patch['mark_failed_reason'] = failure_note
            decision_flags['events'].append('step_failed')
            reliability_tracker = ports.get_reliability_tracker()
            planner_control_profile = dict(_build_planner_control_profile(ports.meta_control.for_planner_replan(
                episode=ports.get_episode(),
                tick=ports.get_tick(),
                context={'gate': 'planner_progress_revise', 'reason': failure_reason},
            )))
            failure_strategy_context = {
                'task_family': ports.infer_task_family(),
                'phase': 'progress',
                'available_functions': ports.extract_available_functions(obs),
                'discovered_functions': ports.extract_available_functions(obs),
                'visible_functions': ports.extract_available_functions(obs),
                'world_model_hidden_state': dict(_extract_hidden_state_guidance(
                    obs.get('world_model', {}),
                    WorldModelControlProtocol.from_context({'world_model_summary': obs.get('world_model', {})}),
                )),
            }
            strategy_profile = reliability_tracker.build_failure_strategy(
                fn_name,
                short_term_pressure=1.0,
                context=failure_strategy_context,
                planner_control_profile=planner_control_profile,
            ) if reliability_tracker else None
            revise_ctx = {
                'episode': ports.get_episode(),
                'tick': ports.get_tick(),
                'block_reason': 'negative_outcome_on_target_step',
                'reward': reward,
                'discovered_functions': ports.extract_available_functions(obs),
                'available_functions': ports.extract_available_functions(obs),
                'visible_functions': ports.extract_available_functions(obs),
                'failure_strategy_profile': strategy_profile.to_dict() if strategy_profile else {},
                'planner_control_profile': dict(planner_control_profile),
                'world_model_summary': ports.build_world_model_context(obs.get('perception') if isinstance(obs, dict) else {}),
                'world_model_transition_priors': ports.build_world_model_transition_priors(obs.get('perception') if isinstance(obs, dict) else {}),
                'world_model_hidden_state': dict(_extract_hidden_state_guidance(obs.get('world_model', {}), WorldModelControlProtocol.from_context({'world_model_summary': obs.get('world_model', {})}))),
                'failure_reason': failure_reason,
                'blocked_by_attempts': blocked_by_attempts,
                'attempts': attempts + 1,
            }
            plan = plan_state.current_plan
            if plan and ports.plan_reviser.should_revise(plan, 'plan_blocked', revise_ctx):
                revised = ports.plan_reviser.revise(plan, 'plan_blocked', revise_ctx)
                if revised:
                    patch['set_plan'] = revised
                    decision_flags['events'].append('plan_revised_blocked')

        return self._finalize_result(selected_action=selected_action, state_patch=patch, decision_flags=decision_flags, telemetry={'phase': 'progress'})

    @staticmethod
    def _should_trigger_policy_replan(
        *,
        planner_bias: float,
        tick: int,
        reward_trend: str,
        world_shift_risk: float,
        hidden_phase: str,
        hidden_state_depth: int,
        hidden_drift_score: float,
        current_step: Any,
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if tick <= 0:
            return False
        profile = planner_control_profile if isinstance(planner_control_profile, dict) else {}
        periodic_interval = max(2, int(profile.get('periodic_replan_interval', 3) or 3))
        periodic_bias_threshold = float(profile.get('periodic_bias_threshold', 0.8) or 0.8)
        world_shift_threshold = float(profile.get('world_shift_replan_threshold', 0.7) or 0.7)
        world_shift_bias_threshold = float(profile.get('world_shift_bias_threshold', 0.6) or 0.6)
        hidden_drift_threshold = float(profile.get('hidden_drift_replan_threshold', 0.72) or 0.72)
        hidden_disruption_threshold = float(profile.get('hidden_disruption_threshold', 0.58) or 0.58)
        negative_reward_bias_threshold = float(profile.get('negative_reward_bias_threshold', 0.7) or 0.7)
        periodic_bias_replan = planner_bias >= periodic_bias_threshold and tick % periodic_interval == 0
        high_world_shift = world_shift_risk >= world_shift_threshold and planner_bias >= world_shift_bias_threshold
        hidden_state_shift = hidden_drift_score >= hidden_drift_threshold and hidden_state_depth >= 1
        hidden_disruption = hidden_phase == 'disrupted' and hidden_drift_score >= hidden_disruption_threshold
        blocked_step = bool(getattr(current_step, 'status', None)) and str(getattr(current_step.status, 'value', current_step.status)).lower() == 'failed'
        sustained_negative = reward_trend == 'negative' and planner_bias >= negative_reward_bias_threshold
        return periodic_bias_replan or high_world_shift or hidden_state_shift or hidden_disruption or blocked_step or sustained_negative

    def _evaluate_plan_lookahead(
        self,
        *,
        current_plan: Any,
        current_step: Any,
        obs: Dict[str, Any],
        wm_transition_priors: Dict[str, Dict[str, float]],
        wm_summary: Dict[str, Any],
        world_shift_risk: float,
        hidden_guidance: Optional[Dict[str, Any]],
        tick: int,
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        priors = wm_transition_priors if isinstance(wm_transition_priors, dict) else {}
        control_profile = planner_control_profile if isinstance(planner_control_profile, dict) else {}
        target_action = str(getattr(current_step, 'target_function', None) or 'wait')
        alternatives = ['wait', 'probe', 'safer_action']
        action_set = [target_action] + [fn for fn in alternatives if fn != target_action]
        hidden = dict(hidden_guidance or {})
        hidden_phase = str(hidden.get('phase', '') or '')
        hidden_depth = int(hidden.get('depth', 0) or 0)
        hidden_phase_confidence = float(hidden.get('phase_confidence', 0.0) or 0.0)
        hidden_drift_score = float(hidden.get('drift_score', 0.0) or 0.0)
        hidden_uncertainty_score = float(hidden.get('uncertainty_score', 0.0) or 0.0)
        hidden_focus_functions = set(hidden.get('focus_functions', []) or [])

        evals: Dict[str, Dict[str, float]] = {}
        for fn_name in action_set:
            evals[fn_name] = self._extract_transition_quad(priors.get(fn_name, {}))

        alpha = 0.7
        beta = 0.35
        gamma = 0.45
        scored: Dict[str, float] = {}
        for fn_name, quad in evals.items():
            scored[fn_name] = (
                quad['long_reward']
                - alpha * quad['risk']
                + beta * quad['reversibility']
                + gamma * quad['info_gain']
            )
            focus_bonus = 0.0
            if fn_name and fn_name in hidden_focus_functions and hidden_phase in {'stabilizing', 'committed'}:
                focus_bonus += min(0.22, hidden_depth * 0.05) * max(0.4, hidden_phase_confidence)
            if hidden_phase == 'disrupted' and fn_name not in {'wait', 'probe'}:
                focus_bonus -= 0.12 * max(0.4, hidden_phase_confidence)
            if hidden_phase == 'exploring' and fn_name == 'probe':
                focus_bonus += 0.10
            scored[fn_name] += focus_bonus
        target_score = float(scored.get(target_action, 0.0))
        safest_alt = min((fn for fn in alternatives), key=lambda fn: evals.get(fn, {'risk': 1.0})['risk'])
        safest_alt_risk = float(evals.get(safest_alt, {'risk': 1.0})['risk'])
        target_risk = float(evals.get(target_action, {'risk': 0.0})['risk'])
        safe_value_gap = float(scored.get(safest_alt, 0.0)) - target_score

        uncertainty = self._estimate_uncertainty(wm_summary, world_shift_risk)
        uncertainty = max(uncertainty, hidden_uncertainty_score * 0.92)
        requires_probe = self._requires_probe_step(current_step=current_step)
        probe_attempts = int(getattr(current_step, 'execution_attempts', 0) or 0) if current_step is not None else 0
        hidden_probe_required = bool(hidden_drift_score >= 0.62 and 'probe_hidden_state_transition' in set((wm_summary.get('required_probes', []) if isinstance(wm_summary.get('required_probes', []), list) else [])))

        high_risk_threshold = float(control_profile.get('high_risk_replan_threshold', 0.75) or 0.75)
        safer_risk_margin = float(control_profile.get('safer_risk_margin', 0.15) or 0.15)
        value_drop_threshold = float(control_profile.get('value_drop_threshold', 0.4) or 0.4)
        uncertainty_threshold = float(control_profile.get('uncertainty_threshold', 0.7) or 0.7)
        verification_budget = max(0, int(control_profile.get('verification_budget', 0) or 0))
        force_events: List[str] = []

        if target_risk >= high_risk_threshold and (target_risk - safest_alt_risk) >= safer_risk_margin and safe_value_gap >= -0.1:
            force_events.append('wm_high_risk_replan')

        if self._last_lookahead_plan_value is not None:
            value_drop = target_score - self._last_lookahead_plan_value
            if world_shift_risk >= 0.55 and value_drop <= -value_drop_threshold:
                force_events.append('wm_value_drop_replan')
        else:
            value_drop = 0.0

        if requires_probe and probe_attempts <= 0 and uncertainty >= uncertainty_threshold:
            force_events.append('wm_probe_required_replan')
        if verification_budget >= 2 and target_action != 'probe' and hidden_uncertainty_score >= max(0.0, uncertainty_threshold - 0.12):
            force_events.append('wm_meta_verification_replan')
        hidden_meta_threshold = float(control_profile.get('hidden_drift_replan_threshold', 0.72) or 0.72)
        if hidden_drift_score >= hidden_meta_threshold and hidden_phase in {'disrupted', 'exploring'}:
            force_events.append('wm_hidden_drift_replan')
        if hidden_probe_required and target_action != 'probe' and probe_attempts <= 0:
            force_events.append('wm_hidden_probe_required_replan')

        lookahead_horizon = self._resolve_lookahead_horizon(control_profile)
        rollout_actions = self._collect_rollout_actions(
            current_plan=current_plan,
            current_step=current_step,
            horizon=lookahead_horizon,
        )
        rollout_eval = self._score_rollout_actions(
            rollout_actions=rollout_actions,
            priors=priors,
            hidden=hidden,
            obs=obs,
            wm_summary=wm_summary,
            planner_control_profile=control_profile,
        )
        frontier_candidate = self._best_frontier_rollout(
            current_step=current_step,
            priors=priors,
            hidden=hidden,
            obs=obs,
            wm_summary=wm_summary,
            planner_control_profile=control_profile,
            horizon=lookahead_horizon,
        )
        negative_rollout_threshold = float(control_profile.get('negative_rollout_threshold', -0.18) or -0.18)
        frontier_margin_threshold = float(control_profile.get('frontier_value_margin_threshold', 0.24) or 0.24)
        risky_rollout_threshold = float(control_profile.get('risky_rollout_threshold', 0.78) or 0.78)
        belief_branch_margin_threshold = float(control_profile.get('belief_branch_margin_threshold', 0.10) or 0.10)
        persistence_margin_threshold = float(control_profile.get('branch_persistence_margin_threshold', 0.18) or 0.18)
        low_persistence_threshold = float(control_profile.get('low_branch_persistence_threshold', 0.38) or 0.38)
        if len(rollout_actions) >= 2 and rollout_eval['value'] <= negative_rollout_threshold:
            force_events.append('wm_rollout_value_replan')
        if frontier_candidate:
            frontier_value_gap = float(frontier_candidate['value']) - float(rollout_eval['value'])
            if frontier_value_gap >= frontier_margin_threshold and (
                rollout_eval['value'] <= 0.0 or rollout_eval['risk'] >= risky_rollout_threshold
            ):
                force_events.append('wm_branch_salvage_replan')
            frontier_belief_gap = float(frontier_candidate.get('belief_value', 0.0) or 0.0) - float(rollout_eval.get('belief_value', 0.0) or 0.0)
            if (
                frontier_belief_gap >= belief_branch_margin_threshold
                and str(frontier_candidate.get('belief_branch_id', '') or '') != str(rollout_eval.get('belief_branch_id', '') or '')
                and float(frontier_candidate.get('belief_branch_confidence', 0.0) or 0.0) >= 0.45
            ):
                force_events.append('wm_belief_branch_replan')
            frontier_persistence_gap = float(frontier_candidate.get('persistence_value', 0.0) or 0.0) - float(rollout_eval.get('persistence_value', 0.0) or 0.0)
            frontier_persistence_ratio_gap = float(frontier_candidate.get('branch_persistence_ratio', 0.0) or 0.0) - float(rollout_eval.get('branch_persistence_ratio', 0.0) or 0.0)
            if (
                frontier_persistence_gap >= persistence_margin_threshold
                or frontier_persistence_ratio_gap >= max(0.18, persistence_margin_threshold * 0.8)
            ) and (
                float(rollout_eval.get('branch_persistence_ratio', 0.0) or 0.0) <= low_persistence_threshold
                or str(rollout_eval.get('final_phase', '') or '') == 'disrupted'
                or float(rollout_eval['risk']) >= risky_rollout_threshold * 0.9
            ):
                force_events.append('wm_branch_persistence_replan')

        telemetry = {
            'tick': tick,
            'target_action': target_action,
            'alternatives': alternatives,
            'transition_quads': evals,
            'plan_value': target_score,
            'plan_value_by_action': scored,
            'safest_alternative': safest_alt,
            'safest_alternative_risk': safest_alt_risk,
            'target_risk': target_risk,
            'plan_value_delta': value_drop,
            'world_shift_risk': float(world_shift_risk),
            'estimated_uncertainty': uncertainty,
            'planner_control_profile': dict(control_profile),
            'probe_required': requires_probe,
            'hidden_probe_required': hidden_probe_required,
            'verification_budget': verification_budget,
            'probe_attempts': probe_attempts,
            'hidden_state': dict(hidden),
            'lookahead_horizon': lookahead_horizon,
            'rollout_plan_value': rollout_eval['value'],
            'rollout_plan_risk': rollout_eval['risk'],
            'rollout_belief_value': rollout_eval.get('belief_value', 0.0),
            'rollout_belief_branch_id': rollout_eval.get('belief_branch_id', ''),
            'rollout_belief_target_phase': rollout_eval.get('belief_target_phase', ''),
            'rollout_persistence_value': rollout_eval.get('persistence_value', 0.0),
            'rollout_branch_persistence_ratio': rollout_eval.get('branch_persistence_ratio', 0.0),
            'rollout_branch_id': rollout_eval.get('rollout_branch_id', ''),
            'rollout_branch_target_phase': rollout_eval.get('rollout_branch_target_phase', ''),
            'rollout_final_phase': rollout_eval.get('final_phase', ''),
            'rollout_phase_path': list(rollout_eval.get('phase_path', []) or []),
            'rollout_trace': list(rollout_eval['trace']),
            'rollout_used_learned_dynamics': bool(rollout_eval.get('used_learned_dynamics', False)),
            'rollout_learned_coverage_ratio': float(rollout_eval.get('learned_coverage_ratio', 0.0) or 0.0),
            'frontier_alternative': frontier_candidate,
            'forced_replan_events': list(force_events),
            'formula_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
        }
        self._last_lookahead_plan_value = target_score
        return {'force_replan': bool(force_events), 'events': force_events, 'telemetry': telemetry}

    @staticmethod
    def _resolve_lookahead_horizon(planner_control_profile: Optional[Dict[str, Any]] = None) -> int:
        profile = planner_control_profile if isinstance(planner_control_profile, dict) else {}
        try:
            horizon = int(profile.get('lookahead_horizon', 3) or 3)
        except (TypeError, ValueError):
            horizon = 3
        return max(1, min(4, horizon))

    @staticmethod
    def _step_constraints(step: Any) -> Dict[str, Any]:
        raw = getattr(step, 'constraints', {})
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _step_action_kwargs(step: Any) -> Dict[str, Any]:
        constraints = PlannerRuntime._step_constraints(step)
        if isinstance(constraints.get('action_kwargs', {}), dict):
            return dict(constraints.get('action_kwargs', {}))
        if isinstance(constraints.get('kwargs', {}), dict):
            return dict(constraints.get('kwargs', {}))
        for key in ('raw_action', 'selected_action', 'action'):
            payload = constraints.get(key, {})
            if not isinstance(payload, dict):
                continue
            tool_args = payload.get('payload', {}).get('tool_args', {}) if isinstance(payload.get('payload', {}), dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            if kwargs:
                return dict(kwargs)
        kwargs: Dict[str, Any] = {}
        for key in ('x', 'y', 'anchor_ref', 'target_family', 'object_color', 'role'):
            if key in constraints:
                kwargs[key] = constraints.get(key)
        target_point = constraints.get('target_point', {}) if isinstance(constraints.get('target_point', {}), dict) else {}
        if 'x' not in kwargs and target_point.get('x') is not None:
            kwargs['x'] = target_point.get('x')
        if 'y' not in kwargs and target_point.get('y') is not None:
            kwargs['y'] = target_point.get('y')
        return kwargs

    @staticmethod
    def _planner_rollout_action_to_synthetic_action(row: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(row, dict):
            return {'kind': 'call_tool', 'payload': {'tool_args': {'function_name': 'wait', 'kwargs': {}}}}
        constraints = row.get('constraints', {}) if isinstance(row.get('constraints', {}), dict) else {}
        synthetic: Dict[str, Any] = {}

        def _merge_action(source: Any) -> None:
            nonlocal synthetic
            if not isinstance(source, dict) or not source:
                return
            if not synthetic:
                synthetic = deepcopy(source)
                return
            payload = synthetic.get('payload', {}) if isinstance(synthetic.get('payload', {}), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
            source_payload = source.get('payload', {}) if isinstance(source.get('payload', {}), dict) else {}
            source_tool_args = source_payload.get('tool_args', {}) if isinstance(source_payload.get('tool_args', {}), dict) else {}
            if source_tool_args:
                if source_tool_args.get('function_name'):
                    tool_args['function_name'] = str(source_tool_args.get('function_name') or '')
                source_kwargs = source_tool_args.get('kwargs', {})
                if isinstance(source_kwargs, dict) and source_kwargs:
                    merged_kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
                    merged_kwargs.update(deepcopy(source_kwargs))
                    tool_args['kwargs'] = merged_kwargs
            payload['tool_args'] = tool_args
            synthetic['payload'] = payload
            source_meta = source.get('_candidate_meta', {}) if isinstance(source.get('_candidate_meta', {}), dict) else {}
            if source_meta:
                meta = synthetic.get('_candidate_meta', {}) if isinstance(synthetic.get('_candidate_meta', {}), dict) else {}
                meta.update(deepcopy(source_meta))
                synthetic['_candidate_meta'] = meta
            for key, value in source.items():
                if key in {'payload', '_candidate_meta'}:
                    continue
                synthetic.setdefault(key, deepcopy(value))

        for key in ('raw_action', 'selected_action', 'action'):
            _merge_action(row.get(key, {}))
            _merge_action(constraints.get(key, {}))

        action_payload = row.get('action_payload', {})
        if isinstance(action_payload, dict) and action_payload:
            _merge_action(action_payload)
        tool_args = row.get('tool_args', {})
        if isinstance(tool_args, dict) and tool_args:
            _merge_action({'kind': 'call_tool', 'payload': {'tool_args': deepcopy(tool_args)}})

        if not synthetic:
            synthetic = {'kind': 'call_tool', 'payload': {'tool_args': {}}}
        synthetic['kind'] = str(synthetic.get('kind', '') or 'call_tool')
        payload = synthetic.get('payload', {}) if isinstance(synthetic.get('payload', {}), dict) else {}
        tool_args_payload = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
        kwargs = tool_args_payload.get('kwargs', {}) if isinstance(tool_args_payload.get('kwargs', {}), dict) else {}

        fn_name = str(
            row.get('function', '')
            or row.get('function_name', '')
            or tool_args_payload.get('function_name', '')
            or 'wait'
        )
        tool_args_payload['function_name'] = fn_name

        for source in (constraints, row):
            source_kwargs = source.get('kwargs', {}) if isinstance(source.get('kwargs', {}), dict) else {}
            if source_kwargs:
                kwargs.update(deepcopy(source_kwargs))
        for source in (constraints, row):
            for key in ('x', 'y', 'anchor_ref', 'target_family', 'relation_type', 'object_color', 'role'):
                if key in source and source.get(key) is not None:
                    kwargs[key] = deepcopy(source.get(key))

        tool_args_payload['kwargs'] = kwargs
        payload['tool_args'] = tool_args_payload
        synthetic['payload'] = payload

        meta = synthetic.get('_candidate_meta', {}) if isinstance(synthetic.get('_candidate_meta', {}), dict) else {}
        anchor_ref = kwargs.get('anchor_ref')
        if anchor_ref not in (None, ''):
            meta['anchor_ref'] = anchor_ref
        target_family = kwargs.get('target_family')
        if target_family not in (None, ''):
            meta['target_family'] = str(target_family)
        relation_type = kwargs.get('relation_type')
        if relation_type not in (None, ''):
            meta['goal_progress_relation_type'] = str(relation_type)
        object_color = kwargs.get('object_color')
        if object_color not in (None, '') and meta.get('object_color') in (None, ''):
            meta['object_color'] = object_color
        if meta:
            synthetic['_candidate_meta'] = meta
        return synthetic

    @staticmethod
    def _is_probe_like_action(fn_name: str, constraints: Optional[Dict[str, Any]] = None) -> bool:
        name = str(fn_name or '').strip().lower()
        if name in {'probe', 'inspect'}:
            return True
        if any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test')):
            return True
        if isinstance(constraints, dict):
            return bool(
                constraints.get('require_probe')
                or constraints.get('requires_probe')
                or constraints.get('must_probe')
            )
        return False

    @staticmethod
    def _action_family(fn_name: str) -> str:
        name = str(fn_name or '').strip().lower()
        if not name:
            return 'generic'
        if any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test')):
            return 'probe'
        if 'scan' in name:
            return 'scan'
        if any(token in name for token in ('calibrate', 'align', 'tune')):
            return 'calibrate'
        if any(token in name for token in ('route', 'select', 'choose', 'rank')):
            return 'route'
        if any(token in name for token in ('commit', 'apply', 'submit', 'advance', 'finalize', 'seal')):
            return 'commit'
        if any(token in name for token in ('compute', 'aggregate', 'transform', 'join', 'filter', 'group')):
            return 'compute'
        return 'generic'

    @staticmethod
    def _phase_alias(raw_phase: Any) -> str:
        phase = str(raw_phase or '').strip().lower()
        if not phase:
            return ''
        aliases = {
            'explore': 'exploring',
            'exploration': 'exploring',
            'stabilize': 'stabilizing',
            'stable': 'stabilizing',
            'commit': 'committed',
            'complete': 'committed',
            'completed': 'committed',
            'solve': 'committed',
            'solved': 'committed',
            'fail': 'disrupted',
            'failed': 'disrupted',
            'error': 'disrupted',
            'drift': 'disrupted',
        }
        return aliases.get(phase, phase)

    @staticmethod
    def _normalize_hidden_branches(hidden: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = hidden.get('latent_branches', []) if isinstance(hidden, dict) else []
        if not isinstance(raw, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            rows.append({
                'branch_id': str(item.get('branch_id', '') or '').strip(),
                'current_phase': PlannerRuntime._phase_alias(item.get('current_phase', '')) or 'exploring',
                'target_phase': PlannerRuntime._phase_alias(item.get('target_phase', '')) or 'exploring',
                'confidence': max(0.0, min(1.0, float(item.get('confidence', 0.0) or 0.0))),
                'anchor_functions': [
                    str(value or '').strip()
                    for value in list(item.get('anchor_functions', []) or [])[:4]
                    if str(value or '').strip()
                ],
                'risky_functions': [
                    str(value or '').strip()
                    for value in list(item.get('risky_functions', []) or [])[:4]
                    if str(value or '').strip()
                ],
            })
        return rows

    def _resolve_rollout_branch(
        self,
        *,
        rollout_actions: List[Dict[str, Any]],
        hidden: Dict[str, Any],
    ) -> Dict[str, Any]:
        latent_branches = self._normalize_hidden_branches(hidden)
        dominant_branch_id = str(hidden.get('dominant_branch_id', '') or '').strip()
        dominant_branch = next((dict(row) for row in latent_branches if str(row.get('branch_id', '')) == dominant_branch_id), {})
        if not dominant_branch and latent_branches:
            dominant_branch = dict(latent_branches[0])

        explicit_branch_id = ''
        explicit_target_phase = ''
        explicit_confidence = 0.0
        explicit_anchor_functions: List[str] = []
        for row in rollout_actions:
            constraints = row.get('constraints', {}) if isinstance(row.get('constraints', {}), dict) else {}
            explicit_branch_id = str(constraints.get('belief_branch_id', '') or '').strip() or explicit_branch_id
            explicit_target_phase = self._phase_alias(constraints.get('belief_target_phase', '')) or explicit_target_phase
            explicit_confidence = max(explicit_confidence, float(constraints.get('belief_branch_confidence', 0.0) or 0.0))
            explicit_anchor_functions = [
                str(value or '').strip()
                for value in (constraints.get('belief_anchor_functions', []) if isinstance(constraints.get('belief_anchor_functions', []), list) else [])
                if str(value or '').strip()
            ] or explicit_anchor_functions
            if explicit_branch_id or explicit_target_phase or explicit_anchor_functions:
                break

        branch = {}
        if explicit_branch_id:
            branch = next((dict(row) for row in latent_branches if str(row.get('branch_id', '')) == explicit_branch_id), {})
        if not branch and dominant_branch:
            branch = dict(dominant_branch)
        if explicit_branch_id:
            branch['branch_id'] = explicit_branch_id
        if explicit_target_phase:
            branch['target_phase'] = explicit_target_phase
        if explicit_confidence > 0.0:
            branch['confidence'] = max(float(branch.get('confidence', 0.0) or 0.0), explicit_confidence)
        if explicit_anchor_functions:
            branch['anchor_functions'] = list(explicit_anchor_functions)
        if 'anchor_functions' not in branch:
            branch['anchor_functions'] = []
        if 'risky_functions' not in branch:
            branch['risky_functions'] = list(dominant_branch.get('risky_functions', []) or []) if dominant_branch else []
        if 'target_phase' not in branch:
            branch['target_phase'] = self._phase_alias(hidden.get('expected_next_phase', '')) or self._phase_alias(hidden.get('phase', '')) or 'exploring'
        if 'current_phase' not in branch:
            branch['current_phase'] = self._phase_alias(hidden.get('phase', '')) or 'exploring'
        if 'branch_id' not in branch:
            branch['branch_id'] = str(dominant_branch.get('branch_id', '') or '')
        if 'confidence' not in branch:
            branch['confidence'] = float(dominant_branch.get('confidence', 0.0) or 0.0) if dominant_branch else 0.0
        return branch if branch else {}

    def _branch_guided_next_phase(
        self,
        *,
        current_phase: str,
        rollout_branch: Dict[str, Any],
        fn_name: str,
        step_index: int,
        anchor_index: int,
        risky_match: bool,
        probe_like: bool,
    ) -> str:
        if not rollout_branch:
            return current_phase
        target_phase = self._phase_alias(rollout_branch.get('target_phase', '')) or current_phase
        branch_confidence = max(0.0, min(1.0, float(rollout_branch.get('confidence', 0.0) or 0.0)))
        anchor_functions = [
            str(value or '').strip()
            for value in list(rollout_branch.get('anchor_functions', []) or [])[:4]
            if str(value or '').strip()
        ]
        commit_like = self._action_family(fn_name) == 'commit'
        if risky_match and target_phase in {'stabilizing', 'committed'}:
            return 'disrupted'
        if anchor_index >= 0:
            if target_phase == 'committed':
                if anchor_index >= max(0, len(anchor_functions) - 2) or step_index >= 1 or branch_confidence >= 0.84:
                    return 'committed'
                return 'stabilizing'
            if target_phase == 'stabilizing':
                return 'stabilizing'
            return target_phase
        if probe_like and target_phase in {'exploring', 'disrupted'}:
            return target_phase
        if target_phase == 'committed' and current_phase in {'exploring', 'disrupted'} and not commit_like:
            return 'stabilizing'
        if target_phase == 'stabilizing' and current_phase == 'exploring':
            return 'stabilizing'
        return current_phase if current_phase in {'stabilizing', 'committed'} else target_phase

    def _collect_rollout_actions(
        self,
        *,
        current_plan: Any,
        current_step: Any,
        horizon: int,
    ) -> List[Dict[str, Any]]:
        rollout: List[Dict[str, Any]] = []
        steps = getattr(current_plan, 'steps', None)
        if isinstance(steps, list) and steps:
            try:
                start = max(0, int(getattr(current_plan, 'current_step_index', 0) or 0))
            except (TypeError, ValueError):
                start = 0
            for step in steps[start:start + horizon]:
                fn_name = str(getattr(step, 'target_function', None) or 'wait')
                rollout.append({
                    'function': fn_name,
                    'kwargs': self._step_action_kwargs(step),
                    'constraints': self._step_constraints(step),
                    'intent': str(getattr(step, 'intent', '') or ''),
                })
        elif current_step is not None:
            rollout.append({
                'function': str(getattr(current_step, 'target_function', None) or 'wait'),
                'kwargs': self._step_action_kwargs(current_step),
                'constraints': self._step_constraints(current_step),
                'intent': str(getattr(current_step, 'intent', '') or ''),
            })
        return rollout

    def _score_rollout_actions(
        self,
        *,
        rollout_actions: List[Dict[str, Any]],
        priors: Dict[str, Dict[str, float]],
        hidden: Dict[str, Any],
        obs: Optional[Dict[str, Any]] = None,
        wm_summary: Optional[Dict[str, Any]] = None,
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = planner_control_profile if isinstance(planner_control_profile, dict) else {}
        learned_rollout = self._score_learned_rollout_actions(
            rollout_actions=rollout_actions,
            hidden=hidden,
            obs=obs if isinstance(obs, dict) else {},
            wm_summary=wm_summary if isinstance(wm_summary, dict) else {},
            planner_control_profile=profile,
        )
        if learned_rollout is not None:
            return learned_rollout
        hidden_phase = str(hidden.get('phase', '') or '')
        hidden_confidence = max(0.0, min(1.0, float(hidden.get('phase_confidence', 0.0) or 0.0)))
        hidden_uncertainty = max(0.0, min(1.0, float(hidden.get('uncertainty_score', 0.0) or 0.0)))
        risk_tolerance = max(0.0, min(1.0, float(profile.get('risk_tolerance', 0.5) or 0.5)))
        verification_budget = max(0, int(profile.get('verification_budget', 0) or 0))
        current_phase = self._phase_alias(hidden_phase) or 'exploring'
        rollout_branch = self._resolve_rollout_branch(rollout_actions=rollout_actions, hidden=hidden)
        rollout_branch_id = str(rollout_branch.get('branch_id', '') or '').strip()
        rollout_branch_target_phase = self._phase_alias(rollout_branch.get('target_phase', '')) or ''
        rollout_branch_confidence = max(0.0, min(1.0, float(rollout_branch.get('confidence', 0.0) or 0.0)))
        branch_anchor_functions = [
            str(value or '').strip()
            for value in list(rollout_branch.get('anchor_functions', []) or [])[:4]
            if str(value or '').strip()
        ]
        branch_risky_functions = {
            str(value or '').strip()
            for value in list(rollout_branch.get('risky_functions', []) or [])[:4]
            if str(value or '').strip()
        }
        phase_path = [current_phase]
        persistence_value = 0.0
        branch_guided_steps = 0.0
        last_anchor_index = -1

        total_value = 0.0
        total_risk = 0.0
        belief_value = 0.0
        probe_seen = False
        trace: List[Dict[str, Any]] = []
        active_belief_branch_id = ''
        active_belief_target_phase = ''

        for index, row in enumerate(rollout_actions):
            fn_name = str(row.get('function', 'wait') or 'wait')
            constraints = row.get('constraints', {}) if isinstance(row.get('constraints', {}), dict) else {}
            quad = self._extract_transition_quad(priors.get(fn_name, {}))
            belief_target_phase = str(constraints.get('belief_target_phase', '') or '').strip().lower()
            belief_branch_id = str(constraints.get('belief_branch_id', '') or '').strip()
            belief_branch_confidence = max(0.0, min(1.0, float(constraints.get('belief_branch_confidence', 0.0) or 0.0)))
            belief_anchor_functions = [
                str(value or '').strip()
                for value in (constraints.get('belief_anchor_functions', []) if isinstance(constraints.get('belief_anchor_functions', []), list) else [])
                if str(value or '').strip()
            ]
            belief_uncertainty_reduction = max(0.0, float(constraints.get('belief_uncertainty_reduction', 0.0) or 0.0))
            risk_weight = 0.42 + (1.0 - risk_tolerance) * 0.20
            info_gain_weight = 0.16 + min(0.10, verification_budget * 0.04)
            step_value = (
                quad['long_reward']
                - risk_weight * quad['risk']
                + 0.18 * quad['reversibility']
                + info_gain_weight * quad['info_gain']
            )
            if hidden_phase in {'disrupted', 'exploring'} and not probe_seen and not self._is_probe_like_action(fn_name, constraints):
                step_value -= 0.08 * max(0.4, hidden_confidence)
            if self._is_probe_like_action(fn_name, constraints):
                probe_seen = True
                if hidden_phase in {'disrupted', 'exploring'}:
                    step_value += 0.06
            if 'commit' in fn_name.lower() and not probe_seen and hidden_uncertainty >= 0.55:
                step_value -= 0.12
            belief_bonus = 0.0
            if belief_target_phase in {'exploring', 'disrupted'}:
                if self._is_probe_like_action(fn_name, constraints):
                    belief_bonus += 0.08 * max(0.4, belief_branch_confidence) + belief_uncertainty_reduction * 0.35
                elif index <= 1 and 'commit' in fn_name.lower():
                    belief_bonus -= 0.14 * max(0.4, belief_branch_confidence)
            elif belief_target_phase == 'stabilizing':
                if self._action_family(fn_name) in {'calibrate', 'probe'}:
                    belief_bonus += 0.05 * max(0.4, belief_branch_confidence)
                elif 'commit' in fn_name.lower() and index <= 1:
                    belief_bonus -= 0.10 * max(0.4, belief_branch_confidence)
            elif belief_target_phase == 'committed':
                if 'commit' in fn_name.lower():
                    belief_bonus += (0.10 if probe_seen or index > 0 else -0.08) * max(0.4, belief_branch_confidence)
            if belief_anchor_functions and fn_name in belief_anchor_functions:
                belief_bonus += 0.06 * max(0.4, belief_branch_confidence)
            elif belief_anchor_functions and index == 0 and not self._is_probe_like_action(fn_name, constraints):
                belief_bonus -= 0.04 * max(0.4, belief_branch_confidence)
            probe_like = self._is_probe_like_action(fn_name, constraints)
            action_family = self._action_family(fn_name)
            anchor_match = bool(fn_name and fn_name in branch_anchor_functions)
            risky_match = bool(fn_name and fn_name in branch_risky_functions)
            anchor_index = branch_anchor_functions.index(fn_name) if anchor_match else -1
            persistence_bonus = 0.0
            if rollout_branch:
                if anchor_match:
                    persistence_bonus += 0.08 + rollout_branch_confidence * 0.10
                    if anchor_index == last_anchor_index + 1:
                        persistence_bonus += 0.06
                    elif anchor_index > last_anchor_index + 1:
                        persistence_bonus -= 0.04
                if risky_match:
                    persistence_bonus -= 0.16 + rollout_branch_confidence * 0.12
                if rollout_branch_target_phase in {'exploring', 'disrupted'} and probe_like:
                    persistence_bonus += 0.05 + rollout_branch_confidence * 0.04
                elif rollout_branch_target_phase == 'stabilizing' and action_family in {'calibrate', 'probe'}:
                    persistence_bonus += 0.05 + rollout_branch_confidence * 0.03
                elif rollout_branch_target_phase == 'committed' and action_family == 'commit':
                    if anchor_match and anchor_index >= max(0, len(branch_anchor_functions) - 1):
                        persistence_bonus += 0.10 + rollout_branch_confidence * 0.06
                    elif not probe_seen and index == 0:
                        persistence_bonus -= 0.12 + rollout_branch_confidence * 0.06
                    elif last_anchor_index >= max(0, len(branch_anchor_functions) - 2):
                        persistence_bonus += 0.06 + rollout_branch_confidence * 0.04
                if branch_anchor_functions and index == 0 and not anchor_match and not probe_like and action_family != 'calibrate':
                    persistence_bonus -= 0.05 * max(0.4, rollout_branch_confidence)
            step_value += belief_bonus + persistence_bonus

            discount = max(0.45, 1.0 - index * 0.15)
            discounted_value = step_value * discount
            discounted_risk = quad['risk'] * discount
            total_value += discounted_value
            total_risk += discounted_risk
            belief_value += belief_bonus * discount
            persistence_value += persistence_bonus * discount
            if belief_branch_id and not active_belief_branch_id:
                active_belief_branch_id = belief_branch_id
                active_belief_target_phase = belief_target_phase
            if rollout_branch_id and not active_belief_branch_id:
                active_belief_branch_id = rollout_branch_id
                active_belief_target_phase = rollout_branch_target_phase
            if anchor_match and anchor_index > last_anchor_index:
                last_anchor_index = anchor_index
            next_phase = self._branch_guided_next_phase(
                current_phase=current_phase,
                rollout_branch=rollout_branch,
                fn_name=fn_name,
                step_index=index,
                anchor_index=anchor_index,
                risky_match=risky_match,
                probe_like=probe_like,
            )
            branch_memory_retained = bool(rollout_branch) and persistence_bonus > 0.0 and not risky_match
            if branch_memory_retained:
                branch_guided_steps += 1.0
            current_phase = next_phase
            phase_path.append(next_phase)
            trace.append({
                'index': index,
                'function': fn_name,
                'value': discounted_value,
                'risk': discounted_risk,
                'probe_like': probe_like,
                'belief_branch_id': belief_branch_id,
                'belief_target_phase': belief_target_phase,
                'belief_bonus': belief_bonus * discount,
                'persistence_bonus': persistence_bonus * discount,
                'rollout_branch_id': rollout_branch_id,
                'rollout_branch_target_phase': rollout_branch_target_phase,
                'anchor_match': anchor_match,
                'risky_match': risky_match,
                'next_phase': next_phase,
                'branch_memory_retained': branch_memory_retained,
            })

        return {
            'value': float(total_value),
            'risk': float(total_risk),
            'belief_value': float(belief_value),
            'persistence_value': float(persistence_value),
            'branch_persistence_ratio': float(branch_guided_steps / max(len(rollout_actions), 1)),
            'rollout_branch_id': rollout_branch_id,
            'rollout_branch_target_phase': rollout_branch_target_phase,
            'rollout_branch_confidence': float(rollout_branch_confidence),
            'phase_path': list(phase_path),
            'final_phase': str(phase_path[-1] if phase_path else current_phase),
            'belief_branch_id': active_belief_branch_id,
            'belief_target_phase': active_belief_target_phase,
            'trace': trace,
            'used_learned_dynamics': False,
            'learned_coverage_ratio': 0.0,
        }

    def _score_learned_rollout_actions(
        self,
        *,
        rollout_actions: List[Dict[str, Any]],
        hidden: Dict[str, Any],
        obs: Dict[str, Any],
        wm_summary: Dict[str, Any],
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        ports = self._ports
        predictor = ports.get_learned_dynamics_predictor()
        deployment_mode = str(ports.get_learned_dynamics_deployment_mode() or "shadow").strip().lower()
        if predictor is None or deployment_mode != "planner_rollout_dependence" or not rollout_actions or not isinstance(obs, dict) or not obs:
            return None

        hidden_summary = {
            'phase': str(hidden.get('phase', '') or ''),
            'phase_confidence': float(hidden.get('phase_confidence', 0.0) or 0.0),
            'hidden_state_depth': int(hidden.get('depth', 0) or 0),
            'drift_score': float(hidden.get('drift_score', 0.0) or 0.0),
            'uncertainty_score': float(hidden.get('uncertainty_score', 0.0) or 0.0),
            'expected_next_phase': str(hidden.get('expected_next_phase', hidden.get('phase', '')) or ''),
        }
        tick = int(self._ports.get_tick() or 0)
        state_snapshot = build_learned_dynamics_state_snapshot(
            obs,
            world_model_summary=wm_summary,
            hidden_state_summary=hidden_summary,
            belief_summary=wm_summary,
            identity_tracker=ports.get_persistent_object_identity_tracker(),
            tick=tick * 2,
        )

        trace: List[Dict[str, Any]] = []
        phase_path = [str(state_snapshot.get('phase', 'unknown') or 'unknown')]
        total_value = 0.0
        total_risk = 0.0
        used_steps = 0
        information_gain_bonus = 0.0
        current_snapshot = state_snapshot

        for index, row in enumerate(rollout_actions):
            fn_name = str(row.get('function', 'wait') or 'wait')
            rollout_kwargs = dict(row.get('kwargs', {}) or {}) if isinstance(row.get('kwargs', {}), dict) else {}
            synthetic_action = self._planner_rollout_action_to_synthetic_action(row)
            prediction = predictor.predict(current_snapshot, synthetic_action)
            if not isinstance(prediction, dict) or not prediction:
                break
            confidence = float(prediction.get('confidence', 0.0) or 0.0)
            support = int(prediction.get('support', 0) or 0)
            valid_state_change = bool(prediction.get('valid_state_change', False))
            reward_sign = str(prediction.get('reward_sign', '') or '')
            information_gain = float(prediction.get('information_gain', 0.0) or 0.0)
            risk_type = str(prediction.get('risk_type', '') or '')
            routing_active = bool(
                confidence >= 0.6
                and support >= 2
                and valid_state_change
                and reward_sign in {'positive', 'zero'}
            )
            if not routing_active:
                break

            step_value = 0.0
            if reward_sign == 'positive':
                step_value += 0.24
            elif reward_sign == 'zero':
                step_value += 0.06
            else:
                step_value -= 0.18
            step_value += information_gain * 0.35
            if valid_state_change:
                step_value += 0.05
            if risk_type in {'execution_failure', 'dead_end'}:
                step_value -= 0.12

            step_risk = 0.18 if risk_type in {'execution_failure', 'dead_end'} else 0.06
            discount = max(0.45, 1.0 - index * 0.15)
            total_value += step_value * discount
            total_risk += step_risk * discount
            information_gain_bonus += information_gain * discount
            used_steps += 1
            next_phase = str(prediction.get('next_phase', phase_path[-1]) or phase_path[-1])
            phase_path.append(next_phase)
            trace.append({
                'index': index,
                'function': fn_name,
                'kwargs': dict(rollout_kwargs),
                'source': 'learned_dynamics_rollout',
                'confidence': round(confidence, 4),
                'support': support,
                'reward_sign': reward_sign,
                'info_gain': round(information_gain, 4),
                'risk_type': risk_type,
                'next_phase': next_phase,
                'routing_active': True,
            })
            current_snapshot = _apply_prediction_to_snapshot(current_snapshot, prediction)

        coverage_ratio = float(used_steps / max(len(rollout_actions), 1))
        if used_steps <= 0 or coverage_ratio < 0.5:
            return None
        return {
            'value': float(total_value),
            'risk': float(total_risk),
            'belief_value': 0.0,
            'persistence_value': float(information_gain_bonus * 0.08),
            'branch_persistence_ratio': coverage_ratio,
            'rollout_branch_id': '',
            'rollout_branch_target_phase': '',
            'rollout_branch_confidence': 0.0,
            'phase_path': list(phase_path),
            'final_phase': str(phase_path[-1] if phase_path else current_snapshot.get('phase', 'unknown')),
            'belief_branch_id': '',
            'belief_target_phase': '',
            'trace': trace,
            'used_learned_dynamics': True,
            'learned_coverage_ratio': coverage_ratio,
        }

    def _best_frontier_rollout(
        self,
        *,
        current_step: Any,
        priors: Dict[str, Dict[str, float]],
        hidden: Dict[str, Any],
        obs: Optional[Dict[str, Any]] = None,
        wm_summary: Optional[Dict[str, Any]] = None,
        planner_control_profile: Optional[Dict[str, Any]] = None,
        horizon: int,
    ) -> Optional[Dict[str, Any]]:
        constraints = self._step_constraints(current_step)
        frontier_actions: List[Dict[str, Any]] = []

        raw_frontier = constraints.get('branch_frontier', [])
        if isinstance(raw_frontier, list):
            for row in raw_frontier[:3]:
                if not isinstance(row, dict):
                    continue
                residual_chain = [
                    str(fn or '').strip()
                    for fn in (row.get('residual_chain', []) if isinstance(row.get('residual_chain', []), list) else [])
                    if str(fn or '').strip()
                ]
                target_function = str(row.get('target_function', '') or '').strip()
                if not residual_chain and target_function:
                    residual_chain = [target_function]
                if not residual_chain:
                    continue
                branch_constraints = {
                    'belief_branch_id': str(row.get('belief_branch_id', '') or ''),
                    'belief_target_phase': str(row.get('belief_target_phase', '') or ''),
                    'belief_branch_confidence': float(row.get('belief_branch_confidence', 0.0) or 0.0),
                    'belief_anchor_functions': list(row.get('belief_anchor_functions', []) or []),
                    'belief_uncertainty_reduction': float(row.get('uncertainty_reduction_score', 0.0) or 0.0),
                }
                frontier_actions.append({
                    'rank': int(row.get('rank', 0) or 0),
                    'source': 'branch_frontier',
                    'target_function': target_function or residual_chain[0],
                    'score_gap': float(row.get('score_gap', 0.0) or 0.0),
                    'actions': [
                        {'function': fn_name, 'constraints': dict(branch_constraints), 'intent': 'explore'}
                        for fn_name in residual_chain[:horizon]
                    ],
                    'belief_branch_id': str(row.get('belief_branch_id', '') or ''),
                    'belief_target_phase': str(row.get('belief_target_phase', '') or ''),
                    'belief_branch_confidence': float(row.get('belief_branch_confidence', 0.0) or 0.0),
                })

        if not frontier_actions:
            fallback = constraints.get('fallback_functions', []) if isinstance(constraints.get('fallback_functions', []), list) else []
            for rank, fn_name in enumerate([str(fn).strip() for fn in fallback[:2] if str(fn).strip()], start=1):
                frontier_actions.append({
                    'rank': rank,
                    'source': 'fallback_functions',
                    'target_function': fn_name,
                    'score_gap': 0.0,
                    'actions': [{'function': fn_name, 'constraints': {}, 'intent': 'explore'}],
                })

        best_frontier: Optional[Dict[str, Any]] = None
        for row in frontier_actions:
            rollout_eval = self._score_rollout_actions(
                rollout_actions=row['actions'],
                priors=priors,
                hidden=hidden,
                obs=obs,
                wm_summary=wm_summary,
                planner_control_profile=planner_control_profile,
            )
            candidate = {
                'rank': row['rank'],
                'source': row['source'],
                'target_function': row['target_function'],
                'score_gap': row['score_gap'],
                'value': rollout_eval['value'],
                'risk': rollout_eval['risk'],
                'belief_value': rollout_eval.get('belief_value', 0.0),
                'persistence_value': rollout_eval.get('persistence_value', 0.0),
                'branch_persistence_ratio': rollout_eval.get('branch_persistence_ratio', 0.0),
                'rollout_branch_id': str(rollout_eval.get('rollout_branch_id', '') or ''),
                'rollout_branch_target_phase': str(rollout_eval.get('rollout_branch_target_phase', '') or ''),
                'rollout_branch_confidence': float(rollout_eval.get('rollout_branch_confidence', 0.0) or 0.0),
                'phase_path': list(rollout_eval.get('phase_path', []) or []),
                'final_phase': str(rollout_eval.get('final_phase', '') or ''),
                'belief_branch_id': str(row.get('belief_branch_id', '') or rollout_eval.get('belief_branch_id', '') or ''),
                'belief_target_phase': str(row.get('belief_target_phase', '') or rollout_eval.get('belief_target_phase', '') or ''),
                'belief_branch_confidence': float(row.get('belief_branch_confidence', 0.0) or 0.0),
                'trace': rollout_eval['trace'],
                'used_learned_dynamics': bool(rollout_eval.get('used_learned_dynamics', False)),
                'learned_coverage_ratio': float(rollout_eval.get('learned_coverage_ratio', 0.0) or 0.0),
            }
            candidate['selection_score'] = (
                float(candidate['value'])
                + float(candidate.get('persistence_value', 0.0) or 0.0) * 0.65
                + float(candidate.get('belief_value', 0.0) or 0.0) * 0.25
            )
            if (
                best_frontier is None
                or float(candidate['selection_score']) > float(best_frontier.get('selection_score', best_frontier['value']))
            ):
                best_frontier = candidate
        return best_frontier

    @staticmethod
    def _extract_transition_quad(prior: Any) -> Dict[str, float]:
        data = prior if isinstance(prior, dict) else {}
        return {
            'long_reward': float(data.get('long_horizon_reward', 0.0) or 0.0),
            'risk': float(data.get('predicted_risk', 0.0) or 0.0),
            'reversibility': float(data.get('reversibility', 0.0) or 0.0),
            'info_gain': float(data.get('info_gain', 0.0) or 0.0),
        }

    @staticmethod
    def _requires_probe_step(*, current_step: Any) -> bool:
        if current_step is None:
            return False
        target_function = str(getattr(current_step, 'target_function', '') or '')
        constraints = getattr(current_step, 'constraints', {}) if isinstance(getattr(current_step, 'constraints', {}), dict) else {}
        return bool(
            target_function == 'probe'
            or constraints.get('require_probe')
            or constraints.get('requires_probe')
            or constraints.get('must_probe')
        )

    @staticmethod
    def _estimate_uncertainty(wm_summary: Dict[str, Any], world_shift_risk: float) -> float:
        summary = wm_summary if isinstance(wm_summary, dict) else {}
        belief_state = summary.get('belief_state', {}) if isinstance(summary.get('belief_state', {}), dict) else {}
        uncertain = float(belief_state.get('uncertain_count', 0.0) or 0.0)
        total = float(belief_state.get('total_beliefs', 0.0) or 0.0)
        belief_uncertainty = (uncertain / total) if total > 0 else 0.0
        return max(float(world_shift_risk), belief_uncertainty)


def _extract_hidden_state_guidance(wm_summary: Dict[str, Any], wm_control: WorldModelControlProtocol) -> Dict[str, Any]:
    summary = wm_summary if isinstance(wm_summary, dict) else {}
    hidden = summary.get('hidden_state', {}) if isinstance(summary.get('hidden_state', {}), dict) else {}
    transition_memory = hidden.get('transition_memory', {}) if isinstance(hidden.get('transition_memory', {}), dict) else {}
    latent_branches = list(wm_control.latent_branches or hidden.get('latent_branches', transition_memory.get('latent_branches', [])) or [])
    dominant_branch_id = str(
        wm_control.dominant_branch_id
        or hidden.get('dominant_branch_id', transition_memory.get('dominant_branch_id', ''))
        or ''
    )
    return {
        'phase': str(wm_control.hidden_state_phase or hidden.get('phase', '') or ''),
        'phase_confidence': max(0.0, min(1.0, float(wm_control.hidden_phase_confidence or hidden.get('phase_confidence', 0.0) or 0.0))),
        'depth': max(0, int(wm_control.hidden_state_depth or hidden.get('hidden_state_depth', 0) or 0)),
        'drift_score': max(0.0, min(1.0, float(wm_control.hidden_drift_score or hidden.get('drift_score', 0.0) or 0.0))),
        'uncertainty_score': max(0.0, min(1.0, float(wm_control.hidden_uncertainty_score or hidden.get('uncertainty_score', 0.0) or 0.0))),
        'focus_functions': list(wm_control.hidden_focus_functions or hidden.get('focus_functions', []) or []),
        'latent_signature': str(wm_control.hidden_latent_signature or hidden.get('latent_signature', '') or ''),
        'dominant_branch_id': dominant_branch_id,
        'latent_branches': [dict(item) for item in latent_branches if isinstance(item, dict)],
        'expected_next_phase': str(wm_control.expected_next_phase or hidden.get('expected_next_phase', transition_memory.get('expected_next_phase', '')) or ''),
        'expected_next_phase_confidence': max(0.0, min(1.0, float(wm_control.expected_next_phase_confidence or hidden.get('expected_next_phase_confidence', transition_memory.get('expected_next_phase_confidence', 0.0)) or 0.0))),
        'transition_entropy': max(0.0, min(1.0, float(wm_control.phase_transition_entropy or hidden.get('transition_entropy', transition_memory.get('phase_transition_entropy', 1.0)) or 1.0))),
        'stabilizing_functions': list(wm_control.stabilizing_focus_functions or []),
        'risky_functions': list(wm_control.risky_focus_functions or []),
    }


def _build_planner_control_profile(planner_controls: Dict[str, Any]) -> Dict[str, Any]:
    controls = planner_controls if isinstance(planner_controls, dict) else {}
    policy = controls.get('policy_profile', {}) if isinstance(controls.get('policy_profile', {}), dict) else {}
    representation = controls.get('representation_profile', {}) if isinstance(controls.get('representation_profile', {}), dict) else {}
    retention_tuning = controls.get('retention_tuning', policy.get('retention_tuning', {}))
    retention_tuning = retention_tuning if isinstance(retention_tuning, dict) else {}
    planner_bias = max(0.0, min(1.0, float(controls.get('planner_bias', policy.get('planner_bias', 0.5)) or 0.5)))
    retrieval_pressure = max(
        0.0,
        min(
            1.0,
            float(
                representation.get(
                    'retrieval_pressure',
                    representation.get('retrieval_aggressiveness', policy.get('retrieval_aggressiveness', 0.5)),
                ) or 0.5
            ),
        ),
    )
    verification_bias = max(
        0.0,
        min(1.0, float(policy.get('verification_bias', representation.get('verification_bias', 0.5)) or 0.5)),
    )
    risk_tolerance = max(0.0, min(1.0, float(policy.get('risk_tolerance', 0.5) or 0.5)))
    recovery_bias = max(0.0, min(1.0, float(policy.get('recovery_bias', representation.get('recovery_bias', 0.5)) or 0.5)))
    stability_bias = max(0.0, min(1.0, float(policy.get('stability_bias', representation.get('stability_bias', 0.5)) or 0.5)))
    strategy_mode = str(policy.get('strategy_mode', representation.get('strategy_mode', 'balanced')) or 'balanced')

    branch_budget = 2
    if strategy_mode == 'explore':
        branch_budget = 4 if retrieval_pressure >= 0.62 else 3
    elif strategy_mode in {'verify', 'recover'}:
        branch_budget = 3
    elif verification_bias >= 0.74 or retrieval_pressure >= 0.70:
        branch_budget = 3

    verification_budget = 0
    if strategy_mode in {'recover', 'verify'}:
        verification_budget = 2
    elif verification_bias >= 0.72 or recovery_bias >= 0.74:
        verification_budget = 1

    periodic_replan_interval = 3
    periodic_bias_threshold = 0.80
    world_shift_replan_threshold = 0.70
    world_shift_bias_threshold = 0.60
    hidden_drift_replan_threshold = 0.72
    hidden_disruption_threshold = 0.58
    negative_reward_bias_threshold = 0.70
    high_risk_replan_threshold = 0.75
    safer_risk_margin = 0.15
    value_drop_threshold = 0.40
    uncertainty_threshold = 0.70
    belief_branch_margin_threshold = 0.10
    branch_persistence_margin_threshold = 0.18
    low_branch_persistence_threshold = 0.38

    if strategy_mode == 'recover':
        periodic_replan_interval = 2
        periodic_bias_threshold = 0.72
        world_shift_replan_threshold = 0.55
        world_shift_bias_threshold = 0.45
        hidden_drift_replan_threshold = 0.64
        hidden_disruption_threshold = 0.50
        negative_reward_bias_threshold = 0.55
        high_risk_replan_threshold = 0.62
        safer_risk_margin = 0.08
        value_drop_threshold = 0.22
        uncertainty_threshold = 0.52
        belief_branch_margin_threshold = 0.06
        branch_persistence_margin_threshold = 0.10
        low_branch_persistence_threshold = 0.32
    elif strategy_mode == 'verify':
        periodic_replan_interval = 3
        periodic_bias_threshold = 0.76
        world_shift_replan_threshold = 0.60
        world_shift_bias_threshold = 0.50
        hidden_drift_replan_threshold = 0.68
        hidden_disruption_threshold = 0.54
        negative_reward_bias_threshold = 0.60
        high_risk_replan_threshold = 0.68
        safer_risk_margin = 0.10
        value_drop_threshold = 0.28
        uncertainty_threshold = 0.58
        belief_branch_margin_threshold = 0.07
        branch_persistence_margin_threshold = 0.12
        low_branch_persistence_threshold = 0.34
    elif strategy_mode == 'explore':
        periodic_replan_interval = 4
        periodic_bias_threshold = 0.78
        world_shift_replan_threshold = 0.68
        world_shift_bias_threshold = 0.55
        hidden_drift_replan_threshold = 0.72
        hidden_disruption_threshold = 0.58
        negative_reward_bias_threshold = 0.68
        high_risk_replan_threshold = 0.72
        safer_risk_margin = 0.12
        value_drop_threshold = 0.36
        uncertainty_threshold = 0.64
        belief_branch_margin_threshold = 0.09
        branch_persistence_margin_threshold = 0.16
        low_branch_persistence_threshold = 0.36
    elif strategy_mode == 'exploit':
        periodic_replan_interval = 5
        periodic_bias_threshold = 0.84
        world_shift_replan_threshold = 0.76
        world_shift_bias_threshold = 0.68
        hidden_drift_replan_threshold = 0.78
        hidden_disruption_threshold = 0.62
        negative_reward_bias_threshold = 0.80
        high_risk_replan_threshold = 0.82
        safer_risk_margin = 0.20
        value_drop_threshold = 0.50
        uncertainty_threshold = 0.78
        belief_branch_margin_threshold = 0.14
        branch_persistence_margin_threshold = 0.22
        low_branch_persistence_threshold = 0.42

    branch_budget += max(0, min(2, int(round(float(retention_tuning.get('branch_budget_bonus', 0.0) or 0.0)))))
    verification_budget += max(0, min(2, int(round(float(retention_tuning.get('verification_budget_bonus', 0.0) or 0.0)))))
    branch_budget = max(1, min(4, int(branch_budget)))
    verification_budget = max(0, min(3, int(verification_budget)))

    world_shift_replan_threshold = max(
        0.35,
        min(0.90, float(world_shift_replan_threshold) + float(retention_tuning.get('world_shift_replan_threshold_delta', 0.0) or 0.0)),
    )
    hidden_drift_replan_threshold = max(
        0.35,
        min(0.90, float(hidden_drift_replan_threshold) + float(retention_tuning.get('hidden_drift_replan_threshold_delta', 0.0) or 0.0)),
    )
    high_risk_replan_threshold = max(
        0.45,
        min(0.92, float(high_risk_replan_threshold) + float(retention_tuning.get('high_risk_replan_threshold_delta', 0.0) or 0.0)),
    )
    value_drop_threshold = max(
        0.12,
        min(0.60, float(value_drop_threshold) + float(retention_tuning.get('value_drop_threshold_delta', 0.0) or 0.0)),
    )
    uncertainty_threshold = max(
        0.35,
        min(0.88, float(uncertainty_threshold) + float(retention_tuning.get('uncertainty_threshold_delta', 0.0) or 0.0)),
    )
    belief_branch_margin_threshold = max(
        0.02,
        min(0.22, float(belief_branch_margin_threshold) + float(retention_tuning.get('belief_branch_margin_threshold_delta', 0.0) or 0.0)),
    )
    branch_persistence_margin_threshold = max(
        0.04,
        min(0.30, float(branch_persistence_margin_threshold) + float(retention_tuning.get('branch_persistence_margin_threshold_delta', 0.0) or 0.0)),
    )
    low_branch_persistence_threshold = max(
        0.18,
        min(0.65, float(low_branch_persistence_threshold) + float(retention_tuning.get('low_branch_persistence_threshold_delta', 0.0) or 0.0)),
    )

    return {
        'strategy_mode': strategy_mode,
        'branch_budget': int(branch_budget),
        'verification_budget': int(verification_budget),
        'periodic_replan_interval': int(periodic_replan_interval),
        'periodic_bias_threshold': float(periodic_bias_threshold),
        'world_shift_replan_threshold': float(world_shift_replan_threshold),
        'world_shift_bias_threshold': float(world_shift_bias_threshold),
        'hidden_drift_replan_threshold': float(hidden_drift_replan_threshold),
        'hidden_disruption_threshold': float(hidden_disruption_threshold),
        'negative_reward_bias_threshold': float(negative_reward_bias_threshold),
        'high_risk_replan_threshold': float(high_risk_replan_threshold),
        'safer_risk_margin': float(safer_risk_margin),
        'value_drop_threshold': float(value_drop_threshold),
        'uncertainty_threshold': float(uncertainty_threshold),
        'belief_branch_margin_threshold': float(belief_branch_margin_threshold),
        'branch_persistence_margin_threshold': float(branch_persistence_margin_threshold),
        'low_branch_persistence_threshold': float(low_branch_persistence_threshold),
        'planner_bias': float(planner_bias),
        'retrieval_pressure': float(retrieval_pressure),
        'verification_bias': float(verification_bias),
        'risk_tolerance': float(risk_tolerance),
        'recovery_bias': float(recovery_bias),
        'stability_bias': float(stability_bias),
        'retention_tuning': dict(retention_tuning),
    }


def _apply_self_model_planner_modulation(
    profile: Dict[str, Any],
    self_model_summary: Dict[str, Any],
) -> Dict[str, Any]:
    base = dict(profile or {})
    summary = self_model_summary if isinstance(self_model_summary, dict) else {}
    capability_envelope = summary.get('capability_envelope', {})
    capability_envelope = capability_envelope if isinstance(capability_envelope, dict) else {}
    modulation = summary.get('planner_control_profile', {})
    modulation = modulation if isinstance(modulation, dict) else {}

    strategy_mode = str(
        modulation.get(
            'strategy_mode',
            capability_envelope.get('strategy_mode_hint', base.get('strategy_mode', 'balanced')),
        ) or base.get('strategy_mode', 'balanced')
    ).strip() or 'balanced'
    base['strategy_mode'] = strategy_mode

    branch_delta = int(modulation.get('branch_budget_delta', capability_envelope.get('branch_budget_delta', 0)) or 0)
    verification_delta = int(modulation.get('verification_budget_delta', capability_envelope.get('verification_budget_delta', 0)) or 0)
    search_depth_bias = int(modulation.get('search_depth_bias', capability_envelope.get('search_depth_bias', 0)) or 0)

    base['branch_budget'] = max(1, min(4, int(base.get('branch_budget', 2) or 2) + branch_delta))
    base['verification_budget'] = max(0, min(3, int(base.get('verification_budget', 0) or 0) + verification_delta))
    if 'lookahead_horizon' in base:
        base['lookahead_horizon'] = max(1, min(6, int(base.get('lookahead_horizon', 2) or 2) + search_depth_bias))
    base['fallback_bias'] = str(modulation.get('fallback_bias', capability_envelope.get('fallback_bias', 'balanced')) or 'balanced')
    base['teacher_off_escalation'] = bool(
        modulation.get('teacher_off_escalation', capability_envelope.get('teacher_off_escalation', False))
    )
    if base['teacher_off_escalation']:
        base['strategy_mode'] = 'recover'
        base['verification_budget'] = max(1, int(base.get('verification_budget', 0) or 0))
        base['branch_budget'] = max(2, int(base.get('branch_budget', 2) or 2))
    return base


def _synthetic_goal_from_plan(plan: Any) -> Any:
    if plan is None:
        return None
    goal_hint = str(getattr(plan, 'goal', '') or '').lower()
    if any(token in goal_hint for token in ('test', 'probe', '测试')):
        goal_id = 'test_replan'
    elif any(token in goal_hint for token in ('confirm', 'verify', '确认')):
        goal_id = 'confirm_replan'
    elif any(token in goal_hint for token in ('exploit', '利用')):
        goal_id = 'exploit_replan'
    else:
        goal_id = 'explore_replan'
    return SimpleNamespace(goal_id=goal_id)

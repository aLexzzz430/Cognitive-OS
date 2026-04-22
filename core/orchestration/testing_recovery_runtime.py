from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.orchestration.adjudication_flow import AdjudicationFlow
from core.orchestration.prediction_flow import PredictionFlow
from core.orchestration.probe_flow import ProbeFlow
from core.orchestration.recovery_flow import RecoveryFlow


@dataclass
class TeacherInterventionPolicy:
    """Centralized teacher intervention policy for adjudication/critique/task injection."""

    experiment_mode: str
    episode: int
    tick: int

    def allows_intervention(self) -> bool:
        mode = str(self.experiment_mode or 'full').lower()
        if mode in ('no', 'no_teacher'):
            return False
        if mode == 'freeze':
            return self.episode <= 1
        if mode == 'delayed':
            return self.episode >= 3
        if mode == 'weaken':
            return (self.tick % 2) == 0
        return True

    def should_adjudicate(self, competing_pair: Any) -> bool:
        return bool(competing_pair) and self.allows_intervention()

    def should_critique(self, has_test_result: bool, has_probe_candidates: bool) -> bool:
        return bool(has_test_result and has_probe_candidates and self.allows_intervention())

    def should_inject_task(self, recovery_path: Any) -> bool:
        return recovery_path is not None and self.allows_intervention()


@dataclass
class TestingRecoveryDeps:
    probe: Any
    teacher: Any
    prediction: Any
    recovery: Any
    world: Any
    event: Any
    continuity: Any


@dataclass
class TestingRecoveryResult:
    effects: List[Dict[str, Any]] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    state_patch: Dict[str, Any] = field(default_factory=dict)


class TestingRecoveryRuntime:
    """Owns unified testing + recovery runtime flow and returns patch outputs."""

    __test__ = False

    def __init__(self, loop: Any):
        self.loop = loop
        self.deps = TestingRecoveryDeps(
            probe=getattr(loop, '_probe_designer', None),
            teacher=getattr(loop, '_teacher', None),
            prediction=getattr(loop, '_prediction_engine', None),
            recovery=getattr(loop, '_recovery', None),
            world=getattr(loop, '_world', None),
            event=getattr(loop, '_event_bus', None),
            continuity=getattr(getattr(loop, '_continuity', None), 'agenda', None),
        )
        self.adjudication_flow = AdjudicationFlow(loop)
        self.prediction_flow = PredictionFlow(loop)
        self.probe_flow = ProbeFlow(loop, self.adjudication_flow, self.prediction_flow)
        self.recovery_flow = RecoveryFlow(loop, self.adjudication_flow)

    def run_testing_and_recovery(self, obs_before: dict, surfaced: list, action_to_use: dict, result: dict, frame: Any) -> TestingRecoveryResult:
        out = TestingRecoveryResult(
            telemetry={
                'probe_result': None,
                'teacher_adjudication': None,
                'recovery_decision': None,
            },
            state_patch={
                'pending_replan_patch': None,
                'pending_probe_patch': None,
            },
        )
        policy = TeacherInterventionPolicy(
            experiment_mode=str(getattr(self.loop, '_teacher_experiment_mode', 'full')),
            episode=int(self.loop._episode),
            tick=int(self.loop._tick),
        )

        if surfaced:
            known_fns = self._extract_known_functions(obs_before)
            competing_pair = self.loop._hypotheses.discriminating_pair()
            diagnosis = self._build_recovery_diagnosis(action_to_use, result, obs_before)
            out.telemetry['recovery_diagnosis'] = diagnosis
            out.telemetry['teacher_adjudication'] = self.adjudication_flow.handle_teacher_adjudication(competing_pair, policy)
            test_advice = self.loop._probe_designer.should_run_test(competing_pair)
            if not isinstance(test_advice, dict):
                test_advice = {'should_test': bool(test_advice), 'urgency': 'medium', 'reason': 'non_dict_probe_advice'}
            test_advice = self._apply_probe_policy_bias(test_advice)
            self.loop._llm_advice_log.append({'tick': self.loop._tick, 'episode': self.loop._episode, 'kind': 'probe_gate', 'advice': test_advice})

            pending_probe_patch = self.probe_flow.consume_pending_recovery_probe(known_fns)
            if pending_probe_patch is not None:
                out.state_patch['pending_probe_patch'] = pending_probe_patch

            probe_candidates = self.loop._probe_designer.generate_probe_candidates(competing_pair, known_fns, top_k=3)
            probe_candidates = self.loop._rank_probe_candidates_by_prediction(probe_candidates, obs_before, surfaced, frame)
            probe_candidates = self._rank_probe_candidates_with_diagnosis(probe_candidates, diagnosis)
            test = self.loop._test_engine.generate(known_fns) if test_advice.get('should_test', True) else None
            if test and probe_candidates:
                probe = probe_candidates[0]
                test.test_function = probe.target_function
                test.test_params = probe.test_params
                self.probe_flow.maybe_emit_test_generated(test, probe, out.effects, out.telemetry)
                out.telemetry['probe_result'] = self.probe_flow.run_probe_test(
                    test=test,
                    probe_candidates=probe_candidates,
                    obs_before=obs_before,
                    surfaced=surfaced,
                    frame=frame,
                    policy=policy,
                    effects=out.effects,
                    telemetry=out.telemetry,
                )
                if isinstance(out.telemetry.get('probe_result'), dict):
                    expected_info_gain = float(getattr(probe, 'expected_information_gain', 0.0) or 0.0)
                    actual_info_gain = self._estimate_actual_probe_info_gain(out.telemetry['probe_result'])
                    out.telemetry['probe_result']['info_gain_calibration'] = {
                        'expected_info_gain': round(expected_info_gain, 4),
                        'actual_info_gain': round(actual_info_gain, 4),
                        'delta': round(actual_info_gain - expected_info_gain, 4),
                        'ranking_details': getattr(probe, 'ranking_details', {}),
                    }

        out.telemetry['recovery_decision'] = self.recovery_flow.handle_recovery_if_needed(
            action=action_to_use,
            result=result,
            policy=policy,
            effects=out.effects,
        )
        if isinstance(out.telemetry['recovery_decision'], dict):
            out.state_patch['pending_replan_patch'] = out.telemetry['recovery_decision'].get('pending_replan_patch')
            if out.state_patch['pending_probe_patch'] is None:
                out.state_patch['pending_probe_patch'] = out.telemetry['recovery_decision'].get('pending_probe_patch')
            self._enrich_pending_probe_target_metadata(out.state_patch, diagnosis=out.telemetry.get('recovery_diagnosis', {}), frame=frame)
            out.telemetry['recovery_path_rank'] = self._rank_recovery_path(out.telemetry['recovery_decision'])
        return out

    def apply_effects(self, effects: List[Dict[str, Any]], telemetry: Optional[Dict[str, Any]] = None) -> None:
        for effect in effects:
            command = effect.get('command')
            args = effect.get('args', {})
            try:
                if command == 'event_emit':
                    event_bus = getattr(self.loop, '_event_bus', None)
                    if event_bus is None:
                        raise RuntimeError('event_bus_unavailable')
                    event_bus.emit(args['event'])
                elif command == 'world_act':
                    action = args.get('action', {})
                    raw_result = self.loop._world.act(action)
                    test_function = action.get('payload', {}).get('tool_args', {}).get('function_name')
                    action_id = str(raw_result.get('action_id', f"test_{self.loop._episode}_{self.loop._tick}_{test_function}")) if isinstance(raw_result, dict) else f"test_{self.loop._episode}_{self.loop._tick}_{test_function}"
                    self.loop._governance_log.append({
                        'tick': self.loop._tick,
                        'episode': self.loop._episode,
                        'entry': 'test_execution',
                        'source_stage': 'testing',
                        'test_lane': True,
                        'is_test_execution': True,
                        'counts_toward_main_action': False,
                        'test_function': test_function,
                        'action_id': action_id,
                    })
                elif command == 'continuity_add_task':
                    self.loop._continuity.agenda.add_task(
                        args['task_id'],
                        args['description'],
                        priority=args.get('priority', 0.95),
                        metadata=args.get('metadata', {}),
                    )
            except Exception as exc:
                if telemetry is not None:
                    telemetry.setdefault('warnings', []).append({'command': command, 'error': str(exc)})

    def _extract_known_functions(self, obs_before: dict) -> List[str]:
        api_raw = obs_before.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else []
        return list(discovered)

    def _execute_test_action(self, test: Any) -> dict:
        return self.probe_flow.execute_test_action(test)

    def _apply_probe_policy_bias(self, test_advice: Dict[str, Any]) -> Dict[str, Any]:
        adjusted = dict(test_advice or {})
        if hasattr(self.loop, '_meta_control'):
            controls = self.loop._meta_control.for_probe_gate(episode=self.loop._episode, tick=self.loop._tick, context={'gate': 'probe'})
            probe_bias = float(controls['probe_bias'])
            adjusted['probe_bias'] = probe_bias
            if probe_bias >= 0.75:
                adjusted['should_test'] = True
                adjusted['urgency'] = 'high'
                adjusted['policy_reason'] = 'meta_probe_bias_high'
            elif probe_bias <= 0.25 and not self.loop._pending_recovery_probe:
                adjusted['should_test'] = False
                adjusted['urgency'] = 'low'
                adjusted['policy_reason'] = 'meta_probe_bias_low'
            adjusted['meta_control_snapshot_id'] = controls.get('meta_control_snapshot_id', '')
            adjusted['meta_control_inputs_hash'] = controls.get('meta_control_inputs_hash', '')
        return adjusted

    def _build_recovery_diagnosis(self, action: Dict[str, Any], result: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
        fn_name = (
            action.get('payload', {}).get('tool_args', {}).get('function_name')
            if isinstance(action, dict) else None
        ) or ('wait' if isinstance(action, dict) and action.get('kind') == 'wait' else 'unknown')
        na_result = result.get('novel_api', {}) if isinstance(result, dict) else {}
        if hasattr(na_result, '_data'):
            na_result = na_result._data
        error = str(na_result.get('error', '') or '') if isinstance(na_result, dict) else ''
        discovered = self._extract_known_functions(obs_before)
        return {
            'function_name': fn_name,
            'has_error': bool(error),
            'error': error,
            'known_function': fn_name in discovered if fn_name and fn_name != 'wait' else True,
            'surface_signal_strength': len(discovered),
        }

    def _rank_probe_candidates_with_diagnosis(self, probe_candidates: List[Any], diagnosis: Dict[str, Any]) -> List[Any]:
        if not probe_candidates:
            return []
        has_error = bool((diagnosis or {}).get('has_error'))
        known_fn = bool((diagnosis or {}).get('known_function', True))

        def _score(candidate: Any) -> float:
            fn = str(getattr(candidate, 'target_function', '') or '')
            base = 1.0
            if has_error:
                base += 0.4
            if not known_fn and fn:
                base += 0.3
            if fn and fn == str((diagnosis or {}).get('function_name', '')):
                base += 0.2
            return base

        return sorted(probe_candidates, key=_score, reverse=True)

    def _rank_recovery_path(self, recovery_decision: Dict[str, Any]) -> Dict[str, Any]:
        path = recovery_decision.get('path', {}) if isinstance(recovery_decision, dict) else {}
        recovery_type = str(path.get('recovery_type', '') or '').lower()
        estimated_success = float(path.get('estimated_success', 0.0) or 0.0)
        priority = estimated_success
        if recovery_type == 'request_replan':
            priority += 0.1
        elif recovery_type == 'request_probe':
            priority += 0.05
        return {
            'recovery_type': recovery_type,
            'estimated_success': estimated_success,
            'priority_score': round(priority, 3),
        }

    def _handle_recovery_if_needed(self, action: dict, result: dict) -> Optional[Dict[str, Any]]:
        policy = TeacherInterventionPolicy(
            experiment_mode=str(getattr(self.loop, '_teacher_experiment_mode', 'full')),
            episode=int(self.loop._episode),
            tick=int(self.loop._tick),
        )
        effects: List[Dict[str, Any]] = []
        recovery_event = self.recovery_flow.handle_recovery_if_needed(action, result, policy, effects)
        self.apply_effects(effects)
        return recovery_event

    def _enrich_pending_probe_target_metadata(self, state_patch: Dict[str, Any], diagnosis: Dict[str, Any], frame: Any) -> None:
        pending = state_patch.get('pending_probe_patch')
        if not isinstance(pending, dict):
            return
        wm_summary = getattr(frame, 'world_model_summary', {}) if frame is not None else {}
        beliefs = wm_summary.get('beliefs', {}) if isinstance(wm_summary, dict) else {}
        high_value = wm_summary.get('high_value_beliefs', []) if isinstance(wm_summary, dict) else []
        hv_lookup = {
            str(item.get('variable', '')): item
            for item in high_value if isinstance(item, dict) and item.get('variable')
        }
        targets: List[Dict[str, Any]] = []
        for variable, payload in beliefs.items() if isinstance(beliefs, dict) else []:
            if not isinstance(payload, dict):
                continue
            confidence = float(payload.get('confidence', 0.0) or 0.0)
            if confidence >= 0.65:
                continue
            impact_scope = str(hv_lookup.get(str(variable), {}).get('impact_scope', 'decision_only'))
            impact = 1.0 if impact_scope == 'planner+decision' else 0.65
            if impact < 0.65:
                continue
            targets.append({
                'belief': str(variable),
                'confidence': round(max(0.0, min(1.0, confidence)), 4),
                'impact': round(impact, 4),
                'priority': round((1.0 - max(0.0, min(1.0, confidence))) * impact, 4),
            })
        targets.sort(key=lambda item: item.get('priority', 0.0), reverse=True)
        diagnosis_text = f"{diagnosis.get('function_name', '')} {diagnosis.get('error', '')} {diagnosis.get('root_cause', '')}".strip().lower()
        mechanism_hint = ''
        if targets:
            mechanism_hint = f"disambiguate_{targets[0]['belief']}"
        if diagnosis_text:
            mechanism_hint = f"{mechanism_hint}::{diagnosis.get('function_name', 'unknown')}" if mechanism_hint else diagnosis.get('function_name', 'unknown')
        pending['target_metadata'] = {
            'belief_targets': targets[:3],
            'mechanism_hint': mechanism_hint or 'recovery_probe_disambiguation',
        }
        pending['selection_reason'] = 'uncertainty_aware_recovery_probe'

    def _estimate_actual_probe_info_gain(self, probe_result: Dict[str, Any]) -> float:
        if not isinstance(probe_result, dict):
            return 0.0
        raw = probe_result.get('test_result', {})
        if not isinstance(raw, dict):
            return 0.0
        novel_api = raw.get('novel_api', {})
        if hasattr(novel_api, '_data'):
            novel_api = novel_api._data
        if not isinstance(novel_api, dict):
            novel_api = {}
        if novel_api.get('error'):
            return 0.05
        signal = 0.2 + (0.12 * len([k for k, v in novel_api.items() if v not in (None, '', [], {})]))
        return max(0.0, min(1.0, signal))

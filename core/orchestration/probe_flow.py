from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.world_model.events import EventType, WorldModelEvent
from modules.world_model.mechanism import MechanismHypothesis


class ProbeFlow:
    """Probe/test workflow that emits side effects into an effect list."""

    def __init__(self, loop: Any, adjudication_flow: Any, prediction_flow: Any):
        self.loop = loop
        self.adjudication_flow = adjudication_flow
        self.prediction_flow = prediction_flow

    def consume_pending_recovery_probe(self, known_fns: List[str]) -> Optional[Dict[str, Any]]:
        pending = self.loop._pending_recovery_probe
        if not pending:
            return None
        diagnosis = pending['diagnosis']
        recovery_probe = self.loop._probe_designer.generate_probe_candidates(
            competing_pair=None,
            known_fns=known_fns,
            top_k=1,
            recovery_context={
                'diagnosis': diagnosis.description,
                'error_type': diagnosis.error_type.value,
                'root_cause': diagnosis.root_cause_hypothesis,
                'target_metadata': pending.get('target_metadata', {}) if isinstance(pending, dict) else {},
            },
        )
        recovery_task_id = pending.get('task_id') if isinstance(pending, dict) else None
        patch: Dict[str, Any] = {'clear_pending_probe': True, 'task_id': recovery_task_id}
        if recovery_probe:
            probe = recovery_probe[0]
            test = self.loop._test_engine.generate(known_fns)
            test.test_function = probe.target_function
            test.test_params = probe.test_params
            test.related_hypothesis = f"recovery_{diagnosis.error_type.value}"
            self.loop._teacher_log.append({
                'tick': self.loop._tick,
                'episode': self.loop._episode,
                'entry': 'recovery_probe_injected',
                'diagnosis': diagnosis.error_type.value,
                'recovery_task': pending['task_id'],
                'target_metadata': pending.get('target_metadata', {}) if isinstance(pending, dict) else {},
            })
            self.loop._mark_continuity_task_completed(recovery_task_id, reason='recovery_probe_injected')
            patch['probe_injected'] = {'function': probe.target_function, 'params': probe.test_params}
        else:
            self.loop._mark_continuity_task_cancelled(recovery_task_id, reason='recovery_probe_unavailable')
            patch['probe_injected'] = None
        return patch

    def maybe_emit_test_generated(self, test: Any, probe: Any, effects: list, telemetry: dict) -> None:
        if not test or not probe:
            return
        effects.append({
            'command': 'event_emit',
            'args': {
                'event': WorldModelEvent(
                    event_type=EventType.TEST_GENERATED,
                    episode=self.loop._episode,
                    tick=self.loop._tick,
                    data={'test_function': test.test_function, 'test_params': test.test_params, 'has_probe': probe is not None},
                    source_stage='testing',
                )
            },
        })

    def run_probe_test(self, test: Any, probe_candidates: List[Any], obs_before: dict, surfaced: list, frame: Any, policy: Any, effects: list, telemetry: dict) -> Dict[str, Any]:
        probe_hyp_before = list(self.loop._hypotheses.get_active())
        probe_bundle = self.prediction_flow.maybe_predict_probe(probe_candidates, obs_before, surfaced, frame, probe_hyp_before)

        test_result = self.execute_test_action(test, effects, telemetry)
        self.loop._test_engine.receive(test, test_result)
        extracted_mechanisms: List[MechanismHypothesis] = self.loop._extract_mechanism_candidates({
            'test_function': test.test_function,
            'test_result': test.result,
            'evidence_id': f"test_{self.loop._episode}_{self.loop._tick}",
        })
        committed_mechanism_ids: List[str] = self.loop._commit_mechanism_candidates_formal(extracted_mechanisms)
        if committed_mechanism_ids:
            self.loop._governance_log.append({'tick': self.loop._tick, 'episode': self.loop._episode, 'entry': 'mechanism_committed', 'count': len(committed_mechanism_ids), 'mechanism_object_ids': committed_mechanism_ids})
            supports = bool(test.result is True)
            for mechanism_id in committed_mechanism_ids:
                effects.append({
                    'command': 'event_emit',
                    'args': {
                        'event': WorldModelEvent(
                            event_type=EventType.MECHANISM_EVIDENCE_ADDED,
                            episode=self.loop._episode,
                            tick=self.loop._tick,
                            data={
                                'mechanism_id': mechanism_id,
                                'target_function': test.test_function,
                                'supports': supports,
                                'evidence_id': f"test_{self.loop._episode}_{self.loop._tick}",
                            },
                            source_stage='testing',
                        ),
                    },
                })

        if self.loop._prediction_enabled and probe_bundle is not None:
            self.prediction_flow.record_probe_prediction_feedback(probe_bundle, test, test_result, obs_before, probe_hyp_before)

        if probe_candidates:
            info_gain_note = self.loop._probe_designer.explain_information_gain(test_result.get('novel_api', {}), probe_candidates[0])
            self.loop._llm_advice_log.append({'tick': self.loop._tick, 'episode': self.loop._episode, 'kind': 'probe_result', 'note': info_gain_note})
            self.adjudication_flow.emit_teacher_critique(test, probe_candidates[0], policy)

        if test.result is not None and probe_candidates:
            probe = probe_candidates[0]
            confirmed_hyp = probe.hypothesis_a if test.result else probe.hypothesis_b
            falsified_hyp = probe.hypothesis_b if test.result else probe.hypothesis_a
            effects.append({'command': 'event_emit', 'args': {'event': WorldModelEvent(event_type=EventType.HYPOTHESIS_UPDATED, episode=self.loop._episode, tick=self.loop._tick, data={'hypothesis_id': confirmed_hyp, 'verdict': 'confirmed', 'test_function': test.test_function}, source_stage='testing')}})
            effects.append({'command': 'event_emit', 'args': {'event': WorldModelEvent(event_type=EventType.HYPOTHESIS_UPDATED, episode=self.loop._episode, tick=self.loop._tick, data={'hypothesis_id': falsified_hyp, 'verdict': 'falsified', 'test_function': test.test_function}, source_stage='testing')}})

        effects.append({'command': 'event_emit', 'args': {'event': WorldModelEvent(event_type=EventType.TEST_EXECUTED, episode=self.loop._episode, tick=self.loop._tick, data={'test_function': test.test_function, 'test_result': test.result}, source_stage='testing')}})
        related_hypothesis = str(getattr(test, 'related_hypothesis', '') or '')
        if related_hypothesis.startswith('recovery_'):
            recovery_type = related_hypothesis.replace('recovery_', '', 1) or 'unknown'
            effects.append({
                'command': 'event_emit',
                'args': {
                    'event': WorldModelEvent(
                        event_type=EventType.RECOVERY_OUTCOME_OBSERVED,
                        episode=self.loop._episode,
                        tick=self.loop._tick,
                        data={
                            'recovery_type': recovery_type,
                            'success': bool(test.result is True),
                            'function_name': test.test_function,
                            'evidence_id': f"recovery_probe_{self.loop._episode}_{self.loop._tick}",
                        },
                        source_stage='testing',
                    )
                },
            })
        return {'test_function': test.test_function, 'test_result': test.result}

    def execute_test_action(self, test: Any, effects: Optional[list] = None, telemetry: Optional[dict] = None) -> dict:
        action = {
            'kind': 'action',
            'payload': {'tool_name': 'call_hidden_function', 'tool_args': {'function_name': test.test_function, 'kwargs': test.test_params}},
            '_source': 'test_lane',
        }
        if effects is not None:
            effects.append({'command': 'world_act', 'args': {'action': action, 'label': 'test_execution'}})
            return {}
        try:
            raw_result = self.loop._world.act(action)
        except Exception as exc:
            if telemetry is not None:
                telemetry.setdefault('warnings', []).append({'kind': 'world_act_failed', 'error': str(exc)})
            return {}
        action_id = str(raw_result.get('action_id', f"test_{self.loop._episode}_{self.loop._tick}_{test.test_function}")) if isinstance(raw_result, dict) else f"test_{self.loop._episode}_{self.loop._tick}_{test.test_function}"
        self.loop._governance_log.append({
            'tick': self.loop._tick,
            'episode': self.loop._episode,
            'entry': 'test_execution',
            'source_stage': 'testing',
            'test_lane': True,
            'is_test_execution': True,
            'counts_toward_main_action': False,
            'test_function': test.test_function,
            'action_id': action_id,
        })
        return raw_result if isinstance(raw_result, dict) else {}

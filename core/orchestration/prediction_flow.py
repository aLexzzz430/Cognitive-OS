from __future__ import annotations

from typing import Any, List, Optional


class PredictionFlow:
    """Prediction-specific steps used by testing/probe runtime."""

    def __init__(self, loop: Any):
        self.loop = loop

    @staticmethod
    def _world_model_support(summary: Any) -> dict:
        if not isinstance(summary, dict):
            return {}
        return {
            'predicted_transitions': [
                dict(row) for row in list(summary.get('predicted_transitions', []) or [])[:4]
                if isinstance(row, dict)
            ],
            'mechanism_hypotheses': [
                dict(row) for row in list(summary.get('mechanism_hypotheses', []) or [])[:3]
                if isinstance(row, dict)
            ],
            'candidate_intervention_targets': [
                dict(row) for row in list(summary.get('candidate_intervention_targets', []) or [])[:3]
                if isinstance(row, dict)
            ],
            'discriminating_tests': [
                dict(row) for row in list(summary.get('discriminating_tests', []) or [])[:4]
                if isinstance(row, dict)
            ],
            'expected_information_gain': float(summary.get('expected_information_gain', 0.0) or 0.0),
            'rollout_uncertainty': float(summary.get('rollout_uncertainty', 0.5) or 0.5),
        }

    def maybe_predict_probe(self, probe_candidates: List[Any], obs_before: dict, surfaced: list, frame: Any, probe_hyp_before: List[Any]) -> Optional[Any]:
        if not self.loop._prediction_runtime_active() or not probe_candidates:
            return None
        world_model_summary = frame.world_model_summary if isinstance(getattr(frame, 'world_model_summary', {}), dict) else {}
        world_model_support = self._world_model_support(world_model_summary)
        probe_action = {
            'kind': 'probe',
            '_source': 'probe_gate',
            'payload': {'tool_args': {'function_name': probe_candidates[0].target_function, 'kwargs': probe_candidates[0].test_params}},
            '_candidate_meta': {
                'world_model_support': world_model_support,
                'expected_information_gain': float(world_model_support.get('expected_information_gain', 0.0) or 0.0),
            },
        }
        self.loop._build_action_id(probe_action)
        return self.loop._prediction_engine.predict_action(
            episode=self.loop._episode,
            tick=self.loop._tick,
            action=probe_action,
            obs=obs_before,
            surfaced=surfaced,
            hypotheses=probe_hyp_before,
            belief_summary=world_model_summary,
            plan_summary=self.loop._plan_state.get_plan_summary(),
            step_intent=self.loop._plan_state.get_intent_for_step(),
            recent_trace=self.loop._episode_trace[-5:],
            self_model_summary=frame.self_model_summary,
            policy_profile=self.loop._get_policy_profile(),
            recovery_context=self.loop._build_recovery_prediction_context(),
        )

    def record_probe_prediction_feedback(self, probe_bundle: Any, test: Any, test_result: dict, obs_before: dict, probe_hyp_before: List[Any]) -> None:
        if not self.loop._prediction_runtime_active() or probe_bundle is None:
            return
        probe_hyp_after = list(self.loop._hypotheses.get_active())
        probe_outcome = self.loop._prediction_adjudicator.build_outcome_record(
            episode=self.loop._episode,
            tick=self.loop._tick,
            action_id=probe_bundle.action_id,
            function_name=test.test_function or 'probe',
            result=test_result if isinstance(test_result, dict) else {},
            reward=float(test_result.get('reward', 0.0)) if isinstance(test_result, dict) else 0.0,
            obs_before=obs_before,
            hypotheses_before=probe_hyp_before,
            hypotheses_after=probe_hyp_after,
        )
        probe_error = self.loop._prediction_adjudicator.compare(probe_bundle, probe_outcome)
        self.loop._prediction_registry.record_prediction(probe_bundle)
        self.loop._prediction_registry.record_outcome(probe_outcome)
        self.loop._prediction_registry.record_error(probe_error)
        self.loop._prediction_registry.update_calibration(probe_bundle, probe_outcome, probe_error)
        for predictor in self.loop._prediction_engine.predictors:
            predictor.update(probe_bundle, probe_outcome, probe_error)
        self.loop._record_prediction_trace(probe_bundle, probe_outcome, probe_error)
        self.loop._apply_prediction_error_feedback(probe_error, bundle=probe_bundle, outcome=probe_outcome)

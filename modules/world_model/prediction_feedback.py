"""Prediction miss feedback runtime for world-model belief updates."""

from __future__ import annotations

from typing import Any, Dict


class PredictionMissFeedbackRuntime:
    """Small bridge from prediction miss signals to belief updater hooks."""

    def __init__(self, belief_updater: Any):
        self._belief_updater = belief_updater

    def record_prediction_miss(
        self,
        *,
        episode: int,
        tick: int,
        function_name: str,
        prediction_error: Dict[str, Any],
        reward: float,
        action_id: str,
    ) -> None:
        if self._belief_updater is None or not hasattr(self._belief_updater, 'on_prediction_miss'):
            return
        predicted_transition = {}
        observed_transition = {}
        if isinstance(prediction_error, dict):
            raw_predicted = prediction_error.get('predicted_transition', {})
            raw_observed = prediction_error.get('observed_transition', {})
            predicted_transition = raw_predicted if isinstance(raw_predicted, dict) else {}
            observed_transition = raw_observed if isinstance(raw_observed, dict) else {}
        payload = {
            'episode': int(episode),
            'tick': int(tick),
            'function_name': str(function_name or ''),
            'prediction_error': prediction_error if isinstance(prediction_error, dict) else {},
            'reward': float(reward or 0.0),
            'action_id': str(action_id or ''),
        }
        if isinstance(prediction_error, dict):
            payload['prediction_magnitude'] = float(
                prediction_error.get('total_error', prediction_error.get('magnitude', 0.0)) or 0.0
            )
            payload['predicted_phase'] = str(predicted_transition.get('to_phase', '') or '')
            payload['observed_phase'] = str(observed_transition.get('to_phase', '') or '')
            payload['state_shift_risk'] = float(predicted_transition.get('state_shift_risk', 0.0) or 0.0)
            payload['expected_information_gain'] = float(predicted_transition.get('expected_information_gain', 0.0) or 0.0)
        self._belief_updater.on_prediction_miss(payload)

"""Prediction feedback pipeline for post-action adjudication and reliability feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PredictionFeedbackInput:
    """Stable input contract for post-action prediction feedback."""

    episode: int
    tick: int
    action_id: str
    function_name: str
    result: Dict[str, Any]
    reward: float
    obs_before: Dict[str, Any]
    hypotheses_before: List[Any]
    hypotheses_after: List[Any]
    prediction_bundle: Optional[Any]
    prediction_enabled: bool


@dataclass
class PredictionFeedbackOutput:
    """Stable output contract for post-action prediction feedback."""

    pending_replan_patch: Optional[Dict[str, Any]]
    prediction_positive_miss_streak: int
    failure_mode: Optional[str]
    trace_entry: Optional[Dict[str, Any]]


class PredictionFeedbackPipeline:
    """Single pipeline: summary adjudication -> error feedback -> replan/reliability updates."""

    def __init__(self, *, trace_limit: int = 200):
        self._trace_limit = trace_limit

    def apply_after_action(
        self,
        payload: PredictionFeedbackInput,
        *,
        prediction_adjudicator: Any,
        prediction_registry: Any,
        prediction_engine: Any,
        reliability_tracker: Any,
        meta_control: Any,
        governance_log: List[Dict[str, Any]],
        prediction_trace_log: List[Dict[str, Any]],
        prediction_positive_miss_streak: int,
        world_model_feedback_port: Any = None,
    ) -> PredictionFeedbackOutput:
        failure_mode = self._classify_failure_mode(payload.result, payload.reward)
        if failure_mode is None:
            if hasattr(reliability_tracker, 'decay_failure_recency'):
                reliability_tracker.decay_failure_recency()
        else:
            reliability_tracker.record_failure_mode(payload.function_name or 'unknown', failure_mode)

        trace_entry: Optional[Dict[str, Any]] = None
        pending_replan_patch: Optional[Dict[str, Any]] = None

        if (
            payload.prediction_enabled
            and payload.prediction_bundle is not None
            and prediction_adjudicator is not None
            and prediction_registry is not None
            and prediction_engine is not None
        ):
            outcome = prediction_adjudicator.build_outcome_record(
                episode=payload.episode,
                tick=payload.tick,
                action_id=payload.action_id,
                function_name=payload.function_name,
                result=payload.result,
                reward=payload.reward,
                obs_before=payload.obs_before,
                hypotheses_before=payload.hypotheses_before,
                hypotheses_after=payload.hypotheses_after,
            )
            error = prediction_adjudicator.compare(payload.prediction_bundle, outcome)

            prediction_registry.record_prediction(payload.prediction_bundle)
            prediction_registry.record_outcome(outcome)
            prediction_registry.record_error(error)
            prediction_registry.update_calibration(payload.prediction_bundle, outcome, error)
            for predictor in prediction_engine.predictors:
                predictor.update(payload.prediction_bundle, outcome, error)

            trace_entry = {
                'episode': payload.episode,
                'tick': payload.tick,
                'prediction': payload.prediction_bundle.to_dict(),
                'outcome': outcome.to_dict(),
                'error': error.to_dict(),
            }
            prediction_trace_log.append(trace_entry)
            del prediction_trace_log[:-self._trace_limit]

            recent = prediction_registry.get_recent_errors(3)
            if len(recent) == 3 and all(float(e.total_error) > 0.6 for e in recent):
                if hasattr(meta_control, 'apply_runtime_hints'):
                    meta_control.apply_runtime_hints(retrieval_delta=0.1, reason='prediction_high_error')
                governance_log.append({'episode': payload.episode, 'tick': payload.tick, 'entry': 'prediction_high_error_retrieval_pressure'})

            pred_positive = str(payload.prediction_bundle.reward_sign.value) == 'positive'
            actual_negative = str(outcome.actual_reward_sign) == 'negative'
            prediction_positive_miss_streak = prediction_positive_miss_streak + 1 if (pred_positive and actual_negative) else 0
            if prediction_positive_miss_streak >= 2:
                pending_replan_patch = {'trigger': 'prediction_positive_miss', 'tick': payload.tick}
                governance_log.append({'episode': payload.episode, 'tick': payload.tick, 'entry': 'prediction_replan_hint'})

            trust = prediction_registry.get_predictor_trust()
            low_predictors = [k for k, v in trust.items() if v == 'low']
            if low_predictors:
                reliability_tracker.record_failure_mode('prediction', f"low_trust:{','.join(low_predictors[:2])}")

            if world_model_feedback_port is not None and hasattr(world_model_feedback_port, 'record_prediction_miss'):
                world_model_feedback_port.record_prediction_miss(
                    episode=payload.episode,
                    tick=payload.tick,
                    function_name=payload.function_name,
                    prediction_error=error.to_dict(),
                    reward=payload.reward,
                    action_id=payload.action_id,
                )

        return PredictionFeedbackOutput(
            pending_replan_patch=pending_replan_patch,
            prediction_positive_miss_streak=prediction_positive_miss_streak,
            failure_mode=failure_mode,
            trace_entry=trace_entry,
        )

    @staticmethod
    def _classify_failure_mode(result: Dict[str, Any], reward: float) -> Optional[str]:
        if reward >= 0 and bool(result.get('success', True)):
            return None
        err = result.get('error', {}) if isinstance(result.get('error'), dict) else {}
        err_type = str(err.get('type', '') or '').lower()
        if 'resource' in err_type or 'timeout' in err_type:
            return 'resource_failure'
        if 'bind' in err_type or 'schema' in err_type:
            return 'representation_failure'
        if 'world' in err_type or 'state' in err_type:
            return 'world_model_failure'
        if 'plan' in err_type:
            return 'planner_failure'
        if 'teacher' in err_type:
            return 'teacher_dependency_failure'
        return 'execution_failure'

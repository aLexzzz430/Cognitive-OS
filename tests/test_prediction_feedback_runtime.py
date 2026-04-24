from types import SimpleNamespace

from core.orchestration.prediction_feedback import (
    apply_prediction_error_feedback,
    prediction_bundle_to_dict,
    record_prediction_trace,
)


class _Bundle:
    def __init__(self, *, reward_sign="positive", function_name="move", action_id="a1"):
        self.reward_sign = SimpleNamespace(value=reward_sign)
        self.function_name = function_name
        self.action_id = action_id

    def to_dict(self):
        return {
            "reward_sign": self.reward_sign.value,
            "function_name": self.function_name,
            "action_id": self.action_id,
        }


class _Outcome:
    actual_reward_sign = "negative"
    actual_reward = -1.0

    def to_dict(self):
        return {
            "actual_reward_sign": self.actual_reward_sign,
            "actual_reward": self.actual_reward,
        }


class _Error:
    def __init__(self, total_error=0.75):
        self.total_error = total_error

    def to_dict(self):
        return {"total_error": self.total_error}


class _MetaControl:
    def __init__(self):
        self.hints = []

    def apply_runtime_hints(self, **kwargs):
        self.hints.append(kwargs)


class _PredictionMissFeedback:
    def __init__(self):
        self.records = []

    def record_prediction_miss(self, **kwargs):
        self.records.append(kwargs)


def test_prediction_bundle_to_dict_uses_bundle_contract():
    bundle = _Bundle(reward_sign="neutral", function_name="inspect", action_id="a2")

    assert prediction_bundle_to_dict(bundle) == {
        "reward_sign": "neutral",
        "function_name": "inspect",
        "action_id": "a2",
    }
    assert prediction_bundle_to_dict(object()) == {}


def test_record_prediction_trace_skips_missing_parts_and_keeps_recent_window():
    log = [{"old": index} for index in range(200)]

    assert record_prediction_trace(
        episode=1,
        tick=2,
        prediction_trace_log=log,
        bundle=None,
        outcome=_Outcome(),
        error=_Error(),
    ) is None
    assert len(log) == 200

    entry = record_prediction_trace(
        episode=1,
        tick=2,
        prediction_trace_log=log,
        bundle=_Bundle(),
        outcome=_Outcome(),
        error=_Error(),
    )

    assert len(log) == 200
    assert log[-1] == entry
    assert log[0] == {"old": 1}
    assert entry == {
        "episode": 1,
        "tick": 2,
        "prediction": {
            "reward_sign": "positive",
            "function_name": "move",
            "action_id": "a1",
        },
        "outcome": {"actual_reward_sign": "negative", "actual_reward": -1.0},
        "error": {"total_error": 0.75},
    }


def test_apply_prediction_error_feedback_records_high_error_and_world_model_miss():
    meta_control = _MetaControl()
    governance_log = []
    miss_feedback = _PredictionMissFeedback()

    feedback = apply_prediction_error_feedback(
        episode=3,
        tick=4,
        error=_Error(total_error=0.7),
        meta_control=meta_control,
        governance_log=governance_log,
        prediction_miss_feedback=miss_feedback,
        prediction_positive_miss_streak=0,
        bundle=_Bundle(function_name="click", action_id="a3"),
        outcome=_Outcome(),
    )

    assert feedback.applied_high_error_hint is True
    assert feedback.recorded_prediction_miss is True
    assert feedback.prediction_positive_miss_streak == 1
    assert feedback.pending_replan_patch is None
    assert meta_control.hints == [
        {"retrieval_delta": 0.1, "reason": "prediction_high_error"}
    ]
    assert governance_log == [
        {
            "episode": 3,
            "tick": 4,
            "entry": "prediction_high_error_retrieval_pressure",
        }
    ]
    assert miss_feedback.records == [
        {
            "episode": 3,
            "tick": 4,
            "function_name": "click",
            "prediction_error": {"total_error": 0.7},
            "reward": -1.0,
            "action_id": "a3",
        }
    ]


def test_apply_prediction_error_feedback_triggers_replan_after_second_positive_miss():
    governance_log = []

    feedback = apply_prediction_error_feedback(
        episode=5,
        tick=6,
        error=_Error(total_error=0.2),
        meta_control=_MetaControl(),
        governance_log=governance_log,
        prediction_miss_feedback=_PredictionMissFeedback(),
        prediction_positive_miss_streak=1,
        bundle=_Bundle(),
        outcome=_Outcome(),
    )

    assert feedback.prediction_positive_miss_streak == 2
    assert feedback.pending_replan_patch == {
        "trigger": "prediction_positive_miss",
        "tick": 6,
    }
    assert governance_log == [
        {"episode": 5, "tick": 6, "entry": "prediction_replan_hint"}
    ]


def test_apply_prediction_error_feedback_resets_streak_on_non_miss_and_noops_without_error():
    governance_log = []
    outcome = _Outcome()
    outcome.actual_reward_sign = "positive"
    outcome.actual_reward = 1.0

    feedback = apply_prediction_error_feedback(
        episode=7,
        tick=8,
        error=_Error(total_error=0.1),
        meta_control=_MetaControl(),
        governance_log=governance_log,
        prediction_miss_feedback=_PredictionMissFeedback(),
        prediction_positive_miss_streak=3,
        bundle=_Bundle(),
        outcome=outcome,
    )
    assert feedback.prediction_positive_miss_streak == 0
    assert feedback.pending_replan_patch is None
    assert governance_log == []

    none_feedback = apply_prediction_error_feedback(
        episode=7,
        tick=9,
        error=None,
        meta_control=_MetaControl(),
        governance_log=governance_log,
        prediction_miss_feedback=_PredictionMissFeedback(),
        prediction_positive_miss_streak=2,
        bundle=_Bundle(),
        outcome=outcome,
    )
    assert none_feedback.prediction_positive_miss_streak == 2
    assert none_feedback.pending_replan_patch is None
    assert none_feedback.applied_high_error_hint is False

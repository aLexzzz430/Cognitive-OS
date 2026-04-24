from types import SimpleNamespace

from core.orchestration.prediction_context_runtime import (
    build_recovery_prediction_context,
    build_self_model_prediction_summary,
)


class _SelfModelFacade:
    def __init__(self):
        self.calls = []

    def build_prediction_summary(self, **kwargs):
        self.calls.append(kwargs)
        return {"facade": "summary", "kwargs": kwargs}


class _ReliabilityTracker:
    def get_reliability_by_action_type(self):
        return {"move": 0.8, "inspect": 0.4}

    def get_recent_failure_profile(self, limit=8):
        return [{"mode": "timeout", "limit": limit}]

    def get_overall_recovery_success_rate(self):
        return 1.4


class _ResourceState:
    def budget_band(self):
        return "tight"

    def is_tight_budget(self):
        return True


def test_build_self_model_prediction_summary_delegates_to_facade_with_ablation_flag():
    facade = _SelfModelFacade()
    resource_state = _ResourceState()

    summary = build_self_model_prediction_summary(
        causal_ablation=SimpleNamespace(enable_high_level_self_model=False),
        self_model_facade=facade,
        reliability_tracker=_ReliabilityTracker(),
        resource_state=resource_state,
    )

    assert summary == {
        "facade": "summary",
        "kwargs": {
            "resource_state": resource_state,
            "include_high_level_state": False,
        },
    }
    assert facade.calls == [
        {
            "resource_state": resource_state,
            "include_high_level_state": False,
        }
    ]


def test_build_self_model_prediction_summary_fallback_collects_reliability_and_resources():
    summary = build_self_model_prediction_summary(
        causal_ablation=SimpleNamespace(enable_high_level_self_model=True),
        reliability_tracker=_ReliabilityTracker(),
        resource_state=_ResourceState(),
    )

    assert summary["high_level_state_included"] is True
    assert summary["reliability_by_function"] == {"move": 0.8, "inspect": 0.4}
    assert summary["global_reliability"] == 0.6000000000000001
    assert summary["recovery_availability"] == 1.0
    assert summary["resource_tightness"] == "tight"
    assert summary["budget_tight"] is True
    assert summary["recent_failure_modes"] == [{"mode": "timeout", "limit": 8}]
    assert summary["self_model_state"]["known_failure_modes"] == [
        {"mode": "timeout", "limit": 8}
    ]
    assert summary["self_model_state"]["continuity_confidence"] == 1.0
    assert summary["reliability_subscores"] == {
        "reliability_by_function": {"move": 0.8, "inspect": 0.4},
        "global_reliability": 0.6000000000000001,
        "recovery_availability": 1.0,
    }


def test_build_self_model_prediction_summary_can_emit_minimal_high_level_state():
    summary = build_self_model_prediction_summary(
        causal_ablation=SimpleNamespace(enable_high_level_self_model=False),
        reliability_tracker=None,
        resource_state=None,
    )

    assert summary["high_level_state_included"] is False
    assert summary["self_model_state"] == {
        "capabilities_by_domain": {},
        "capabilities_by_condition": {},
    }
    assert summary["global_reliability"] == 0.5
    assert summary["recovery_availability"] == 0.5
    assert summary["budget_tight"] is False


def test_build_recovery_prediction_context_uses_only_latest_dict_diagnosis():
    context = build_recovery_prediction_context(
        pending_recovery_probe={"probe": True},
        pending_replan=None,
        recovery_log=[{"older": True}, {"latest": True}],
    )

    assert context == {
        "pending_recovery_probe": True,
        "pending_replan": False,
        "last_recovery_diagnosis": {"latest": True},
    }

    non_dict_context = build_recovery_prediction_context(
        pending_recovery_probe=None,
        pending_replan={"replan": True},
        recovery_log=["not-a-dict"],
    )
    assert non_dict_context == {
        "pending_recovery_probe": False,
        "pending_replan": True,
        "last_recovery_diagnosis": {},
    }

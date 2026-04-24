from pytest import approx

from core.orchestration.retention_failure_runtime import (
    classify_retention_failure,
    latest_governance_entry_for_tick,
    latest_plan_lookahead_telemetry,
)


def _classification_defaults(**overrides):
    payload = {
        "reward": -1.0,
        "prediction_mismatch": 0.2,
        "task_family": "arc",
        "phase": "solve",
        "observation_mode": "grid",
        "resource_band": "normal",
        "action_name": "inspect",
        "lookahead": {},
        "governance_entry": {},
        "action_meta": {},
    }
    payload.update(overrides)
    return payload


def test_latest_governance_entry_for_tick_returns_recent_relevant_row():
    rows = [
        {"episode": 1, "tick": 2, "reason": "old"},
        {"episode": 2, "tick": 3},
        "ignore",
        {"episode": 2, "tick": 3, "entry": "prediction_high_error_retrieval_pressure"},
    ]

    assert latest_governance_entry_for_tick(rows, episode=2, tick=3) == {
        "episode": 2,
        "tick": 3,
        "entry": "prediction_high_error_retrieval_pressure",
    }
    assert latest_governance_entry_for_tick(rows, episode=9, tick=9) == {}


def test_latest_plan_lookahead_telemetry_prefers_current_payload_then_runtime_log():
    payload = {
        "episode": 4,
        "tick": 5,
        "telemetry": {"plan_lookahead": {"source": "payload"}},
    }
    log = [
        {
            "episode": 4,
            "tick": 5,
            "telemetry": {"plan_lookahead": {"source": "log"}},
        }
    ]

    assert latest_plan_lookahead_telemetry(
        last_planner_runtime_payload=payload,
        planner_runtime_log=log,
        episode=4,
        tick=5,
    ) == {"source": "payload"}
    assert latest_plan_lookahead_telemetry(
        last_planner_runtime_payload={"episode": 4, "tick": 4},
        planner_runtime_log=log,
        episode=4,
        tick=5,
    ) == {"source": "log"}
    assert latest_plan_lookahead_telemetry(
        last_planner_runtime_payload={},
        planner_runtime_log=[],
        episode=4,
        tick=5,
    ) == {}


def test_classify_retention_failure_detects_governance_overrule_misfire():
    result = classify_retention_failure(
        **_classification_defaults(
            reward=-0.1,
            prediction_mismatch=0.5,
            governance_entry={"reason": "counterfactual_oppose: selected_alt"},
            action_meta={
                "counterfactual_confidence": "high",
                "counterfactual_advantage": False,
            },
        )
    )

    assert result["failure_type"] == "governance_overrule_misfire"
    assert result["severity"] == approx(0.845)
    context = result["context"]
    assert context["strategy_mode_hint"] == "verify"
    assert context["verification_budget_hint"] == 2
    assert context["counterfactual_confidence"] == 0.9
    assert context["context_key"].endswith("|failure=governance_overrule_misfire")


def test_classify_retention_failure_detects_branch_persistence_collapse():
    result = classify_retention_failure(
        **_classification_defaults(
            reward=0.2,
            prediction_mismatch=0.1,
            lookahead={
                "forced_replan_events": ["wm_branch_persistence_replan"],
                "rollout_branch_persistence_ratio": 0.2,
                "rollout_branch_id": "branch-1",
                "rollout_final_phase": "disrupted",
            },
        )
    )

    assert result["failure_type"] == "branch_persistence_collapse"
    assert result["severity"] == approx(0.8)
    context = result["context"]
    assert context["branch_budget_hint"] == 2
    assert context["verification_budget_hint"] == 1
    assert context["forced_replan_events"] == ["wm_branch_persistence_replan"]
    assert context["rollout_branch_id"] == "branch-1"


def test_classify_retention_failure_detects_planner_target_switch():
    result = classify_retention_failure(
        **_classification_defaults(
            reward=0.0,
            prediction_mismatch=0.2,
            lookahead={
                "forced_replan_events": [
                    "wm_branch_salvage_replan",
                    "wm_rollout_value_replan",
                ],
                "rollout_branch_persistence_ratio": 0.75,
            },
        )
    )

    assert result["failure_type"] == "planner_target_switch"
    assert result["severity"] == approx(0.38)
    assert result["context"]["strategy_mode_hint"] == "recover"
    assert result["context"]["branch_budget_hint"] == 2


def test_classify_retention_failure_detects_prediction_drift_and_clean_case():
    drift = classify_retention_failure(
        **_classification_defaults(
            reward=-0.2,
            prediction_mismatch=0.36,
            lookahead={"rollout_branch_persistence_ratio": 0.9},
            governance_entry={
                "entry": "prediction_high_error_retrieval_pressure",
                "selected": "move",
            },
            action_name="fallback",
        )
    )

    assert drift["failure_type"] == "prediction_drift"
    assert drift["severity"] == approx(0.36)
    assert drift["context"]["selected_name"] == "move"
    assert drift["context"]["strategy_mode_hint"] == "verify"

    clean = classify_retention_failure(
        **_classification_defaults(
            reward=1.0,
            prediction_mismatch=0.1,
            lookahead={},
            governance_entry={},
        )
    )
    assert clean == {"failure_type": "", "severity": 0.0, "context": {}}

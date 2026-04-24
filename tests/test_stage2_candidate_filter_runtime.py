from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.runtime_stage_contracts import (
    Stage2PlanConstraintsInput,
    Stage2SelfModelSuppressionInput,
)
from core.orchestration import stage2_candidate_filter_runtime as runtime


class _PlanState:
    has_plan = True
    current_step = {"step_id": "step-1"}


class _ResourceState:
    def is_tight_budget(self):
        return True

    def budget_band(self):
        return "tight"


class _Loop:
    def __init__(self) -> None:
        self._plan_state = _PlanState()
        self._tick = 8
        self._episode = 3
        self._last_mechanism_runtime_view = {
            "mechanism_control_summary": {"mode": "guarded"},
        }
        self._pending_replan = None
        self._candidate_viability_log = []
        self._governance_log = []
        self._episode_trace = [
            {"reward": 1.0},
            {"reward": -1.0},
            {"reward": 0.0},
            {"reward": -0.5},
        ]
        self._resource_state = _ResourceState()
        self._viability_audit_log = []
        self._reliability_tracker = SimpleNamespace(name="reliability")
        self._prediction_enabled = True

    def _extract_action_function_name(self, action, default=""):
        return action.get("function", default) if isinstance(action, dict) else default

    def _infer_task_family(self, obs_before):
        return "test-family"

    def _extract_phase_hint(self, action):
        return "test-phase"


def test_stage2_plan_constraints_records_replan_and_viability(monkeypatch) -> None:
    loop = _Loop()
    captured = {}

    def fake_apply_plan_constraints(constraint_input):
        captured["input"] = constraint_input
        return SimpleNamespace(
            pending_replan_patch={"reason": "blocked"},
            viability_entry={"entry": "candidate_viability", "kept": 1},
            filtered_candidates=[{"function": "kept"}],
        )

    monkeypatch.setattr(runtime, "apply_plan_constraints", fake_apply_plan_constraints)

    filtered = runtime.run_stage2_plan_constraints(
        loop,
        Stage2PlanConstraintsInput(
            obs_before={"state": "before"},
            candidate_actions=[{"function": "kept"}, {"function": "drop"}],
        ),
    )

    assert filtered == [{"function": "kept"}]
    assert loop._pending_replan == {"reason": "blocked"}
    assert loop._candidate_viability_log == [{"entry": "candidate_viability", "kept": 1}]
    assert loop._governance_log == [{"entry": "candidate_viability", "kept": 1}]
    constraint_input = captured["input"]
    assert constraint_input.has_plan is True
    assert constraint_input.current_step == {"step_id": "step-1"}
    assert constraint_input.tick == 8
    assert constraint_input.episode == 3
    assert constraint_input.mechanism_control_summary == {"mode": "guarded"}


def test_stage2_self_model_suppression_enriches_snapshot_and_records_audit(monkeypatch) -> None:
    loop = _Loop()
    captured = {}

    def fake_apply_self_model_suppression(suppression_input, **ports):
        captured["input"] = suppression_input
        captured["ports"] = ports
        return SimpleNamespace(
            filtered_candidates=[{"function": "allowed"}],
            audit_records=[{"entry": "self_model_audit"}],
        )

    monkeypatch.setattr(runtime, "apply_self_model_suppression", fake_apply_self_model_suppression)

    filtered = runtime.run_stage2_self_model_suppression(
        loop,
        Stage2SelfModelSuppressionInput(
            candidate_actions=[{"function": "allowed"}],
            continuity_snapshot={"identity": "agent"},
            obs_before={
                "novel_api": {
                    "visible_functions": ["see"],
                    "discovered_functions": ["found"],
                },
                "perception": {"screen": "grid"},
                "world_model": {"objects": 2},
            },
        ),
    )

    assert filtered == [{"function": "allowed"}]
    assert loop._viability_audit_log == [{"entry": "self_model_audit"}]
    suppression_input = captured["input"]
    assert suppression_input.recent_failure_summary == {"recent_failures": 2}
    assert suppression_input.resource_state == {
        "is_tight_budget": True,
        "budget_band": "tight",
        "observation_mode": "unknown",
    }
    assert suppression_input.continuity_snapshot["identity"] == "agent"
    assert suppression_input.continuity_snapshot["visible_functions"] == ["see"]
    assert suppression_input.continuity_snapshot["discovered_functions"] == ["found"]
    assert suppression_input.continuity_snapshot["perception"] == {"screen": "grid"}
    assert suppression_input.continuity_snapshot["world_model_summary"] == {"objects": 2}
    assert captured["ports"]["extract_action_function_name"]({"function": "allowed"}) == "allowed"
    assert captured["ports"]["infer_task_family"]({}) == "test-family"
    assert captured["ports"]["extract_phase_hint"]({}) == "test-phase"
    assert captured["ports"]["reliability_tracker"].name == "reliability"


def test_prediction_fallback_materializes_bounded_prediction_metadata() -> None:
    loop = _Loop()
    candidate_actions = [
        {"function": "advantaged", "_candidate_meta": {"counterfactual_delta": 0.25, "counterfactual_advantage": True}},
        {"function": "existing", "_candidate_meta": {"prediction": {"source": "real_predictor"}}},
        {"function": "plain"},
        "not-an-action",
    ]

    runtime.materialize_stage2_prediction_fallback(loop, candidate_actions)

    advantaged_prediction = candidate_actions[0]["_candidate_meta"]["prediction"]
    assert advantaged_prediction["source"] == "counterfactual_fallback"
    assert advantaged_prediction["success"]["value"] == pytest.approx(0.68)
    assert advantaged_prediction["information_gain"]["value"] == 0.25
    assert advantaged_prediction["reward_sign"]["value"] == "positive"
    assert advantaged_prediction["overall_confidence"] == 0.58
    assert candidate_actions[1]["_candidate_meta"]["prediction"] == {"source": "real_predictor"}
    assert candidate_actions[2]["_candidate_meta"]["prediction"]["success"]["value"] == 0.5

    loop._prediction_enabled = False
    disabled_candidate = {"function": "disabled"}
    runtime.materialize_stage2_prediction_fallback(loop, [disabled_candidate])
    assert disabled_candidate == {"function": "disabled"}


def test_counterfactual_rank_candidates_orders_by_delta_advantage_and_confidence() -> None:
    candidate_actions = [
        {"function": "low", "_candidate_meta": {"counterfactual_delta": 0.1}},
        {
            "function": "high",
            "_candidate_meta": {
                "counterfactual_delta": 0.3,
                "counterfactual_advantage": True,
                "counterfactual_confidence": "high",
            },
        },
        {"function": "medium", "_candidate_meta": {"counterfactual_delta": 0.35}},
        {"function": "plain"},
    ]

    ranked = runtime.rank_counterfactual_candidates(candidate_actions)

    assert [row["function"] for row in ranked] == ["high", "medium", "low", "plain"]
    assert [row["_candidate_meta"]["counterfactual_rank"] for row in ranked] == [0, 1, 2, 3]
    assert runtime.rank_counterfactual_candidates([]) == []

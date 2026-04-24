from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.main_loop_components import CAPABILITY_PRIMARY_CONTROL
from core.orchestration.governance_state import GovernanceState
from core.orchestration.main_loop_ports import MainLoopContextProvider, MainLoopGovernancePorts


class _BeliefLedger:
    def get_active_beliefs(self):
        return [{"belief": "active"}]

    def get_established_beliefs(self):
        return [{"belief": "stable"}, {"belief": "confirmed"}]


class _PlanState:
    def get_plan_summary(self):
        return {"plan": "summary"}

    def get_intent_for_step(self):
        return "inspect"


class _MetaSnapshot:
    snapshot_id = "snap-1"
    inputs_hash = "hash-1"
    planner_bias = 0.1
    retrieval_aggressiveness = 0.2
    retrieval_pressure = 0.3
    probe_bias = 0.4
    verification_bias = 0.5
    risk_tolerance = 0.6
    recovery_bias = 0.7
    stability_bias = 0.8
    strategy_mode = "balanced"

    def to_policy_profile(self):
        return {"policy": "profile"}

    def to_representation_profile(self):
        return {"representation": "profile"}


class _MetaControl:
    def get_snapshot(self, episode, tick, context=None):
        assert episode == 2
        assert tick == 3
        assert context == {"ctx": True}
        return _MetaSnapshot()


class _StateManager:
    def get_state(self):
        return {
            "object_workspace": {"existing": True},
            "goal_stack": {"active": "goal"},
            "ignored": {"not": "included"},
        }


class _SharedStore:
    def get_by_object_type(self, object_type):
        assert object_type == "representation"
        return [
            {
                "object_id": "plain",
                "memory_layer": "semantic",
                "confidence": 0.9,
            },
            {
                "object_id": "mechanism",
                "memory_layer": "mechanism",
                "confidence": 0.1,
            },
        ]


def _loop():
    return SimpleNamespace(
        _belief_ledger=_BeliefLedger(),
        _episode_trace=[{"tick": 1}],
        _plan_state=_PlanState(),
        _meta_control=_MetaControl(),
        _hypotheses=SimpleNamespace(get_active=lambda: []),
        _hidden_state_tracker=SimpleNamespace(summary=lambda: {"hidden": False}),
        _learning_policy_snapshot={"learning": "snapshot"},
        _state_mgr=_StateManager(),
        _episode=2,
        _tick=3,
        _llm_initial_goal_hypothesis_candidates=[
            {"object_id": "goal-prior", "source_episode": 2, "summary": "goal"},
            {"object_id": "old-prior", "source_episode": 1, "summary": "old"},
        ],
        _llm_analyst_hypothesis_candidates=[
            {"object_id": "analyst", "source_tick": 3, "summary": "analyst"},
            {"object_id": "stale", "source_tick": 1, "summary": "stale"},
        ],
        _llm_world_model_proposal_candidates=[{"proposal": "candidate", "source_episode": 2}],
        _llm_world_model_validation_feedback=[{"validation": "ok"}],
        _llm_world_model_snapshot={"world": "snapshot"},
        _shared_store=_SharedStore(),
        _last_retrieval_decision={"should_query": True},
        _extract_action_function_name=lambda action, default="": action.get("function_name", default),
        _build_self_model_prediction_summary=lambda: {"self": "summary"},
    )


def test_main_loop_context_provider_exposes_loop_state_without_owning_runtime() -> None:
    provider = MainLoopContextProvider(_loop())

    assert provider.beliefs()["established_count"] == 2
    assert provider.episode_trace() == [{"tick": 1}]
    assert provider.plan_snapshot()["current_task"] == "inspect"
    assert provider.meta_control_snapshot(2, 3, context={"ctx": True})["policy_profile"] == {"policy": "profile"}
    assert provider.self_model_summary() == {"self": "summary"}
    assert provider.hidden_state_summary() == {"hidden": False}
    assert provider.learning_policy_snapshot() == {"learning": "snapshot"}
    assert provider.extraction_function_name({"function_name": "ACTION1"}) == "ACTION1"
    assert provider.retrieval_should_query() is True


def test_context_provider_workspace_and_representation_records_are_filtered() -> None:
    provider = MainLoopContextProvider(_loop())

    workspace = provider.workspace_state()
    object_workspace = workspace["object_workspace"]
    assert workspace["goal_stack"] == {"active": "goal"}
    assert [row["object_id"] for row in object_workspace["analyst_hypothesis_candidates"]] == [
        "goal-prior",
        "analyst",
    ]
    assert object_workspace["llm_proposal_candidates"] == [{"proposal": "candidate", "source_episode": 2}]
    assert object_workspace["llm_proposal_validation_feedback"] == [{"validation": "ok"}]
    assert object_workspace["llm_world_model_snapshot"] == {"world": "snapshot"}

    records = provider.cognitive_object_records("representation", limit=2)
    assert [row["object_id"] for row in records] == ["mechanism", "plain"]


def test_main_loop_governance_ports_bridge_logs_and_capability() -> None:
    loop = SimpleNamespace(
        _governance_log=[],
        _candidate_viability_log=[],
        _reliability_tracker=SimpleNamespace(
            build_global_failure_strategy=lambda short_term_pressure: {"pressure": short_term_pressure}
        ),
        _counterfactual=SimpleNamespace(
            simulate_action_difference=lambda state_slice, action_a, action_b, context: {
                "state": state_slice,
                "a": action_a,
                "b": action_b,
                "context": context,
            }
        ),
    )
    ports = MainLoopGovernancePorts(loop)

    ports.append_governance({"verdict": "passed"})
    ports.append_candidate_viability({"candidate": "ACTION1"})
    assert loop._governance_log == [{"verdict": "passed"}]
    assert loop._candidate_viability_log == [{"candidate": "ACTION1"}]
    assert ports.build_global_failure_strategy(short_term_pressure=0.25) == {"pressure": 0.25}
    assert ports.simulate_action_difference("s", {"a": 1}, {"b": 2}, context={"ctx": True}) == {
        "state": "s",
        "a": {"a": 1},
        "b": {"b": 2},
        "context": {"ctx": True},
    }
    assert ports.get_capability(
        "planner",
        GovernanceState(
            organ_failure_streaks={"planner": 0},
            organ_capability_flags={"planner": CAPABILITY_PRIMARY_CONTROL},
            organ_failure_threshold=2,
        ),
    ) == CAPABILITY_PRIMARY_CONTROL

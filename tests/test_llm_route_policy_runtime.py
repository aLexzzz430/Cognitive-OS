from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.llm_route_policy_runtime import (
    build_llm_route_context,
    route_capability_requirements,
)


def _loop(unified_context):
    return SimpleNamespace(
        _active_tick_context_frame=SimpleNamespace(unified_context=unified_context),
        _llm_route_feedback_summary=lambda: {
            "planner": {"score": 0.5, "samples": 2},
            "analyst": {"score": -0.25, "samples": 1},
        },
    )


def test_route_capability_requirements_are_stable_defaults() -> None:
    assert route_capability_requirements("general") == ["reasoning"]
    assert route_capability_requirements("probe") == ["verification", "reasoning"]
    assert route_capability_requirements("structured_answer") == ["structured_output", "reasoning"]
    assert route_capability_requirements("unknown") == ["reasoning"]


def test_build_llm_route_context_biases_for_pending_verification_and_resource_pressure() -> None:
    unified_context = SimpleNamespace(
        plan_state_summary={
            "goal_contract": {"goal_id": "goal-1"},
            "task_graph": {
                "nodes": [
                    {
                        "node_id": "task-1",
                        "provenance": {"step_id": "step-1"},
                        "verification_gate": {
                            "required": True,
                            "verifier_function": "check_goal",
                        },
                    }
                ]
            },
            "task_contract": {
                "verification_requirement": {
                    "verifier_authority": {
                        "required": True,
                        "verdict": "pending",
                        "verifier_function": "check_goal",
                    }
                }
            },
            "completion_gate": {
                "blocked_reasons": ["verification_incomplete"],
                "requires_verification": True,
            },
            "execution_authority": {},
        },
        uncertainty_vector={"overall": 0.4},
        posterior_summary={
            "execution_snapshot": {"transition_uncertainty": 0.66},
            "deliberation_snapshot": {"uncertainty_focus": 0.2},
        },
        compute_budget={"resource_pressure": "tight", "compute_budget": 0.3},
        world_shift_risk=0.1,
        retrieval_pressure=0.7,
    )

    context = build_llm_route_context(
        _loop(unified_context),
        "retrieval",
        capability_request="retrieval.query",
        capability_resolution={
            "capability": "retrieval.query",
            "route_name": "retrieval",
            "policy_source": "task_node",
            "required_capabilities": ["retrieval"],
        },
    )

    assert context["required_capabilities"] == ["retrieval", "verification"]
    assert context["uncertainty_level"] == 0.66
    assert context["verification_pressure"] == 0.72
    assert context["prefer_low_cost"] == 0.82
    assert context["prefer_low_latency"] == 0.72
    assert context["prefer_high_trust"] == 0.9
    assert context["route_feedback"] == {
        "planner": {"score": 0.5, "samples": 2},
        "analyst": {"score": -0.25, "samples": 1},
    }
    assert context["metadata"]["goal_id"] == "goal-1"
    assert context["metadata"]["active_task_id"] == "task-1"
    assert context["metadata"]["completion_gate_blocked_reasons"] == ["verification_incomplete"]
    assert context["metadata"]["verifier_verdict"] == "pending"
    assert context["metadata"]["verifier_required"] is True
    assert context["metadata"]["feedback_available_routes"] == ["analyst", "planner"]
    assert context["metadata"]["capability_request"] == "retrieval.query"
    assert context["metadata"]["capability_route_name"] == "retrieval"
    assert context["metadata"]["capability_policy_source"] == "task_node"


def test_build_llm_route_context_handles_empty_unified_context() -> None:
    context = build_llm_route_context(_loop(None), "structured_answer")

    assert context["required_capabilities"] == ["structured_output", "reasoning"]
    assert context["uncertainty_level"] == 0.0
    assert context["verification_pressure"] == 0.0
    assert context["prefer_structured_output"] == 1.0
    assert context["metadata"]["goal_id"] == ""

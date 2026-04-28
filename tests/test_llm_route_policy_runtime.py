from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.llm_route_policy_runtime import (
    build_llm_route_context,
    goal_task_binding_for_llm_policy,
    goal_task_capability_specs,
    goal_task_route_specs,
    resolved_llm_capability_specs,
    resolved_llm_route_specs,
    route_capability_requirements,
    runtime_budget_capability_specs,
    runtime_budget_route_specs,
)
from modules.llm.model_router import ModelRouter
from modules.llm.status_escalation import decide_status_escalation


def _loop(unified_context):
    return SimpleNamespace(
        _active_tick_context_frame=SimpleNamespace(unified_context=unified_context),
        _llm_route_feedback_summary=lambda: {
            "planner": {"score": 0.5, "samples": 2},
            "analyst": {"score": -0.25, "samples": 1},
        },
    )


class _BudgetWithResolvers:
    llm_route_specs = {
        "legacy": {
            "client_alias": "legacy-client",
            "budget": {"request_budget": 99},
        }
    }
    llm_capability_specs = {
        "legacy-capability": {
            "route_name": "legacy",
        }
    }

    def resolve_llm_route_specs(self):
        return {
            "runtime": {
                "client_alias": "runtime-client",
                "budget": {"request_budget": 2},
            }
        }

    def resolve_llm_capability_specs(self):
        return {
            "reasoning": {
                "route_name": "runtime",
                "required_capabilities": ["reasoning"],
            }
        }


class _GoalTaskRuntime:
    def __init__(self, binding):
        self.binding = binding
        self.refresh_calls = []

    def current_binding(self):
        return self.binding

    def refresh(self, **kwargs):
        self.refresh_calls.append(dict(kwargs))
        return self.binding


def _policy_loop(*, unified_context=None, binding=None):
    return SimpleNamespace(
        _runtime_budget=_BudgetWithResolvers(),
        _llm_route_specs={
            "manual": {
                "client_alias": "manual-client",
                "budget": {"request_budget": 1},
            },
        },
        _llm_capability_policies={
            "structured_output": {
                "route_name": "manual",
                "required_capabilities": ["structured_output"],
            }
        },
        _goal_task_runtime=_GoalTaskRuntime(binding) if binding is not None else None,
        _active_tick_context_frame=SimpleNamespace(unified_context=unified_context),
        _state_mgr=SimpleNamespace(name="state"),
        _episode=3,
        _tick=7,
    )


def _goal_task_binding():
    return SimpleNamespace(
        goal_contract=SimpleNamespace(
            goal_id="goal-1",
            planning=SimpleNamespace(
                llm_route_policies={
                    "planner": {
                        "client_alias": "goal-planner",
                        "budget": {"request_budget": 1},
                        "metadata": {"layer": "goal"},
                    }
                },
                llm_capability_policies={
                    "reasoning": {
                        "route_name": "planner",
                        "required_capabilities": ["reasoning"],
                        "metadata": {"layer": "goal"},
                    }
                },
            ),
        ),
        active_task=SimpleNamespace(
            goal_id="goal-1",
            node_id="task-1",
            llm_route_policies={
                "planner": {
                    "client_alias": "task-planner",
                    "budget": {"token_budget": 32},
                    "metadata": {"layer": "task"},
                }
            },
            llm_capability_policies={
                "reasoning": {
                    "route_name": "analyst",
                    "required_capabilities": ["verification"],
                    "metadata": {"layer": "task"},
                }
            },
        ),
    )


def test_route_capability_requirements_are_stable_defaults() -> None:
    assert route_capability_requirements("general") == ["reasoning"]
    assert route_capability_requirements("probe") == ["verification", "reasoning"]
    assert route_capability_requirements("structured_answer") == ["structured_output", "reasoning"]
    assert route_capability_requirements("unknown") == ["reasoning"]


def test_runtime_budget_policy_specs_prefer_resolvers_over_legacy_specs() -> None:
    loop = _policy_loop()

    assert runtime_budget_route_specs(loop) == {
        "runtime": {
            "client_alias": "runtime-client",
            "budget": {"request_budget": 2},
        }
    }
    assert runtime_budget_capability_specs(loop) == {
        "reasoning": {
            "route_name": "runtime",
            "required_capabilities": ["reasoning"],
        }
    }


def test_goal_task_policy_resolution_annotates_and_merges_goal_task_layers() -> None:
    binding = _goal_task_binding()
    loop = _policy_loop(unified_context=SimpleNamespace(), binding=binding)

    assert goal_task_binding_for_llm_policy(loop) is binding
    assert loop._goal_task_runtime.refresh_calls
    route_specs = goal_task_route_specs(loop)
    planner = route_specs["planner"]
    assert planner["client_alias"] == "task-planner"
    assert planner["budget"] == {"request_budget": 1, "token_budget": 32}
    assert planner["metadata"]["policy_source"] == "task_node"
    assert planner["metadata"]["goal_ref"] == "goal-1"
    assert planner["metadata"]["task_ref"] == "task-1"
    assert planner["metadata"]["layer"] == "task"

    capability_specs = goal_task_capability_specs(loop)
    reasoning = capability_specs["reasoning"]
    assert reasoning["route_name"] == "analyst"
    assert reasoning["required_capabilities"] == ["reasoning", "verification"]
    assert reasoning["metadata"]["policy_source"] == "task_node"
    assert reasoning["metadata"]["goal_ref"] == "goal-1"
    assert reasoning["metadata"]["task_ref"] == "task-1"
    assert reasoning["metadata"]["layer"] == "task"


def test_resolved_policy_specs_merge_runtime_goal_task_and_manual_layers() -> None:
    loop = _policy_loop(unified_context=None, binding=_goal_task_binding())

    route_specs = resolved_llm_route_specs(loop)
    assert sorted(route_specs) == ["manual", "planner", "runtime"]
    assert route_specs["runtime"]["client_alias"] == "runtime-client"
    assert route_specs["planner"]["metadata"]["policy_source"] == "task_node"
    assert route_specs["manual"]["budget"]["request_budget"] == 1

    capability_specs = resolved_llm_capability_specs(loop)
    assert sorted(capability_specs) == ["reasoning", "structured_output"]
    assert capability_specs["reasoning"]["route_name"] == "analyst"
    assert capability_specs["structured_output"]["route_name"] == "manual"


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


def test_status_monitor_escalates_route_context_from_degraded_status_file(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "DEGRADED",
                "failure_count": 3,
                "watchdog": {"ollama_connected": False, "ollama_latency_ms": 18000},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_FILE", str(status_path))
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_ENABLED", "1")
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_USE_LLM", "0")

    context = build_llm_route_context(_loop(SimpleNamespace()), "retrieval")

    monitor = context["metadata"]["status_monitor"]
    assert monitor["enabled"] is True
    assert monitor["should_escalate"] is True
    assert monitor["signals"]["failure_count"] == 3
    assert context["prefer_high_trust"] >= 0.96
    assert context["prefer_low_cost"] <= 0.12
    assert context["metadata"]["cloud_escalation_recommended"] is True
    assert context["metadata"]["cloud_route_bias"] > 0.0


def test_status_monitor_keeps_nominal_status_on_fast_path(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": "OK", "failure_count": 0}), encoding="utf-8")
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_FILE", str(status_path))
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_ENABLED", "1")
    monkeypatch.setenv("CONOS_LLM_STATUS_MONITOR_USE_LLM", "0")

    context = build_llm_route_context(_loop(SimpleNamespace()), "retrieval")

    assert context["metadata"]["status_monitor"]["should_escalate"] is False
    assert context["metadata"]["cloud_escalation_recommended"] is False
    assert context["metadata"]["cloud_route_bias"] == 0.0


def test_status_monitor_can_use_local_model_decision(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": "OK", "failure_count": 0}), encoding="utf-8")

    class FakeClient:
        def complete_json(self, *args, **kwargs):
            return {
                "should_escalate": True,
                "confidence": 0.81,
                "reason": "local_monitor_detected_risk",
            }

    monkeypatch.setattr("modules.llm.factory.build_llm_client", lambda **kwargs: FakeClient())
    decision = decide_status_escalation(
        route_name="root_cause",
        route_context={"uncertainty_level": 0.2, "verification_pressure": 0.1, "metadata": {}},
        environ={
            "CONOS_LLM_STATUS_MONITOR_FILE": str(status_path),
            "CONOS_LLM_STATUS_MONITOR_ENABLED": "1",
            "CONOS_LLM_STATUS_MONITOR_MODEL": "qwen-small",
            "CONOS_LLM_STATUS_MONITOR_USE_LLM": "1",
        },
    )

    assert decision.source == "local_model"
    assert decision.should_escalate is True
    assert decision.cloud_route_bias == 0.81
    assert decision.local_model == "qwen-small"


def test_model_router_cloud_route_bias_prefers_cloud_candidate() -> None:
    router = ModelRouter(
        route_specs={
            "local_small": {
                "served_routes": ["general"],
                "client_alias": "local",
                "provider": "ollama",
                "capability_profile": {
                    "capabilities": ["reasoning"],
                    "trust_score": 0.55,
                    "cost_efficiency": 0.95,
                    "latency_efficiency": 0.9,
                    "uncertainty_tolerance": 0.5,
                    "verification_strength": 0.45,
                },
            },
            "cloud_big": {
                "served_routes": ["general"],
                "client_alias": "cloud",
                "provider": "openai",
                "capability_profile": {
                    "capabilities": ["reasoning"],
                    "trust_score": 0.98,
                    "cost_efficiency": 0.1,
                    "latency_efficiency": 0.2,
                    "uncertainty_tolerance": 0.95,
                    "verification_strength": 0.95,
                },
            },
        }
    )
    router.register_client("local", object())
    router.register_client("cloud", object())

    fast_path = router.decide(
        "general",
        context={
            "required_capabilities": ["reasoning"],
            "prefer_low_cost": 1.0,
            "metadata": {},
        },
    )
    escalated = router.decide(
        "general",
        context={
            "required_capabilities": ["reasoning"],
            "prefer_high_trust": 0.96,
            "prefer_low_cost": 0.0,
            "metadata": {"cloud_route_bias": 1.0},
        },
    )

    assert fast_path.route_name == "local_small"
    assert escalated.route_name == "cloud_big"
    assert escalated.metadata["score_breakdown"]["cloud_bonus"] > 0.0

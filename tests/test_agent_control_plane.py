from __future__ import annotations

import json
from pathlib import Path

from modules.control_plane import (
    AGENT_CONTROL_PLANE_VERSION,
    AgentControlPlane,
    AgentControlRequest,
    agent_specs_from_model_route_policies,
    load_agent_registry,
    write_agent_registry,
)
from modules.llm.cli import main as llm_cli_main
from modules.llm.model_profile import MODEL_PROFILE_VERSION, route_policies_from_profiles, write_model_route_policies


def _route_policies() -> dict:
    return route_policies_from_profiles(
        [
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "ollama",
                "base_url": "http://fake-ollama",
                "model": "fast-small",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.45,
                    "planning": 0.35,
                    "structured_output": 0.35,
                    "verification": 0.3,
                    "speed": 0.95,
                    "instruction_following": 0.9,
                    "retrieval": 0.9,
                    "coding": 0.35,
                },
            },
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "ollama",
                "base_url": "http://fake-ollama",
                "model": "json-strong",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.8,
                    "planning": 0.7,
                    "structured_output": 0.95,
                    "verification": 0.9,
                    "speed": 0.5,
                    "instruction_following": 0.8,
                    "retrieval": 0.5,
                    "coding": 0.55,
                },
            },
        ],
        base_url="http://fake-ollama",
    )


def test_control_plane_selects_profile_backed_model_and_audits_decision() -> None:
    agents = agent_specs_from_model_route_policies(_route_policies())
    decision = AgentControlPlane(agents).decide(
        AgentControlRequest(
            task_type="answer_with_schema",
            route_name="structured_answer",
            capability_request="structured.output",
            required_capabilities=["structured_output"],
            permissions_required=["generate_text", "structured_output"],
            risk_level="low",
            prefer_high_trust=0.8,
        )
    )

    assert decision.status == "SELECTED"
    assert decision.selected_agent["model"] == "json-strong"
    assert decision.audit_event["schema_version"] == AGENT_CONTROL_PLANE_VERSION
    assert decision.audit_event["selected_agent_id"] == decision.selected_agent_id
    assert decision.audit_event["permissions_required"] == ["generate_text", "structured_output"]


def test_control_plane_can_select_non_model_coding_agent() -> None:
    agents = agent_specs_from_model_route_policies(_route_policies())
    agents.append(
        {
            "agent_id": "codex-local",
            "display_name": "Codex local coding agent",
            "agent_kind": "coding_agent",
            "provider": "codex",
            "served_routes": ["coding", "recovery"],
            "capabilities": ["coding", "tool_use", "reasoning", "verification"],
            "allowed_permissions": ["read_files", "propose_patch", "run_tests"],
            "approval_required_permissions": ["write_source"],
            "trust_score": 0.88,
            "cost_efficiency": 0.65,
            "latency_efficiency": 0.55,
            "max_risk_level": "medium",
        }
    )

    decision = AgentControlPlane(agents).decide(
        {
            "task_type": "small_code_improvement",
            "route": "coding",
            "required_capabilities": ["coding", "tool_use"],
            "permissions": ["read_files", "propose_patch", "run_tests"],
            "risk_level": "medium",
            "prefer_high_trust": 1.0,
        }
    )

    assert decision.status == "SELECTED"
    assert decision.selected_agent_id == "codex-local"
    assert decision.selected_agent["agent_kind"] == "coding_agent"
    assert any(row["agent_id"] == "codex-local" and row["eligible"] for row in decision.candidates)


def test_control_plane_marks_sensitive_permission_waiting_approval() -> None:
    plane = AgentControlPlane(
        [
            {
                "agent_id": "codex-local",
                "agent_kind": "coding_agent",
                "provider": "codex",
                "served_routes": ["coding"],
                "capabilities": ["coding", "tool_use"],
                "allowed_permissions": ["read_files", "propose_patch"],
                "approval_required_permissions": ["write_source"],
                "max_risk_level": "medium",
                "trust_score": 0.9,
            }
        ]
    )

    decision = plane.decide(
        {
            "task_type": "apply_patch_after_review",
            "route": "coding",
            "required_capabilities": ["coding"],
            "permissions": ["read_files", "write_source"],
            "risk_level": "medium",
        }
    )

    assert decision.status == "WAITING_APPROVAL"
    assert decision.approval_required is True
    assert decision.approval_permissions == ["write_source"]
    assert decision.audit_event["approval_required"] is True


def test_agent_registry_round_trips_and_cli_control_plane(tmp_path: Path, capsys) -> None:
    policies = _route_policies()
    policy_path = tmp_path / "route_policies.json"
    write_model_route_policies(policies, policy_path)
    registry_path = tmp_path / "agents.json"
    write_agent_registry(
        [
            {
                "agent_id": "ci-runner",
                "agent_kind": "ci",
                "provider": "ci",
                "served_routes": ["validation"],
                "capabilities": ["verification", "testing"],
                "allowed_permissions": ["run_tests"],
                "max_risk_level": "medium",
                "trust_score": 0.75,
            }
        ],
        registry_path,
    )

    loaded = load_agent_registry(registry_path)
    assert loaded[0].agent_id == "ci-runner"

    assert (
        llm_cli_main(
            [
                "control-plane",
                "--route-policy-file",
                str(policy_path),
                "--agent-registry",
                str(registry_path),
                "--task-type",
                "validation",
                "--route",
                "validation",
                "--required-capability",
                "testing",
                "--permission",
                "run_tests",
                "--risk-level",
                "medium",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == AGENT_CONTROL_PLANE_VERSION
    assert payload["agent_count"] == 3
    assert payload["decision"]["selected_agent_id"] == "ci-runner"

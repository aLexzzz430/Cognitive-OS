from __future__ import annotations

from modules.control_plane import (
    ACTION_GOVERNANCE_VERSION,
    ActionGovernancePolicy,
    ActionGovernanceState,
    derive_action_governance_request,
    evaluate_action_governance,
    record_action_governance_result,
)


def test_action_governance_blocks_mirror_write_without_evidence() -> None:
    request = derive_action_governance_request(
        "apply_patch",
        {"patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n"},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "BLOCKED"
    assert decision.blocked_reason == "evidence_refs_required_before_mirror_write"
    assert decision.audit_event["schema_version"] == ACTION_GOVERNANCE_VERSION


def test_action_governance_allows_mirror_write_with_evidence() -> None:
    request = derive_action_governance_request(
        "edit_replace_range",
        {"path": "app.py", "evidence_refs": ["file:app.py:1-20"]},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "ALLOWED"
    assert decision.effective_permissions == ["propose_patch", "edit_mirror"]


def test_action_governance_requires_validation_for_code_sync_but_not_docs() -> None:
    code_request = derive_action_governance_request(
        "mirror_apply",
        {"plan_id": "plan-1"},
        agent_id="agent-a",
        metadata={
            "changed_paths": ["app.py"],
            "approval_status": "machine_approved",
            "approved_by": "machine",
        },
    )
    docs_request = derive_action_governance_request(
        "mirror_apply",
        {"plan_id": "plan-2"},
        agent_id="agent-a",
        metadata={
            "changed_paths": ["README.md"],
            "approval_status": "machine_approved",
            "approved_by": "machine",
        },
    )

    blocked = evaluate_action_governance(code_request, ActionGovernanceState())
    allowed_docs = evaluate_action_governance(docs_request, ActionGovernanceState())
    allowed_code = evaluate_action_governance(
        code_request,
        ActionGovernanceState(
            validation_runs=[{"run_ref": "run_1", "success": True, "returncode": 0}],
            passing_tests=1,
        ),
    )

    assert blocked.status == "BLOCKED"
    assert blocked.blocked_reason == "passing_validation_required_before_source_sync"
    assert allowed_docs.status == "ALLOWED"
    assert allowed_code.status == "ALLOWED"


def test_action_governance_waits_for_source_sync_approval() -> None:
    request = derive_action_governance_request(
        "mirror_apply",
        {"plan_id": "plan-1"},
        agent_id="agent-a",
        metadata={"changed_paths": ["README.md"]},
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "WAITING_APPROVAL"
    assert decision.required_approval is True
    assert decision.blocked_reason == "source_sync_requires_approved_plan"


def test_action_governance_downgrades_agent_after_repeated_failures() -> None:
    request = derive_action_governance_request(
        "apply_patch",
        {"patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n", "evidence_refs": ["file:app.py:1-5"]},
        agent_id="agent-a",
    )

    state = ActionGovernanceState(evidence_refs=["file:app.py:1-5"])
    state = record_action_governance_result(state, request, success=False, failure_reason="context mismatch")
    state = record_action_governance_result(state, request, success=False, failure_reason="context mismatch")
    decision = evaluate_action_governance(request, state)

    assert state.failure_count_by_agent["agent-a"] == 2
    assert "agent-a" in state.downgraded_agents
    assert decision.status == "DOWNGRADED"


def test_action_governance_blocks_relative_path_escape(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    request = derive_action_governance_request(
        "edit_replace_range",
        {"path": "../outside.py", "evidence_refs": ["file:app.py:1-2"]},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(
        request,
        ActionGovernanceState(
            source_root=str(source_root),
            allowed_roots=[str(source_root)],
            evidence_refs=["file:app.py:1-2"],
        ),
    )

    assert decision.status == "BLOCKED"
    assert decision.blocked_reason.startswith("path_outside_allowed_roots:")


def test_action_governance_blocks_inline_credentials() -> None:
    request = derive_action_governance_request(
        "internet_fetch",
        {"url": "https://example.com/data.json", "headers": {"Authorization": "Bearer secret"}},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "BLOCKED"
    assert decision.blocked_reason.startswith("inline_credentials_not_allowed:")


def test_action_governance_requires_bounded_generated_exec() -> None:
    request = derive_action_governance_request(
        "mirror_exec",
        {"command": ["python", "-m", "pytest"]},
        agent_id="agent-a",
        metadata={"generated_command": True, "purpose": "test", "timeout_seconds_present": True},
    )
    allowed = derive_action_governance_request(
        "mirror_exec",
        {"command": ["python", "-m", "pytest"], "target": "."},
        agent_id="agent-a",
        metadata={
            "generated_command": True,
            "purpose": "test",
            "timeout_seconds_present": True,
            "bounded_target_present": True,
        },
    )

    blocked = evaluate_action_governance(request, ActionGovernanceState())
    accepted = evaluate_action_governance(allowed, ActionGovernanceState())

    assert blocked.status == "BLOCKED"
    assert blocked.blocked_reason == "bounded_exec_requires_target"
    assert accepted.status == "ALLOWED"


def test_action_governance_requires_approval_for_private_network_access() -> None:
    request = derive_action_governance_request(
        "internet_fetch",
        {"url": "http://192.168.0.2/data.json"},
        agent_id="agent-a",
        metadata={"private_networks_allowed": True},
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())
    relaxed = evaluate_action_governance(
        request,
        ActionGovernanceState(),
        ActionGovernancePolicy(require_approval_for_private_network=False),
    )

    assert decision.status == "WAITING_APPROVAL"
    assert decision.required_approval is True
    assert decision.blocked_reason == "private_network_access_requires_approval"
    assert relaxed.status == "ALLOWED"

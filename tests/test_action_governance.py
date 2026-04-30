from __future__ import annotations

from modules.control_plane import (
    ACTION_GOVERNANCE_VERSION,
    ActionGovernancePolicy,
    ActionGovernanceState,
    canonical_capability_layers,
    derive_action_governance_request,
    evaluate_action_governance,
    record_action_governance_result,
    side_effect_class_for_permissions,
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
    assert decision.audit_event["capability_layers"] == ["propose_patch"]
    assert decision.audit_event["side_effect"] is True
    assert decision.audit_event["side_effect_class"] == "mirror_patch"


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
            "sync_gate_mode": "patch_gate_added_or_modified_files_only",
            "apply_method": "unified_text_patch",
            "requires_source_hash_match": True,
            "creates_rollback_checkpoint": True,
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
            "sync_gate_mode": "patch_gate_added_or_modified_files_only",
            "apply_method": "unified_text_patch",
            "requires_source_hash_match": True,
            "creates_rollback_checkpoint": True,
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
        metadata={
            "changed_paths": ["README.md"],
            "sync_gate_mode": "patch_gate_added_or_modified_files_only",
            "apply_method": "unified_text_patch",
            "requires_source_hash_match": True,
            "creates_rollback_checkpoint": True,
        },
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "WAITING_APPROVAL"
    assert decision.required_approval is True
    assert decision.blocked_reason == "source_sync_requires_approved_plan"
    assert decision.audit_event["approval_state"] == "waiting_approval"
    assert decision.audit_event["capability_layers"] == ["sync_back"]


def test_action_governance_rejects_source_sync_without_patch_gate() -> None:
    request = derive_action_governance_request(
        "mirror_apply",
        {"plan_id": "plan-1"},
        agent_id="agent-a",
        metadata={
            "changed_paths": ["README.md"],
            "approval_status": "machine_approved",
            "approved_by": "machine",
            "sync_gate_mode": "copy_back",
            "apply_method": "copy_file",
        },
    )

    decision = evaluate_action_governance(request, ActionGovernanceState())

    assert decision.status == "BLOCKED"
    assert decision.blocked_reason == "source_sync_requires_patch_gate"


def test_action_governance_canonical_permission_layers() -> None:
    request = derive_action_governance_request(
        "internet_fetch",
        {"url": "https://example.com", "headers": {"Authorization": "Bearer redacted"}},
        agent_id="agent-a",
    )

    assert canonical_capability_layers(request.permissions_required) == ["network", "credential"]
    assert side_effect_class_for_permissions(request.permissions_required) == "network_access"


def test_action_governance_blocks_capability_layer_not_allowed() -> None:
    request = derive_action_governance_request(
        "run_test",
        {"target": "tests", "timeout_seconds": 30},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(
        request,
        ActionGovernanceState(),
        ActionGovernancePolicy(allowed_capability_layers=("read", "propose_patch")),
    )

    assert decision.status == "BLOCKED"
    assert decision.blocked_reason == "capability_layer_not_allowed:execute"
    assert decision.audit_event["allowed_capability_layers"] == ["read", "propose_patch"]


def test_action_governance_waits_for_generic_capability_layer_approval() -> None:
    request = derive_action_governance_request(
        "internet_fetch",
        {"url": "https://example.com/data.json"},
        agent_id="agent-a",
    )
    policy = ActionGovernancePolicy(approval_required_capability_layers=("network",))

    waiting = evaluate_action_governance(request, ActionGovernanceState(), policy)
    approved = evaluate_action_governance(
        request,
        ActionGovernanceState(approved_capability_layers=["network"]),
        policy,
    )

    assert waiting.status == "WAITING_APPROVAL"
    assert waiting.blocked_reason == "capability_layer_requires_approval:network"
    assert waiting.audit_event["approval_required_capability_layers"] == ["network"]
    assert approved.status == "ALLOWED"


def test_action_governance_treats_credential_lease_as_credential_capability() -> None:
    request = derive_action_governance_request(
        "mirror_exec",
        {},
        agent_id="agent-a",
        metadata={"credential_lease_ids": ["lease_openai_test"]},
    )
    policy = ActionGovernancePolicy(approval_required_capability_layers=("credential",))

    waiting = evaluate_action_governance(request, ActionGovernanceState(), policy)
    approved = evaluate_action_governance(
        request,
        ActionGovernanceState(approved_capability_layers=["credential"]),
        policy,
    )

    assert canonical_capability_layers(request.permissions_required) == ["execute", "credential"]
    assert waiting.status == "WAITING_APPROVAL"
    assert waiting.blocked_reason == "capability_layer_requires_approval:credential"
    assert approved.status == "ALLOWED"


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


def test_action_governance_inline_credentials_block_before_capability_approval() -> None:
    request = derive_action_governance_request(
        "internet_fetch",
        {"url": "https://example.com/data.json", "headers": {"Authorization": "Bearer secret"}},
        agent_id="agent-a",
    )

    decision = evaluate_action_governance(
        request,
        ActionGovernanceState(),
        ActionGovernancePolicy(approval_required_capability_layers=("credential", "network")),
    )

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


def test_runtime_mode_policy_limits_action_permissions() -> None:
    creating_exec = derive_action_governance_request(
        "run_test",
        {"target": "tests", "timeout_seconds": 30},
        agent_id="agent-a",
        metadata={"runtime_mode": "CREATING"},
    )
    creating_patch = derive_action_governance_request(
        "apply_patch",
        {"patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n", "evidence_refs": ["file:app.py:1-5"]},
        agent_id="agent-a",
        metadata={"runtime_mode": "CREATING"},
    )
    waiting_patch = derive_action_governance_request(
        "apply_patch",
        {"patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n", "evidence_refs": ["file:app.py:1-5"]},
        agent_id="agent-a",
        metadata={"runtime_mode": "WAITING_HUMAN"},
    )

    blocked_exec = evaluate_action_governance(creating_exec, ActionGovernanceState())
    allowed_patch = evaluate_action_governance(
        creating_patch,
        ActionGovernanceState(evidence_refs=["file:app.py:1-5"]),
    )
    waiting = evaluate_action_governance(
        waiting_patch,
        ActionGovernanceState(evidence_refs=["file:app.py:1-5"]),
    )

    assert blocked_exec.status == "BLOCKED"
    assert blocked_exec.blocked_reason == "capability_layer_not_allowed:execute"
    assert blocked_exec.audit_event["runtime_mode"] == "CREATING"
    assert allowed_patch.status == "ALLOWED"
    assert waiting.status == "BLOCKED"
    assert waiting.blocked_reason == "capability_layer_not_allowed:propose_patch"

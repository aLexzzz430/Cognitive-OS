from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from modules.control_plane.action_governance import ActionGovernancePolicy, coerce_action_governance_policy


PRODUCT_DEPLOYMENT_GATE_VERSION = "conos.product_deployment_gate/v0.1"


def product_deployment_gate_report(
    *,
    checks: Sequence[Mapping[str, Any]],
    vm_setup_plan: Mapping[str, Any] | None = None,
    include_vm: bool = True,
    governance_policy: Mapping[str, Any] | ActionGovernancePolicy | None = None,
) -> Dict[str, Any]:
    """Summarize whether this install is safe to call deployable.

    The normal development runtime can still run without a VM, but product AGI
    deployment must have a real default side-effect boundary. This gate keeps
    that distinction explicit instead of letting optional warnings masquerade as
    production readiness.
    """

    policy = coerce_action_governance_policy(governance_policy)
    vm_plan = dict(vm_setup_plan or {})
    check_rows = [dict(row) for row in list(checks or []) if isinstance(row, Mapping)]
    blockers: list[Dict[str, Any]] = []

    for row in check_rows:
        if bool(row.get("required", False)) and not bool(row.get("ok", False)):
            blockers.append(
                {
                    "check": str(row.get("name") or ""),
                    "reason": str(row.get("detail") or ""),
                    "kind": "required_install_check",
                }
            )

    vm_required = bool(include_vm)
    vm_ready = bool(vm_plan.get("safe_to_run_tasks", False))
    real_vm_boundary = bool(vm_plan.get("real_vm_boundary", False) or vm_plan.get("vm_ready", False))
    if vm_required and not vm_ready:
        blockers.append(
            {
                "check": "vm_default_boundary",
                "reason": str(vm_plan.get("operator_summary") or vm_plan.get("status") or "VM default boundary is not ready"),
                "kind": "default_side_effect_boundary",
            }
        )
    if vm_required and vm_ready and not real_vm_boundary:
        blockers.append(
            {
                "check": "vm_real_boundary",
                "reason": "VM setup plan is ready but did not assert a real VM boundary",
                "kind": "default_side_effect_boundary",
            }
        )

    policy_requirements = [
        (
            "side_effect_audit_event",
            bool(policy.require_side_effect_audit_event),
            "every side-effect action must write an audit event",
        ),
        (
            "patch_gate_source_sync",
            bool(policy.require_patch_gate_for_source_sync),
            "source sync must use patch gate, not copy-back",
        ),
        (
            "source_sync_approval",
            bool(policy.require_approval_for_source_sync),
            "sync-back must require approval",
        ),
        (
            "test_before_source_sync",
            bool(policy.require_test_before_source_sync),
            "source sync must require verifier evidence",
        ),
        (
            "credential_boundary",
            str(policy.credential_boundary_policy) == "vm_guest_isolated_explicit_env_only",
            "credentials must stay isolated from host passthrough",
        ),
        (
            "network_policy",
            str(policy.network_default_policy) == "deny_private_by_default",
            "network access must be policy controlled by default",
        ),
    ]
    policy_checks = []
    for name, ok, description in policy_requirements:
        policy_checks.append({"name": name, "ok": bool(ok), "description": description})
        if not ok:
            blockers.append({"check": name, "reason": description, "kind": "governance_policy"})

    status = "PASSED" if not blockers else "BLOCKED"
    return {
        "schema_version": PRODUCT_DEPLOYMENT_GATE_VERSION,
        "status": status,
        "deployable": status == "PASSED",
        "include_vm": bool(include_vm),
        "default_side_effect_boundary": {
            "required": vm_required,
            "safe_to_run_tasks": vm_ready,
            "real_vm_boundary": real_vm_boundary,
            "host_side_effect_execution_allowed_by_default": False,
            "vm_setup_status": str(vm_plan.get("status") or ""),
        },
        "governance_policy": {
            "allowed_capability_layers": list(policy.allowed_capability_layers),
            "approval_required_capability_layers": list(policy.approval_required_capability_layers),
            "network_default_policy": str(policy.network_default_policy),
            "credential_boundary_policy": str(policy.credential_boundary_policy),
            "sync_back_policy": str(policy.sync_back_policy),
            "approval_state_model": str(policy.approval_state_model),
        },
        "policy_checks": policy_checks,
        "blockers": blockers,
        "operator_summary": (
            "产品部署门已通过：默认 side-effect 边界、patch gate、审批、凭证和网络策略均满足。"
            if status == "PASSED"
            else "产品部署门未通过：不能宣称可部署 AGI，先处理 blockers。"
        ),
    }

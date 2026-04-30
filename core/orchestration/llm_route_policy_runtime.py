from __future__ import annotations

from typing import Any, Dict, Optional

from core.orchestration.goal_task_control import resolve_plan_summary_active_task
from core.orchestration.verifier_runtime import build_verifier_runtime
from core.runtime_budget import (
    merge_llm_capability_specs,
    merge_llm_route_specs,
    resolve_llm_capability_policies,
    resolve_llm_route_policies,
)
from core.runtime.runtime_modes import infer_route_runtime_mode
from modules.llm.status_escalation import apply_status_escalation_to_route_context


DEFAULT_ROUTE_CAPABILITY_REQUIREMENTS: Dict[str, list[str]] = {
    "general": ["reasoning"],
    "deliberation": ["reasoning", "planning"],
    "planning": ["reasoning", "planning"],
    "planner": ["reasoning", "planning"],
    "plan_generation": ["reasoning", "planning"],
    "retrieval": ["retrieval", "grounding"],
    "hypothesis": ["reasoning", "uncertainty"],
    "probe": ["verification", "reasoning"],
    "skill": ["instruction_following"],
    "recovery": ["recovery", "reasoning"],
    "root_cause": ["reasoning", "verification", "uncertainty"],
    "test_failure": ["reasoning", "verification"],
    "patch_proposal": ["reasoning", "coding", "verification"],
    "final_audit": ["analysis", "verification", "reasoning"],
    "representation": ["representation"],
    "structured_answer": ["structured_output", "reasoning"],
    "shadow": ["analysis"],
    "analyst": ["analysis", "verification"],
}


def route_capability_requirements(route_name: str) -> list[str]:
    route_key = str(route_name or "general").strip() or "general"
    return list(DEFAULT_ROUTE_CAPABILITY_REQUIREMENTS.get(route_key, ["reasoning"]))


def runtime_budget_route_specs(loop: Any) -> Dict[str, Dict[str, Any]]:
    budget = getattr(loop, "_runtime_budget", None)
    resolver = getattr(budget, "resolve_llm_route_specs", None)
    if callable(resolver):
        return dict(resolver() or {})
    legacy_specs = getattr(budget, "llm_route_specs", {}) if budget is not None else {}
    if isinstance(legacy_specs, dict):
        return dict(legacy_specs)
    return {}


def runtime_budget_capability_specs(loop: Any) -> Dict[str, Dict[str, Any]]:
    budget = getattr(loop, "_runtime_budget", None)
    resolver = getattr(budget, "resolve_llm_capability_specs", None)
    if callable(resolver):
        return dict(resolver() or {})
    legacy_specs = getattr(budget, "llm_capability_specs", {}) if budget is not None else {}
    if isinstance(legacy_specs, dict):
        return resolve_llm_capability_policies(legacy_specs)
    legacy_policies = getattr(budget, "llm_capability_policies", {}) if budget is not None else {}
    return resolve_llm_capability_policies(legacy_policies)


def goal_task_binding_for_llm_policy(loop: Any) -> Any:
    runtime = getattr(loop, "_goal_task_runtime", None)
    if runtime is None:
        return None
    existing_binding = None
    current_binding = getattr(runtime, "current_binding", None)
    if callable(current_binding):
        try:
            existing_binding = current_binding()
        except Exception:
            existing_binding = None
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None) if active_frame is not None else None
    if unified_context is None and _has_goal_task_binding(existing_binding):
        return existing_binding
    state_mgr = getattr(loop, "_state_mgr", None)
    refresher = getattr(runtime, "refresh", None)
    if callable(refresher):
        try:
            binding = refresher(
                unified_context=unified_context,
                state_mgr=state_mgr,
                episode=int(getattr(loop, "_episode", 0) or 0),
                tick=int(getattr(loop, "_tick", 0) or 0),
            )
            if _has_goal_task_binding(binding):
                return binding
        except Exception:
            pass
    if _has_goal_task_binding(existing_binding):
        return existing_binding
    if callable(current_binding):
        try:
            return current_binding()
        except Exception:
            return None
    return None


def _has_goal_task_binding(binding: Any) -> bool:
    return binding is not None and any(
        getattr(binding, attr, None) is not None
        for attr in ("goal_contract", "task_graph", "active_task")
    )


def goal_task_route_specs(loop: Any) -> Dict[str, Dict[str, Any]]:
    binding = goal_task_binding_for_llm_policy(loop)
    if binding is None:
        return {}
    goal_contract = getattr(binding, "goal_contract", None)
    active_task = getattr(binding, "active_task", None)
    goal_specs: Dict[str, Dict[str, Any]] = {}
    if goal_contract is not None:
        planning = getattr(goal_contract, "planning", None)
        goal_specs = resolve_llm_route_policies(
            getattr(planning, "llm_route_policies", {}) if planning is not None else {}
        )
        _annotate_goal_policy_metadata(goal_specs, getattr(goal_contract, "goal_id", ""))
    task_specs: Dict[str, Dict[str, Any]] = {}
    if active_task is not None:
        task_specs = resolve_llm_route_policies(getattr(active_task, "llm_route_policies", {}))
        _annotate_task_policy_metadata(
            task_specs,
            getattr(active_task, "goal_id", ""),
            getattr(active_task, "node_id", ""),
        )
    return merge_llm_route_specs(goal_specs, task_specs)


def goal_task_capability_specs(loop: Any) -> Dict[str, Dict[str, Any]]:
    binding = goal_task_binding_for_llm_policy(loop)
    if binding is None:
        return {}
    goal_contract = getattr(binding, "goal_contract", None)
    active_task = getattr(binding, "active_task", None)
    goal_specs: Dict[str, Dict[str, Any]] = {}
    if goal_contract is not None:
        planning = getattr(goal_contract, "planning", None)
        goal_specs = resolve_llm_capability_policies(
            getattr(planning, "llm_capability_policies", {}) if planning is not None else {}
        )
        _annotate_goal_policy_metadata(goal_specs, getattr(goal_contract, "goal_id", ""))
    task_specs: Dict[str, Dict[str, Any]] = {}
    if active_task is not None:
        task_specs = resolve_llm_capability_policies(getattr(active_task, "llm_capability_policies", {}))
        _annotate_task_policy_metadata(
            task_specs,
            getattr(active_task, "goal_id", ""),
            getattr(active_task, "node_id", ""),
        )
    return merge_llm_capability_specs(goal_specs, task_specs)


def resolved_llm_route_specs(loop: Any) -> Dict[str, Any]:
    return merge_llm_route_specs(
        runtime_budget_route_specs(loop),
        goal_task_route_specs(loop),
        getattr(loop, "_llm_route_specs", {}) or {},
    )


def resolved_llm_capability_specs(loop: Any) -> Dict[str, Dict[str, Any]]:
    return merge_llm_capability_specs(
        runtime_budget_capability_specs(loop),
        goal_task_capability_specs(loop),
        getattr(loop, "_llm_capability_policies", {}) or {},
    )


def _annotate_goal_policy_metadata(specs: Dict[str, Dict[str, Any]], goal_id: Any) -> None:
    if not specs:
        return
    goal_ref = str(goal_id or "")
    for spec in specs.values():
        metadata = dict(spec.get("metadata", {}) or {})
        metadata.setdefault("policy_source", "goal_contract")
        if goal_ref:
            metadata.setdefault("goal_ref", goal_ref)
        spec["metadata"] = metadata


def _annotate_task_policy_metadata(specs: Dict[str, Dict[str, Any]], goal_id: Any, task_id: Any) -> None:
    if not specs:
        return
    goal_ref = str(goal_id or "")
    task_ref = str(task_id or "")
    for spec in specs.values():
        metadata = dict(spec.get("metadata", {}) or {})
        metadata["policy_source"] = "task_node"
        if goal_ref:
            metadata.setdefault("goal_ref", goal_ref)
        if task_ref:
            metadata.setdefault("task_ref", task_ref)
        spec["metadata"] = metadata


def build_llm_route_context(
    loop: Any,
    route_name: str,
    *,
    capability_request: str = "",
    capability_resolution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_key = str(route_name or "general").strip() or "general"
    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None) if active_frame is not None else None
    plan_state_summary = (
        dict(getattr(unified_context, "plan_state_summary", {}) or {})
        if unified_context is not None and isinstance(getattr(unified_context, "plan_state_summary", {}), dict)
        else {}
    )
    task_contract = (
        dict(plan_state_summary.get("task_contract", {}) or {})
        if isinstance(plan_state_summary.get("task_contract", {}), dict)
        else {}
    )
    completion_gate = (
        dict(plan_state_summary.get("completion_gate", {}) or {})
        if isinstance(plan_state_summary.get("completion_gate", {}), dict)
        else {}
    )
    execution_authority = (
        dict(plan_state_summary.get("execution_authority", {}) or {})
        if isinstance(plan_state_summary.get("execution_authority", {}), dict)
        else {}
    )
    verifier_runtime = build_verifier_runtime(
        task_contract=task_contract,
        completion_gate=completion_gate,
        execution_authority=execution_authority,
        context=plan_state_summary,
    )
    verifier_authority = dict(verifier_runtime.verifier_authority)
    active_task_node = resolve_plan_summary_active_task(plan_state_summary)
    goal_contract = (
        dict(plan_state_summary.get("goal_contract", {}) or {})
        if isinstance(plan_state_summary.get("goal_contract", {}), dict)
        else {}
    )
    active_verification_gate = (
        dict(active_task_node.get("verification_gate", {}) or {})
        if isinstance(active_task_node.get("verification_gate", {}), dict)
        else {}
    )
    uncertainty_vector = (
        dict(getattr(unified_context, "uncertainty_vector", {}) or {})
        if unified_context is not None and isinstance(getattr(unified_context, "uncertainty_vector", {}), dict)
        else {}
    )
    posterior_summary = (
        dict(getattr(unified_context, "posterior_summary", {}) or {})
        if unified_context is not None and isinstance(getattr(unified_context, "posterior_summary", {}), dict)
        else {}
    )
    execution_snapshot = (
        dict(posterior_summary.get("execution_snapshot", {}) or {})
        if isinstance(posterior_summary.get("execution_snapshot", {}), dict)
        else {}
    )
    deliberation_snapshot = (
        dict(posterior_summary.get("deliberation_snapshot", {}) or {})
        if isinstance(posterior_summary.get("deliberation_snapshot", {}), dict)
        else {}
    )
    compute_budget = (
        dict(getattr(unified_context, "compute_budget", {}) or {})
        if unified_context is not None and isinstance(getattr(unified_context, "compute_budget", {}), dict)
        else {}
    )

    uncertainty_candidates = [
        float(getattr(unified_context, "world_shift_risk", 0.0) or 0.0) if unified_context is not None else 0.0,
        float(uncertainty_vector.get("overall", 0.0) or 0.0),
        float(execution_snapshot.get("transition_uncertainty", 0.0) or 0.0),
        float(deliberation_snapshot.get("uncertainty_focus", 0.0) or 0.0),
    ]
    uncertainty_level = max(0.0, min(1.0, max(uncertainty_candidates or [0.0])))

    verification_pressure = 0.0
    blocked_reasons = {
        str(item or "").strip()
        for item in list(completion_gate.get("blocked_reasons", []) or [])
        if str(item or "").strip()
    }
    verifier_verdict = str(verifier_authority.get("verdict", "") or "").strip().lower()
    verifier_required = bool(verifier_authority.get("required", False))
    # Pending verification should bias toward trusted/verification-capable routes,
    # but only failed verdicts represent a full contradiction that maxes out pressure.
    if verifier_verdict == "failed":
        verification_pressure = 1.0
    elif verifier_verdict == "pending":
        verification_pressure = 0.72
    elif "verification_incomplete" in blocked_reasons:
        verification_pressure = 0.72
    elif verifier_required or bool(active_verification_gate.get("required", False)) or bool(completion_gate.get("requires_verification", False)):
        verification_pressure = 0.55

    resource_pressure = str(compute_budget.get("resource_pressure", getattr(unified_context, "resource_pressure", "normal")) or "normal").strip().lower()
    compute_remaining = float(compute_budget.get("compute_budget", 1.0) or 1.0)
    prefer_low_cost = 0.0
    if resource_pressure in {"tight", "constrained", "scarce", "low"} or compute_remaining <= 0.45:
        prefer_low_cost = 0.82
    elif resource_pressure in {"elevated", "warm"} or compute_remaining <= 0.7:
        prefer_low_cost = 0.45

    prefer_low_latency = 0.0
    if route_key in {"retrieval", "probe", "skill"}:
        prefer_low_latency = 0.55
    retrieval_pressure = float(getattr(unified_context, "retrieval_pressure", 0.0) or 0.0) if unified_context is not None else 0.0
    if retrieval_pressure >= 0.6:
        prefer_low_latency = max(prefer_low_latency, 0.72)
    if route_key in {"planning", "planner", "plan_generation", "deliberation"}:
        prefer_low_cost = 0.0
        prefer_low_latency = 0.0

    prefer_high_trust = 0.0
    if uncertainty_level >= 0.55:
        prefer_high_trust = max(prefer_high_trust, 0.65)
    if verification_pressure >= 0.6:
        prefer_high_trust = max(prefer_high_trust, 0.9)
    if route_key in {"planning", "planner", "plan_generation", "deliberation"}:
        prefer_high_trust = max(prefer_high_trust, 0.98)

    resolved_capability = (
        dict(capability_resolution or {})
        if isinstance(capability_resolution, dict)
        else {}
    )
    prefer_structured_output = 1.0 if route_key == "structured_answer" else 0.0
    required_capabilities = [
        str(item).strip()
        for item in list(resolved_capability.get("required_capabilities", []) or [])
        if str(item).strip()
    ]
    if not required_capabilities:
        required_capabilities = route_capability_requirements(route_key)
    if verification_pressure >= 0.5 and "verification" not in required_capabilities:
        required_capabilities.append("verification")

    feedback_summary = loop._llm_route_feedback_summary()
    context_metadata = {
        "goal_id": str(goal_contract.get("goal_id", "") or ""),
        "active_task_id": str(active_task_node.get("node_id", "") or ""),
        "resource_pressure": resource_pressure,
        "completion_gate_blocked_reasons": sorted(blocked_reasons),
        "verifier_verdict": verifier_verdict,
        "verifier_required": verifier_required,
        "verifier_runtime_version": str(verifier_runtime.runtime_version),
        "feedback_available_routes": sorted(feedback_summary.keys()),
        "capability_request": str(capability_request or resolved_capability.get("capability", "") or ""),
        "capability_route_name": str(resolved_capability.get("route_name", "") or route_key),
        "capability_policy_source": str(resolved_capability.get("policy_source", "") or ""),
    }
    if route_key in {"planning", "planner", "plan_generation", "deliberation"}:
        context_metadata["thinking_policy"] = "unbounded_plan_generation"
        context_metadata["prefer_strongest_model"] = True
    route_context = {
        "required_capabilities": required_capabilities,
        "uncertainty_level": round(uncertainty_level, 4),
        "verification_pressure": round(verification_pressure, 4),
        "prefer_low_cost": round(prefer_low_cost, 4),
        "prefer_low_latency": round(prefer_low_latency, 4),
        "prefer_high_trust": round(prefer_high_trust, 4),
        "prefer_structured_output": round(prefer_structured_output, 4),
        "route_feedback": feedback_summary,
        "metadata": context_metadata,
    }
    route_mode = infer_route_runtime_mode(route_key, route_context).to_dict()
    route_context["runtime_mode"] = route_mode["mode"]
    route_context["metadata"]["runtime_mode"] = route_mode
    return apply_status_escalation_to_route_context(route_context, route_name=route_key)

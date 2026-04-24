from __future__ import annotations

from typing import Any, Dict, Optional

from core.orchestration.goal_task_control import resolve_plan_summary_active_task
from core.orchestration.verifier_runtime import build_verifier_runtime


DEFAULT_ROUTE_CAPABILITY_REQUIREMENTS: Dict[str, list[str]] = {
    "general": ["reasoning"],
    "deliberation": ["reasoning", "planning"],
    "retrieval": ["retrieval", "grounding"],
    "hypothesis": ["reasoning", "uncertainty"],
    "probe": ["verification", "reasoning"],
    "skill": ["instruction_following"],
    "recovery": ["recovery", "reasoning"],
    "representation": ["representation"],
    "structured_answer": ["structured_output", "reasoning"],
    "shadow": ["analysis"],
    "analyst": ["analysis", "verification"],
}


def route_capability_requirements(route_name: str) -> list[str]:
    route_key = str(route_name or "general").strip() or "general"
    return list(DEFAULT_ROUTE_CAPABILITY_REQUIREMENTS.get(route_key, ["reasoning"]))


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

    prefer_high_trust = 0.0
    if uncertainty_level >= 0.55:
        prefer_high_trust = max(prefer_high_trust, 0.65)
    if verification_pressure >= 0.6:
        prefer_high_trust = max(prefer_high_trust, 0.9)

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
    return {
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

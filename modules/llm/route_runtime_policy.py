from __future__ import annotations

from typing import Any, Dict, Mapping

from core.runtime.runtime_modes import infer_route_runtime_mode, mode_policy_for_mode, normalize_runtime_mode
from .thinking_policy import thinking_policy_for_route


ROUTE_RUNTIME_POLICY_VERSION = "conos.llm.route_runtime_policy/v1"


_MAX_TOKENS_BY_ROUTE: Dict[str, int] = {
    "retrieval": 64,
    "file_classification": 64,
    "log_summary": 128,
    "json_output": 256,
    "structured_answer": 256,
    "skill": 128,
    "representation": 256,
    "candidate_ranking": 192,
    "probe": 256,
    "hypothesis": 512,
    "root_cause": 768,
    "test_failure": 768,
    "patch_proposal": 900,
    "recovery": 512,
    "final_audit": 1024,
    "analyst": 1024,
    "planning": 2048,
    "planner": 2048,
    "plan_generation": 2048,
    "deliberation": 2048,
}


_BUDGET_BY_TIER: Dict[int, Dict[str, Any]] = {
    0: {
        "request_budget": 8,
        "token_budget": 12_000,
        "priority": "cheap",
        "runtime_tier": "tier_0_deterministic",
    },
    1: {
        "request_budget": 6,
        "token_budget": 16_000,
        "priority": "normal",
        "runtime_tier": "tier_1_lightweight",
    },
    2: {
        "request_budget": 4,
        "token_budget": 24_000,
        "priority": "reasoning",
        "runtime_tier": "tier_2_reasoning",
    },
    3: {
        "request_budget": 3,
        "token_budget": 48_000,
        "priority": "critical",
        "runtime_tier": "tier_3_strong_model",
    },
}


def _normalize_route(route_name: Any) -> str:
    route = str(route_name or "general").strip().lower() or "general"
    if "." in route:
        route = route.split(".", 1)[0]
    return route


def _runtime_mode_for_route(route_name: Any, runtime_mode: Any = None) -> str:
    normalized = normalize_runtime_mode(runtime_mode)
    if normalized:
        return normalized
    return str(infer_route_runtime_mode(_normalize_route(route_name)).mode)


def _mode_budget(runtime_mode: Any) -> Dict[str, Any]:
    policy = mode_policy_for_mode(runtime_mode)
    return dict(policy.get("llm_budget", {}) or {})


def _mode_model_selection(runtime_mode: Any) -> Dict[str, Any]:
    policy = mode_policy_for_mode(runtime_mode)
    return dict(policy.get("model_selection", {}) or {})


def _cap_int(value: Any, cap: Any) -> int:
    selected = max(0, int(value or 0))
    if cap in (None, ""):
        return selected
    return min(selected, max(0, int(cap or 0)))


def route_runtime_call_defaults(route_name: Any, *, mode: Any = "auto", runtime_mode: Any = None) -> Dict[str, Any]:
    route = _normalize_route(route_name)
    thinking = thinking_policy_for_route(route, mode=mode)
    selected_runtime_mode = _runtime_mode_for_route(route, runtime_mode)
    budget = _mode_budget(selected_runtime_mode)
    model_selection = _mode_model_selection(selected_runtime_mode)
    defaults: Dict[str, Any] = {
        "max_tokens": _cap_int(_MAX_TOKENS_BY_ROUTE.get(route, 256), budget.get("max_completion_tokens")),
        "temperature": 0.0,
        "think": bool(thinking.think),
        "runtime_mode": selected_runtime_mode,
        "model_tier": str(model_selection.get("model_tier") or ""),
        "prefer_strongest_model": bool(thinking.prefer_strongest_model or model_selection.get("prefer_strongest_model", False)),
    }
    if thinking.thinking_budget is not None:
        defaults["thinking_budget"] = int(thinking.thinking_budget)
    if thinking.timeout_sec is not None:
        timeout_cap = budget.get("max_wall_clock_seconds")
        defaults["timeout_sec"] = min(float(thinking.timeout_sec), float(timeout_cap)) if timeout_cap not in (None, "") else float(thinking.timeout_sec)
    if int(budget.get("max_llm_calls", 1) or 0) <= 0:
        defaults["think"] = False
        defaults["thinking_budget"] = 0
        defaults["max_tokens"] = 0
        defaults["timeout_sec"] = 0.0
    if str(model_selection.get("thinking_mode") or "") == "off":
        defaults["think"] = False
        defaults["thinking_budget"] = 0
    return defaults


def apply_route_runtime_call_defaults(
    route_name: Any,
    kwargs: Mapping[str, Any] | None = None,
    *,
    mode: Any = "auto",
    runtime_mode: Any = None,
) -> Dict[str, Any]:
    merged = dict(kwargs or {})
    selected_runtime_mode = runtime_mode if runtime_mode is not None else merged.get("runtime_mode")
    defaults = route_runtime_call_defaults(route_name, mode=mode, runtime_mode=selected_runtime_mode)
    for key, value in defaults.items():
        if key == "timeout_sec":
            try:
                current = float(merged.get(key)) if merged.get(key) is not None else 0.0
            except (TypeError, ValueError):
                current = 0.0
            merged[key] = max(current, float(value))
            continue
        merged.setdefault(key, value)
    return merged


def route_runtime_policy_for_route(route_name: Any, *, mode: Any = "auto", runtime_mode: Any = None) -> Dict[str, Any]:
    route = _normalize_route(route_name)
    thinking = thinking_policy_for_route(route, mode=mode)
    tier = int(max(0, min(3, int(thinking.tier or 0))))
    budget = dict(_BUDGET_BY_TIER.get(tier, _BUDGET_BY_TIER[1]))
    selected_runtime_mode = _runtime_mode_for_route(route, runtime_mode)
    mode_policy = mode_policy_for_mode(selected_runtime_mode)
    mode_budget = dict(mode_policy.get("llm_budget", {}) or {})
    if mode_budget.get("max_llm_calls") is not None:
        budget["request_budget"] = min(int(budget.get("request_budget", 0) or 0), int(mode_budget.get("max_llm_calls", 0) or 0))
    token_cap = int(mode_budget.get("max_prompt_tokens", 0) or 0) + int(mode_budget.get("max_completion_tokens", 0) or 0)
    if token_cap > 0:
        budget["token_budget"] = min(int(budget.get("token_budget", 0) or 0), token_cap)
    budget["max_wall_clock_seconds"] = mode_budget.get("max_wall_clock_seconds")
    budget["max_retry_count"] = mode_budget.get("max_retry_count")
    budget["escalation_allowed"] = bool(mode_budget.get("escalation_allowed", True))
    budget["metadata"] = {
        "policy_source": "route_runtime_policy",
        "thinking_tier": tier,
        "prefer_strongest_model": bool(thinking.prefer_strongest_model),
        "runtime_mode": selected_runtime_mode,
        "runtime_mode_policy_version": str(mode_policy.get("schema_version") or ""),
    }
    return {
        "schema_version": ROUTE_RUNTIME_POLICY_VERSION,
        "route_name": route,
        "runtime_mode": selected_runtime_mode,
        "runtime_tier": str(budget.get("runtime_tier", "")),
        "timeout_sec": thinking.timeout_sec,
        "max_tokens": _cap_int(_MAX_TOKENS_BY_ROUTE.get(route, 256), mode_budget.get("max_completion_tokens")),
        "thinking_policy": thinking.to_dict(),
        "runtime_mode_policy": mode_policy,
        "budget": budget,
        "model_selection": dict(mode_policy.get("model_selection", {}) or {}),
        "call_defaults": route_runtime_call_defaults(route, mode=mode, runtime_mode=selected_runtime_mode),
    }


def route_runtime_policies_for_routes(routes: Any, *, mode: Any = "auto", runtime_mode: Any = None) -> Dict[str, Dict[str, Any]]:
    policies: Dict[str, Dict[str, Any]] = {}
    for route_name in list(routes or []):
        route = _normalize_route(route_name)
        if route and route not in policies:
            policies[route] = route_runtime_policy_for_route(route, mode=mode, runtime_mode=runtime_mode)
    return policies


def route_runtime_policy_from_metadata(route_name: Any, metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    route = _normalize_route(route_name)
    payload = dict(metadata or {})
    nested_metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {}
    policies = (
        nested_metadata.get("route_runtime_policies", {})
        if isinstance(nested_metadata.get("route_runtime_policies", {}), Mapping)
        else {}
    )
    if isinstance(policies.get(route), Mapping):
        return dict(policies.get(route) or {})
    direct = payload.get("runtime_policy", {})
    if isinstance(direct, Mapping) and str(direct.get("route_name", "") or "") == route:
        return dict(direct)
    runtime_mode = payload.get("runtime_mode") or nested_metadata.get("runtime_mode")
    if isinstance(runtime_mode, Mapping):
        runtime_mode = runtime_mode.get("mode")
    return route_runtime_policy_for_route(route, runtime_mode=runtime_mode)

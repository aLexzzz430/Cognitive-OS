from __future__ import annotations

from typing import Any, Dict, Mapping

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


def route_runtime_call_defaults(route_name: Any, *, mode: Any = "auto") -> Dict[str, Any]:
    route = _normalize_route(route_name)
    thinking = thinking_policy_for_route(route, mode=mode)
    defaults: Dict[str, Any] = {
        "max_tokens": int(_MAX_TOKENS_BY_ROUTE.get(route, 256)),
        "temperature": 0.0,
        "think": bool(thinking.think),
    }
    if thinking.thinking_budget is not None:
        defaults["thinking_budget"] = int(thinking.thinking_budget)
    if thinking.timeout_sec is not None:
        defaults["timeout_sec"] = float(thinking.timeout_sec)
    return defaults


def apply_route_runtime_call_defaults(
    route_name: Any,
    kwargs: Mapping[str, Any] | None = None,
    *,
    mode: Any = "auto",
) -> Dict[str, Any]:
    merged = dict(kwargs or {})
    defaults = route_runtime_call_defaults(route_name, mode=mode)
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


def route_runtime_policy_for_route(route_name: Any, *, mode: Any = "auto") -> Dict[str, Any]:
    route = _normalize_route(route_name)
    thinking = thinking_policy_for_route(route, mode=mode)
    tier = int(max(0, min(3, int(thinking.tier or 0))))
    budget = dict(_BUDGET_BY_TIER.get(tier, _BUDGET_BY_TIER[1]))
    budget["metadata"] = {
        "policy_source": "route_runtime_policy",
        "thinking_tier": tier,
        "prefer_strongest_model": bool(thinking.prefer_strongest_model),
    }
    return {
        "schema_version": ROUTE_RUNTIME_POLICY_VERSION,
        "route_name": route,
        "runtime_tier": str(budget.get("runtime_tier", "")),
        "timeout_sec": thinking.timeout_sec,
        "max_tokens": int(_MAX_TOKENS_BY_ROUTE.get(route, 256)),
        "thinking_policy": thinking.to_dict(),
        "budget": budget,
        "call_defaults": route_runtime_call_defaults(route, mode=mode),
    }


def route_runtime_policies_for_routes(routes: Any, *, mode: Any = "auto") -> Dict[str, Dict[str, Any]]:
    policies: Dict[str, Dict[str, Any]] = {}
    for route_name in list(routes or []):
        route = _normalize_route(route_name)
        if route and route not in policies:
            policies[route] = route_runtime_policy_for_route(route, mode=mode)
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
    return route_runtime_policy_for_route(route)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


THINKING_POLICY_VERSION = "conos.llm.thinking_policy/v1"


@dataclass(frozen=True)
class ThinkingPolicyDecision:
    route_name: str
    mode: str
    tier: int
    think: bool
    thinking_budget: Optional[int] = None
    timeout_sec: Optional[float] = None
    prefer_strongest_model: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_NO_THINK_ROUTES: Dict[str, Dict[str, Any]] = {
    "retrieval": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 5.0,
        "reason": "cheap retrieval and grounding should be fast and deterministic",
    },
    "file_classification": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 5.0,
        "reason": "file classification is a cheap routing step",
    },
    "log_summary": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 6.0,
        "reason": "log summarization is compression, not deep causal reasoning",
    },
    "json_output": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 8.0,
        "reason": "strict JSON output is more stable without hidden deliberation",
    },
    "structured_answer": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 8.0,
        "reason": "tool kwargs require format stability and low latency",
    },
    "skill": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 5.0,
        "reason": "skill routing and parameter drafts should stay cheap",
    },
    "representation": {
        "tier": 0,
        "thinking_budget": 0,
        "timeout_sec": 6.0,
        "reason": "representation snapshots are mostly compression",
    },
}


_ROUTE_POLICIES: Dict[str, Dict[str, Any]] = {
    **_NO_THINK_ROUTES,
    "candidate_ranking": {
        "tier": 1,
        "think": False,
        "thinking_budget": 0,
        "timeout_sec": 12.0,
        "reason": "candidate ranking should stay light unless escalated by a harder route",
    },
    "probe": {
        "tier": 1,
        "think": True,
        "thinking_budget": 256,
        "timeout_sec": 20.0,
        "reason": "probe selection benefits from small bounded deliberation",
    },
    "hypothesis": {
        "tier": 2,
        "think": True,
        "thinking_budget": 768,
        "timeout_sec": 60.0,
        "reason": "hypothesis construction needs a causal chain but should remain bounded",
    },
    "root_cause": {
        "tier": 2,
        "think": True,
        "thinking_budget": 1024,
        "timeout_sec": 90.0,
        "reason": "root-cause analysis needs explicit causal reasoning",
    },
    "test_failure": {
        "tier": 2,
        "think": True,
        "thinking_budget": 1024,
        "timeout_sec": 90.0,
        "reason": "test failure review should explain the failure before acting",
    },
    "patch_proposal": {
        "tier": 2,
        "think": True,
        "thinking_budget": 1024,
        "timeout_sec": 90.0,
        "reason": "bounded patch design needs cross-file reasoning but remains budgeted",
    },
    "recovery": {
        "tier": 2,
        "think": True,
        "thinking_budget": 1024,
        "timeout_sec": 90.0,
        "reason": "recovery decisions should reason over failure evidence",
    },
    "final_audit": {
        "tier": 3,
        "think": True,
        "thinking_budget": 2048,
        "timeout_sec": 120.0,
        "reason": "final audit is reliability-critical",
    },
    "analyst": {
        "tier": 3,
        "think": True,
        "thinking_budget": 2048,
        "timeout_sec": 120.0,
        "reason": "analyst review is reliability-critical",
    },
    "planning": {
        "tier": 3,
        "think": True,
        "thinking_budget": None,
        "timeout_sec": 300.0,
        "prefer_strongest_model": True,
        "reason": "plan generation should use the strongest available model with unbounded thinking",
    },
    "planner": {
        "tier": 3,
        "think": True,
        "thinking_budget": None,
        "timeout_sec": 300.0,
        "prefer_strongest_model": True,
        "reason": "planner route should use the strongest available model with unbounded thinking",
    },
    "plan_generation": {
        "tier": 3,
        "think": True,
        "thinking_budget": None,
        "timeout_sec": 300.0,
        "prefer_strongest_model": True,
        "reason": "plan generation should use the strongest available model with unbounded thinking",
    },
    "deliberation": {
        "tier": 3,
        "think": True,
        "thinking_budget": None,
        "timeout_sec": 300.0,
        "prefer_strongest_model": True,
        "reason": "planner deliberation should use the strongest available model with unbounded thinking",
    },
}


def _normalize_route(route_name: Any) -> str:
    route = str(route_name or "general").strip().lower() or "general"
    if "." in route:
        route = route.split(".", 1)[0]
    return route


def _normalize_mode(mode: Any) -> str:
    value = str(mode or "auto").strip().lower() or "auto"
    if value in {"default", "route"}:
        return "auto"
    if value in {"no", "none", "false", "no_think", "no-thinking", "off"}:
        return "off"
    if value in {"yes", "true", "think", "thinking", "on"}:
        return "on"
    return "auto"


def thinking_policy_for_route(route_name: Any, *, mode: Any = "auto") -> ThinkingPolicyDecision:
    route = _normalize_route(route_name)
    normalized_mode = _normalize_mode(mode)
    policy = dict(_ROUTE_POLICIES.get(route, {}) or {})
    if not policy:
        policy = {
            "tier": 1,
            "think": False,
            "thinking_budget": 256,
            "timeout_sec": 30.0,
            "reason": "unknown route defaults to cheap bounded behavior",
        }
    raw_tier = policy.get("tier", 1)
    tier = 1 if raw_tier in (None, "") else int(raw_tier)
    think = bool(policy.get("think", False))
    budget = policy.get("thinking_budget")
    if budget is not None:
        budget = max(0, int(budget))
    timeout = policy.get("timeout_sec")
    if timeout is not None:
        timeout = float(timeout)
    if normalized_mode == "off":
        think = False
        budget = 0
        timeout = min(float(timeout or 30.0), 30.0)
    elif normalized_mode == "on":
        think = True
        if budget == 0:
            budget = 512
        if timeout is None:
            timeout = 60.0
    elif budget == 0:
        think = False
    return ThinkingPolicyDecision(
        route_name=route,
        mode=normalized_mode,
        tier=tier,
        think=think,
        thinking_budget=budget,
        timeout_sec=timeout,
        prefer_strongest_model=bool(policy.get("prefer_strongest_model", False)),
        reason=str(policy.get("reason", "") or ""),
    )


def apply_thinking_policy(
    route_name: Any,
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    mode: Any = "auto",
) -> Dict[str, Any]:
    merged = dict(kwargs or {})
    decision = thinking_policy_for_route(route_name, mode=mode)
    merged["think"] = bool(decision.think)
    if decision.thinking_budget is None:
        merged.pop("thinking_budget", None)
    else:
        merged["thinking_budget"] = int(decision.thinking_budget)
    if decision.timeout_sec is not None:
        current = merged.get("timeout_sec")
        try:
            current_timeout = float(current) if current is not None else 0.0
        except (TypeError, ValueError):
            current_timeout = 0.0
        merged["timeout_sec"] = max(current_timeout, float(decision.timeout_sec))
    return merged

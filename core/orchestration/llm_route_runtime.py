from __future__ import annotations

import math
from typing import Any, Dict, Optional

from core.conos_kernel import build_model_call_ticket


def _route_key(route_name: str) -> str:
    return str(route_name or "general").strip() or "general"


def _current_episode_tick(loop: Any) -> tuple[int, int]:
    return int(getattr(loop, "_episode", 0) or 0), int(getattr(loop, "_tick", 0) or 0)


def _json_safe(loop: Any, value: Any) -> Any:
    sanitizer = getattr(loop, "_json_safe", None)
    if callable(sanitizer):
        return sanitizer(value)
    return value


def _cooldown_ready(loop: Any, last_tick: int, cooldown_ticks: int) -> bool:
    checker = getattr(loop, "_cooldown_ready", None)
    if callable(checker):
        return bool(checker(last_tick, cooldown_ticks))
    _, current_tick = _current_episode_tick(loop)
    return current_tick - int(last_tick or 0) >= int(cooldown_ticks or 0)


def _route_context_metadata(route_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    route_context = (route_metadata or {}).get("route_context", {})
    if isinstance(route_context, dict) and isinstance(route_context.get("metadata", {}), dict):
        return dict(route_context.get("metadata", {}) or {})
    return {}


def _model_call_ticket(route_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ticket = (route_metadata or {}).get("model_call_ticket", {})
    return dict(ticket or {}) if isinstance(ticket, dict) else {}


def estimate_llm_token_units(*values: Any) -> int:
    total_chars = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            total_chars += len(value)
        else:
            total_chars += len(repr(value))
    if total_chars <= 0:
        return 0
    return max(1, int(math.ceil(total_chars / 4.0)))


def llm_route_state(loop: Any) -> Dict[str, Any]:
    state = getattr(loop, "_llm_route_runtime_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(loop, "_llm_route_runtime_state", state)
    state.setdefault("per_tick_usage", {})
    state.setdefault("last_call", {})
    state.setdefault("lifetime_usage", {})
    state.setdefault("blocked", {})
    return state


def llm_route_usage_bucket(loop: Any, route_name: str) -> Dict[str, Any]:
    route_name = _route_key(route_name)
    state = llm_route_state(loop)
    per_tick_usage = state.setdefault("per_tick_usage", {})
    bucket = per_tick_usage.get(route_name)
    current_episode, current_tick = _current_episode_tick(loop)
    bucket_episode = int(bucket.get("episode", -1)) if isinstance(bucket, dict) else -1
    bucket_tick = int(bucket.get("tick", -1)) if isinstance(bucket, dict) else -1
    if not isinstance(bucket, dict) or bucket_episode != current_episode or bucket_tick != current_tick:
        bucket = {
            "episode": current_episode,
            "tick": current_tick,
            "request_count": 0,
            "token_count": 0,
        }
        per_tick_usage[route_name] = bucket
    return bucket


def llm_route_budget_status(
    loop: Any,
    *,
    route_name: str,
    route_metadata: Optional[Dict[str, Any]] = None,
    prompt_tokens: int = 0,
    reserved_response_tokens: int = 0,
) -> Dict[str, Any]:
    route_name = _route_key(route_name)
    metadata = dict(route_metadata or {})
    budget = metadata.get("budget", {}) if isinstance(metadata.get("budget", {}), dict) else {}
    request_budget = int(max(0, budget.get("request_budget", 0) or 0))
    token_budget = int(max(0, budget.get("token_budget", 0) or 0))
    cooldown_ticks = int(max(0, budget.get("cooldown_ticks", 0) or 0))
    bucket = llm_route_usage_bucket(loop, route_name)
    state = llm_route_state(loop)
    last_call = state.get("last_call", {}).get(route_name, {})
    current_episode, _ = _current_episode_tick(loop)
    last_call_tick = -10_000
    if isinstance(last_call, dict) and int(last_call.get("episode", -1)) == current_episode:
        last_call_tick = int(last_call.get("tick", -10_000))
    estimated_total_tokens = int(bucket.get("token_count", 0) or 0) + max(0, int(prompt_tokens or 0)) + max(
        0, int(reserved_response_tokens or 0)
    )
    allowed = True
    blocked_reason = ""
    if request_budget and int(bucket.get("request_count", 0) or 0) >= request_budget:
        allowed = False
        blocked_reason = "request_budget_exceeded"
    elif token_budget and estimated_total_tokens > token_budget:
        allowed = False
        blocked_reason = "token_budget_exceeded"
    elif cooldown_ticks and last_call_tick > -10_000 and not _cooldown_ready(loop, last_call_tick, cooldown_ticks):
        allowed = False
        blocked_reason = "cooldown_active"
    return {
        "route_name": route_name,
        "allowed": allowed,
        "blocked_reason": blocked_reason,
        "request_budget": request_budget,
        "token_budget": token_budget,
        "cooldown_ticks": cooldown_ticks,
        "request_count": int(bucket.get("request_count", 0) or 0),
        "token_count": int(bucket.get("token_count", 0) or 0),
        "remaining_request_budget": max(0, request_budget - int(bucket.get("request_count", 0) or 0))
        if request_budget
        else None,
        "remaining_token_budget": max(0, token_budget - int(bucket.get("token_count", 0) or 0))
        if token_budget
        else None,
        "last_call_tick": last_call_tick if last_call_tick > -10_000 else None,
        "prompt_tokens": int(max(0, prompt_tokens or 0)),
        "reserved_response_tokens": int(max(0, reserved_response_tokens or 0)),
        "estimated_total_tokens": estimated_total_tokens,
    }


def record_llm_route_blocked(
    loop: Any,
    *,
    route_name: str,
    method_name: str,
    route_metadata: Optional[Dict[str, Any]],
    budget_status: Dict[str, Any],
    entry_kind: str,
) -> None:
    route_name = _route_key(route_name)
    current_episode, current_tick = _current_episode_tick(loop)
    blocked_entry = {
        "episode": current_episode,
        "tick": current_tick,
        "route_name": route_name,
        "requested_route": str((route_metadata or {}).get("requested_route", route_name) or route_name),
        "method_name": str(method_name or "complete"),
        "entry_kind": str(entry_kind or "runtime_gate"),
        "blocked_reason": str(budget_status.get("blocked_reason", "") or "budget_blocked"),
        "budget_status": dict(budget_status or {}),
        "route_budget": dict((route_metadata or {}).get("budget", {}) or {}),
        "decision_explanation": list((route_metadata or {}).get("decision_explanation", []) or []),
        "selected_route": str((route_metadata or {}).get("selected_route", route_name) or route_name),
        "goal_id": "",
        "active_task_id": "",
    }
    route_context_metadata = _route_context_metadata(route_metadata)
    model_call_ticket = _model_call_ticket(route_metadata)
    blocked_entry["goal_id"] = str(route_context_metadata.get("goal_id", "") or "")
    blocked_entry["active_task_id"] = str(route_context_metadata.get("active_task_id", "") or "")
    blocked_entry["model_call_ticket_id"] = str(model_call_ticket.get("ticket_id", "") or "")
    blocked_entry["audit_event_id"] = str(model_call_ticket.get("audit_event_id", "") or "")
    llm_route_state(loop).setdefault("blocked", {}).setdefault(route_name, []).append(blocked_entry)
    log = getattr(loop, "_llm_route_usage_log", None)
    if isinstance(log, list):
        log.append({**blocked_entry, "event": "blocked"})
    advice_log = getattr(loop, "_llm_advice_log", None)
    if isinstance(advice_log, list):
        advice_log.append(
            {
                "episode": blocked_entry["episode"],
                "tick": blocked_entry["tick"],
                "kind": f"route_blocked::{route_name}",
                "entry": "llm_route_budget_block",
                "route_name": route_name,
                "method_name": blocked_entry["method_name"],
                "blocked_reason": blocked_entry["blocked_reason"],
                "budget_status": dict(budget_status or {}),
            }
        )


def record_llm_route_usage(
    loop: Any,
    *,
    route_name: str,
    method_name: str,
    prompt_tokens: int,
    response_tokens: int,
    reserved_response_tokens: int,
    route_metadata: Optional[Dict[str, Any]],
) -> None:
    route_name = _route_key(route_name)
    current_episode, current_tick = _current_episode_tick(loop)
    bucket = llm_route_usage_bucket(loop, route_name)
    prompt_count = int(max(0, prompt_tokens or 0))
    response_count = int(max(0, response_tokens or 0))
    bucket["request_count"] = int(bucket.get("request_count", 0) or 0) + 1
    bucket["token_count"] = int(bucket.get("token_count", 0) or 0) + prompt_count + response_count
    state = llm_route_state(loop)
    state.setdefault("last_call", {})[route_name] = {
        "episode": current_episode,
        "tick": current_tick,
    }
    lifetime = state.setdefault("lifetime_usage", {}).setdefault(
        route_name,
        {"request_count": 0, "token_count": 0},
    )
    lifetime["request_count"] = int(lifetime.get("request_count", 0) or 0) + 1
    lifetime["token_count"] = int(lifetime.get("token_count", 0) or 0) + prompt_count + response_count
    setattr(loop, "_llm_calls_this_tick", int(getattr(loop, "_llm_calls_this_tick", 0) or 0) + 1)
    usage_entry = {
        "episode": current_episode,
        "tick": current_tick,
        "route_name": route_name,
        "requested_route": str((route_metadata or {}).get("requested_route", route_name) or route_name),
        "method_name": str(method_name or "complete"),
        "entry_kind": "request",
        "prompt_tokens": prompt_count,
        "response_tokens": response_count,
        "reserved_response_tokens": int(max(0, reserved_response_tokens or 0)),
        "request_count_this_tick": int(bucket.get("request_count", 0) or 0),
        "token_count_this_tick": int(bucket.get("token_count", 0) or 0),
        "route_budget": dict((route_metadata or {}).get("budget", {}) or {}),
        "decision_explanation": list((route_metadata or {}).get("decision_explanation", []) or []),
        "selected_route": str((route_metadata or {}).get("selected_route", route_name) or route_name),
        "goal_id": "",
        "active_task_id": "",
    }
    route_context_metadata = _route_context_metadata(route_metadata)
    model_call_ticket = _model_call_ticket(route_metadata)
    usage_entry["goal_id"] = str(route_context_metadata.get("goal_id", "") or "")
    usage_entry["active_task_id"] = str(route_context_metadata.get("active_task_id", "") or "")
    usage_entry["model_call_ticket_id"] = str(model_call_ticket.get("ticket_id", "") or "")
    usage_entry["audit_event_id"] = str(model_call_ticket.get("audit_event_id", "") or "")
    log = getattr(loop, "_llm_route_usage_log", None)
    if isinstance(log, list):
        log.append({**usage_entry, "event": "request"})
    advice_log = getattr(loop, "_llm_advice_log", None)
    if isinstance(advice_log, list):
        advice_log.append(
            {
                "episode": usage_entry["episode"],
                "tick": usage_entry["tick"],
                "kind": f"route_usage::{route_name}",
                "entry": "llm_route_request",
                "route_name": route_name,
                "method_name": usage_entry["method_name"],
                "prompt_tokens": usage_entry["prompt_tokens"],
                "response_tokens": usage_entry["response_tokens"],
                "request_count_this_tick": usage_entry["request_count_this_tick"],
                "token_count_this_tick": usage_entry["token_count_this_tick"],
            }
        )


def llm_route_usage_summary(loop: Any) -> Dict[str, Any]:
    state = llm_route_state(loop)
    feedback_summary = getattr(loop, "_llm_route_feedback_summary", None)
    feedback = feedback_summary() if callable(feedback_summary) else {}
    return {
        "per_tick_usage": _json_safe(loop, dict(state.get("per_tick_usage", {}) or {})),
        "last_call": _json_safe(loop, dict(state.get("last_call", {}) or {})),
        "lifetime_usage": _json_safe(loop, dict(state.get("lifetime_usage", {}) or {})),
        "feedback": _json_safe(loop, feedback),
    }


class RouteBudgetedLLMClient:
    """Route-aware LLM proxy that enforces runtime budget policy at call time."""

    def __init__(
        self,
        *,
        route_name: str,
        client: Any,
        route_metadata: Optional[Dict[str, Any]],
        preflight_budget_check: Any,
        record_usage: Any,
        record_blocked: Any,
    ) -> None:
        self._route_name = str(route_name or "general").strip() or "general"
        self._client = client
        self._route_metadata = dict(route_metadata or {})
        self._preflight_budget_check = preflight_budget_check
        self._record_usage = record_usage
        self._record_blocked = record_blocked

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def complete(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return str(self._invoke("complete", prompt, *args, **kwargs) or "")

    def complete_raw(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return str(self._invoke("complete_raw", prompt, *args, **kwargs) or "")

    def complete_json(self, prompt: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        payload = self._invoke("complete_json", prompt, *args, **kwargs)
        return payload if isinstance(payload, dict) else {}

    def budget_status(self) -> Dict[str, Any]:
        return dict(
            self._preflight_budget_check(
                route_name=self._route_name,
                route_metadata=self._route_metadata,
                prompt_tokens=0,
                reserved_response_tokens=0,
            )
        )

    def is_available(self) -> bool:
        return bool(self.budget_status().get("allowed", False))

    def _invoke(self, method_name: str, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        if self._client is None:
            return self._neutral_result(method_name)
        internal_capability_request = str(kwargs.get("capability_request", "") or "")
        internal_response_schema_name = str(
            kwargs.get("response_schema_name", "")
            or kwargs.get("schema_name", "")
            or kwargs.get("output_schema_name", "")
            or ""
        )
        prompt_tokens = estimate_llm_token_units(prompt, kwargs.get("system_prompt"))
        reserved_response_tokens = int(max(0, kwargs.get("max_tokens", 0) or 0))
        invoke_metadata = dict(self._route_metadata or {})
        route_context = invoke_metadata.get("route_context", {})
        route_context_metadata = (
            dict(route_context.get("metadata", {}) or {})
            if isinstance(route_context, dict) and isinstance(route_context.get("metadata", {}), dict)
            else {}
        )
        model_call_ticket = build_model_call_ticket(
            route_name=self._route_name,
            method_name=method_name,
            capability_request=str(
                internal_capability_request
                or invoke_metadata.get("capability_request", "")
                or self._route_name
            ),
            schema_name=internal_response_schema_name,
            fallback_route=str(invoke_metadata.get("fallback_route", "") or ""),
            goal_ref=str(route_context_metadata.get("goal_id", "") or ""),
            task_ref=str(route_context_metadata.get("active_task_id", "") or ""),
            graph_ref=str(route_context_metadata.get("graph_ref", "") or ""),
            prompt_tokens=prompt_tokens,
            reserved_response_tokens=reserved_response_tokens,
            budget=dict(invoke_metadata.get("budget", {}) or {}),
            metadata={
                "requested_route": str(invoke_metadata.get("requested_route", self._route_name) or self._route_name),
                "selected_route": str(invoke_metadata.get("selected_route", self._route_name) or self._route_name),
                "capability_route_name": str(
                    kwargs.get("capability_route_name", "")
                    or invoke_metadata.get("capability_route_name", self._route_name)
                    or self._route_name
                ),
                "capability_policy_source": str(
                    route_context_metadata.get("capability_policy_source", "")
                    or invoke_metadata.get("capability_policy_source", "")
                    or ""
                ),
                "decision_explanation": list(invoke_metadata.get("decision_explanation", []) or []),
            },
        )
        invoke_metadata["model_call_ticket"] = model_call_ticket.to_dict()
        budget_status = dict(
            self._preflight_budget_check(
                route_name=self._route_name,
                route_metadata=invoke_metadata,
                prompt_tokens=prompt_tokens,
                reserved_response_tokens=reserved_response_tokens,
            )
        )
        if not bool(budget_status.get("allowed", False)):
            self._record_blocked(
                route_name=self._route_name,
                method_name=method_name,
                route_metadata=invoke_metadata,
                budget_status=budget_status,
                entry_kind="runtime_gate",
            )
            return self._neutral_result(method_name)
        forwarded_kwargs = dict(kwargs)
        for internal_key in (
            "capability_request",
            "capability_route_name",
            "response_schema_name",
            "schema_name",
            "output_schema_name",
        ):
            forwarded_kwargs.pop(internal_key, None)
        result = getattr(self._client, method_name)(prompt, *args, **forwarded_kwargs)
        response_tokens = estimate_llm_token_units(result)
        self._record_usage(
            route_name=self._route_name,
            method_name=method_name,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            reserved_response_tokens=reserved_response_tokens,
            route_metadata=invoke_metadata,
        )
        return result

    def _neutral_result(self, method_name: str) -> Any:
        if method_name == "complete_json":
            return {}
        return ""

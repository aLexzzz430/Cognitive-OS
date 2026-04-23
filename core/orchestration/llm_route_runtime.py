from __future__ import annotations

import math
from typing import Any, Dict, Optional

from core.conos_kernel import build_model_call_ticket


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

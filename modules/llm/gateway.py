from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

from .capabilities import capability_name
from .capability_registry import LLMCapabilityResolution
from .route_runtime_policy import apply_route_runtime_call_defaults


@dataclass(frozen=True)
class LLMCapabilityRequest:
    capability: str
    prompt: str
    method: str = "text"
    route_name: str = "general"
    schema_name: str = ""
    max_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMGateway:
    """Capability-first gateway in front of route-selected model clients."""

    def __init__(
        self,
        *,
        route_name: str = "general",
        capability_prefix: str = "",
        client_resolver: Optional[Callable[..., Any]] = None,
        capability_resolver: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._route_name = str(route_name or "general").strip() or "general"
        self._capability_prefix = str(capability_prefix or "").strip()
        self._client_resolver = client_resolver
        self._capability_resolver = capability_resolver
        self._last_error = ""

    @property
    def route_name(self) -> str:
        return self._route_name

    @property
    def capability_prefix(self) -> str:
        return self._capability_prefix

    def resolve_client(self, capability: Any = "") -> Any:
        resolution = self._resolve_capability(self._normalize_capability(capability))
        if callable(self._client_resolver):
            try:
                return self._client_resolver(resolution.route_name, resolution)
            except TypeError:
                return self._client_resolver()
        return None

    def is_available(self) -> bool:
        client = self.resolve_client()
        return client is not None and (
            not hasattr(client, "is_available") or bool(client.is_available())
        )

    def request_text(self, capability: Any, prompt: str, **kwargs: Any) -> str:
        kwargs = self._with_route_defaults(capability, kwargs)
        request = self._build_request(capability, prompt, method="text", kwargs=kwargs)
        client = self.resolve_client(request.capability)
        if client is None:
            return ""
        try:
            result = self._call_client_method(
                    client,
                    "complete",
                    prompt,
                    capability_request=request.capability,
                    capability_route_name=request.route_name,
                    response_schema_name=request.schema_name,
                    **kwargs,
                )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return ""
        self._last_error = ""
        return str(result or "")

    @property
    def last_error(self) -> str:
        return self._last_error

    def _with_route_defaults(self, capability: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(kwargs or {})
        route = self._route_name
        capability_name_value = capability_name(capability)
        capability_route = str(getattr(capability, "route_name", "") or "").strip()
        if capability_route and capability_route != "general":
            route = capability_route
        known_routes = {
            "retrieval",
            "hypothesis",
            "representation",
            "skill",
            "recovery",
            "structured_answer",
            "patch_proposal",
            "root_cause",
            "test_failure",
            "planning",
            "planner",
            "plan_generation",
            "deliberation",
        }
        if not capability_route and "." in capability_name_value:
            route_prefix = capability_name_value.split(".", 1)[0]
            if route_prefix in known_routes:
                route = route_prefix
        return apply_route_runtime_call_defaults(route, merged, mode=merged.pop("thinking_mode", "auto"))

    def request_raw(self, capability: Any, prompt: str, **kwargs: Any) -> str:
        kwargs = self._with_route_defaults(capability, kwargs)
        request = self._build_request(capability, prompt, method="raw", kwargs=kwargs)
        client = self.resolve_client(request.capability)
        if client is None:
            return ""
        if hasattr(client, "complete_raw"):
            try:
                result = self._call_client_method(
                    client,
                    "complete_raw",
                    prompt,
                    capability_request=request.capability,
                    capability_route_name=request.route_name,
                    response_schema_name=request.schema_name,
                    **kwargs,
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return ""
            self._last_error = ""
            return str(result or "")
        return self.request_text(capability, prompt, **kwargs)

    def request_json(
        self,
        capability: Any,
        prompt: str,
        *,
        schema_name: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        kwargs = self._with_route_defaults(capability, kwargs)
        request = self._build_request(
            capability,
            prompt,
            method="json",
            kwargs=kwargs,
            schema_name=schema_name,
        )
        client = self.resolve_client(request.capability)
        if client is None:
            return {}
        if hasattr(client, "complete_json"):
            try:
                payload = self._call_client_method(
                    client,
                    "complete_json",
                    prompt,
                    capability_request=request.capability,
                    capability_route_name=request.route_name,
                    response_schema_name=request.schema_name,
                    **kwargs,
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return {}
            self._last_error = ""
            return payload if isinstance(payload, dict) else {}
        text = self.request_text(
            capability,
            prompt,
            response_schema_name=request.schema_name,
            **kwargs,
        )
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_request(
        self,
        capability: Any,
        prompt: str,
        *,
        method: str,
        kwargs: Dict[str, Any],
        schema_name: str = "",
    ) -> LLMCapabilityRequest:
        clean_capability = self._normalize_capability(capability)
        resolution = self._resolve_capability(clean_capability)
        return LLMCapabilityRequest(
            capability=resolution.capability,
            prompt=str(prompt or ""),
            method=method,
            route_name=resolution.route_name,
            schema_name=str(schema_name or resolution.schema_name or ""),
            max_tokens=int(max(0, kwargs.get("max_tokens", 0) or 0)),
            metadata={
                "temperature": kwargs.get("temperature"),
                "system_prompt": kwargs.get("system_prompt"),
                "required_capabilities": list(resolution.required_capabilities),
                "capability_policy_source": str(resolution.policy_source or ""),
                "capability_metadata": dict(resolution.metadata or {}),
            },
        )

    def _resolve_capability(self, capability: str) -> LLMCapabilityResolution:
        clean_capability = capability_name(capability)
        if callable(self._capability_resolver):
            try:
                resolved = self._capability_resolver(clean_capability, self._route_name)
            except TypeError:
                resolved = self._capability_resolver(clean_capability)
            if isinstance(resolved, LLMCapabilityResolution):
                return resolved
            if isinstance(resolved, dict):
                return LLMCapabilityResolution(
                    capability=str(resolved.get("capability", clean_capability) or clean_capability or self._route_name),
                    route_name=str(resolved.get("route_name", self._route_name) or self._route_name),
                    required_capabilities=list(resolved.get("required_capabilities", []) or []),
                    schema_name=str(resolved.get("schema_name", "") or ""),
                    fallback_route=str(resolved.get("fallback_route", self._route_name) or self._route_name),
                    policy_source=str(resolved.get("policy_source", "") or ""),
                    metadata=dict(resolved.get("metadata", {}) or {}) if isinstance(resolved.get("metadata", {}), dict) else {},
                )
        return LLMCapabilityResolution(
            capability=clean_capability or self._route_name,
            route_name=self._route_name,
            required_capabilities=[self._route_name],
            fallback_route=self._route_name,
            policy_source="gateway_fallback",
            metadata={"policy_source": "gateway_fallback"},
        )

    def _normalize_capability(self, capability: Any) -> str:
        clean_capability = capability_name(capability)
        if clean_capability and "." not in clean_capability and self._capability_prefix:
            clean_capability = f"{self._capability_prefix}.{clean_capability}"
        if not clean_capability:
            clean_capability = self._capability_prefix or self._route_name
        return clean_capability

    def _call_client_method(self, client: Any, method_name: str, prompt: str, **kwargs: Any) -> Any:
        method = getattr(client, method_name, None)
        if not callable(method):
            return None
        try:
            return method(prompt, **kwargs)
        except TypeError:
            sanitized = self._kwargs_supported_by_method(method, kwargs)
            try:
                return method(prompt, **sanitized)
            except TypeError:
                return method(prompt)

    @staticmethod
    def _kwargs_supported_by_method(method: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raw = {
            key: value
            for key, value in dict(kwargs or {}).items()
            if key not in {"capability_request", "capability_route_name", "response_schema_name"}
        }
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return raw
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return raw
        return {key: value for key, value in raw.items() if key in parameters}


def ensure_llm_gateway(
    llm_client: Any,
    *,
    route_name: str = "general",
    capability_prefix: str = "",
    capability_resolver: Optional[Callable[..., Any]] = None,
) -> Optional[LLMGateway]:
    if llm_client is None:
        return None
    if isinstance(llm_client, LLMGateway):
        return llm_client
    if hasattr(llm_client, "request_text"):
        return llm_client
    fallback_route = str(route_name or "general").strip() or "general"

    def _client_resolver(_route_name: Optional[str] = None, _resolution: Optional[Any] = None) -> Any:
        return llm_client

    def _capability_resolver(capability: Any, requested_route: str = fallback_route) -> Dict[str, Any]:
        capability_name_value = capability_name(capability) or requested_route or fallback_route
        if callable(capability_resolver):
            try:
                resolved = capability_resolver(capability_name_value, requested_route)
            except TypeError:
                resolved = capability_resolver(capability_name_value)
            if isinstance(resolved, dict):
                return resolved
        return {
            "capability": capability_name_value,
            "route_name": requested_route or fallback_route,
            "fallback_route": fallback_route,
            "policy_source": "legacy_gateway_adapter",
            "metadata": {"policy_source": "legacy_gateway_adapter"},
        }

    return LLMGateway(
        route_name=fallback_route,
        capability_prefix=capability_prefix,
        client_resolver=_client_resolver,
        capability_resolver=_capability_resolver,
    )

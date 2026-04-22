from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from .capabilities import (
    LLM_DEFAULT_CAPABILITY_POLICIES,
    LLM_DEFAULT_CAPABILITY_PREFIX_POLICIES,
)
from core.runtime_budget import merge_llm_capability_specs, resolve_llm_capability_policies


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in dict(value).items()}


@dataclass(frozen=True)
class LLMCapabilityResolution:
    capability: str
    route_name: str
    required_capabilities: list[str] = field(default_factory=list)
    schema_name: str = ""
    fallback_route: str = ""
    policy_source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DEFAULT_CAPABILITY_POLICIES: Dict[str, Dict[str, Any]] = {
    **LLM_DEFAULT_CAPABILITY_POLICIES,
    **LLM_DEFAULT_CAPABILITY_PREFIX_POLICIES,
}


class LLMCapabilityRegistry:
    def __init__(self, policies: Optional[Mapping[str, Any]] = None) -> None:
        self._policies: Dict[str, Dict[str, Any]] = {}
        self.set_policies(policies or {})

    def set_policies(self, policies: Mapping[str, Any]) -> None:
        self._policies = merge_llm_capability_specs(_DEFAULT_CAPABILITY_POLICIES, policies)

    def policies(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._policies)

    def resolve(self, capability: str, *, fallback_route: str = "general") -> LLMCapabilityResolution:
        clean_capability = str(capability or "").strip() or "general"
        exact_match = self._policies.get(clean_capability)
        if isinstance(exact_match, dict):
            return self._resolution_from_policy(
                clean_capability,
                exact_match,
                fallback_route=fallback_route,
            )
        prefix_match_key = self._longest_prefix_match(clean_capability)
        if prefix_match_key:
            return self._resolution_from_policy(
                clean_capability,
                dict(self._policies.get(prefix_match_key, {}) or {}),
                fallback_route=fallback_route,
            )
        return LLMCapabilityResolution(
            capability=clean_capability,
            route_name=str(fallback_route or "general").strip() or "general",
            required_capabilities=[str(fallback_route or "general").strip() or "general"],
            policy_source="fallback",
            metadata={"policy_source": "fallback"},
        )

    def _longest_prefix_match(self, capability: str) -> str:
        clean_capability = str(capability or "").strip()
        matches = [
            key
            for key in self._policies.keys()
            if key.endswith(".*") and clean_capability.startswith(key[:-1])
        ]
        if not matches:
            return ""
        matches.sort(key=len, reverse=True)
        return matches[0]

    def _resolution_from_policy(
        self,
        capability: str,
        policy: Dict[str, Any],
        *,
        fallback_route: str,
    ) -> LLMCapabilityResolution:
        route_name = str(policy.get("route_name", "") or fallback_route or "general").strip() or "general"
        metadata = _dict_or_empty(policy.get("metadata", {}))
        return LLMCapabilityResolution(
            capability=str(capability or "").strip() or route_name,
            route_name=route_name,
            required_capabilities=_string_list(policy.get("required_capabilities", [])),
            schema_name=str(policy.get("schema_name", "") or ""),
            fallback_route=str(policy.get("fallback_route", "") or fallback_route or "").strip(),
            policy_source=str(metadata.get("policy_source", "") or policy.get("policy_source", "") or ""),
            metadata=metadata,
        )


def resolve_llm_capability_registry_policies(value: Any) -> Dict[str, Dict[str, Any]]:
    return resolve_llm_capability_policies(value)

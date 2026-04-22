from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in dict(value).items()}


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _capability_name(value: Any) -> str:
    from modules.llm.capabilities import capability_name

    return capability_name(value)


def _canonical_capability_name(value: Any, *, fallback: str = "", namespace: str = "") -> str:
    from modules.llm.capabilities import canonical_capability_name

    return canonical_capability_name(value, fallback=fallback, namespace=namespace)


def _capability_spec(value: Any, *, namespace: str = "") -> Any:
    from modules.llm.capabilities import capability_spec

    return capability_spec(value, namespace=namespace)


@dataclass(frozen=True)
class LLMCapabilityPolicy:
    route_name: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    schema_name: str = ""
    fallback_route: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }


def coerce_llm_capability_policy(value: Any) -> LLMCapabilityPolicy:
    if isinstance(value, LLMCapabilityPolicy):
        return value
    if not isinstance(value, Mapping):
        return LLMCapabilityPolicy()
    payload = dict(value)
    return LLMCapabilityPolicy(
        route_name=str(payload.get("route_name", "") or ""),
        required_capabilities=_string_list(payload.get("required_capabilities", [])),
        schema_name=str(payload.get("schema_name", "") or ""),
        fallback_route=str(payload.get("fallback_route", "") or ""),
        metadata=_dict_or_empty(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class LLMCapabilityPolicyEntry:
    capability: str = ""
    capability_ref: str = ""
    namespace: str = ""
    route_name: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    schema_name: str = ""
    fallback_route: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_policy(self) -> LLMCapabilityPolicy:
        return LLMCapabilityPolicy(
            route_name=self.route_name,
            required_capabilities=list(self.required_capabilities),
            schema_name=self.schema_name,
            fallback_route=self.fallback_route,
            metadata=dict(self.metadata),
        )

    def to_policy_dict(self) -> Dict[str, Any]:
        return self.to_policy().to_dict()

    def to_dict(self) -> Dict[str, Any]:
        canonical_name = _canonical_capability_name(self.capability, namespace=self.namespace) or _capability_name(
            self.capability
        )
        capability_ref = _capability_name(self.capability_ref) or canonical_name
        payload = {
            "capability": canonical_name,
            "capability_ref": capability_ref,
            "namespace": self.namespace,
            "route_name": self.route_name,
            "required_capabilities": list(self.required_capabilities),
            "schema_name": self.schema_name,
            "fallback_route": self.fallback_route,
            "metadata": dict(self.metadata),
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }


def coerce_llm_capability_policy_entry(
    value: Any,
    *,
    fallback_capability: Any = "",
    namespace: str = "",
) -> LLMCapabilityPolicyEntry:
    payload = value.to_dict() if isinstance(value, LLMCapabilityPolicyEntry) else dict(value or {}) if isinstance(value, Mapping) else {}
    capability_ref = (
        payload.pop("capability", None)
        or payload.pop("capability_ref", None)
        or payload.pop("capability_name", None)
        or payload.pop("full_name", None)
        or payload.pop("name", None)
        or fallback_capability
    )
    namespace_hint = str(
        payload.pop("namespace", "")
        or payload.pop("capability_namespace", "")
        or namespace
        or ""
    ).strip()
    canonical_name = _canonical_capability_name(
        capability_ref,
        namespace=namespace_hint,
    )
    if not canonical_name:
        return LLMCapabilityPolicyEntry()
    spec = _capability_spec(canonical_name, namespace=namespace_hint)
    normalized_policy = coerce_llm_capability_policy(payload)
    metadata = _dict_or_empty(normalized_policy.metadata)
    metadata.setdefault("capability_ref", _capability_name(capability_ref) or canonical_name)
    metadata.setdefault("catalog_name", canonical_name)
    metadata.setdefault("catalog_validated", spec is not None)
    if spec is not None:
        metadata.setdefault("catalog_namespace", spec.namespace)
    if namespace_hint:
        metadata.setdefault("capability_namespace_hint", namespace_hint)
    return LLMCapabilityPolicyEntry(
        capability=canonical_name,
        capability_ref=_capability_name(capability_ref) or canonical_name,
        namespace=str(spec.namespace if spec is not None else namespace_hint),
        route_name=normalized_policy.route_name,
        required_capabilities=list(normalized_policy.required_capabilities),
        schema_name=normalized_policy.schema_name,
        fallback_route=normalized_policy.fallback_route,
        metadata=metadata,
    )


@dataclass(frozen=True)
class LLMRouteBudget:
    request_budget: Optional[int] = None
    token_budget: Optional[int] = None
    cooldown_ticks: Optional[int] = None
    priority: str = ""
    runtime_tier: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }


def coerce_llm_route_budget(value: Any) -> LLMRouteBudget:
    if isinstance(value, LLMRouteBudget):
        return value
    if not isinstance(value, Mapping):
        return LLMRouteBudget()
    payload = dict(value)
    return LLMRouteBudget(
        request_budget=_optional_int(payload.get("request_budget")),
        token_budget=_optional_int(payload.get("token_budget")),
        cooldown_ticks=_optional_int(payload.get("cooldown_ticks")),
        priority=str(payload.get("priority", "") or ""),
        runtime_tier=str(payload.get("runtime_tier", "") or ""),
        metadata=_dict_or_empty(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class LLMRoutePolicy:
    served_routes: list[str] = field(default_factory=list)
    client_alias: str = ""
    provider: str = ""
    token_file: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    enabled_modes: list[str] = field(default_factory=list)
    disabled_modes: list[str] = field(default_factory=list)
    fallback_route: str = ""
    budget: LLMRouteBudget = field(default_factory=LLMRouteBudget)
    capability_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "served_routes": list(self.served_routes),
            "client_alias": self.client_alias,
            "provider": self.provider,
            "token_file": self.token_file,
            "base_url": self.base_url,
            "model": self.model,
            "enabled_modes": list(self.enabled_modes),
            "disabled_modes": list(self.disabled_modes),
            "fallback_route": self.fallback_route,
            "budget": self.budget.to_dict(),
            "capability_profile": dict(self.capability_profile),
            "metadata": dict(self.metadata),
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }

    def to_route_spec(self) -> Dict[str, Any]:
        return self.to_dict()


def coerce_llm_route_policy(value: Any) -> LLMRoutePolicy:
    if isinstance(value, LLMRoutePolicy):
        return value
    if not isinstance(value, Mapping):
        return LLMRoutePolicy()
    payload = dict(value)
    return LLMRoutePolicy(
        served_routes=_string_list(payload.get("served_routes", [])),
        client_alias=str(payload.get("client_alias", "") or ""),
        provider=str(payload.get("provider", "") or ""),
        token_file=payload.get("token_file") or None,
        base_url=payload.get("base_url") or None,
        model=payload.get("model") or None,
        enabled_modes=_string_list(payload.get("enabled_modes", [])),
        disabled_modes=_string_list(payload.get("disabled_modes", [])),
        fallback_route=str(payload.get("fallback_route", "") or ""),
        budget=coerce_llm_route_budget(payload.get("budget", {})),
        capability_profile=_dict_or_empty(payload.get("capability_profile", {})),
        metadata=_dict_or_empty(payload.get("metadata", {})),
    )


def resolve_llm_route_policies(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(route_name).strip(): coerce_llm_route_policy(policy).to_route_spec()
        for route_name, policy in dict(value).items()
        if str(route_name).strip()
    }


def _capability_policy_items(value: Any) -> list[tuple[Any, Any]]:
    if not isinstance(value, (Mapping, list, tuple)):
        return []
    if isinstance(value, Mapping):
        return list(dict(value).items())
    items: list[tuple[Any, Any]] = []
    for row in list(value or []):
        if isinstance(row, LLMCapabilityPolicyEntry):
            items.append((row.capability or row.capability_ref, row))
            continue
        if isinstance(row, Mapping):
            items.append(
                (
                    row.get("capability", "")
                    or row.get("capability_ref", "")
                    or row.get("capability_name", "")
                    or row.get("full_name", "")
                    or row.get("name", ""),
                    row,
                )
            )
    return items


def _resolved_llm_capability_policy_entries(value: Any) -> list[LLMCapabilityPolicyEntry]:
    raw_items = _capability_policy_items(value)
    if not raw_items:
        return []
    resolved: Dict[str, LLMCapabilityPolicyEntry] = {}
    for raw_key, raw_policy in raw_items:
        entry = coerce_llm_capability_policy_entry(
            raw_policy,
            fallback_capability=raw_key,
        )
        if not entry.capability:
            continue
        if entry.capability in resolved:
            del resolved[entry.capability]
        resolved[entry.capability] = entry
    return list(resolved.values())


def resolve_llm_capability_policy_entries(value: Any) -> list[Dict[str, Any]]:
    return [entry.to_dict() for entry in _resolved_llm_capability_policy_entries(value)]


def resolve_llm_capability_policies(value: Any) -> Dict[str, Dict[str, Any]]:
    return {
        entry.capability: entry.to_policy_dict()
        for entry in _resolved_llm_capability_policy_entries(value)
    }


def merge_llm_capability_specs(*layers: Any) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        normalized = resolve_llm_capability_policies(layer)
        for capability_name, spec in normalized.items():
            current = dict(merged.get(capability_name, {}) or {})
            incoming = dict(spec or {})
            current_metadata = _dict_or_empty(current.get("metadata", {}))
            incoming_metadata = _dict_or_empty(incoming.get("metadata", {}))
            merged_required_capabilities = _string_list(current.get("required_capabilities", [])) + _string_list(
                incoming.get("required_capabilities", [])
            )
            current.update(
                {
                    key: value
                    for key, value in incoming.items()
                    if key not in {"metadata", "required_capabilities"}
                }
            )
            if current_metadata or incoming_metadata:
                current["metadata"] = {**current_metadata, **incoming_metadata}
            if merged_required_capabilities:
                deduped: list[str] = []
                seen: set[str] = set()
                for item in merged_required_capabilities:
                    if item in seen:
                        continue
                    seen.add(item)
                    deduped.append(item)
                current["required_capabilities"] = deduped
            merged[capability_name] = current
    return merged


def merge_llm_route_specs(*layers: Any) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        normalized = resolve_llm_route_policies(layer)
        for route_name, spec in normalized.items():
            current = dict(merged.get(route_name, {}) or {})
            incoming = dict(spec or {})
            current_budget = _dict_or_empty(current.get("budget", {}))
            incoming_budget = _dict_or_empty(incoming.get("budget", {}))
            current_metadata = _dict_or_empty(current.get("metadata", {}))
            incoming_metadata = _dict_or_empty(incoming.get("metadata", {}))
            current_profile = _dict_or_empty(current.get("capability_profile", {}))
            incoming_profile = _dict_or_empty(incoming.get("capability_profile", {}))
            merged_served_routes = _string_list(current.get("served_routes", [])) + _string_list(incoming.get("served_routes", []))
            current.update(
                {
                    key: value
                    for key, value in incoming.items()
                    if key not in {"budget", "metadata", "capability_profile", "served_routes"}
                }
            )
            if current_budget or incoming_budget:
                current["budget"] = {**current_budget, **incoming_budget}
            if current_profile or incoming_profile:
                current["capability_profile"] = {**current_profile, **incoming_profile}
            if current_metadata or incoming_metadata:
                current["metadata"] = {**current_metadata, **incoming_metadata}
            if merged_served_routes:
                deduped: list[str] = []
                seen: set[str] = set()
                for item in merged_served_routes:
                    if item in seen:
                        continue
                    seen.add(item)
                    deduped.append(item)
                current["served_routes"] = deduped
            merged[route_name] = current
    return merged


@dataclass
class RuntimeBudgetConfig:
    """
    Budget policy for slow-path cognition inside CoreMainLoop.

    The fast path should stay available every tick. Expensive retrieval-side
    LLM calls are only allowed when a cheap trigger fires and cooldown permits.
    """

    retrieval_cooldown_ticks: int = 8
    hypothesis_augment_cooldown_ticks: int = 12
    llm_rerank_cooldown_ticks: int = 8
    force_retrieval_on_tick0: bool = True
    enable_llm_retrieval_gate: bool = False
    reward_stagnation_window: int = 4
    repeated_action_window: int = 3
    candidate_margin_threshold: float = 0.15
    retrieval_runtime_tier: str = "llm_assisted"
    # Prediction runtime switch (config-driven). Keep enabled by default so
    # candidate scoring can observe prediction influence unless explicitly disabled.
    enable_prediction: bool = True
    # Learning runtime switch. Keep enabled by default so outcome feedback can
    # persist and shape later candidate arbitration.
    enable_learning: bool = True
    # Budget/degrade control for heavy mechanism formal commits.
    # IMPORTANT: this does not disable mechanism matching experiment-wide;
    # that ownership belongs to CausalLayerAblationConfig.enable_mechanism_matching.
    enable_mechanism_formal_path: bool = True
    # Route-level model routing policy. This lets runtime policy choose
    # provider/model/budget per route instead of hard-coding them in code.
    llm_route_policies: Dict[str, Any] = field(default_factory=dict)
    # Capability-level routing policy. This lets runtime policy pin or refine
    # capability requests independently from module-owned route declarations.
    llm_capability_policies: Dict[str, Any] = field(default_factory=dict)

    def resolve_llm_route_specs(self) -> Dict[str, Dict[str, Any]]:
        return resolve_llm_route_policies(self.llm_route_policies)

    def resolve_llm_capability_specs(self) -> Dict[str, Dict[str, Any]]:
        return resolve_llm_capability_policies(self.llm_capability_policies)

    def resolve_llm_capability_policy_entries(self) -> list[Dict[str, Any]]:
        return resolve_llm_capability_policy_entries(self.llm_capability_policies)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["llm_route_policies"] = resolve_llm_route_policies(self.llm_route_policies)
        payload["llm_route_specs"] = self.resolve_llm_route_specs()
        payload["llm_capability_policies"] = resolve_llm_capability_policies(self.llm_capability_policies)
        payload["llm_capability_policy_entries"] = self.resolve_llm_capability_policy_entries()
        payload["llm_capability_specs"] = self.resolve_llm_capability_specs()
        return payload

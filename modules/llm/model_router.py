from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from modules.llm.factory import build_llm_client


def _normalized_mode(value: Any) -> str:
    return str(value or "integrated").strip().lower() or "integrated"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in dict(value).items()}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _score_from_tier(
    value: Any,
    *,
    default: float = 0.5,
    low_words: tuple[str, ...] = ("low", "slow", "expensive", "weak"),
    high_words: tuple[str, ...] = ("high", "fast", "cheap", "strong"),
) -> float:
    if isinstance(value, (int, float)):
        return _bounded_float(value, default=default)
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"very_low", "lowest", "minimal"} or any(word in text for word in low_words):
        return 0.2
    if text in {"medium", "normal", "balanced"}:
        return 0.5
    if text in {"very_high", "highest", "critical"} or any(word in text for word in high_words):
        return 0.85
    return default


def _efficiency_score(value: Any, *, default: float = 0.5) -> float:
    if isinstance(value, (int, float)):
        return _bounded_float(value, default=default)
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"cheap", "fast", "low", "low_cost", "low_latency"}:
        return 0.85
    if text in {"expensive", "slow", "high", "high_cost", "high_latency"}:
        return 0.2
    return _score_from_tier(text, default=default)


def _feedback_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(-1.0, min(1.0, float(value)))
    if not isinstance(value, Mapping):
        return 0.0
    payload = dict(value)
    if isinstance(payload.get("score"), (int, float)):
        return max(-1.0, min(1.0, float(payload.get("score", 0.0))))
    successes = int(max(0, payload.get("successes", 0) or 0))
    failures = int(max(0, payload.get("failures", 0) or 0))
    total = successes + failures
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(successes - failures) / float(total)))


@dataclass(frozen=True)
class ModelRouteCapabilityProfile:
    capabilities: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    cost_efficiency: float = 0.5
    latency_efficiency: float = 0.5
    uncertainty_tolerance: float = 0.5
    verification_strength: float = 0.5
    structured_output_strength: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRouteContext:
    required_capabilities: list[str] = field(default_factory=list)
    uncertainty_level: float = 0.0
    verification_pressure: float = 0.0
    prefer_low_cost: float = 0.0
    prefer_low_latency: float = 0.0
    prefer_high_trust: float = 0.0
    prefer_structured_output: float = 0.0
    route_feedback: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce_model_route_capability_profile(value: Any, *, fallback_route: str = "") -> ModelRouteCapabilityProfile:
    if isinstance(value, ModelRouteCapabilityProfile):
        return value
    payload = dict(value) if isinstance(value, Mapping) else {}
    route_defaults = _DEFAULT_CAPABILITY_PROFILES.get(str(fallback_route or "").strip(), {})
    capabilities = _string_list(payload.get("capabilities", route_defaults.get("capabilities", [])))
    return ModelRouteCapabilityProfile(
        capabilities=capabilities,
        trust_score=_score_from_tier(payload.get("trust_score", payload.get("trust", route_defaults.get("trust_score", 0.5))), default=route_defaults.get("trust_score", 0.5)),
        cost_efficiency=_efficiency_score(payload.get("cost_efficiency", payload.get("cost_tier", route_defaults.get("cost_efficiency", 0.5))), default=route_defaults.get("cost_efficiency", 0.5)),
        latency_efficiency=_efficiency_score(payload.get("latency_efficiency", payload.get("latency_tier", route_defaults.get("latency_efficiency", 0.5))), default=route_defaults.get("latency_efficiency", 0.5)),
        uncertainty_tolerance=_score_from_tier(payload.get("uncertainty_tolerance", payload.get("uncertainty_support", route_defaults.get("uncertainty_tolerance", 0.5))), default=route_defaults.get("uncertainty_tolerance", 0.5), low_words=("low", "weak"), high_words=("high", "strong", "robust")),
        verification_strength=_score_from_tier(payload.get("verification_strength", route_defaults.get("verification_strength", 0.5)), default=route_defaults.get("verification_strength", 0.5), low_words=("low", "weak"), high_words=("high", "strong", "robust")),
        structured_output_strength=_score_from_tier(payload.get("structured_output_strength", route_defaults.get("structured_output_strength", 0.5)), default=route_defaults.get("structured_output_strength", 0.5), low_words=("low", "weak"), high_words=("high", "strong", "reliable")),
        metadata=_dict_or_empty(payload.get("metadata", route_defaults.get("metadata", {}))),
    )


def coerce_model_route_context(value: Any) -> ModelRouteContext:
    if isinstance(value, ModelRouteContext):
        return value
    payload = dict(value) if isinstance(value, Mapping) else {}
    return ModelRouteContext(
        required_capabilities=_string_list(payload.get("required_capabilities", [])),
        uncertainty_level=_bounded_float(payload.get("uncertainty_level", 0.0), default=0.0),
        verification_pressure=_bounded_float(payload.get("verification_pressure", 0.0), default=0.0),
        prefer_low_cost=_bounded_float(payload.get("prefer_low_cost", 0.0), default=0.0),
        prefer_low_latency=_bounded_float(payload.get("prefer_low_latency", 0.0), default=0.0),
        prefer_high_trust=_bounded_float(payload.get("prefer_high_trust", 0.0), default=0.0),
        prefer_structured_output=_bounded_float(payload.get("prefer_structured_output", 0.0), default=0.0),
        route_feedback=_dict_or_empty(payload.get("route_feedback", {})),
        metadata=_dict_or_empty(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class ModelRouteSpec:
    served_routes: list[str] = field(default_factory=list)
    client_alias: str = ""
    provider: str = ""
    token_file: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    enabled_modes: list[str] = field(default_factory=list)
    disabled_modes: list[str] = field(default_factory=list)
    fallback_route: str = ""
    budget: Dict[str, Any] = field(default_factory=dict)
    capability_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRouteDecision:
    route_name: str
    mode: str
    requested_route: str = ""
    client: Any = None
    source: str = ""
    disabled_reason: str = ""
    explanation: list[str] = field(default_factory=list)
    score: float = 0.0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    candidate_routes: list[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["client"] = "<client>" if self.client is not None else None
        return payload


_GENERAL_DISABLED_MODES = ["shadow", "analyst", "final_candidate", "candidate_only_final"]

_DEFAULT_ROUTE_SPECS: Dict[str, ModelRouteSpec] = {
    "general": ModelRouteSpec(
        client_alias="default",
        disabled_modes=list(_GENERAL_DISABLED_MODES),
        metadata={"route_group": "general"},
    ),
    "deliberation": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "deliberation"},
    ),
    "retrieval": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "retrieval"},
    ),
    "hypothesis": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "hypothesis"},
    ),
    "probe": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "probe"},
    ),
    "skill": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "skill"},
    ),
    "recovery": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "recovery"},
    ),
    "representation": ModelRouteSpec(
        fallback_route="general",
        metadata={"route_group": "general", "role": "representation"},
    ),
    "structured_answer": ModelRouteSpec(
        client_alias="default",
        disabled_modes=["shadow", "analyst"],
        metadata={"route_group": "structured_answer"},
    ),
    "shadow": ModelRouteSpec(
        client_alias="shadow",
        enabled_modes=["shadow"],
        metadata={"route_group": "shadow"},
    ),
    "analyst": ModelRouteSpec(
        client_alias="analyst",
        enabled_modes=["analyst"],
        metadata={"route_group": "analyst"},
    ),
}

_DEFAULT_CAPABILITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "general": {
        "capabilities": ["reasoning", "general"],
        "trust_score": 0.55,
        "cost_efficiency": 0.55,
        "latency_efficiency": 0.55,
        "uncertainty_tolerance": 0.55,
        "verification_strength": 0.45,
        "structured_output_strength": 0.5,
    },
    "deliberation": {
        "capabilities": ["reasoning", "planning"],
        "trust_score": 0.62,
        "cost_efficiency": 0.45,
        "latency_efficiency": 0.4,
        "uncertainty_tolerance": 0.76,
        "verification_strength": 0.58,
        "structured_output_strength": 0.55,
    },
    "retrieval": {
        "capabilities": ["retrieval", "grounding"],
        "trust_score": 0.6,
        "cost_efficiency": 0.72,
        "latency_efficiency": 0.78,
        "uncertainty_tolerance": 0.55,
        "verification_strength": 0.48,
        "structured_output_strength": 0.42,
    },
    "hypothesis": {
        "capabilities": ["reasoning", "uncertainty", "hypothesis"],
        "trust_score": 0.64,
        "cost_efficiency": 0.42,
        "latency_efficiency": 0.38,
        "uncertainty_tolerance": 0.82,
        "verification_strength": 0.52,
        "structured_output_strength": 0.45,
    },
    "probe": {
        "capabilities": ["verification", "reasoning", "probe"],
        "trust_score": 0.72,
        "cost_efficiency": 0.5,
        "latency_efficiency": 0.52,
        "uncertainty_tolerance": 0.78,
        "verification_strength": 0.88,
        "structured_output_strength": 0.58,
    },
    "skill": {
        "capabilities": ["instruction_following", "rewriting"],
        "trust_score": 0.56,
        "cost_efficiency": 0.62,
        "latency_efficiency": 0.68,
        "uncertainty_tolerance": 0.4,
        "verification_strength": 0.32,
        "structured_output_strength": 0.44,
    },
    "recovery": {
        "capabilities": ["recovery", "reasoning"],
        "trust_score": 0.76,
        "cost_efficiency": 0.46,
        "latency_efficiency": 0.44,
        "uncertainty_tolerance": 0.72,
        "verification_strength": 0.64,
        "structured_output_strength": 0.48,
    },
    "representation": {
        "capabilities": ["representation", "creative"],
        "trust_score": 0.5,
        "cost_efficiency": 0.48,
        "latency_efficiency": 0.46,
        "uncertainty_tolerance": 0.44,
        "verification_strength": 0.28,
        "structured_output_strength": 0.38,
    },
    "structured_answer": {
        "capabilities": ["structured_output", "reasoning"],
        "trust_score": 0.78,
        "cost_efficiency": 0.42,
        "latency_efficiency": 0.4,
        "uncertainty_tolerance": 0.66,
        "verification_strength": 0.72,
        "structured_output_strength": 0.94,
    },
    "shadow": {
        "capabilities": ["analysis", "reasoning"],
        "trust_score": 0.7,
        "cost_efficiency": 0.4,
        "latency_efficiency": 0.34,
        "uncertainty_tolerance": 0.76,
        "verification_strength": 0.54,
        "structured_output_strength": 0.44,
    },
    "analyst": {
        "capabilities": ["analysis", "verification", "reasoning"],
        "trust_score": 0.84,
        "cost_efficiency": 0.38,
        "latency_efficiency": 0.32,
        "uncertainty_tolerance": 0.88,
        "verification_strength": 0.92,
        "structured_output_strength": 0.62,
    },
}


def coerce_model_route_spec(value: Any) -> ModelRouteSpec:
    if isinstance(value, ModelRouteSpec):
        return value
    if not isinstance(value, dict):
        return ModelRouteSpec()
    return ModelRouteSpec(
        served_routes=_string_list(value.get("served_routes", [])),
        client_alias=str(value.get("client_alias", "") or ""),
        provider=str(value.get("provider", "") or ""),
        token_file=value.get("token_file"),
        base_url=value.get("base_url"),
        model=value.get("model"),
        enabled_modes=_string_list(value.get("enabled_modes", [])),
        disabled_modes=_string_list(value.get("disabled_modes", [])),
        fallback_route=str(value.get("fallback_route", "") or ""),
        budget=dict(value.get("budget", {}) or {}) if isinstance(value.get("budget", {}), dict) else {},
        capability_profile=dict(value.get("capability_profile", {}) or {}) if isinstance(value.get("capability_profile", {}), dict) else {},
        metadata=dict(value.get("metadata", {}) or {}) if isinstance(value.get("metadata", {}), dict) else {},
    )


class ModelRouter:
    def __init__(
        self,
        *,
        default_client: Any = None,
        shadow_client: Any = None,
        analyst_client: Any = None,
        llm_mode: str = "integrated",
        route_specs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._mode = _normalized_mode(llm_mode)
        self._clients: Dict[str, Any] = {}
        self._route_specs: Dict[str, ModelRouteSpec] = {}
        self._client_cache: Dict[tuple[Any, ...], Any] = {}
        self.register_client("default", default_client)
        self.register_client("shadow", shadow_client)
        self.register_client("analyst", analyst_client)
        self.set_route_specs(route_specs or {})

    def register_client(self, alias: str, client: Any) -> None:
        key = str(alias or "").strip().lower()
        if not key:
            return
        self._clients[key] = client

    def set_mode(self, llm_mode: str) -> None:
        self._mode = _normalized_mode(llm_mode)

    def set_route_specs(self, route_specs: Mapping[str, Any]) -> None:
        self._route_specs = {
            str(name or "").strip(): coerce_model_route_spec(spec)
            for name, spec in dict(route_specs or {}).items()
            if str(name or "").strip()
        }

    def resolve(self, route_name: str = "general", context: Optional[Mapping[str, Any]] = None) -> Any:
        return self.decide(route_name, context=context).client

    def decide(self, route_name: str = "general", context: Optional[Mapping[str, Any]] = None) -> ModelRouteDecision:
        requested_route = str(route_name or "general").strip() or "general"
        route_context = coerce_model_route_context(context)
        candidate_names = self._candidate_route_names_for(requested_route)
        ranked_candidates: list[tuple[float, int, ModelRouteDecision, Dict[str, Any]]] = []
        seen_candidate_keys: set[tuple[str, str]] = set()
        for index, candidate_name in enumerate(candidate_names):
            candidate = self._decide_single(candidate_name, requested_route=requested_route, seen_routes=set())
            if candidate.client is None:
                continue
            candidate_key = (
                str(candidate.route_name or candidate_name).strip() or candidate_name,
                str(candidate.source or ""),
            )
            if candidate_key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(candidate_key)
            scored = self._score_candidate(
                requested_route=requested_route,
                decision=candidate,
                context=route_context,
            )
            ranked_candidates.append((float(scored["total_score"]), -index, candidate, scored))
        if not ranked_candidates:
            unresolved = self._decide_single(requested_route, requested_route=requested_route, seen_routes=set())
            return self._decorate_decision(
                decision=unresolved,
                requested_route=requested_route,
                context=route_context,
                scored=None,
                candidate_payloads=[],
            )
        ranked_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        winning_candidate = ranked_candidates[0][2]
        winning_score = ranked_candidates[0][3]
        candidate_payloads = [
            self._candidate_payload(candidate, scored)
            for _, _, candidate, scored in ranked_candidates
        ]
        return self._decorate_decision(
            decision=winning_candidate,
            requested_route=requested_route,
            context=route_context,
            scored=winning_score,
            candidate_payloads=candidate_payloads,
        )

    def describe_route(self, route_name: str = "general", context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        decision = self.decide(route_name, context=context)
        return decision.to_dict()

    def _spec_for(self, route_name: str) -> ModelRouteSpec:
        if route_name in self._route_specs:
            return self._route_specs[route_name]
        if route_name in _DEFAULT_ROUTE_SPECS:
            return _DEFAULT_ROUTE_SPECS[route_name]
        return _DEFAULT_ROUTE_SPECS["general"]

    def _all_route_specs(self) -> Dict[str, ModelRouteSpec]:
        combined = dict(_DEFAULT_ROUTE_SPECS)
        combined.update(self._route_specs)
        return combined

    def _candidate_route_names_for(self, requested_route: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        requested = str(requested_route or "general").strip() or "general"
        for candidate_name in [requested, *list(self._all_route_specs().keys())]:
            clean = str(candidate_name or "").strip()
            if not clean or clean in seen:
                continue
            spec = self._spec_for(clean)
            served_routes = {str(item or "").strip() for item in list(spec.served_routes or []) if str(item or "").strip()}
            if clean != requested and requested not in served_routes:
                continue
            seen.add(clean)
            candidates.append(clean)
        fallback_route = str(self._spec_for(requested).fallback_route or "").strip()
        if fallback_route and fallback_route not in seen:
            candidates.append(fallback_route)
        return candidates or [requested]

    def _decide_single(self, route_name: str, *, requested_route: str, seen_routes: set[str]) -> ModelRouteDecision:
        if route_name in seen_routes:
            return ModelRouteDecision(
                requested_route=requested_route,
                route_name=route_name,
                mode=self._mode,
                client=None,
                source="router",
                disabled_reason="route_cycle",
            )
        seen_routes.add(route_name)
        spec = self._spec_for(route_name)
        enabled_modes = {str(item).strip().lower() for item in list(spec.enabled_modes or []) if str(item).strip()}
        disabled_modes = {str(item).strip().lower() for item in list(spec.disabled_modes or []) if str(item).strip()}
        if enabled_modes and self._mode not in enabled_modes:
            return ModelRouteDecision(
                requested_route=requested_route,
                route_name=route_name,
                mode=self._mode,
                client=None,
                source="route_spec",
                disabled_reason=f"mode_not_enabled:{self._mode}",
                metadata=spec.to_dict(),
            )
        if self._mode in disabled_modes:
            return ModelRouteDecision(
                requested_route=requested_route,
                route_name=route_name,
                mode=self._mode,
                client=None,
                source="route_spec",
                disabled_reason=f"mode_disabled:{self._mode}",
                metadata=spec.to_dict(),
            )

        client = None
        source = ""
        if spec.client_alias:
            client = self._clients.get(str(spec.client_alias).strip().lower())
            source = f"alias:{str(spec.client_alias).strip().lower()}"
        elif spec.provider:
            client = self._build_cached_client(spec)
            source = f"provider:{spec.provider}"

        if client is None and spec.fallback_route and spec.fallback_route != route_name:
            return self._decide_single(
                spec.fallback_route,
                requested_route=requested_route,
                seen_routes=seen_routes,
            )

        return ModelRouteDecision(
            requested_route=requested_route,
            route_name=route_name,
            mode=self._mode,
            client=client,
            source=source or "unresolved",
            disabled_reason="" if client is not None else "no_client_resolved",
            metadata=spec.to_dict(),
        )

    def _capability_profile_for(self, route_name: str, metadata: Optional[Dict[str, Any]]) -> ModelRouteCapabilityProfile:
        payload = {}
        if isinstance(metadata, dict):
            payload = dict(metadata.get("capability_profile", {}) or {})
        return coerce_model_route_capability_profile(payload, fallback_route=route_name)

    def _score_candidate(
        self,
        *,
        requested_route: str,
        decision: ModelRouteDecision,
        context: ModelRouteContext,
    ) -> Dict[str, Any]:
        profile = self._capability_profile_for(decision.route_name, decision.metadata)
        required_capabilities = [item for item in list(context.required_capabilities or []) if item]
        matched_capabilities = [item for item in required_capabilities if item in set(profile.capabilities)]
        capability_coverage = (
            float(len(matched_capabilities)) / float(len(required_capabilities))
            if required_capabilities
            else 1.0
        )
        capability_bonus = (0.42 * capability_coverage) - (0.25 if required_capabilities and not matched_capabilities else 0.0)
        uncertainty_bonus = float(context.uncertainty_level) * float(profile.uncertainty_tolerance) * 0.34
        verification_bonus = float(context.verification_pressure) * float(profile.verification_strength) * 0.34
        trust_bonus = float(context.prefer_high_trust) * float(profile.trust_score) * 0.28
        cost_bonus = float(context.prefer_low_cost) * float(profile.cost_efficiency) * 0.18
        latency_bonus = float(context.prefer_low_latency) * float(profile.latency_efficiency) * 0.18
        structured_bonus = float(context.prefer_structured_output) * float(profile.structured_output_strength) * 0.22
        feedback_entry = context.route_feedback.get(decision.route_name)
        if feedback_entry is None and str(decision.route_name or "") == str(requested_route or ""):
            feedback_entry = context.route_feedback.get(requested_route)
        if feedback_entry is None:
            feedback_entry = context.route_feedback.get("*")
        feedback_value = _feedback_score(feedback_entry)
        feedback_bonus = feedback_value * 0.26
        request_bonus = 0.06 if str(decision.route_name or "") == str(requested_route or "") else 0.0
        total_score = (
            request_bonus
            + capability_bonus
            + uncertainty_bonus
            + verification_bonus
            + trust_bonus
            + cost_bonus
            + latency_bonus
            + structured_bonus
            + feedback_bonus
        )
        explanation: list[str] = []
        if matched_capabilities:
            explanation.append(f"matched capabilities: {', '.join(matched_capabilities)}")
        if float(context.verification_pressure) >= 0.4 and float(profile.verification_strength) >= 0.6:
            explanation.append(
                f"verification pressure favored strength={profile.verification_strength:.2f}"
            )
        if float(context.uncertainty_level) >= 0.4 and float(profile.uncertainty_tolerance) >= 0.6:
            explanation.append(
                f"uncertainty favored tolerance={profile.uncertainty_tolerance:.2f}"
            )
        if float(context.prefer_high_trust) >= 0.4 and float(profile.trust_score) >= 0.6:
            explanation.append(f"trust preference favored trust={profile.trust_score:.2f}")
        if float(context.prefer_low_latency) >= 0.4 and float(profile.latency_efficiency) >= 0.6:
            explanation.append(f"latency preference favored latency={profile.latency_efficiency:.2f}")
        if float(context.prefer_low_cost) >= 0.4 and float(profile.cost_efficiency) >= 0.6:
            explanation.append(f"cost preference favored cost={profile.cost_efficiency:.2f}")
        if float(context.prefer_structured_output) >= 0.4 and float(profile.structured_output_strength) >= 0.6:
            explanation.append(
                f"structured output favored strength={profile.structured_output_strength:.2f}"
            )
        if abs(feedback_value) >= 0.2:
            label = "positive" if feedback_value > 0.0 else "negative"
            explanation.append(f"{label} feedback adjusted score={feedback_value:.2f}")
        if not explanation:
            explanation.append("selected by baseline route preference")
        return {
            "profile": profile.to_dict(),
            "required_capabilities": required_capabilities,
            "matched_capabilities": matched_capabilities,
            "total_score": round(total_score, 4),
            "score_breakdown": {
                "request_bonus": round(request_bonus, 4),
                "capability_bonus": round(capability_bonus, 4),
                "uncertainty_bonus": round(uncertainty_bonus, 4),
                "verification_bonus": round(verification_bonus, 4),
                "trust_bonus": round(trust_bonus, 4),
                "cost_bonus": round(cost_bonus, 4),
                "latency_bonus": round(latency_bonus, 4),
                "structured_bonus": round(structured_bonus, 4),
                "feedback_bonus": round(feedback_bonus, 4),
            },
            "feedback_score": round(feedback_value, 4),
            "explanation": explanation,
        }

    def _candidate_payload(self, decision: ModelRouteDecision, scored: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "route_name": str(decision.route_name or ""),
            "source": str(decision.source or ""),
            "score": float(scored.get("total_score", 0.0) or 0.0),
            "score_breakdown": dict(scored.get("score_breakdown", {}) or {}),
            "required_capabilities": list(scored.get("required_capabilities", []) or []),
            "matched_capabilities": list(scored.get("matched_capabilities", []) or []),
            "capability_profile": dict(scored.get("profile", {}) or {}),
            "explanation": list(scored.get("explanation", []) or []),
        }

    def _decorate_decision(
        self,
        *,
        decision: ModelRouteDecision,
        requested_route: str,
        context: ModelRouteContext,
        scored: Optional[Dict[str, Any]],
        candidate_payloads: list[Dict[str, Any]],
    ) -> ModelRouteDecision:
        metadata = dict(decision.metadata or {})
        metadata.setdefault("requested_route", requested_route)
        metadata.setdefault("selected_route", decision.route_name)
        metadata["route_context"] = context.to_dict()
        if scored is not None:
            metadata["decision_explanation"] = list(scored.get("explanation", []) or [])
            metadata["score_breakdown"] = dict(scored.get("score_breakdown", {}) or {})
            metadata["capability_profile"] = dict(scored.get("profile", {}) or {})
            metadata["matched_capabilities"] = list(scored.get("matched_capabilities", []) or [])
            metadata["required_capabilities"] = list(scored.get("required_capabilities", []) or [])
            metadata["feedback_score"] = float(scored.get("feedback_score", 0.0) or 0.0)
        if candidate_payloads:
            metadata["candidate_routes"] = list(candidate_payloads)
        return ModelRouteDecision(
            requested_route=requested_route,
            route_name=decision.route_name,
            mode=decision.mode,
            client=decision.client,
            source=decision.source,
            disabled_reason=decision.disabled_reason,
            explanation=list((scored or {}).get("explanation", []) or []),
            score=float((scored or {}).get("total_score", 0.0) or 0.0),
            score_breakdown=dict((scored or {}).get("score_breakdown", {}) or {}),
            candidate_routes=list(candidate_payloads),
            metadata=metadata,
        )

    def _build_cached_client(self, spec: ModelRouteSpec) -> Any:
        cache_key = (
            str(spec.provider or "").strip().lower(),
            spec.token_file,
            spec.base_url,
            spec.model,
        )
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = build_llm_client(
                provider=spec.provider,
                token_file=spec.token_file,
                base_url=spec.base_url,
                model=spec.model,
            )
        return self._client_cache[cache_key]

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence


AGENT_CONTROL_PLANE_VERSION = "conos.agent_control_plane/v1"

RISK_LEVELS = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

DEFAULT_LLM_PERMISSIONS = (
    "read_context",
    "generate_text",
    "propose_text",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _now() -> float:
    return float(time.time())


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        return []
    result: list[str] = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _risk_value(value: Any) -> int:
    text = str(value or "low").strip().lower()
    return int(RISK_LEVELS.get(text, RISK_LEVELS["low"]))


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    display_name: str = ""
    agent_kind: str = "llm"
    provider: str = ""
    model: str = ""
    base_url: str = ""
    served_routes: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    allowed_permissions: list[str] = field(default_factory=list)
    approval_required_permissions: list[str] = field(default_factory=list)
    denied_permissions: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    cost_efficiency: float = 0.5
    latency_efficiency: float = 0.5
    max_risk_level: str = "low"
    status: str = "available"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentControlRequest:
    task_type: str
    route_name: str = ""
    capability_request: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    permissions_required: list[str] = field(default_factory=list)
    risk_level: str = "low"
    prefer_low_cost: float = 0.0
    prefer_low_latency: float = 0.0
    prefer_high_trust: float = 0.0
    prefer_local: float = 0.0
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentControlDecision:
    status: str
    selected_agent_id: str = ""
    selected_agent: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    blocked_reason: str = ""
    approval_required: bool = False
    approval_permissions: list[str] = field(default_factory=list)
    candidates: list[Dict[str, Any]] = field(default_factory=list)
    audit_event: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce_agent_spec(value: Any) -> AgentSpec:
    if isinstance(value, AgentSpec):
        return value
    payload = _dict_or_empty(value)
    agent_id = str(
        payload.get("agent_id")
        or payload.get("id")
        or payload.get("route_policy")
        or payload.get("name")
        or ""
    ).strip()
    if not agent_id:
        agent_id = f"agent_{_hash_payload(payload)[:12]}"
    capability_profile = _dict_or_empty(payload.get("capability_profile", {}))
    metadata = _dict_or_empty(payload.get("metadata", {}))
    allowed_permissions = _string_list(payload.get("allowed_permissions"))
    if not allowed_permissions and str(payload.get("agent_kind", "llm") or "llm") == "llm":
        allowed_permissions = list(DEFAULT_LLM_PERMISSIONS)
    return AgentSpec(
        agent_id=agent_id,
        display_name=str(payload.get("display_name") or payload.get("name") or agent_id),
        agent_kind=str(payload.get("agent_kind") or payload.get("kind") or "llm"),
        provider=str(payload.get("provider") or ""),
        model=str(payload.get("model") or ""),
        base_url=str(payload.get("base_url") or ""),
        served_routes=_string_list(payload.get("served_routes")),
        capabilities=_string_list(payload.get("capabilities") or capability_profile.get("capabilities")),
        allowed_permissions=allowed_permissions,
        approval_required_permissions=_string_list(payload.get("approval_required_permissions")),
        denied_permissions=_string_list(payload.get("denied_permissions")),
        trust_score=_clamp01(payload.get("trust_score", capability_profile.get("trust_score", 0.5)), 0.5),
        cost_efficiency=_clamp01(payload.get("cost_efficiency", capability_profile.get("cost_efficiency", 0.5)), 0.5),
        latency_efficiency=_clamp01(payload.get("latency_efficiency", capability_profile.get("latency_efficiency", 0.5)), 0.5),
        max_risk_level=str(payload.get("max_risk_level") or "low"),
        status=str(payload.get("status") or "available"),
        metadata={**metadata, "capability_profile": capability_profile} if capability_profile else metadata,
    )


def _extra_llm_permissions(capabilities: Sequence[str]) -> list[str]:
    caps = set(_string_list(capabilities))
    permissions = list(DEFAULT_LLM_PERMISSIONS)
    if "structured_output" in caps:
        permissions.append("structured_output")
    if "coding" in caps:
        permissions.append("propose_patch")
    if "tool_use" in caps:
        permissions.append("draft_tool_call")
    return permissions


def agent_specs_from_model_route_policies(policies: Mapping[str, Any]) -> list[AgentSpec]:
    agents: list[AgentSpec] = []
    for policy_name, raw_policy in dict(policies or {}).items():
        if not isinstance(raw_policy, Mapping):
            continue
        policy = dict(raw_policy)
        profile = _dict_or_empty(policy.get("capability_profile", {}))
        capabilities = _string_list(profile.get("capabilities"))
        disabled_reason = str(_dict_or_empty(policy.get("metadata", {})).get("disabled_reason", "") or "")
        status = "available" if _string_list(policy.get("served_routes")) and not disabled_reason else "deprioritized"
        agents.append(
            AgentSpec(
                agent_id=str(policy_name),
                display_name=str(policy.get("model") or policy_name),
                agent_kind="llm",
                provider=str(policy.get("provider") or ""),
                model=str(policy.get("model") or ""),
                base_url=str(policy.get("base_url") or ""),
                served_routes=_string_list(policy.get("served_routes")),
                capabilities=capabilities,
                allowed_permissions=_extra_llm_permissions(capabilities),
                trust_score=_clamp01(profile.get("trust_score"), 0.5),
                cost_efficiency=_clamp01(profile.get("cost_efficiency"), 0.5),
                latency_efficiency=_clamp01(profile.get("latency_efficiency"), 0.5),
                max_risk_level="low",
                status=status,
                metadata={
                    "source": "llm_route_policy",
                    "route_policy": str(policy_name),
                    "policy_metadata": _dict_or_empty(policy.get("metadata", {})),
                    "capability_profile": profile,
                },
            )
        )
    return agents


def load_agent_registry(path: str | Path | None = None) -> list[AgentSpec]:
    if not path:
        return []
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    raw_agents = payload.get("agents", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(raw_agents, list):
        return []
    return [coerce_agent_spec(row) for row in raw_agents if isinstance(row, Mapping)]


def write_agent_registry(agents: Sequence[AgentSpec | Mapping[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": AGENT_CONTROL_PLANE_VERSION,
        "agents": [
            coerce_agent_spec(agent).to_dict()
            for agent in list(agents or [])
        ],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    return output


class AgentControlPlane:
    def __init__(self, agents: Sequence[AgentSpec | Mapping[str, Any]] = ()) -> None:
        self._agents: Dict[str, AgentSpec] = {}
        for agent in list(agents or []):
            self.register(agent)

    def register(self, agent: AgentSpec | Mapping[str, Any]) -> None:
        spec = coerce_agent_spec(agent)
        self._agents[spec.agent_id] = spec

    def list_agents(self) -> list[Dict[str, Any]]:
        return [agent.to_dict() for agent in self._agents.values()]

    def decide(self, request: AgentControlRequest | Mapping[str, Any]) -> AgentControlDecision:
        req = request if isinstance(request, AgentControlRequest) else self._coerce_request(request)
        candidates: list[Dict[str, Any]] = []
        for agent in self._agents.values():
            candidate = self._score_agent(agent, req)
            candidates.append(candidate)
        eligible = [row for row in candidates if row["eligible"]]
        eligible.sort(key=lambda row: (float(row["score"]), str(row["agent_id"])), reverse=True)
        if not eligible:
            return self._decision(
                req,
                status="BLOCKED",
                candidates=candidates,
                blocked_reason="no_eligible_agent",
            )
        selected = eligible[0]
        approval_permissions = list(selected.get("approval_permissions", []) or [])
        status = "WAITING_APPROVAL" if approval_permissions else "SELECTED"
        agent = self._agents[str(selected["agent_id"])]
        return self._decision(
            req,
            status=status,
            selected_agent=agent,
            candidates=candidates,
            score=float(selected.get("score", 0.0) or 0.0),
            approval_permissions=approval_permissions,
        )

    def _coerce_request(self, value: Mapping[str, Any]) -> AgentControlRequest:
        payload = dict(value or {})
        return AgentControlRequest(
            request_id=str(payload.get("request_id") or ""),
            task_type=str(payload.get("task_type") or payload.get("task") or "general"),
            route_name=str(payload.get("route_name") or payload.get("route") or ""),
            capability_request=str(payload.get("capability_request") or payload.get("capability") or ""),
            required_capabilities=_string_list(payload.get("required_capabilities")),
            permissions_required=_string_list(payload.get("permissions_required") or payload.get("permissions")),
            risk_level=str(payload.get("risk_level") or "low"),
            prefer_low_cost=_clamp01(payload.get("prefer_low_cost"), 0.0),
            prefer_low_latency=_clamp01(payload.get("prefer_low_latency"), 0.0),
            prefer_high_trust=_clamp01(payload.get("prefer_high_trust"), 0.0),
            prefer_local=_clamp01(payload.get("prefer_local"), 0.0),
            metadata=_dict_or_empty(payload.get("metadata", {})),
        )

    def _score_agent(self, agent: AgentSpec, request: AgentControlRequest) -> Dict[str, Any]:
        reasons: list[str] = []
        blocked_reasons: list[str] = []
        status = str(agent.status or "available").lower()
        required = set(_string_list(request.required_capabilities))
        capabilities = set(agent.capabilities)
        missing_capabilities = sorted(required - capabilities)
        required_permissions = set(_string_list(request.permissions_required))
        allowed_permissions = set(agent.allowed_permissions)
        approval_permissions = sorted(required_permissions & set(agent.approval_required_permissions))
        denied_permissions = sorted(required_permissions & set(agent.denied_permissions))
        missing_permissions = sorted(required_permissions - allowed_permissions - set(agent.approval_required_permissions))
        route_name = str(request.route_name or "").strip()
        served_routes = set(agent.served_routes)
        if status not in {"available", "ready", "enabled"}:
            blocked_reasons.append(f"status:{agent.status}")
        if route_name and served_routes and route_name not in served_routes:
            blocked_reasons.append("route_not_served")
        if missing_capabilities:
            blocked_reasons.append("missing_capabilities:" + ",".join(missing_capabilities))
        if denied_permissions:
            blocked_reasons.append("denied_permissions:" + ",".join(denied_permissions))
        if missing_permissions:
            blocked_reasons.append("missing_permissions:" + ",".join(missing_permissions))
        if _risk_value(request.risk_level) > _risk_value(agent.max_risk_level):
            blocked_reasons.append(f"risk_exceeds_max:{request.risk_level}>{agent.max_risk_level}")
        capability_score = 1.0 if not required else (len(required & capabilities) / max(1, len(required)))
        route_bonus = 0.08 if route_name and (not served_routes or route_name in served_routes) else 0.0
        local_bonus = 0.0
        if str(agent.provider).lower() in {"ollama", "local", "codex", "browser", "ci", "vm"}:
            local_bonus = float(request.prefer_local) * 0.12
        score = (
            capability_score * 0.42
            + float(agent.trust_score) * (0.18 + 0.18 * float(request.prefer_high_trust))
            + float(agent.cost_efficiency) * 0.14 * float(request.prefer_low_cost)
            + float(agent.latency_efficiency) * 0.14 * float(request.prefer_low_latency)
            + route_bonus
            + local_bonus
        )
        if approval_permissions:
            score -= 0.05
            reasons.append("selected path requires approval")
        if required & capabilities:
            reasons.append("matched capabilities: " + ", ".join(sorted(required & capabilities)))
        if route_bonus:
            reasons.append(f"serves route: {route_name}")
        if not reasons:
            reasons.append("baseline candidate")
        eligible = not blocked_reasons
        return {
            "agent_id": agent.agent_id,
            "agent_kind": agent.agent_kind,
            "provider": agent.provider,
            "model": agent.model,
            "eligible": bool(eligible),
            "score": round(max(0.0, float(score)), 6) if eligible else 0.0,
            "blocked_reasons": blocked_reasons,
            "approval_permissions": approval_permissions,
            "matched_capabilities": sorted(required & capabilities),
            "reasons": reasons,
        }

    def _decision(
        self,
        request: AgentControlRequest,
        *,
        status: str,
        candidates: Sequence[Mapping[str, Any]],
        selected_agent: Optional[AgentSpec] = None,
        score: float = 0.0,
        blocked_reason: str = "",
        approval_permissions: Sequence[str] = (),
    ) -> AgentControlDecision:
        request_payload = request.to_dict()
        request_id = request.request_id or f"agent_req_{_hash_payload(request_payload)[:16]}"
        selected_payload = selected_agent.to_dict() if selected_agent is not None else {}
        audit_payload = {
            "schema_version": AGENT_CONTROL_PLANE_VERSION,
            "event_type": "agent_control_decision",
            "request_id": request_id,
            "status": status,
            "selected_agent_id": str(selected_agent.agent_id if selected_agent else ""),
            "blocked_reason": str(blocked_reason or ""),
            "approval_required": bool(approval_permissions),
            "approval_permissions": list(approval_permissions or []),
            "task_type": request.task_type,
            "route_name": request.route_name,
            "capability_request": request.capability_request,
            "required_capabilities": list(request.required_capabilities or []),
            "permissions_required": list(request.permissions_required or []),
            "risk_level": request.risk_level,
            "candidate_count": len(list(candidates or [])),
            "created_at": _now(),
        }
        audit_payload["audit_event_id"] = f"agent_audit_{_hash_payload(audit_payload)[:20]}"
        return AgentControlDecision(
            status=status,
            selected_agent_id=str(selected_agent.agent_id if selected_agent else ""),
            selected_agent=selected_payload,
            score=round(float(score), 6),
            blocked_reason=str(blocked_reason or ""),
            approval_required=bool(approval_permissions),
            approval_permissions=list(approval_permissions or []),
            candidates=[dict(row) for row in list(candidates or [])],
            audit_event=audit_payload,
            request={**request_payload, "request_id": request_id},
        )


def render_agent_control_decision(decision: Mapping[str, Any] | AgentControlDecision) -> str:
    payload = decision.to_dict() if isinstance(decision, AgentControlDecision) else dict(decision or {})
    lines = [
        "Con OS agent control decision",
        f"schema: {AGENT_CONTROL_PLANE_VERSION}",
        f"status: {payload.get('status', '')}",
        f"selected_agent: {payload.get('selected_agent_id', '') or '-'}",
    ]
    if payload.get("blocked_reason"):
        lines.append(f"blocked_reason: {payload.get('blocked_reason')}")
    if payload.get("approval_required"):
        lines.append("approval_required: " + ", ".join(list(payload.get("approval_permissions", []) or [])))
    candidates = [row for row in list(payload.get("candidates", []) or []) if isinstance(row, Mapping)]
    if candidates:
        lines.extend(["", f"{'agent':<28} {'kind':<14} {'score':>7} status"])
        lines.append("-" * 72)
        for row in candidates:
            status = "eligible" if row.get("eligible") else ";".join(list(row.get("blocked_reasons", []) or []))
            lines.append(
                f"{str(row.get('agent_id', '') or ''):<28} "
                f"{str(row.get('agent_kind', '') or ''):<14} "
                f"{float(row.get('score', 0.0) or 0.0):>7.3f} "
                f"{status}"
            )
    return "\n".join(lines)

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional


def _payload_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    if is_dataclass(value):
        payload = asdict(value)
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): raw for key, raw in dict(value).items()}


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_VERIFIER_FRESHNESS_FUNCTION_TOKENS = frozenset(
    {"inspect", "probe", "verify", "check", "measure", "audit"}
)


def _looks_like_verifier_function(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in _VERIFIER_FRESHNESS_FUNCTION_TOKENS:
        return True
    return any(token in text for token in _VERIFIER_FRESHNESS_FUNCTION_TOKENS)


def _issued_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class TaskContract:
    contract_id: str
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    title: str = ""
    status: str = ""
    intent: str = ""
    target_function: str = ""
    success_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    planning: Dict[str, Any] = field(default_factory=dict)
    approval: Dict[str, Any] = field(default_factory=dict)
    verification_requirement: Dict[str, Any] = field(default_factory=dict)
    completion: Dict[str, Any] = field(default_factory=dict)
    assigned_worker: Dict[str, Any] = field(default_factory=dict)
    governance_memory: Dict[str, Any] = field(default_factory=dict)
    freshness_binding: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = "conos.task_contract/v2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextFrame:
    frame_id: str
    episode: int = 0
    tick: int = 0
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    perception_summary: Dict[str, Any] = field(default_factory=dict)
    meta_control_snapshot: Dict[str, Any] = field(default_factory=dict)
    world_model_summary: Dict[str, Any] = field(default_factory=dict)
    self_model_summary: Dict[str, Any] = field(default_factory=dict)
    task_contract: Dict[str, Any] = field(default_factory=dict)
    unified_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_version: str = "conos.context_frame/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCallTicket:
    ticket_id: str
    issued_at: float
    route_name: str
    method_name: str = "complete"
    capability_request: str = ""
    schema_name: str = ""
    fallback_route: str = ""
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    prompt_tokens: int = 0
    reserved_response_tokens: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)
    audit_event_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    ticket_version: str = "conos.model_call_ticket/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCapabilityManifest:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    capability_class: str = ""
    side_effect_class: str = ""
    approval_required: bool = False
    risk_level: str = "low"
    source: str = "surface"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""
    manifest_version: str = "conos.tool_capability_manifest/v1"

    def __post_init__(self) -> None:
        if not str(self.tool_id or "").strip():
            object.__setattr__(self, "tool_id", str(self.name or "").strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionTicket:
    ticket_id: str
    issued_at: float
    policy_name: str
    tool_id: str
    function_name: str
    capability_class: str
    side_effect_class: str
    approval_required: bool
    approved: bool
    decision_reason: str
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    run_id: str = ""
    episode: int = 0
    tick: int = 0
    approval_result: Dict[str, Any] = field(default_factory=dict)
    execution_scope: Dict[str, Any] = field(default_factory=dict)
    verification_requirement: Dict[str, Any] = field(default_factory=dict)
    secret_lease_requirement: Dict[str, Any] = field(default_factory=dict)
    secret_lease_result: Dict[str, Any] = field(default_factory=dict)
    audit_event_id: str = ""
    task_contract: Dict[str, Any] = field(default_factory=dict)
    goal_contract: Dict[str, Any] = field(default_factory=dict)
    task_graph: Dict[str, Any] = field(default_factory=dict)
    task_node: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ticket_version: str = "conos.execution_ticket/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    result_id: str
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    verifier_function: str = ""
    passed: bool = False
    requirement: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    failure_mode: str = "block"
    source: str = ""
    audit_event_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_version: str = "conos.verification_result/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditEvent:
    audit_event_id: str
    event_type: str
    issued_at: float
    goal_ref: str = ""
    task_ref: str = ""
    graph_ref: str = ""
    tool_id: str = ""
    execution_ticket_id: str = ""
    verification_result_id: str = ""
    model_call_ticket_id: str = ""
    source_stage: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_version: str = "conos.audit_event/v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_task_contract(
    *,
    goal_contract: Any = None,
    task_graph: Any = None,
    task_node: Any = None,
    completion_gate: Any = None,
    authority_snapshot: Any = None,
    run_id: str = "",
    episode: int = 0,
    tick: int = 0,
) -> TaskContract:
    goal_payload = _payload_dict(goal_contract)
    graph_payload = _payload_dict(task_graph)
    task_payload = _payload_dict(task_node)
    authority_payload = _dict_or_empty(authority_snapshot)
    title = str(task_payload.get("title", "") or goal_payload.get("title", "") or "").strip()
    goal_ref = str(goal_payload.get("goal_id", "") or graph_payload.get("goal_id", "") or "").strip()
    task_ref = str(task_payload.get("node_id", "") or "").strip()
    graph_ref = str(graph_payload.get("graph_id", "") or "").strip()
    goal_provenance = _dict_or_empty(goal_payload.get("provenance", {}))
    goal_metadata = _dict_or_empty(goal_payload.get("metadata", {}))
    graph_metadata = _dict_or_empty(graph_payload.get("metadata", {}))
    task_provenance = _dict_or_empty(task_payload.get("provenance", {}))
    task_metadata = _dict_or_empty(task_payload.get("metadata", {}))
    completion_payload = _dict_or_empty(goal_payload.get("completion", {}))
    verifier_authority = {}
    if isinstance(completion_gate, Mapping):
        completion_gate_payload = dict(completion_gate)
        completion_payload["completion_gate"] = completion_gate_payload
        if isinstance(completion_gate_payload.get("verifier_authority", {}), Mapping):
            verifier_authority = dict(completion_gate_payload.get("verifier_authority", {}) or {})
    approval_payload = _dict_or_empty(goal_payload.get("approval", {}))
    task_approval = _dict_or_empty(task_payload.get("approval_requirement", {}))
    if task_approval:
        approval_payload["task_requirement"] = task_approval
    verification_requirement = (
        _dict_or_empty(task_payload.get("verification_gate", {}))
        or _dict_or_empty(goal_payload.get("verification", {}))
    )
    if verifier_authority:
        verification_requirement["verifier_authority"] = verifier_authority
    source_plan_id = str(
        graph_metadata.get("plan_id", "")
        or goal_provenance.get("plan_id", "")
        or goal_metadata.get("plan_id", "")
        or task_provenance.get("plan_id", "")
        or ""
    ).strip()
    source_plan_revision = max(
        0,
        _int_or_default(
            graph_metadata.get(
                "revision_count",
                goal_metadata.get("revision_count", task_provenance.get("revision_count", 0)),
            ),
            0,
        ),
    )
    source_graph_revision = max(
        0,
        _int_or_default(
            graph_metadata.get("graph_revision", graph_metadata.get("revision_count", source_plan_revision)),
            source_plan_revision,
        ),
    )
    issued_episode = max(0, _int_or_default(episode, 0))
    issued_tick = max(0, _int_or_default(tick, 0))
    assigned_worker = _dict_or_empty(task_payload.get("assigned_worker", {}))
    task_worker_type = str(assigned_worker.get("worker_type", "") or "").strip().lower()
    task_target_function = str(task_metadata.get("target_function", "") or "").strip().lower()
    task_intent = str(task_metadata.get("intent", "") or "").strip().lower()
    task_capability_class = str(task_approval.get("capability_class", "") or "").strip().lower()
    required_secret_leases = list(task_approval.get("required_secret_leases", []) or [])
    is_verifier_contract = bool(
        task_worker_type == "verifier"
        or task_capability_class.startswith("verification")
        or _looks_like_verifier_function(task_target_function)
        or _looks_like_verifier_function(task_intent)
    )
    freshness_policy_class = "verifier_short_window" if is_verifier_contract else "same_tick_only"
    binding_kind = "verifier_contract" if is_verifier_contract else "execution_contract"
    tick_window = 2 if is_verifier_contract else 0
    valid_through_tick = issued_tick + tick_window
    renewal_strategy = "explicit_renewal"
    if is_verifier_contract:
        renewal_strategy = "renew_verifier_contract"
    if bool(task_approval.get("required", False)) or required_secret_leases:
        renewal_strategy = "explicit_renewal"
    freshness_binding = {
        "binding_version": "task_contract_freshness/v1",
        "binding_kind": binding_kind,
        "authority_source": str(authority_payload.get("source", "") or ""),
        "authority_integrity": str(authority_payload.get("integrity", "") or ""),
        "authority_warnings": _string_list(authority_payload.get("warnings", [])),
        "contract_source": str(
            authority_payload.get("source", "")
            or task_provenance.get("source", "")
            or goal_provenance.get("source", "")
            or ""
        ).strip(),
        "source_plan_id": source_plan_id,
        "source_plan_revision": source_plan_revision,
        "source_graph_revision": source_graph_revision,
        "source_step_id": str(task_provenance.get("step_id", "") or ""),
        "issued_run_id": str(run_id or "").strip(),
        "issued_episode": issued_episode,
        "issued_tick": issued_tick,
        "freshness_policy_class": freshness_policy_class,
        "acceptance_scope": "verification" if is_verifier_contract else "execution",
        "tick_window": tick_window,
        "cross_tick_valid": bool(tick_window > 0),
        "same_tick_only": bool(tick_window == 0),
        "renewal_strategy": renewal_strategy,
        "valid_through_tick": valid_through_tick,
        "expired": False,
        "stale": False,
        "stale_reasons": [],
        "renewal_required": False,
        "renewal_reasons": [],
        "acceptable_for_execution": True,
    }
    return TaskContract(
        contract_id=str(task_ref or graph_ref or goal_ref or _issued_id("task-contract")),
        goal_ref=goal_ref,
        task_ref=task_ref,
        graph_ref=graph_ref,
        title=title,
        status=str(task_payload.get("status", "") or graph_payload.get("status", "") or ""),
        intent=str(_dict_or_empty(task_payload.get("metadata", {})).get("intent", "") or ""),
        target_function=str(_dict_or_empty(task_payload.get("metadata", {})).get("target_function", "") or ""),
        success_criteria=_string_list(task_payload.get("success_criteria", []))
        or _string_list(goal_payload.get("success_criteria", [])),
        dependencies=_string_list(task_payload.get("dependencies", [])),
        planning=_dict_or_empty(goal_payload.get("planning", {})),
        approval=approval_payload,
        verification_requirement=verification_requirement,
        completion=completion_payload,
        assigned_worker=_dict_or_empty(task_payload.get("assigned_worker", {})),
        governance_memory=_dict_or_empty(task_payload.get("governance_memory", {})),
        freshness_binding=freshness_binding,
        metadata={
            "goal_title": str(goal_payload.get("title", "") or ""),
            "task_metadata": task_metadata,
            "goal_metadata": _dict_or_empty(goal_payload.get("metadata", {})),
            "authority_snapshot": authority_payload,
        },
    )


def build_context_frame(
    frame: Any,
    *,
    task_contract: Any = None,
    goal_ref: str = "",
    task_ref: str = "",
    graph_ref: str = "",
) -> ContextFrame:
    unified_context = _payload_dict(getattr(frame, "unified_context", None))
    task_contract_payload = _payload_dict(task_contract)
    goal_value = str(goal_ref or task_contract_payload.get("goal_ref", "") or getattr(frame, "current_goal", "") or "").strip()
    task_value = str(task_ref or task_contract_payload.get("task_ref", "") or getattr(frame, "current_task", "") or "").strip()
    graph_value = str(graph_ref or task_contract_payload.get("graph_ref", "") or "").strip()
    return ContextFrame(
        frame_id=_issued_id("ctx"),
        episode=int(getattr(frame, "episode", 0) or 0),
        tick=int(getattr(frame, "tick", 0) or 0),
        goal_ref=goal_value,
        task_ref=task_value,
        graph_ref=graph_value,
        perception_summary=_dict_or_empty(getattr(frame, "perception_summary", {})),
        meta_control_snapshot=_dict_or_empty(getattr(frame, "meta_control_snapshot", {})),
        world_model_summary=_dict_or_empty(getattr(frame, "world_model_summary", {})),
        self_model_summary=_dict_or_empty(getattr(frame, "self_model_summary", {})),
        task_contract=task_contract_payload,
        unified_context=unified_context,
        metadata={"source_frame_type": type(frame).__name__},
    )


def build_model_call_ticket(
    *,
    route_name: str,
    method_name: str = "complete",
    capability_request: str = "",
    schema_name: str = "",
    fallback_route: str = "",
    goal_ref: str = "",
    task_ref: str = "",
    graph_ref: str = "",
    prompt_tokens: int = 0,
    reserved_response_tokens: int = 0,
    budget: Optional[Mapping[str, Any]] = None,
    audit_event_id: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> ModelCallTicket:
    return ModelCallTicket(
        ticket_id=_issued_id("llm"),
        issued_at=float(time.time()),
        route_name=str(route_name or "general").strip() or "general",
        method_name=str(method_name or "complete"),
        capability_request=str(capability_request or route_name or "general").strip(),
        schema_name=str(schema_name or "").strip(),
        fallback_route=str(fallback_route or "").strip(),
        goal_ref=str(goal_ref or "").strip(),
        task_ref=str(task_ref or "").strip(),
        graph_ref=str(graph_ref or "").strip(),
        prompt_tokens=int(max(0, prompt_tokens or 0)),
        reserved_response_tokens=int(max(0, reserved_response_tokens or 0)),
        budget=dict(budget or {}),
        audit_event_id=str(audit_event_id or _issued_id("audit")),
        metadata=dict(metadata or {}),
    )


def build_tool_capability_manifest(tool: Any) -> ToolCapabilityManifest:
    payload = _payload_dict(tool)
    name = str(payload.get("name", "") or payload.get("tool_id", "") or "").strip()
    return ToolCapabilityManifest(
        name=name,
        description=str(payload.get("description", "") or ""),
        input_schema=_dict_or_empty(payload.get("input_schema", {})),
        side_effects=_string_list(payload.get("side_effects", [])),
        risk_notes=_string_list(payload.get("risk_notes", [])),
        capability_class=str(payload.get("capability_class", "") or ""),
        side_effect_class=str(payload.get("side_effect_class", "") or ""),
        approval_required=bool(payload.get("approval_required", False)),
        risk_level=str(payload.get("risk_level", "low") or "low"),
        source=str(payload.get("source", "surface") or "surface"),
        metadata=_dict_or_empty(payload.get("metadata", {})),
        tool_id=str(payload.get("tool_id", "") or name),
    )


def build_verification_result(
    *,
    goal_ref: str = "",
    task_ref: str = "",
    graph_ref: str = "",
    verifier_function: str = "",
    passed: bool = False,
    requirement: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
    failure_mode: str = "block",
    source: str = "",
    audit_event_id: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> VerificationResult:
    return VerificationResult(
        result_id=_issued_id("verify"),
        goal_ref=str(goal_ref or "").strip(),
        task_ref=str(task_ref or "").strip(),
        graph_ref=str(graph_ref or "").strip(),
        verifier_function=str(verifier_function or "").strip(),
        passed=bool(passed),
        requirement=dict(requirement or {}),
        evidence=dict(evidence or {}),
        failure_mode=str(failure_mode or "block"),
        source=str(source or "").strip(),
        audit_event_id=str(audit_event_id or _issued_id("audit")),
        metadata=dict(metadata or {}),
    )


def build_audit_event(
    event_type: str,
    *,
    goal_ref: str = "",
    task_ref: str = "",
    graph_ref: str = "",
    tool_id: str = "",
    execution_ticket_id: str = "",
    verification_result_id: str = "",
    model_call_ticket_id: str = "",
    source_stage: str = "",
    payload: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    audit_event_id: str = "",
) -> AuditEvent:
    return AuditEvent(
        audit_event_id=str(audit_event_id or _issued_id("audit")),
        event_type=str(event_type or "audit_event"),
        issued_at=float(time.time()),
        goal_ref=str(goal_ref or "").strip(),
        task_ref=str(task_ref or "").strip(),
        graph_ref=str(graph_ref or "").strip(),
        tool_id=str(tool_id or "").strip(),
        execution_ticket_id=str(execution_ticket_id or "").strip(),
        verification_result_id=str(verification_result_id or "").strip(),
        model_call_ticket_id=str(model_call_ticket_id or "").strip(),
        source_stage=str(source_stage or "").strip(),
        payload=dict(payload or {}),
        metadata=dict(metadata or {}),
    )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional

from core.cognition.unified_context import UnifiedCognitiveContext
from core.conos_kernel import build_task_contract
from core.runtime_budget import (
    merge_llm_capability_specs,
    resolve_llm_capability_policies,
    resolve_llm_capability_policy_entries,
    resolve_llm_route_policies,
)


_HIGH_RISK_TITLE_TOKENS = (
    "submit",
    "checkout",
    "purchase",
    "delete",
    "remove",
    "write",
    "patch",
    "commit",
    "publish",
    "deploy",
    "finalize",
)

_VERIFICATION_TITLE_TOKENS = (
    "verify",
    "verification",
    "inspect",
    "probe",
    "check",
    "measure",
    "test",
)


def _stable_id(prefix: str, *parts: Any) -> str:
    blob = "||".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not blob:
        blob = prefix
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)

def _unique_strings(values: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _task_contract_verifier_authority(task_contract: Any) -> Dict[str, Any]:
    payload = _dict_or_empty(task_contract)
    verification_requirement = _dict_or_empty(payload.get("verification_requirement", {}))
    return _dict_or_empty(verification_requirement.get("verifier_authority", {}))


def resolve_task_graph_active_task_payload(
    task_graph: Any,
    *,
    fallback: Any = None,
) -> Dict[str, Any]:
    graph_payload = _dict_or_empty(task_graph)
    active_node_id = str(graph_payload.get("active_node_id", "") or "").strip()
    nodes = _dict_list(graph_payload.get("nodes", []))
    if active_node_id:
        for node in nodes:
            if str(node.get("node_id", "") or "").strip() == active_node_id:
                return dict(node)
    if nodes:
        return dict(nodes[0])
    return _dict_or_empty(fallback)


def resolve_plan_summary_active_task(plan_summary: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan_summary, Mapping):
        return {}
    return resolve_task_graph_active_task_payload(
        plan_summary.get("task_graph", {}),
        fallback=plan_summary.get("active_task_node", {}),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_int(value: Any, *, default: int = 0, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


_GOAL_RISK_LEVEL_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def _normalize_goal_risk_level(value: Any, *, default: str = "low") -> str:
    clean = str(value or "").strip().lower()
    if clean in _GOAL_RISK_LEVEL_RANK:
        return clean
    return default


def _default_goal_risk_level(approval: GoalApprovalSchema) -> str:
    if list(approval.required_functions or []):
        return "high"
    return "low"


def _default_goal_verification_plan(verification: GoalVerificationSchema) -> Dict[str, Any]:
    return {
        "required_for_completion": bool(verification.required_for_completion),
        "verification_scope": str(verification.verification_scope or "task_graph"),
        "failure_mode": str(verification.failure_mode or "block_completion"),
        "verification_functions": list(verification.verification_functions or []),
        "success_criteria": list(verification.success_criteria or []),
    }


@dataclass(frozen=True)
class GoalPlanningSchema:
    max_steps: int = 0
    max_ticks: int = 0
    target_reward: Optional[float] = None
    success_indicator: str = ""
    allowed_step_intents: List[str] = field(default_factory=list)
    blocked_functions: List[str] = field(default_factory=list)
    replanning_allowed: bool = True
    llm_route_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_capability_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["llm_route_policies"] = resolve_llm_route_policies(self.llm_route_policies)
        payload["llm_capability_policies"] = resolve_llm_capability_policies(self.llm_capability_policies)
        payload["llm_capability_policy_entries"] = resolve_llm_capability_policy_entries(
            payload["llm_capability_policies"]
        )
        return payload


@dataclass(frozen=True)
class GoalApprovalSchema:
    require_explicit_approval_for_high_risk: bool = True
    required_functions: List[str] = field(default_factory=list)
    blocked_functions: List[str] = field(default_factory=list)
    required_secret_leases: List[Dict[str, Any]] = field(default_factory=list)
    approval_scope: str = "goal_task_binding"


@dataclass(frozen=True)
class GoalVerificationSchema:
    required_for_completion: bool = False
    verification_functions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_mode: str = "block_completion"
    verification_scope: str = "task_graph"


@dataclass(frozen=True)
class GoalCompletionSchema:
    completion_mode: str = "terminal_state"
    requires_all_nodes_terminal: bool = True
    terminal_statuses: List[str] = field(default_factory=lambda: ["completed", "skipped"])
    requires_goal_success_criteria: bool = False
    requires_verification: bool = False


@dataclass(frozen=True)
class TaskVerificationGate:
    required: bool = False
    mode: str = "none"
    verifier_function: str = ""
    success_criteria: List[str] = field(default_factory=list)
    failure_mode: str = "block"
    last_verified: bool = False
    last_verdict: str = "pending"
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRetryPolicy:
    max_attempts: int = 1
    on_failure: str = "block"
    backoff: str = "none"


@dataclass(frozen=True)
class TaskAssignment:
    worker_id: str = "primary"
    worker_type: str = "executor"
    ownership: str = "exclusive"
    source: str = "derived"


@dataclass(frozen=True)
class TaskApprovalRequirement:
    required: bool = False
    risk_level: str = "low"
    capability_class: str = ""
    reason: str = ""
    required_secret_leases: List[Dict[str, Any]] = field(default_factory=list)
    allow_high_risk_without_approval: bool = False


@dataclass(frozen=True)
class GoalContract:
    goal_id: str
    title: str
    objective: str = ""
    success_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    approval_points: List[Dict[str, Any]] = field(default_factory=list)
    verification_plan: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    priority: str = "normal"
    planning: GoalPlanningSchema = field(default_factory=GoalPlanningSchema)
    approval: GoalApprovalSchema = field(default_factory=GoalApprovalSchema)
    verification: GoalVerificationSchema = field(default_factory=GoalVerificationSchema)
    completion: GoalCompletionSchema = field(default_factory=GoalCompletionSchema)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = "goal_contract/v2"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["objective"] = str(self.objective or self.title or "")
        payload["deliverables"] = _string_list(payload.get("deliverables", []))
        if not payload["deliverables"] and list(self.success_criteria or []):
            payload["deliverables"] = list(self.success_criteria or [])
        payload["risk_level"] = _normalize_goal_risk_level(
            payload.get("risk_level", ""),
            default=_default_goal_risk_level(self.approval),
        )
        payload["allowed_tools"] = _string_list(payload.get("allowed_tools", []))
        if not payload["allowed_tools"]:
            payload["allowed_tools"] = list(self.approval.required_functions or [])
        payload["forbidden_actions"] = _string_list(payload.get("forbidden_actions", []))
        if not payload["forbidden_actions"]:
            payload["forbidden_actions"] = _unique_strings(
                list(self.planning.blocked_functions or []) + list(self.approval.blocked_functions or [])
            )
        if not _dict_or_empty(payload.get("verification_plan", {})):
            payload["verification_plan"] = _default_goal_verification_plan(self.verification)
        payload["planning"] = self.planning.to_dict()
        return payload


@dataclass(frozen=True)
class TaskNode:
    node_id: str
    title: str
    status: str
    goal_id: str
    success_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    verification_gate: TaskVerificationGate = field(default_factory=TaskVerificationGate)
    retry_policy: TaskRetryPolicy = field(default_factory=TaskRetryPolicy)
    retry_state: Dict[str, Any] = field(default_factory=dict)
    assigned_worker: TaskAssignment = field(default_factory=TaskAssignment)
    approval_requirement: TaskApprovalRequirement = field(default_factory=TaskApprovalRequirement)
    approval_state: Dict[str, Any] = field(default_factory=dict)
    governance_memory: Dict[str, Any] = field(default_factory=dict)
    branch_targets: List[Dict[str, Any]] = field(default_factory=list)
    rollback_edge: Dict[str, Any] = field(default_factory=dict)
    verifier_node_id: str = ""
    approval_node_id: str = ""
    llm_route_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_capability_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["llm_route_policies"] = resolve_llm_route_policies(self.llm_route_policies)
        payload["llm_capability_policies"] = resolve_llm_capability_policies(self.llm_capability_policies)
        payload["llm_capability_policy_entries"] = resolve_llm_capability_policy_entries(
            payload["llm_capability_policies"]
        )
        return payload


@dataclass(frozen=True)
class TaskGraph:
    graph_id: str
    goal_id: str
    status: str = "active"
    active_node_id: str = ""
    nodes: List[TaskNode] = field(default_factory=list)
    approval_nodes: List[Dict[str, Any]] = field(default_factory=list)
    verifier_nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    active_control_node_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_version: str = "task_graph/v2"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["nodes"] = [node.to_dict() for node in self.nodes]
        return payload


GOAL_TASK_AUTHORITY_BUILDER_VERSION = "goal_task_authority_builder/v1"


@dataclass(frozen=True)
class GoalTaskBinding:
    goal_contract: Optional[GoalContract] = None
    task_graph: Optional[TaskGraph] = None
    active_task: Optional[TaskNode] = None
    completion_gate: Dict[str, Any] = field(default_factory=dict)
    verifier_authority_override: Dict[str, Any] = field(default_factory=dict)
    authority_source: str = "none"
    authority_integrity: str = "none"
    authority_warnings: List[str] = field(default_factory=list)
    run_id: str = ""
    episode: int = 0
    tick: int = 0

    def to_context(self) -> Dict[str, Any]:
        from core.orchestration.verifier_runtime import build_verifier_runtime

        verifier_runtime = build_verifier_runtime(
            goal_contract=self.goal_contract,
            task_graph=self.task_graph,
            active_task=self.active_task,
            completion_gate=self.completion_gate,
            verifier_authority_override=self.verifier_authority_override,
        )
        verifier_authority = dict(verifier_runtime.verifier_authority)
        completion_gate = dict(verifier_runtime.completion_gate)
        authority_snapshot = {
            "source": str(self.authority_source or "none"),
            "integrity": str(self.authority_integrity or "none"),
            "warnings": list(self.authority_warnings or []),
            "builder_version": GOAL_TASK_AUTHORITY_BUILDER_VERSION,
        }
        task_contract = build_task_contract(
            goal_contract=self.goal_contract,
            task_graph=self.task_graph,
            task_node=self.active_task,
            completion_gate=completion_gate,
            authority_snapshot=authority_snapshot,
            run_id=self.run_id,
            episode=int(self.episode or 0),
            tick=int(self.tick or 0),
        )
        execution_authority = dict(verifier_runtime.execution_authority)
        return {
            "goal_ref": self.goal_contract.goal_id if self.goal_contract is not None else "",
            "task_ref": self.active_task.node_id if self.active_task is not None else "",
            "graph_ref": self.task_graph.graph_id if self.task_graph is not None else "",
            "task_contract": task_contract.to_dict(),
            "goal_contract": self.goal_contract.to_dict() if self.goal_contract is not None else {},
            "task_graph": self.task_graph.to_dict() if self.task_graph is not None else {},
            "task_node": self.active_task.to_dict() if self.active_task is not None else {},
            "completion_gate": completion_gate,
            "verifier_authority": verifier_authority,
            "verifier_runtime": verifier_runtime.to_dict(),
            "authority_snapshot": authority_snapshot,
            "execution_authority": execution_authority,
        }


def _normalized_name_set(values: Any) -> set[str]:
    return {
        str(item).strip().casefold()
        for item in list(values or [])
        if str(item).strip()
    }


def _plan_summary_contains_native_goal_task_authority(plan_summary: Mapping[str, Any]) -> bool:
    return any(
        key in plan_summary
        for key in (
            "goal_contract",
            "task_graph",
            "active_task_node",
            "task_contract",
            "completion_gate",
            "execution_authority",
        )
    )


def _normalized_verification_verdict(
    *,
    required: bool,
    last_verified: bool,
    explicit_verdict: str = "",
    evidence: Optional[Mapping[str, Any]] = None,
) -> str:
    clean = str(explicit_verdict or "").strip().lower()
    if clean in {"passed", "failed", "pending", "not_required"}:
        if clean == "pending" and not required and not bool(last_verified):
            return "not_required"
        return clean
    if bool(last_verified):
        return "passed"
    if not bool(required):
        return "not_required"
    if isinstance(evidence, Mapping) and dict(evidence):
        return "pending"
    return "pending"


def _coerce_goal_contract(value: Any) -> Optional[GoalContract]:
    if isinstance(value, GoalContract):
        return value
    contract = _goal_contract_from_payload(value)
    if contract is not None:
        return contract
    payload = _dict_or_empty(value)
    if not payload:
        return None
    title = str(
        payload.get("title", "")
        or payload.get("goal", "")
        or payload.get("summary", "")
        or "goal_contract"
    ).strip()
    goal_id = str(payload.get("goal_id", "") or "").strip() or _stable_id("goal", title)
    success_criteria = _string_list(payload.get("success_criteria", []))
    planning = _goal_planning_from_payload(payload.get("planning", {}))
    approval = _goal_approval_from_payload(payload.get("approval", {}))
    verification = _goal_verification_from_payload(payload.get("verification", {}))
    verification_schema = GoalVerificationSchema(
        required_for_completion=bool(
            verification.required_for_completion or success_criteria or verification.verification_functions
        ),
        verification_functions=list(verification.verification_functions),
        success_criteria=list(verification.success_criteria or success_criteria),
        failure_mode=verification.failure_mode,
        verification_scope=verification.verification_scope,
    )
    return GoalContract(
        goal_id=goal_id,
        title=title,
        objective=str(payload.get("objective", title) or title),
        success_criteria=success_criteria,
        abort_criteria=_string_list(payload.get("abort_criteria", [])),
        deliverables=(
            _string_list(payload.get("deliverables", []))
            or list(success_criteria or [])
        ),
        constraints=_dict_or_empty(payload.get("constraints", {})),
        risk_level=_normalize_goal_risk_level(
            payload.get("risk_level", ""),
            default=_default_goal_risk_level(approval),
        ),
        allowed_tools=(
            _string_list(payload.get("allowed_tools", []))
            or list(approval.required_functions or [])
        ),
        forbidden_actions=(
            _string_list(payload.get("forbidden_actions", []))
            or _unique_strings(list(planning.blocked_functions or []) + list(approval.blocked_functions or []))
        ),
        approval_points=_dict_list(payload.get("approval_points", [])),
        verification_plan=_dict_or_empty(payload.get("verification_plan", {}))
        or _default_goal_verification_plan(verification_schema),
        assumptions=_string_list(payload.get("assumptions", [])),
        unknowns=_string_list(payload.get("unknowns", [])),
        priority=str(payload.get("priority", "normal") or "normal"),
        planning=planning,
        approval=approval,
        verification=verification_schema,
        completion=_goal_completion_from_payload(payload.get("completion", {})),
        provenance=_dict_or_empty(payload.get("provenance", {})),
        metadata=_dict_or_empty(payload.get("metadata", {})),
        contract_version=str(payload.get("contract_version", "goal_contract/v2") or "goal_contract/v2"),
    )


def _coerce_task_graph(value: Any, goal_contract: Optional[GoalContract]) -> Optional[TaskGraph]:
    if isinstance(value, TaskGraph):
        return value
    return _task_graph_from_payload(value, goal_contract)


def _coerce_task_node(value: Any, goal_id: str = "") -> Optional[TaskNode]:
    if isinstance(value, TaskNode):
        return value
    return _task_node_from_payload(value, goal_id)


def _select_active_task(
    task_graph: Optional[TaskGraph],
    active_task: Optional[TaskNode],
) -> Optional[TaskNode]:
    if active_task is not None:
        return active_task
    if task_graph is None:
        return None
    return next(
        (
            node for node in list(task_graph.nodes or [])
            if str(node.node_id or "") == str(task_graph.active_node_id or "")
        ),
        task_graph.nodes[0] if task_graph.nodes else None,
    )


def build_goal_task_binding(
    *,
    goal_contract: Any = None,
    task_graph: Any = None,
    active_task: Any = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    verifier_authority_override: Optional[Mapping[str, Any]] = None,
    authority_source: str = "goal_task_authority_builder",
    authority_integrity: str = "",
    authority_warnings: Optional[List[str]] = None,
    run_id: str = "",
    episode: int = 0,
    tick: int = 0,
) -> GoalTaskBinding:
    contract = _coerce_goal_contract(goal_contract)
    graph = _coerce_task_graph(task_graph, contract)
    task = _coerce_task_node(active_task, contract.goal_id if contract is not None else "")
    task = _select_active_task(graph, task)
    source = str(authority_source or "goal_task_authority_builder")
    integrity = str(authority_integrity or "").strip()
    if not integrity:
        integrity = "complete" if contract is not None and graph is not None else "incomplete"
    warnings = _string_list(authority_warnings or [])
    if source != "none":
        if contract is None and "goal_contract_missing" not in warnings:
            warnings.append("goal_contract_missing")
        if graph is None and "task_graph_missing" not in warnings:
            warnings.append("task_graph_missing")
        if integrity == "complete" and (contract is None or graph is None):
            integrity = "incomplete"
    return GoalTaskBinding(
        goal_contract=contract,
        task_graph=graph,
        active_task=task,
        completion_gate=_derive_binding_completion_gate(
            goal_contract=contract,
            task_graph=graph,
            active_task=task,
            payload=completion_gate,
        ),
        verifier_authority_override=_dict_or_empty(verifier_authority_override),
        authority_source=source,
        authority_integrity=integrity,
        authority_warnings=warnings,
        run_id=str(run_id or ""),
        episode=int(episode or 0),
        tick=int(tick or 0),
    )


def build_goal_task_authority_context(
    *,
    goal_contract: Any = None,
    task_graph: Any = None,
    active_task: Any = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    verifier_authority_override: Optional[Mapping[str, Any]] = None,
    authority_source: str = "goal_task_authority_builder",
    authority_integrity: str = "",
    authority_warnings: Optional[List[str]] = None,
    run_id: str = "",
    episode: int = 0,
    tick: int = 0,
) -> Dict[str, Any]:
    return build_goal_task_binding(
        goal_contract=goal_contract,
        task_graph=task_graph,
        active_task=active_task,
        completion_gate=completion_gate,
        verifier_authority_override=verifier_authority_override,
        authority_source=authority_source,
        authority_integrity=authority_integrity,
        authority_warnings=authority_warnings,
        run_id=run_id,
        episode=episode,
        tick=tick,
    ).to_context()


def _task_target_function(task_node: Optional[TaskNode]) -> str:
    if task_node is None:
        return ""
    metadata = _dict_or_empty(task_node.metadata)
    target_function = str(metadata.get("target_function", "") or "").strip()
    if target_function:
        return target_function
    for criterion in list(task_node.success_criteria or []):
        text = str(criterion or "").strip()
        if text.startswith("invoke:"):
            return text.split(":", 1)[1].strip()
    return ""


def _task_intent(task_node: Optional[TaskNode]) -> str:
    if task_node is None:
        return ""
    metadata = _dict_or_empty(task_node.metadata)
    return str(metadata.get("intent", "") or "").strip()


def _function_looks_high_risk(function_name: str) -> bool:
    lowered = str(function_name or "").strip().lower()
    return bool(lowered) and any(token in lowered for token in _HIGH_RISK_TITLE_TOKENS)


def _infer_target_function_from_title(title: str, goal_contract: Optional[GoalContract]) -> str:
    lowered = str(title or "").strip().lower()
    if not lowered:
        return ""
    candidate_functions: List[str] = []
    if goal_contract is not None:
        candidate_functions.extend(list(goal_contract.approval.required_functions or []))
        candidate_functions.extend(list(goal_contract.approval.blocked_functions or []))
        candidate_functions.extend(list(goal_contract.planning.blocked_functions or []))
    for function_name in _unique_strings(candidate_functions):
        clean = str(function_name or "").strip().lower()
        if clean and clean in lowered:
            return str(function_name or "").strip()
    for token in _HIGH_RISK_TITLE_TOKENS:
        if token in lowered:
            return token
    return ""


def _task_looks_high_risk(task_node: Optional[TaskNode], *, function_name: str = "", intent: str = "") -> bool:
    if _function_looks_high_risk(function_name):
        return True
    clean_intent = str(intent or "").strip().lower()
    if clean_intent and _function_looks_high_risk(clean_intent):
        return True
    if task_node is None:
        return False
    if _title_looks_high_risk(task_node.title):
        return True
    return _function_looks_high_risk(_task_target_function(task_node))


def resolve_effective_task_verification_gate(
    goal_contract: Any,
    task_node: Any,
) -> Dict[str, Any]:
    contract = _coerce_goal_contract(goal_contract)
    goal_id = contract.goal_id if contract is not None else ""
    node = _coerce_task_node(task_node, goal_id)
    gate = node.verification_gate if node is not None else TaskVerificationGate()
    verification = contract.verification if contract is not None else GoalVerificationSchema()
    goal_success_criteria = list(contract.success_criteria) if contract is not None else []
    verifier_function = str(gate.verifier_function or "").strip()
    if not verifier_function and verification.verification_functions:
        verifier_function = str(verification.verification_functions[0] or "").strip()
    success_criteria = _unique_strings(
        list(gate.success_criteria or [])
        + (list(node.success_criteria or []) if node is not None else [])
        + list(verification.success_criteria or [])
        + goal_success_criteria
    )
    required = bool(gate.required)
    mode = str(gate.mode or ("before_completion" if required else "none") or "none")
    if mode == "none" and verifier_function:
        mode = "observation"
    evidence = _dict_or_empty(gate.evidence)
    last_verified = bool(gate.last_verified)
    last_verdict = _normalized_verification_verdict(
        required=required,
        last_verified=last_verified,
        explicit_verdict=str(getattr(gate, "last_verdict", "") or ""),
        evidence=evidence,
    )
    return {
        "required": required,
        "mode": mode,
        "verifier_function": verifier_function,
        "success_criteria": success_criteria,
        "failure_mode": str(gate.failure_mode or verification.failure_mode or "block"),
        "last_verified": last_verified,
        "last_verdict": last_verdict,
        "evidence": evidence,
        "evidence_present": bool(evidence),
        "evidence_keys": sorted(evidence.keys()),
    }


def resolve_effective_task_approval_requirement(
    goal_contract: Any,
    task_node: Any,
    *,
    function_name: str = "",
    capability_class: str = "",
) -> Dict[str, Any]:
    contract = _coerce_goal_contract(goal_contract)
    goal_id = contract.goal_id if contract is not None else ""
    node = _coerce_task_node(task_node, goal_id)
    requirement = node.approval_requirement if node is not None else TaskApprovalRequirement()
    approval = contract.approval if contract is not None else GoalApprovalSchema()
    planning = contract.planning if contract is not None else GoalPlanningSchema()
    effective_function = str(
        function_name
        or _task_target_function(node)
        or _infer_target_function_from_title(node.title if node is not None else "", contract)
        or ""
    ).strip()
    effective_intent = _task_intent(node)
    blocked_functions = _normalized_name_set(list(approval.blocked_functions) + list(planning.blocked_functions))
    required_functions = _normalized_name_set(approval.required_functions)
    goal_secret_leases = _dict_list(approval.required_secret_leases)
    task_secret_leases = _dict_list(requirement.required_secret_leases)
    high_risk = _task_looks_high_risk(node, function_name=effective_function, intent=effective_intent)
    allow_high_risk_without_approval = bool(getattr(requirement, "allow_high_risk_without_approval", False))
    required = bool(
        requirement.required
        or (effective_function and effective_function.casefold() in required_functions)
        or (approval.require_explicit_approval_for_high_risk and high_risk and not allow_high_risk_without_approval)
    )
    resolved_capability_class = str(requirement.capability_class or capability_class or "").strip()
    if not resolved_capability_class and required and high_risk:
        resolved_capability_class = "high_risk_action"
    reason = str(requirement.reason or "").strip()
    if not reason and effective_function and effective_function.casefold() in required_functions:
        reason = f"goal_contract.required_function:{effective_function}"
    if not reason and required and high_risk:
        reason = f"goal_contract.high_risk:{effective_function or (node.title if node is not None else '')}"
    required_secret_leases: List[Dict[str, Any]] = []
    if goal_secret_leases:
        required_secret_leases.extend(goal_secret_leases)
    if task_secret_leases:
        required_secret_leases.extend(task_secret_leases)
    return {
        "required": required,
        "risk_level": str(requirement.risk_level or ("high" if required and high_risk else "low") or "low"),
        "capability_class": resolved_capability_class,
        "reason": reason,
        "blocked": bool(effective_function and effective_function.casefold() in blocked_functions),
        "required_secret_leases": required_secret_leases,
        "allow_high_risk_without_approval": allow_high_risk_without_approval,
    }


def resolve_goal_contract_authority(
    *,
    goal_contract: Any,
    task_graph: Any = None,
    active_task: Any = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    contract = _coerce_goal_contract(goal_contract)
    graph = _coerce_task_graph(task_graph, contract)
    task = _coerce_task_node(active_task, contract.goal_id if contract is not None else "")
    if task is None and graph is not None:
        task = next(
            (
                node
                for node in list(graph.nodes or [])
                if str(node.node_id or "") == str(graph.active_node_id or "")
            ),
            graph.nodes[0] if graph.nodes else None,
        )
    planning = contract.planning if contract is not None else GoalPlanningSchema()
    approval = contract.approval if contract is not None else GoalApprovalSchema()
    verification = contract.verification if contract is not None else GoalVerificationSchema()
    completion = contract.completion if contract is not None else GoalCompletionSchema()
    effective_verification_gate = resolve_effective_task_verification_gate(contract, task)
    effective_approval_requirement = resolve_effective_task_approval_requirement(contract, task)
    verifier_authority = derive_verifier_authority_snapshot(
        goal_contract=contract,
        task_graph=graph,
        active_task=task,
        completion_gate=completion_gate,
    )
    active_task_intent = _task_intent(task)
    allowed_step_intents = list(planning.allowed_step_intents or [])
    allowed_by_planning = True
    if allowed_step_intents and active_task_intent:
        allowed_by_planning = active_task_intent in {
            str(item or "").strip()
            for item in allowed_step_intents
            if str(item or "").strip()
        }
    active_target_function = _task_target_function(task) or _infer_target_function_from_title(
        task.title if task is not None else "",
        contract,
    )
    planning_blocked_functions = list(planning.blocked_functions or [])
    if active_target_function and active_target_function.casefold() in _normalized_name_set(planning_blocked_functions):
        allowed_by_planning = False
    completion_snapshot = _dict_or_empty(completion_gate)
    return {
        "goal_ref": contract.goal_id if contract is not None else "",
        "graph_ref": graph.graph_id if graph is not None else "",
        "task_ref": task.node_id if task is not None else "",
        "contract_version": contract.contract_version if contract is not None else "goal_contract/v2",
        "task_graph": {
            "status": graph.status if graph is not None else "inactive",
            "active_node_id": graph.active_node_id if graph is not None else "",
            "active_control_node_id": graph.active_control_node_id if graph is not None else "",
            "approval_nodes": [dict(item) for item in list(graph.approval_nodes or [])] if graph is not None else [],
            "verifier_nodes": [dict(item) for item in list(graph.verifier_nodes or [])] if graph is not None else [],
            "edges": [dict(item) for item in list(graph.edges or [])] if graph is not None else [],
        },
        "planning": {
            "max_steps": int(planning.max_steps or 0),
            "max_ticks": int(planning.max_ticks or 0),
            "target_reward": planning.target_reward,
            "success_indicator": str(planning.success_indicator or ""),
            "allowed_step_intents": list(allowed_step_intents),
            "blocked_functions": list(planning_blocked_functions),
            "replanning_allowed": bool(planning.replanning_allowed),
            "active_task_allowed": bool(allowed_by_planning),
        },
        "approval": {
            "require_explicit_approval_for_high_risk": bool(approval.require_explicit_approval_for_high_risk),
            "required_functions": list(approval.required_functions or []),
            "blocked_functions": list(approval.blocked_functions or []),
            "required_secret_leases": _dict_list(approval.required_secret_leases),
            "approval_scope": str(approval.approval_scope or "goal_task_binding"),
            "active_task_requirement": dict(effective_approval_requirement),
        },
        "verification": {
            "required_for_completion": bool(verification.required_for_completion),
            "verification_functions": list(verification.verification_functions or []),
            "success_criteria": list(verification.success_criteria or []),
            "failure_mode": str(verification.failure_mode or "block_completion"),
            "verification_scope": str(verification.verification_scope or "task_graph"),
            "active_task_gate": dict(effective_verification_gate),
        },
        "verifier_authority": verifier_authority,
        "completion": {
            "completion_mode": str(completion.completion_mode or "terminal_state"),
            "requires_all_nodes_terminal": bool(completion.requires_all_nodes_terminal),
            "terminal_statuses": list(completion.terminal_statuses or ["completed", "skipped"]),
            "requires_goal_success_criteria": bool(completion.requires_goal_success_criteria),
            "requires_verification": bool(completion.requires_verification or verification.required_for_completion),
            "completion_gate": completion_snapshot,
        },
        "active_task": {
            "node_id": task.node_id if task is not None else "",
            "title": task.title if task is not None else "",
            "status": task.status if task is not None else "",
            "intent": active_task_intent,
            "target_function": active_target_function,
            "assigned_worker": asdict(task.assigned_worker) if task is not None else asdict(TaskAssignment()),
            "retry_policy": asdict(task.retry_policy) if task is not None else asdict(TaskRetryPolicy()),
            "retry_state": dict(task.retry_state) if task is not None else {},
            "llm_route_policies": dict(task.llm_route_policies) if task is not None else {},
            "approval_requirement": dict(effective_approval_requirement),
            "approval_state": dict(task.approval_state) if task is not None else {},
            "governance_memory": dict(task.governance_memory) if task is not None else {},
            "approval_node_id": str(task.approval_node_id or "") if task is not None else "",
            "verification_gate": dict(effective_verification_gate),
            "verifier_node_id": str(task.verifier_node_id or "") if task is not None else "",
            "branch_targets": [dict(item) for item in list(task.branch_targets or [])] if task is not None else [],
            "rollback_edge": dict(task.rollback_edge) if task is not None else {},
            "allowed_by_planning": bool(allowed_by_planning),
        },
    }


class GoalTaskRuntime:
    def __init__(self) -> None:
        self._binding = GoalTaskBinding()

    def current_binding(self) -> GoalTaskBinding:
        return self._binding

    def refresh(
        self,
        *,
        unified_context: Optional[UnifiedCognitiveContext],
        state_mgr: Any = None,
        run_id: str = "",
        episode: int = 0,
        tick: int = 0,
    ) -> GoalTaskBinding:
        goal_title = ""
        current_task = ""
        plan_summary: Dict[str, Any] = {}
        task_frame_summary: Dict[str, Any] = {}
        goal_agenda: List[Dict[str, Any]] = []
        long_horizon_commitments: List[Dict[str, Any]] = []
        if isinstance(unified_context, UnifiedCognitiveContext) or _looks_like_unified_context(unified_context):
            goal_title = str(getattr(unified_context, "current_goal", "") or "")
            current_task = str(getattr(unified_context, "current_task", "") or "")
            plan_summary = dict(getattr(unified_context, "plan_state_summary", {}) or {})
            task_frame_summary = dict(getattr(unified_context, "task_frame_summary", {}) or {})
            goal_agenda = _dict_list(getattr(unified_context, "goal_agenda", []))
            long_horizon_commitments = _dict_list(getattr(unified_context, "long_horizon_commitments", []))
        native_authority_present = _plan_summary_contains_native_goal_task_authority(plan_summary)
        native_task_contract = _dict_or_empty(plan_summary.get("task_contract", {}))
        native_verifier_authority = _task_contract_verifier_authority(native_task_contract)
        native_authority_snapshot = _dict_or_empty(plan_summary.get("authority_snapshot", {}))
        native_goal_contract = _goal_contract_from_payload(plan_summary.get("goal_contract", {}))
        native_task_graph = _task_graph_from_payload(plan_summary.get("task_graph", {}), native_goal_contract)
        native_active_task = _task_node_from_payload(
            resolve_plan_summary_active_task(plan_summary),
            native_goal_contract.goal_id if native_goal_contract is not None else "",
        )
        if native_goal_contract is not None and not goal_title:
            goal_title = native_goal_contract.title
        if native_active_task is not None and not current_task:
            current_task = native_active_task.title
        if not goal_title and state_mgr is not None and hasattr(state_mgr, "get"):
            goal_title = str(state_mgr.get("goal_stack.top_goal", "") or "")
        if not current_task:
            current_task = str(
                plan_summary.get("current_step_intent")
                or plan_summary.get("current_step_description")
                or goal_title
                or ""
            )
        goal_title = str(goal_title or "").strip()
        current_task = str(current_task or "").strip()
        native_authority_source = str(native_authority_snapshot.get("source", "") or "planner_native")
        native_authority_integrity = str(
            native_authority_snapshot.get(
                "integrity",
                "complete" if native_goal_contract is not None and native_task_graph is not None else "incomplete",
            )
            or ("complete" if native_goal_contract is not None and native_task_graph is not None else "incomplete")
        )
        native_authority_warnings = _string_list(native_authority_snapshot.get("warnings", []))
        if native_authority_integrity != "complete" and not native_authority_warnings:
            native_authority_warnings = ["planner_native_goal_task_authority_incomplete"]

        if native_authority_present:
            self._binding = build_goal_task_binding(
                goal_contract=native_goal_contract,
                task_graph=native_task_graph,
                active_task=native_active_task,
                completion_gate=plan_summary.get("completion_gate", {}),
                verifier_authority_override=native_verifier_authority,
                authority_source=native_authority_source,
                authority_integrity=native_authority_integrity,
                authority_warnings=native_authority_warnings,
                run_id=str(run_id or ""),
                episode=int(episode or 0),
                tick=int(tick or 0),
            )
            return self._binding

        if not goal_title:
            self._binding = build_goal_task_binding(
                authority_source="none",
                authority_integrity="none",
                run_id=str(run_id or ""),
                episode=int(episode or 0),
                tick=int(tick or 0),
            )
            return self._binding

        success_criteria = _derive_goal_success_criteria(
            task_frame_summary=task_frame_summary,
            goal_agenda=goal_agenda,
        )
        planning = _derive_goal_planning_schema(
            task_frame_summary=task_frame_summary,
            goal_agenda=goal_agenda,
            plan_summary=plan_summary,
        )
        approval = _derive_goal_approval_schema(
            task_frame_summary=task_frame_summary,
            goal_agenda=goal_agenda,
            plan_summary=plan_summary,
        )
        verification = _derive_goal_verification_schema(
            success_criteria=success_criteria,
            task_frame_summary=task_frame_summary,
            plan_summary=plan_summary,
        )
        completion = _derive_goal_completion_schema(
            success_criteria=success_criteria,
            verification=verification,
        )
        goal_contract = GoalContract(
            goal_id=_stable_id("goal", goal_title),
            title=goal_title,
            objective=goal_title,
            success_criteria=success_criteria,
            abort_criteria=_derive_abort_criteria(task_frame_summary=task_frame_summary),
            deliverables=_derive_goal_deliverables(
                success_criteria=success_criteria,
                task_frame_summary=task_frame_summary,
                goal_agenda=goal_agenda,
                current_task=current_task,
            ),
            constraints=_derive_goal_constraints(
                task_frame_summary=task_frame_summary,
                plan_summary=plan_summary,
            ),
            risk_level=_derive_goal_risk_level(
                task_frame_summary=task_frame_summary,
                approval=approval,
            ),
            allowed_tools=_derive_goal_allowed_tools(
                task_frame_summary=task_frame_summary,
                approval=approval,
            ),
            forbidden_actions=_derive_goal_forbidden_actions(
                task_frame_summary=task_frame_summary,
                planning=planning,
                approval=approval,
            ),
            approval_points=_derive_goal_approval_points(
                task_frame_summary=task_frame_summary,
                goal_agenda=goal_agenda,
                approval=approval,
            ),
            verification_plan=_derive_goal_verification_plan(
                task_frame_summary=task_frame_summary,
                verification=verification,
            ),
            assumptions=_derive_goal_assumptions(
                task_frame_summary=task_frame_summary,
                goal_agenda=goal_agenda,
            ),
            unknowns=_derive_goal_unknowns(
                task_frame_summary=task_frame_summary,
                goal_agenda=goal_agenda,
            ),
            priority=_derive_goal_priority(goal_agenda=goal_agenda),
            planning=planning,
            approval=approval,
            verification=verification,
            completion=completion,
            provenance={
                "source": "unified_context" if isinstance(unified_context, UnifiedCognitiveContext) else "state_manager",
                "plan_id": str(plan_summary.get("plan_id", "") or ""),
                "episode": int(episode or 0),
                "tick": int(tick or 0),
            },
            metadata={
                "plan_id": str(plan_summary.get("plan_id", "") or ""),
                "revision_count": int(plan_summary.get("revision_count", 0) or 0),
                "goal_agenda_size": len(goal_agenda),
                "commitment_count": len(long_horizon_commitments),
            },
        )
        task_nodes = _build_task_nodes(
            goal_contract=goal_contract,
            current_task=current_task,
            goal_agenda=goal_agenda,
            long_horizon_commitments=long_horizon_commitments,
            plan_summary=plan_summary,
        )
        active_task = next(
            (node for node in task_nodes if node.title == current_task),
            task_nodes[0] if task_nodes else None,
        )
        task_graph = TaskGraph(
            graph_id=_stable_id("task_graph", goal_contract.goal_id, current_task or goal_title),
            goal_id=goal_contract.goal_id,
            status=str(plan_summary.get("status", "active") or "active"),
            active_node_id=active_task.node_id if active_task is not None else "",
            nodes=task_nodes,
            metadata={
                "current_task": current_task,
                "plan_has_plan": bool(plan_summary.get("has_plan", False)),
                "plan_id": str(plan_summary.get("plan_id", "") or ""),
                "plan_status": str(plan_summary.get("status", "") or ""),
                "revision_count": int(plan_summary.get("revision_count", 0) or 0),
                "graph_revision": int(plan_summary.get("revision_count", 0) or 0),
                "issued_episode": int(episode or 0),
                "issued_tick": int(tick or 0),
            },
        )
        self._binding = build_goal_task_binding(
            goal_contract=goal_contract,
            task_graph=task_graph,
            active_task=active_task,
            completion_gate=plan_summary.get("completion_gate", {}),
            authority_source="context_derived",
            authority_integrity="derived",
            authority_warnings=["goal_task_binding_derived_from_context"],
            run_id=str(run_id or ""),
            episode=int(episode or 0),
            tick=int(tick or 0),
        )
        return self._binding


def _looks_like_unified_context(value: Any) -> bool:
    return value is not None and any(
        hasattr(value, attr)
        for attr in (
            "current_goal",
            "current_task",
            "plan_state_summary",
            "task_frame_summary",
        )
    )


def _goal_planning_from_payload(payload: Any) -> GoalPlanningSchema:
    data = _dict_or_empty(payload)
    capability_specs = merge_llm_capability_specs(
        data.get("llm_capability_policies", {}),
        data.get("llm_capability_policy_entries", []),
    )
    return GoalPlanningSchema(
        max_steps=_bounded_int(data.get("max_steps", 0), default=0),
        max_ticks=_bounded_int(data.get("max_ticks", 0), default=0),
        target_reward=_optional_float(data.get("target_reward")),
        success_indicator=str(data.get("success_indicator", "") or ""),
        allowed_step_intents=_string_list(data.get("allowed_step_intents", [])),
        blocked_functions=_string_list(data.get("blocked_functions", [])),
        replanning_allowed=bool(data.get("replanning_allowed", True)),
        llm_route_policies=resolve_llm_route_policies(data.get("llm_route_policies", {})),
        llm_capability_policies=capability_specs,
    )


def _goal_approval_from_payload(payload: Any) -> GoalApprovalSchema:
    data = _dict_or_empty(payload)
    return GoalApprovalSchema(
        require_explicit_approval_for_high_risk=bool(
            data.get("require_explicit_approval_for_high_risk", True)
        ),
        required_functions=_string_list(data.get("required_functions", [])),
        blocked_functions=_string_list(data.get("blocked_functions", [])),
        required_secret_leases=_dict_list(data.get("required_secret_leases", [])),
        approval_scope=str(data.get("approval_scope", "goal_task_binding") or "goal_task_binding"),
    )


def _goal_verification_from_payload(payload: Any) -> GoalVerificationSchema:
    data = _dict_or_empty(payload)
    return GoalVerificationSchema(
        required_for_completion=bool(data.get("required_for_completion", False)),
        verification_functions=_string_list(data.get("verification_functions", [])),
        success_criteria=_string_list(data.get("success_criteria", [])),
        failure_mode=str(data.get("failure_mode", "block_completion") or "block_completion"),
        verification_scope=str(data.get("verification_scope", "task_graph") or "task_graph"),
    )


def _goal_completion_from_payload(payload: Any) -> GoalCompletionSchema:
    data = _dict_or_empty(payload)
    return GoalCompletionSchema(
        completion_mode=str(data.get("completion_mode", "terminal_state") or "terminal_state"),
        requires_all_nodes_terminal=bool(data.get("requires_all_nodes_terminal", True)),
        terminal_statuses=_string_list(data.get("terminal_statuses", ["completed", "skipped"])) or ["completed", "skipped"],
        requires_goal_success_criteria=bool(data.get("requires_goal_success_criteria", False)),
        requires_verification=bool(data.get("requires_verification", False)),
    )


def _verification_gate_from_payload(payload: Any) -> TaskVerificationGate:
    data = _dict_or_empty(payload)
    required = bool(data.get("required", False))
    default_mode = "before_completion" if required else "none"
    evidence = _dict_or_empty(data.get("evidence", {}))
    last_verified = bool(data.get("last_verified", False))
    return TaskVerificationGate(
        required=required,
        mode=str(data.get("mode", default_mode) or default_mode),
        verifier_function=str(data.get("verifier_function", "") or ""),
        success_criteria=_string_list(data.get("success_criteria", [])),
        failure_mode=str(data.get("failure_mode", "block") or "block"),
        last_verified=last_verified,
        last_verdict=_normalized_verification_verdict(
            required=required,
            last_verified=last_verified,
            explicit_verdict=str(data.get("last_verdict", "") or ""),
            evidence=evidence,
        ),
        evidence=evidence,
    )


def _retry_policy_from_payload(payload: Any) -> TaskRetryPolicy:
    data = _dict_or_empty(payload)
    return TaskRetryPolicy(
        max_attempts=_bounded_int(data.get("max_attempts", 1), default=1, minimum=1),
        on_failure=str(data.get("on_failure", "block") or "block"),
        backoff=str(data.get("backoff", "none") or "none"),
    )


def _assignment_from_payload(payload: Any) -> TaskAssignment:
    data = _dict_or_empty(payload)
    return TaskAssignment(
        worker_id=str(data.get("worker_id", "primary") or "primary"),
        worker_type=str(data.get("worker_type", "executor") or "executor"),
        ownership=str(data.get("ownership", "exclusive") or "exclusive"),
        source=str(data.get("source", "explicit") or "explicit"),
    )


def _approval_requirement_from_payload(payload: Any) -> TaskApprovalRequirement:
    data = _dict_or_empty(payload)
    return TaskApprovalRequirement(
        required=bool(data.get("required", False)),
        risk_level=str(data.get("risk_level", "low") or "low"),
        capability_class=str(data.get("capability_class", "") or ""),
        reason=str(data.get("reason", "") or ""),
        required_secret_leases=_dict_list(data.get("required_secret_leases", [])),
        allow_high_risk_without_approval=bool(data.get("allow_high_risk_without_approval", False)),
    )


def _goal_contract_from_payload(payload: Any) -> Optional[GoalContract]:
    if not isinstance(payload, dict):
        return None
    goal_id = str(payload.get("goal_id", "") or "").strip()
    title = str(payload.get("title", "") or "").strip()
    if not goal_id or not title:
        return None
    success_criteria = _string_list(payload.get("success_criteria", []))
    planning = _goal_planning_from_payload(payload.get("planning", {}))
    approval = _goal_approval_from_payload(payload.get("approval", {}))
    verification = _goal_verification_from_payload(payload.get("verification", {}))
    verification_schema = GoalVerificationSchema(
        required_for_completion=bool(
            verification.required_for_completion or success_criteria or verification.verification_functions
        ),
        verification_functions=list(verification.verification_functions),
        success_criteria=list(verification.success_criteria or success_criteria),
        failure_mode=verification.failure_mode,
        verification_scope=verification.verification_scope,
    )
    return GoalContract(
        goal_id=goal_id,
        title=title,
        objective=str(payload.get("objective", title) or title),
        success_criteria=success_criteria,
        abort_criteria=_string_list(payload.get("abort_criteria", [])),
        deliverables=(
            _string_list(payload.get("deliverables", []))
            or list(success_criteria or [])
        ),
        constraints=_dict_or_empty(payload.get("constraints", {})),
        risk_level=_normalize_goal_risk_level(
            payload.get("risk_level", ""),
            default=_default_goal_risk_level(approval),
        ),
        allowed_tools=(
            _string_list(payload.get("allowed_tools", []))
            or list(approval.required_functions or [])
        ),
        forbidden_actions=(
            _string_list(payload.get("forbidden_actions", []))
            or _unique_strings(list(planning.blocked_functions or []) + list(approval.blocked_functions or []))
        ),
        approval_points=_dict_list(payload.get("approval_points", [])),
        verification_plan=_dict_or_empty(payload.get("verification_plan", {}))
        or _default_goal_verification_plan(verification_schema),
        assumptions=_string_list(payload.get("assumptions", [])),
        unknowns=_string_list(payload.get("unknowns", [])),
        priority=str(payload.get("priority", "normal") or "normal"),
        planning=planning,
        approval=approval,
        verification=verification_schema,
        completion=_goal_completion_from_payload(payload.get("completion", {})),
        provenance=_dict_or_empty(payload.get("provenance", {})),
        metadata=_dict_or_empty(payload.get("metadata", {})),
        contract_version=str(payload.get("contract_version", "goal_contract/v2") or "goal_contract/v2"),
    )


def _task_node_from_payload(payload: Any, goal_id: str) -> Optional[TaskNode]:
    if not isinstance(payload, dict):
        return None
    node_id = str(payload.get("node_id", "") or "").strip()
    title = str(payload.get("title", "") or "").strip()
    if not node_id or not title:
        return None
    capability_specs = merge_llm_capability_specs(
        payload.get("llm_capability_policies", {}),
        payload.get("llm_capability_policy_entries", []),
    )
    return TaskNode(
        node_id=node_id,
        title=title,
        status=str(payload.get("status", "queued") or "queued"),
        goal_id=str(payload.get("goal_id", goal_id) or goal_id),
        success_criteria=_string_list(payload.get("success_criteria", [])),
        dependencies=_string_list(payload.get("dependencies", [])),
        verification_gate=_verification_gate_from_payload(payload.get("verification_gate", {})),
        retry_policy=_retry_policy_from_payload(payload.get("retry_policy", {})),
        retry_state=_dict_or_empty(payload.get("retry_state", {})),
        assigned_worker=_assignment_from_payload(payload.get("assigned_worker", {})),
        approval_requirement=_approval_requirement_from_payload(payload.get("approval_requirement", {})),
        approval_state=_dict_or_empty(payload.get("approval_state", {})),
        governance_memory=_dict_or_empty(payload.get("governance_memory", {})),
        branch_targets=_dict_list(payload.get("branch_targets", [])),
        rollback_edge=_dict_or_empty(payload.get("rollback_edge", {})),
        verifier_node_id=str(payload.get("verifier_node_id", "") or ""),
        approval_node_id=str(payload.get("approval_node_id", "") or ""),
        llm_route_policies=resolve_llm_route_policies(payload.get("llm_route_policies", {})),
        llm_capability_policies=capability_specs,
        provenance=_dict_or_empty(payload.get("provenance", {})),
        metadata=_dict_or_empty(payload.get("metadata", {})),
    )


def _task_graph_from_payload(payload: Any, goal_contract: Optional[GoalContract]) -> Optional[TaskGraph]:
    if not isinstance(payload, dict):
        return None
    graph_id = str(payload.get("graph_id", "") or "").strip()
    if not graph_id:
        return None
    nodes = [
        node
        for node in (
            _task_node_from_payload(row, goal_contract.goal_id if goal_contract is not None else "")
            for row in list(payload.get("nodes", []) or [])
        )
        if node is not None
    ]
    return TaskGraph(
        graph_id=graph_id,
        goal_id=str(payload.get("goal_id", goal_contract.goal_id if goal_contract is not None else "") or ""),
        status=str(payload.get("status", "active") or "active"),
        active_node_id=str(payload.get("active_node_id", "") or ""),
        nodes=nodes,
        approval_nodes=_dict_list(payload.get("approval_nodes", [])),
        verifier_nodes=_dict_list(payload.get("verifier_nodes", [])),
        edges=_dict_list(payload.get("edges", [])),
        active_control_node_id=str(payload.get("active_control_node_id", "") or ""),
        metadata=_dict_or_empty(payload.get("metadata", {})),
        graph_version=str(payload.get("graph_version", "task_graph/v2") or "task_graph/v2"),
    )


def _derive_goal_success_criteria(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
) -> List[str]:
    criteria = _string_list(task_frame_summary.get("success_criteria", []))
    if criteria:
        return criteria
    for item in goal_agenda:
        item_criteria = _string_list(item.get("success_criteria", []))
        if item_criteria:
            return item_criteria
    inferred_goal = task_frame_summary.get("inferred_level_goal", {})
    if isinstance(inferred_goal, dict):
        summary = str(inferred_goal.get("summary", "") or "").strip()
        if summary:
            return [summary]
    return []


def _derive_abort_criteria(*, task_frame_summary: Dict[str, Any]) -> List[str]:
    abort_criteria = _string_list(task_frame_summary.get("abort_criteria", []))
    if abort_criteria:
        return abort_criteria
    blocked = _string_list(task_frame_summary.get("blocked_functions", []))
    if blocked:
        return [f"avoid_blocked_functions:{item}" for item in blocked[:4]]
    return []


def _derive_goal_priority(*, goal_agenda: List[Dict[str, Any]]) -> str:
    if not goal_agenda:
        return "normal"
    for item in goal_agenda:
        priority = str(item.get("priority", "") or "").strip().lower()
        if priority:
            return priority
    return "normal"


def _derive_goal_planning_schema(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
    plan_summary: Mapping[str, Any],
) -> GoalPlanningSchema:
    allowed_step_intents = _string_list(plan_summary.get("allowed_step_intents", []))
    if not allowed_step_intents:
        agenda_intents = [
            str(item.get("intent", "") or "").strip()
            for item in goal_agenda
            if str(item.get("intent", "") or "").strip()
        ]
        allowed_step_intents = agenda_intents
    success_indicator = str(plan_summary.get("goal_success_indicator", "") or "").strip()
    if not success_indicator:
        success_indicator = str(task_frame_summary.get("completion_signal", "") or "").strip()
    capability_specs = merge_llm_capability_specs(
        plan_summary.get("llm_capability_policies", {}),
        plan_summary.get("llm_capability_policy_entries", []),
    )
    return GoalPlanningSchema(
        max_steps=_bounded_int(plan_summary.get("total_steps", 0), default=0),
        max_ticks=_bounded_int(plan_summary.get("max_ticks", 0), default=0),
        target_reward=_optional_float(plan_summary.get("target_reward")),
        success_indicator=success_indicator,
        allowed_step_intents=allowed_step_intents,
        blocked_functions=_string_list(task_frame_summary.get("blocked_functions", [])),
        replanning_allowed=bool(plan_summary.get("has_plan", False) or goal_agenda),
        llm_route_policies=resolve_llm_route_policies(plan_summary.get("llm_route_policies", {})),
        llm_capability_policies=capability_specs,
    )


def _derive_goal_deliverables(
    *,
    success_criteria: List[str],
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
    current_task: str,
) -> List[str]:
    deliverables = _string_list(task_frame_summary.get("deliverables", []))
    if deliverables:
        return deliverables
    for item in goal_agenda:
        item_deliverables = _string_list(item.get("deliverables", []))
        if item_deliverables:
            return item_deliverables
        single = str(item.get("deliverable", "") or "").strip()
        if single:
            return [single]
    if success_criteria:
        return list(success_criteria)
    if str(current_task or "").strip():
        return [str(current_task or "").strip()]
    return []


def _derive_goal_constraints(
    *,
    task_frame_summary: Dict[str, Any],
    plan_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    constraints = _dict_or_empty(task_frame_summary.get("constraints", {}))
    if constraints:
        return constraints
    derived: Dict[str, Any] = {}
    blocked_functions = _string_list(task_frame_summary.get("blocked_functions", []))
    if blocked_functions:
        derived["blocked_functions"] = blocked_functions
    execution_limits: Dict[str, Any] = {}
    total_steps = _bounded_int(plan_summary.get("total_steps", 0), default=0)
    max_ticks = _bounded_int(plan_summary.get("max_ticks", 0), default=0)
    target_reward = _optional_float(plan_summary.get("target_reward"))
    if total_steps > 0:
        execution_limits["max_steps"] = total_steps
    if max_ticks > 0:
        execution_limits["max_ticks"] = max_ticks
    if target_reward is not None:
        execution_limits["target_reward"] = target_reward
    if execution_limits:
        derived["execution_limits"] = execution_limits
    return derived


def _derive_goal_risk_level(
    *,
    task_frame_summary: Dict[str, Any],
    approval: GoalApprovalSchema,
) -> str:
    explicit = _normalize_goal_risk_level(task_frame_summary.get("risk_level", ""), default="")
    if explicit:
        return explicit
    if list(approval.required_functions or []):
        return "high"
    if list(approval.blocked_functions or []):
        return "medium"
    return _default_goal_risk_level(approval)


def _derive_goal_allowed_tools(
    *,
    task_frame_summary: Dict[str, Any],
    approval: GoalApprovalSchema,
) -> List[str]:
    explicit = _string_list(task_frame_summary.get("allowed_tools", []))
    if explicit:
        return explicit
    return list(approval.required_functions or [])


def _derive_goal_forbidden_actions(
    *,
    task_frame_summary: Dict[str, Any],
    planning: GoalPlanningSchema,
    approval: GoalApprovalSchema,
) -> List[str]:
    return _unique_strings(
        _string_list(task_frame_summary.get("forbidden_actions", []))
        + _string_list(task_frame_summary.get("blocked_functions", []))
        + list(planning.blocked_functions or [])
        + list(approval.blocked_functions or [])
    )


def _derive_goal_approval_points(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
    approval: GoalApprovalSchema,
) -> List[Dict[str, Any]]:
    points = _dict_list(task_frame_summary.get("approval_points", []))
    if points:
        return points
    derived: List[Dict[str, Any]] = []
    for item in goal_agenda:
        title = str(item.get("title") or item.get("goal") or item.get("summary") or "").strip()
        if not title or not _title_looks_high_risk(title):
            continue
        derived.append(
            {
                "title": title,
                "reason": "goal_agenda_high_risk",
                "risk_level": "high",
                "scope": approval.approval_scope,
            }
        )
    if not derived and list(approval.required_functions or []):
        derived.append(
            {
                "reason": "required_functions",
                "required_functions": list(approval.required_functions or []),
                "risk_level": _default_goal_risk_level(approval),
                "scope": approval.approval_scope,
            }
        )
    return derived


def _derive_goal_verification_plan(
    *,
    task_frame_summary: Dict[str, Any],
    verification: GoalVerificationSchema,
) -> Dict[str, Any]:
    explicit = _dict_or_empty(task_frame_summary.get("verification_plan", {}))
    if explicit:
        return explicit
    return _default_goal_verification_plan(verification)


def _derive_goal_assumptions(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
) -> List[str]:
    assumptions = _string_list(task_frame_summary.get("assumptions", []))
    if assumptions:
        return assumptions
    for item in goal_agenda:
        item_assumptions = _string_list(item.get("assumptions", []))
        if item_assumptions:
            return item_assumptions
    return []


def _derive_goal_unknowns(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
) -> List[str]:
    unknowns = _string_list(task_frame_summary.get("unknowns", []))
    if unknowns:
        return unknowns
    for item in goal_agenda:
        item_unknowns = _string_list(item.get("unknowns", []))
        if item_unknowns:
            return item_unknowns
    return []


def _derive_goal_approval_schema(
    *,
    task_frame_summary: Dict[str, Any],
    goal_agenda: List[Dict[str, Any]],
    plan_summary: Mapping[str, Any],
) -> GoalApprovalSchema:
    required_functions = _string_list(plan_summary.get("approval_required_functions", []))
    if not required_functions:
        for item in goal_agenda:
            title = str(item.get("title") or item.get("goal") or item.get("summary") or "").strip()
            if title and _title_looks_high_risk(title):
                required_functions.append(title)
    return GoalApprovalSchema(
        require_explicit_approval_for_high_risk=True,
        required_functions=required_functions,
        blocked_functions=_string_list(task_frame_summary.get("blocked_functions", [])),
        approval_scope="goal_task_binding",
    )


def _derive_goal_verification_schema(
    *,
    success_criteria: List[str],
    task_frame_summary: Dict[str, Any],
    plan_summary: Mapping[str, Any],
) -> GoalVerificationSchema:
    verification_functions = _string_list(plan_summary.get("verification_functions", []))
    if not verification_functions:
        verification_functions = _string_list(task_frame_summary.get("verification_functions", []))
    required_for_completion = bool(
        plan_summary.get("requires_verification", False)
        or task_frame_summary.get("requires_verification", False)
        or verification_functions
    )
    return GoalVerificationSchema(
        required_for_completion=required_for_completion,
        verification_functions=verification_functions,
        success_criteria=list(success_criteria),
        failure_mode=str(plan_summary.get("verification_failure_mode", "block_completion") or "block_completion"),
        verification_scope="task_graph",
    )


def _derive_goal_completion_schema(
    *,
    success_criteria: List[str],
    verification: GoalVerificationSchema,
) -> GoalCompletionSchema:
    requires_goal_success_criteria = bool(success_criteria)
    requires_verification = bool(verification.required_for_completion)
    completion_mode = "verified_terminal_state" if requires_verification else "terminal_state"
    return GoalCompletionSchema(
        completion_mode=completion_mode,
        requires_all_nodes_terminal=True,
        terminal_statuses=["completed", "skipped"],
        requires_goal_success_criteria=requires_goal_success_criteria,
        requires_verification=requires_verification,
    )


def _title_looks_high_risk(title: str) -> bool:
    lowered = str(title or "").strip().lower()
    return bool(lowered) and any(token in lowered for token in _HIGH_RISK_TITLE_TOKENS)


def _title_looks_like_verification(title: str) -> bool:
    lowered = str(title or "").strip().lower()
    if not lowered:
        return False
    tokens = {
        part
        for part in re.split(r"[^a-z0-9]+", lowered)
        if part
    }
    return bool(tokens.intersection(set(_VERIFICATION_TITLE_TOKENS)))


def _fallback_verification_gate(title: str, success_criteria: List[str]) -> TaskVerificationGate:
    is_verification_step = _title_looks_like_verification(title)
    return TaskVerificationGate(
        required=False,
        mode="observation" if is_verification_step else "none",
        verifier_function="",
        success_criteria=list(success_criteria),
        failure_mode="block",
        last_verified=False,
        last_verdict="not_required",
        evidence={},
    )


def _fallback_retry_policy(title: str) -> TaskRetryPolicy:
    high_risk = _title_looks_high_risk(title)
    return TaskRetryPolicy(
        max_attempts=2 if high_risk else 1,
        on_failure="block" if high_risk else "revise",
        backoff="none",
    )


def _fallback_assignment(title: str) -> TaskAssignment:
    worker_type = "verifier" if _title_looks_like_verification(title) else "executor"
    return TaskAssignment(
        worker_id="primary",
        worker_type=worker_type,
        ownership="exclusive",
        source="derived",
    )


def _fallback_approval_requirement(title: str) -> TaskApprovalRequirement:
    if not _title_looks_high_risk(title):
        return TaskApprovalRequirement()
    return TaskApprovalRequirement(
        required=True,
        risk_level="high",
        capability_class="high_risk_action",
        reason=f"title:{title}",
    )


def _build_task_nodes(
    *,
    goal_contract: GoalContract,
    current_task: str,
    goal_agenda: List[Dict[str, Any]],
    long_horizon_commitments: List[Dict[str, Any]],
    plan_summary: Mapping[str, Any],
) -> List[TaskNode]:
    nodes: List[TaskNode] = []
    seen: set[str] = set()

    def add_node(
        title: str,
        *,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[List[str]] = None,
    ) -> None:
        clean_title = str(title or "").strip()
        if not clean_title or clean_title in seen:
            return
        seen.add(clean_title)
        node_success_criteria = list(success_criteria or [])
        nodes.append(
            TaskNode(
                node_id=_stable_id("task", goal_contract.goal_id, clean_title),
                title=clean_title,
                status=status,
                goal_id=goal_contract.goal_id,
                success_criteria=node_success_criteria,
                verification_gate=_fallback_verification_gate(clean_title, node_success_criteria),
                retry_policy=_fallback_retry_policy(clean_title),
                assigned_worker=_fallback_assignment(clean_title),
                approval_requirement=_fallback_approval_requirement(clean_title),
                llm_route_policies={},
                provenance={"source": "goal_task_runtime"},
                metadata=dict(metadata or {}),
            )
        )

    if current_task:
        add_node(
            current_task,
            status="active",
            metadata={
                "current_step_index": int(plan_summary.get("current_step_index", 0) or 0),
                "total_steps": int(plan_summary.get("total_steps", 0) or 0),
            },
        )
    for row in goal_agenda[:6]:
        title = str(row.get("title") or row.get("goal") or row.get("summary") or "").strip()
        add_node(
            title,
            status="queued" if title and title != current_task else "active",
            metadata={"source": "goal_agenda"},
            success_criteria=_string_list(row.get("success_criteria", [])),
        )
    for row in long_horizon_commitments[:4]:
        title = str(row.get("title") or row.get("commitment") or row.get("summary") or "").strip()
        add_node(
            title,
            status="committed",
            metadata={"source": "long_horizon_commitment"},
            success_criteria=_string_list(row.get("success_criteria", [])),
        )
    if not nodes:
        add_node(goal_contract.title, status="active", metadata={"source": "goal_fallback"})
    return nodes


def _derive_binding_completion_gate(
    *,
    goal_contract: Optional[GoalContract],
    task_graph: Optional[TaskGraph],
    active_task: Optional[TaskNode],
    payload: Any,
) -> Dict[str, Any]:
    gate = _dict_or_empty(payload)
    if not gate:
        completion = goal_contract.completion if goal_contract is not None else GoalCompletionSchema()
        verification = goal_contract.verification if goal_contract is not None else GoalVerificationSchema()
        nodes = list(task_graph.nodes or []) if task_graph is not None else []
        required_verification_node_ids: List[str] = []
        pending_verification_node_ids: List[str] = []
        failed_verification_node_ids: List[str] = []
        verified_node_ids: List[str] = []
        for node in nodes:
            gate_payload = resolve_effective_task_verification_gate(goal_contract, node)
            if not bool(gate_payload.get("required", False)):
                continue
            node_id = str(node.node_id or "")
            required_verification_node_ids.append(node_id)
            verdict = str(gate_payload.get("last_verdict", "pending") or "pending")
            if verdict == "failed":
                failed_verification_node_ids.append(node_id)
            elif bool(gate_payload.get("last_verified", False)):
                verified_node_ids.append(node_id)
            else:
                pending_verification_node_ids.append(node_id)
        active_gate = resolve_effective_task_verification_gate(goal_contract, active_task)
        active_task_completion_ready = not bool(active_gate.get("required", False)) or bool(active_gate.get("last_verified", False))
        all_nodes_terminal = bool(nodes) and all(
            str(node.status or "") in set(completion.terminal_statuses or ["completed", "skipped"])
            for node in nodes
        )
        requires_verification = bool(completion.requires_verification or verification.required_for_completion)
        verification_ready = not requires_verification
        if requires_verification:
            verification_ready = (
                not pending_verification_node_ids
                and not failed_verification_node_ids
                and bool(required_verification_node_ids)
            )
        blocked_reasons: List[str] = []
        if completion.requires_all_nodes_terminal and not all_nodes_terminal:
            blocked_reasons.append("awaiting_terminal_nodes")
        if requires_verification and failed_verification_node_ids:
            blocked_reasons.append("verification_failed")
        elif requires_verification and not verification_ready:
            blocked_reasons.append("verification_incomplete")
        if completion.requires_goal_success_criteria:
            blocked_reasons.append("goal_success_criteria_unsatisfied")
        gate = {
            "active_task_completion_ready": active_task_completion_ready,
            "goal_completion_ready": not blocked_reasons,
            "verification_ready": verification_ready,
            "goal_success_satisfied": not completion.requires_goal_success_criteria,
            "all_nodes_terminal": all_nodes_terminal,
            "requires_verification": requires_verification,
            "requires_goal_success_criteria": bool(completion.requires_goal_success_criteria),
            "requires_all_nodes_terminal": bool(completion.requires_all_nodes_terminal),
            "required_verification_node_ids": required_verification_node_ids,
            "pending_verification_node_ids": pending_verification_node_ids,
            "failed_verification_node_ids": failed_verification_node_ids,
            "verified_node_ids": verified_node_ids,
            "completed_verification_node_ids": [],
            "blocked_reasons": blocked_reasons,
        }
    gate.setdefault("failed_verification_node_ids", [])
    from core.orchestration.verifier_runtime import build_verifier_runtime

    verifier_runtime = build_verifier_runtime(
        goal_contract=goal_contract,
        task_graph=task_graph,
        active_task=active_task,
        completion_gate=gate,
    )
    gate["verifier_authority"] = dict(verifier_runtime.verifier_authority)
    gate["verifier_runtime_version"] = str(verifier_runtime.runtime_version)
    return gate


def derive_verifier_authority_snapshot(
    *,
    goal_contract: Any,
    task_graph: Any = None,
    active_task: Any = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    contract = _coerce_goal_contract(goal_contract)
    graph = _coerce_task_graph(task_graph, contract)
    task = _coerce_task_node(active_task, contract.goal_id if contract is not None else "")
    if task is None and graph is not None:
        task = next(
            (
                node
                for node in list(graph.nodes or [])
                if str(node.node_id or "") == str(graph.active_node_id or "")
            ),
            graph.nodes[0] if graph.nodes else None,
        )
    verification = contract.verification if contract is not None else GoalVerificationSchema()
    completion_snapshot = _dict_or_empty(completion_gate)
    active_gate = resolve_effective_task_verification_gate(contract, task)
    required = bool(
        completion_snapshot.get("requires_verification", False)
        or verification.required_for_completion
        or active_gate.get("required", False)
    )
    blocked_reasons = {
        str(item or "").strip()
        for item in list(completion_snapshot.get("blocked_reasons", []) or [])
        if str(item or "").strip()
    }
    pending_node_ids = _string_list(completion_snapshot.get("pending_verification_node_ids", []))
    failed_node_ids = _string_list(completion_snapshot.get("failed_verification_node_ids", []))
    verified_node_ids = _string_list(completion_snapshot.get("verified_node_ids", []))
    active_verdict = str(active_gate.get("last_verdict", "not_required") or "not_required")
    if required:
        if failed_node_ids or "verification_failed" in blocked_reasons or active_verdict == "failed":
            verdict = "failed"
        elif pending_node_ids or "verification_incomplete" in blocked_reasons:
            verdict = "pending"
        elif bool(completion_snapshot.get("verification_ready", False)) or verified_node_ids or active_verdict == "passed":
            verdict = "passed"
        else:
            verdict = "pending"
    else:
        verdict = "not_required"
    blocked_reason = ""
    if verdict == "failed":
        blocked_reason = "verification_failed"
    elif verdict == "pending" and required:
        blocked_reason = "verification_incomplete"
    rollback_edge = dict(task.rollback_edge) if task is not None else {}
    rollback_target_node_id = str(rollback_edge.get("target_node_id", "") or "")
    rollback_target_step_id = str(rollback_edge.get("target_step_id", "") or "")
    contradiction_detected = verdict == "failed"
    rollback_eligible = contradiction_detected and bool(rollback_target_node_id or rollback_target_step_id)
    teaching_signal = "positive" if verdict == "passed" else ("negative" if verdict == "failed" else "none")
    teaching_signal_score = 0.75 if teaching_signal == "positive" else (-1.0 if teaching_signal == "negative" else 0.0)
    evidence = _dict_or_empty(active_gate.get("evidence", {}))
    return {
        "goal_ref": contract.goal_id if contract is not None else "",
        "graph_ref": graph.graph_id if graph is not None else "",
        "task_ref": task.node_id if task is not None else "",
        "verification_scope": str(verification.verification_scope or "task_graph"),
        "required": required,
        "verdict": verdict,
        "active_task_verdict": active_verdict,
        "decision": (
            "allow_completion"
            if verdict in {"passed", "not_required"}
            else ("block_completion" if verdict == "failed" else "await_verification")
        ),
        "completion_blocked": bool(required and verdict in {"pending", "failed"}),
        "blocked_reason": blocked_reason,
        "contradiction_detected": contradiction_detected,
        "verifier_function": str(active_gate.get("verifier_function", "") or ""),
        "evidence_present": bool(evidence),
        "evidence_keys": sorted(evidence.keys()),
        "pending_node_ids": pending_node_ids,
        "failed_node_ids": failed_node_ids,
        "verified_node_ids": verified_node_ids,
        "rollback_eligible": rollback_eligible,
        "rollback_target_node_id": rollback_target_node_id,
        "rollback_target_step_id": rollback_target_step_id,
        "rollback_reason": str(rollback_edge.get("reason", "") or ""),
        "teaching_signal": teaching_signal,
        "teaching_signal_score": teaching_signal_score,
    }

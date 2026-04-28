from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence


ACTION_GOVERNANCE_VERSION = "conos.action_governance/v1"

READ_PERMISSIONS = frozenset({"read_files", "read_context", "network_read"})
MIRROR_WRITE_PERMISSIONS = frozenset({"propose_patch", "edit_mirror"})
SOURCE_WRITE_PERMISSIONS = frozenset({"write_source", "sync_source"})
VALIDATION_PERMISSIONS = frozenset({"run_tests"})
EXEC_PERMISSIONS = frozenset({"exec_command"})

SENSITIVE_ARG_TOKENS = frozenset({
    "api_key",
    "auth",
    "authorization",
    "bearer",
    "credential",
    "oauth",
    "password",
    "private_key",
    "secret",
    "secret_id",
    "token",
})

CODE_SUFFIXES = frozenset({
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
})


def _now() -> float:
    return float(time.time())


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


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


def _path_in_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _changed_path_needs_validation(path: str, code_suffixes: Sequence[str]) -> bool:
    suffix = Path(path).suffix.lower()
    return bool(suffix and suffix in set(code_suffixes))


@dataclass(frozen=True)
class ActionGovernancePolicy:
    require_evidence_before_patch: bool = True
    require_test_before_source_sync: bool = True
    require_approval_for_source_sync: bool = True
    forbid_cross_root_paths: bool = True
    forbid_inline_credentials: bool = True
    require_bounded_exec: bool = True
    require_approval_for_private_network: bool = True
    max_failures_before_downgrade: int = 2
    code_suffixes_requiring_validation: tuple[str, ...] = tuple(sorted(CODE_SUFFIXES))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionGovernanceState:
    evidence_refs: list[str] = field(default_factory=list)
    validation_runs: list[Dict[str, Any]] = field(default_factory=list)
    passing_tests: int = 0
    failure_count_by_agent: Dict[str, int] = field(default_factory=dict)
    downgraded_agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_root: str = ""
    allowed_roots: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionGovernanceRequest:
    agent_id: str
    action_name: str
    permissions_required: list[str] = field(default_factory=list)
    path_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    risk_level: str = "low"
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionGovernanceDecision:
    status: str
    blocked_reason: str = ""
    required_evidence: list[str] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)
    required_approval: bool = False
    effective_permissions: list[str] = field(default_factory=list)
    audit_event: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce_action_governance_policy(value: Any = None) -> ActionGovernancePolicy:
    if isinstance(value, ActionGovernancePolicy):
        return value
    payload = _dict_or_empty(value)
    if not payload:
        return ActionGovernancePolicy()
    suffixes = _string_list(payload.get("code_suffixes_requiring_validation"))
    return ActionGovernancePolicy(
        require_evidence_before_patch=bool(payload.get("require_evidence_before_patch", True)),
        require_test_before_source_sync=bool(payload.get("require_test_before_source_sync", True)),
        require_approval_for_source_sync=bool(payload.get("require_approval_for_source_sync", True)),
        forbid_cross_root_paths=bool(payload.get("forbid_cross_root_paths", True)),
        forbid_inline_credentials=bool(payload.get("forbid_inline_credentials", True)),
        require_bounded_exec=bool(payload.get("require_bounded_exec", True)),
        require_approval_for_private_network=bool(payload.get("require_approval_for_private_network", True)),
        max_failures_before_downgrade=max(1, int(payload.get("max_failures_before_downgrade", 2) or 2)),
        code_suffixes_requiring_validation=tuple(suffixes or sorted(CODE_SUFFIXES)),
    )


def derive_action_permissions(action_name: str) -> list[str]:
    name = str(action_name or "").strip()
    if name in {"repo_tree", "repo_find", "repo_grep", "file_read", "file_outline", "file_summary", "read_run_output", "read_test_failure"}:
        return ["read_files"]
    if name in {"note_write", "hypothesis_add", "hypothesis_update", "hypothesis_compete", "discriminating_test_add", "candidate_files_set", "candidate_files_update", "investigation_status"}:
        return ["read_context"]
    if name in {"apply_patch", "edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        return ["propose_patch", "edit_mirror"]
    if name in {"run_test", "run_lint", "run_typecheck", "run_build"}:
        return ["run_tests"]
    if name == "mirror_apply":
        return ["sync_source", "write_source"]
    if name == "mirror_exec":
        return ["exec_command"]
    if name in {"internet_fetch", "internet_fetch_project"}:
        return ["network_read"]
    return []


def _sensitive_arg_paths(payload: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key or "").strip()
            lowered = key_text.lower()
            path = f"{prefix}.{key_text}" if prefix else key_text
            if lowered in SENSITIVE_ARG_TOKENS or any(token in lowered for token in SENSITIVE_ARG_TOKENS):
                paths.append(path)
            paths.extend(_sensitive_arg_paths(value, prefix=path))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            paths.extend(_sensitive_arg_paths(item, prefix=f"{prefix}[{index}]" if prefix else f"[{index}]"))
    return paths


def derive_action_governance_request(
    action_name: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    agent_id: str = "local_machine",
    metadata: Mapping[str, Any] | None = None,
) -> ActionGovernanceRequest:
    payload = dict(kwargs or {})
    meta = dict(metadata or {})
    path_refs: list[str] = []
    for key in ("path", "root", "target", "plan_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            path_refs.append(value.strip())
    path_refs.extend(_string_list(payload.get("paths") or payload.get("files") or meta.get("changed_paths")))
    evidence_refs = _string_list(payload.get("evidence_refs") or payload.get("evidence_ref"))
    sensitive_arg_paths = _sensitive_arg_paths(payload)
    request_id = str(meta.get("request_id") or "") or f"agov_{_hash_payload([action_name, payload, meta])[:16]}"
    if sensitive_arg_paths:
        meta["sensitive_arg_paths"] = sensitive_arg_paths
    return ActionGovernanceRequest(
        agent_id=str(agent_id or "local_machine"),
        action_name=str(action_name or ""),
        permissions_required=derive_action_permissions(action_name),
        path_refs=path_refs,
        evidence_refs=evidence_refs,
        risk_level=str(meta.get("risk_level") or "low"),
        request_id=request_id,
        metadata=meta,
    )


def _audit_event(
    request: ActionGovernanceRequest,
    decision_status: str,
    *,
    reason: str = "",
    required_approval: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": ACTION_GOVERNANCE_VERSION,
        "event_type": "action_governance_decision",
        "created_at": _now(),
        "request_id": request.request_id,
        "agent_id": request.agent_id,
        "action_name": request.action_name,
        "permissions_required": list(request.permissions_required),
        "status": decision_status,
        "reason": str(reason or ""),
        "required_approval": bool(required_approval),
    }


def _decision(
    status: str,
    request: ActionGovernanceRequest,
    *,
    reason: str = "",
    required_evidence: Sequence[str] = (),
    required_tests: Sequence[str] = (),
    required_approval: bool = False,
) -> ActionGovernanceDecision:
    return ActionGovernanceDecision(
        status=status,
        blocked_reason=str(reason or ""),
        required_evidence=list(required_evidence),
        required_tests=list(required_tests),
        required_approval=bool(required_approval),
        effective_permissions=list(request.permissions_required),
        audit_event=_audit_event(request, status, reason=reason, required_approval=required_approval),
        request=request.to_dict(),
    )


def evaluate_action_governance(
    request: ActionGovernanceRequest | Mapping[str, Any],
    state: ActionGovernanceState | Mapping[str, Any] | None = None,
    policy: ActionGovernancePolicy | Mapping[str, Any] | None = None,
) -> ActionGovernanceDecision:
    req = request if isinstance(request, ActionGovernanceRequest) else ActionGovernanceRequest(**dict(request))
    gov_state = state if isinstance(state, ActionGovernanceState) else ActionGovernanceState(**_dict_or_empty(state))
    gov_policy = coerce_action_governance_policy(policy)
    permissions = set(req.permissions_required)

    if req.agent_id in gov_state.downgraded_agents and (permissions & (MIRROR_WRITE_PERMISSIONS | SOURCE_WRITE_PERMISSIONS | EXEC_PERMISSIONS)):
        return _decision(
            "DOWNGRADED",
            req,
            reason="agent_downgraded_after_repeated_action_failures",
        )

    sensitive_arg_paths = _string_list(req.metadata.get("sensitive_arg_paths"))
    if gov_policy.forbid_inline_credentials and sensitive_arg_paths:
        return _decision(
            "BLOCKED",
            req,
            reason=f"inline_credentials_not_allowed:{','.join(sensitive_arg_paths[:4])}",
        )

    if gov_policy.forbid_cross_root_paths and req.path_refs and gov_state.allowed_roots:
        roots = [Path(root) for root in gov_state.allowed_roots if str(root).strip()]
        for raw_path in req.path_refs:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                if not gov_state.source_root:
                    continue
                candidate = Path(gov_state.source_root).expanduser() / candidate
            if not any(_path_in_root(candidate, root) for root in roots):
                return _decision("BLOCKED", req, reason=f"path_outside_allowed_roots:{raw_path}")

    if "exec_command" in permissions and gov_policy.require_bounded_exec:
        if bool(req.metadata.get("generated_command", False)):
            purpose = str(req.metadata.get("purpose") or "").strip().lower()
            if purpose not in {"inspect", "test", "format", "build"}:
                return _decision("BLOCKED", req, reason="bounded_exec_requires_purpose")
            if not bool(req.metadata.get("timeout_seconds_present", False)):
                return _decision("BLOCKED", req, reason="bounded_exec_requires_timeout")
            if not bool(req.metadata.get("bounded_target_present", False)):
                return _decision("BLOCKED", req, reason="bounded_exec_requires_target")

    if "network_read" in permissions and gov_policy.require_approval_for_private_network:
        if bool(req.metadata.get("private_networks_allowed", False)):
            return _decision(
                "WAITING_APPROVAL",
                req,
                reason="private_network_access_requires_approval",
                required_approval=True,
            )

    available_evidence = _string_list(req.evidence_refs) or _string_list(gov_state.evidence_refs)
    if permissions & MIRROR_WRITE_PERMISSIONS and gov_policy.require_evidence_before_patch and not available_evidence:
        return _decision(
            "BLOCKED",
            req,
            reason="evidence_refs_required_before_mirror_write",
            required_evidence=["file_read", "repo_grep", "note_write.evidence_refs", "hypothesis_update.evidence_refs"],
        )

    if permissions & SOURCE_WRITE_PERMISSIONS:
        changed_paths = _string_list(req.metadata.get("changed_paths"))
        code_changes = [
            path
            for path in changed_paths
            if _changed_path_needs_validation(path, gov_policy.code_suffixes_requiring_validation)
        ]
        if gov_policy.require_test_before_source_sync and code_changes and int(gov_state.passing_tests) <= 0:
            return _decision(
                "BLOCKED",
                req,
                reason="passing_validation_required_before_source_sync",
                required_tests=["run_test", "run_lint", "run_typecheck", "run_build"],
            )
        approval_status = str(req.metadata.get("approval_status") or "").strip()
        approved_by = str(req.metadata.get("approved_by") or "").strip()
        already_approved = approval_status in {"machine_approved", "human_approved", "approved"} or bool(approved_by)
        if gov_policy.require_approval_for_source_sync and not already_approved:
            return _decision(
                "WAITING_APPROVAL",
                req,
                reason="source_sync_requires_approved_plan",
                required_approval=True,
            )

    return _decision("ALLOWED", req)


def record_action_governance_result(
    state: ActionGovernanceState | Mapping[str, Any] | None,
    request: ActionGovernanceRequest | Mapping[str, Any],
    *,
    success: bool,
    failure_reason: str = "",
    policy: ActionGovernancePolicy | Mapping[str, Any] | None = None,
) -> ActionGovernanceState:
    gov_state = state if isinstance(state, ActionGovernanceState) else ActionGovernanceState(**_dict_or_empty(state))
    req = request if isinstance(request, ActionGovernanceRequest) else ActionGovernanceRequest(**dict(request))
    gov_policy = coerce_action_governance_policy(policy)
    failures = dict(gov_state.failure_count_by_agent)
    downgraded = dict(gov_state.downgraded_agents)
    if success:
        failures[req.agent_id] = 0
    else:
        failures[req.agent_id] = int(failures.get(req.agent_id, 0) or 0) + 1
        if failures[req.agent_id] >= gov_policy.max_failures_before_downgrade:
            downgraded[req.agent_id] = {
                "agent_id": req.agent_id,
                "downgraded_at": _now(),
                "failure_count": failures[req.agent_id],
                "last_action_name": req.action_name,
                "last_failure_reason": str(failure_reason or ""),
                "restricted_permissions": sorted(MIRROR_WRITE_PERMISSIONS | SOURCE_WRITE_PERMISSIONS | EXEC_PERMISSIONS),
            }
    return ActionGovernanceState(
        evidence_refs=list(gov_state.evidence_refs),
        validation_runs=list(gov_state.validation_runs),
        passing_tests=int(gov_state.passing_tests),
        failure_count_by_agent=failures,
        downgraded_agents=downgraded,
        source_root=gov_state.source_root,
        allowed_roots=list(gov_state.allowed_roots),
        candidate_files=list(gov_state.candidate_files),
        metadata=dict(gov_state.metadata),
    )


def governance_state_from_local_machine_investigation(
    investigation: Mapping[str, Any] | None,
    *,
    source_root: str = "",
    allowed_roots: Sequence[str] = (),
) -> ActionGovernanceState:
    payload = _dict_or_empty(investigation)
    refs: list[str] = []
    for note in list(payload.get("notes", []) or []):
        if isinstance(note, Mapping):
            refs.extend(_string_list(note.get("evidence_refs")))
    for row in list(payload.get("hypotheses", []) or []):
        if isinstance(row, Mapping):
            refs.extend(_string_list(row.get("evidence_refs")))
    for row in list(payload.get("hypothesis_events", []) or []):
        if isinstance(row, Mapping):
            refs.extend(_string_list(row.get("evidence_refs")))
    last_read = _dict_or_empty(payload.get("last_read"))
    if last_read.get("path"):
        refs.append(
            f"file:{last_read.get('path')}:{last_read.get('start_line', '')}-{last_read.get('end_line', '')}"
        )
    last_search = _dict_or_empty(payload.get("last_search"))
    if last_search.get("query"):
        refs.append(
            f"grep:{last_search.get('root', '.') or '.'}:{last_search.get('query')}:{last_search.get('match_count', '')}"
        )
    last_tree = _dict_or_empty(payload.get("last_tree"))
    if last_tree.get("path") or last_tree.get("root"):
        refs.append(f"tree:{last_tree.get('path') or last_tree.get('root')}")
    validation_runs = [
        dict(row)
        for row in list(payload.get("validation_runs", []) or [])
        if isinstance(row, Mapping)
    ]
    passing = sum(1 for row in validation_runs if bool(row.get("success")) or int(row.get("returncode", 1) or 1) == 0)
    governance = _dict_or_empty(payload.get("action_governance"))
    return ActionGovernanceState(
        evidence_refs=_string_list(refs),
        validation_runs=validation_runs,
        passing_tests=passing,
        failure_count_by_agent=_dict_or_empty(governance.get("failure_count_by_agent")),
        downgraded_agents=_dict_or_empty(governance.get("downgraded_agents")),
        source_root=str(source_root or ""),
        allowed_roots=_string_list(allowed_roots),
        candidate_files=_string_list(payload.get("candidate_files")),
        metadata={"source": "local_machine_investigation"},
    )


def render_action_governance_decision(decision: ActionGovernanceDecision | Mapping[str, Any]) -> str:
    payload = decision.to_dict() if isinstance(decision, ActionGovernanceDecision) else dict(decision)
    status = str(payload.get("status") or "")
    reason = str(payload.get("blocked_reason") or "")
    action = str(_dict_or_empty(payload.get("request")).get("action_name") or "")
    if reason:
        return f"{status}: {action} blocked by action governance ({reason})"
    return f"{status}: {action} accepted by action governance"

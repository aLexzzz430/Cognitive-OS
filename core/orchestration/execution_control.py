from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

from core.conos_kernel import (
    ExecutionTicket,
    ToolCapabilityManifest,
    build_audit_event,
    build_task_contract,
)
from core.orchestration.action_utils import extract_action_function_name, extract_action_signature_kwargs
from core.orchestration.goal_task_control import (
    resolve_effective_task_approval_requirement,
)
from core.orchestration.verifier_runtime import (
    build_verifier_runtime,
    canonicalize_execution_authority_verifier_authority as runtime_canonicalize_execution_authority_verifier_authority,
    canonicalize_task_contract_verifier_authority as runtime_canonicalize_task_contract_verifier_authority,
    derive_contextual_execution_authority,
    resolve_effective_verifier_authority as runtime_resolve_effective_verifier_authority,
    verifier_authority_from_task_contract,
)


_DEFAULT_APPROVAL_REQUIRED_SIDE_EFFECTS = frozenset(
    {
        "external_submission",
        "filesystem_write",
        "system_mutation",
        "network_mutation",
    }
)

SANDBOX_BEST_EFFORT_AUDIT_VERSION = "conos.sandbox_best_effort_audit/v1"

_NETWORK_FUNCTION_TOKENS = frozenset(
    {
        "api",
        "connect",
        "download",
        "fetch",
        "http",
        "request",
        "send",
        "socket",
        "submit",
        "tcp",
        "upload",
        "url",
        "web",
    }
)

_NETWORK_ARG_TOKENS = frozenset(
    {
        "base_url",
        "endpoint",
        "host",
        "hostname",
        "url",
        "uri",
        "webhook",
    }
)

_FILE_ARG_TOKENS = frozenset(
    {
        "dir",
        "directory",
        "file",
        "filename",
        "folder",
        "input_path",
        "output",
        "output_file",
        "output_path",
        "path",
        "save_path",
        "source_path",
        "target_path",
        "workspace",
    }
)

_WRITE_ARG_TOKENS = frozenset(
    {
        "dest",
        "destination",
        "output",
        "output_file",
        "output_path",
        "patch",
        "save",
        "save_path",
        "target_path",
        "write",
    }
)

_WRITE_FUNCTION_TOKENS = frozenset(
    {
        "append",
        "commit",
        "create",
        "delete",
        "deploy",
        "edit",
        "move",
        "patch",
        "publish",
        "remove",
        "save",
        "update",
        "upload",
        "write",
    }
)

_CREDENTIAL_ARG_TOKENS = frozenset(
    {
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
    }
)

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

_OBSERVED_TOOL_READ_ONLY_TOKENS = frozenset(
    {
        "inspect",
        "probe",
        "wait",
        "measure",
        "scan",
        "survey",
        "read",
        "observe",
        "check",
        "verify",
    }
)

_VERIFIER_COMPATIBLE_FUNCTION_NAMES = frozenset(
    {
        "inspect",
        "probe",
        "verify",
        "check",
        "measure",
    }
)

_OBSERVED_TOOL_APPROVAL_EXACT_NAMES = (
    (
        "external_submission",
        "form_submission",
        "high",
        frozenset(
            {
                "submit",
                "submit_form",
                "submit_order",
                "submit_checkout_form",
                "checkout",
                "place_order",
                "complete_purchase",
                "confirm_purchase",
                "send_message",
                "send_email",
                "publish_post",
            }
        ),
    ),
    (
        "filesystem_write",
        "filesystem_mutation",
        "high",
        frozenset(
            {
                "write_file",
                "save_file",
                "delete_file",
                "remove_file",
                "update_file",
                "patch_file",
                "apply_patch",
            }
        ),
    ),
    (
        "system_mutation",
        "system_mutation",
        "high",
        frozenset(
            {
                "commit",
                "finalize",
                "publish",
                "deploy",
                "force_submit",
                "force_delete",
            }
        ),
    ),
)


RegisteredTool = ToolCapabilityManifest


@dataclass(frozen=True)
class ApprovalDecision:
    allowed: bool
    reason: str
    policy_name: str
    function_name: str
    matched_tool_name: str = ""
    capability_class: str = ""
    side_effect_class: str = ""
    approval_required: bool = False
    risk_level: str = "low"
    approval_granted: bool = False
    approval_sources: List[str] = field(default_factory=list)
    blocked_by_contract: bool = False
    contract_scope: str = ""
    approval_grant_id: str = ""
    approval_scope_snapshot: Dict[str, Any] = field(default_factory=dict)
    required_secret_lease_ids: List[str] = field(default_factory=list)
    granted_secret_lease_ids: List[str] = field(default_factory=list)
    missing_secret_lease_ids: List[str] = field(default_factory=list)
    secret_lease_scope_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def serialize_tool_spec(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, dict):
        return {
            "tool_id": str(tool.get("tool_id", "") or tool.get("name", "") or ""),
            "name": str(tool.get("name", "") or ""),
            "description": str(tool.get("description", "") or ""),
            "input_schema": dict(tool.get("input_schema", {}) or {}) if isinstance(tool.get("input_schema", {}), dict) else {},
            "side_effects": [str(item) for item in list(tool.get("side_effects", []) or []) if str(item)],
            "risk_notes": [str(item) for item in list(tool.get("risk_notes", []) or []) if str(item)],
            "capability_class": str(tool.get("capability_class", "") or ""),
            "side_effect_class": str(tool.get("side_effect_class", "") or ""),
            "approval_required": bool(tool.get("approval_required", False)),
            "risk_level": str(tool.get("risk_level", "low") or "low"),
            "source": str(tool.get("source", "surface") or "surface"),
            "metadata": dict(tool.get("metadata", {}) or {}) if isinstance(tool.get("metadata", {}), dict) else {},
        }
    if is_dataclass(tool):
        payload = asdict(tool)
        return serialize_tool_spec(payload)
    name = str(getattr(tool, "name", "") or "")
    return {
        "tool_id": str(getattr(tool, "tool_id", "") or name),
        "name": name,
        "description": str(getattr(tool, "description", "") or ""),
        "input_schema": dict(getattr(tool, "input_schema", {}) or {}) if isinstance(getattr(tool, "input_schema", {}), dict) else {},
        "side_effects": [str(item) for item in list(getattr(tool, "side_effects", []) or []) if str(item)],
        "risk_notes": [str(item) for item in list(getattr(tool, "risk_notes", []) or []) if str(item)],
        "capability_class": str(getattr(tool, "capability_class", "") or ""),
        "side_effect_class": str(getattr(tool, "side_effect_class", "") or ""),
        "approval_required": bool(getattr(tool, "approval_required", False)),
        "risk_level": str(getattr(tool, "risk_level", "low") or "low"),
        "source": str(getattr(tool, "source", "surface") or "surface"),
        "metadata": dict(getattr(tool, "metadata", {}) or {}) if isinstance(getattr(tool, "metadata", {}), dict) else {},
    }


def coerce_registered_tool(tool: Any) -> Optional[RegisteredTool]:
    payload = serialize_tool_spec(tool)
    name = str(payload.get("name", "") or payload.get("tool_id", "") or "").strip()
    if not name:
        return None
    return RegisteredTool(
        name=name,
        description=str(payload.get("description", "") or ""),
        input_schema=dict(payload.get("input_schema", {}) or {}),
        side_effects=[str(item) for item in list(payload.get("side_effects", []) or []) if str(item)],
        risk_notes=[str(item) for item in list(payload.get("risk_notes", []) or []) if str(item)],
        capability_class=str(payload.get("capability_class", "") or ""),
        side_effect_class=str(payload.get("side_effect_class", "") or ""),
        approval_required=bool(payload.get("approval_required", False)),
        risk_level=str(payload.get("risk_level", "low") or "low"),
        source=str(payload.get("source", "surface") or "surface"),
        metadata=dict(payload.get("metadata", {}) or {}),
        tool_id=str(payload.get("tool_id", "") or name),
    )


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalized_name_set(values: Any) -> set[str]:
    return {
        str(item).strip().casefold()
        for item in list(values or [])
        if str(item).strip()
    }


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _audit_key_tokens(key: str) -> set[str]:
    clean = str(key or "").strip().lower().replace("-", "_").replace(".", "_")
    return {token for token in clean.split("_") if token}


def _audit_key_has_token(key: str, token_set: Iterable[str]) -> bool:
    clean = str(key or "").strip().lower().replace("-", "_").replace(".", "_")
    if not clean:
        return False
    tokens = _audit_key_tokens(clean)
    for token in token_set:
        needle = str(token or "").strip().lower().replace("-", "_")
        if not needle:
            continue
        if needle == clean or needle in tokens or needle in clean:
            return True
    return False


def _iter_audit_values(value: Any, *, prefix: str = "", depth: int = 0) -> List[tuple[str, Any]]:
    if depth > 4:
        return []
    rows: List[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in dict(value).items():
            child_key = str(key or "")
            child_prefix = f"{prefix}.{child_key}" if prefix else child_key
            rows.append((child_prefix, child))
            rows.extend(_iter_audit_values(child, prefix=child_prefix, depth=depth + 1))
    elif isinstance(value, list):
        for index, child in enumerate(value[:12]):
            child_prefix = f"{prefix}[{index}]"
            rows.append((child_prefix, child))
            rows.extend(_iter_audit_values(child, prefix=child_prefix, depth=depth + 1))
    return rows


def _action_audit_payload(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
    kwargs = extract_action_signature_kwargs(action)
    merged: Dict[str, Any] = {}
    for source in (action, payload, tool_args, kwargs):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if str(key).startswith("_") or key in {"payload", "tool_args"}:
                continue
            merged[str(key)] = value
    if kwargs:
        merged["kwargs"] = dict(kwargs)
    return merged


def _looks_like_file_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or _URL_RE.match(text):
        return False
    return (
        "/" in text
        or "\\" in text
        or text.startswith(("~", "."))
        or bool(re.search(r"\.[a-zA-Z0-9]{1,8}$", text))
    )


def _path_scope(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return "unknown"
    expanded = text.replace("\\", "/")
    if expanded.startswith("/Users/alexhuang/Downloads/Cognitive-OS-main/"):
        relative = expanded.split("/Users/alexhuang/Downloads/Cognitive-OS-main/", 1)[1]
        if relative.startswith(("runtime/", "reports/", "audit/")):
            return "repo_runtime_artifact"
        return "repo_source_tree"
    if expanded.startswith(("/tmp/", "/private/tmp/", "/var/folders/")):
        return "temporary_path"
    if expanded.startswith("/"):
        return "absolute_external_path"
    if expanded.startswith(("runtime/", "reports/", "audit/")):
        return "repo_runtime_artifact"
    if expanded.startswith(("~", "../")):
        return "relative_external_path"
    return "relative_path"


def _redacted_url_host(value: str) -> Dict[str, str]:
    parsed = urlparse(str(value or "").strip())
    return {
        "scheme": str(parsed.scheme or ""),
        "host": str(parsed.hostname or ""),
        "port": str(parsed.port or "") if parsed.port else "",
    }


def _build_sandbox_best_effort_audit(
    *,
    action: Optional[Dict[str, Any]],
    decision: ApprovalDecision,
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ctx = dict(context or {})
    fn_name = str(decision.function_name or extract_action_function_name(action, default="") or "").strip()
    fn_tokens = set(_function_name_tokens(fn_name))
    payload = _action_audit_payload(action)
    rows = _iter_audit_values(payload)
    side_effect_class = str(decision.side_effect_class or "").strip().lower()
    capability_class = str(decision.capability_class or "").strip().lower()
    write_like_function = bool(
        side_effect_class in {"filesystem_write", "filesystem_mutation", "system_mutation", "network_mutation"}
        or fn_tokens.intersection(_WRITE_FUNCTION_TOKENS)
    )
    network_like_function = bool(
        side_effect_class in {"network_mutation", "external_submission"}
        or capability_class in {"network", "form_submission"}
        or fn_tokens.intersection(_NETWORK_FUNCTION_TOKENS)
    )

    path_rows: List[Dict[str, Any]] = []
    write_paths: List[Dict[str, Any]] = []
    network_targets: List[Dict[str, str]] = []
    sensitive_keys: List[str] = []
    for key, value in rows:
        key_text = str(key or "")
        if _audit_key_has_token(key_text, _CREDENTIAL_ARG_TOKENS):
            sensitive_keys.append(key_text)
        if isinstance(value, str) and _URL_RE.match(value):
            target = _redacted_url_host(value)
            if target.get("host"):
                network_targets.append(target)
        elif _audit_key_has_token(key_text, _NETWORK_ARG_TOKENS) and isinstance(value, str) and value.strip():
            network_targets.append({"scheme": "", "host": str(value or "").strip(), "port": ""})
        if not isinstance(value, str):
            continue
        if not (_audit_key_has_token(key_text, _FILE_ARG_TOKENS) or _looks_like_file_path(value)):
            continue
        path_intent = "write" if (
            write_like_function or _audit_key_has_token(key_text, _WRITE_ARG_TOKENS)
        ) else "read_or_reference"
        path_row = {
            "key": key_text,
            "path": str(value or "").strip(),
            "intent": path_intent,
            "scope": _path_scope(str(value or "")),
        }
        path_rows.append(path_row)
        if path_intent == "write":
            write_paths.append(path_row)

    network_targets = [
        dict(item)
        for item in {
            json.dumps(item, sort_keys=True, separators=(",", ":")): item
            for item in network_targets
            if item.get("host")
        }.values()
    ]
    sensitive_keys = _ordered_unique(sensitive_keys)
    path_rows = [
        dict(item)
        for item in {
            json.dumps(item, sort_keys=True, separators=(",", ":")): item
            for item in path_rows
        }.values()
    ]
    write_paths = [item for item in path_rows if str(item.get("intent", "") or "") == "write"]
    write_path_unknown = bool(write_like_function and not write_paths)
    warnings: List[str] = ["sandbox_boundary_best_effort_only"]
    if write_path_unknown:
        warnings.append("write_effect_without_explicit_write_path")
    if sensitive_keys and not decision.required_secret_lease_ids and not decision.granted_secret_lease_ids:
        warnings.append("credential_like_argument_without_secret_lease")
    if network_like_function and not network_targets:
        warnings.append("network_effect_without_explicit_target")

    return {
        "audit_version": SANDBOX_BEST_EFFORT_AUDIT_VERSION,
        "sandbox_label": "best_effort",
        "security_boundary": "best_effort_policy_ticket_audit",
        "not_os_security_sandbox": True,
        "enforcement_model": {
            "policy_ticketing": True,
            "approval_gate": bool(decision.approval_required),
            "secret_lease_gate": bool(decision.required_secret_lease_ids),
            "os_process_isolation": False,
            "network_isolation": False,
            "filesystem_isolation": False,
        },
        "limitations": [
            "audit_and_policy_only",
            "does_not_guarantee_os_level_file_isolation",
            "does_not_guarantee_network_blocking",
            "credential_values_are_not_logged_but_callable_tool_may_receive_supplied_args",
        ],
        "file_audit": {
            "detected": bool(path_rows),
            "paths": path_rows,
            "path_count": len(path_rows),
        },
        "write_path_audit": {
            "detected": bool(write_paths or write_path_unknown),
            "paths": write_paths,
            "unknown_write_path": write_path_unknown,
            "path_scopes": _ordered_unique([str(item.get("scope", "") or "") for item in write_paths]),
        },
        "network_audit": {
            "detected": bool(network_like_function or network_targets),
            "targets": network_targets,
            "target_count": len(network_targets),
            "network_like_function": network_like_function,
        },
        "credential_audit": {
            "detected": bool(sensitive_keys or decision.required_secret_lease_ids or decision.granted_secret_lease_ids),
            "sensitive_arg_keys": sensitive_keys,
            "values_redacted": True,
            "required_secret_lease_ids": list(decision.required_secret_lease_ids or []),
            "granted_secret_lease_ids": list(decision.granted_secret_lease_ids or []),
            "missing_secret_lease_ids": list(decision.missing_secret_lease_ids or []),
            "secret_lease_scope_snapshots": [
                dict(item)
                for item in list(decision.secret_lease_scope_snapshots or [])
                if isinstance(item, dict)
            ],
        },
        "scope_binding": {
            "goal_ref": str(ctx.get("goal_ref", "") or ""),
            "task_ref": str(ctx.get("task_ref", "") or ""),
            "graph_ref": str(ctx.get("graph_ref", "") or ""),
            "run_id": str(ctx.get("run_id", "") or ""),
            "episode": int(ctx.get("episode", 0) or 0),
            "tick": int(ctx.get("tick", 0) or 0),
        },
        "warnings": _ordered_unique(warnings),
    }


def _evaluate_task_contract_freshness(
    *,
    task_contract: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    contract_payload = _dict_or_empty(task_contract)
    freshness = _dict_or_empty(contract_payload.get("freshness_binding", {}))
    if not freshness:
        return {}
    ctx = dict(context or {})
    goal_contract = _dict_or_empty(ctx.get("goal_contract", {}))
    task_graph = _dict_or_empty(ctx.get("task_graph", {}))
    goal_metadata = _dict_or_empty(goal_contract.get("metadata", {}))
    graph_metadata = _dict_or_empty(task_graph.get("metadata", {}))
    current_plan_id = str(
        graph_metadata.get("plan_id", "")
        or goal_metadata.get("plan_id", "")
        or freshness.get("source_plan_id", "")
        or ""
    ).strip()
    current_plan_revision = max(
        0,
        _int_or_default(
            graph_metadata.get(
                "revision_count",
                goal_metadata.get("revision_count", freshness.get("source_plan_revision", 0)),
            ),
            0,
        ),
    )
    current_graph_revision = max(
        0,
        _int_or_default(
            graph_metadata.get("graph_revision", graph_metadata.get("revision_count", freshness.get("source_graph_revision", 0))),
            0,
        ),
    )
    current_run_id = str(ctx.get("run_id", "") or "").strip()
    current_episode = max(0, _int_or_default(ctx.get("episode", 0), 0))
    current_tick = max(0, _int_or_default(ctx.get("tick", 0), 0))
    issued_run_id = str(freshness.get("issued_run_id", "") or "").strip()
    issued_episode = max(0, _int_or_default(freshness.get("issued_episode", 0), 0))
    issued_tick = max(0, _int_or_default(freshness.get("issued_tick", 0), 0))
    freshness_policy_class = str(
        freshness.get("freshness_policy_class", "same_tick_only") or "same_tick_only"
    ).strip().lower()
    tick_window = max(
        0,
        _int_or_default(
            freshness.get(
                "tick_window",
                2 if freshness_policy_class == "verifier_short_window" else 0,
            ),
            0,
        ),
    )
    valid_through_tick = max(
        issued_tick,
        _int_or_default(
            freshness.get("valid_through_tick", issued_tick + tick_window),
            issued_tick + tick_window,
        ),
    )
    stale_reasons: List[str] = []
    source_plan_id = str(freshness.get("source_plan_id", "") or "").strip()
    source_plan_revision = max(0, _int_or_default(freshness.get("source_plan_revision", 0), 0))
    source_graph_revision = max(0, _int_or_default(freshness.get("source_graph_revision", 0), 0))
    if source_plan_id and current_plan_id and source_plan_id != current_plan_id:
        stale_reasons.append("plan_mismatch")
    if source_plan_revision and current_plan_revision and source_plan_revision != current_plan_revision:
        stale_reasons.append("plan_revision_mismatch")
    if source_graph_revision and current_graph_revision and source_graph_revision != current_graph_revision:
        stale_reasons.append("graph_revision_mismatch")
    if issued_run_id and current_run_id and issued_run_id != current_run_id:
        stale_reasons.append("run_mismatch")
    if issued_episode and current_episode and issued_episode != current_episode:
        stale_reasons.append("episode_mismatch")
    if current_tick and valid_through_tick and current_tick > valid_through_tick:
        stale_reasons.append("tick_expired")
    stale = bool(stale_reasons)
    renewal_strategy = str(freshness.get("renewal_strategy", "explicit_renewal") or "explicit_renewal").strip()
    renewal_reasons = [
        reason
        for reason in stale_reasons
        if reason in {"tick_expired", "run_mismatch", "episode_mismatch"}
    ]
    renewal_required = bool(renewal_reasons) and renewal_strategy in {
        "explicit_renewal",
        "renew_verifier_contract",
    }
    evaluated = dict(freshness)
    evaluated.update(
        {
            "current_plan_id": current_plan_id,
            "current_plan_revision": current_plan_revision,
            "current_graph_revision": current_graph_revision,
            "current_run_id": current_run_id,
            "current_episode": current_episode,
            "current_tick": current_tick,
            "freshness_policy_class": freshness_policy_class,
            "tick_window": tick_window,
            "cross_tick_valid": bool(tick_window > 0),
            "same_tick_only": bool(tick_window == 0),
            "renewal_strategy": renewal_strategy,
            "expired": stale,
            "stale": stale,
            "stale_reasons": stale_reasons,
            "renewal_required": renewal_required,
            "renewal_reasons": renewal_reasons,
            "acceptable_for_execution": not stale,
        }
    )
    return evaluated


def _verifier_authority_from_task_contract(task_contract: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return verifier_authority_from_task_contract(task_contract)


def resolve_task_contract_verifier_authority(task_contract: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return _verifier_authority_from_task_contract(task_contract)


def _derive_contextual_execution_authority(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return derive_contextual_execution_authority(context)


def resolve_effective_verifier_authority(
    *,
    task_contract: Optional[Mapping[str, Any]] = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    execution_authority: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return runtime_resolve_effective_verifier_authority(
        task_contract=task_contract,
        completion_gate=completion_gate,
        execution_authority=execution_authority,
        context=context,
    )


def _authority_signature(authority: Mapping[str, Any]) -> str:
    payload = _dict_or_empty(authority)
    if not payload:
        return ""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        safe_payload = json.loads(json.dumps(payload, default=str))
        return json.dumps(safe_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _collect_task_contract_authority_drift(
    *,
    context: Optional[Mapping[str, Any]],
    task_contract: Optional[Mapping[str, Any]],
    execution_authority: Optional[Mapping[str, Any]],
) -> List[str]:
    contract_payload = _dict_or_empty(task_contract)
    canonical_authority = _verifier_authority_from_task_contract(contract_payload)
    if not contract_payload or not canonical_authority:
        return []

    drift_sources: List[str] = []
    canonical_signature = _authority_signature(canonical_authority)
    verification_authority = dict(
        _dict_or_empty(
            _dict_or_empty(contract_payload.get("verification_requirement", {})).get("verifier_authority", {})
        )
    )
    completion_authority = dict(
        _dict_or_empty(
            _dict_or_empty(_dict_or_empty(contract_payload.get("completion", {})).get("completion_gate", {})).get(
                "verifier_authority", {}
            )
        )
    )
    if (
        verification_authority
        and completion_authority
        and _authority_signature(verification_authority) != _authority_signature(completion_authority)
    ):
        drift_sources.append("task_contract.completion_gate.verifier_authority")

    ctx = dict(context or {})
    contextual_authorities = (
        ("context.verifier_authority", _dict_or_empty(ctx.get("verifier_authority", {}))),
        (
            "context.completion_gate.verifier_authority",
            _dict_or_empty(_dict_or_empty(ctx.get("completion_gate", {})).get("verifier_authority", {})),
        ),
        (
            "execution_authority.verifier_authority",
            _dict_or_empty(_dict_or_empty(execution_authority or {}).get("verifier_authority", {})),
        ),
    )
    for label, payload in contextual_authorities:
        if payload and _authority_signature(payload) != canonical_signature:
            drift_sources.append(label)
    return _ordered_unique(drift_sources)


def _resolve_verifier_authority_context(
    *,
    context: Optional[Mapping[str, Any]],
    execution_authority: Mapping[str, Any],
) -> Dict[str, Any]:
    ctx = dict(context or {})
    runtime = build_verifier_runtime(
        task_contract=_dict_or_empty(ctx.get("task_contract", {})),
        completion_gate=_dict_or_empty(ctx.get("completion_gate", {})),
        execution_authority=execution_authority,
        context=ctx,
    )
    return dict(runtime.verifier_authority)


def _action_satisfies_verifier_authority(
    *,
    function_name: str,
    capability_class: str,
    verifier_authority: Mapping[str, Any],
) -> bool:
    function_key = str(function_name or "").strip().casefold()
    if not function_key:
        return False
    verifier_function = str(verifier_authority.get("verifier_function", "") or "").strip().casefold()
    capability_key = str(capability_class or "").strip().casefold()
    if verifier_function and function_key == verifier_function:
        return True
    if function_key in _VERIFIER_COMPATIBLE_FUNCTION_NAMES:
        return True
    return capability_key.startswith("verification")


def _canonical_secret_lease_id(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("lease_id", "")
        or payload.get("grant_id", "")
        or payload.get("ticket_scope_id", "")
        or payload.get("secret_id", "")
        or ""
    ).strip()


def _normalized_secret_lease_records(values: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(values or []):
        if isinstance(item, str):
            payload: Dict[str, Any] = {
                "lease_id": str(item).strip(),
                "required": True,
            }
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            continue
        lease_id = _canonical_secret_lease_id(payload)
        secret_id = str(payload.get("secret_id", "") or lease_id).strip()
        if not lease_id and not secret_id:
            continue
        normalized = dict(payload)
        if lease_id:
            normalized["lease_id"] = lease_id
        if secret_id:
            normalized["secret_id"] = secret_id
        normalized["required"] = bool(payload.get("required", True))
        rows.append(normalized)
    return rows


class ToolCapabilityRegistry:
    def __init__(self, tools: Optional[Iterable[Any]] = None) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        for tool in list(tools or []):
            self.register(tool)

    @staticmethod
    def _key(name: str) -> str:
        return str(name or "").strip().casefold()

    def register(self, tool: Any) -> Optional[RegisteredTool]:
        spec = coerce_registered_tool(tool)
        if spec is None:
            return None
        self._tools[self._key(spec.name)] = spec
        return spec

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(self._key(name))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {spec.name: spec.to_dict() for spec in self._tools.values()}

    @classmethod
    def from_available_tools(cls, tools: Optional[Iterable[Any]]) -> "ToolCapabilityRegistry":
        return cls(tools=tools)


class ApprovalPolicy:
    def __init__(
        self,
        *,
        policy_name: str = "default_execution_policy",
        deny_unregistered: bool = True,
        approval_required_side_effects: Optional[Iterable[str]] = None,
    ) -> None:
        self.policy_name = str(policy_name or "default_execution_policy")
        side_effects = approval_required_side_effects
        if side_effects is None:
            side_effects = _DEFAULT_APPROVAL_REQUIRED_SIDE_EFFECTS
        self._approval_required_side_effects = {
            str(item).strip()
            for item in list(side_effects or [])
            if str(item).strip()
        }
        self._deny_unregistered = bool(deny_unregistered)

    def evaluate(
        self,
        action: Optional[Dict[str, Any]],
        registry: ToolCapabilityRegistry,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ApprovalDecision:
        fn_name = str(extract_action_function_name(action, default="wait") or "wait").strip()
        if fn_name in {"", "wait"}:
            return ApprovalDecision(
                allowed=True,
                reason="non_side_effecting_action",
                policy_name=self.policy_name,
                function_name=fn_name or "wait",
                matched_tool_name=fn_name or "wait",
            )

        contract_signals = _approval_contract_signals(
            context=context,
            function_name=fn_name,
            capability_class="",
        )
        if bool(contract_signals.get("blocked_by_contract", False)):
            return ApprovalDecision(
                allowed=False,
                reason=str(contract_signals.get("blocked_reason", f"goal_contract_blocked_function:{fn_name}") or f"goal_contract_blocked_function:{fn_name}"),
                policy_name=self.policy_name,
                function_name=fn_name,
                matched_tool_name=fn_name,
                approval_required=True,
                risk_level="high",
                approval_granted=False,
                approval_sources=list(contract_signals.get("approval_sources", []) or []),
                blocked_by_contract=True,
                contract_scope=str(contract_signals.get("contract_scope", "goal_contract.approval") or "goal_contract.approval"),
            )

        if fn_name in {"inspect", "probe"} and not bool(contract_signals.get("approval_required", False)):
            return ApprovalDecision(
                allowed=True,
                reason="non_side_effecting_action",
                policy_name=self.policy_name,
                function_name=fn_name,
                matched_tool_name=fn_name,
                approval_sources=list(contract_signals.get("approval_sources", []) or []),
            )

        registered = registry.get(fn_name)
        if registered is None:
            return ApprovalDecision(
                allowed=not self._deny_unregistered,
                reason="tool_not_registered" if self._deny_unregistered else "tool_not_registered_default_allow",
                policy_name=self.policy_name,
                function_name=fn_name,
                matched_tool_name=fn_name,
                approval_sources=list(contract_signals.get("approval_sources", []) or []),
                contract_scope=str(contract_signals.get("contract_scope", "") or ""),
            )

        contract_signals = _approval_contract_signals(
            context=context,
            function_name=fn_name,
            capability_class=registered.capability_class,
        )
        if bool(contract_signals.get("blocked_by_contract", False)):
            return ApprovalDecision(
                allowed=False,
                reason=str(contract_signals.get("blocked_reason", f"goal_contract_blocked_function:{fn_name}") or f"goal_contract_blocked_function:{fn_name}"),
                policy_name=self.policy_name,
                function_name=fn_name,
                matched_tool_name=registered.name,
                capability_class=registered.capability_class,
                side_effect_class=registered.side_effect_class,
                approval_required=True,
                risk_level=registered.risk_level,
                approval_granted=False,
                approval_sources=list(contract_signals.get("approval_sources", []) or []),
                blocked_by_contract=True,
                contract_scope=str(contract_signals.get("contract_scope", "goal_contract.approval") or "goal_contract.approval"),
            )

        approval_required = bool(registered.approval_required)
        side_effect_class = str(registered.side_effect_class or "").strip()
        if side_effect_class and side_effect_class in self._approval_required_side_effects:
            approval_required = True
        approval_sources: List[str] = []
        if bool(registered.approval_required):
            approval_sources.append("tool_spec.approval_required")
        if side_effect_class and side_effect_class in self._approval_required_side_effects:
            approval_sources.append(f"side_effect_class:{side_effect_class}")
        if bool(contract_signals.get("approval_required", False)):
            approval_required = True
            approval_sources.extend(
                str(item)
                for item in list(contract_signals.get("approval_sources", []) or [])
                if str(item)
            )
        required_secret_leases = _normalized_secret_lease_records(
            contract_signals.get("required_secret_leases", [])
        )
        required_secret_lease_ids = _ordered_unique(
            [
                _canonical_secret_lease_id(item) or str(item.get("secret_id", "") or "").strip()
                for item in required_secret_leases
            ]
        )
        if required_secret_lease_ids:
            approval_sources.extend(
                str(item)
                for item in list(contract_signals.get("secret_lease_sources", []) or [])
                if str(item)
            )
        approval_sources = _ordered_unique(approval_sources)
        goal_ref = str((context or {}).get("goal_ref", "") or "").strip()
        task_ref = str((context or {}).get("task_ref", "") or "").strip()
        if (approval_required or required_secret_lease_ids) and (not goal_ref or not task_ref):
            return ApprovalDecision(
                allowed=False,
                reason=f"missing_goal_task_binding:{fn_name}",
                policy_name=self.policy_name,
                function_name=fn_name,
                matched_tool_name=registered.name,
                capability_class=registered.capability_class,
                side_effect_class=registered.side_effect_class,
                approval_required=approval_required,
                risk_level=registered.risk_level,
                approval_granted=False,
                approval_sources=approval_sources,
                contract_scope=str(contract_signals.get("contract_scope", "") or ""),
                required_secret_lease_ids=required_secret_lease_ids,
                missing_secret_lease_ids=required_secret_lease_ids,
            )

        approval_grant = _approval_granted(
            action=action,
            function_name=fn_name,
            capability_class=registered.capability_class,
            context=context,
        )
        secret_lease_result = _secret_lease_granted(
            required_secret_leases=required_secret_leases,
            function_name=fn_name,
            capability_class=registered.capability_class,
            context=context,
        )
        approval_granted = bool(approval_grant.get("granted", False))
        if approval_granted and str(approval_grant.get("source", "") or ""):
            approval_sources = _ordered_unique(
                approval_sources + [str(approval_grant.get("source", "") or "")]
            )
        granted_secret_lease_ids = _ordered_unique(
            [
                str(item).strip()
                for item in list(secret_lease_result.get("granted_secret_lease_ids", []) or [])
                if str(item).strip()
            ]
        )
        missing_secret_lease_ids = _ordered_unique(
            [
                str(item).strip()
                for item in list(secret_lease_result.get("missing_secret_lease_ids", []) or [])
                if str(item).strip()
            ]
        )
        if granted_secret_lease_ids:
            approval_sources = _ordered_unique(
                approval_sources + [str(secret_lease_result.get("source", "") or "approved_secret_lease")]
            )
        allowed = ((not approval_required) or approval_granted) and not missing_secret_lease_ids
        if missing_secret_lease_ids:
            reason = f"required_secret_leases_missing:{','.join(missing_secret_lease_ids)}"
        else:
            reason = "approved_or_not_required" if allowed else f"approval_required:{fn_name}"
        return ApprovalDecision(
            allowed=allowed,
            reason=reason,
            policy_name=self.policy_name,
            function_name=fn_name,
            matched_tool_name=registered.name,
            capability_class=registered.capability_class,
            side_effect_class=registered.side_effect_class,
            approval_required=approval_required,
            risk_level=registered.risk_level,
            approval_granted=approval_granted,
            approval_sources=approval_sources,
            contract_scope=str(contract_signals.get("contract_scope", "") or ""),
            approval_grant_id=str(approval_grant.get("grant_id", "") or ""),
            approval_scope_snapshot=dict(approval_grant.get("scope_snapshot", {}) or {}),
            required_secret_lease_ids=required_secret_lease_ids,
            granted_secret_lease_ids=granted_secret_lease_ids,
            missing_secret_lease_ids=missing_secret_lease_ids,
            secret_lease_scope_snapshots=[
                dict(item)
                for item in list(secret_lease_result.get("scope_snapshots", []) or [])
                if isinstance(item, dict)
            ],
        )


def issue_execution_ticket(
    *,
    action: Optional[Dict[str, Any]],
    decision: ApprovalDecision,
    context: Optional[Mapping[str, Any]] = None,
) -> ExecutionTicket:
    ctx = dict(context or {})
    task_contract_supplied = isinstance(ctx.get("task_contract", {}), dict) and bool(ctx.get("task_contract", {}))
    execution_authority = (
        dict(ctx.get("execution_authority", {}) or {})
        if isinstance(ctx.get("execution_authority", {}), dict)
        else {}
    )
    if not execution_authority:
        execution_authority = _derive_contextual_execution_authority(ctx)
    task_contract = (
        dict(ctx.get("task_contract", {}) or {})
        if isinstance(ctx.get("task_contract", {}), dict)
        else build_task_contract(
            goal_contract=ctx.get("goal_contract", {}),
            task_graph=ctx.get("task_graph", {}),
            task_node=ctx.get("task_node", {}),
            completion_gate=ctx.get("completion_gate", {}),
            authority_snapshot=ctx.get("authority_snapshot", {}),
            run_id=str(ctx.get("run_id", "") or ""),
            episode=int(ctx.get("episode", 0) or 0),
            tick=int(ctx.get("tick", 0) or 0),
        ).to_dict()
    )
    task_contract_verification_pre = _dict_or_empty(
        _dict_or_empty(task_contract.get("verification_requirement", {})).get("verifier_authority", {})
    )
    task_contract_completion_pre = _dict_or_empty(
        _dict_or_empty(_dict_or_empty(task_contract.get("completion", {})).get("completion_gate", {})).get(
            "verifier_authority", {}
        )
    )
    execution_authority_pre = _dict_or_empty(execution_authority.get("verifier_authority", {}))
    task_contract_freshness = _evaluate_task_contract_freshness(task_contract=task_contract, context=ctx)
    if task_contract_freshness:
        task_contract = dict(task_contract)
        task_contract["freshness_binding"] = task_contract_freshness
    task_governance_memory = _dict_or_empty(task_contract.get("governance_memory", {}))
    if decision.approval_required or decision.approval_granted:
        approval_events = [
            dict(item)
            for item in list(task_governance_memory.get("approval_events", []) or [])
            if isinstance(item, dict)
        ]
        approval_events.append(
            {
                "event_type": "approval_ticket",
                "approved": bool(decision.approval_granted),
                "approval_grant_id": str(decision.approval_grant_id or ""),
                "approval_sources": [str(item) for item in list(decision.approval_sources or []) if str(item)],
                "contract_scope": str(decision.contract_scope or ""),
            }
        )
        task_governance_memory["approval_events"] = approval_events[-6:]
        task_governance_memory["approval_pending"] = bool(decision.approval_required and not decision.approval_granted)
        task_governance_memory["last_approval_required"] = bool(decision.approval_required)
        task_governance_memory["last_approval_grant_id"] = str(decision.approval_grant_id or "")
    if task_governance_memory:
        task_contract = dict(task_contract)
        task_contract["governance_memory"] = task_governance_memory
    verification_requirement = (
        dict(task_contract.get("verification_requirement", {}) or {})
        if isinstance(task_contract.get("verification_requirement", {}), dict)
        else {}
    )
    verifier_authority = _resolve_verifier_authority_snapshot(
        context=ctx,
        task_contract=task_contract,
        execution_authority=execution_authority,
    )
    task_contract = _canonicalize_task_contract_verifier_authority(task_contract, verifier_authority)
    execution_authority = _canonicalize_execution_authority_verifier_authority(execution_authority, verifier_authority)
    if not verification_requirement:
        verification_requirement = (
            dict(
                _dict_or_empty(_dict_or_empty(execution_authority.get("active_task", {})).get("verification_gate", {}))
            )
            or dict(_dict_or_empty(ctx.get("goal_contract", {})).get("verification", {}) or {})
        )
    if verifier_authority:
        verification_requirement = dict(verification_requirement)
        verification_requirement["verifier_authority"] = verifier_authority
    verifier_authority_source = _ticket_verifier_authority_source(
        context=ctx,
        task_contract_supplied=task_contract_supplied,
        task_contract_verification_pre=task_contract_verification_pre,
        task_contract_completion_pre=task_contract_completion_pre,
        execution_authority_pre=execution_authority_pre,
    )
    verifier_authority_views = _ticket_verifier_authority_views(
        source=verifier_authority_source,
        verifier_authority=verifier_authority,
        task_contract_verification_pre=task_contract_verification_pre,
        task_contract_completion_pre=task_contract_completion_pre,
        execution_authority_pre=execution_authority_pre,
    )
    tool_id = str(decision.matched_tool_name or decision.function_name or "").strip()
    audit_event_id = f"audit-{uuid.uuid4().hex[:12]}"
    execution_scope = {
        "goal_ref": str(ctx.get("goal_ref", "") or ""),
        "task_ref": str(ctx.get("task_ref", "") or ""),
        "graph_ref": str(ctx.get("graph_ref", "") or ""),
        "run_id": str(ctx.get("run_id", "") or ""),
        "episode": int(ctx.get("episode", 0) or 0),
        "tick": int(ctx.get("tick", 0) or 0),
        "contract_scope": str(decision.contract_scope or ""),
        "policy_name": str(decision.policy_name or ""),
    }
    sandbox_audit = _build_sandbox_best_effort_audit(
        action=action,
        decision=decision,
        context=ctx,
    )
    execution_scope["sandbox_boundary"] = {
        "label": "best_effort",
        "audit_version": SANDBOX_BEST_EFFORT_AUDIT_VERSION,
        "not_os_security_sandbox": True,
    }
    return ExecutionTicket(
        ticket_id=f"exec-{uuid.uuid4().hex[:12]}",
        issued_at=float(time.time()),
        policy_name=decision.policy_name,
        tool_id=tool_id,
        function_name=decision.function_name,
        capability_class=decision.capability_class,
        side_effect_class=decision.side_effect_class,
        approval_required=decision.approval_required,
        approved=decision.allowed,
        decision_reason=decision.reason,
        goal_ref=str(ctx.get("goal_ref", "") or ""),
        task_ref=str(ctx.get("task_ref", "") or ""),
        graph_ref=str(ctx.get("graph_ref", "") or ""),
        run_id=str(ctx.get("run_id", "") or ""),
        episode=int(ctx.get("episode", 0) or 0),
        tick=int(ctx.get("tick", 0) or 0),
        approval_result=decision.to_dict(),
        execution_scope=execution_scope,
        verification_requirement=verification_requirement,
        secret_lease_requirement={
            "required": bool(decision.required_secret_lease_ids),
            "required_secret_lease_ids": list(decision.required_secret_lease_ids or []),
        },
        secret_lease_result={
            "granted_secret_lease_ids": list(decision.granted_secret_lease_ids or []),
            "missing_secret_lease_ids": list(decision.missing_secret_lease_ids or []),
            "scope_snapshots": [
                dict(item)
                for item in list(decision.secret_lease_scope_snapshots or [])
                if isinstance(item, dict)
            ],
        },
        audit_event_id=audit_event_id,
        task_contract=task_contract,
        goal_contract=dict(ctx.get("goal_contract", {}) or {}) if isinstance(ctx.get("goal_contract", {}), dict) else {},
        task_graph=dict(ctx.get("task_graph", {}) or {}) if isinstance(ctx.get("task_graph", {}), dict) else {},
        task_node=dict(ctx.get("task_node", {}) or {}) if isinstance(ctx.get("task_node", {}), dict) else {},
        metadata={
            "tool_id": tool_id,
            "matched_tool_name": decision.matched_tool_name,
            "approval_granted": bool(decision.approval_granted),
            "risk_level": decision.risk_level,
            "approval_sources": [str(item) for item in list(decision.approval_sources or []) if str(item)],
            "blocked_by_contract": bool(decision.blocked_by_contract),
            "contract_scope": str(decision.contract_scope or ""),
            "approval_grant_id": str(decision.approval_grant_id or ""),
            "approval_scope_snapshot": dict(decision.approval_scope_snapshot or {}),
            "required_secret_lease_ids": list(decision.required_secret_lease_ids or []),
            "granted_secret_lease_ids": list(decision.granted_secret_lease_ids or []),
            "missing_secret_lease_ids": list(decision.missing_secret_lease_ids or []),
            "secret_lease_scope_snapshots": [
                dict(item)
                for item in list(decision.secret_lease_scope_snapshots or [])
                if isinstance(item, dict)
            ],
            "sandbox_audit": sandbox_audit,
            "sandbox_boundary": {
                "label": "best_effort",
                "security_boundary": "best_effort_policy_ticket_audit",
                "not_os_security_sandbox": True,
                "audit_version": SANDBOX_BEST_EFFORT_AUDIT_VERSION,
            },
            "execution_authority": execution_authority,
            "verifier_authority_ref": "verification_requirement.verifier_authority",
            "verifier_authority_source": verifier_authority_source,
            "verifier_authority_views": verifier_authority_views,
            "scope_binding": {
                "goal_ref": str(ctx.get("goal_ref", "") or ""),
                "task_ref": str(ctx.get("task_ref", "") or ""),
                "graph_ref": str(ctx.get("graph_ref", "") or ""),
                "run_id": str(ctx.get("run_id", "") or ""),
                "episode": int(ctx.get("episode", 0) or 0),
                "tick": int(ctx.get("tick", 0) or 0),
            },
            "task_governance_memory": dict(task_contract.get("governance_memory", {}) or {}),
            "task_contract_freshness": dict(task_contract_freshness or {}),
            "task_contract": task_contract,
        },
    )


def resolve_execution_ticket_verifier_authority(ticket: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = _dict_or_empty(ticket)
    verification_requirement = _dict_or_empty(payload.get("verification_requirement", {}))
    authority = _dict_or_empty(verification_requirement.get("verifier_authority", {}))
    if authority or verification_requirement:
        return authority
    task_contract = _dict_or_empty(payload.get("task_contract", {}))
    contract_authority = _verifier_authority_from_task_contract(task_contract)
    if contract_authority or task_contract:
        return contract_authority
    metadata = _dict_or_empty(payload.get("metadata", {}))
    return _dict_or_empty(metadata.get("verifier_authority", {}))


def build_policy_block_result(*, ticket: ExecutionTicket, decision: ApprovalDecision) -> Dict[str, Any]:
    audit_event = build_audit_event(
        "execution_blocked",
        goal_ref=ticket.goal_ref,
        task_ref=ticket.task_ref,
        graph_ref=ticket.graph_ref,
        tool_id=ticket.tool_id,
        execution_ticket_id=ticket.ticket_id,
        source_stage="execution_policy",
        payload={
            "reason": decision.reason,
            "function_name": ticket.function_name,
            "approval_required": bool(decision.approval_required),
            "missing_secret_lease_ids": list(decision.missing_secret_lease_ids or []),
            "sandbox_audit": dict(_dict_or_empty(ticket.metadata).get("sandbox_audit", {})),
        },
        metadata={
            "policy_name": ticket.policy_name,
            "approval_sources": [str(item) for item in list(decision.approval_sources or []) if str(item)],
            "required_secret_lease_ids": list(decision.required_secret_lease_ids or []),
            "granted_secret_lease_ids": list(decision.granted_secret_lease_ids or []),
            "sandbox_boundary": dict(_dict_or_empty(ticket.metadata).get("sandbox_boundary", {})),
        },
        audit_event_id=ticket.audit_event_id,
    )
    return {
        "success": False,
        "ok": False,
        "blocked_by_policy": True,
        "state": "POLICY_BLOCKED",
        "failure_reason": decision.reason,
        "reward": 0.0,
        "terminal": False,
        "done": False,
        "execution_ticket": ticket.to_dict(),
        "policy_decision": decision.to_dict(),
        "audit_event": audit_event.to_dict(),
        "events": [
            {
                "type": "policy_blocked",
                "function_name": ticket.function_name,
                "reason": decision.reason,
                "ticket_id": ticket.ticket_id,
                "audit_event_id": ticket.audit_event_id,
            }
        ],
    }


def attach_execution_ticket(action: Any, ticket: ExecutionTicket, decision: ApprovalDecision) -> Any:
    if not isinstance(action, dict):
        return action
    action["_execution_ticket"] = ticket.to_dict()
    meta = action.setdefault("_candidate_meta", {})
    if not isinstance(meta, dict):
        meta = {}
        action["_candidate_meta"] = meta
    meta["execution_ticket_id"] = ticket.ticket_id
    meta["execution_policy"] = {
        "approved": bool(decision.allowed),
        "reason": decision.reason,
        "policy_name": decision.policy_name,
        "approval_required": bool(decision.approval_required),
        "tool_id": ticket.tool_id,
        "goal_ref": ticket.goal_ref,
        "task_ref": ticket.task_ref,
        "approval_sources": [str(item) for item in list(decision.approval_sources or []) if str(item)],
        "blocked_by_contract": bool(decision.blocked_by_contract),
        "contract_scope": str(decision.contract_scope or ""),
        "approval_grant_id": str(decision.approval_grant_id or ""),
        "required_secret_lease_ids": list(decision.required_secret_lease_ids or []),
        "granted_secret_lease_ids": list(decision.granted_secret_lease_ids or []),
        "missing_secret_lease_ids": list(decision.missing_secret_lease_ids or []),
        "audit_event_id": ticket.audit_event_id,
        "sandbox_boundary": dict(_dict_or_empty(ticket.metadata).get("sandbox_boundary", {})),
        "sandbox_audit": dict(_dict_or_empty(ticket.metadata).get("sandbox_audit", {})),
    }
    return action


def consume_approval_grant(
    *,
    approval_context: Any,
    decision: ApprovalDecision,
) -> bool:
    if not isinstance(approval_context, dict):
        return False
    consumed = False
    grant_id = str(decision.approval_grant_id or "").strip()
    if grant_id:
        scopes = approval_context.get("approved_ticket_scopes", [])
        if isinstance(scopes, list):
            for scope in scopes:
                if not isinstance(scope, dict):
                    continue
                scope_id = str(scope.get("grant_id", "") or scope.get("ticket_scope_id", "") or "").strip()
                if scope_id != grant_id:
                    continue
                remaining_uses = scope.get("remaining_uses", scope.get("max_uses", 1))
                try:
                    next_uses = max(0, int(remaining_uses) - 1)
                except (TypeError, ValueError):
                    next_uses = 0
                scope["remaining_uses"] = next_uses
                if next_uses <= 0:
                    consumed_ids = approval_context.setdefault("consumed_ticket_scope_ids", [])
                    if isinstance(consumed_ids, list) and grant_id not in consumed_ids:
                        consumed_ids.append(grant_id)
                consumed = True
                break
    secret_scopes = approval_context.get("approved_secret_leases", [])
    if not isinstance(secret_scopes, list):
        return consumed
    for lease_id in _ordered_unique(list(decision.granted_secret_lease_ids or [])):
        if not lease_id:
            continue
        for scope in secret_scopes:
            if not isinstance(scope, dict):
                continue
            scope_id = _canonical_secret_lease_id(scope)
            if scope_id != lease_id:
                continue
            remaining_uses = scope.get("remaining_uses", scope.get("max_uses", 1))
            try:
                next_uses = max(0, int(remaining_uses) - 1)
            except (TypeError, ValueError):
                next_uses = 0
            scope["remaining_uses"] = next_uses
            if next_uses <= 0:
                consumed_ids = approval_context.setdefault("consumed_secret_lease_ids", [])
                if isinstance(consumed_ids, list) and scope_id not in consumed_ids:
                    consumed_ids.append(scope_id)
            consumed = True
            break
    return consumed


def _approval_granted(
    *,
    action: Optional[Dict[str, Any]],
    function_name: str,
    capability_class: str,
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ctx = dict(context or {})
    ticket_scopes = [
        _dict_or_empty(item)
        for item in list(ctx.get("approved_ticket_scopes", []) or [])
        if isinstance(item, dict)
    ]
    scope_result = _match_approval_scope_grant(
        ticket_scopes=ticket_scopes,
        function_name=function_name,
        capability_class=capability_class,
        context=ctx,
    )
    if bool(scope_result.get("granted", False)):
        return scope_result

    approved_functions = {
        str(item).strip()
        for item in list(ctx.get("approved_functions", []) or [])
        if str(item).strip()
    }
    approved_capabilities = {
        str(item).strip()
        for item in list(ctx.get("approved_capabilities", []) or [])
        if str(item).strip()
    }
    if function_name in approved_functions:
        return {
            "granted": True,
            "source": "legacy_approved_functions",
            "grant_id": "",
            "scope_snapshot": {},
        }
    if capability_class and capability_class in approved_capabilities:
        return {
            "granted": True,
            "source": "legacy_approved_capabilities",
            "grant_id": "",
            "scope_snapshot": {},
        }
    return {
        "granted": False,
        "source": "",
        "grant_id": "",
        "scope_snapshot": {},
    }


def _secret_lease_granted(
    *,
    required_secret_leases: List[Dict[str, Any]],
    function_name: str,
    capability_class: str,
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    required = _normalized_secret_lease_records(required_secret_leases)
    if not required:
        return {
            "granted_secret_lease_ids": [],
            "missing_secret_lease_ids": [],
            "scope_snapshots": [],
            "source": "",
        }
    ctx = dict(context or {})
    grants = _normalized_secret_lease_records(ctx.get("approved_secret_leases", []))
    granted_ids: List[str] = []
    missing_ids: List[str] = []
    scope_snapshots: List[Dict[str, Any]] = []
    for requirement in required:
        match = _match_secret_lease_grant(
            requirement=requirement,
            grants=grants,
            function_name=function_name,
            capability_class=capability_class,
            context=ctx,
        )
        requirement_id = _canonical_secret_lease_id(requirement) or str(requirement.get("secret_id", "") or "").strip()
        if match:
            granted_ids.append(_canonical_secret_lease_id(match) or requirement_id)
            scope_snapshots.append(dict(match))
            continue
        if bool(requirement.get("required", True)):
            missing_ids.append(requirement_id)
    return {
        "granted_secret_lease_ids": _ordered_unique(granted_ids),
        "missing_secret_lease_ids": _ordered_unique(missing_ids),
        "scope_snapshots": scope_snapshots,
        "source": "approved_secret_lease" if granted_ids else "",
    }


def _scope_field_matches(expected: Any, actual: Any) -> bool:
    if expected in (None, "", [], {}):
        return True
    if isinstance(expected, int):
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return False
    return str(actual or "").strip() == str(expected or "").strip()


def _secret_lease_requirement_applies(
    *,
    requirement: Mapping[str, Any],
    function_name: str,
    capability_class: str,
    context: Mapping[str, Any],
) -> bool:
    requirement_function = str(requirement.get("function_name", "") or "").strip().casefold()
    requirement_capability = str(requirement.get("capability_class", "") or "").strip()
    fn_key = str(function_name or "").strip().casefold()
    if requirement_function and requirement_function != fn_key:
        return False
    if requirement_capability and capability_class and requirement_capability != capability_class:
        return False
    if requirement_capability and not capability_class:
        return False
    for field_name in ("goal_ref", "task_ref", "graph_ref", "run_id", "episode", "tick"):
        if not _scope_field_matches(requirement.get(field_name), context.get(field_name)):
            return False
    return True


def _match_secret_lease_grant(
    *,
    requirement: Mapping[str, Any],
    grants: List[Dict[str, Any]],
    function_name: str,
    capability_class: str,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    requirement_id = _canonical_secret_lease_id(requirement)
    requirement_secret_id = str(requirement.get("secret_id", "") or "").strip()
    for grant in grants:
        if not grant:
            continue
        remaining_uses = grant.get("remaining_uses", grant.get("max_uses", 1))
        try:
            if int(remaining_uses) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        grant_id = _canonical_secret_lease_id(grant)
        grant_secret_id = str(grant.get("secret_id", "") or "").strip()
        if requirement_id and grant_id != requirement_id:
            if not (
                requirement_secret_id
                and grant_secret_id
                and requirement_secret_id == grant_secret_id
            ):
                continue
        if requirement_secret_id and grant_secret_id and requirement_secret_id != grant_secret_id:
            continue
        if not _secret_lease_requirement_applies(
            requirement=grant,
            function_name=function_name,
            capability_class=capability_class,
            context=context,
        ):
            continue
        if not _secret_lease_requirement_applies(
            requirement=requirement,
            function_name=function_name,
            capability_class=capability_class,
            context=context,
        ):
            continue
        return dict(grant)
    return {}


def _match_approval_scope_grant(
    *,
    ticket_scopes: List[Dict[str, Any]],
    function_name: str,
    capability_class: str,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    function_key = str(function_name or "").strip().casefold()
    capability_key = str(capability_class or "").strip()
    for index, grant in enumerate(ticket_scopes):
        if not grant:
            continue
        remaining_uses = grant.get("remaining_uses", grant.get("max_uses", 1))
        try:
            if int(remaining_uses) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        grant_function = str(grant.get("function_name", "") or "").strip().casefold()
        grant_capability = str(grant.get("capability_class", "") or "").strip()
        if grant_function and grant_function != function_key:
            continue
        if grant_capability and capability_key and grant_capability != capability_key:
            continue
        if not grant_function and not grant_capability:
            continue
        if not _scope_field_matches(grant.get("goal_ref"), context.get("goal_ref")):
            continue
        if not _scope_field_matches(grant.get("task_ref"), context.get("task_ref")):
            continue
        if not _scope_field_matches(grant.get("graph_ref"), context.get("graph_ref")):
            continue
        if not _scope_field_matches(grant.get("run_id"), context.get("run_id")):
            continue
        if not _scope_field_matches(grant.get("episode"), context.get("episode")):
            continue
        if not _scope_field_matches(grant.get("tick"), context.get("tick")):
            continue
        grant_id = str(grant.get("grant_id", "") or grant.get("ticket_scope_id", "") or f"approval-scope-{index}")
        return {
            "granted": True,
            "source": "approved_ticket_scope",
            "grant_id": grant_id,
            "scope_snapshot": dict(grant),
        }
    return {
        "granted": False,
        "source": "",
        "grant_id": "",
        "scope_snapshot": {},
    }


def _task_node_target_matches(
    *,
    task_node: Mapping[str, Any],
    function_name: str,
    capability_class: str,
) -> bool:
    fn_key = str(function_name or "").strip().casefold()
    if not fn_key:
        return False
    metadata = _dict_or_empty(task_node.get("metadata", {}))
    target_function = str(metadata.get("target_function", "") or "").strip().casefold()
    if target_function and target_function == fn_key:
        return True
    approval_requirement = _dict_or_empty(task_node.get("approval_requirement", {}))
    required_capability_class = str(approval_requirement.get("capability_class", "") or "").strip()
    if capability_class and required_capability_class and capability_class == required_capability_class:
        return True
    title_tokens = set(_function_name_tokens(str(task_node.get("title", "") or "")))
    fn_tokens = set(_function_name_tokens(function_name))
    if fn_tokens and title_tokens.intersection(fn_tokens):
        return True
    return False


def _resolve_verifier_authority_snapshot(
    *,
    context: Optional[Mapping[str, Any]],
    task_contract: Optional[Mapping[str, Any]],
    execution_authority: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    ctx = dict(context or {})
    runtime = build_verifier_runtime(
        task_contract=task_contract,
        completion_gate=_dict_or_empty(ctx.get("completion_gate", {})),
        execution_authority=execution_authority,
        context=ctx,
    )
    return dict(runtime.verifier_authority)


def _canonicalize_task_contract_verifier_authority(
    task_contract: Optional[Mapping[str, Any]],
    verifier_authority: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return runtime_canonicalize_task_contract_verifier_authority(task_contract, verifier_authority)


def _canonicalize_execution_authority_verifier_authority(
    execution_authority: Optional[Mapping[str, Any]],
    verifier_authority: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return runtime_canonicalize_execution_authority_verifier_authority(execution_authority, verifier_authority)


def _ticket_verifier_authority_source(
    *,
    context: Optional[Mapping[str, Any]],
    task_contract_supplied: bool,
    task_contract_verification_pre: Optional[Mapping[str, Any]],
    task_contract_completion_pre: Optional[Mapping[str, Any]],
    execution_authority_pre: Optional[Mapping[str, Any]],
) -> str:
    ctx = dict(context or {})
    if task_contract_supplied and (
        _dict_or_empty(task_contract_verification_pre) or _dict_or_empty(task_contract_completion_pre)
    ):
        return "task_contract"
    if _dict_or_empty(execution_authority_pre):
        return "execution_authority"
    if _dict_or_empty(_dict_or_empty(ctx.get("completion_gate", {})).get("verifier_authority", {})):
        return "completion_gate_mirror"
    if _dict_or_empty(ctx.get("verifier_authority", {})):
        return "context_mirror"
    if _dict_or_empty(task_contract_verification_pre) or _dict_or_empty(task_contract_completion_pre):
        return "task_contract"
    return "unresolved"


def _ticket_verifier_authority_views(
    *,
    source: str,
    verifier_authority: Optional[Mapping[str, Any]],
    task_contract_verification_pre: Optional[Mapping[str, Any]],
    task_contract_completion_pre: Optional[Mapping[str, Any]],
    execution_authority_pre: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    canonical_authority = _dict_or_empty(verifier_authority)
    canonical_signature = _authority_signature(canonical_authority)

    def _mirror_view(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        row = _dict_or_empty(payload)
        pre_aligned = bool(row) and _authority_signature(row) == canonical_signature
        return {
            "role": "mirror",
            "source": "ticket.verification_requirement.verifier_authority",
            "aligned": True,
            "warnings": [] if not row or pre_aligned else ["verifier_authority_mirror_normalized"],
        }

    return {
        "ticket.verification_requirement.verifier_authority": {
            "role": "canonical",
            "source": str(source or "unresolved"),
            "aligned": True,
            "warnings": [],
        },
        "ticket.task_contract.verification_requirement.verifier_authority": _mirror_view(task_contract_verification_pre),
        "ticket.task_contract.completion.completion_gate.verifier_authority": _mirror_view(task_contract_completion_pre),
        "ticket.metadata.execution_authority.verifier_authority": _mirror_view(execution_authority_pre),
    }


def _approval_contract_signals(
    *,
    context: Optional[Mapping[str, Any]],
    function_name: str,
    capability_class: str,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    goal_contract = _dict_or_empty(ctx.get("goal_contract", {}))
    task_contract = _dict_or_empty(ctx.get("task_contract", {}))
    task_node = _dict_or_empty(ctx.get("task_node", {}))
    execution_authority = _dict_or_empty(ctx.get("execution_authority", {}))
    if not execution_authority:
        if task_contract:
            execution_authority = {
                "planning": _dict_or_empty(task_contract.get("planning", {})),
                "approval": _dict_or_empty(task_contract.get("approval", {})),
                "active_task": {
                    "title": str(task_contract.get("title", "") or ""),
                    "target_function": str(task_contract.get("target_function", "") or ""),
                    "approval_requirement": dict(
                        _dict_or_empty(_dict_or_empty(task_contract.get("approval", {})).get("task_requirement", {}))
                    ),
                    "governance_memory": _dict_or_empty(task_contract.get("governance_memory", {})),
                },
                "verifier_authority": _dict_or_empty(
                    _dict_or_empty(task_contract.get("verification_requirement", {})).get("verifier_authority", {})
                ),
            }
        else:
            execution_authority = _derive_contextual_execution_authority(ctx)
    authority_drift_sources = _collect_task_contract_authority_drift(
        context=ctx,
        task_contract=task_contract,
        execution_authority=execution_authority,
    )
    verifier_authority = _resolve_verifier_authority_context(
        context=ctx,
        execution_authority=execution_authority,
    )
    task_contract_freshness = _evaluate_task_contract_freshness(
        task_contract=task_contract,
        context=ctx,
    )
    planning = _dict_or_empty(execution_authority.get("planning", {})) or _dict_or_empty(task_contract.get("planning", {}))
    goal_approval = _dict_or_empty(execution_authority.get("approval", {})) or _dict_or_empty(task_contract.get("approval", {}))
    active_task = _dict_or_empty(execution_authority.get("active_task", {}))
    contract_task_requirement = _dict_or_empty(_dict_or_empty(task_contract.get("approval", {})).get("task_requirement", {}))
    pseudo_task_node = task_node or {
        "title": str(active_task.get("title", "") or task_contract.get("title", "") or ""),
        "metadata": {
            "target_function": str(
                active_task.get("target_function", "") or task_contract.get("target_function", "") or ""
            ),
            "intent": str(active_task.get("intent", "") or task_contract.get("intent", "") or ""),
        },
        "approval_requirement": dict(
            _dict_or_empty(task_node.get("approval_requirement", {}))
            or contract_task_requirement
            or _dict_or_empty(active_task.get("approval_requirement", {}))
        ),
        "governance_memory": _dict_or_empty(
            task_node.get("governance_memory", {})
            or task_contract.get("governance_memory", {})
            or active_task.get("governance_memory", {})
        ),
    }
    task_approval = dict(contract_task_requirement)
    if not task_approval:
        task_approval = resolve_effective_task_approval_requirement(
            goal_contract,
            pseudo_task_node or active_task,
            function_name=function_name,
            capability_class=capability_class,
        )
    explicit_task_requirement = _dict_or_empty(task_node.get("approval_requirement", {})) or dict(contract_task_requirement)
    if explicit_task_requirement:
        if "required" in explicit_task_requirement:
            task_approval["required"] = bool(explicit_task_requirement.get("required", False))
        if str(explicit_task_requirement.get("risk_level", "") or "").strip():
            task_approval["risk_level"] = str(explicit_task_requirement.get("risk_level", "") or "").strip()
        if str(explicit_task_requirement.get("capability_class", "") or "").strip():
            task_approval["capability_class"] = str(
                explicit_task_requirement.get("capability_class", "") or ""
            ).strip()
        if str(explicit_task_requirement.get("reason", "") or "").strip():
            task_approval["reason"] = str(explicit_task_requirement.get("reason", "") or "").strip()
        explicit_secret_leases = _normalized_secret_lease_records(
            explicit_task_requirement.get("required_secret_leases", [])
        )
        if explicit_secret_leases:
            task_approval["required_secret_leases"] = explicit_secret_leases

    blocked_functions = _normalized_name_set(
        list(planning.get("blocked_functions", []))
        + list(goal_approval.get("blocked_functions", []))
        + list(goal_contract.get("forbidden_actions", []))
    )
    required_functions = _normalized_name_set(goal_approval.get("required_functions", []))
    function_key = str(function_name or "").strip().casefold()

    approval_sources: List[str] = []
    blocked_reason = ""
    contract_scope = ""
    blocked_by_contract = False
    approval_required = False
    required_secret_leases: List[Dict[str, Any]] = []
    secret_lease_sources: List[str] = []

    if task_contract_freshness and not bool(task_contract_freshness.get("acceptable_for_execution", True)):
        blocked_by_contract = True
        blocked_reason = "stale_task_contract"
        stale_reasons = [
            str(item).strip()
            for item in list(task_contract_freshness.get("stale_reasons", []) or [])
            if str(item).strip()
        ]
        renewal_reasons = [
            str(item).strip()
            for item in list(task_contract_freshness.get("renewal_reasons", []) or [])
            if str(item).strip()
        ]
        if bool(task_contract_freshness.get("renewal_required", False)):
            blocked_reason = "stale_task_contract:renewal_required"
            if renewal_reasons:
                blocked_reason = f"stale_task_contract:renewal_required:{','.join(renewal_reasons)}"
        elif stale_reasons:
            blocked_reason = f"stale_task_contract:{','.join(stale_reasons)}"
        contract_scope = "task_contract.freshness"

    if not blocked_by_contract and authority_drift_sources:
        blocked_by_contract = True
        blocked_reason = f"task_contract_authority_drift:{','.join(authority_drift_sources)}"
        contract_scope = "task_contract.authority"

    if not blocked_by_contract and function_key and function_key in blocked_functions:
        blocked_by_contract = True
        blocked_reason = f"goal_contract_blocked_function:{function_name}"
        if function_key in _normalized_name_set(planning.get("blocked_functions", [])):
            contract_scope = "goal_contract.planning"
        else:
            contract_scope = "goal_contract.approval"

    active_task_matches = _task_node_target_matches(
        task_node=(
            pseudo_task_node
            or {
                "title": str(active_task.get("title", "") or ""),
                "metadata": {
                    "target_function": str(active_task.get("target_function", "") or ""),
                    "intent": str(active_task.get("intent", "") or ""),
                },
                "approval_requirement": dict(task_approval),
            }
        ),
        function_name=function_name,
        capability_class=capability_class,
    )
    if (
        not blocked_by_contract
        and active_task_matches
        and not bool(planning.get("active_task_allowed", True))
    ):
        blocked_by_contract = True
        blocked_reason = (
            f"goal_contract_disallowed_step_intent:{str(active_task.get('intent', '') or function_name)}"
        )
        contract_scope = "goal_contract.planning"

    verifier_decision = str(verifier_authority.get("decision", "") or "").strip().lower()
    if not blocked_by_contract and function_key and verifier_decision == "block_completion":
        if not _action_satisfies_verifier_authority(
            function_name=function_name,
            capability_class=capability_class,
            verifier_authority=verifier_authority,
        ):
            blocked_by_contract = True
            blocked_reason = f"verifier_block_completion:{function_name}"
            contract_scope = "goal_contract.verification"

    if function_key and function_key in required_functions:
        approval_required = True
        approval_sources.append("goal_contract.approval.required_functions")
        if not contract_scope:
            contract_scope = "goal_contract.approval"

    task_required = bool(task_approval.get("required", False))
    if task_required and active_task_matches:
        approval_required = True
        if bool(explicit_task_requirement.get("required", False)):
            approval_sources.append("task_node.approval_requirement")
        elif function_key and function_key in required_functions:
            approval_sources.append("goal_contract.approval.required_functions")
        else:
            approval_sources.append("goal_contract.approval.high_risk")
        if not contract_scope:
            if bool(explicit_task_requirement.get("required", False)):
                contract_scope = "task_node.approval_requirement"
            else:
                contract_scope = "goal_contract.approval"

    goal_secret_leases = _normalized_secret_lease_records(goal_approval.get("required_secret_leases", []))
    for requirement in goal_secret_leases:
        if not _secret_lease_requirement_applies(
            requirement=requirement,
            function_name=function_name,
            capability_class=capability_class,
            context=ctx,
        ):
            continue
        required_secret_leases.append(dict(requirement))
    if goal_secret_leases and required_secret_leases:
        secret_lease_sources.append("goal_contract.approval.required_secret_leases")
        if not contract_scope:
            contract_scope = "goal_contract.approval"

    task_secret_leases = _normalized_secret_lease_records(task_approval.get("required_secret_leases", []))
    if active_task_matches and task_secret_leases:
        required_secret_leases.extend(dict(item) for item in task_secret_leases)
        explicit_task_requirement = _dict_or_empty(task_node.get("approval_requirement", {}))
        if _normalized_secret_lease_records(explicit_task_requirement.get("required_secret_leases", [])):
            secret_lease_sources.append("task_node.approval_requirement.required_secret_leases")
            if not contract_scope:
                contract_scope = "task_node.approval_requirement"
        else:
            secret_lease_sources.append("goal_contract.approval.required_secret_leases")
            if not contract_scope:
                contract_scope = "goal_contract.approval"

    return {
        "approval_required": approval_required,
        "approval_sources": _ordered_unique(approval_sources),
        "blocked_by_contract": blocked_by_contract,
        "blocked_reason": blocked_reason,
        "contract_scope": contract_scope,
        "required_secret_leases": required_secret_leases,
        "secret_lease_sources": _ordered_unique(secret_lease_sources),
    }


def _function_name_tokens(function_name: str) -> List[str]:
    text = str(function_name or "").strip().lower()
    if not text:
        return []
    return [token for token in re.split(r"[^a-z0-9]+", text.replace("::", "_")) if token]


def _classify_observed_tool_name(function_name: str) -> Dict[str, Any]:
    normalized_name = str(function_name or "").strip().lower()
    tokens = _function_name_tokens(function_name)
    token_set = set(tokens)
    if not normalized_name:
        return {
            "approval_required": False,
            "capability_class": "",
            "side_effect_class": "",
            "risk_level": "low",
            "metadata": {"inferred_from_observation": True},
        }
    if normalized_name in _OBSERVED_TOOL_READ_ONLY_TOKENS or any(
        normalized_name.startswith(f"{token}_") for token in _OBSERVED_TOOL_READ_ONLY_TOKENS
    ):
        return {
            "approval_required": False,
            "capability_class": "inspection",
            "side_effect_class": "read_only",
            "risk_level": "low",
            "metadata": {
                "inferred_from_observation": True,
                "classification_source": "heuristic_read_only",
            },
        }
    exact_high_risk_match = any(
        normalized_name in exact_names
        for _side_effect_class, _capability_class, _risk_level, exact_names in _OBSERVED_TOOL_APPROVAL_EXACT_NAMES
    )
    if token_set.intersection(_NETWORK_FUNCTION_TOKENS) and not exact_high_risk_match:
        return {
            "approval_required": True,
            "capability_class": "network_mutation",
            "side_effect_class": "network_mutation",
            "risk_level": "high",
            "metadata": {
                "inferred_from_observation": True,
                "classification_source": "heuristic_network_effect",
                "classification_name": normalized_name,
            },
        }
    if token_set.intersection(_WRITE_FUNCTION_TOKENS) and not exact_high_risk_match:
        return {
            "approval_required": True,
            "capability_class": "filesystem_mutation",
            "side_effect_class": "filesystem_write",
            "risk_level": "high",
            "metadata": {
                "inferred_from_observation": True,
                "classification_source": "heuristic_write_effect",
                "classification_name": normalized_name,
            },
        }
    for side_effect_class, capability_class, risk_level, exact_names in _OBSERVED_TOOL_APPROVAL_EXACT_NAMES:
        if normalized_name in exact_names:
            return {
                "approval_required": True,
                "capability_class": capability_class,
                "side_effect_class": side_effect_class,
                "risk_level": risk_level,
                "metadata": {
                    "inferred_from_observation": True,
                    "classification_source": "heuristic_high_risk",
                    "classification_name": normalized_name,
                },
            }
    return {
        "approval_required": False,
        "capability_class": "",
        "side_effect_class": "",
        "risk_level": "low",
        "metadata": {
            "inferred_from_observation": True,
            "classification_source": "unclassified_surface",
        },
    }


def infer_available_tools_from_observation(observation: Any) -> List[Dict[str, Any]]:
    obs = dict(observation) if isinstance(observation, dict) else {}
    inferred: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _remember(tool_spec: Dict[str, Any]) -> None:
        name = str(tool_spec.get("name", "") or "").strip()
        if not name:
            return
        key = name.casefold()
        if key in seen:
            return
        seen.add(key)
        inferred.append(tool_spec)

    raw_tools = obs.get("available_tools", [])
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            _remember(serialize_tool_spec(tool))

    novel_api = obs.get("novel_api", {}) if isinstance(obs.get("novel_api", {}), dict) else {}
    backend_functions = obs.get("backend_functions", {}) if isinstance(obs.get("backend_functions", {}), dict) else {}
    function_signatures = obs.get("function_signatures", {}) if isinstance(obs.get("function_signatures", {}), dict) else {}

    candidate_names: List[str] = []
    for key in ("available_functions", "available_action_names", "available_actions"):
        values = obs.get(key, [])
        if isinstance(values, list):
            candidate_names.extend(str(item).strip() for item in values if str(item).strip())
    for key in ("visible_functions", "discovered_functions"):
        values = novel_api.get(key, [])
        if isinstance(values, list):
            candidate_names.extend(str(item).strip() for item in values if str(item).strip())
    candidate_names.extend(str(name).strip() for name in backend_functions.keys() if str(name).strip())
    candidate_names.extend(str(name).strip() for name in function_signatures.keys() if str(name).strip())

    for name in candidate_names:
        signature = function_signatures.get(name, {})
        input_schema = dict(signature) if isinstance(signature, dict) else {}
        inferred_classification = _classify_observed_tool_name(name)
        _remember(
            {
                "name": name,
                "description": "",
                "input_schema": input_schema,
                "side_effects": [],
                "risk_notes": [],
                "capability_class": str(inferred_classification.get("capability_class", "") or ""),
                "side_effect_class": str(inferred_classification.get("side_effect_class", "") or ""),
                "approval_required": bool(inferred_classification.get("approval_required", False)),
                "risk_level": str(inferred_classification.get("risk_level", "low") or "low"),
                "source": "observed_surface",
                "metadata": dict(inferred_classification.get("metadata", {}) or {}),
            }
        )
    return inferred

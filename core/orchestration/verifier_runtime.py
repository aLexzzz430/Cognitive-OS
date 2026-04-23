from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from core.orchestration.goal_task_control import (
    derive_verifier_authority_snapshot,
    resolve_goal_contract_authority,
)


VERIFIER_RUNTIME_VERSION = "conos.verifier_runtime/v1"


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def verifier_authority_from_task_contract(task_contract: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    contract_payload = _dict_or_empty(task_contract)
    if not contract_payload:
        return {}
    verification_authority = dict(
        _dict_or_empty(
            _dict_or_empty(contract_payload.get("verification_requirement", {})).get("verifier_authority", {})
        )
    )
    if verification_authority:
        return verification_authority
    return dict(
        _dict_or_empty(
            _dict_or_empty(_dict_or_empty(contract_payload.get("completion", {})).get("completion_gate", {})).get(
                "verifier_authority", {}
            )
        )
    )


def derive_contextual_execution_authority(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    ctx = dict(context or {})
    goal_contract = _dict_or_empty(ctx.get("goal_contract", {}))
    task_graph = ctx.get("task_graph", {})
    task_node = _dict_or_empty(ctx.get("task_node", {}))
    completion_gate = ctx.get("completion_gate", {})
    if not goal_contract and not task_node and not _dict_or_empty(task_graph):
        return {}
    return resolve_goal_contract_authority(
        goal_contract=goal_contract,
        task_graph=task_graph,
        active_task=task_node,
        completion_gate=completion_gate,
    )


def _resolve_authority_with_source(
    *,
    task_contract: Optional[Mapping[str, Any]] = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    execution_authority: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    goal_contract: Any = None,
    task_graph: Any = None,
    active_task: Any = None,
    verifier_authority_override: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Any], str]:
    override = _dict_or_empty(verifier_authority_override)
    if override:
        return override, "override"

    contract_payload = _dict_or_empty(task_contract)
    contract_authority = verifier_authority_from_task_contract(contract_payload)
    if contract_authority or contract_payload:
        return contract_authority, "task_contract"

    execution_payload = _dict_or_empty(execution_authority)
    if not execution_payload:
        execution_payload = derive_contextual_execution_authority(context)
    execution_verifier_authority = dict(_dict_or_empty(execution_payload.get("verifier_authority", {})))
    if execution_verifier_authority:
        return execution_verifier_authority, "execution_authority"

    ctx = dict(context or {})
    completion_authority = dict(
        _dict_or_empty(
            _dict_or_empty(completion_gate or ctx.get("completion_gate", {})).get("verifier_authority", {})
        )
    )
    if completion_authority:
        return completion_authority, "completion_gate"

    context_authority = dict(_dict_or_empty(ctx.get("verifier_authority", {})))
    if context_authority:
        return context_authority, "context"

    if goal_contract is not None or task_graph is not None or active_task is not None:
        return (
            derive_verifier_authority_snapshot(
                goal_contract=goal_contract,
                task_graph=task_graph,
                active_task=active_task,
                completion_gate=completion_gate,
            ),
            "derived_goal_task",
        )

    return {}, "none"


def resolve_effective_verifier_authority(
    *,
    task_contract: Optional[Mapping[str, Any]] = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    execution_authority: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    goal_contract: Any = None,
    task_graph: Any = None,
    active_task: Any = None,
    verifier_authority_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    authority, _source = _resolve_authority_with_source(
        task_contract=task_contract,
        completion_gate=completion_gate,
        execution_authority=execution_authority,
        context=context,
        goal_contract=goal_contract,
        task_graph=task_graph,
        active_task=active_task,
        verifier_authority_override=verifier_authority_override,
    )
    return dict(authority)


def canonicalize_task_contract_verifier_authority(
    task_contract: Optional[Mapping[str, Any]],
    verifier_authority: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = _dict_or_empty(task_contract)
    canonical_authority = _dict_or_empty(verifier_authority)
    if not payload or not canonical_authority:
        return payload
    verification_requirement = _dict_or_empty(payload.get("verification_requirement", {}))
    verification_requirement["verifier_authority"] = dict(canonical_authority)
    completion = _dict_or_empty(payload.get("completion", {}))
    completion_gate = _dict_or_empty(completion.get("completion_gate", {}))
    completion_gate["verifier_authority"] = dict(canonical_authority)
    completion["completion_gate"] = completion_gate
    payload["verification_requirement"] = verification_requirement
    payload["completion"] = completion
    return payload


def canonicalize_execution_authority_verifier_authority(
    execution_authority: Optional[Mapping[str, Any]],
    verifier_authority: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = _dict_or_empty(execution_authority)
    canonical_authority = _dict_or_empty(verifier_authority)
    if not payload or not canonical_authority:
        return payload
    payload["verifier_authority"] = dict(canonical_authority)
    completion = _dict_or_empty(payload.get("completion", {}))
    completion_gate = _dict_or_empty(completion.get("completion_gate", {}))
    completion_gate["verifier_authority"] = dict(canonical_authority)
    completion["completion_gate"] = completion_gate
    payload["completion"] = completion
    return payload


def _rollback_from_authority(authority: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _dict_or_empty(authority)
    target_node_id = str(payload.get("rollback_target_node_id", "") or "")
    target_step_id = str(payload.get("rollback_target_step_id", "") or "")
    eligible = bool(payload.get("rollback_eligible", False) and (target_node_id or target_step_id))
    return {
        "source": "verifier_authority",
        "eligible": eligible,
        "contradiction_detected": bool(payload.get("contradiction_detected", False)),
        "target_node_id": target_node_id,
        "target_step_id": target_step_id,
        "reason": str(payload.get("rollback_reason", "") or payload.get("blocked_reason", "") or ""),
        "decision": str(payload.get("decision", "") or ""),
    }


def _posterior_teaching_from_authority(authority: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _dict_or_empty(authority)
    signal = str(payload.get("teaching_signal", "") or "none").strip().lower()
    if signal not in {"positive", "negative", "none"}:
        signal = "none"
    score = _float_or_default(payload.get("teaching_signal_score", 0.0), 0.0)
    return {
        "source": "verifier_authority",
        "teaching_signal": signal,
        "teaching_signal_score": score,
        "verdict": str(payload.get("verdict", "") or ""),
        "decision": str(payload.get("decision", "") or ""),
        "contradiction_detected": bool(payload.get("contradiction_detected", False)),
        "evidence_present": bool(payload.get("evidence_present", False)),
        "verifier_function": str(payload.get("verifier_function", "") or ""),
    }


@dataclass(frozen=True)
class VerifierRuntimeSnapshot:
    verifier_authority: Dict[str, Any] = field(default_factory=dict)
    completion_gate: Dict[str, Any] = field(default_factory=dict)
    execution_authority: Dict[str, Any] = field(default_factory=dict)
    rollback: Dict[str, Any] = field(default_factory=dict)
    posterior_teaching: Dict[str, Any] = field(default_factory=dict)
    authority_source: str = "none"
    runtime_version: str = VERIFIER_RUNTIME_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_verifier_runtime(
    *,
    task_contract: Optional[Mapping[str, Any]] = None,
    completion_gate: Optional[Mapping[str, Any]] = None,
    execution_authority: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    goal_contract: Any = None,
    task_graph: Any = None,
    active_task: Any = None,
    verifier_authority_override: Optional[Mapping[str, Any]] = None,
) -> VerifierRuntimeSnapshot:
    authority, authority_source = _resolve_authority_with_source(
        task_contract=task_contract,
        completion_gate=completion_gate,
        execution_authority=execution_authority,
        context=context,
        goal_contract=goal_contract,
        task_graph=task_graph,
        active_task=active_task,
        verifier_authority_override=verifier_authority_override,
    )
    authority = dict(authority)
    gate = _dict_or_empty(completion_gate)
    ctx = dict(context or {})
    if not gate:
        gate = _dict_or_empty(ctx.get("completion_gate", {}))
    if not gate and task_contract:
        gate = _dict_or_empty(_dict_or_empty(_dict_or_empty(task_contract).get("completion", {})).get("completion_gate", {}))
    if authority:
        gate["verifier_authority"] = dict(authority)

    execution_payload = _dict_or_empty(execution_authority)
    if not execution_payload:
        execution_payload = _dict_or_empty(ctx.get("execution_authority", {}))
    if not execution_payload:
        execution_payload = derive_contextual_execution_authority(ctx)
    if not execution_payload and (goal_contract is not None or task_graph is not None or active_task is not None):
        execution_payload = resolve_goal_contract_authority(
            goal_contract=goal_contract,
            task_graph=task_graph,
            active_task=active_task,
            completion_gate=gate,
        )
    execution_payload = canonicalize_execution_authority_verifier_authority(execution_payload, authority)

    return VerifierRuntimeSnapshot(
        verifier_authority=authority,
        completion_gate=gate,
        execution_authority=execution_payload,
        rollback=_rollback_from_authority(authority),
        posterior_teaching=_posterior_teaching_from_authority(authority),
        authority_source=authority_source,
    )

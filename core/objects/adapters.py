from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from core.objects.lifecycle import (
    STATUS_QUALIFIED,
    bootstrap_lifecycle,
    normalize_lifecycle_status,
)
from core.objects.registry import resolve_object_class
from core.reasoning.hypothesis_schema import normalize_hypothesis_row
from core.objects.schema import (
    CognitiveObjectBase,
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_DISCRIMINATING_TEST,
    OBJECT_TYPE_HYPOTHESIS,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
)


_TYPE_ALIASES = {
    "representation": OBJECT_TYPE_REPRESENTATION,
    "generic_object": OBJECT_TYPE_REPRESENTATION,
    "evidence_object": OBJECT_TYPE_REPRESENTATION,
    "learning_update": OBJECT_TYPE_REPRESENTATION,
    "control_profile": OBJECT_TYPE_REPRESENTATION,
    "hypothesis": OBJECT_TYPE_HYPOTHESIS,
    "mechanism_hypothesis": OBJECT_TYPE_HYPOTHESIS,
    "world_model_hypothesis": OBJECT_TYPE_HYPOTHESIS,
    "discriminating_test": OBJECT_TYPE_DISCRIMINATING_TEST,
    "test": OBJECT_TYPE_DISCRIMINATING_TEST,
    "probe": OBJECT_TYPE_DISCRIMINATING_TEST,
    "skill": OBJECT_TYPE_SKILL,
    "skill_card": OBJECT_TYPE_SKILL,
    "procedure_chain": OBJECT_TYPE_SKILL,
    "transfer": OBJECT_TYPE_TRANSFER,
    "transfer_object": OBJECT_TYPE_TRANSFER,
    "identity": OBJECT_TYPE_IDENTITY,
    "identity_object": OBJECT_TYPE_IDENTITY,
    "autobiographical": OBJECT_TYPE_AUTOBIOGRAPHICAL,
    "autobiographical_object": OBJECT_TYPE_AUTOBIOGRAPHICAL,
}

_DEFAULT_SURFACE_PRIORITIES = {
    OBJECT_TYPE_REPRESENTATION: 0.65,
    OBJECT_TYPE_HYPOTHESIS: 0.55,
    OBJECT_TYPE_DISCRIMINATING_TEST: 0.7,
    OBJECT_TYPE_SKILL: 0.8,
    OBJECT_TYPE_TRANSFER: 0.75,
    OBJECT_TYPE_IDENTITY: 0.95,
    OBJECT_TYPE_AUTOBIOGRAPHICAL: 0.9,
}


def canonical_object_type(raw_type: Any) -> str:
    text = str(raw_type or "").strip().lower()
    if not text:
        return ""
    return _TYPE_ALIASES.get(text, text)


def infer_object_type(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return OBJECT_TYPE_REPRESENTATION

    for candidate in (
        payload.get("object_type"),
        payload.get("type"),
        payload.get("memory_type"),
    ):
        normalized = canonical_object_type(candidate)
        if normalized in _DEFAULT_SURFACE_PRIORITIES:
            return normalized

    content = payload.get("content", {})
    if isinstance(content, dict):
        for candidate in (
            content.get("object_type"),
            content.get("type"),
            content.get("memory_type"),
        ):
            normalized = canonical_object_type(candidate)
            if normalized in _DEFAULT_SURFACE_PRIORITIES:
                return normalized

    memory_layer = str(payload.get("memory_layer") or "").strip().lower()
    if memory_layer == "procedural":
        return OBJECT_TYPE_SKILL
    if memory_layer == "episodic":
        return OBJECT_TYPE_AUTOBIOGRAPHICAL
    return OBJECT_TYPE_REPRESENTATION


def _extract_structured_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    content = payload.get("content")
    if isinstance(content, dict):
        return copy.deepcopy(content)
    structured_payload = payload.get("structured_payload")
    if isinstance(structured_payload, dict):
        return copy.deepcopy(structured_payload)
    return copy.deepcopy(payload) if isinstance(payload, dict) else {}


def _resolve_function_name(structured_payload: Dict[str, Any]) -> str:
    return str(
        structured_payload.get("tool_args", {}).get("function_name")
        or structured_payload.get("function_name")
        or structured_payload.get("card_id")
        or structured_payload.get("task_signature")
        or ""
    ).strip()


def _resolve_summary(payload: Dict[str, Any], object_type: str, structured_payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("summary") or structured_payload.get("summary") or "").strip()
    if explicit:
        return explicit
    for key in ("assertion", "mechanism_summary", "goal", "task_signature", "claim"):
        text = str(structured_payload.get(key) or "").strip()
        if text:
            return text
    fn_name = _resolve_function_name(structured_payload)
    if fn_name:
        return f"{object_type}:{fn_name}"
    return object_type


def _resolve_family(payload: Dict[str, Any], object_type: str, structured_payload: Dict[str, Any]) -> str:
    for candidate in (
        payload.get("family"),
        structured_payload.get("family"),
        payload.get("memory_type"),
        structured_payload.get("memory_type"),
        structured_payload.get("task_signature"),
        _resolve_function_name(structured_payload),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return object_type


def _normalize_tags(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw_values = values
    elif isinstance(values, (tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    normalized: List[str] = []
    seen = set()
    for value in raw_values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_str_list(values: Any) -> List[str]:
    return _normalize_tags(values)


def _normalize_dict_list(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [copy.deepcopy(item) for item in values if isinstance(item, dict)]


def _evidence_ref_list(explicit: Any, rows: Any) -> List[str]:
    normalized = _normalize_str_list(explicit)
    if normalized:
        return normalized
    derived: List[str] = []
    for row in _normalize_dict_list(rows):
        for candidate in (
            row.get("id"),
            row.get("evidence_id"),
            row.get("observation_id"),
            row.get("reason"),
            row.get("summary"),
            row.get("event_type"),
            row.get("label"),
        ):
            text = str(candidate or "").strip()
            if not text:
                continue
            derived.append(text)
            break
    return _normalize_str_list(derived)


def _normalize_hypothesis_structured_payload(
    payload: Dict[str, Any],
    structured_payload: Dict[str, Any],
) -> Dict[str, Any]:
    raw = copy.deepcopy(structured_payload)
    for key, value in payload.items():
        if key == "content":
            continue
        if key in {"supporting_evidence", "contradicting_evidence"}:
            incoming_rows = _normalize_dict_list(value)
            if incoming_rows:
                raw[key] = copy.deepcopy(incoming_rows)
                continue
            if _normalize_dict_list(raw.get(key)):
                continue
        raw[key] = copy.deepcopy(value)
    normalized = normalize_hypothesis_row(raw, fallback_id_prefix="obj")
    merged = copy.deepcopy(structured_payload)
    for key in (
        "hypothesis_id",
        "hypothesis_type",
        "posterior",
        "support_count",
        "contradiction_count",
        "status",
        "scope",
        "source",
        "predictions",
        "falsifiers",
        "conflicts_with",
        "supporting_evidence",
        "contradicting_evidence",
        "tags",
        "metadata",
        "predicted_action_effects",
        "predicted_action_effects_by_signature",
        "predicted_observation_tokens",
        "predicted_phase_shift",
        "predicted_information_gain",
    ):
        if key in normalized:
            merged[key] = copy.deepcopy(normalized[key])
    if "type" not in merged:
        merged["type"] = str(payload.get("type") or structured_payload.get("type") or "hypothesis")
    return merged


def _resolve_provenance(payload: Dict[str, Any]) -> List[str]:
    provenance = payload.get("provenance", [])
    normalized = _normalize_str_list(provenance)
    if normalized:
        return normalized
    derived = []
    for candidate in (
        payload.get("trigger_source"),
        payload.get("source_stage"),
        payload.get("source_module"),
        payload.get("source"),
    ):
        text = str(candidate or "").strip()
        if text:
            derived.append(text)
    return _normalize_str_list(derived)


def _resolve_applicability(payload: Dict[str, Any], structured_payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("applicability", structured_payload.get("applicability", {}))
    return dict(raw) if isinstance(raw, dict) else {}


def _resolve_failure_conditions(payload: Dict[str, Any], structured_payload: Dict[str, Any]) -> List[str]:
    return _normalize_str_list(
        payload.get("failure_conditions", structured_payload.get("failure_conditions", []))
    )


def _resolve_surface_priority(
    payload: Dict[str, Any],
    object_type: str,
    structured_payload: Dict[str, Any],
) -> float:
    raw = payload.get("surface_priority", structured_payload.get("surface_priority"))
    if raw is None:
        return float(_DEFAULT_SURFACE_PRIORITIES.get(object_type, 0.5))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(_DEFAULT_SURFACE_PRIORITIES.get(object_type, 0.5))


def _resolve_extra_fields(
    payload: Dict[str, Any],
    object_type: str,
    structured_payload: Dict[str, Any],
) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    if object_type == OBJECT_TYPE_HYPOTHESIS:
        supporting_rows = _normalize_dict_list(
            payload.get("supporting_evidence_rows", structured_payload.get("supporting_evidence", []))
        )
        contradicting_rows = _normalize_dict_list(
            payload.get("contradicting_evidence_rows", structured_payload.get("contradicting_evidence", []))
        )
        extras["hypothesis_type"] = str(
            payload.get("hypothesis_type")
            or structured_payload.get("hypothesis_type")
            or structured_payload.get("family")
            or "generic"
        ).strip() or "generic"
        try:
            extras["posterior"] = float(
                payload.get("posterior", structured_payload.get("posterior", payload.get("confidence", 0.0)) or 0.0)
            )
        except (TypeError, ValueError):
            extras["posterior"] = float(payload.get("confidence", 0.0) or 0.0)
        try:
            extras["support_count"] = int(
                payload.get(
                    "support_count",
                    structured_payload.get("support_count", len(supporting_rows)) or len(supporting_rows),
                )
                or 0
            )
        except (TypeError, ValueError):
            extras["support_count"] = len(supporting_rows)
        try:
            extras["contradiction_count"] = int(
                payload.get(
                    "contradiction_count",
                    structured_payload.get("contradiction_count", len(contradicting_rows)) or len(contradicting_rows),
                )
                or 0
            )
        except (TypeError, ValueError):
            extras["contradiction_count"] = len(contradicting_rows)
        extras["scope"] = str(payload.get("scope") or structured_payload.get("scope") or "local").strip() or "local"
        extras["source"] = str(payload.get("source") or structured_payload.get("source") or "workspace").strip() or "workspace"
        extras["predictions"] = copy.deepcopy(
            payload.get("predictions", structured_payload.get("predictions", {}))
            if isinstance(payload.get("predictions", structured_payload.get("predictions", {})), dict)
            else {}
        )
        extras["falsifiers"] = _normalize_str_list(
            payload.get("falsifiers", structured_payload.get("falsifiers", []))
        )
        extras["conflicts_with"] = _normalize_str_list(
            payload.get("conflicts_with", structured_payload.get("conflicts_with", []))
        )
        extras["supporting_evidence"] = _evidence_ref_list(
            payload.get("supporting_evidence", []),
            supporting_rows,
        )
        extras["contradicting_evidence"] = _evidence_ref_list(
            payload.get("contradicting_evidence", []),
            contradicting_rows,
        )
        extras["supporting_evidence_rows"] = supporting_rows
        extras["contradicting_evidence_rows"] = contradicting_rows
        extras["tags"] = _normalize_str_list(
            payload.get("tags", structured_payload.get("tags", []))
        )
        extras["hypothesis_metadata"] = copy.deepcopy(
            payload.get("hypothesis_metadata", structured_payload.get("metadata", payload.get("metadata", {})))
            if isinstance(payload.get("hypothesis_metadata", structured_payload.get("metadata", payload.get("metadata", {}))), dict)
            else {}
        )
    elif object_type == OBJECT_TYPE_DISCRIMINATING_TEST:
        raw_test_spec = payload.get("test_spec", structured_payload.get("test_spec", structured_payload))
        extras["test_spec"] = dict(raw_test_spec) if isinstance(raw_test_spec, dict) else {}
    elif object_type == OBJECT_TYPE_SKILL:
        extras["skill_kind"] = str(
            payload.get("skill_kind")
            or structured_payload.get("skill_kind")
            or structured_payload.get("type")
            or structured_payload.get("memory_type")
            or ""
        ).strip()
    elif object_type == OBJECT_TYPE_TRANSFER:
        extras["source_family"] = str(
            payload.get("source_family") or structured_payload.get("source_family") or ""
        ).strip()
        extras["target_family"] = str(
            payload.get("target_family") or structured_payload.get("target_family") or ""
        ).strip()
        extras["reuse_evidence"] = _normalize_str_list(
            payload.get("reuse_evidence", structured_payload.get("reuse_evidence", []))
        )
    elif object_type == OBJECT_TYPE_IDENTITY:
        raw_profile = payload.get("identity_profile", structured_payload.get("identity_profile", structured_payload))
        extras["identity_profile"] = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    elif object_type == OBJECT_TYPE_AUTOBIOGRAPHICAL:
        extras["episode_refs"] = _normalize_str_list(
            payload.get("episode_refs", structured_payload.get("episode_refs", []))
        )
        raw_markers = payload.get(
            "continuity_markers",
            structured_payload.get("continuity_markers", {}),
        )
        extras["continuity_markers"] = dict(raw_markers) if isinstance(raw_markers, dict) else {}
    return extras


def proposal_to_cognitive_object(
    proposal: Dict[str, Any],
    *,
    object_id: str = "",
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> CognitiveObjectBase:
    payload = dict(proposal) if isinstance(proposal, dict) else {}
    structured_payload = _extract_structured_payload(payload)
    object_type = infer_object_type(payload)
    if object_type == OBJECT_TYPE_HYPOTHESIS:
        structured_payload = _normalize_hypothesis_structured_payload(payload, structured_payload)
    lifecycle = bootstrap_lifecycle(
        reason=str(payload.get("lifecycle_reason") or "validator_commit"),
        timestamp=created_at,
    )
    lifecycle_events = copy.deepcopy(payload.get("lifecycle_events", lifecycle["history"]))
    lifecycle_status = normalize_lifecycle_status(
        payload.get("status"),
        history=lifecycle_events,
        default=lifecycle["status"],
    )
    created = created_at or datetime.utcnow().isoformat()
    updated = updated_at or created
    object_class = resolve_object_class(object_type)
    base_kwargs = {
        "object_id": str(object_id or payload.get("object_id") or "").strip(),
        "object_type": object_type,
        "family": _resolve_family(payload, object_type, structured_payload),
        "summary": _resolve_summary(payload, object_type, structured_payload),
        "structured_payload": structured_payload,
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "evidence_ids": _normalize_str_list(payload.get("evidence_ids", [])),
        "provenance": _resolve_provenance(payload),
        "created_at": str(payload.get("created_at") or created),
        "updated_at": str(payload.get("updated_at") or updated),
        "status": lifecycle_status,
        "applicability": _resolve_applicability(payload, structured_payload),
        "failure_conditions": _resolve_failure_conditions(payload, structured_payload),
        "source_stage": str(payload.get("source_stage") or "").strip(),
        "commit_epoch": int(
            payload.get("commit_epoch", payload.get("episode", payload.get("trigger_episode", 0)) or 0)
        ),
        "version": int(payload.get("version", 1) or 1),
        "supersedes": _normalize_str_list(payload.get("supersedes", [])),
        "reopened_from": str(payload.get("reopened_from") or "").strip(),
        "lifecycle_events": lifecycle_events,
        "surface_priority": _resolve_surface_priority(payload, object_type, structured_payload),
        "asset_status": str(payload.get("asset_status") or "new_asset"),
        "memory_type": str(payload.get("memory_type") or "").strip(),
        "memory_layer": str(payload.get("memory_layer") or "").strip(),
        "retrieval_tags": _normalize_tags(payload.get("retrieval_tags", [])),
        "memory_metadata": dict(payload.get("memory_metadata", {}))
        if isinstance(payload.get("memory_metadata", {}), dict)
        else {},
        "content_hash": str(payload.get("content_hash") or "").strip(),
        "trigger_source": str(payload.get("trigger_source") or "unknown").strip(),
        "trigger_episode": int(payload.get("trigger_episode", 0) or 0),
        "consumption_count": int(payload.get("consumption_count", 0) or 0),
        "last_consumed_tick": payload.get("last_consumed_tick"),
        "reuse_history": list(payload.get("reuse_history", []))
        if isinstance(payload.get("reuse_history", []), list)
        else [],
    }
    base_kwargs.update(_resolve_extra_fields(payload, object_type, structured_payload))
    return object_class(**base_kwargs)


def proposal_to_object_record(
    proposal: Dict[str, Any],
    *,
    object_id: str = "",
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    return proposal_to_cognitive_object(
        proposal,
        object_id=object_id,
        created_at=created_at,
        updated_at=updated_at,
    ).to_record()


def record_to_cognitive_object(record: Dict[str, Any]) -> CognitiveObjectBase:
    payload = dict(record) if isinstance(record, dict) else {}
    object_type = infer_object_type(payload)
    if object_type == OBJECT_TYPE_HYPOTHESIS:
        return proposal_to_cognitive_object(payload)
    object_class = resolve_object_class(object_type)
    return object_class.from_record(payload)


def annotate_records_with_object_type(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for record in records:
        payload = dict(record) if isinstance(record, dict) else {}
        payload["object_type"] = infer_object_type(payload)
        annotated.append(payload)
    return annotated

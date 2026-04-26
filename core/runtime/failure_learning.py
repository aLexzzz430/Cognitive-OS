from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence


FAILURE_LEARNING_VERSION = "conos.failure_learning/v1"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload if payload is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _str_list(value: Any) -> list[str]:
    result: list[str] = []
    seen = set()
    for item in _as_list(value):
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _excerpt(value: Any, *, limit: int = 280) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


@dataclass(frozen=True)
class FailureLearningObject:
    schema_version: str = FAILURE_LEARNING_VERSION
    failure_mode: str = "unknown_failure"
    title: str = ""
    summary: str = ""
    violated_assumption: str = ""
    failed_action: Dict[str, Any] = field(default_factory=dict)
    failure_result: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    missing_tool: str = ""
    bad_policy: str = ""
    new_regression_test: Dict[str, Any] = field(default_factory=dict)
    new_governance_rule: Dict[str, Any] = field(default_factory=dict)
    future_retrieval_object: Dict[str, Any] = field(default_factory=dict)
    retrieval_tags: list[str] = field(default_factory=list)
    source_run_id: str = ""
    source_task_family: str = "generic"
    confidence: float = 0.5
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_failure_learning_object(
    *,
    failure_mode: str,
    title: str,
    summary: str,
    violated_assumption: str,
    failed_action: Mapping[str, Any] | None = None,
    failure_result: Mapping[str, Any] | None = None,
    evidence_refs: Sequence[Any] = (),
    missing_tool: str = "",
    bad_policy: str = "",
    new_regression_test: Mapping[str, Any] | None = None,
    new_governance_rule: Mapping[str, Any] | None = None,
    retrieval_tags: Sequence[Any] = (),
    source_run_id: str = "",
    source_task_family: str = "generic",
    confidence: float = 0.5,
    status: str = "active",
) -> Dict[str, Any]:
    tags = _str_list(retrieval_tags)
    mode = str(failure_mode or "unknown_failure").strip() or "unknown_failure"
    if mode not in tags:
        tags.insert(0, mode)
    future_retrieval_object = {
        "object_type": "failure_learning_object",
        "failure_mode": mode,
        "query_keys": sorted(set(tags + _str_list([missing_tool, bad_policy]))),
        "preferred_next_actions": _preferred_actions_for(mode, missing_tool=missing_tool),
        "avoid_actions": _avoid_actions_for(mode, bad_policy=bad_policy),
    }
    return FailureLearningObject(
        failure_mode=mode,
        title=str(title or mode).strip(),
        summary=_excerpt(summary, limit=360),
        violated_assumption=str(violated_assumption or "").strip(),
        failed_action=_as_dict(failed_action),
        failure_result=_as_dict(failure_result),
        evidence_refs=_str_list(evidence_refs),
        missing_tool=str(missing_tool or "").strip(),
        bad_policy=str(bad_policy or "").strip(),
        new_regression_test=_as_dict(new_regression_test),
        new_governance_rule=_as_dict(new_governance_rule),
        future_retrieval_object=future_retrieval_object,
        retrieval_tags=tags,
        source_run_id=str(source_run_id or ""),
        source_task_family=str(source_task_family or "generic"),
        confidence=max(0.0, min(1.0, float(confidence))),
        status=str(status or "active"),
    ).to_dict()


def failure_hash(failure_object: Mapping[str, Any]) -> str:
    payload = dict(failure_object or {})
    comparable = {
        "failure_mode": payload.get("failure_mode"),
        "violated_assumption": payload.get("violated_assumption"),
        "missing_tool": payload.get("missing_tool"),
        "bad_policy": payload.get("bad_policy"),
        "new_regression_test": payload.get("new_regression_test"),
        "new_governance_rule": payload.get("new_governance_rule"),
        "retrieval_tags": payload.get("retrieval_tags"),
    }
    return _hash_payload(comparable)


def failure_object_matches_tags(row: Mapping[str, Any], objective_tags: set[str]) -> bool:
    obj = _as_dict(row.get("failure_object") or row.get("object") or row)
    tags = {str(item).strip().lower() for item in _as_list(obj.get("retrieval_tags")) if str(item).strip()}
    tags.update(
        str(item).strip().lower()
        for item in _as_list(_as_dict(obj.get("future_retrieval_object")).get("query_keys"))
        if str(item).strip()
    )
    if not objective_tags:
        return not (tags - {"general", "local_machine"})
    if tags & objective_tags:
        return True
    if "ai_project_generation" in objective_tags and tags & {"python_generation", "requires_tests", "api_surface", "llm_generation"}:
        return True
    if "project_maintenance" in objective_tags and "ai_project_generation" in tags and not (tags & objective_tags):
        return False
    return bool(tags <= {"general", "local_machine"})


def build_failure_learning_hint_text(failure_objects: Sequence[Mapping[str, Any]], *, limit: int = 4) -> str:
    selected = [dict(item) for item in list(failure_objects or [])[: max(1, int(limit or 4))]]
    if not selected:
        return ""
    lines = ["Structured failure objects from prior runs:"]
    for index, row in enumerate(selected, start=1):
        obj = _as_dict(row.get("failure_object") or row.get("object") or row)
        title = str(obj.get("title") or obj.get("failure_mode") or "failure").strip()
        assumption = str(obj.get("violated_assumption") or "").strip()
        rule = _as_dict(obj.get("new_governance_rule"))
        test = _as_dict(obj.get("new_regression_test"))
        fragments = []
        if assumption:
            fragments.append(f"violated assumption: {assumption}")
        if test.get("description"):
            fragments.append(f"regression: {test.get('description')}")
        if rule.get("description"):
            fragments.append(f"governance: {rule.get('description')}")
        detail = "; ".join(fragments) or str(obj.get("summary") or "").strip()
        if detail:
            lines.append(f"{index}. {title}: {_excerpt(detail, limit=320)}")
    return "\n".join(lines)


def failure_objects_to_context_entries(failure_objects: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    for row in failure_objects:
        obj = _as_dict(row.get("failure_object") or row.get("object") or row)
        failure_id = str(row.get("failure_id") or obj.get("failure_id") or "")
        entries.append(
            {
                "source": "failure_learning",
                "schema_version": FAILURE_LEARNING_VERSION,
                "failure_id": failure_id,
                "failure_mode": str(obj.get("failure_mode") or ""),
                "title": str(obj.get("title") or ""),
                "violated_assumption": str(obj.get("violated_assumption") or ""),
                "evidence_refs": _str_list(obj.get("evidence_refs")),
                "new_regression_test": _as_dict(obj.get("new_regression_test")),
                "new_governance_rule": _as_dict(obj.get("new_governance_rule")),
                "future_retrieval_object": _as_dict(obj.get("future_retrieval_object")),
                "confidence": float(row.get("confidence", obj.get("confidence", 0.0)) or 0.0),
            }
        )
    return entries


def _preferred_actions_for(failure_mode: str, *, missing_tool: str = "") -> list[str]:
    mode = str(failure_mode or "")
    if mode in {"placeholder_generation", "missing_required_artifact"}:
        return ["file_read", "apply_patch", "run_test", "mirror_plan"]
    if mode == "api_surface_mismatch":
        return ["file_outline", "file_read", "run_test"]
    if mode == "governance_block":
        return ["investigation_status", "note_write", "run_test"]
    if missing_tool:
        return [missing_tool]
    return ["investigation_status", "file_read", "run_test"]


def _avoid_actions_for(failure_mode: str, *, bad_policy: str = "") -> list[str]:
    mode = str(failure_mode or "")
    avoid = []
    if mode in {"placeholder_generation", "missing_required_artifact"}:
        avoid.append("mirror_apply_before_artifact_contract")
    if mode == "api_surface_mismatch":
        avoid.append("invent_api_without_file_outline")
    if bad_policy:
        avoid.append(bad_policy)
    return avoid

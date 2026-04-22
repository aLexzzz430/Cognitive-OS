from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.objects import OBJECT_TYPE_SKILL, OBJECT_TYPE_TRANSFER, infer_object_type
from modules.memory.router import MEMORY_ROUTE_PROCEDURAL, route_object_record


def _function_name(record: Dict[str, Any]) -> str:
    content = record.get("content", {})
    if not isinstance(content, dict):
        return ""
    invocation = content.get("invocation_schema", {})
    if isinstance(invocation, dict) and invocation.get("function_name"):
        return str(invocation.get("function_name") or "").strip()
    tool_args = content.get("tool_args", {})
    if isinstance(tool_args, dict) and tool_args.get("function_name"):
        return str(tool_args.get("function_name") or "").strip()
    return str(content.get("function_name") or "").strip()


def _beneficial_reuse_count(record: Dict[str, Any]) -> int:
    return sum(1 for item in record.get("reuse_history", []) or [] if item.get("was_beneficial", False))


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, (tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


class ProceduralMemoryStore:
    """Read facade over procedural memories and transfer-capable objects."""

    def __init__(self, object_store):
        self._store = object_store

    def list_active(self, *, limit: int = 50, include_transfers: bool = True) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in self._store.retrieve(sort_by="surface_priority", limit=max(limit * 5, 50)):
            if route_object_record(record) != MEMORY_ROUTE_PROCEDURAL:
                continue
            object_type = infer_object_type(record)
            if not include_transfers and object_type == OBJECT_TYPE_TRANSFER:
                continue
            rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def retrieve_for_function(
        self,
        function_name: str,
        *,
        task_family: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        fn = str(function_name or "").strip()
        family_filter = str(task_family or "").strip()
        rows: List[Dict[str, Any]] = []
        for record in self.list_active(limit=max(limit * 5, 50), include_transfers=False):
            if fn and _function_name(record) != fn:
                continue
            if family_filter:
                applicability = record.get("applicability", {}) if isinstance(record.get("applicability"), dict) else {}
                applicable_families = _normalize_list(applicability.get("task_family", []))
                if applicable_families and family_filter not in applicable_families:
                    continue
            rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def summarize_skill(self, record: Dict[str, Any]) -> Dict[str, Any]:
        content = record.get("content", {})
        skill_kind = str(
            record.get("skill_kind")
            or (content.get("skill_kind") if isinstance(content, dict) else "")
            or record.get("memory_type")
            or ""
        ).strip()
        applicability = record.get("applicability", {}) if isinstance(record.get("applicability"), dict) else {}
        failure_conditions = _normalize_list(
            record.get("failure_conditions")
            or (content.get("failure_conditions") if isinstance(content, dict) else [])
        )
        return {
            "object_id": str(record.get("object_id") or ""),
            "object_type": infer_object_type(record),
            "skill_kind": skill_kind,
            "function_name": _function_name(record),
            "beneficial_reuse_count": _beneficial_reuse_count(record),
            "applicability": applicability,
            "failure_conditions": failure_conditions,
            "asset_status": str(record.get("asset_status") or ""),
            "confidence": float(record.get("confidence", 0.0) or 0.0),
        }

    def summarize(self) -> Dict[str, Any]:
        skills = [self.summarize_skill(record) for record in self.list_active(limit=500)]
        callable_count = sum(1 for row in skills if row.get("function_name"))
        transfer_count = sum(1 for row in skills if row.get("object_type") == OBJECT_TYPE_TRANSFER)
        return {
            "total": len(skills),
            "callable_count": callable_count,
            "transfer_count": transfer_count,
            "high_reuse_count": sum(1 for row in skills if int(row.get("beneficial_reuse_count", 0) or 0) >= 2),
        }

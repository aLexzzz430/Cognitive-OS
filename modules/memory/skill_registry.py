from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from core.objects import OBJECT_TYPE_SKILL, infer_object_type
from modules.memory.procedural_store import ProceduralMemoryStore


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


class SkillRegistry:
    """Canonical view of reusable callable skills formed from procedural memory."""

    def __init__(self, object_store):
        self._store = object_store
        self._procedural_store = ProceduralMemoryStore(object_store)

    def _beneficial_reuse_count(self, record: Dict[str, Any]) -> int:
        return sum(1 for item in record.get("reuse_history", []) or [] if item.get("was_beneficial", False))

    def _post_teacher_beneficial_count(self, record: Dict[str, Any]) -> int:
        count = 0
        for item in record.get("reuse_history", []) or []:
            if not item.get("was_beneficial", False):
                continue
            if item.get("teacher_present") is False:
                count += 1
        return count

    def _invocation_schema(self, record: Dict[str, Any]) -> Dict[str, Any]:
        content = record.get("content", {})
        if not isinstance(content, dict):
            return {}
        invocation = content.get("invocation_schema", {})
        if isinstance(invocation, dict) and invocation.get("function_name"):
            return dict(invocation)
        tool_args = content.get("tool_args", {})
        if isinstance(tool_args, dict) and tool_args.get("function_name"):
            return {
                "callable_form": "call_tool",
                "function_name": str(tool_args.get("function_name") or ""),
                "kwargs_schema": dict(tool_args.get("kwargs_schema", {})) if isinstance(tool_args.get("kwargs_schema"), dict) else {},
            }
        function_name = str(content.get("function_name") or "").strip()
        if function_name:
            return {
                "callable_form": str(content.get("callable_form") or "call_tool"),
                "function_name": function_name,
            }
        return {}

    def _source_evidence_chain(self, record: Dict[str, Any]) -> List[str]:
        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        evidence = []
        evidence.extend(_normalize_list(record.get("evidence_ids", [])))
        evidence.extend(_normalize_list(record.get("supporting_evidence", [])))
        evidence.extend(_normalize_list(record.get("reuse_evidence", [])))
        evidence.extend(_normalize_list(content.get("source_evidence_chain", [])))
        return _normalize_list(evidence)

    def _negative_examples(self, record: Dict[str, Any]) -> List[str]:
        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        explicit = _normalize_list(content.get("negative_examples", []))
        if explicit:
            return explicit
        derived = []
        for item in record.get("reuse_history", []) or []:
            if item.get("was_beneficial", False):
                continue
            reason = str(item.get("reason") or item.get("failure") or item.get("context_family") or "non_beneficial_reuse")
            derived.append(reason)
        return _normalize_list(derived)

    def _expected_gains(self, record: Dict[str, Any]) -> List[str]:
        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        explicit = _normalize_list(content.get("expected_gains", []))
        if explicit:
            return explicit
        beneficial = self._beneficial_reuse_count(record)
        if beneficial > 0:
            return [f"beneficial_reuse:{beneficial}"]
        return []

    def _is_agent_owned(self, record: Dict[str, Any]) -> bool:
        memory_metadata = record.get("memory_metadata", {}) if isinstance(record.get("memory_metadata"), dict) else {}
        owner_scope = str(record.get("owner_scope") or memory_metadata.get("owner_scope") or "system").strip().lower()
        if owner_scope != "teacher":
            return True
        return self._post_teacher_beneficial_count(record) > 0

    def _materialize_skill(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if infer_object_type(record) != OBJECT_TYPE_SKILL:
            return None
        invocation_schema = self._invocation_schema(record)
        function_name = str(invocation_schema.get("function_name") or "").strip()
        if not function_name or not self._is_agent_owned(record):
            return None

        content = record.get("content", {}) if isinstance(record.get("content"), dict) else {}
        applicability = record.get("applicability", {}) if isinstance(record.get("applicability"), dict) else {}
        failure_conditions = _normalize_list(record.get("failure_conditions") or content.get("failure_conditions", []))
        skill_id = f"s_{str(record.get('object_id') or '')[:8]}"
        return {
            "skill_id": skill_id,
            "object_id": str(record.get("object_id") or ""),
            "skill_type": str(record.get("skill_kind") or content.get("skill_kind") or record.get("memory_type") or "rewrite"),
            "content": content,
            "hints": dict(content.get("rewrite_hints", {})) if isinstance(content.get("rewrite_hints"), dict) else {},
            "invocation_schema": invocation_schema,
            "callable_form": str(invocation_schema.get("callable_form") or content.get("callable_form") or "call_tool"),
            "applicability": applicability,
            "failure_conditions": failure_conditions,
            "expected_gains": self._expected_gains(record),
            "negative_examples": self._negative_examples(record),
            "source_evidence_chain": self._source_evidence_chain(record),
            "beneficial_reuse_count": self._beneficial_reuse_count(record),
            "confidence": float(record.get("confidence", 0.0) or 0.0),
            "conditions": [],
        }

    def list_registered_skills(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        skills: List[Dict[str, Any]] = []
        for record in self._procedural_store.list_active(limit=max(limit * 5, 50), include_transfers=False):
            skill = self._materialize_skill(record)
            if skill is None:
                continue
            skills.append(skill)
        skills.sort(
            key=lambda row: (
                int(row.get("beneficial_reuse_count", 0) or 0),
                float(row.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return skills[:limit]

    def match_hypothesis(
        self,
        hyp,
        *,
        top_k: int = 3,
        invalidated_skill_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        invalidated = set(invalidated_skill_ids or set())
        claim = str(getattr(hyp, "claim", "") or "")
        family = str(getattr(hyp, "family", "") or "")
        quoted = set(re.findall(r"'([^']+)'", claim))
        matched: List[Dict[str, Any]] = []

        for skill in self.list_registered_skills(limit=max(top_k * 4, 12)):
            if skill["skill_id"] in invalidated:
                continue
            function_name = str(skill.get("invocation_schema", {}).get("function_name") or "").strip()
            conditions: List[str] = []
            if function_name and (function_name in quoted or function_name in claim):
                conditions.append(f"applies:{function_name}")
            applicable_families = _normalize_list(skill.get("applicability", {}).get("task_family", []))
            if family and applicable_families and family in applicable_families:
                conditions.append(f"family:{family}")
            if not conditions and quoted:
                continue
            enriched = dict(skill)
            enriched["conditions"] = conditions
            matched.append(enriched)
            if len(matched) >= top_k:
                break
        return matched

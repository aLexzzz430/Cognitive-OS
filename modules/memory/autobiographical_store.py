from __future__ import annotations

from typing import Any, Dict, List

from core.objects import OBJECT_TYPE_AUTOBIOGRAPHICAL, infer_object_type
from modules.memory.router import MEMORY_ROUTE_AUTOBIOGRAPHICAL, MEMORY_ROUTE_EPISODIC, route_object_record
from modules.memory.schema import MemoryType


def _episode_ref_for_record(record: Dict[str, Any]) -> str:
    if str(record.get("memory_type") or "") == MemoryType.EPISODE_RECORD.value:
        content = record.get("content", {})
        episode_id = ""
        if isinstance(content, dict):
            episode_id = str(content.get("episode_id") or "").strip()
        return f"ep-{episode_id}" if episode_id else ""
    refs = record.get("episode_refs", [])
    if isinstance(refs, list) and refs:
        return str(refs[0] or "").strip()
    content = record.get("content", {})
    if isinstance(content, dict):
        content_refs = content.get("episode_refs", [])
        if isinstance(content_refs, list) and content_refs:
            return str(content_refs[0] or "").strip()
    return ""


class AutobiographicalMemoryStore:
    """Read facade for autobiographical objects plus episode records that ground continuity."""

    def __init__(self, object_store):
        self._store = object_store

    def list_entries(self, *, limit: int = 20, include_episode_records: bool = True) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in self._store.retrieve(sort_by="surface_priority", limit=max(limit * 5, 50)):
            route = route_object_record(record)
            if route == MEMORY_ROUTE_AUTOBIOGRAPHICAL:
                rows.append(record)
            elif include_episode_records and route == MEMORY_ROUTE_EPISODIC:
                rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def retrieve_by_episode_ref(self, episode_ref: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        target = str(episode_ref or "").strip()
        if not target:
            return []
        rows: List[Dict[str, Any]] = []
        for record in self.list_entries(limit=max(limit * 5, 30)):
            if _episode_ref_for_record(record) == target:
                rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def identity_salient(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in self.list_entries(limit=max(limit * 5, 30)):
            content = record.get("content", {})
            if infer_object_type(record) == OBJECT_TYPE_AUTOBIOGRAPHICAL:
                rows.append(record)
                continue
            if isinstance(content, dict) and content.get("identity_implications"):
                rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def summarize(self) -> Dict[str, Any]:
        rows = self.list_entries(limit=500)
        return {
            "total": len(rows),
            "autobiographical_objects": sum(1 for row in rows if route_object_record(row) == MEMORY_ROUTE_AUTOBIOGRAPHICAL),
            "episode_records": sum(1 for row in rows if str(row.get("memory_type") or "") == MemoryType.EPISODE_RECORD.value),
        }

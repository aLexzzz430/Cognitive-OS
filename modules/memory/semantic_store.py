from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from modules.memory.router import MEMORY_ROUTE_MECHANISM, MEMORY_ROUTE_SEMANTIC, route_object_record


def _search_blob(record: Dict[str, Any]) -> str:
    content = record.get("content", {})
    return " ".join(
        [
            str(record.get("summary") or ""),
            str(record.get("family") or ""),
            str(content if isinstance(content, dict) else ""),
            " ".join(str(tag) for tag in record.get("retrieval_tags", []) or []),
        ]
    ).lower()


class SemanticMemoryStore:
    """Read facade over semantic and mechanism memory without creating a second truth store."""

    def __init__(self, object_store):
        self._store = object_store

    def list_active(self, *, limit: int = 50, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in self._store.retrieve(sort_by="confidence", limit=max(limit * 5, 50)):
            if route_object_record(record) not in (MEMORY_ROUTE_SEMANTIC, MEMORY_ROUTE_MECHANISM):
                continue
            if float(record.get("confidence", 0.0) or 0.0) < float(min_confidence):
                continue
            rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def retrieve(
        self,
        *,
        query: str = "",
        retrieval_tags: Optional[Iterable[str]] = None,
        family: str = "",
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        normalized_query = str(query or "").strip().lower()
        required_tags = {str(tag).strip() for tag in retrieval_tags or [] if str(tag).strip()}
        family_filter = str(family or "").strip()
        rows = []
        for record in self.list_active(limit=max(limit * 5, 50), min_confidence=min_confidence):
            if family_filter and str(record.get("family") or "").strip() != family_filter:
                continue
            if required_tags and not (required_tags & set(record.get("retrieval_tags", []) or [])):
                continue
            if normalized_query and normalized_query not in _search_blob(record):
                continue
            rows.append(record)
            if len(rows) >= limit:
                break
        return rows

    def summarize(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for record in self.list_active(limit=500):
            memory_type = str(record.get("memory_type") or record.get("type") or "unknown")
            counts[memory_type] = counts.get(memory_type, 0) + 1
        return {
            "total": sum(counts.values()),
            "by_memory_type": counts,
        }

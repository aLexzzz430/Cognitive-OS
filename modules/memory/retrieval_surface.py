from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.objects import (
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
    infer_object_type,
)
from modules.memory.router import MemoryRouter


def _sort_surface(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(records),
        key=lambda row: (
            float(row.get("surface_priority", 0.0) or 0.0),
            float(row.get("confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )


class RetrievalSurface:
    """Protocol layer that groups surfaced objects by downstream consumer."""

    def __init__(self, object_store=None):
        self._store = object_store
        self._router = MemoryRouter()

    def surface_object_records(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        limit: int = 10,
    ) -> Dict[str, Any]:
        surface_ids: List[str] = []
        semantic_ids: List[str] = []
        mechanism_ids: List[str] = []
        skill_ids: List[str] = []
        transfer_ids: List[str] = []
        autobiographical_ids: List[str] = []
        identity_ids: List[str] = []

        for record in _sort_surface(records):
            object_id = str(record.get("object_id") or "").strip()
            if not object_id:
                continue
            object_type = infer_object_type(record)
            memory_layer = str(record.get("memory_layer") or "").strip().lower()
            if object_type == OBJECT_TYPE_REPRESENTATION and memory_layer == "mechanism":
                mechanism_ids.append(object_id)
            elif object_type == OBJECT_TYPE_REPRESENTATION:
                semantic_ids.append(object_id)
            elif object_type == OBJECT_TYPE_SKILL:
                skill_ids.append(object_id)
            elif object_type == OBJECT_TYPE_TRANSFER:
                transfer_ids.append(object_id)
            elif object_type == OBJECT_TYPE_AUTOBIOGRAPHICAL:
                autobiographical_ids.append(object_id)
            elif object_type == OBJECT_TYPE_IDENTITY:
                identity_ids.append(object_id)
            if len(surface_ids) < limit:
                surface_ids.append(object_id)

        route_ids = self._router.summarize_ids(records)
        return {
            "surfaced_object_ids": surface_ids,
            "semantic_object_ids": semantic_ids[:limit],
            "mechanism_object_ids": mechanism_ids[:limit],
            "planner_prior_object_ids": skill_ids[:limit],
            "cross_domain_prior_object_ids": transfer_ids[:limit],
            "autobiographical_object_ids": autobiographical_ids[:limit],
            "identity_object_ids": identity_ids[:limit],
            "memory_route_ids": route_ids,
            "memory_route_counts": {
                route: len(object_ids)
                for route, object_ids in route_ids.items()
                if object_ids
            },
        }

    def surface_from_bundle(self, bundle, *, limit: Optional[int] = None) -> Dict[str, Any]:
        if self._store is None:
            return self.surface_object_records([], limit=limit or 10)
        records = self._store.retrieve_with_bundle(bundle)
        return self.surface_object_records(records, limit=limit or getattr(bundle, "max_results", 10))

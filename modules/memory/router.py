from __future__ import annotations

from typing import Any, Dict, Iterable, List

from core.objects import (
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
    infer_object_type,
)


MEMORY_ROUTE_SEMANTIC = "semantic"
MEMORY_ROUTE_PROCEDURAL = "procedural"
MEMORY_ROUTE_AUTOBIOGRAPHICAL = "autobiographical"
MEMORY_ROUTE_CONTINUITY = "continuity"
MEMORY_ROUTE_EPISODIC = "episodic"
MEMORY_ROUTE_MECHANISM = "mechanism"

ALL_MEMORY_ROUTES = (
    MEMORY_ROUTE_SEMANTIC,
    MEMORY_ROUTE_PROCEDURAL,
    MEMORY_ROUTE_AUTOBIOGRAPHICAL,
    MEMORY_ROUTE_CONTINUITY,
    MEMORY_ROUTE_EPISODIC,
    MEMORY_ROUTE_MECHANISM,
)


def route_object_record(record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return MEMORY_ROUTE_SEMANTIC

    object_type = infer_object_type(record)
    memory_layer = str(record.get("memory_layer") or "").strip().lower()

    if memory_layer == "episodic":
        return MEMORY_ROUTE_EPISODIC
    if object_type == OBJECT_TYPE_AUTOBIOGRAPHICAL:
        return MEMORY_ROUTE_AUTOBIOGRAPHICAL
    if object_type == OBJECT_TYPE_IDENTITY or memory_layer == "continuity":
        return MEMORY_ROUTE_CONTINUITY
    if object_type in (OBJECT_TYPE_SKILL, OBJECT_TYPE_TRANSFER) or memory_layer == "procedural":
        return MEMORY_ROUTE_PROCEDURAL
    if memory_layer == "episodic":
        return MEMORY_ROUTE_EPISODIC
    if memory_layer == "mechanism":
        return MEMORY_ROUTE_MECHANISM
    return MEMORY_ROUTE_SEMANTIC


class MemoryRouter:
    """Canonical routing helper for formal memory objects."""

    def route(self, record: Dict[str, Any]) -> str:
        return route_object_record(record)

    def group_records(self, records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {route: [] for route in ALL_MEMORY_ROUTES}
        for record in records:
            route = self.route(record)
            grouped.setdefault(route, []).append(record)
        return grouped

    def summarize_ids(self, records: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
        grouped_ids: Dict[str, List[str]] = {route: [] for route in ALL_MEMORY_ROUTES}
        for record in records:
            route = self.route(record)
            object_id = str(record.get("object_id") or "").strip()
            if object_id:
                grouped_ids.setdefault(route, []).append(object_id)
        return grouped_ids

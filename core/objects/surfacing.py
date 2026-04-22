from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from core.objects.adapters import record_to_cognitive_object


def _sort_key(record: Dict[str, object]) -> tuple[float, float, int]:
    typed = record_to_cognitive_object(record)
    return (
        float(getattr(typed, "surface_priority", 0.0) or 0.0),
        float(getattr(typed, "confidence", 0.0) or 0.0),
        int(getattr(typed, "version", 0) or 0),
    )


def select_surfaced_objects(
    records: Iterable[Dict[str, object]],
    *,
    object_types: Optional[Sequence[str]] = None,
    asset_statuses: Optional[Sequence[str]] = None,
    limit: int = 10,
) -> List[Dict[str, object]]:
    allowed_types = {str(item).strip() for item in list(object_types or []) if str(item).strip()}
    allowed_statuses = {str(item).strip() for item in list(asset_statuses or []) if str(item).strip()}

    filtered: List[Dict[str, object]] = []
    for record in records:
        typed = record_to_cognitive_object(record)
        if allowed_types and typed.object_type not in allowed_types:
            continue
        if allowed_statuses and str(record.get("asset_status") or "") not in allowed_statuses:
            continue
        filtered.append(dict(record))

    filtered.sort(key=_sort_key, reverse=True)
    return filtered[: max(0, int(limit))]


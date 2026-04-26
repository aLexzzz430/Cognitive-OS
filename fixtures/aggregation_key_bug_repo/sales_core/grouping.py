from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def group_rows(rows: Iterable[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        bucket_key = row["region"]
        grouped.setdefault(bucket_key, []).append(row)
    return grouped

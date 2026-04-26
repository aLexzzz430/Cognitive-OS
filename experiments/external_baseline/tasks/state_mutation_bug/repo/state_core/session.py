from __future__ import annotations

from collections.abc import Iterable

from .state import ItemStore


def visible_items(items: Iterable[str]) -> list[str]:
    store = ItemStore(items)
    exposed = store.snapshot()
    exposed.append("debug-marker")
    return store.snapshot()

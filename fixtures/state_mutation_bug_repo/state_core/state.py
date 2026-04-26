from __future__ import annotations

from collections.abc import Iterable


class ItemStore:
    def __init__(self, items: Iterable[str]) -> None:
        self._items = list(items)

    def snapshot(self) -> list[str]:
        return self._items

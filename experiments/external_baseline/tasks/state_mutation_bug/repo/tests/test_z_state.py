from __future__ import annotations

from state_core.state import ItemStore


def test_snapshot_returns_defensive_copy() -> None:
    store = ItemStore(["alpha", "beta"])
    leaked = store.snapshot()
    leaked.append("debug-marker")

    assert store.snapshot() == ["alpha", "beta"], "snapshot should be a defensive copy"

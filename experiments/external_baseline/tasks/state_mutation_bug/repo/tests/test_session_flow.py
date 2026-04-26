from __future__ import annotations

from state_core.session import visible_items


def test_visible_items_does_not_leak_internal_state() -> None:
    assert visible_items(["alpha", "beta"]) == ["alpha", "beta"], "internal state leak through exposed snapshot"

"""Minimal surface data types retained by the distilled runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    side_effects: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    capability_class: str = ""
    side_effect_class: str = ""
    approval_required: bool = False
    risk_level: str = "low"


@dataclass
class SurfaceObservation:
    text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    available_tools: list[ToolSpec] = field(default_factory=list)
    terminal: bool = False
    reward: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    ok: bool
    observation: SurfaceObservation
    events: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Dict-like access delegated to raw for legacy call sites."""
        return self.raw.get(key, default)

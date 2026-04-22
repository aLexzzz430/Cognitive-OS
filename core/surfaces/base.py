"""
core/surfaces/base.py

Unified surface + abstraction + research agenda ports.
CoreMainLoop only knows these protocols, not any specific domain adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Surface Port
# =============================================================================

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
class SurfaceAction:
    kind: str  # "call_tool" | "wait" | "submit"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    ok: bool
    observation: SurfaceObservation
    events: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Dict-like access delegated to raw."""
        return self.raw.get(key, default)


@runtime_checkable
class SurfaceAdapter(Protocol):
    """
    Surface port: only handles observation and tool execution.
    MUST NOT directly write to long-term state.
    """

    def reset(self, seed: int | None = None) -> SurfaceObservation: ...
    def observe(self) -> SurfaceObservation: ...
    def act(self, action: SurfaceAction | dict | str) -> ActionResult: ...


# =============================================================================
# Abstraction Port
# =============================================================================

@dataclass
class AbstractionProposal:
    """
    A candidate abstraction for the formal object layer.
    Produced by AbstractionProvider.propose().
    Not yet committed — must go through Step 9/10.
    """
    kind: str  # "create" | "update" | "retire" | "freeze" | "redirect"
    abstraction_id: str
    claim: str  # What this abstraction represents
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractionProvider(Protocol):
    """
    Abstraction port: proposes candidate abstractions from surfaced items.
    CoreMainLoop calls this in Step 3/4 to get abstraction proposals.
    """

    def propose(
        self,
        surfaced_items: list[dict[str, Any]],
        runtime_state: dict[str, Any],
        object_context: dict[str, Any],
    ) -> list[AbstractionProposal]:
        """Generate abstraction proposals from surfaced items."""
        ...


class NullAbstractionProvider:
    """
    Null implementation: returns no proposals.
    Graceful degradation when no abstraction system is available.
    """

    def propose(self, surfaced_items, runtime_state, object_context) -> list[AbstractionProposal]:
        return []


# =============================================================================
# Research Agenda Port
# =============================================================================

@dataclass
class ResearchProposal:
    """
    A candidate research agenda item.
    Produced by ResearchAgendaProvider.propose().
    Not yet committed — must go through Step 9/10.
    """
    title: str
    question: str
    rationale: str
    suggested_tests: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchAgendaProvider(Protocol):
    """
    Research agenda port: proposes research agenda items from surfaced conflicts.
    CoreMainLoop calls this in Step 3/4 to get agenda proposals.
    """

    def propose(
        self,
        surfaced_conflicts: list[dict[str, Any]],
        runtime_state: dict[str, Any],
        object_context: dict[str, Any],
    ) -> list[ResearchProposal]:
        """Generate research proposals from surfaced conflicts."""
        ...


class NullResearchAgendaProvider:
    """
    Null implementation: returns no proposals.
    Graceful degradation when no research agenda system is available.
    """

    def propose(self, surfaced_conflicts, runtime_state, object_context) -> list[ResearchProposal]:
        return []


# =============================================================================
# Evidence Packet (Step 9 output)
# =============================================================================

@dataclass
class EvidencePacket:
    """
    Normalized evidence from Step 9.

    All providers (C1, C5, OA, etc.) output proposals.
    Step 9 translates proposals into evidence packets.
    Step 10 commits evidence packets to formal long-term objects.
    """
    type: str  # "hypothesis" | "abstraction" | "risk" | "recovery" (legacy alias: kind)
    content: dict[str, Any]
    source_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5

    # Legacy compatibility
    kind: str = ""  # alias for type
    claim: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Support both naming conventions
        if not self.type and self.kind:
            self.type = self.kind
        if self.kind == "":
            self.kind = self.type
        if not self.payload and self.content:
            self.payload = self.content
        if self.claim and not self.content:
            self.content = {"claim": self.claim}


# =============================================================================
# Raw Evidence Extractor (Step 9 Phase 1)
# =============================================================================

class RawEvidenceExtractor(Protocol):
    """
    Extracts raw evidence from tool calls, observations, errors, and state deltas.

    This is C1/C5-agnostic. Step 9 Phase 1 always runs this first,
    regardless of whether AbstractionProvider or ResearchAgendaProvider are available.

    Core principle:
      - Raw evidence exists FIRST
      - Proposals are just extra benefit
      - Without proposals, it should NOT equal no evidence
    """

    def extract(
        self,
        observation_before: dict[str, Any],
        action_taken: dict[str, Any],
        action_result: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> list[EvidencePacket]:
        """
        Extract raw evidence packets from a single tool call interaction.
        """
        ...


class NullRawEvidenceExtractor:
    """Null extractor: returns no evidence. Use when no domain-specific extractor exists."""

    def extract(self, observation_before, action_taken, action_result, runtime_state) -> list[EvidencePacket]:
        return []

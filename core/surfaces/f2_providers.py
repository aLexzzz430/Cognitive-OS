"""
core/surfaces/f2_abstraction_provider.py

Wraps existing F2 C1AbstractionAdapter behind AbstractionProvider protocol.
Preserves existing behavior while making it port-compatible.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.c1_adapter import C1AbstractionAdapter

from core.surfaces.base import AbstractionProvider, AbstractionProposal


class F2AbstractionProvider:
    """
    Wraps F2's C1AbstractionAdapter behind AbstractionProvider protocol.

    C1AbstractionAdapter methods used:
      - get_active_abstractions() → AbstractionProposal list
      - create_abstraction()     → creates new abstraction
      - update_abstraction()      → updates existing
      - retire_abstraction()       → retires

    This adapter maps C1's output to the generic AbstractionProposal format.
    """

    def __init__(self, c1_adapter: Any):
        self._c1 = c1_adapter

    def propose(
        self,
        surfaced_items: list[dict[str, Any]],
        runtime_state: dict[str, Any],
        object_context: dict[str, Any],
    ) -> list[AbstractionProposal]:
        """
        Generate abstraction proposals from surfaced items.

        C1 logic: abstractions are formed from:
          - Observed patterns in surface items
          - Runtime state (gaps, contradictions, etc.)
          - Object context (existing abstractions)
        """
        proposals = []

        # Get active abstractions from C1
        active_abstractions = []
        if hasattr(self._c1, 'get_active_abstractions'):
            try:
                raw_abstractions = self._c1.get_active_abstractions()
                active_abstractions = raw_abstractions if raw_abstractions else []
            except Exception:
                active_abstractions = []

        # Convert C1 abstractions to AbstractionProposals
        for abstr in active_abstractions:
            if hasattr(abstr, 'abstraction_id'):
                prop = AbstractionProposal(
                    kind='retain',  # Default: keep existing abstraction
                    abstraction_id=abstr.abstraction_id,
                    claim=getattr(abstr, 'claim', str(abstr)),
                    evidence_refs=[],
                    metadata={'status': getattr(abstr, 'status', 'active')},
                )
                proposals.append(prop)

        # Also propose from surfaced items
        # C1 creates abstractions from observed patterns
        for item in surfaced_items:
            # Pattern: if item has no matching abstraction, propose creation
            item_id = item.get('id', item.get('name', ''))
            if item_id:
                # Check if this item has an abstraction
                has_abstraction = any(
                    p.abstraction_id == item_id for p in proposals
                )
                if not has_abstraction:
                    proposals.append(AbstractionProposal(
                        kind='create',
                        abstraction_id=f"ab_{item_id}",
                        claim=f"Abstraction for {item_id}",
                        evidence_refs=[item_id],
                        metadata={'source': 'surfaced_item'},
                    ))

        return proposals

    def apply_verdict(self, verdict: dict[str, Any]) -> None:
        """
        Apply an object-level verdict from Step 10.
        verdict: {abstraction_id, verdict_type, evidence_refs}
        """
        if not hasattr(self._c1, 'apply_verdict'):
            return

        abstraction_id = verdict.get('abstraction_id')
        verdict_type = verdict.get('verdict_type')
        evidence_refs = verdict.get('evidence_refs', [])

        if verdict_type == 'retire':
            if hasattr(self._c1, 'retire_abstraction'):
                self._c1.retire_abstraction(abstraction_id)
        elif verdict_type == 'freeze':
            if hasattr(self._c1, 'freeze_abstraction'):
                self._c1.freeze_abstraction(abstraction_id)
        elif verdict_type == 'update':
            if hasattr(self._c1, 'update_abstraction'):
                self._c1.update_abstraction(abstraction_id, evidence_refs)


# =============================================================================
# F2 Research Agenda Provider
# =============================================================================

class F2ResearchAgendaProvider:
    """
    Wraps F2's C5PlannerAdapter behind ResearchAgendaProvider protocol.

    C5 logic: research agenda items are formed from:
      - Gap analysis (from C5PlannerAdapter)
      - Contradictions between world model and observations
      - Runtime state (unresolved tensions)
    """

    def __init__(self, c5_adapter: Any):
        self._c5 = c5_adapter

    def propose(
        self,
        surfaced_conflicts: list[dict[str, Any]],
        runtime_state: dict[str, Any],
        object_context: dict[str, Any],
    ) -> list[dict]:  # Returns ResearchProposal dataclasses
        """
        Generate research proposals from surfaced conflicts.

        Returns list of proposal dicts (compatible with ResearchProposal).
        """
        from core.surfaces.base import ResearchProposal

        proposals = []

        # Get open gaps from C5
        open_gaps = []
        if hasattr(self._c5, 'get_active_gaps'):
            try:
                raw_gaps = self._c5.get_active_gaps()
                open_gaps = raw_gaps if raw_gaps else []
            except Exception:
                open_gaps = []

        # Convert C5 gaps to ResearchProposals
        for gap in open_gaps:
            gap_id = getattr(gap, 'gap_id', str(gap))
            description = getattr(gap, 'description', '')

            proposals.append(ResearchProposal(
                title=f"Research Gap: {gap_id}",
                question=f"What is the cause of {gap_id}?",
                rationale=description or f"Gap {gap_id} requires investigation",
                suggested_tests=[],
                metadata={'gap_id': gap_id, 'source': 'c5_adapter'},
            ))

        # Propose from surfaced conflicts
        for conflict in surfaced_conflicts:
            conflict_type = conflict.get('type', 'unknown')
            proposals.append(ResearchProposal(
                title=f"Conflict: {conflict_type}",
                question=f"How to resolve {conflict_type} conflict?",
                rationale=str(conflict.get('details', '')),
                suggested_tests=[],
                metadata={'source': 'surfaced_conflict'},
            ))

        return proposals

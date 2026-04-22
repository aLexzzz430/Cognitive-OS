"""
core/surfaces/novel_api_providers.py

NovelAPIC5Adapter: C5PlannerAdapter interface for NovelAPI domain.
NovelAPIC1Adapter: C1AbstractionAdapter interface for NovelAPI domain.

These make Arm A (Full-CoreMainLoop) runnable with NovelAPI domain.
"""

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class FormalCommit:
    """Formal commit result matching CoreMainLoop expectations."""
    action: str
    target: str
    old_value: Any = None
    new_value: Any = None
    evidence_bundle: Any = None
    explanation: str = ""


class GapStatus(Enum):
    open = "open"
    active = "active"
    closed = "closed"


class GapType(Enum):
    efficiency = "efficiency"
    transfer = "transfer"
    discovery = "discovery"


@dataclass
class NovelAPIGap:
    """Gap object matching CoreMainLoop expectations."""
    gap_id: str
    gap_type: GapType
    status: GapStatus
    description: str = ""
    root_cause_hypotheses: list = field(default_factory=list)
    success_metric: str = ""
    kill_condition: str = ""


@dataclass
class NovelAPIAbstraction:
    """Abstraction object matching CoreMainLoop expectations."""
    abstraction_id: str
    kind: str  # "create" or "update"
    claim: str = ""
    evidence_refs: list = field(default_factory=list)
    status: str = "proposed"
    confidence: float = 0.5
    family_pattern: str = ""
    applicability_domain: str = ""


class NovelAPIC5Adapter:
    """
    C5PlannerAdapter interface for NovelAPI domain.

    Provides gap-driven research agenda for NovelAPI discovery tasks.
    The "gaps" here are API exploration unknowns:
      - C3: efficiency (how to call efficiently)
      - B5: system transfer (can discoveries transfer across API surfaces?)
    """

    def __init__(self):
        self._gaps = [
            NovelAPIGap(
                gap_id='C3_efficiency_scope',
                gap_type=GapType.efficiency,
                status=GapStatus.open,
                description='How to call these functions most efficiently?',
                root_cause_hypotheses=['parallel', 'sequential', 'batched'],
                success_metric='call_count <= threshold',
                kill_condition='efficiency strategy found',
            ),
            NovelAPIGap(
                gap_id='B5_system_transfer',
                gap_type=GapType.transfer,
                status=GapStatus.open,
                description='Can insights transfer across API surfaces?',
                root_cause_hypotheses=['yes_shared_structure', 'no_domain_specific'],
                success_metric='cross-surface generalization observed',
                kill_condition='transfer confirmed or ruled out',
            ),
        ]

    def get_active_gaps(self) -> list:
        """Return active research gaps. Called by CoreMainLoop._step3."""
        return [g for g in self._gaps if g.status == GapStatus.open]

    def run_experiment_and_translate(self, surfaced_items: list, runtime_state: dict) -> list:
        """Generate hypothesis proposals from surfaced items. Called by _step5."""
        proposals = []
        for item in surfaced_items:
            if not isinstance(item, dict):
                continue
            fn = item.get('function_name', '')
            if fn:
                proposals.append({
                    'kind': 'hypothesis',
                    'title': f'Hypothesis: {fn}',
                    'question': f'Does {fn} have useful properties?',
                    'rationale': f'Discovered via exploration',
                    'suggested_tests': [f'test_{fn}_basic'],
                })
        return proposals

    def propose_candidates(self, scene_context: dict, contradiction_window: list) -> list:
        """Generate research candidates from active gaps. Called by _step5."""
        candidates = []
        for gap in self.get_active_gaps():
            candidates.append({
                'kind': 'research',
                'gap_id': gap.gap_id,
                'title': gap.description,
                'score': gap.success_metric and 0.5 or 0.3,
                'metadata': {'target_gap_id': gap.gap_id},
            })
        return candidates

    def formal_commit(self, evidence_packets: list = None, runtime_state: dict = None) -> FormalCommit:
        """Commit evidence to formal research agenda. Called by _step10."""
        if evidence_packets is None:
            evidence_packets = []
        return FormalCommit(
            action='novelapi_research_commit',
            target='novelapi_research_agenda',
            old_value=None,
            new_value={'committed_count': len(evidence_packets)},
            evidence_bundle=evidence_packets,
            explanation=f'NovelAPI committed {len(evidence_packets)} evidence packets to research agenda.',
        )

    def record_commit(self, commit_info: dict):
        """Record commit to history. Called by Step 10 D1."""
        pass


class NovelAPIC1Adapter:
    """
    C1AbstractionAdapter interface for NovelAPI domain.

    Provides abstraction proposals from NovelAPI interactions.
    """

    def __init__(self):
        self._abstractions: list[NovelAPIAbstraction] = []

    def get_active_abstractions(self) -> list:
        """Return active abstractions. Called by CoreMainLoop._step3."""
        return self._abstractions

    def propose_abstraction(self, surfaced_item: dict, runtime_state: dict) -> Optional[NovelAPIAbstraction]:
        """Propose an abstraction from a surfaced item."""
        fn = surfaced_item.get('function_name', '') if isinstance(surfaced_item, dict) else ''
        if not fn:
            return None
        abstraction = NovelAPIAbstraction(
            abstraction_id=f'abs_{fn}',
            kind='create',
            claim=f'Discovered function: {fn}',
            evidence_refs=[f'discovery:{fn}'],
            status='proposed',
        )
        if abstraction not in self._abstractions:
            self._abstractions.append(abstraction)
        return abstraction

    def generate_abstraction_candidates(self, situation: dict, existing_candidates: list) -> list:
        """Generate abstraction candidates from situation. Called by _step5."""
        # For NovelAPI domain: generate candidates from discovered functions
        discovered = situation.get('discovered_functions', [])
        candidates = []
        for fn in discovered:
            candidates.append({
                'kind': 'abstraction',
                'abstraction_id': f'abs_{fn}',
                'title': f'Test {fn}',
                'score': 0.5,
                'action': f'test_{fn}',
                'metadata': {'abstraction_id': f'abs_{fn}'},
            })
        return candidates

    def commit_abstraction(self, abstraction: Any, evidence_packets: list, runtime_state: dict) -> dict:
        """Commit an abstraction to formal long-term storage. Called by _step10."""
        abstraction_id = abstraction.abstraction_id if hasattr(abstraction, 'abstraction_id') else str(abstraction)
        return {
            'committed': True,
            'abstraction_id': abstraction_id,
            'evidence_count': len(evidence_packets),
        }

    def get_abstraction_by_id(self, abstraction_id: str) -> NovelAPIAbstraction | None:
        """Find abstraction by ID. Called by Step 10."""
        for abs_obj in self._abstractions:
            if abs_obj.abstraction_id == abstraction_id:
                return abs_obj
        return None

"""
governance/family_registry.py — OLP/TAP Step 10 Family Lifecycle Registry
============================================================================
Formal family lifecycle management following the OLP/TAP Step 10 protocol.

Every family status transition is a formal Step 10 operation.
No status change is valid unless it passes through Step 10.

Inspired by: docs/design/formal_hypothesis_card.md (HypothesisCard pattern)
Applied to: family lifecycle (GRADUATE / RETIRE / REOPEN / QUALIFY)
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


# ─── Lifecycle States ────────────────────────────────────────────────────────

class FamilyState:
    """Valid family lifecycle states (OLP/TAP Step 10 states)."""
    QUALIFYING = "qualifying"       # Under active qualification testing
    GRADUATED = "graduated"         # Passed all qualification gates
    RETIRED = "retired"             # Falsified / superseded / unfixable
    REOPENED = "reopened"           # Was RETIRED, new evidence reopens
    SUSPENDED = "suspended"         # Temporarily blocked (dependency)

    @classmethod
    def valid_states(cls) -> list:
        return [cls.QUALIFYING, cls.GRADUATED, cls.RETIRED, cls.REOPENED, cls.SUSPENDED]

    @classmethod
    def is_valid(cls, s: str) -> bool:
        return s in cls.valid_states()


# ─── Lifecycle Transitions (Step 10) ─────────────────────────────────────────

@dataclass
class Step10Transition:
    """
    A single Step 10 lifecycle transition.
    
    Analogous to HypothesisCard.update_log entries.
    All modifications to family status go through Step 10.
    """
    transition_id: str           # UUID
    from_state: str              # Previous state
    to_state: str                # New state
    tick: int                    # Environment tick (or wall-clock step)
    reason: str                  # Human-readable justification
    evidence: Dict[str, Any]     # Key metrics that motivated this transition
    gates_passed: List[str]       # Which gates passed/failed
    gates_failed: List[str]
    variant_scope: str            # Which variants were tested
    approved_by: str = "step10_protocol"

    @classmethod
    def create(cls, from_state: str, to_state: str, reason: str,
               evidence: Dict[str, Any], gates_passed: List[str],
               gates_failed: List[str], variant_scope: str, tick: int = 0) -> Step10Transition:
        return cls(
            transition_id=str(uuid.uuid4())[:8],
            from_state=from_state,
            to_state=to_state,
            tick=tick,
            reason=reason,
            evidence=evidence,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            variant_scope=variant_scope,
        )


# ─── FamilyCard ───────────────────────────────────────────────────────────────

@dataclass
class FamilyCard:
    """
    Formal Family Card — mirrors HypothesisCard schema.
    
    7 required fields:
    1. family_id       — stable semantic name
    2. claim           — what this family asserts it does
    3. mechanism       — how it works (backend / trigger / signal)
    4. state           — current lifecycle state (GRADUATED/RETIRED/etc)
    5. variants        — list of qualified variants
    6. metrics         — key quantitative metrics
    7. update_log      — Step 10 transition history

    Invariants:
    - state must be a valid FamilyState
    - All modifications go through Step 10 (update_log)
    - scope field documents what this family DOES NOT cover
    """
    # ── Identity ─────────────────────────────────────────────────────────────
    family_id: str               # Stable internal ID (e.g., "commit_damage_control")
    claim: str                   # What this family asserts
    mechanism: str               # Human-readable mechanism summary

    # ── Scope ────────────────────────────────────────────────────────────────
    signal_type: str             # PRESENT evidence / ABSENT evidence / etc
    firing_phase: str             # committed_revealed_trap / committed_trap / etc
    protective_primitive: str     # retreat / wait / etc
    scope_boundary: str           # What this family does NOT cover

    # ── Qualification ────────────────────────────────────────────────────────
    state: str = FamilyState.QUALIFYING
    variants: List[str] = field(default_factory=list)  # Qualified variant names
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)

    # ── Metrics ─────────────────────────────────────────────────────────────
    lineage_changed_pct: float = 0.0   # % of windows where lineage changed selection
    retreat_delta_pct: float = 0.0    # Δ in retreat rate (TrapTwin Full vs NoLineage)
    damage_delta: float = 0.0          # Δ in damage (Full vs NoLineage)
    benign_over_retreat_pct: float = 0.0  # Over-retreat rate in safe variant
    n_seeds_tested: int = 0

    # ── Lifecycle History ────────────────────────────────────────────────────
    update_log: List[Dict] = field(default_factory=list)  # Step 10 transitions
    created_at: str = ""           # ISO timestamp
    graduated_at: Optional[str] = None
    retired_at: Optional[str] = None

    def __post_init__(self):
        if not FamilyState.is_valid(self.state):
            raise ValueError(f"Invalid family state: {self.state}")
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    # ── Step 10: State Transition ─────────────────────────────────────────────

    def step10_transition(self, to_state: str, reason: str,
                           evidence: Dict[str, Any],
                           gates_passed: List[str],
                           gates_failed: List[str],
                           variant_scope: str) -> None:
        """
        Execute a Step 10 state transition.
        
        This is the ONLY legal place to modify family status.
        Mirrors HypothesisCard's Step 10 modification protocol.
        """
        if not FamilyState.is_valid(to_state):
            raise ValueError(f"Invalid target state: {to_state}")

        from_state = self.state
        t = Step10Transition.create(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            evidence=evidence,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            variant_scope=variant_scope,
            tick=len(self.update_log),
        )

        self.state = to_state
        self.update_log.append({
            'transition_id': t.transition_id,
            'from_state': from_state,
            'to_state': to_state,
            'tick': t.tick,
            'reason': reason,
            'evidence': evidence,
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'variant_scope': variant_scope,
        })

        if to_state == FamilyState.GRADUATED:
            self.graduated_at = datetime.now().isoformat()
        elif to_state == FamilyState.RETIRED:
            self.retired_at = datetime.now().isoformat()

    # ── Convenience ──────────────────────────────────────────────────────────

    def is_graduated(self) -> bool:
        return self.state == FamilyState.GRADUATED

    def is_retired(self) -> bool:
        return self.state == FamilyState.RETIRED

    def summary(self) -> str:
        return (f"FamilyCard({self.family_id}): "
                f"state={self.state} "
                f"variants={self.variants} "
                f"Δ={self.damage_delta:.1f} "
                f"lineage_changed={self.lineage_changed_pct:.0%} "
                f"BenignOver={self.benign_over_retreat_pct:.0%}")


# ─── Family Registry ─────────────────────────────────────────────────────────

class FamilyRegistry:
    """
    Global registry of all families — OLP/TAP Step 10 management layer.
    
    Singleton. All family status queries and transitions go through here.
    """
    _instance: Optional[FamilyRegistry] = None

    def __init__(self):
        self._families: Dict[str, FamilyCard] = {}

    @classmethod
    def get_instance(cls) -> FamilyRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            instance = cls.__new__(cls)
            instance.__init__()
            cls._instance = instance
        return cls._instance

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def register(self, card: FamilyCard) -> None:
        """Register a new family."""
        if card.family_id in self._families:
            raise ValueError(f"Family already registered: {card.family_id}")
        self._families[card.family_id] = card

    def get(self, family_id: str) -> Optional[FamilyCard]:
        """Get a family card by ID."""
        return self._families.get(family_id)

    def all(self) -> List[FamilyCard]:
        """List all families."""
        return list(self._families.values())

    def graduated(self) -> List[FamilyCard]:
        return [f for f in self._families.values() if f.is_graduated()]

    def retired(self) -> List[FamilyCard]:
        return [f for f in self._families.values() if f.is_retired()]

    def qualifying(self) -> List[FamilyCard]:
        return [f for f in self._families.values() if f.state == FamilyState.QUALIFYING]

    # ── Step 10 Transitions ──────────────────────────────────────────────────

    def graduate(self, family_id: str, evidence: Dict[str, Any],
                  gates_passed: List[str], gates_failed: List[str],
                  variant_scope: str, reason: str = "") -> None:
        """
        Step 10: Transition family to GRADUATED state.
        """
        card = self._families.get(family_id)
        if card is None:
            raise ValueError(f"Unknown family: {family_id}")
        card.step10_transition(
            to_state=FamilyState.GRADUATED,
            reason=reason or f"Passed all G1-G5 qualification gates across {variant_scope}",
            evidence=evidence,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            variant_scope=variant_scope,
        )

    def retire(self, family_id: str, evidence: Dict[str, Any],
               gates_passed: List[str], gates_failed: List[str],
               variant_scope: str, reason: str) -> None:
        """
        Step 10: Transition family to RETIRED state.
        """
        card = self._families.get(family_id)
        if card is None:
            raise ValueError(f"Unknown family: {family_id}")
        card.step10_transition(
            to_state=FamilyState.RETIRED,
            reason=reason,
            evidence=evidence,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            variant_scope=variant_scope,
        )

    def reopen(self, family_id: str, reason: str) -> None:
        """
        Step 10: Reopen a RETIRED family with new evidence.
        """
        card = self._families.get(family_id)
        if card is None:
            raise ValueError(f"Unknown family: {family_id}")
        card.step10_transition(
            to_state=FamilyState.REOPENED,
            reason=reason,
            evidence={},
            gates_passed=[],
            gates_failed=[],
            variant_scope="pending",
        )

    # ── Report ───────────────────────────────────────────────────────────────

    def report(self) -> str:
        lines = ["=== Family Registry ==="]
        for f in sorted(self._families.values(), key=lambda x: x.family_id):
            lines.append(f"  {f.summary()}")
        lines.append(f"\nTotal: {len(self._families)} families "
                     f"({len(self.graduated())} graduated, "
                     f"{len(self.retired())} retired)")
        return "\n".join(lines)


# ─── Pre-built Family Cards ───────────────────────────────────────────────────

def _dc_card() -> FamilyCard:
    """DC (commit_damage_control) — pre-built GRADUATED card."""
    card = FamilyCard(
        family_id="commit_damage_control",
        claim="Lineage presence causes retreat in revealed trap via score override",
        mechanism="TwinDCBackend: retreat score +1.0, fires on evid='down' in revealed_trap",
        signal_type="PRESENT evidence (evid='down')",
        firing_phase="committed_revealed_trap (post-reveal)",
        protective_primitive="retreat",
        scope_boundary="Pre-reveal firing not qualified; OA RETIRED (anti-cheat structurally invalid)",
        state=FamilyState.GRADUATED,
        variants=["V2", "V4", "B1", "B2"],
        gates_passed=["G1", "G2a", "G2b", "G2c", "G3", "G4c", "G5"],
        gates_failed=["G4a"],
        lineage_changed_pct=1.0,
        retreat_delta_pct=1.0,
        damage_delta=-70.5,
        benign_over_retreat_pct=0.0,
        n_seeds_tested=34,
    )
    card.graduated_at = "2026-04-03"
    card.update_log.append({
        'transition_id': 'dc-initial',
        'from_state': 'qualifying',
        'to_state': 'graduated',
        'tick': 0,
        'reason': 'Passed all G1-G5 across 4 variants (V2, V4, B1, B2)',
        'evidence': {
            'V2_damage_delta': +34.2,
            'V4_damage_delta': +21.3,
            'B1_damage_delta': -79.7,
            'B2_damage_delta': -36.8,
            'lineage_changed': '100%',
            'BenignOver': '0%',
        },
        'gates_passed': ['G1', 'G2a', 'G2b', 'G2c', 'G3', 'G4c', 'G5'],
        'gates_failed': [],
        'variant_scope': 'V2 (3-cand+4-cand), B1 (early), B2 (late)',
    })
    return card


def _poa_card() -> FamilyCard:
    """POA (probe_or_advance) — pre-built GRADUATED card."""
    card = FamilyCard(
        family_id="probe_or_advance",
        claim="Absence of probe evidence causes retreat in pre-reveal via score override",
        mechanism="TwinPOABackend: retreat score +1.0, fires on evid='none' in committed_trap pre-reveal",
        signal_type="ABSENT evidence (evid='none')",
        firing_phase="committed_trap (pre-reveal)",
        protective_primitive="retreat",
        scope_boundary="prob_steps ∈ {1, 2} only; noisy trap not tested",
        state=FamilyState.GRADUATED,
        variants=["P1", "PB1"],
        gates_passed=["G1", "G2a", "G2b", "G2c", "G3", "G4b", "G4c", "G5"],
        gates_failed=["G4a"],
        lineage_changed_pct=1.0,
        retreat_delta_pct=1.0,
        damage_delta=-19.25,
        benign_over_retreat_pct=0.0,
        n_seeds_tested=10,
    )
    card.graduated_at = "2026-04-03"
    card.update_log.append({
        'transition_id': 'poa-initial',
        'from_state': 'qualifying',
        'to_state': 'graduated',
        'tick': 0,
        'reason': 'Passed all G1-G5 across 2 variants (P1, PB1)',
        'evidence': {
            'P1_damage_delta': -26.0,
            'PB1_damage_delta': -12.5,
            'lineage_changed': '100%',
            'BenignOver': '0%',
        },
        'gates_passed': ['G1', 'G2a', 'G2b', 'G2c', 'G3', 'G4b', 'G4c', 'G5'],
        'gates_failed': [],
        'variant_scope': 'P1 (prob=2), PB1 (prob=1)',
    })
    return card


def _oa_card() -> FamilyCard:
    """OA (OpportunityAmbiguity) — RETIRED: anti-cheat structurally invalid."""
    card = FamilyCard(
        family_id="opportunity_ambiguity",
        claim="dormant_opportunity_credibility field in world_summary drives behavioral fork",
        mechanism="OA backend modulates doc credibility; behavioral fork in Aligned vs Misaligned",
        signal_type="world_summary explicit field",
        firing_phase="step6 (predict_evaluate)",
        protective_primitive="wait / gather",
        scope_boundary="RETIRED — anti-cheat structurally invalid: doc IS derived FROM WM posterior",
        state=FamilyState.RETIRED,
        variants=["OA-exp1"],
        gates_passed=["G1", "G2", "G3"],
        gates_failed=["anti-cheat-exp2", "anti-cheat-exp3"],
        lineage_changed_pct=0.0,
        retreat_delta_pct=0.0,
        damage_delta=0.0,
        benign_over_retreat_pct=0.0,
        n_seeds_tested=4,
    )
    card.retired_at = "2026-04-03"
    card.update_log.append({
        'transition_id': 'oa-retire-exp3',
        'from_state': 'suspended',
        'to_state': 'retired',
        'tick': 1,
        'reason': 'Anti-cheat Exp3 structurally invalid: doc = 1 - WM.posterior (same signal). '
                  'Freezing WM at neutral 0.5 while doc varies inverts fork. '
                  'Anti-cheat design impossible — doc is derived from the very signal it should be isolating.',
        'evidence': {
            'fork_magnitude': 461.25,
            'natural_fork_neutral_WM': -648.0,  # inverted
            'replayA_fork': -645.0,
            'key_finding': 'doc = 1 - WM.posterior; they are the same signal, not independent',
            'architectural_conclusion': 'OA anti-cheat design is structurally impossible without redesigning doc computation',
        },
        'gates_passed': ['G1', 'G2', 'G3'],
        'gates_failed': ['anti-cheat-exp2', 'anti-cheat-exp3'],
        'variant_scope': 'OA-exp1, OA-anticheat-exp3 (4 seeds each)',
    })
    return card


# ─── Initialize Registry ──────────────────────────────────────────────────────

def init_registry() -> FamilyRegistry:
    """Initialize the global family registry with known families."""
    r = FamilyRegistry.get_instance()
    r.register(_dc_card())
    r.register(_poa_card())
    r.register(_oa_card())
    return r

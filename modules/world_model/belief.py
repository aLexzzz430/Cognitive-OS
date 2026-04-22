"""
modules/world_model/belief.py

Stage B1: Belief State Ledger

Belief = explicit representation of hidden or uncertain world variables.

Core principle:
- Beliefs are derived from events, not direct observation
- Beliefs do NOT replace formal object store
- Beliefs are advisory (for governance, recovery, planning)
- All belief mutations go through the event bus

Belief schema:
- belief_id: unique identifier
- variable_name: what this belief is about
- hypothesized_values: possible values with probabilities
- posterior: most likely value
- confidence: how confident we are (0-1)
- uncertainty: entropy-like measure
- evidence_ids: supporting evidence
- last_updated_tick: when last updated

Usage:
- WorldModelOmega subscribes to event bus
- When events arrive, beliefs are updated
- Updated beliefs can influence governance/recovery/planning
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import time
import re


class BeliefStatus(Enum):
    """Lifecycle state of a belief."""
    HYPOTHESIS = "hypothesis"       # Initial, low confidence
    PROBABLE = "probable"           # Growing confidence
    ESTABLISHED = "established"     # High confidence, stable
    CONTRADICTED = "contradicted"   # Evidence against, pending review
    RETIRED = "retired"             # No longer relevant


@dataclass
class BeliefValue:
    """A possible value for a belief variable."""
    value: str                      # e.g., "function_exists", "hidden_precondition"
    probability: float              # 0.0 - 1.0
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class Belief:
    """
    A belief about a hidden or uncertain world variable.
    
    Beliefs are derived from evidence, not direct access.
    They represent the system's best guess about things it cannot observe directly.
    """
    belief_id: str
    variable_name: str              # e.g., "function_completion_order", "hidden_precondition_type"
    hypothesized_values: List[BeliefValue]  # All considered possibilities
    posterior: str                   # Most likely value
    confidence: float               # 0.0 - 1.0 (posterior probability)
    uncertainty: float              # Entropy measure (higher = less sure)
    evidence_ids: List[str] = field(default_factory=list)  # Supporting evidence
    last_updated_tick: int = 0
    last_updated_episode: int = 0
    status: BeliefStatus = BeliefStatus.HYPOTHESIS
    source_hypothesis_id: Optional[str] = None  # Which hypothesis led to this belief
    
    def update_from_evidence(
        self,
        new_evidence: Dict[str, Any],
        confidence_boost: float = 0.0,
    ) -> None:
        """
        Update belief based on new evidence.
        
        This is a simple Bayesian-like update:
        - If evidence supports current posterior, boost confidence
        - If evidence contradicts, reduce confidence
        - If evidence suggests different value, update posterior
        """
        # Check if evidence supports or contradicts
        previous_status = self.status
        evidence_value = new_evidence.get('value', '')
        evidence_strength = new_evidence.get('strength', 0.5)  # 0.0-1.0
        
        # Find matching hypothesized value
        matching = [v for v in self.hypothesized_values if v.value == evidence_value]
        
        if matching:
            # Evidence supports a hypothesized value
            match = matching[0]
            match.evidence_ids.append(new_evidence.get('evidence_id', ''))
            
            # Update confidence (weighted by evidence strength)
            self.confidence = min(1.0, self.confidence + (evidence_strength * confidence_boost))
            
            # Recalculate posterior
            best = max(self.hypothesized_values, key=lambda v: len(v.evidence_ids))
            if best.probability > 0.3:
                self.posterior = best.value

            # Explicit recovery condition for contradicted beliefs:
            # only recover on later supportive rounds with enough strength.
            can_recover = (
                previous_status == BeliefStatus.CONTRADICTED
                and evidence_strength >= 0.6
                and self.confidence >= 0.4
            )
        else:
            # Evidence contradicts all current hypotheses
            self.confidence = max(0.0, self.confidence - evidence_strength * 0.2)
            self.status = BeliefStatus.CONTRADICTED
            can_recover = False
        
        # Update uncertainty (simplified: more evidence = less uncertainty)
        total_evidence = sum(len(v.evidence_ids) for v in self.hypothesized_values)
        self.uncertainty = max(0.0, 1.0 - (total_evidence / 10.0))
        
        # Contradiction path takes precedence for this update round.
        if not matching:
            return

        # If belief is contradicted, keep it contradicted until explicit recovery condition is met.
        if previous_status == BeliefStatus.CONTRADICTED and not can_recover:
            self.status = BeliefStatus.CONTRADICTED
            return

        # Update status based on confidence for support path (or allowed recovery path).
        if self.confidence >= 0.8:
            self.status = BeliefStatus.ESTABLISHED
        elif self.confidence >= 0.4:
            self.status = BeliefStatus.PROBABLE


class BeliefLedger:
    """
    Ledger of all active beliefs.
    
    CoreMainLoop does not write here directly.
    WorldModelOmega updates beliefs based on events.
    """
    
    def __init__(self):
        self._beliefs: Dict[str, Belief] = {}
        self._variable_index: Dict[str, List[str]] = {}  # variable_name -> [belief_ids]
    
    def add_belief(self, belief: Belief) -> None:
        """Add a new belief to the ledger."""
        self._beliefs[belief.belief_id] = belief
        if belief.variable_name not in self._variable_index:
            self._variable_index[belief.variable_name] = []
        self._variable_index[belief.variable_name].append(belief.belief_id)
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Get a belief by ID."""
        return self._beliefs.get(belief_id)
    
    def get_beliefs_for_variable(self, variable_name: str) -> List[Belief]:
        """Get all beliefs about a variable."""
        belief_ids = self._variable_index.get(variable_name, [])
        return [self._beliefs[bid] for bid in belief_ids if bid in self._beliefs]
    
    def get_active_beliefs(self) -> List[Belief]:
        """Get all beliefs that are not retired or contradicted."""
        return [
            b for b in self._beliefs.values()
            if b.status not in (BeliefStatus.RETIRED, BeliefStatus.CONTRADICTED)
        ]
    
    def get_established_beliefs(self) -> List[Belief]:
        """Get beliefs with ESTABLISHED status (high confidence)."""
        return [b for b in self._beliefs.values() if b.status == BeliefStatus.ESTABLISHED]
    
    def update_belief(
        self,
        belief_id: str,
        new_evidence: Dict[str, Any],
        confidence_boost: float = 0.1,
    ) -> bool:
        """
        Update a belief with new evidence.
        
        Returns True if belief was found and updated.
        """
        belief = self._beliefs.get(belief_id)
        if not belief:
            return False
        belief.update_from_evidence(new_evidence, confidence_boost)
        return True
    
    def retire_belief(self, belief_id: str) -> None:
        """Retire a belief (no longer relevant)."""
        belief = self._beliefs.get(belief_id)
        if belief:
            belief.status = BeliefStatus.RETIRED
    
    def belief_count(self) -> int:
        """Total number of beliefs."""
        return len(self._beliefs)
    
    def active_count(self) -> int:
        """Number of active (non-retired) beliefs."""
        return len(self.get_active_beliefs())


class BeliefUpdater:
    """
    Updates beliefs based on events from the event bus.
    
    This is the bridge between CoreMainLoop events and belief state.
    
    Issue 2 fix: Added on_object_created for belief self-bootstrapping.
    """
    
    def __init__(self, ledger: BeliefLedger):
        self._ledger = ledger
        self._mechanism_evidence_tally: Dict[str, Dict[str, int]] = {}

    def _slug(self, value: Any) -> str:
        text = str(value or '').strip().lower()
        text = re.sub(r'[^a-z0-9_]+', '_', text)
        text = re.sub(r'_+', '_', text).strip('_')
        return text or 'unknown'

    def _canonical_variable_name(self, prefix: str, key: Any) -> str:
        prefix_norm = self._slug(prefix)
        if prefix_norm not in {'fn', 'test', 'observation', 'mechanism'}:
            prefix_norm = 'mechanism'
        return f"{prefix_norm}_{self._slug(key)}"

    def _upsert_binary_belief(
        self,
        variable_name: str,
        positive_value: str,
        evidence_id: str,
        confidence: float,
        confidence_boost: float,
    ) -> None:
        existing = self._ledger.get_beliefs_for_variable(variable_name)
        if existing:
            self._ledger.update_belief(
                existing[0].belief_id,
                {'value': positive_value, 'strength': confidence, 'evidence_id': evidence_id},
                confidence_boost=confidence_boost,
            )
            return
        belief = Belief(
            belief_id=f"belief_{variable_name}",
            variable_name=variable_name,
            hypothesized_values=[
                BeliefValue(value=positive_value, probability=max(0.3, confidence), evidence_ids=[evidence_id]),
                BeliefValue(value="unknown", probability=max(0.0, 1.0 - confidence), evidence_ids=[]),
            ],
            posterior=positive_value,
            confidence=max(0.0, min(1.0, confidence)),
            uncertainty=max(0.0, 1.0 - confidence),
            evidence_ids=[evidence_id],
            status=BeliefStatus.HYPOTHESIS,
        )
        self._ledger.add_belief(belief)

    def _upsert_observation_belief(
        self,
        variable_name: str,
        posterior: str,
        confidence: float,
        evidence_id: str,
    ) -> None:
        """Create or update a lightweight observation-derived belief."""
        existing = self._ledger.get_beliefs_for_variable(variable_name)
        clamped_confidence = max(0.0, min(1.0, float(confidence)))
        uncertainty = max(0.0, 1.0 - clamped_confidence)

        if existing:
            belief = existing[0]
            if all(v.value != posterior for v in belief.hypothesized_values):
                belief.hypothesized_values.append(BeliefValue(value=posterior, probability=clamped_confidence, evidence_ids=[]))
            belief.posterior = posterior
            belief.confidence = max(belief.confidence, clamped_confidence)
            belief.uncertainty = min(belief.uncertainty, uncertainty)
            belief.evidence_ids.append(evidence_id)
            if clamped_confidence >= 0.8:
                belief.status = BeliefStatus.ESTABLISHED
            elif clamped_confidence >= 0.55:
                belief.status = BeliefStatus.PROBABLE
            else:
                belief.status = BeliefStatus.HYPOTHESIS
            return

        belief = Belief(
            belief_id=f"belief_{variable_name}",
            variable_name=variable_name,
            hypothesized_values=[
                BeliefValue(value=posterior, probability=clamped_confidence, evidence_ids=[evidence_id]),
                BeliefValue(value="unknown", probability=max(0.0, 1.0 - clamped_confidence), evidence_ids=[]),
            ],
            posterior=posterior,
            confidence=clamped_confidence,
            uncertainty=uncertainty,
            evidence_ids=[evidence_id],
            status=(
                BeliefStatus.ESTABLISHED
                if clamped_confidence >= 0.8
                else BeliefStatus.PROBABLE
                if clamped_confidence >= 0.55
                else BeliefStatus.HYPOTHESIS
            ),
        )
        self._ledger.add_belief(belief)

    def _recovery_variable_name(self, recovery_type: Any) -> str:
        return self._canonical_variable_name('mechanism', f"recovery_{self._slug(recovery_type)}")

    def _mechanism_prior_variable_name(self, function_name: Any) -> str:
        return self._canonical_variable_name('mechanism', f"prior_{self._slug(function_name)}")

    def on_observation_received(self, event_data: Dict[str, Any]) -> None:
        """
        Use perception summaries to maintain lightweight world-model beliefs.

        This keeps perception from being an isolated demo by exposing
        observation structure as advisory beliefs for decision-making.
        """
        perception = event_data.get('perception', {})
        if not isinstance(perception, dict) or not perception:
            return

        coordinate_type = perception.get('coordinate_type', 'unknown')
        coordinate_confidence = float(perception.get('coordinate_confidence', 0.0) or 0.0)
        if coordinate_type and coordinate_type != 'unknown' and coordinate_confidence >= 0.4:
            self._upsert_observation_belief(
                variable_name='observation_coordinate_type',
                posterior=str(coordinate_type),
                confidence=coordinate_confidence,
                evidence_id=f"ev_obs_coord_{coordinate_type}",
            )

        camera_motion_score = float(perception.get('camera_motion_score', 0.0) or 0.0)
        if camera_motion_score > 0.0:
            self._upsert_observation_belief(
                variable_name='observation_camera_motion',
                posterior='high_motion' if camera_motion_score >= 0.6 else 'low_motion',
                confidence=max(0.35, camera_motion_score),
                evidence_id='ev_obs_camera_motion',
            )

        dynamic_entities = perception.get('dynamic_entities', [])
        dynamic_count = len(dynamic_entities) if isinstance(dynamic_entities, list) else 0
        if dynamic_count > 0:
            dynamic_confidence = min(0.95, 0.45 + dynamic_count * 0.15)
            self._upsert_observation_belief(
                variable_name='observation_dynamic_entities',
                posterior='present',
                confidence=dynamic_confidence,
                evidence_id='ev_obs_dynamic_entities',
            )
    
    def on_object_created(self, event_data: Dict[str, Any]) -> None:
        """
        Issue 2 fix: Create belief when a new object is committed.
        
        This is the secondary path for self-bootstrapping the belief layer.
        """
        object_id = event_data.get('object_id')
        if not object_id:
            return
        
        # Create a belief about this object
        variable_name = self._canonical_variable_name('mechanism', f'object_exists_{object_id[:8]}')
        # Check if belief already exists
        existing = self._ledger.get_beliefs_for_variable(variable_name)
        if existing:
            return  # Already have a belief for this object
        
        belief = Belief(
            belief_id=f"belief_{variable_name}",
            variable_name=variable_name,
            hypothesized_values=[
                BeliefValue(value="exists", probability=0.5, evidence_ids=[object_id]),
                BeliefValue(value="unknown", probability=0.5, evidence_ids=[]),
            ],
            posterior="exists",
            confidence=0.5,
            uncertainty=0.7,
            evidence_ids=[object_id],
            status=BeliefStatus.HYPOTHESIS,
        )
        self._ledger.add_belief(belief)
    
    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """
        When action is executed, update related beliefs.
        
        Example: belief that "function X exists" gets confirmed when action succeeds.
        """
        fn_name = event_data.get('function_name', '')
        success = bool(event_data.get('success', True))
        if fn_name and fn_name != 'wait':
            variable_name = self._canonical_variable_name('fn', fn_name)
            if success:
                self._upsert_binary_belief(
                    variable_name=variable_name,
                    positive_value='function_exists',
                    evidence_id=f'ev_action_{self._slug(fn_name)}',
                    confidence=0.65,
                    confidence_boost=0.15,
                )
            else:
                existing = self._ledger.get_beliefs_for_variable(variable_name)
                for belief in existing:
                    self._ledger.update_belief(
                        belief.belief_id,
                        {'value': '', 'strength': 0.6, 'evidence_id': f'ev_action_fail_{self._slug(fn_name)}'},
                        confidence_boost=-0.15,
                    )
    
    def on_hypothesis_updated(self, event_data: Dict[str, Any]) -> None:
        """
        When hypothesis is confirmed/falsified, update related beliefs.
        """
        hypothesis_id = event_data.get('hypothesis_id', '')
        verdict = event_data.get('verdict', '')  # 'confirmed' or 'falsified'
        
        # Find belief sourced from this hypothesis
        for belief in self._ledger.get_active_beliefs():
            if belief.source_hypothesis_id == hypothesis_id:
                if verdict == 'confirmed':
                    self._ledger.update_belief(
                        belief.belief_id,
                        {'value': belief.posterior, 'strength': 0.8, 'evidence_id': f'ev_hyp_{hypothesis_id}'},
                        confidence_boost=0.2,
                    )
                else:
                    # Falsified - reduce confidence
                    self._ledger.update_belief(
                        belief.belief_id,
                        {'value': '', 'strength': 0.6, 'evidence_id': f'ev_hyp_false_{hypothesis_id}'},
                        confidence_boost=-0.3,
                    )
    
    def on_test_executed(self, event_data: Dict[str, Any]) -> None:
        """
        When test is executed, update belief about tested hypothesis.
        """
        test_result = event_data.get('test_result')
        test_fn = event_data.get('test_function', '')
        variable_name = self._canonical_variable_name('test', test_fn)
        
        # Tests that pass (result=True) confirm the targeted hypothesis
        if test_result is True and test_fn:
            # Find beliefs related to this test
            self._upsert_binary_belief(
                variable_name=variable_name,
                positive_value='test_passes',
                evidence_id=f'ev_test_pass_{self._slug(test_fn)}',
                confidence=0.75,
                confidence_boost=0.25,
            )
        elif test_result is False:
            related = self._ledger.get_beliefs_for_variable(variable_name)
            for belief in related:
                self._ledger.update_belief(
                    belief.belief_id,
                    {'value': '', 'strength': 0.7, 'evidence_id': f'ev_test_fail_{self._slug(test_fn)}'},
                    confidence_boost=-0.2,
                )

    def on_recovery_executed(self, event_data: Dict[str, Any]) -> None:
        """Track recovery attempts as beliefs that can later be reinforced/refuted."""
        recovery_type = event_data.get('recovery_type', 'unknown')
        variable_name = self._recovery_variable_name(recovery_type)
        evidence_id = str(event_data.get('recovery_task_id') or f"ev_recovery_exec_{self._slug(recovery_type)}")
        self._upsert_observation_belief(
            variable_name=variable_name,
            posterior='attempted',
            confidence=0.45,
            evidence_id=evidence_id,
        )

    def on_recovery_outcome_observed(self, event_data: Dict[str, Any]) -> None:
        """Map recovery success/failure to confidence updates on recovery and function beliefs."""
        recovery_type = event_data.get('recovery_type', 'unknown')
        recovery_success = bool(event_data.get('success', False))
        target_fn = event_data.get('function_name', '')
        variable_name = self._recovery_variable_name(recovery_type)
        evidence_id = str(event_data.get('evidence_id') or f"ev_recovery_outcome_{self._slug(recovery_type)}")
        if recovery_success:
            self._upsert_observation_belief(
                variable_name=variable_name,
                posterior='successful',
                confidence=0.82,
                evidence_id=evidence_id,
            )
        else:
            existing = self._ledger.get_beliefs_for_variable(variable_name)
            if existing:
                self._ledger.update_belief(
                    existing[0].belief_id,
                    {'value': '', 'strength': 0.75, 'evidence_id': evidence_id},
                    confidence_boost=-0.25,
                )
            else:
                self._upsert_observation_belief(
                    variable_name=variable_name,
                    posterior='failed',
                    confidence=0.3,
                    evidence_id=evidence_id,
                )

        if target_fn and target_fn != 'wait':
            fn_variable = self._canonical_variable_name('fn', target_fn)
            if recovery_success:
                self._upsert_binary_belief(
                    variable_name=fn_variable,
                    positive_value='function_exists',
                    evidence_id=f'ev_recovery_fn_{self._slug(target_fn)}',
                    confidence=0.7,
                    confidence_boost=0.15,
                )
            else:
                for belief in self._ledger.get_beliefs_for_variable(fn_variable):
                    self._ledger.update_belief(
                        belief.belief_id,
                        {'value': '', 'strength': 0.65, 'evidence_id': f'ev_recovery_fn_fail_{self._slug(target_fn)}'},
                        confidence_boost=-0.15,
                    )

    def on_mechanism_evidence_added(self, event_data: Dict[str, Any]) -> None:
        """
        Accumulate support/refute evidence for mechanism priors so priors can feed scoring.
        """
        fn_name = event_data.get('target_function') or event_data.get('function_name')
        if not fn_name:
            return
        mechanism_id = str(event_data.get('mechanism_id') or 'unknown')
        supports = bool(event_data.get('supports', True))
        key = f"{self._slug(fn_name)}::{mechanism_id}"
        tally = self._mechanism_evidence_tally.setdefault(key, {'support': 0, 'refute': 0})
        if supports:
            tally['support'] += 1
        else:
            tally['refute'] += 1
        total = tally['support'] + tally['refute']
        if total <= 0:
            return
        support_ratio = tally['support'] / float(total)
        variable_name = self._mechanism_prior_variable_name(fn_name)
        posterior = 'supported_transition' if support_ratio >= 0.5 else 'refuted_transition'
        confidence = max(0.35, min(0.95, abs(support_ratio - 0.5) * 2.0))
        self._upsert_observation_belief(
            variable_name=variable_name,
            posterior=posterior,
            confidence=confidence,
            evidence_id=f"ev_mech_{mechanism_id}_{tally['support']}_{tally['refute']}",
        )

    def on_prediction_miss(self, event_data: Dict[str, Any]) -> None:
        """
        Feed prediction misses back into mechanism priors as negative evidence.

        This keeps prediction calibration connected to world-model belief updates.
        """
        fn_name = str(event_data.get('function_name', '') or '').strip()
        if not fn_name or fn_name == 'wait':
            return
        error = event_data.get('prediction_error', {}) if isinstance(event_data.get('prediction_error'), dict) else {}
        total_error = float(error.get('total_error', 0.0) or 0.0)
        if total_error <= 0.0:
            return
        variable_name = self._mechanism_prior_variable_name(fn_name)
        penalty_confidence = max(0.3, min(0.9, total_error))
        evidence_id = str(
            event_data.get('action_id')
            or f"ev_pred_miss_{self._slug(fn_name)}_{int(total_error * 100)}"
        )
        existing = self._ledger.get_beliefs_for_variable(variable_name)
        if existing:
            self._ledger.update_belief(
                existing[0].belief_id,
                {'value': '', 'strength': penalty_confidence, 'evidence_id': evidence_id},
                confidence_boost=-min(0.35, total_error * 0.4),
            )
            return
        self._upsert_observation_belief(
            variable_name=variable_name,
            posterior='refuted_transition',
            confidence=max(0.25, 1.0 - penalty_confidence),
            evidence_id=evidence_id,
        )
    
    def create_belief_from_hypothesis(
        self,
        hypothesis_id: str,
        variable_name: str,
        proposed_value: str,
        initial_confidence: float = 0.3,
    ) -> Belief:
        """
        Create a new belief derived from a hypothesis.
        
        This links hypotheses to the belief system.
        """
        belief = Belief(
            belief_id=f"belief_{variable_name}_{hypothesis_id[:8]}",
            variable_name=variable_name,
            hypothesized_values=[
                BeliefValue(value=proposed_value, probability=initial_confidence, evidence_ids=[]),
                BeliefValue(value="unknown", probability=1.0 - initial_confidence, evidence_ids=[]),
            ],
            posterior=proposed_value,
            confidence=initial_confidence,
            uncertainty=0.8,
            evidence_ids=[],
            status=BeliefStatus.HYPOTHESIS,
            source_hypothesis_id=hypothesis_id,
        )
        self._ledger.add_belief(belief)
        return belief

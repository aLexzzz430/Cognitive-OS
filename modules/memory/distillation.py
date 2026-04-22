"""
modules/memory/distillation.py

Phase 6: Distillation As Capability

Distillation = verified behavior change from stored memory.

Key principle:
"stored" and "distilled" are NOT the same thing.
- stored: in object store, may or may not influence behavior
- distilled: PROVEN to alter later behavior measurably

Rules:
- Distillation requires evidence of policy change
- Teacher-absent or teacher-reduced evidence is required
- Only verified assets can be promoted to distilled_asset
- distiller does not bypass formal validation

Contract:
- A memory object becomes "distilled" when it demonstrably changes behavior
- This is verified through consumption tracking + post-teacher evidence
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import time

from modules.memory.schema import MemoryType, RetrievalTag
from modules.memory.promotion_rules import MemoryPromotionRules


class DistillationState(Enum):
    """States for distillation pipeline."""
    STORED = "stored"           # In object store, not yet verified
    CANDIDATE = "candidate"     # Has consumption evidence
    COMPILED = "compiled"       # Policy change evidence exists
    DISTILLED = "distilled"     # VERIFIED to change behavior


@dataclass
class DistillationEvidence:
    """Evidence of behavior change from a memory object."""
    object_id: str
    memory_type: str
    
    # When was it consumed?
    consumption_episodes: List[int] = field(default_factory=list)
    consumption_ticks: List[int] = field(default_factory=list)
    
    # Was it beneficial?
    beneficial_consumptions: int = 0
    total_consumptions: int = 0
    
    # Policy change tracking
    policy_change_detected: bool = False
    policy_change_episodes: List[int] = field(default_factory=list)
    
    # Teacher presence at consumption
    teacher_present_at_consumption: List[bool] = field(default_factory=list)
    post_teacher_consumptions: int = 0
    
    # Timestamps
    first_consumed_at: Optional[str] = None
    last_consumed_at: Optional[str] = None
    distilled_at: Optional[str] = None
    
    @property
    def benefit_ratio(self) -> float:
        """Ratio of beneficial consumptions."""
        if self.total_consumptions == 0:
            return 0.0
        return self.beneficial_consumptions / self.total_consumptions
    
    @property
    def post_teacher_ratio(self) -> float:
        """Ratio of consumptions that occurred after teacher exit."""
        if self.total_consumptions == 0:
            return 0.0
        return self.post_teacher_consumptions / self.total_consumptions
    
    def to_dict(self) -> dict:
        """Serialize distillation evidence to dict."""
        return {
            'object_id': self.object_id,
            'memory_type': self.memory_type,
            'consumption_episodes': self.consumption_episodes,
            'consumption_ticks': self.consumption_ticks,
            'beneficial_consumptions': self.beneficial_consumptions,
            'total_consumptions': self.total_consumptions,
            'policy_change_detected': self.policy_change_detected,
            'policy_change_episodes': self.policy_change_episodes,
            'post_teacher_consumptions': self.post_teacher_consumptions,
            'benefit_ratio': self.benefit_ratio,
            'post_teacher_ratio': self.post_teacher_ratio,
            'first_consumed_at': self.first_consumed_at,
            'last_consumed_at': self.last_consumed_at,
            'distilled_at': self.distilled_at,
        }


@dataclass
class DistillationReport:
    """Report of distillation verification."""
    distilled_count: int = 0
    compiled_count: int = 0
    candidate_count: int = 0
    stored_count: int = 0
    new_distilled: List[str] = field(default_factory=list)  # object_ids
    new_compiled: List[str] = field(default_factory=list)
    failed_candidates: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    episode: int = 0
    
    def to_dict(self) -> dict:
        return {
            'distilled_count': self.distilled_count,
            'compiled_count': self.compiled_count,
            'candidate_count': self.candidate_count,
            'stored_count': self.stored_count,
            'new_distilled': self.new_distilled,
            'new_compiled': self.new_compiled,
            'failed_candidates': self.failed_candidates,
            'timestamp': self.timestamp,
            'episode': self.episode,
        }


# Thresholds for distillation
MIN_CONSUMPTIONS_FOR_CANDIDATE = 2
MIN_BENEFIT_RATIO_FOR_COMPILE = 0.5
MIN_POST_TEACHER_RATIO_FOR_DISTILL = 0.3
MIN_POST_TEACHER_CONSUMPTIONS = 2


class DistillationTracker:
    """
    Tracks which memory objects have demonstrable behavior change.
    
    Key insight:
    - A memory is "distilled" when it provably changes policy
    - This requires evidence of post-teacher consumption + benefit
    
    Integration:
    - Consume events come from ObjectStore.record_consumption()
    - Policy change detection requires comparing behavior before/after consumption
    - distillation.py wraps object store with distillation semantics
    """
    
    def __init__(self, object_store, teacher_exit_episode: int = 5):
        self._store = object_store
        self._teacher_exit_episode = teacher_exit_episode
        self._promotion_rules = MemoryPromotionRules(object_store)
        
        # Track distillation evidence per object
        self._evidence: Dict[str, DistillationEvidence] = {}
        
        # Report cache
        self._last_report: Optional[DistillationReport] = None
    
    def record_consumption(
        self,
        object_id: str,
        tick: int,
        episode: int,
        was_beneficial: bool,
        teacher_present: bool,
        policy_changed: bool = False,
    ) -> None:
        """
        Record that an object was consumed.
        
        This updates distillation evidence for the object.
        """
        # Get or create evidence record
        if object_id not in self._evidence:
            obj = self._store.get(object_id)
            memory_type = obj.get('memory_type', 'unknown') if obj else 'unknown'
            self._evidence[object_id] = DistillationEvidence(
                object_id=object_id,
                memory_type=memory_type,
            )
        
        ev = self._evidence[object_id]
        
        # Record consumption
        ev.consumption_episodes.append(episode)
        ev.consumption_ticks.append(tick)
        ev.total_consumptions += 1
        
        if was_beneficial:
            ev.beneficial_consumptions += 1
        
        # Track teacher presence
        ev.teacher_present_at_consumption.append(teacher_present)
        if not teacher_present or episode > self._teacher_exit_episode:
            ev.post_teacher_consumptions += 1
        
        # Track policy change
        if policy_changed:
            ev.policy_change_detected = True
            ev.policy_change_episodes.append(episode)
        
        # Update timestamps
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if ev.first_consumed_at is None:
            ev.first_consumed_at = now
        ev.last_consumed_at = now
        
        # Update object store consumption record
        self._store.record_consumption(object_id, tick, episode, was_beneficial)
    
    def evaluate_distillation(self, current_episode: int) -> DistillationReport:
        """
        Evaluate distillation status of all tracked objects.
        
        Returns DistillationReport with counts and changes.
        """
        report = DistillationReport(episode=current_episode)
        
        for obj_id, ev in self._evidence.items():
            state = self._evaluate_single(ev, current_episode)
            
            if state == DistillationState.DISTILLED:
                report.distilled_count += 1
                if self._maybe_promote_to_distilled(obj_id):
                    report.new_distilled.append(obj_id)
            elif state == DistillationState.COMPILED:
                report.compiled_count += 1
                if self._maybe_promote_to_compiled(obj_id):
                    report.new_compiled.append(obj_id)
            elif state == DistillationState.CANDIDATE:
                report.candidate_count += 1
            else:
                report.stored_count += 1
        
        self._last_report = report
        return report
    
    def _evaluate_single(self, ev: DistillationEvidence, current_episode: int) -> DistillationState:
        """
        Evaluate distillation state for a single object.
        
        State machine:
        STORED -> CANDIDATE: has consumption evidence
        CANDIDATE -> COMPILED: policy change detected + enough post-teacher benefit
        COMPILED -> DISTILLED: verified post-teacher behavior change
        """
        from modules.governance.object_store import AssetStatus
        
        obj = self._store.get(ev.object_id)
        if not obj:
            return DistillationState.STORED
        
        current_asset_status = obj.get('asset_status', AssetStatus.NEW_ASSET.value)
        
        # STORED: No consumption yet
        if ev.total_consumptions == 0:
            return DistillationState.STORED
        
        # CANDIDATE: Has some consumption
        if ev.total_consumptions < MIN_CONSUMPTIONS_FOR_CANDIDATE:
            return DistillationState.CANDIDATE
        
        # Check if can move to COMPILED
        if current_asset_status in (AssetStatus.NEW_ASSET, AssetStatus.LIVE_ASSET):
            # Need policy change evidence + minimum benefit
            if ev.policy_change_detected and ev.benefit_ratio >= MIN_BENEFIT_RATIO_FOR_COMPILE:
                # But still need post-teacher consumption
                if ev.post_teacher_ratio >= MIN_POST_TEACHER_RATIO_FOR_DISTILL:
                    return DistillationState.DISTILLED
                return DistillationState.COMPILED
        
        # Check if can move to DISTILLED
        if current_asset_status in (AssetStatus.COMPILED_ASSET, AssetStatus.REUSABLE_ASSET):
            if (ev.post_teacher_consumptions >= MIN_POST_TEACHER_CONSUMPTIONS and
                ev.benefit_ratio >= MIN_BENEFIT_RATIO_FOR_COMPILE and
                ev.policy_change_detected):
                return DistillationState.DISTILLED
            return DistillationState.COMPILED
        
        return DistillationState.STORED
    
    def _maybe_promote_to_compiled(self, object_id: str) -> bool:
        """Try to promote object to compiled status in object store."""
        from modules.governance.object_store import AssetStatus
        
        obj = self._store.get(object_id)
        if not obj:
            return False
        
        current = AssetStatus(obj.get('asset_status', AssetStatus.NEW_ASSET.value))
        
        # Can only promote from REUSABLE or lower
        if current == AssetStatus.REUSABLE_ASSET:
            obj['asset_status'] = AssetStatus.COMPILED_ASSET.value
            return True
        
        return False
    
    def _maybe_promote_to_distilled(self, object_id: str) -> bool:
        """Try to promote object to distilled status in object store."""
        return self._store.promote_to_distilled(object_id)
    
    def get_distillation_evidence(self, object_id: str) -> Optional[DistillationEvidence]:
        """Get distillation evidence for an object."""
        return self._evidence.get(object_id)
    
    def get_distilled_objects(self) -> List[dict]:
        """Get all objects that are distilled."""
        from modules.governance.object_store import AssetStatus
        
        objs = []
        for obj in self._store.retrieve():
            if obj.get('asset_status') == AssetStatus.DISTILLED_ASSET.value:
                ev = self._evidence.get(obj['object_id'])
                objs.append({
                    **obj,
                    'distillation_evidence': ev.to_dict() if ev else None,
                })
        return objs
    
    def get_summary(self) -> dict:
        """Get distillation summary."""
        from modules.governance.object_store import AssetStatus
        
        total = len(self._evidence)
        by_state = {s.value: 0 for s in DistillationState}
        
        for ev in self._evidence.values():
            obj = self._store.get(ev.object_id)
            if obj:
                state = self._evaluate_single(ev, 0)
                by_state[state.value] += 1
        
        distilled_objs = self._store.get_objects_by_status(AssetStatus.DISTILLED_ASSET)
        
        return {
            'total_tracked': total,
            'by_state': by_state,
            'distilled_in_store': len(distilled_objs),
            'skill_compilation_candidates': len(self.iter_skill_compilation_candidates()),
            'teacher_exit_episode': self._teacher_exit_episode,
        }

    def iter_skill_compilation_candidates(self) -> List[Dict[str, Any]]:
        """
        Surface promotable skill/transfer objects without creating a second write path.
        """
        candidates: List[Dict[str, Any]] = []
        for obj in self._store.retrieve(sort_by='confidence', limit=200):
            skill_candidate = self._promotion_rules.propose_skill_from_object(obj)
            if skill_candidate:
                candidates.append(skill_candidate)
            transfer_candidate = self._promotion_rules.propose_transfer_from_object(obj)
            if transfer_candidate:
                candidates.append(transfer_candidate)
        return candidates


def verify_distillation(
    object_store,
    distillation_tracker: DistillationTracker,
    current_episode: int,
) -> DistillationReport:
    """
    Verify distillation status of all memory objects.
    
    Call at episode end or at graduation boundaries.
    """
    return distillation_tracker.evaluate_distillation(current_episode)


def check_policy_change(
    before_state: Dict,
    after_state: Dict,
) -> bool:
    """
    Check if policy changed between two states.
    
    Simple heuristic:
    - Different selected actions
    - Different retrieval results used
    - Different confidence adjustments
    
    Returns True if policy change detected.
    """
    # Check if action selection changed
    before_action = before_state.get('selected_action')
    after_action = after_state.get('selected_action')
    if before_action != after_action:
        return True
    
    # Check if memory retrieval influenced decision
    before_memory = before_state.get('memory_retrieved', [])
    after_memory = after_state.get('memory_retrieved', [])
    if set(before_memory) != set(after_memory):
        return True
    
    # Check if confidence was adjusted based on memory
    before_conf = before_state.get('confidence', 0.5)
    after_conf = after_state.get('confidence', 0.5)
    if abs(before_conf - after_conf) > 0.2:
        return True
    
    return False

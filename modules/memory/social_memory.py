"""
modules/memory/social_memory.py

Phase 7: Minimal Social Memory

Minimal social-memory slice without building a large side subsystem.

Goals:
- Log teacher and collaborator interactions into raw event log
- Track trust/relevance for collaborators
- Enable preserving collaborator-relevant memory

Non-goals:
- Full user modeling platform
- Complex relationship graphs
- Second control chain

Rules:
- Social memory goes through same formal path as other memory
- Only minimal relationship_model type when genuinely needed
- Raw event log is the audit trail for social interactions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import time

from modules.memory.schema import MemoryType, MemoryLayer, RetrievalTag


class TrustLevel(Enum):
    """Trust level for collaborators."""
    UNKNOWN = "unknown"
    NEW = "new"
    TRUSTED = "trusted"
    VERIFIED = "verified"
    UNTRUSTED = "untrusted"


class InteractionType(Enum):
    """Types of social interactions."""
    TEACHER_GUIDANCE = "teacher_guidance"
    TEACHER_INTERVENTION = "teacher_intervention"
    TEACHER_FEEDBACK = "teacher_feedback"
    COLLABORATOR_QUERY = "collaborator_query"
    COLLABORATOR_SHARE = "collaborator_share"
    TRUST_UPDATE = "trust_update"


@dataclass
class RelationshipModel:
    """
    Minimal model of a collaborator/teacher relationship.
    
    Stored as a memory object with memory_type=relationship_model.
    """
    entity_id: str  # Who this relationship is with
    entity_type: str  # 'teacher' | 'collaborator' | 'user'
    
    # Trust tracking
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    trust_score: float = 0.5  # 0.0 to 1.0
    
    # Interaction history (summary)
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    
    # Last interaction
    last_interaction_at: Optional[str] = None
    last_interaction_type: Optional[str] = None
    
    # Relevance to current goals
    relevant_goals: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    
    # Flags
    teacher_exit_reported: bool = False
    exit_episode: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'trust_level': self.trust_level.value,
            'trust_score': self.trust_score,
            'total_interactions': self.total_interactions,
            'successful_interactions': self.successful_interactions,
            'failed_interactions': self.failed_interactions,
            'last_interaction_at': self.last_interaction_at,
            'last_interaction_type': self.last_interaction_type,
            'relevant_goals': self.relevant_goals,
            'expertise_areas': self.expertise_areas,
            'teacher_exit_reported': self.teacher_exit_reported,
            'exit_episode': self.exit_episode,
        }
    
    @property
    def success_ratio(self) -> float:
        """Ratio of successful interactions."""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions


@dataclass
class SocialMemoryReport:
    """Report of social memory operations."""
    relationships_tracked: int = 0
    new_relationships: int = 0
    trust_updates: int = 0
    interactions_logged: int = 0
    teacher_exit_detected: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    episode: int = 0
    
    def to_dict(self) -> dict:
        return {
            'relationships_tracked': self.relationships_tracked,
            'new_relationships': self.new_relationships,
            'trust_updates': self.trust_updates,
            'interactions_logged': self.interactions_logged,
            'teacher_exit_detected': self.teacher_exit_detected,
            'timestamp': self.timestamp,
            'episode': self.episode,
        }


# Trust adjustment rates
TRUST_INCREMENT_SUCCESS = 0.05
TRUST_DECREMENT_FAILURE = 0.10
TRUST_DECREMENT_TIMEOUT = 0.05
TRUST_THRESHOLD_TRUSTED = 0.7
TRUST_THRESHOLD_VERIFIED = 0.9


class SocialMemoryTracker:
    """
    Minimal social memory tracking.
    
    Tracks:
    - Teacher interactions (exit detection, guidance quality)
    - Collaborator relationships (trust, relevance)
    
    Integration:
    - Teacher interactions go to event log (already exists)
    - relationship_model objects go to object store through normal path
    - SocialMemoryTracker wraps tracking + object store updates
    """
    
    def __init__(self, object_store, event_log=None, teacher_entity_id: str = "teacher"):
        self._store = object_store
        self._event_log = event_log
        self._teacher_entity_id = teacher_entity_id
        
        # In-memory relationship models (keyed by entity_id)
        self._relationships: Dict[str, RelationshipModel] = {}
        
        # Track teacher exit
        self._teacher_exit_episode: Optional[int] = None
        self._teacher_exit_reported: bool = False
    
    def record_interaction(
        self,
        entity_id: str,
        entity_type: str,
        interaction_type: InteractionType,
        was_successful: bool,
        guidance_content: Optional[Dict] = None,
        current_episode: int = 0,
    ) -> None:
        """
        Record a social interaction.
        
        - Logs to event log (raw history)
        - Updates relationship model
        - Updates trust score
        """
        # Get or create relationship model
        if entity_id not in self._relationships:
            self._relationships[entity_id] = RelationshipModel(
                entity_id=entity_id,
                entity_type=entity_type,
            )
        
        rel = self._relationships[entity_id]
        
        # Update interaction stats
        rel.total_interactions += 1
        if was_successful:
            rel.successful_interactions += 1
        else:
            rel.failed_interactions += 1
        
        # Update trust score
        trust_delta = TRUST_INCREMENT_SUCCESS if was_successful else -TRUST_DECREMENT_FAILURE
        rel.trust_score = max(0.0, min(1.0, rel.trust_score + trust_delta))
        
        # Update trust level
        if rel.trust_score >= TRUST_THRESHOLD_VERIFIED:
            rel.trust_level = TrustLevel.VERIFIED
        elif rel.trust_score >= TRUST_THRESHOLD_TRUSTED:
            rel.trust_level = TrustLevel.TRUSTED
        elif rel.trust_score < 0.3:
            rel.trust_level = TrustLevel.UNTRUSTED
        elif rel.total_interactions > 0:
            rel.trust_level = TrustLevel.NEW
        
        # Update last interaction
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        rel.last_interaction_at = now
        rel.last_interaction_type = interaction_type.value
        
        # Special: detect teacher exit
        if entity_type == 'teacher' and not was_successful:
            # Teacher stopped providing guidance = exit detection
            if self._teacher_exit_episode is None:
                self._teacher_exit_episode = current_episode
                rel.teacher_exit_reported = True
                rel.exit_episode = current_episode
        
        # Log to event log if available
        if self._event_log is not None:
            self._event_log.append({
                'event_type': 'social_interaction',
                'episode': current_episode,
                'tick': 0,
                'data': {
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'interaction_type': interaction_type.value,
                    'was_successful': was_successful,
                    'trust_score_after': rel.trust_score,
                    'guidance_content': guidance_content,
                },
                'source_module': 'social_memory',
            })
    
    def record_teacher_guidance(
        self,
        guidance_type: str,
        content: Dict,
        was_followed: bool,
        was_beneficial: bool,
        current_episode: int,
    ) -> None:
        """
        Record teacher guidance specifically.
        
        Maps to teacher_interaction event in raw log.
        """
        # Log to event log
        if self._event_log is not None:
            self._event_log.append({
                'event_type': 'teacher_guidance',
                'episode': current_episode,
                'tick': 0,
                'data': {
                    'guidance_type': guidance_type,
                    'content': content,
                    'was_followed': was_followed,
                    'was_beneficial': was_beneficial,
                },
                'source_module': 'teacher',
            })
        
        # Update teacher relationship
        self.record_interaction(
            entity_id=self._teacher_entity_id,
            entity_type='teacher',
            interaction_type=InteractionType.TEACHER_GUIDANCE,
            was_successful=was_beneficial,
            guidance_content=content,
            current_episode=current_episode,
        )
    
    def detect_teacher_exit(self, current_episode: int, grace_period: int = 2) -> bool:
        """
        Detect if teacher has exited.
        
        Returns True if teacher exit is confirmed.
        """
        if self._teacher_exit_reported:
            return True
        
        if self._teacher_exit_episode is not None:
            if current_episode > self._teacher_exit_episode + grace_period:
                self._teacher_exit_reported = True
                return True
        
        return False
    
    def get_relationship(self, entity_id: str) -> Optional[RelationshipModel]:
        """Get relationship model for an entity."""
        return self._relationships.get(entity_id)
    
    def get_teacher_relationship(self) -> Optional[RelationshipModel]:
        """Get teacher relationship model."""
        return self._relationships.get(self._teacher_entity_id)
    
    def get_all_relationships(self) -> List[RelationshipModel]:
        """Get all tracked relationships."""
        return list(self._relationships.values())
    
    def get_trusted_collaborators(self) -> List[RelationshipModel]:
        """Get collaborators with trust level >= TRUSTED."""
        return [
            rel for rel in self._relationships.values()
            if rel.entity_type != 'teacher'
            and rel.trust_level in (TrustLevel.TRUSTED, TrustLevel.VERIFIED)
        ]
    
    def create_relationship_proposal(self, entity_id: str) -> Optional[Dict]:
        """
        Create a relationship_model proposal for object store.
        
        Only creates formal memory if relationship is meaningful.
        """
        rel = self._relationships.get(entity_id)
        if not rel:
            return None
        
        # Only create formal memory if relationship is established
        if rel.total_interactions < 3:
            return None
        
        proposal = {
            'content': rel.to_dict(),
            'memory_type': MemoryType.RELATIONSHIP_MODEL.value,
            'memory_layer': MemoryLayer.SOCIAL.value,
            'confidence': rel.trust_score,
            'retrieval_tags': [
                RetrievalTag.SOCIAL_MEMORY.value,
                f"entity:{entity_id}",
                f"trust:{rel.trust_level.value}",
            ],
            'evidence_ids': [],
            'trigger_source': 'social_memory',
            'trigger_episode': self._teacher_exit_episode or 0,
        }
        
        return proposal
    
    def generate_report(self, current_episode: int) -> SocialMemoryReport:
        """Generate social memory activity report."""
        report = SocialMemoryReport(
            relationships_tracked=len(self._relationships),
            interactions_logged=sum(r.total_interactions for r in self._relationships.values()),
            teacher_exit_detected=self._teacher_exit_reported,
            episode=current_episode,
        )
        
        # Count new relationships this episode
        # (simplified - in production would track episode of creation)
        
        return report


def log_teacher_exit(
    event_log,
    exit_episode: int,
    reason: str = "natural_exit",
) -> None:
    """
    Log teacher exit to raw event log.
    
    Called when teacher exit is confirmed.
    """
    if event_log is None:
        return
    
    event_log.append({
        'event_type': 'teacher_exit',
        'episode': exit_episode,
        'tick': 0,
        'data': {
            'reason': reason,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        'source_module': 'teacher',
    })


def compute_collaborator_relevance(
    relationship: RelationshipModel,
    current_goals: List[str],
) -> float:
    """
    Compute relevance score for a collaborator based on current goals.
    
    Returns 0.0 to 1.0.
    """
    if not relationship:
        return 0.0
    
    if not current_goals:
        return 0.5  # Neutral if no current goals
    
    # Check overlap between relevant_goals and current_goals
    if not relationship.relevant_goals:
        return 0.3  # Low relevance if no known goals
    
    overlap = set(relationship.relevant_goals) & set(current_goals)
    relevance = len(overlap) / len(current_goals)
    
    # Adjust by trust
    trust_factor = relationship.trust_score
    
    return relevance * trust_factor

"""
modules/memory/schema.py

Phase 0: Memory Unification Guardrails

Canonical memory metadata fields and type definitions.
All memory objects in the object store should use these fields.

Two-Store Model:
- Raw Memory Log (event_log.py) — append-only, immutable history
- Formal Memory Truth (object_store.py) — typed memory objects via validator/committer

Rules:
- memory_type and memory_layer are REQUIRED for all formal memory objects
- backward compatibility: 'type' field may exist but memory_type takes precedence
- raw log events are NOT formal memory until they pass validator + committer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MemoryLayer(Enum):
    """Which memory layer this object belongs to."""
    EPISODIC = "episodic"           # Episode-level records
    SEMANTIC = "semantic"          # Facts, constraints, confirmed knowledge
    PROCEDURAL = "procedural"       # Skills, recovery paths, biases
    CONTINUITY = "continuity"       # Identity, goals, self-model
    SOCIAL = "social"              # Teacher, collaborators, trust
    MECHANISM = "mechanism"        # World model beliefs, mechanisms


class MemoryType(Enum):
    """
    Specific memory object types.
    Every formal memory object should have a memory_type.
    """
    # Layer 1: Episodic
    EPISODE_RECORD = "episode_record"
    
    # Layer 2: Semantic
    FACT_CARD = "fact_card"
    CONSTRAINT_RECORD = "constraint_record"
    MECHANISM_HYPOTHESIS = "mechanism_hypothesis"
    MECHANISM_SUMMARY = "mechanism_summary"
    
    # Layer 3: Procedural
    SKILL_CARD = "skill_card"
    PROCEDURE_CHAIN = "procedure_chain"
    RECOVERY_PATH = "recovery_path"
    SELECTOR_BIAS = "selector_bias"
    AGENDA_PRIOR = "agenda_prior"
    REPRESENTATION_PRIOR = "representation_prior"
    
    # Layer 4: Continuity
    GOAL_CARD = "goal_card"
    IDENTITY_STATE = "identity_state"
    AGENDA_ITEM = "agenda_item"
    
    # Layer 5: Social
    RELATIONSHIP_MODEL = "relationship_model"
    TEACHER_ASSET = "teacher_asset"
    
    # Layer 6: Mechanism
    BELIEF = "belief"
    DISTILLED_ASSET = "distilled_asset"
    
    # Legacy / Generic
    GENERIC_OBJECT = "generic_object"


class MemoryStatus(Enum):
    """Lifecycle state of a memory object."""
    ACTIVE = "active"
    WEAKENED = "weakened"
    RETIRED = "retired"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"


class RetrievalTag(Enum):
    """Tags for retrieval filtering."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    TESTING = "testing"
    RECOVERY = "recovery"
    CONSOLIDATION = "consolidation"
    TEACHER = "teacher"
    AGENT = "agent"
    SOCIAL_MEMORY = "social_memory"


# =============================================================================
# Canonical Memory Metadata Fields
# =============================================================================

CANONICAL_MEMORY_FIELDS = [
    'object_id',
    'memory_type',
    'memory_layer',
    'status',
    'content',
    'created_at',
    'updated_at',
    'source_event_ids',
    'source_episode_ids',
    'support_evidence',
    'contradiction_evidence',
    'confidence',
    'lifecycle_state',
    'retrieval_tags',
    'linked_objects',
    'utility_score',
    'last_used_at',
    'distillation_status',
    'owner_scope',
]


@dataclass
class MemoryMetadata:
    """
    Canonical metadata for formal memory objects.
    
    All formal memory objects in object_store should include these fields.
    """
    memory_type: MemoryType
    memory_layer: MemoryLayer
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    # Provenance
    source_event_ids: List[str] = field(default_factory=list)
    source_episode_ids: List[int] = field(default_factory=list)
    
    # Evidence tracking
    support_evidence: List[str] = field(default_factory=list)
    contradiction_evidence: List[str] = field(default_factory=list)
    
    # Confidence and utility
    confidence: float = 0.5
    utility_score: float = 0.0
    
    # Retrieval
    retrieval_tags: List[str] = field(default_factory=list)
    
    # Relationships
    linked_objects: List[str] = field(default_factory=list)
    
    # Temporal
    last_used_at: Optional[int] = None
    last_used_episode: Optional[int] = None
    
    # Distillation
    distillation_status: Optional[str] = None  # 'new', 'compiled', 'distilled'
    
    # Scope
    owner_scope: str = "system"  # 'system', 'teacher', 'agent'
    
    def to_dict(self) -> dict:
        """Convert to dict for storage."""
        return {
            'memory_type': self.memory_type.value if isinstance(self.memory_type, Enum) else self.memory_type,
            'memory_layer': self.memory_layer.value if isinstance(self.memory_layer, Enum) else self.memory_layer,
            'status': self.status.value if isinstance(self.status, Enum) else self.status,
            'source_event_ids': self.source_event_ids,
            'source_episode_ids': self.source_episode_ids,
            'support_evidence': self.support_evidence,
            'contradiction_evidence': self.contradiction_evidence,
            'confidence': self.confidence,
            'utility_score': self.utility_score,
            'retrieval_tags': self.retrieval_tags,
            'linked_objects': self.linked_objects,
            'last_used_at': self.last_used_at,
            'last_used_episode': self.last_used_episode,
            'distillation_status': self.distillation_status,
            'owner_scope': self.owner_scope,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryMetadata':
        """Create from dict."""
        return cls(
            memory_type=MemoryType(data.get('memory_type', 'generic_object')),
            memory_layer=MemoryLayer(data.get('memory_layer', 'semantic')),
            status=MemoryStatus(data.get('status', 'active')),
            source_event_ids=data.get('source_event_ids', []),
            source_episode_ids=data.get('source_episode_ids', []),
            support_evidence=data.get('support_evidence', []),
            contradiction_evidence=data.get('contradiction_evidence', []),
            confidence=data.get('confidence', 0.5),
            utility_score=data.get('utility_score', 0.0),
            retrieval_tags=data.get('retrieval_tags', []),
            linked_objects=data.get('linked_objects', []),
            last_used_at=data.get('last_used_at'),
            last_used_episode=data.get('last_used_episode'),
            distillation_status=data.get('distillation_status'),
            owner_scope=data.get('owner_scope', 'system'),
        )


def is_memory_type_field(field_name: str) -> bool:
    """Check if field_name is a canonical memory field."""
    return field_name in CANONICAL_MEMORY_FIELDS


def get_memory_layer_for_type(memory_type: MemoryType) -> MemoryLayer:
    """Get the canonical memory layer for a given memory type."""
    mapping = {
        # Episodic
        MemoryType.EPISODE_RECORD: MemoryLayer.EPISODIC,
        
        # Semantic
        MemoryType.FACT_CARD: MemoryLayer.SEMANTIC,
        MemoryType.CONSTRAINT_RECORD: MemoryLayer.SEMANTIC,
        MemoryType.MECHANISM_HYPOTHESIS: MemoryLayer.SEMANTIC,
        MemoryType.MECHANISM_SUMMARY: MemoryLayer.MECHANISM,
        
        # Procedural
        MemoryType.SKILL_CARD: MemoryLayer.PROCEDURAL,
        MemoryType.PROCEDURE_CHAIN: MemoryLayer.PROCEDURAL,
        MemoryType.RECOVERY_PATH: MemoryLayer.PROCEDURAL,
        MemoryType.SELECTOR_BIAS: MemoryLayer.PROCEDURAL,
        MemoryType.AGENDA_PRIOR: MemoryLayer.PROCEDURAL,
        MemoryType.REPRESENTATION_PRIOR: MemoryLayer.PROCEDURAL,
        
        # Continuity
        MemoryType.GOAL_CARD: MemoryLayer.CONTINUITY,
        MemoryType.IDENTITY_STATE: MemoryLayer.CONTINUITY,
        MemoryType.AGENDA_ITEM: MemoryLayer.CONTINUITY,
        
        # Social
        MemoryType.RELATIONSHIP_MODEL: MemoryLayer.SOCIAL,
        MemoryType.TEACHER_ASSET: MemoryLayer.SOCIAL,
        
        # Mechanism
        MemoryType.BELIEF: MemoryLayer.MECHANISM,
        MemoryType.DISTILLED_ASSET: MemoryLayer.MECHANISM,
        
        # Generic
        MemoryType.GENERIC_OBJECT: MemoryLayer.SEMANTIC,
    }
    return mapping.get(memory_type, MemoryLayer.SEMANTIC)

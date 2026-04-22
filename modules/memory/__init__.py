"""
modules/memory/__init__.py

Memory System - Phases 0-7 (Complete)

modules/memory/schema.py
- MemoryLayer, MemoryType, MemoryStatus, RetrievalTag
- MemoryMetadata, CANONICAL_MEMORY_FIELDS

modules/memory/event_log.py
- EventLog: append-only raw event persistence
- EventLogBuilder: helper to create canonical events

modules/memory/episode_summarizer.py
- EpisodeRecord, EpisodeActionSummary
- EpisodeSummarizer: creates typed episode_record objects

modules/memory/retrieval_bundle.py
- RetrievalBundle: query + memory metadata context
- RetrievalBundleBuilder: creates bundles from contexts
- RetrievalBundleFilter: applies bundle filters

modules/memory/router.py
- MemoryRouter: canonical routing of formal memory objects

modules/memory/promotion_rules.py
- MemoryPromotionRules: episode -> semantic/autobiographical and procedural -> skill/transfer promotion heuristics

modules/memory/semantic_store.py
- SemanticMemoryStore: read facade over semantic/mechanism memory

modules/memory/procedural_store.py
- ProceduralMemoryStore: read facade over callable/reusable procedural memory

modules/memory/autobiographical_store.py
- AutobiographicalMemoryStore: continuity-facing memory retrieval

modules/memory/retrieval_surface.py
- RetrievalSurface: canonical grouped surfacing protocol for downstream consumers

modules/memory/skill_registry.py
- SkillRegistry: callable skill registry built on procedural memory

modules/memory/consolidation.py
- ConsolidationEngine: merges, promotes, weakens, retires memory
- ConsolidationReport: results of consolidation pass
- run_light_consolidation: lightweight pass at episode end
- run_heavy_consolidation: full pass every N episodes

modules/memory/distillation.py
- DistillationTracker: tracks behavior change from stored memory
- DistillationEvidence: evidence of policy change
- DistillationReport: distillation verification results
- verify_distillation: verify and promote distilled assets

modules/memory/social_memory.py
- SocialMemoryTracker: minimal social memory slice
- RelationshipModel: collaborator/teacher relationship model
- TrustLevel: trust states for collaborators
- InteractionType: types of social interactions
- log_teacher_exit: log teacher exit to raw event log
"""

from modules.memory.schema import (
    MemoryLayer,
    MemoryType,
    MemoryStatus,
    RetrievalTag,
    MemoryMetadata,
    CANONICAL_MEMORY_FIELDS,
    is_memory_type_field,
    get_memory_layer_for_type,
)

from modules.memory.event_log import (
    EventLog,
    EventLogBuilder,
)

from modules.memory.episode_summarizer import (
    EpisodeRecord,
    EpisodeActionSummary,
    EpisodeSummarizer,
)

from modules.memory.retrieval_bundle import (
    RetrievalGoal,
    MemoryState,
    RetrievalBundle,
    RetrievalBundleBuilder,
    RetrievalBundleFilter,
)

from modules.memory.router import (
    MemoryRouter,
    route_object_record,
    MEMORY_ROUTE_SEMANTIC,
    MEMORY_ROUTE_PROCEDURAL,
    MEMORY_ROUTE_AUTOBIOGRAPHICAL,
    MEMORY_ROUTE_CONTINUITY,
    MEMORY_ROUTE_EPISODIC,
    MEMORY_ROUTE_MECHANISM,
)

from modules.memory.promotion_rules import (
    MemoryPromotionRules,
)

from modules.memory.semantic_store import (
    SemanticMemoryStore,
)

from modules.memory.procedural_store import (
    ProceduralMemoryStore,
)

from modules.memory.autobiographical_store import (
    AutobiographicalMemoryStore,
)

from modules.memory.retrieval_surface import (
    RetrievalSurface,
)

from modules.memory.skill_registry import (
    SkillRegistry,
)

from modules.memory.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationResult,
    run_light_consolidation,
    run_heavy_consolidation,
    UTILITY_THRESHOLD_WEAKEN,
    UTILITY_THRESHOLD_RETIRE,
)

from modules.memory.distillation import (
    DistillationState,
    DistillationEvidence,
    DistillationTracker,
    DistillationReport,
    verify_distillation,
    check_policy_change,
    MIN_CONSUMPTIONS_FOR_CANDIDATE,
    MIN_BENEFIT_RATIO_FOR_COMPILE,
)

from modules.memory.social_memory import (
    SocialMemoryTracker,
    RelationshipModel,
    SocialMemoryReport,
    TrustLevel,
    InteractionType,
    log_teacher_exit,
    compute_collaborator_relevance,
    TRUST_THRESHOLD_TRUSTED,
    TRUST_THRESHOLD_VERIFIED,
)

__all__ = [
    # Schema
    'MemoryLayer',
    'MemoryType',
    'MemoryStatus',
    'RetrievalTag',
    'MemoryMetadata',
    'CANONICAL_MEMORY_FIELDS',
    'is_memory_type_field',
    'get_memory_layer_for_type',
    # Event Log
    'EventLog',
    'EventLogBuilder',
    # Episodic
    'EpisodeRecord',
    'EpisodeActionSummary',
    'EpisodeSummarizer',
    # Retrieval Bundle
    'RetrievalGoal',
    'MemoryState',
    'RetrievalBundle',
    'RetrievalBundleBuilder',
    'RetrievalBundleFilter',
    'MemoryRouter',
    'route_object_record',
    'MEMORY_ROUTE_SEMANTIC',
    'MEMORY_ROUTE_PROCEDURAL',
    'MEMORY_ROUTE_AUTOBIOGRAPHICAL',
    'MEMORY_ROUTE_CONTINUITY',
    'MEMORY_ROUTE_EPISODIC',
    'MEMORY_ROUTE_MECHANISM',
    'MemoryPromotionRules',
    'SemanticMemoryStore',
    'ProceduralMemoryStore',
    'AutobiographicalMemoryStore',
    'RetrievalSurface',
    'SkillRegistry',
    # Consolidation
    'ConsolidationEngine',
    'ConsolidationReport',
    'ConsolidationResult',
    'run_light_consolidation',
    'run_heavy_consolidation',
    'UTILITY_THRESHOLD_WEAKEN',
    'UTILITY_THRESHOLD_RETIRE',
    # Distillation
    'DistillationState',
    'DistillationEvidence',
    'DistillationTracker',
    'DistillationReport',
    'verify_distillation',
    'check_policy_change',
    'MIN_CONSUMPTIONS_FOR_CANDIDATE',
    'MIN_BENEFIT_RATIO_FOR_COMPILE',
    # Social Memory
    'SocialMemoryTracker',
    'RelationshipModel',
    'SocialMemoryReport',
    'TrustLevel',
    'InteractionType',
    'log_teacher_exit',
    'compute_collaborator_relevance',
    'TRUST_THRESHOLD_TRUSTED',
    'TRUST_THRESHOLD_VERIFIED',
]

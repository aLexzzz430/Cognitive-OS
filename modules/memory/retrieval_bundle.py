"""
modules/memory/retrieval_bundle.py

Phase 4: State-Conditioned Retrieval Bundle

RetrievalBundle = search query + memory metadata context + filtering.

When retrieving memories, include:
- What goal/episode are we in?
- What memory_types are relevant?
- What retrieval_tags are active?
- What's the memory_state (exploration vs exploitation)?

This lets retrieval be conditioned on agent state, not just query similarity.

Rules:
- RetrievalBundle is NOT a new control structure
- It's a query enrichment pattern
- Goes through normal retrieval pipeline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from modules.memory.schema import MemoryLayer, MemoryType, RetrievalTag


class RetrievalGoal(Enum):
    """What the agent is trying to accomplish."""
    EXPLORE = "explore"           # Discovering new patterns/mechanisms
    EXPLOIT = "exploit"          # Using known patterns effectively
    TEST = "test"                # Testing a hypothesis
    RECOVER = "recover"          # Recovering from an error
    CONSOLIDATE = "consolidate"  # Consolidating recent learning


class MemoryState(Enum):
    """Current memory system state."""
    FORMING = "forming"         # New episode, memories being formed
    STABLE = "stable"           # Established knowledge
    TRANSITIONING = "transitioning"  # Teacher exit, regime change
    WEAKENING = "weakening"     # Forgetting/garbage collection


@dataclass
class RetrievalBundle:
    """
    A retrieval query enriched with memory metadata.
    
    Bundles:
    - Base query text
    - Agent state context (goal, episode, memory_state)
    - Memory type filter preferences
    - Retrieval tag filter
    - Desired memory layer
    
    Usage:
        bundle = RetrievalBundle.create(
            query="join tables efficiently",
            goal=RetrievalGoal.EXPLOIT,
            episode=5,
            memory_state=MemoryState.STABLE,
        )
        results = object_store.retrieve_with_bundle(bundle)
    """
    # Base query
    query: str
    
    # Agent state context
    goal: RetrievalGoal
    episode: int
    memory_state: MemoryState
    teacher_present: bool = False
    
    # Filtering preferences
    preferred_memory_types: List[str] = field(default_factory=list)
    preferred_memory_layer: Optional[str] = None
    preferred_tags: List[str] = field(default_factory=list)
    excluded_tags: List[str] = field(default_factory=list)
    
    # Constraints
    max_results: int = 10
    min_confidence: float = 0.0
    
    # Metadata for audit
    context_summary: str = ""  # Brief description of retrieval context
    
    @classmethod
    def create(
        cls,
        query: str,
        goal: RetrievalGoal,
        episode: int,
        memory_state: MemoryState = MemoryState.STABLE,
        teacher_present: bool = False,
    ) -> 'RetrievalBundle':
        """
        Factory method with smart defaults based on goal.
        """
        bundle = cls(
            query=query,
            goal=goal,
            episode=episode,
            memory_state=memory_state,
            teacher_present=teacher_present,
        )
        
        # Set smart defaults based on goal
        if goal == RetrievalGoal.EXPLORE:
            bundle.preferred_memory_layer = MemoryLayer.SEMANTIC.value
            bundle.preferred_tags = [RetrievalTag.EXPLORATION.value]
            bundle.context_summary = "exploration retrieval"
        
        elif goal == RetrievalGoal.EXPLOIT:
            bundle.preferred_memory_layer = MemoryLayer.PROCEDURAL.value
            bundle.preferred_tags = [RetrievalTag.EXPLOITATION.value]
            bundle.context_summary = "exploitation retrieval"
        
        elif goal == RetrievalGoal.TEST:
            bundle.preferred_memory_types = [
                MemoryType.MECHANISM_HYPOTHESIS.value,
                MemoryType.FACT_CARD.value,
            ]
            bundle.preferred_tags = [RetrievalTag.TESTING.value]
            bundle.context_summary = "testing retrieval"
        
        elif goal == RetrievalGoal.RECOVER:
            bundle.preferred_memory_layer = MemoryLayer.PROCEDURAL.value
            bundle.preferred_types = [MemoryType.RECOVERY_PATH.value]
            bundle.preferred_tags = [RetrievalTag.RECOVERY.value]
            bundle.context_summary = "recovery retrieval"
        
        elif goal == RetrievalGoal.CONSOLIDATE:
            bundle.preferred_memory_layer = MemoryLayer.EPISODIC.value
            bundle.preferred_tags = [RetrievalTag.CONSOLIDATION.value]
            bundle.context_summary = "consolidation retrieval"
        
        return bundle
    
    def to_dict(self) -> dict:
        """Serialize for logging and audit."""
        return {
            'query': self.query,
            'goal': self.goal.value,
            'episode': self.episode,
            'memory_state': self.memory_state.value,
            'teacher_present': self.teacher_present,
            'preferred_memory_types': self.preferred_memory_types,
            'preferred_memory_layer': self.preferred_memory_layer,
            'preferred_tags': self.preferred_tags,
            'excluded_tags': self.excluded_tags,
            'max_results': self.max_results,
            'min_confidence': self.min_confidence,
            'context_summary': self.context_summary,
        }


class RetrievalBundleBuilder:
    """
    Helper to build RetrievalBundles from various contexts.
    """
    
    @staticmethod
    def from_goal_type(goal_type: str, query: str, episode: int, teacher_present: bool = False) -> RetrievalBundle:
        """
        Build bundle from goal type string.
        
        Args:
            goal_type: 'exploration', 'exploitation', 'testing', 'recovery', 'consolidation'
            query: retrieval query
            episode: current episode
            teacher_present: whether teacher is active
        """
        # Parse goal
        goal_map = {
            'exploration': RetrievalGoal.EXPLORE,
            'exploitation': RetrievalGoal.EXPLOIT,
            'testing': RetrievalGoal.TEST,
            'recovery': RetrievalGoal.RECOVER,
            'consolidation': RetrievalGoal.CONSOLIDATE,
        }
        goal = goal_map.get(goal_type.lower(), RetrievalGoal.EXPLORE)
        
        # Determine memory state
        if teacher_present:
            memory_state = MemoryState.FORMING
        elif episode <= 3:
            memory_state = MemoryState.FORMING
        elif episode <= 10:
            memory_state = MemoryState.STABLE
        else:
            memory_state = MemoryState.TRANSITIONING
        
        return RetrievalBundle.create(
            query=query,
            goal=goal,
            episode=episode,
            memory_state=memory_state,
            teacher_present=teacher_present,
        )
    
    @staticmethod
    def for_episode_retrieval(episode_id: int, query: str) -> RetrievalBundle:
        """Build bundle for retrieving memories from a specific episode."""
        return RetrievalBundle(
            query=query,
            goal=RetrievalGoal.CONSOLIDATE,
            episode=episode_id,
            memory_state=MemoryState.STABLE,
            preferred_memory_types=[MemoryType.EPISODE_RECORD.value],
            context_summary=f"episode_{episode_id}_retrieval",
        )
    
    @staticmethod
    def for_layer_retrieval(layer: MemoryLayer, query: str, episode: int) -> RetrievalBundle:
        """Build bundle for retrieving from a specific memory layer."""
        return RetrievalBundle(
            query=query,
            goal=RetrievalGoal.EXPLORE,
            episode=episode,
            memory_state=MemoryState.STABLE,
            preferred_memory_layer=layer.value,
            context_summary=f"{layer.value}_layer_retrieval",
        )


class RetrievalBundleFilter:
    """
    Applies RetrievalBundle filters to a set of candidate objects.
    
    Used by object store to filter results based on bundle preferences.
    """
    
    def __init__(self, bundle: RetrievalBundle):
        self._bundle = bundle
    
    def filter(self, objects: List[dict]) -> List[dict]:
        """
        Apply bundle filters to objects.
        
        Filters:
        1. Memory type filter (preferred)
        2. Memory layer filter (preferred)
        3. Tag filter (preferred)
        4. Confidence filter (min_confidence)
        5. Excluded tags filter
        """
        results = list(objects)
        
        # Filter by preferred memory types
        if self._bundle.preferred_memory_types:
            results = [o for o in results if self._is_match_type(o)]
        
        # Filter by preferred memory layer
        if self._bundle.preferred_memory_layer:
            results = [o for o in results if self._is_match_layer(o)]
        
        # Filter by preferred tags (at least one match)
        if self._bundle.preferred_tags:
            results = [o for o in results if self._is_match_tags(o)]
        
        # Exclude tags
        if self._bundle.excluded_tags:
            results = [o for o in results if not self._is_excluded(o)]
        
        # Filter by minimum confidence
        if self._bundle.min_confidence > 0:
            results = [o for o in results if o.get('confidence', 0) >= self._bundle.min_confidence]
        
        # Sort by relevance (simple: confidence + tag match count)
        results = self._score_and_sort(results)
        
        # Limit results
        return results[:self._bundle.max_results]
    
    def _is_match_type(self, obj: dict) -> bool:
        """Check if object matches preferred memory types."""
        obj_type = obj.get('memory_type', obj.get('type', ''))
        return obj_type in self._bundle.preferred_memory_types
    
    def _is_match_layer(self, obj: dict) -> bool:
        """Check if object matches preferred memory layer."""
        obj_layer = obj.get('memory_layer', '')
        return obj_layer == self._bundle.preferred_memory_layer
    
    def _is_match_tags(self, obj: dict) -> bool:
        """Check if object has any of the preferred tags."""
        obj_tags = set(obj.get('retrieval_tags', []))
        preferred = set(self._bundle.preferred_tags)
        return bool(obj_tags & preferred)  # Intersection
    
    def _is_excluded(self, obj: dict) -> bool:
        """Check if object has any excluded tags."""
        obj_tags = set(obj.get('retrieval_tags', []))
        excluded = set(self._bundle.excluded_tags)
        return bool(obj_tags & excluded)  # Intersection
    
    def _score_and_sort(self, objects: List[dict]) -> List[dict]:
        """Score objects by relevance and return sorted."""
        scored = []
        for obj in objects:
            score = 0.0
            
            # Confidence contributes to score
            score += obj.get('confidence', 0.5) * 0.5
            
            # Tag matches contribute
            obj_tags = set(obj.get('retrieval_tags', []))
            preferred = set(self._bundle.preferred_tags)
            tag_matches = len(obj_tags & preferred)
            score += tag_matches * 0.2
            
            # Memory type match contributes
            if self._bundle.preferred_memory_types:
                obj_type = obj.get('memory_type', obj.get('type', ''))
                if obj_type in self._bundle.preferred_memory_types:
                    score += 0.3
            
            scored.append((score, obj))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [obj for _, obj in scored]

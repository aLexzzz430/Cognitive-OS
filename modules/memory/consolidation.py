"""
modules/memory/consolidation.py

Phase 5: Consolidation + Forgetting

Consolidation = periodic cleanup that prevents memory sprawl and stale poisoning.

Rules:
- Consolidation MAY produce candidates
- Consolidation MAY NOT bypass formal validation
- Raw event log is NEVER modified by consolidation
- Only formal object store is affected

Operations:
1. merge: Near-duplicate episode_records are merged
2. promote: Repeated patterns become semantic/procedural candidates
3. weaken: Low-utility objects are weakened
4. retire: Stale/harmful objects are retired to garbage
5. conflict: Contradicted objects are marked for adjudication

Forgetting Policy (utility-sensitive):
- recent usage
- historical usefulness
- contradiction count
- age
- retrieval relevance
- whether a stronger merged successor exists

Report Shape:
{
    "merged_count": int,
    "promoted_count": int,
    "weakened_count": int,
    "retired_count": int,
    "conflict_candidates": [object_ids],
    "new_memory_candidates": [proposals],
    "timestamp": str,
    "episode": int,
}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import hashlib

from modules.memory.schema import MemoryType, MemoryLayer, RetrievalTag
from modules.memory.promotion_rules import MemoryPromotionRules


# Utility thresholds
UTILITY_THRESHOLD_WEAKEN = 0.3  # Below this -> weaken
UTILITY_THRESHOLD_RETIRE = 0.1  # Below this -> retire to garbage
USAGE_COUNT_THRESHOLD_WEAK = 3   # Low usage episodes
USAGE_COUNT_THRESHOLD_PROMOTE = 5  # High usage -> promote
AGE_THRESHOLD_STALE = 10  # Episodes old without usage
CONFLICT_TAG_THRESHOLD = 2  # Same tags, different conclusions


@dataclass
class ConsolidationReport:
    """Result of a consolidation run."""
    merged_count: int = 0
    promoted_count: int = 0
    validated_promotion_count: int = 0
    weakened_count: int = 0
    retired_count: int = 0
    conflict_candidates: List[str] = field(default_factory=list)
    new_memory_candidates: List[Dict] = field(default_factory=list)
    promoted_object_ids: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    episode: int = 0
    
    def to_dict(self) -> dict:
        return {
            'merged_count': self.merged_count,
            'promoted_count': self.promoted_count,
            'validated_promotion_count': self.validated_promotion_count,
            'weakened_count': self.weakened_count,
            'retired_count': self.retired_count,
            'conflict_candidates': self.conflict_candidates,
            'new_memory_candidates': self.new_memory_candidates,
            'promoted_object_ids': self.promoted_object_ids,
            'timestamp': self.timestamp,
            'episode': self.episode,
        }


class ConsolidationResult(Enum):
    """Possible outcomes of consolidation operations."""
    MERGED = "merged"
    PROMOTED = "promoted"
    WEAKENED = "weakened"
    RETIRED = "retired"
    CONFLICT = "conflict"
    NO_CHANGE = "no_change"


def _compute_content_fingerprint(obj: dict) -> str:
    """
    Compute a simple fingerprint for content similarity.
    
    Used to find near-duplicate episode records.
    """
    content = obj.get('content', {})
    if isinstance(content, dict):
        # Normalize key elements
        normalized = {
            'goal': content.get('goal', ''),
            'outcome': content.get('outcome', ''),
            'key_actions': tuple(sorted(content.get('key_actions', []))[:3]),  # Top 3
        }
        fingerprint = str(normalized)
    else:
        fingerprint = str(content)[:100]
    
    return hashlib.md5(fingerprint.encode()).hexdigest()[:8]


def _compute_utility_components(obj: dict) -> Dict[str, float]:
    """
    Compute utility decomposition for an object.
    
    Returns:
    - episode_local: short-horizon utility from recent usage quality
    - long_horizon: long-horizon utility from confidence/status/age
    - blended: weighted blend used by legacy thresholds
    """
    from modules.governance.object_store import AssetStatus
    
    base = obj.get('confidence', 0.5)
    
    # Consumption factor
    consumption = obj.get('consumption_count', 0)
    consumption_factor = min(consumption * 0.05, 0.3)  # Cap at 0.3
    
    # Reuse quality factor
    reuse_history = obj.get('reuse_history', [])
    if reuse_history:
        beneficial = sum(1 for r in reuse_history if r.get('was_beneficial', False))
        total = len(reuse_history)
        quality = beneficial / total if total > 0 else 0.5
    else:
        quality = 0.5
    
    # Asset status factor
    asset_status = obj.get('asset_status', AssetStatus.NEW_ASSET.value)
    status_factors = {
        AssetStatus.NEW_ASSET.value: 0.0,
        AssetStatus.LIVE_ASSET.value: 0.1,
        AssetStatus.REUSABLE_ASSET.value: 0.2,
        AssetStatus.COMPILED_ASSET.value: 0.3,
        AssetStatus.DISTILLED_ASSET.value: 0.4,
        AssetStatus.GARBAGE.value: -1.0,
    }
    status_factor = status_factors.get(asset_status, 0.0)
    
    # Age factor (newer is slightly better)
    age_episodes = obj.get('trigger_episode', 0)
    age_factor = max(0, 0.1 - (age_episodes * 0.01))
    
    episode_local = max(0.0, min(1.0, (quality * 0.55) + min(0.35, consumption * 0.08)))
    long_horizon = max(0.0, min(1.0, base + status_factor + age_factor))
    blended = max(0.0, min(1.0, long_horizon * 0.6 + episode_local * 0.4))
    return {
        'episode_local': episode_local,
        'long_horizon': long_horizon,
        'blended': blended,
    }


def _compute_utility_score(obj: dict) -> float:
    """Backward-compatible single utility score (blended utility)."""
    return _compute_utility_components(obj).get('blended', 0.0)


class ConsolidationEngine:
    """
    Memory consolidation engine.
    
    Responsible for:
    1. Merging near-duplicate episode records
    2. Promoting high-utility patterns to semantic/procedural memory
    3. Weakening/retiring low-utility memory
    4. Finding conflict candidates
    
    Usage:
        engine = ConsolidationEngine(object_store, event_log)
        report = engine.run(current_episode)
    """
    
    def __init__(self, object_store, event_log=None):
        """
        Initialize consolidation engine.
        
        Args:
            object_store: ObjectStore instance (formal memory truth)
            event_log: Optional EventLog instance (raw history)
        """
        self._store = object_store
        self._event_log = event_log
        self._report = ConsolidationReport()
        self._promotion_rules = MemoryPromotionRules(object_store)
    
    def run(self, current_episode: int, dry_run: bool = False) -> ConsolidationReport:
        """
        Run full consolidation pass.
        
        Args:
            current_episode: Current episode number
            dry_run: If True, don't modify, just report
        
        Returns:
            ConsolidationReport with results
        """
        self._report = ConsolidationReport(episode=current_episode)
        
        # Phase 1: Find and merge near-duplicate episodes
        self._merge_episode_records(dry_run)
        
        # Phase 2: Find patterns to promote
        self._find_promotion_candidates(dry_run)
        
        # Phase 3: Weaken/retire low-utility memory
        self._apply_forgetting(current_episode=current_episode, dry_run=dry_run)
        
        # Phase 4: Find conflicts
        self._find_conflicts()
        
        return self._report
    
    def _merge_episode_records(self, dry_run: bool) -> None:
        """Find and merge near-duplicate episode records."""
        episodes = self._store.get_by_memory_type(MemoryType.EPISODE_RECORD.value)
        
        if len(episodes) < 2:
            return
        
        # Group by content fingerprint
        groups: Dict[str, List[dict]] = {}
        for ep in episodes:
            if ep.get('status') == 'invalidated':
                continue
            fp = _compute_content_fingerprint(ep)
            if fp not in groups:
                groups[fp] = []
            groups[fp].append(ep)
        
        # Merge groups with 2+ members
        for fp, group in groups.items():
            if len(group) < 2:
                continue
            
            # Keep the most recent/highest confidence as primary
            group.sort(key=lambda x: (x.get('confidence', 0), x.get('trigger_episode', 0)), reverse=True)
            primary = group[0]
            secondary = group[1:]
            
            # Merge evidence IDs and content
            all_evidence = set(primary.get('evidence_ids', []))
            for obj in secondary:
                all_evidence.update(obj.get('evidence_ids', []))
            
            if not dry_run:
                primary_id = primary.get('object_id', '')
                primary_confidence = min(1.0, float(primary.get('confidence', 0.5) or 0.5) + 0.05 * len(secondary))
                if primary_id:
                    self._store.merge_update(
                        primary_id,
                        list(all_evidence),
                        additional_content={
                            'memory_metadata': {
                                'consolidation': {
                                    'merged_fingerprint': fp,
                                    'merged_secondary_count': len(secondary),
                                },
                            },
                        },
                    )
                    self._store.update_fields(
                        primary_id,
                        {'confidence': primary_confidence},
                        reason='consolidation_merge_confidence_boost',
                        evidence_ids=list(all_evidence),
                    )
                
                # Invalidate secondary
                for obj in secondary:
                    self._store.invalidate(obj.get('object_id', ''))
            
            self._report.merged_count += len(secondary)
    
    def _find_promotion_candidates(self, dry_run: bool) -> None:
        """
        Find high-utility patterns that should be promoted.
        
        Promotion: episodic -> semantic/procedural
        """
        episodes = self._store.get_by_memory_type(MemoryType.EPISODE_RECORD.value)
        
        for ep in episodes:
            if ep.get('status') == 'invalidated':
                continue
            
            utility_components = _compute_utility_components(ep)
            utility = utility_components['blended']
            local_utility = utility_components['episode_local']
            long_horizon_utility = utility_components['long_horizon']
            consumption = ep.get('consumption_count', 0)

            proposals = self._promotion_rules.proposals_from_episode(ep)
            if not proposals:
                continue

            repeated_validation_ready = any(
                int(
                    (
                        proposal.get('memory_metadata', {})
                        if isinstance(proposal.get('memory_metadata', {}), dict)
                        else {}
                    ).get('validation_count', 0)
                    or 0
                ) >= 2
                for proposal in proposals
            )
            high_utility_ready = (
                consumption >= USAGE_COUNT_THRESHOLD_PROMOTE
                and local_utility >= 0.55
                and long_horizon_utility >= 0.6
            )
            if not high_utility_ready and not repeated_validation_ready:
                continue

            for proposal in proposals:
                proposal['utility'] = {
                    'episode_local': local_utility,
                    'long_horizon': long_horizon_utility,
                    'blended': utility,
                }
                self._report.new_memory_candidates.append(proposal)
    
    def _create_promotion_proposal(self, episode: dict) -> Optional[Dict]:
        """
        Create a promotion proposal from an episode.
        
        Returns a candidate dict for the validator.
        """
        content = episode.get('content', {})
        if not isinstance(content, dict):
            return None
        
        outcome = content.get('outcome', '') or content.get('summary', '')
        key_actions = content.get('key_actions', []) or content.get('actions', [])
        key_discoveries = content.get('key_discoveries', [])
        
        # Determine target memory type
        if 'skill' in outcome.lower() or 'procedure' in outcome.lower():
            target_type = MemoryType.SKILL_CARD.value
            target_layer = MemoryLayer.PROCEDURAL.value
        elif 'fact' in outcome.lower() or 'learned' in outcome.lower():
            target_type = MemoryType.FACT_CARD.value
            target_layer = MemoryLayer.SEMANTIC.value
        else:
            # Default to mechanism summary
            target_type = MemoryType.MECHANISM_SUMMARY.value
            target_layer = MemoryLayer.MECHANISM.value
        
        proposal = {
            'content': {
                'summary': outcome,
                'procedure': key_actions[0] if key_actions else outcome,
                'key_discoveries': key_discoveries[:5] if isinstance(key_discoveries, list) else [],
                'source_episode': episode.get('object_id'),
            },
            'memory_type': target_type,
            'memory_layer': target_layer,
            'confidence': min(1.0, episode.get('confidence', 0.5) + 0.1),
            'retrieval_tags': [RetrievalTag.CONSOLIDATION.value, RetrievalTag.EXPLOITATION.value],
            'evidence_ids': episode.get('evidence_ids', []),
            'trigger_source': 'consolidation',
            'trigger_episode': episode.get('trigger_episode', 0),
        }
        
        return proposal
    
    def _apply_forgetting(self, current_episode: int, dry_run: bool) -> None:
        """
        Apply forgetting policy to low-utility objects.
        
        Weaken: utility between 0.1 and 0.3
        Retire: utility below 0.1 or repeatedly contradicted
        """
        all_objs = self._store.retrieve()
        
        for obj in all_objs:
            if obj.get('status') == 'invalidated':
                continue
            
            # Skip already garbage
            from modules.governance.object_store import AssetStatus
            if obj.get('asset_status') == AssetStatus.GARBAGE.value:
                continue
            
            utility_components = _compute_utility_components(obj)
            utility = utility_components['blended']
            local_utility = utility_components['episode_local']
            long_horizon_utility = utility_components['long_horizon']
            consumption = obj.get('consumption_count', 0)
            
            # Check for staleness
            last_used = obj.get('last_consumed_tick')
            trigger_ep = obj.get('trigger_episode', 0)
            age = max(0, int(current_episode) - int(trigger_ep))
            
            # Never-used + old = stale
            is_stale = consumption == 0 and age >= AGE_THRESHOLD_STALE
            
            if is_stale or long_horizon_utility < UTILITY_THRESHOLD_RETIRE:
                if not dry_run:
                    self._store.retire_to_garbage(obj.get('object_id', ''))
                self._report.retired_count += 1
            elif local_utility < UTILITY_THRESHOLD_WEAKEN or utility < UTILITY_THRESHOLD_WEAKEN:
                if not dry_run:
                    obj_id = obj.get('object_id', '')
                    if obj_id:
                        self._store.weaken(obj_id, evidence_id='ev_consolidation_weaken')
                        self._store.update_fields(
                            obj_id,
                            {'confidence': max(0.1, utility)},
                            reason='consolidation_weaken_confidence',
                            evidence_ids=['ev_consolidation_weaken'],
                        )
                self._report.weakened_count += 1
    
    def _find_conflicts(self) -> None:
        """
        Find objects that may be contradicted.
        
        Looks for:
        - Same memory_type + same tags + different conclusions
        - Objects weakened by others
        """
        all_objs = self._store.retrieve()
        
        # Group by retrieval tags
        by_tags: Dict[Tuple[str, ...], List[dict]] = {}
        for obj in all_objs:
            if obj.get('status') == 'invalidated':
                continue
            tags = tuple(sorted(obj.get('retrieval_tags', [])))
            if tags:
                if tags not in by_tags:
                    by_tags[tags] = []
                by_tags[tags].append(obj)
        
        # Find conflicts within tag groups
        for tags, group in by_tags.items():
            if len(group) < 2:
                continue
            
            # Check for different conclusions
            conclusions = set()
            for obj in group:
                content = str(obj.get('content', {}))[:50]
                conclusions.add(content)
            
            if len(conclusions) > 1:
                # Potential conflict
                for obj in group:
                    obj_id = obj.get('object_id', '')
                    if obj_id not in self._report.conflict_candidates:
                        self._report.conflict_candidates.append(obj_id)


def run_light_consolidation(
    object_store,
    event_log=None,
    current_episode: int = 0,
) -> ConsolidationReport:
    """
    Light consolidation for episode end.
    
    Called at episode boundary.
    Only does lightweight operations:
    - Merge obvious duplicates
    - Mark low-utility for later review
    
    Heavy consolidation runs less frequently.
    """
    engine = ConsolidationEngine(object_store, event_log)
    engine._report = ConsolidationReport(episode=current_episode)

    # Light pass stays read-only, but it now surfaces promotion candidates so
    # episode-end consolidation can route them through the formal validator+committer.
    engine._merge_episode_records(dry_run=True)
    engine._find_promotion_candidates(dry_run=True)
    return engine._report


def run_heavy_consolidation(
    object_store,
    event_log=None,
    current_episode: int = 0,
) -> ConsolidationReport:
    """
    Full consolidation pass.
    
    Runs all operations:
    - Merge duplicates
    - Promote high-utility patterns
    - Apply forgetting policy
    - Find conflicts
    
    Should run every N episodes, not every episode.
    """
    engine = ConsolidationEngine(object_store, event_log)
    return engine.run(current_episode, dry_run=False)

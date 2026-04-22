"""
modules/governance/object_store.py

ObjectStore + ProposalValidator — 把"候选"变成"合法对象"的门。

Separated from core/main_loop.py for clarity.

P3-B3: Object lifecycle upgrade — asset status tracking.
Asset states: new_asset → live_asset → reusable_asset → compiled_asset → distilled_asset
Bad states: garbage (retired/bad utility)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Iterator, Tuple, Sequence
import copy
from dataclasses import dataclass, field
from enum import Enum
import time

from core.objects import (
    TRANSITION_INVALIDATE,
    TRANSITION_MERGE,
    TRANSITION_WEAKEN,
    apply_lifecycle_transition,
    infer_object_type,
    normalize_lifecycle_status,
    proposal_to_object_record,
)
from modules.memory.schema import MemoryLayer, MemoryType

# Governance decision constants
ACCEPT_NEW = "accept_new"
MERGE_UPDATE_EXISTING = "merge_update_existing"
REJECT = "reject"


class AssetStatus(Enum):
    """
    P3-B3: Asset lifecycle states for committed objects.
    
    new_asset: freshly committed, not yet used
    live_asset: triggered once, demonstrated some value
    reusable_asset: triggered multiple times, consistently beneficial
    compiled_asset: distilled into policy/skill/agenda
    distilled_asset: verified beneficial after teacher exit
    garbage: invalid, never triggered, or contradicted
    """
    NEW_ASSET = "new_asset"
    LIVE_ASSET = "live_asset"
    REUSABLE_ASSET = "reusable_asset"
    COMPILED_ASSET = "compiled_asset"
    DISTILLED_ASSET = "distilled_asset"
    GARBAGE = "garbage"


@dataclass
class GovernanceDecision:
    decision: str  # ACCEPT_NEW / MERGE_UPDATE_EXISTING / REJECT
    reason: str
    object_id: Optional[str] = None
    bypass_leak_check: bool = False
    leak_type: Optional[str] = None
    leak_detection_level: str = "heuristic"
    leak_gate_mode: str = "heuristic"
    confidence_boost: float = 0.0

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured export for audit/trace consumers."""
        return {
            'decision': self.decision,
            'reason': self.reason,
            'object_id': self.object_id,
            'bypass_leak_check': self.bypass_leak_check,
            'leak_type': self.leak_type,
            'leak_detection_level': self.leak_detection_level,
            'leak_gate_mode': self.leak_gate_mode,
            'confidence_boost': self.confidence_boost,
        }


@dataclass
class MergeRecord:
    object_id: str
    merged_from_id: str
    evidence_ids_added: List[str]
    additional_content: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class ObjectStore:
    """
    Persistent object storage with full lifecycle management.
    
    Lifecycle states: active → merged → weakened → invalidated

    NOTE:
    - _objects/_by_family/_merge_history are private internals.
    - External modules must use read-only query helpers (iter_objects/get/count/snapshot_for_audit)
      instead of touching private fields directly.
    """
    def __init__(self):
        self._objects: Dict[str, dict] = {}
        self._by_family: Dict[str, List[str]] = {}  # family → [object_ids]
        self._by_object_type: Dict[str, List[str]] = {}  # object_type → [object_ids]
        self._merge_history: Dict[str, List[MergeRecord]] = {}

    def _index_object(self, obj: Dict[str, Any]) -> None:
        object_id = str(obj.get('object_id') or '').strip()
        family = str(obj.get('family') or '').strip()
        object_type = str(obj.get('object_type') or infer_object_type(obj) or '').strip()
        if object_id and family:
            self._by_family.setdefault(family, [])
            if object_id not in self._by_family[family]:
                self._by_family[family].append(object_id)
        if object_id and object_type:
            self._by_object_type.setdefault(object_type, [])
            if object_id not in self._by_object_type[object_type]:
                self._by_object_type[object_type].append(object_id)

    @staticmethod
    def _current_lifecycle_status(obj: Dict[str, Any]) -> str:
        return normalize_lifecycle_status(
            obj.get('status'),
            history=list(obj.get('lifecycle_events', [])),
        )

    @staticmethod
    def _normalize_retrieval_tags(tags: Any) -> List[str]:
        """Normalize retrieval tags into a stable de-duplicated string list."""
        if tags is None:
            return []
        if isinstance(tags, list):
            values = tags
        elif isinstance(tags, (tuple, set)):
            values = list(tags)
        else:
            values = [tags]

        normalized: List[str] = []
        seen = set()
        for tag in values:
            if tag is None:
                continue
            tag_s = str(tag).strip()
            if not tag_s or tag_s in seen:
                continue
            seen.add(tag_s)
            normalized.append(tag_s)
        return normalized

    @staticmethod
    def _merge_memory_metadata(existing: Any, patch: Any, object_memory_type: str = "", object_memory_layer: str = "") -> Dict[str, Any]:
        """
        Merge memory metadata dictionaries with type/layer safety guarantees.
        - non-dict payloads are ignored
        - top-level memory_type/memory_layer are authoritative and cannot be changed by patch
        """
        base = dict(existing) if isinstance(existing, dict) else {}
        incoming = dict(patch) if isinstance(patch, dict) else {}
        merged = dict(base)
        merged.update(incoming)

        locked_type = object_memory_type or str(base.get('memory_type', '') or '')
        locked_layer = object_memory_layer or str(base.get('memory_layer', '') or '')
        if locked_type:
            merged['memory_type'] = locked_type
        if locked_layer:
            merged['memory_layer'] = locked_layer

        tags = ObjectStore._normalize_retrieval_tags(merged.get('retrieval_tags', []))
        if tags:
            merged['retrieval_tags'] = tags
        return merged

    def _build_new_object(self, proposal: Dict[str, Any], evidence_ids: List[str], obj_id: str) -> Dict[str, Any]:
        """Build canonical object payload for both ACCEPT_NEW and fallback branches."""
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        memory_type = proposal.get('memory_type', 'generic_object')
        memory_layer = proposal.get('memory_layer', 'semantic')
        retrieval_tags = self._normalize_retrieval_tags(proposal.get('retrieval_tags', []))
        memory_metadata = self._merge_memory_metadata(
            {},
            proposal.get('memory_metadata', {}),
            object_memory_type=str(memory_type),
            object_memory_layer=str(memory_layer),
        )
        if retrieval_tags:
            memory_metadata['retrieval_tags'] = retrieval_tags
        normalized_proposal = dict(proposal)
        normalized_proposal['evidence_ids'] = list(evidence_ids)
        normalized_proposal['created_at'] = created_at
        normalized_proposal['updated_at'] = created_at
        normalized_proposal['asset_status'] = AssetStatus.NEW_ASSET.value
        normalized_proposal['trigger_source'] = proposal.get('trigger_source', 'unknown')
        normalized_proposal['trigger_episode'] = proposal.get('trigger_episode', 0)
        normalized_proposal['consumption_count'] = 0
        normalized_proposal['last_consumed_tick'] = None
        normalized_proposal['reuse_history'] = []
        normalized_proposal['memory_type'] = memory_type
        normalized_proposal['memory_layer'] = memory_layer
        normalized_proposal['retrieval_tags'] = retrieval_tags
        normalized_proposal['memory_metadata'] = memory_metadata
        normalized_proposal['content_hash'] = proposal.get('content_hash', '')
        return proposal_to_object_record(
            normalized_proposal,
            object_id=obj_id,
            created_at=created_at,
            updated_at=created_at,
        )

    def add(self, proposal: Dict, decision: str, evidence_ids: List[str]) -> str:
        """
        Add or merge an object based on the governance decision.
        """
        if decision == ACCEPT_NEW:
            import uuid
            obj_id = str(uuid.uuid4())[:12]
            obj = self._build_new_object(proposal, evidence_ids, obj_id)
            self._objects[obj_id] = obj
            self._index_object(obj)
            return obj_id

        elif decision == MERGE_UPDATE_EXISTING:
            existing_id = proposal.get('existing_object_id')
            if existing_id and existing_id in self._objects:
                return self.merge_update(existing_id, evidence_ids, proposal.get('additional_content'))
            # Fallback to add new
            import uuid
            obj_id = str(uuid.uuid4())[:12]
            obj = self._build_new_object(proposal, evidence_ids, obj_id)
            self._objects[obj_id] = obj
            self._index_object(obj)
            return obj_id

        return ""

    def restore_records(self, records: Sequence[Dict[str, Any]], *, replace: bool = False) -> List[str]:
        """
        Restore previously persisted object records without re-validating proposals.

        Intended for bootstrap-time recovery from state snapshots produced by the
        formal write path, not as a second write authority.
        """
        if replace:
            self._objects = {}
            self._by_family = {}
            self._by_object_type = {}
            self._merge_history = {}

        restored: List[str] = []
        for record in list(records or []):
            if not isinstance(record, dict):
                continue
            object_id = str(record.get('object_id') or '').strip()
            if not object_id:
                continue
            obj = copy.deepcopy(record)
            self._objects[object_id] = obj
            self._index_object(obj)
            restored.append(object_id)
        return restored

    def retrieve(
        self,
        sort_by: str = 'confidence',
        limit: int = 50,
        *,
        object_type: Optional[str] = None,
        family: Optional[str] = None,
        min_surface_priority: Optional[float] = None,
        asset_status: Optional[Sequence[str] | str] = None,
    ) -> List[dict]:
        """Retrieve typed cognitive objects with optional typed filters."""
        objs = [o for o in self._objects.values() if o.get('status') != 'invalidated']
        normalized_object_type = str(object_type or '').strip()
        normalized_family = str(family or '').strip()
        if normalized_object_type:
            objs = [o for o in objs if str(o.get('object_type') or infer_object_type(o)) == normalized_object_type]
        if normalized_family:
            objs = [o for o in objs if str(o.get('family') or '').strip() == normalized_family]
        if min_surface_priority is not None:
            objs = [
                o for o in objs
                if float(o.get('surface_priority', 0.0) or 0.0) >= float(min_surface_priority)
            ]
        if asset_status is not None:
            if isinstance(asset_status, str):
                allowed_statuses = {asset_status}
            else:
                allowed_statuses = {
                    str(value).strip()
                    for value in asset_status
                    if str(value).strip()
                }
            objs = [o for o in objs if str(o.get('asset_status') or '') in allowed_statuses]
        if sort_by == 'confidence':
            objs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        elif sort_by == 'surface_priority':
            objs.sort(
                key=lambda x: (
                    float(x.get('surface_priority', 0.0) or 0.0),
                    float(x.get('confidence', 0.0) or 0.0),
                ),
                reverse=True,
            )
        return objs[:limit]

    def get(self, object_id: str) -> Optional[dict]:
        return self._objects.get(object_id)

    def iter_objects(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Read-only iteration over object records (returns detached copies)."""
        count = 0
        for obj in self._objects.values():
            if limit is not None and count >= limit:
                break
            yield copy.deepcopy(obj)
            count += 1

    def count_objects(self) -> int:
        """Return object count without exposing private storage."""
        return len(self._objects)

    def get_object_ids(self) -> List[str]:
        """Return all object IDs without exposing private storage internals."""
        return list(self._objects.keys())

    def snapshot_for_audit(self) -> Tuple[dict, ...]:
        """
        Return immutable audit snapshot.

        Tuple + deep-copied dicts avoids leaking mutable references.
        """
        return tuple(copy.deepcopy(obj) for obj in self._objects.values())

    def merge_update(self, object_id: str, new_evidence_ids: List[str], additional_content: Dict = None) -> str:
        """Merge new evidence into an existing object."""
        obj = self._objects.get(object_id)
        if not obj:
            return ""

        # Add evidence IDs
        existing = set(obj.get('evidence_ids', []))
        existing.update(new_evidence_ids)
        obj['evidence_ids'] = list(existing)

        # Update content if provided
        if additional_content:
            if 'summary' in additional_content:
                obj['summary'] = str(additional_content.get('summary') or obj.get('summary') or '')
            if 'family' in additional_content:
                obj['family'] = str(additional_content.get('family') or obj.get('family') or '')
            if 'applicability' in additional_content and isinstance(additional_content.get('applicability'), dict):
                obj['applicability'] = dict(additional_content.get('applicability', {}))
            if 'failure_conditions' in additional_content:
                obj['failure_conditions'] = self._normalize_retrieval_tags(additional_content.get('failure_conditions', []))
            if 'source_stage' in additional_content:
                obj['source_stage'] = str(additional_content.get('source_stage') or obj.get('source_stage') or '')
            if 'supersedes' in additional_content:
                obj['supersedes'] = self._normalize_retrieval_tags(additional_content.get('supersedes', []))
            if 'reopened_from' in additional_content:
                obj['reopened_from'] = str(additional_content.get('reopened_from') or obj.get('reopened_from') or '')
            if 'surface_priority' in additional_content:
                try:
                    obj['surface_priority'] = float(additional_content.get('surface_priority'))
                except (TypeError, ValueError):
                    pass
            if 'supporting_evidence' in additional_content:
                obj['supporting_evidence'] = self._normalize_retrieval_tags(additional_content.get('supporting_evidence', []))
            if 'contradicting_evidence' in additional_content:
                obj['contradicting_evidence'] = self._normalize_retrieval_tags(additional_content.get('contradicting_evidence', []))
            if 'hypothesis_type' in additional_content:
                obj['hypothesis_type'] = str(additional_content.get('hypothesis_type') or obj.get('hypothesis_type') or '')
            if 'posterior' in additional_content:
                try:
                    obj['posterior'] = float(additional_content.get('posterior'))
                except (TypeError, ValueError):
                    pass
            if 'support_count' in additional_content:
                try:
                    obj['support_count'] = int(additional_content.get('support_count'))
                except (TypeError, ValueError):
                    pass
            if 'contradiction_count' in additional_content:
                try:
                    obj['contradiction_count'] = int(additional_content.get('contradiction_count'))
                except (TypeError, ValueError):
                    pass
            if 'scope' in additional_content:
                obj['scope'] = str(additional_content.get('scope') or obj.get('scope') or '')
            if 'source' in additional_content:
                obj['source'] = str(additional_content.get('source') or obj.get('source') or '')
            if 'predictions' in additional_content and isinstance(additional_content.get('predictions'), dict):
                obj['predictions'] = copy.deepcopy(additional_content.get('predictions', {}))
            if 'falsifiers' in additional_content:
                obj['falsifiers'] = self._normalize_retrieval_tags(additional_content.get('falsifiers', []))
            if 'conflicts_with' in additional_content:
                obj['conflicts_with'] = self._normalize_retrieval_tags(additional_content.get('conflicts_with', []))
            if 'supporting_evidence_rows' in additional_content and isinstance(additional_content.get('supporting_evidence_rows'), list):
                obj['supporting_evidence_rows'] = [
                    copy.deepcopy(item)
                    for item in additional_content.get('supporting_evidence_rows', [])
                    if isinstance(item, dict)
                ]
            if 'contradicting_evidence_rows' in additional_content and isinstance(additional_content.get('contradicting_evidence_rows'), list):
                obj['contradicting_evidence_rows'] = [
                    copy.deepcopy(item)
                    for item in additional_content.get('contradicting_evidence_rows', [])
                    if isinstance(item, dict)
                ]
            if 'tags' in additional_content:
                obj['tags'] = self._normalize_retrieval_tags(additional_content.get('tags', []))
            if 'hypothesis_metadata' in additional_content and isinstance(additional_content.get('hypothesis_metadata'), dict):
                obj['hypothesis_metadata'] = copy.deepcopy(additional_content.get('hypothesis_metadata', {}))
            if 'source_family' in additional_content:
                obj['source_family'] = str(additional_content.get('source_family') or obj.get('source_family') or '')
            if 'target_family' in additional_content:
                obj['target_family'] = str(additional_content.get('target_family') or obj.get('target_family') or '')
            if 'reuse_evidence' in additional_content:
                obj['reuse_evidence'] = self._normalize_retrieval_tags(additional_content.get('reuse_evidence', []))
            content_patch = {
                k: v for k, v in additional_content.items()
                if k not in (
                    'retrieval_tags',
                    'memory_metadata',
                    'summary',
                    'family',
                    'applicability',
                    'failure_conditions',
                    'source_stage',
                    'supersedes',
                    'reopened_from',
                    'surface_priority',
                    'hypothesis_type',
                    'posterior',
                    'support_count',
                    'contradiction_count',
                    'scope',
                    'source',
                    'predictions',
                    'falsifiers',
                    'conflicts_with',
                    'supporting_evidence',
                    'contradicting_evidence',
                    'supporting_evidence_rows',
                    'contradicting_evidence_rows',
                    'tags',
                    'hypothesis_metadata',
                    'source_family',
                    'target_family',
                    'reuse_evidence',
                )
            }
            if content_patch and isinstance(obj.get('content'), dict):
                obj['content'].update(content_patch)
                obj['structured_payload'] = copy.deepcopy(obj['content'])

            if infer_object_type(obj) == 'hypothesis' and isinstance(obj.get('content'), dict):
                hypothesis_content_patch: Dict[str, Any] = {}
                for field_name in (
                    'hypothesis_type',
                    'posterior',
                    'support_count',
                    'contradiction_count',
                    'scope',
                    'source',
                    'falsifiers',
                    'conflicts_with',
                    'tags',
                ):
                    if field_name in additional_content:
                        hypothesis_content_patch[field_name] = copy.deepcopy(obj.get(field_name))
                if 'predictions' in additional_content:
                    predictions = copy.deepcopy(obj.get('predictions', {}))
                    hypothesis_content_patch['predictions'] = predictions
                    hypothesis_content_patch['predicted_action_effects'] = copy.deepcopy(
                        predictions.get('predicted_action_effects', {})
                    )
                    hypothesis_content_patch['predicted_action_effects_by_signature'] = copy.deepcopy(
                        predictions.get('predicted_action_effects_by_signature', {})
                    )
                    hypothesis_content_patch['predicted_observation_tokens'] = list(
                        predictions.get('predicted_observation_tokens', [])
                    )
                    hypothesis_content_patch['predicted_phase_shift'] = str(
                        predictions.get('predicted_phase_shift', '') or ''
                    )
                    hypothesis_content_patch['predicted_information_gain'] = float(
                        predictions.get('predicted_information_gain', 0.0) or 0.0
                    )
                if 'supporting_evidence_rows' in additional_content:
                    hypothesis_content_patch['supporting_evidence'] = copy.deepcopy(
                        obj.get('supporting_evidence_rows', [])
                    )
                if 'contradicting_evidence_rows' in additional_content:
                    hypothesis_content_patch['contradicting_evidence'] = copy.deepcopy(
                        obj.get('contradicting_evidence_rows', [])
                    )
                if 'hypothesis_metadata' in additional_content:
                    hypothesis_content_patch['metadata'] = copy.deepcopy(
                        obj.get('hypothesis_metadata', {})
                    )
                if hypothesis_content_patch:
                    obj['content'].update(hypothesis_content_patch)
                    obj['structured_payload'] = copy.deepcopy(obj['content'])

            merged_tags = self._normalize_retrieval_tags(
                self._normalize_retrieval_tags(obj.get('retrieval_tags', []))
                + self._normalize_retrieval_tags(additional_content.get('retrieval_tags', []))
            )
            if merged_tags:
                obj['retrieval_tags'] = merged_tags

            if 'memory_metadata' in additional_content:
                obj['memory_metadata'] = self._merge_memory_metadata(
                    obj.get('memory_metadata', {}),
                    additional_content.get('memory_metadata', {}),
                    object_memory_type=str(obj.get('memory_type', '')),
                    object_memory_layer=str(obj.get('memory_layer', '')),
                )
                if merged_tags:
                    obj['memory_metadata']['retrieval_tags'] = merged_tags

        lifecycle_result = apply_lifecycle_transition(
            status=self._current_lifecycle_status(obj),
            transition=TRANSITION_MERGE,
            history=list(obj.get('lifecycle_events', [])),
            reason='merge_update_existing',
            metadata={'object_id': object_id},
        )
        obj['status'] = lifecycle_result['status']
        obj['lifecycle_events'] = lifecycle_result['history']
        obj['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        obj['version'] = int(obj.get('version', 1) or 1) + 1
        self._index_object(obj)

        # Record merge
        if object_id not in self._merge_history:
            self._merge_history[object_id] = []
        self._merge_history[object_id].append(MergeRecord(
            object_id=object_id,
            merged_from_id="",
            evidence_ids_added=list(new_evidence_ids),
            additional_content=additional_content,
        ))

        return object_id

    # T1-P4 FIX: Field update whitelist for update_fields() API
    _UPDATE_FIELDS_ALLOWLIST = {
        'confidence', 'retrieval_tags', 'asset_status',
        'consumption_count', 'last_consumed_tick', 'reuse_history',
        'content', 'memory_metadata',  # content sub-fields can be updated
    }
    _UPDATE_FIELDS_FORBIDDEN = {
        'object_id', 'created_at', 'status', 'memory_type', 'memory_layer',
    }
    
    def update_fields(self, object_id: str, patch: Dict[str, Any], reason: str, evidence_ids: List[str]) -> str:
        """
        T1-P4 FIX: Formal update method for learning-driven updates with field constraints.
        
        Updates specific fields on an existing object without bypassing the store.
        Writes merge history for audit trail.
        
        Args:
            object_id: ID of object to update
            patch: Dict of fields to update (e.g., {'confidence': 0.9})
            reason: Reason for update (e.g., 'learning_adjustment')
            evidence_ids: Evidence supporting this update
        
        Returns:
            object_id if successful, "" if failed
        """
        obj = self._objects.get(object_id)
        if not obj:
            return ""
        
        # Check for forbidden fields - reject if any forbidden fields in patch
        forbidden_used = set(patch.keys()) & self._UPDATE_FIELDS_FORBIDDEN
        if forbidden_used:
            # Record rejection in merge history
            if object_id not in self._merge_history:
                self._merge_history[object_id] = []
            self._merge_history[object_id].append(MergeRecord(
                object_id=object_id,
                merged_from_id="learning_update_rejected",
                evidence_ids_added=list(evidence_ids),
                additional_content={'rejected_fields': list(forbidden_used), 'reason': 'forbidden_field'},
            ))
            return ""
        
        # Filter patch to only allowlisted fields
        allowed_patch = {k: v for k, v in patch.items() if k in self._UPDATE_FIELDS_ALLOWLIST}
        rejected_keys = set(patch.keys()) - self._UPDATE_FIELDS_ALLOWLIST
        if rejected_keys:
            # Record partial rejection in merge history
            if object_id not in self._merge_history:
                self._merge_history[object_id] = []
            self._merge_history[object_id].append(MergeRecord(
                object_id=object_id,
                merged_from_id="learning_update_partial",
                evidence_ids_added=list(evidence_ids),
                additional_content={'rejected_keys': list(rejected_keys)},
            ))
        
        if not allowed_patch:
            # Nothing allowed to update
            return ""
        
        # Apply allowed patch to object with explicit merge strategy
        patch_to_apply = dict(allowed_patch)
        if 'retrieval_tags' in patch_to_apply:
            patch_to_apply['retrieval_tags'] = self._normalize_retrieval_tags(
                self._normalize_retrieval_tags(obj.get('retrieval_tags', []))
                + self._normalize_retrieval_tags(patch_to_apply.get('retrieval_tags', []))
            )
        if 'memory_metadata' in patch_to_apply:
            patch_to_apply['memory_metadata'] = self._merge_memory_metadata(
                obj.get('memory_metadata', {}),
                patch_to_apply.get('memory_metadata', {}),
                object_memory_type=str(obj.get('memory_type', '')),
                object_memory_layer=str(obj.get('memory_layer', '')),
            )
            merged_tags = patch_to_apply.get('retrieval_tags', obj.get('retrieval_tags', []))
            if merged_tags:
                patch_to_apply['memory_metadata']['retrieval_tags'] = self._normalize_retrieval_tags(merged_tags)

        obj.update(patch_to_apply)
        
        # Merge evidence IDs
        existing = set(obj.get('evidence_ids', []))
        existing.update(evidence_ids)
        obj['evidence_ids'] = list(existing)
        
        # Record in merge history
        if object_id not in self._merge_history:
            self._merge_history[object_id] = []
        self._merge_history[object_id].append(MergeRecord(
            object_id=object_id,
            merged_from_id="learning_update",
            evidence_ids_added=list(evidence_ids),
            additional_content=patch_to_apply,
        ))
        
        return object_id

    def weaken(self, object_id: str, evidence_id: str) -> GovernanceDecision:
        """Weaken an object based on contradicting evidence."""
        obj = self._objects.get(object_id)
        if not obj:
            return GovernanceDecision(REJECT, "object not found")

        lifecycle_result = apply_lifecycle_transition(
            status=self._current_lifecycle_status(obj),
            transition=TRANSITION_WEAKEN,
            history=list(obj.get('lifecycle_events', [])),
            reason='contradicting_evidence',
            metadata={'evidence_id': evidence_id},
        )
        obj['status'] = lifecycle_result['status']
        obj['lifecycle_events'] = lifecycle_result['history']
        obj.get('evidence_ids', []).append(evidence_id)
        # Reduce confidence
        obj['confidence'] = max(0.1, obj.get('confidence', 0.5) - 0.1)
        obj['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%S")

        return GovernanceDecision(MERGE_UPDATE_EXISTING, "weakened", object_id)

    def invalidate(self, object_id: str) -> GovernanceDecision:
        """Invalidate an object (lifecycle terminal state)."""
        obj = self._objects.get(object_id)
        if not obj:
            return GovernanceDecision(REJECT, "object not found")

        lifecycle_result = apply_lifecycle_transition(
            status=self._current_lifecycle_status(obj),
            transition=TRANSITION_INVALIDATE,
            history=list(obj.get('lifecycle_events', [])),
            reason='formal_invalidation',
            metadata={'object_id': object_id},
        )
        obj['status'] = lifecycle_result['status']
        obj['lifecycle_events'] = lifecycle_result['history']
        obj['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return GovernanceDecision(MERGE_UPDATE_EXISTING, "invalidated", object_id)

    def find_existing(self, content_hash: str, novelty_hash: str = None) -> Optional[str]:
        """Find an existing object matching the given content hash."""
        for obj_id, obj in self._objects.items():
            if obj.get('status') == 'invalidated':
                continue
            obj_hash = obj.get('content_hash', '')
            if obj_hash == content_hash:
                return obj_id
        return None

    # P3-B3: Asset lifecycle methods

    def record_consumption(self, object_id: str, tick: int, episode: int, was_beneficial: bool = False) -> bool:
        """
        Record that an object was consumed in decision-making.
        
        Updates consumption_count, last_consumed_tick, reuse_history.
        Promotes asset_status if beneficial.
        
        Returns True if object was found and updated.
        """
        obj = self._objects.get(object_id)
        if not obj:
            return False
        
        obj['consumption_count'] = obj.get('consumption_count', 0) + 1
        obj['last_consumed_tick'] = tick
        
        reuse_entry = {'tick': tick, 'episode': episode, 'was_beneficial': was_beneficial}
        obj.setdefault('reuse_history', []).append(reuse_entry)
        
        # P3-B3: Promote asset status based on consumption
        self._promote_asset_status(obj, was_beneficial)
        return True
    
    def _promote_asset_status(self, obj: dict, was_beneficial: bool) -> None:
        """
        Promote asset status based on consumption pattern.
        
        Path: NEW_ASSET -> LIVE_ASSET -> REUSABLE_ASSET -> COMPILED_ASSET -> DISTILLED_ASSET
        Bad path: any -> GARBAGE
        """
        current = AssetStatus(obj.get('asset_status', AssetStatus.NEW_ASSET.value))
        count = obj.get('consumption_count', 0)
        
        if not was_beneficial and current != AssetStatus.GARBAGE:
            # Two consecutive non-beneficial uses -> garbage
            recent = obj.get('reuse_history', [])[-2:]
            non_beneficial = [r for r in recent if not r.get('was_beneficial', False)]
            if len(non_beneficial) >= 2:
                obj['asset_status'] = AssetStatus.GARBAGE.value
            return
        
        # Promote on beneficial consumption
        if was_beneficial:
            if current == AssetStatus.NEW_ASSET and count >= 1:
                obj['asset_status'] = AssetStatus.LIVE_ASSET.value
            elif current == AssetStatus.LIVE_ASSET and count >= 2:
                obj['asset_status'] = AssetStatus.REUSABLE_ASSET.value
            elif current == AssetStatus.REUSABLE_ASSET and count >= 3:
                obj['asset_status'] = AssetStatus.COMPILED_ASSET.value
    
    def promote_to_distilled(self, object_id: str) -> bool:
        """
        Promote an asset to DISTILLED_ASSET.
        
        Called when asset demonstrates value after teacher exit.
        
        Returns True if object was found and promoted.
        """
        obj = self._objects.get(object_id)
        if not obj:
            return False
        current = AssetStatus(obj.get('asset_status', AssetStatus.NEW_ASSET.value))
        if current in (AssetStatus.COMPILED_ASSET, AssetStatus.REUSABLE_ASSET, AssetStatus.LIVE_ASSET):
            obj['asset_status'] = AssetStatus.DISTILLED_ASSET.value
            return True
        return False
    
    def retire_to_garbage(self, object_id: str) -> bool:
        """
        Retire an asset to GARBAGE status.
        
        Called when asset is contradicted or has no utility.
        
        Returns True if object was found and retired.
        """
        obj = self._objects.get(object_id)
        if not obj:
            return False
        obj['asset_status'] = AssetStatus.GARBAGE.value
        return True
    
    def get_asset_status(self, object_id: str) -> Optional[str]:
        """Get current asset_status of an object."""
        obj = self._objects.get(object_id)
        return obj.get('asset_status') if obj else None
    
    def get_objects_by_status(self, status: AssetStatus) -> List[dict]:
        """Get all objects with a specific asset_status."""
        return [o for o in self._objects.values() if o.get('asset_status') == status.value]
    
    def get_live_assets(self) -> List[dict]:
        """Get all live/reusable/compiled/distilled assets (non-garbage)."""
        return [
            o for o in self._objects.values()
            if o.get('asset_status') not in (AssetStatus.GARBAGE.value, AssetStatus.NEW_ASSET.value)
        ]

    def get_by_object_type(self, object_type: str) -> List[dict]:
        normalized = str(object_type or '').strip()
        if not normalized:
            return []
        object_ids = self._by_object_type.get(normalized, [])
        return [
            self._objects[obj_id]
            for obj_id in object_ids
            if obj_id in self._objects and self._objects[obj_id].get('status') != 'invalidated'
        ]

    def get_by_family(self, family: str) -> List[dict]:
        normalized = str(family or '').strip()
        if not normalized:
            return []
        object_ids = self._by_family.get(normalized, [])
        return [
            self._objects[obj_id]
            for obj_id in object_ids
            if obj_id in self._objects and self._objects[obj_id].get('status') != 'invalidated'
        ]

    def query_objects(
        self,
        *,
        object_type: Optional[str] = None,
        family: Optional[str] = None,
        min_surface_priority: Optional[float] = None,
        asset_status: Optional[Sequence[str] | str] = None,
        limit: int = 50,
        sort_by: str = 'confidence',
    ) -> List[dict]:
        return self.retrieve(
            sort_by=sort_by,
            limit=limit,
            object_type=object_type,
            family=family,
            min_surface_priority=min_surface_priority,
            asset_status=asset_status,
        )

    def retrieve_with_bundle(self, bundle) -> List[dict]:
        """
        Apply RetrievalBundle preferences through the formal object store.

        The bundle enriches retrieval; it does not create a second memory path.
        """
        from modules.memory.retrieval_bundle import RetrievalBundleFilter

        max_results = int(getattr(bundle, 'max_results', 10) or 10)
        candidates = self.retrieve(sort_by='confidence', limit=max(max_results * 10, 50))
        filtered = RetrievalBundleFilter(bundle).filter(candidates)
        return filtered[:max_results]
    
    # Phase 3: Memory-type filtering helpers
    
    def get_by_memory_type(self, memory_type: str) -> List[dict]:
        """
        Get all objects with a specific memory_type.
        
        Args:
            memory_type: e.g., 'episode_record', 'fact_card', 'skill_card'
        
        Returns:
            List of objects with that memory_type
        """
        return [
            o for o in self._objects.values()
            if o.get('memory_type') == memory_type or o.get('type') == memory_type
        ]
    
    def get_by_memory_layer(self, memory_layer: str) -> List[dict]:
        """
        Get all objects with a specific memory_layer.
        
        Args:
            memory_layer: e.g., 'episodic', 'semantic', 'procedural'
        
        Returns:
            List of objects with that memory_layer
        """
        return [
            o for o in self._objects.values()
            if o.get('memory_layer') == memory_layer
        ]
    
    def find_conflicts(self, memory_type: str, retrieval_tags: List[str]) -> List[dict]:
        """
        Find objects that conflict with a candidate.
        
        Looks for:
        - Same memory_type with contradicting content
        - Same retrieval_tags with different conclusions
        
        Args:
            memory_type: Candidate's memory_type
            retrieval_tags: Candidate's retrieval tags
        
        Returns:
            List of conflicting objects
        """
        conflicts = []
        for obj in self._objects.values():
            # Check same memory_type
            if obj.get('memory_type') == memory_type or obj.get('type') == memory_type:
                # Check for contradiction in content
                obj_tags = obj.get('retrieval_tags', [])
                shared_tags = set(retrieval_tags) & set(obj_tags)
                if shared_tags:
                    # Same tags but could be contradictory
                    conflicts.append(obj)
        return conflicts
    
    def get_memory_summary(self) -> dict:
        """
        Get summary of all memory objects by type and layer.
        
        Returns:
            Dict with counts by memory_type, object_type, and memory_layer
        """
        type_counts = {}
        object_type_counts = {}
        layer_counts = {}
        total = 0
        
        for obj in self._objects.values():
            total += 1
            mt = obj.get('memory_type', obj.get('type', 'unknown'))
            ot = obj.get('object_type', infer_object_type(obj))
            ml = obj.get('memory_layer', 'unknown')
            type_counts[mt] = type_counts.get(mt, 0) + 1
            object_type_counts[ot] = object_type_counts.get(ot, 0) + 1
            layer_counts[ml] = layer_counts.get(ml, 0) + 1
        
        return {
            'total': total,
            'by_object_type': object_type_counts,
            'by_memory_type': type_counts,
            'by_memory_layer': layer_counts,
        }


class ProposalValidator:
    """
    5-gate validator for object proposals.
    
    Gates:
    1. Schema validation
    2. Leak detection (L1/L2/L3 leak types)
    3. Binding validation
    4. Content quality
    5. Merge decision (checks find_existing)
    """
    def __init__(self, agent_id: str, object_store: ObjectStore, leak_detection_level: str = 'heuristic'):
        self.agent_id = agent_id
        self._store = object_store
        normalized_level = str(leak_detection_level or 'heuristic').strip().lower()
        self._leak_detection_level = normalized_level if normalized_level in ('heuristic', 'strict') else 'heuristic'

    def _strict_leak_check(self, proposal: Dict) -> Optional[str]:
        """Strict gate placeholder: denylist + schema + context checks (composed)."""
        content = proposal.get('content', {}) if isinstance(proposal.get('content', {}), dict) else {}
        content_str = str(content).lower()

        # 1) denylist
        denylist_terms = ('password', 'api_key', 'private_key', 'secret', 'hidden')
        if any(term in content_str for term in denylist_terms):
            return 'denylist match'

        # 2) schema presence sanity (placeholder)
        if isinstance(content, dict) and not any(k in content for k in ('summary', 'assertion', 'tool_args', 'card_id')):
            return 'schema/content shape mismatch'

        # 3) context checks (placeholder)
        context = proposal.get('context', {}) if isinstance(proposal.get('context', {}), dict) else {}
        if context.get('source_visibility') == 'private':
            return 'private context visibility'

        return None

    def _detect_contradiction(self, proposal: Dict) -> Optional[Dict]:
        """Detect lightweight object-level contradictions for adjudication."""
        content = proposal.get('content', {}) if isinstance(proposal.get('content', {}), dict) else {}
        retrieval_tags = proposal.get('retrieval_tags', []) if isinstance(proposal.get('retrieval_tags', []), list) else []
        memory_type = proposal.get('memory_type', proposal.get('type', ''))
        conflicts = []
        for obj in self._store.retrieve(sort_by='confidence', limit=200):
            obj_type = obj.get('memory_type', obj.get('type', ''))
            if memory_type and obj_type and obj_type != memory_type:
                continue
            obj_tags = obj.get('retrieval_tags', []) if isinstance(obj.get('retrieval_tags', []), list) else []
            if retrieval_tags and not (set(retrieval_tags) & set(obj_tags)):
                continue
            conflicts.append(obj)
        if not conflicts:
            return None

        proposed_neg = str(content.get('negated', '')).lower() in ('true', '1', 'yes')
        proposed_assertion = str(content.get('assertion', content.get('summary', ''))).strip().lower()
        proposed_object_id = str(proposal.get('object_id') or '').strip()
        for obj in conflicts[:10]:
            obj_content = obj.get('content', {}) if isinstance(obj.get('content', {}), dict) else {}
            obj_neg = str(obj_content.get('negated', '')).lower() in ('true', '1', 'yes')
            obj_assertion = str(obj_content.get('assertion', obj_content.get('summary', ''))).strip().lower()
            same_assertion = proposed_assertion and obj_assertion and proposed_assertion == obj_assertion
            opposite_polarity = proposed_neg != obj_neg
            contradicts_object_id = str(obj_content.get('contradicts_object_id') or '').strip()
            explicit_conflict = bool(
                contradicts_object_id and proposed_object_id and contradicts_object_id == proposed_object_id
            )
            if (same_assertion and opposite_polarity) or explicit_conflict:
                return obj
        return None

    def validate(self, proposal: Dict) -> GovernanceDecision:
        """Run all 5 gates and return governance decision.
        
        Accepts both Dict (proposal format) and EvidencePacket dataclass.
        """
        # Handle EvidencePacket dataclass
        if hasattr(proposal, 'content') and hasattr(proposal, 'evidence_id'):
            # This is an EvidencePacket — convert to dict format
            proposal = {
                'content': proposal.content if isinstance(proposal.content, dict) else {},
                'confidence': proposal.confidence,
                'evidence_id': proposal.evidence_id,
                'evidence_kind': getattr(proposal, 'evidence_kind', ''),
                'source_action': getattr(proposal, 'source_action', None),
            }

        # Gate 2: Leak detection
        leak_mode = str(proposal.get('leak_detection_level') or self._leak_detection_level or 'heuristic').lower()
        if leak_mode not in ('heuristic', 'strict'):
            leak_mode = 'heuristic'

        content_str = str(proposal.get('content', {}))
        lowered = content_str.lower()
        leak_pattern_hit = ('secret' in lowered or 'hidden' in lowered)
        if leak_mode == 'heuristic' and leak_pattern_hit:
            leak_type = proposal.get('leak_type') or 'heuristic_content_leak'
            return GovernanceDecision(
                REJECT,
                "gate2[heuristic]: leak detected",
                bypass_leak_check=True,
                leak_type=leak_type,
                leak_detection_level='heuristic',
                leak_gate_mode='heuristic',
            )

        if leak_mode == 'strict':
            strict_hit_reason = self._strict_leak_check(proposal)
            if strict_hit_reason:
                leak_type = proposal.get('leak_type') or 'strict_composed_guard'
                return GovernanceDecision(
                    REJECT,
                    f"gate2[strict]: leak detected ({strict_hit_reason})",
                    bypass_leak_check=True,
                    leak_type=leak_type,
                    leak_detection_level='strict',
                    leak_gate_mode='strict',
                )

        # Gate 3: Binding validation (simplified — check function_name exists)
        content = proposal.get('content', {})
        tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
        fn = (
            tool_args.get('function_name', '')
            or content.get('function_name', '')
            or content.get('card_id', '')
            or content.get('origin_family', '')
        )
        if not fn and not content:
            return GovernanceDecision(
                REJECT,
                "gate3: binding validation failed — empty content",
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )

        # Gate 4: Content quality (confidence threshold)
        conf = proposal.get('confidence', 0.5)
        if conf < 0.1:
            return GovernanceDecision(
                REJECT,
                "gate4: content quality failed — confidence too low",
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )

        # Gate 4.5: contradiction detection / object-level adjudication
        contradiction = self._detect_contradiction(proposal)
        if contradiction:
            return GovernanceDecision(
                MERGE_UPDATE_EXISTING,
                "gate4.5: contradiction detected — route to adjudicated merge/split workflow",
                object_id=contradiction.get('object_id'),
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )

        # Gate 5: Merge decision — check if object already exists
        memory_type = str(
            proposal.get('memory_type', proposal.get('type', '')) or ''
        ).strip()
        memory_layer = str(proposal.get('memory_layer', '') or '').strip()
        if (
            memory_type == MemoryType.EPISODE_RECORD.value
            or memory_layer == MemoryLayer.EPISODIC.value
        ):
            return GovernanceDecision(
                ACCEPT_NEW,
                "gate5: episodic memories stay append-only",
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )
        content_hash = proposal.get('content_hash', str(proposal.get('content', {})))
        existing = self._store.find_existing(content_hash)
        if existing:
            return GovernanceDecision(
                MERGE_UPDATE_EXISTING,
                "gate5: merge with existing",
                existing,
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )
        else:
            return GovernanceDecision(
                ACCEPT_NEW,
                "all gates passed — accept new",
                leak_detection_level=leak_mode,
                leak_gate_mode=leak_mode,
            )

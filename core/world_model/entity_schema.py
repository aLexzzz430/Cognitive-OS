from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


@dataclass
class StateSlot:
    name: str
    value: Any
    confidence: float = 1.0
    observed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name or ""),
            "value": self.value,
            "confidence": float(self.confidence or 0.0),
            "observed": bool(self.observed),
        }


@dataclass
class EntityNode:
    entity_id: str
    entity_type: str
    semantic_labels: List[str] = field(default_factory=list)
    role_labels: List[str] = field(default_factory=list)
    salience_score: float = 0.0
    actionable_score: float = 0.0
    bbox: Dict[str, Any] = field(default_factory=dict)
    centroid: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)
    state_slots: List[StateSlot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["state_slots"] = [slot.to_dict() for slot in self.state_slots]
        return payload


@dataclass
class RelationEdge:
    relation_id: str
    relation_type: str
    source_entity_id: str
    target_entity_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": str(self.relation_id or ""),
            "relation_type": str(self.relation_type or ""),
            "source_entity_id": str(self.source_entity_id or ""),
            "target_entity_id": str(self.target_entity_id or ""),
            "attributes": _as_dict(self.attributes),
        }


@dataclass
class ObjectGraphState:
    world_state_signature: str
    scene_state: Dict[str, Any] = field(default_factory=dict)
    objects: List[EntityNode] = field(default_factory=list)
    relations: List[RelationEdge] = field(default_factory=list)
    intervention_targets: List[str] = field(default_factory=list)
    focus_object_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_state_signature": str(self.world_state_signature or ""),
            "scene_state": _as_dict(self.scene_state),
            "objects": [row.to_dict() for row in self.objects],
            "relations": [row.to_dict() for row in self.relations],
            "intervention_targets": [str(item or "") for item in self.intervention_targets if str(item or "")],
            "focus_object_ids": [str(item or "") for item in self.focus_object_ids if str(item or "")],
            "object_count": len(self.objects),
            "relation_count": len(self.relations),
        }

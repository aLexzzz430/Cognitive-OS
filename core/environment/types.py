from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GenericEntity:
    entity_id: str
    entity_type: str
    label: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    bbox_or_region: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericRelation:
    relation_id: str
    relation_type: str
    source_entity_id: str
    target_entity_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericActionDescriptor:
    name: str
    action_family: str
    parameter_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    source: str = "environment"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericActionEnvelope:
    action_name: str
    action_family: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_entity_id: str = ""
    target_family: str = ""
    native_action: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericTaskSpec:
    task_id: str
    environment_family: str
    instruction: str = ""
    success_criteria: List[str] = field(default_factory=list)
    available_action_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericObservation:
    observation_id: str
    environment_family: str
    task_id: str
    text: str = ""
    entities: List[GenericEntity] = field(default_factory=list)
    relations: List[GenericRelation] = field(default_factory=list)
    available_actions: List[GenericActionDescriptor] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericStateDelta:
    entity_deltas: List[Dict[str, Any]] = field(default_factory=list)
    relation_deltas: List[Dict[str, Any]] = field(default_factory=list)
    changed_regions: List[Dict[str, Any]] = field(default_factory=list)
    feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericTransition:
    task_id: str
    environment_family: str
    action: GenericActionEnvelope
    before_state_signature: Dict[str, Any] = field(default_factory=dict)
    after_state_signature: Dict[str, Any] = field(default_factory=dict)
    state_delta: GenericStateDelta = field(default_factory=GenericStateDelta)
    reward: float = 0.0
    terminal: bool = False
    success: bool = False
    feedback: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

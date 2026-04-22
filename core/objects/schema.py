from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Type, TypeVar, Union


OBJECT_TYPE_REPRESENTATION = "representation"
OBJECT_TYPE_HYPOTHESIS = "hypothesis"
OBJECT_TYPE_DISCRIMINATING_TEST = "discriminating_test"
OBJECT_TYPE_SKILL = "skill"
OBJECT_TYPE_TRANSFER = "transfer"
OBJECT_TYPE_IDENTITY = "identity"
OBJECT_TYPE_AUTOBIOGRAPHICAL = "autobiographical"

ALL_COGNITIVE_OBJECT_TYPES = (
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_HYPOTHESIS,
    OBJECT_TYPE_DISCRIMINATING_TEST,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
)


@dataclass
class CognitiveObjectBase:
    object_id: str = ""
    object_type: str = OBJECT_TYPE_REPRESENTATION
    family: str = ""
    summary: str = ""
    structured_payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    status: str = "qualified"
    applicability: Dict[str, Any] = field(default_factory=dict)
    failure_conditions: List[str] = field(default_factory=list)
    source_stage: str = ""
    commit_epoch: int = 0
    version: int = 1
    supersedes: List[str] = field(default_factory=list)
    reopened_from: str = ""
    lifecycle_events: List[Dict[str, Any]] = field(default_factory=list)
    surface_priority: float = 0.0
    asset_status: str = "new_asset"
    memory_type: str = ""
    memory_layer: str = ""
    retrieval_tags: List[str] = field(default_factory=list)
    memory_metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    trigger_source: str = ""
    trigger_episode: int = 0
    consumption_count: int = 0
    last_consumed_tick: Any = None
    reuse_history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def field_names(cls) -> set[str]:
        return {field_info.name for field_info in fields(cls)}

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "CognitiveObjectBase":
        payload = dict(record) if isinstance(record, dict) else {}
        if "structured_payload" not in payload:
            content = payload.get("content", {})
            payload["structured_payload"] = copy.deepcopy(content) if isinstance(content, dict) else {}
        filtered = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key in cls.field_names()
        }
        return cls(**filtered)

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["content"] = copy.deepcopy(self.structured_payload)
        return record


@dataclass
class RepresentationObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_REPRESENTATION
    surface_priority: float = 0.65


@dataclass
class HypothesisObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_HYPOTHESIS
    hypothesis_type: str = "generic"
    posterior: float = 0.0
    support_count: int = 0
    contradiction_count: int = 0
    scope: str = "local"
    source: str = "workspace"
    predictions: Dict[str, Any] = field(default_factory=dict)
    falsifiers: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    supporting_evidence_rows: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence_rows: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    hypothesis_metadata: Dict[str, Any] = field(default_factory=dict)
    surface_priority: float = 0.55


@dataclass
class DiscriminatingTestObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_DISCRIMINATING_TEST
    test_spec: Dict[str, Any] = field(default_factory=dict)
    surface_priority: float = 0.7


@dataclass
class SkillObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_SKILL
    skill_kind: str = ""
    surface_priority: float = 0.8


@dataclass
class TransferObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_TRANSFER
    source_family: str = ""
    target_family: str = ""
    reuse_evidence: List[str] = field(default_factory=list)
    surface_priority: float = 0.75


@dataclass
class IdentityObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_IDENTITY
    identity_profile: Dict[str, Any] = field(default_factory=dict)
    surface_priority: float = 0.95


@dataclass
class AutobiographicalObject(CognitiveObjectBase):
    object_type: str = OBJECT_TYPE_AUTOBIOGRAPHICAL
    episode_refs: List[str] = field(default_factory=list)
    continuity_markers: Dict[str, Any] = field(default_factory=dict)
    surface_priority: float = 0.9


AnyCognitiveObject = Union[
    RepresentationObject,
    HypothesisObject,
    DiscriminatingTestObject,
    SkillObject,
    TransferObject,
    IdentityObject,
    AutobiographicalObject,
]

T = TypeVar("T", bound=CognitiveObjectBase)


def clone_object(obj: T) -> T:
    return type(obj).from_record(obj.to_record())


def is_cognitive_object_type(value: Any) -> bool:
    return str(value or "").strip() in ALL_COGNITIVE_OBJECT_TYPES

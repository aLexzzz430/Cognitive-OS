from __future__ import annotations

from typing import Dict, Type

from core.objects.schema import (
    ALL_COGNITIVE_OBJECT_TYPES,
    AutobiographicalObject,
    CognitiveObjectBase,
    DiscriminatingTestObject,
    HypothesisObject,
    IdentityObject,
    RepresentationObject,
    SkillObject,
    TransferObject,
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_DISCRIMINATING_TEST,
    OBJECT_TYPE_HYPOTHESIS,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
)


COGNITIVE_OBJECT_CLASS_REGISTRY: Dict[str, Type[CognitiveObjectBase]] = {
    OBJECT_TYPE_REPRESENTATION: RepresentationObject,
    OBJECT_TYPE_HYPOTHESIS: HypothesisObject,
    OBJECT_TYPE_DISCRIMINATING_TEST: DiscriminatingTestObject,
    OBJECT_TYPE_SKILL: SkillObject,
    OBJECT_TYPE_TRANSFER: TransferObject,
    OBJECT_TYPE_IDENTITY: IdentityObject,
    OBJECT_TYPE_AUTOBIOGRAPHICAL: AutobiographicalObject,
}


def resolve_object_class(object_type: str) -> Type[CognitiveObjectBase]:
    normalized = str(object_type or "").strip().lower()
    return COGNITIVE_OBJECT_CLASS_REGISTRY.get(normalized, RepresentationObject)


def registered_object_types() -> tuple[str, ...]:
    return tuple(ALL_COGNITIVE_OBJECT_TYPES)


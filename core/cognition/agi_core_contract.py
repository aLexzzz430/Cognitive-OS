from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


AGI_CORE_CONTRACT_VERSION = "conos.agi_core_contract/v1"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class CognitiveGoal:
    goal_id: str
    statement: str
    priority: float = 0.5
    source: str = "unknown"
    constraints: list[str] = field(default_factory=list)
    success_signals: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CognitiveGoal":
        values = dict(payload)
        return cls(
            goal_id=str(values.get("goal_id") or values.get("id") or ""),
            statement=str(values.get("statement") or values.get("description") or ""),
            priority=float(values.get("priority", 0.5) or 0.5),
            source=str(values.get("source") or "unknown"),
            constraints=_string_list(values.get("constraints")),
            success_signals=_string_list(values.get("success_signals") or values.get("successSignals")),
            metadata=_mapping(values.get("metadata")),
        )


@dataclass
class CognitiveSituation:
    situation_id: str
    observations: list[Dict[str, Any]] = field(default_factory=list)
    salient_entities: list[str] = field(default_factory=list)
    uncertainty: float = 0.5
    self_state_refs: list[str] = field(default_factory=list)
    world_state_refs: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload


@dataclass
class CognitiveHypothesis:
    hypothesis_id: str
    claim: str
    posterior: float = 0.5
    predictions: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload


@dataclass
class CognitiveExperiment:
    experiment_id: str
    question: str
    discriminates_between: list[str] = field(default_factory=list)
    expected_observations: Dict[str, Any] = field(default_factory=dict)
    information_gain: float = 0.0
    risk: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload


@dataclass
class ActionIntent:
    intent_id: str
    verb: str
    target: str = ""
    expected_effect: str = ""
    capability_required: list[str] = field(default_factory=list)
    risk: float = 0.0
    reversible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload


@dataclass
class CognitiveOutcome:
    outcome_id: str
    action_intent_id: str
    observations: list[Dict[str, Any]] = field(default_factory=list)
    verified: bool = False
    success: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    learning_updates: list[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = AGI_CORE_CONTRACT_VERSION
        return payload


@dataclass
class CognitiveCycleFrame:
    frame_id: str
    goal: CognitiveGoal
    situation: CognitiveSituation
    hypotheses: list[CognitiveHypothesis] = field(default_factory=list)
    experiments: list[CognitiveExperiment] = field(default_factory=list)
    action_intents: list[ActionIntent] = field(default_factory=list)
    outcomes: list[CognitiveOutcome] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": AGI_CORE_CONTRACT_VERSION,
            "frame_id": self.frame_id,
            "goal": self.goal.to_dict(),
            "situation": self.situation.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "experiments": [item.to_dict() for item in self.experiments],
            "action_intents": [item.to_dict() for item in self.action_intents],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "metadata": dict(self.metadata),
        }


DOMAIN_SPECIFIC_TERMS = (
    "local_machine",
    "local-machine",
    "mirror",
    "patch",
    "pytest",
    "repo_tree",
    "file_read",
    "run_test",
    "apply_patch",
)


def validate_domain_neutral_contract(payload: Mapping[str, Any]) -> list[str]:
    """Return contract violations for domain terms leaking into AGI core fields."""

    text = repr(dict(payload)).lower()
    return [term for term in DOMAIN_SPECIFIC_TERMS if term in text]

#!/usr/bin/env python3
from __future__ import annotations

"""
representations/schema.py

RepresentationCard dataclass definition.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


# ============================================================
# Lifecycle Status
# ============================================================

LifecycleStatus = Literal["candidate", "active", "suppressed", "retired", "rejected", "experimental"]

OriginType = Literal["seed", "derived"]

CardFamily = Literal["situation", "pattern", "anomaly", "constraint"]

SemanticClass = Literal["A", "B", "C"]
# A = true middle representation
# B = surface label (direct observation readout)
# C = strategy constraint / governance bias

ActivationFunction = Literal["threshold", "trend", "composite"]


# ============================================================
# Structural Signature
# ============================================================

@dataclass
class ThresholdCondition:
    """Single threshold condition for card activation."""
    observation_key: str           # e.g., "signal", "hazard_exposure"
    operator: str                 # "gt", "lt", "gte", "lte", "eq"
    value: float
    description: str = ""        # human-readable


@dataclass
class StructuralSignature:
    """Describes what observations this card responds to."""
    observation_keys: list[str] = field(default_factory=list)
    threshold_conditions: list[ThresholdCondition] = field(default_factory=list)
    context_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "observation_keys": self.observation_keys,
            "threshold_conditions": [
                {"observation_key": c.observation_key, "operator": c.operator,
                 "value": c.value, "description": c.description}
                for c in self.threshold_conditions
            ],
            "context_requirements": self.context_requirements,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralSignature":
        conds = [
            ThresholdCondition(
                observation_key=c["observation_key"],
                operator=c["operator"],
                value=c["value"],
                description=c.get("description", ""),
            )
            for c in d.get("threshold_conditions", [])
        ]
        return cls(
            observation_keys=d.get("observation_keys", []),
            threshold_conditions=conds,
            context_requirements=d.get("context_requirements", []),
        )


# ============================================================
# Scope
# ============================================================

@dataclass
class CardScope:
    """Defines when this card is valid/invalid."""
    valid_in_regimes: list[str] = field(default_factory=list)
    invalid_in_regimes: list[str] = field(default_factory=list)
    planner_styles: list[str] = field(default_factory=list)  # or ["any"]

    def to_dict(self) -> dict:
        return {
            "valid_in_regimes": self.valid_in_regimes,
            "invalid_in_regimes": self.invalid_in_regimes,
            "planner_styles": self.planner_styles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CardScope":
        return cls(
            valid_in_regimes=d.get("valid_in_regimes", []),
            invalid_in_regimes=d.get("invalid_in_regimes", []),
            planner_styles=d.get("planner_styles", []),
        )


# ============================================================
# Evidence
# ============================================================

@dataclass
class ExampleRecord:
    """Single supporting or refuting example."""
    tick: int
    observation: dict
    outcome: dict
    description: str = ""


@dataclass
class Evidence:
    """Evidence supporting or refuting this representation."""
    positive_examples: list[ExampleRecord] = field(default_factory=list)
    negative_examples: list[ExampleRecord] = field(default_factory=list)
    support_count: int = 0
    counterexample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "positive_examples": [
                {"tick": e.tick, "observation": e.observation,
                 "outcome": e.outcome, "description": e.description}
                for e in self.positive_examples
            ],
            "negative_examples": [
                {"tick": e.tick, "observation": e.observation,
                 "outcome": e.outcome, "description": e.description}
                for e in self.negative_examples
            ],
            "support_count": self.support_count,
            "counterexample_count": self.counterexample_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        pos = [
            ExampleRecord(
                tick=e["tick"], observation=e["observation"],
                outcome=e["outcome"], description=e.get("description", ""),
            )
            for e in d.get("positive_examples", [])
        ]
        neg = [
            ExampleRecord(
                tick=e["tick"], observation=e["observation"],
                outcome=e["outcome"], description=e.get("description", ""),
            )
            for e in d.get("negative_examples", [])
        ]
        return cls(
            positive_examples=pos,
            negative_examples=neg,
            support_count=d.get("support_count", 0),
            counterexample_count=d.get("counterexample_count", 0),
        )


# ============================================================
# Activation Logic
# ============================================================

@dataclass
class ActivationLogic:
    """How to compute activation from observations."""
    function: ActivationFunction = "threshold"   # threshold | trend | composite
    parameters: dict = field(default_factory=dict)  # function-specific params

    def to_dict(self) -> dict:
        return {"function": self.function, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationLogic":
        return cls(
            function=d.get("function", "threshold"),
            parameters=d.get("parameters", {}),
        )


# ============================================================
# Planner Effects
# ============================================================

@dataclass
class PlannerEffects:
    """How this card influences the planner (advisory only)."""
    candidate_augmentation: list[str] = field(default_factory=list)  # extra candidates to add
    weight_adjustments: dict = field(default_factory=dict)           # {"advance": 0.1, ...}
    attention_shift: list[str] = field(default_factory=list)          # observation keys to weight

    def to_dict(self) -> dict:
        return {
            "candidate_augmentation": self.candidate_augmentation,
            "weight_adjustments": self.weight_adjustments,
            "attention_shift": self.attention_shift,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlannerEffects":
        return cls(
            candidate_augmentation=d.get("candidate_augmentation", []),
            weight_adjustments=d.get("weight_adjustments", {}),
            attention_shift=d.get("attention_shift", []),
        )


# ============================================================
# Failure Modes
# ============================================================

@dataclass
class FailureModes:
    """Known failure conditions for this card."""
    false_positive_conditions: list[str] = field(default_factory=list)
    false_negative_conditions: list[str] = field(default_factory=list)
    suppression_triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "false_positive_conditions": self.false_positive_conditions,
            "false_negative_conditions": self.false_negative_conditions,
            "suppression_triggers": self.suppression_triggers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureModes":
        return cls(
            false_positive_conditions=d.get("false_positive_conditions", []),
            false_negative_conditions=d.get("false_negative_conditions", []),
            suppression_triggers=d.get("suppression_triggers", []),
        )


# ============================================================
# Lifecycle
# ============================================================

@dataclass
class CardLifecycle:
    """Runtime lifecycle state of the card."""
    created_at_tick: int = 0
    times_activated: int = 0
    times_helpful: int = 0
    times_harmful: int = 0
    last_support_tick: int = 0
    last_activated_tick: int = 0
    current_status: LifecycleStatus = "candidate"

    def to_dict(self) -> dict:
        return {
            "created_at_tick": self.created_at_tick,
            "times_activated": self.times_activated,
            "times_helpful": self.times_helpful,
            "times_harmful": self.times_harmful,
            "last_support_tick": self.last_support_tick,
            "last_activated_tick": self.last_activated_tick,
            "current_status": self.current_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CardLifecycle":
        return cls(
            created_at_tick=d.get("created_at_tick", 0),
            times_activated=d.get("times_activated", 0),
            times_helpful=d.get("times_helpful", 0),
            times_harmful=d.get("times_harmful", 0),
            last_support_tick=d.get("last_support_tick", 0),
            last_activated_tick=d.get("last_activated_tick", 0),
            current_status=d.get("current_status", "candidate"),
        )


# ============================================================
# RepresentationCard (Core Object)
# ============================================================

@dataclass
class RepresentationCard:
    """
    A mid-level representation that captures a recurring pattern,
    situation, or anomaly in the agent's experience.

    This is a READ-ONLY schema object stored in the card warehouse.
    Runtime state (activation, lifecycle status) lives in representation_context.
    Evidence/lifecycle updates persist in RuntimeUpdateStore.
    """

    rep_id: str
    name: str
    origin_type: OriginType                       # "seed" or "derived"
    family: CardFamily                           # situation | pattern | anomaly | constraint
    summary: str                                # one-line human-readable description
    structural_signature: StructuralSignature
    scope: CardScope
    evidence: Evidence
    activation_logic: ActivationLogic
    planner_effects: PlannerEffects
    failure_modes: FailureModes
    lifecycle: CardLifecycle
    semantic_class: SemanticClass = "A"          # A | B | C semantic tier
    experimental_note: Optional[str] = None     # explanation if C-class or non-standard

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "rep_id": self.rep_id,
            "name": self.name,
            "origin_type": self.origin_type,
            "family": self.family,
            "summary": self.summary,
            "semantic_class": self.semantic_class,
            "experimental_note": self.experimental_note,
            "structural_signature": self.structural_signature.to_dict(),
            "scope": self.scope.to_dict(),
            "evidence": self.evidence.to_dict(),
            "activation_logic": self.activation_logic.to_dict(),
            "planner_effects": self.planner_effects.to_dict(),
            "failure_modes": self.failure_modes.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RepresentationCard":
        return cls(
            rep_id=d["rep_id"],
            name=d["name"],
            origin_type=d["origin_type"],
            family=d["family"],
            summary=d["summary"],
            semantic_class=d.get("semantic_class", "A"),
            experimental_note=d.get("experimental_note"),
            structural_signature=StructuralSignature.from_dict(d.get("structural_signature", {})),
            scope=CardScope.from_dict(d.get("scope", {})),
            evidence=Evidence.from_dict(d.get("evidence", {})),
            activation_logic=ActivationLogic.from_dict(d.get("activation_logic", {})),
            planner_effects=PlannerEffects.from_dict(d.get("planner_effects", {})),
            failure_modes=FailureModes.from_dict(d.get("failure_modes", {})),
            lifecycle=CardLifecycle.from_dict(d.get("lifecycle", {})),
        )

    # ---- Convenience ----

    def is_active(self) -> bool:
        return self.lifecycle.current_status == "active"

    def is_suppressed(self) -> bool:
        return self.lifecycle.current_status == "suppressed"

    def is_retired(self) -> bool:
        return self.lifecycle.current_status in ("retired", "rejected")

    def activation_score_description(self, score: float) -> str:
        """Human-readable description of activation score."""
        if score >= 0.8:
            return f"strong match ({score:.2f})"
        elif score >= 0.5:
            return f"moderate match ({score:.2f})"
        elif score >= 0.3:
            return f"weak match ({score:.2f})"
        else:
            return f"no significant match ({score:.2f})"

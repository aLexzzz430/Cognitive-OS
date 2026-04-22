"""
decision/utility_schema.py

Core decision data structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class CandidateSource(Enum):
    """候选来源"""
    BASE_GENERATION = "base_generation"
    SKILL_REWRITE = "skill_rewrite"
    LLM_REWRITE = "llm_rewrite"
    ARM_EVALUATION = "arm_evaluation"
    RECOVERY = "recovery"
    PROBE = "probe"
    RETRIEVAL = "retrieval"
    PLANNER = "planner"
    SELF_MODEL = "self_model"
    HISTORY_REUSE = "history_reuse"
    PROCEDURE_REUSE = "procedure_reuse"
    WAIT_FALLBACK = "wait_fallback"
    INTERVENTION = "intervention_compiler"


@dataclass
class DecisionCandidate:
    action: Dict[str, Any]
    candidate_id: str = ""
    source: CandidateSource = CandidateSource.BASE_GENERATION
    surfaced_from: List[str] = field(default_factory=list)
    function_name: str = ""
    action_kind: str = "call_tool"
    is_wait: bool = False
    is_probe: bool = False
    episode: int = 0
    tick: int = 0

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source.value,
            "surfaced_from": list(self.surfaced_from),
            "function_name": self.function_name,
            "action_kind": self.action_kind,
            "is_wait": self.is_wait,
            "is_probe": self.is_probe,
            "episode": self.episode,
            "tick": self.tick,
        }


@dataclass
class ValueScore:
    utility: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.5
    goal_alignment: float = 0.5
    intervention_value: float = 0.0
    mechanism_value: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "utility": 0.4,
        "novelty": 0.2,
        "confidence": 0.2,
        "goal_alignment": 0.2,
        "intervention": 0.0,
        "mechanism": 0.0,
    })

    @property
    def total(self) -> float:
        default_weights = {
            "utility": 0.4,
            "novelty": 0.2,
            "confidence": 0.2,
            "goal_alignment": 0.2,
            "intervention": 0.0,
            "mechanism": 0.0,
        }
        weights = self.weights if isinstance(self.weights, dict) else default_weights
        comps = {
            "utility": float(self.utility),
            "novelty": float(self.novelty),
            "confidence": float(self.confidence),
            "goal_alignment": float(self.goal_alignment),
            "intervention": float(self.intervention_value),
            "mechanism": float(self.mechanism_value),
        }
        norm_weights: Dict[str, float] = {}
        total_weight = 0.0
        for key, default in default_weights.items():
            w = max(0.0, float(weights.get(key, default)))
            norm_weights[key] = w
            total_weight += w
        if total_weight <= 1e-8:
            norm_weights = dict(default_weights)
            total_weight = sum(norm_weights.values())
        return sum(comps[k] * (norm_weights[k] / total_weight) for k in comps)

    def to_dict(self) -> dict:
        return {
            "utility": self.utility,
            "novelty": self.novelty,
            "confidence": self.confidence,
            "goal_alignment": self.goal_alignment,
            "intervention_value": self.intervention_value,
            "mechanism_value": self.mechanism_value,
            "weights": dict(self.weights),
            "total": self.total,
        }


@dataclass
class RiskScore:
    uncertainty: float = 0.5
    failure_likelihood: float = 0.5
    recovery_difficulty: float = 0.5

    @property
    def level(self) -> str:
        total = self.failure_likelihood
        if total < 0.3:
            return "low"
        elif total < 0.6:
            return "medium"
        return "high"

    @property
    def is_blocked(self) -> bool:
        return self.failure_likelihood > 0.8

    def to_dict(self) -> dict:
        return {
            "uncertainty": self.uncertainty,
            "failure_likelihood": self.failure_likelihood,
            "recovery_difficulty": self.recovery_difficulty,
            "level": self.level,
            "is_blocked": self.is_blocked,
        }


@dataclass
class ScoreComponent:
    name: str
    raw: float
    weight: float = 1.0
    source: str = "heuristic"

    @property
    def weighted(self) -> float:
        return self.raw * self.weight

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw": self.raw,
            "weight": self.weight,
            "weighted": self.weighted,
            "source": self.source,
        }


@dataclass
class DecisionScoreBreakdown:
    value_component: ScoreComponent
    intervention_component: ScoreComponent
    mechanism_component: ScoreComponent
    risk_penalty_component: ScoreComponent
    deliberation_component: ScoreComponent
    learning_component: ScoreComponent
    prediction_component: ScoreComponent
    world_model_component: ScoreComponent
    raw_final_score: float

    def to_dict(self) -> dict:
        return {
            "value_component": self.value_component.to_dict(),
            "intervention_component": self.intervention_component.to_dict(),
            "mechanism_component": self.mechanism_component.to_dict(),
            "risk_penalty_component": self.risk_penalty_component.to_dict(),
            "deliberation_component": self.deliberation_component.to_dict(),
            "learning_component": self.learning_component.to_dict(),
            "prediction_component": self.prediction_component.to_dict(),
            "world_model_component": self.world_model_component.to_dict(),
            "raw_final_score": self.raw_final_score,
        }


@dataclass
class DecisionScore:
    candidate_id: str
    value_score: ValueScore
    risk_score: RiskScore
    deliberation_bonus: float = 0.0
    learning_bonus: float = 0.0
    prediction_score: float = 0.0
    prediction_confidence: float = 0.5
    world_model_penalty: float = 0.0
    plan_criticality: float = 0.5
    shift_risk: float = 0.0
    self_model_reliability: float = 0.5
    recovery_availability: float = 0.5

    @property
    def penalty_factor(self) -> float:
        base = 0.35
        base += 0.25 * max(0.0, min(1.0, self.plan_criticality))
        base += 0.25 * max(0.0, min(1.0, self.shift_risk))
        base += 0.15 * (1.0 - max(0.0, min(1.0, self.self_model_reliability)))
        base -= 0.10 * max(0.0, min(1.0, self.recovery_availability))
        return max(0.15, min(1.2, base))

    @property
    def raw_final_score(self) -> float:
        risk_penalty = self.risk_score.failure_likelihood * self.penalty_factor
        return (
            self.value_score.total
            - risk_penalty
            + self.deliberation_bonus
            + self.learning_bonus
            + (self.prediction_score * max(0.0, min(1.0, self.prediction_confidence)))
            - self.world_model_penalty
        )

    @property
    def final_score(self) -> float:
        return max(0.0, self.raw_final_score)

    @property
    def breakdown(self) -> DecisionScoreBreakdown:
        return DecisionScoreBreakdown(
            value_component=ScoreComponent(
                name="value",
                raw=self.value_score.total - self.value_score.intervention_value,
                weight=1.0,
                source="value_model",
            ),
            intervention_component=ScoreComponent(
                name="intervention",
                raw=self.value_score.intervention_value,
                weight=1.0,
                source="intervention_value_model",
            ),
            mechanism_component=ScoreComponent(
                name="mechanism",
                raw=self.value_score.mechanism_value,
                weight=1.0,
                source="mechanism_value_model",
            ),
            risk_penalty_component=ScoreComponent(
                name="risk_penalty",
                raw=-self.risk_score.failure_likelihood,
                weight=self.penalty_factor,
                source="risk_model",
            ),
            deliberation_component=ScoreComponent(
                name="deliberation",
                raw=self.deliberation_bonus,
                weight=1.0,
                source="deliberation_rollout",
            ),
            learning_component=ScoreComponent(
                name="learning",
                raw=self.learning_bonus,
                weight=1.0,
                source="learning_runtime",
            ),
            prediction_component=ScoreComponent(
                name="prediction",
                raw=self.prediction_score,
                weight=max(0.0, min(1.0, self.prediction_confidence)),
                source="prediction_runtime",
            ),
            world_model_component=ScoreComponent(
                name="world_model_penalty",
                raw=-self.world_model_penalty,
                weight=1.0,
                source="world_model_protocol",
            ),
            raw_final_score=self.raw_final_score,
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "value_score": self.value_score.to_dict(),
            "risk_score": self.risk_score.to_dict(),
            "deliberation_bonus": self.deliberation_bonus,
            "learning_bonus": self.learning_bonus,
            "prediction_score": self.prediction_score,
            "prediction_confidence": self.prediction_confidence,
            "world_model_penalty": self.world_model_penalty,
            "plan_criticality": self.plan_criticality,
            "shift_risk": self.shift_risk,
            "self_model_reliability": self.self_model_reliability,
            "recovery_availability": self.recovery_availability,
            "penalty_factor": self.penalty_factor,
            "raw_final_score": self.raw_final_score,
            "final_score": self.final_score,
            "breakdown": self.breakdown.to_dict(),
        }


@dataclass
class DecisionOutcome:
    selected_candidate: Optional[DecisionCandidate]
    selected_score: Optional[DecisionScore]
    all_scores: List[DecisionScore] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    primary_reason: str = ""
    secondary_reasons: List[str] = field(default_factory=list)
    score_breakdowns: List[Dict[str, Any]] = field(default_factory=list)
    execute_as: str = "call_tool"

    def to_dict(self) -> dict:
        return {
            "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "selected_score": self.selected_score.to_dict() if self.selected_score else None,
            "all_scores": [s.to_dict() for s in self.all_scores],
            "rejected_ids": list(self.rejected_ids),
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "score_breakdowns": list(self.score_breakdowns),
            "execute_as": self.execute_as,
        }

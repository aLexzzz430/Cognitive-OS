from __future__ import annotations

"""Generic affordance inference over world anchors and intervention targets.

This version adds outcome-aware target ranking signals so downstream systems can
prefer anchors/targets that historically caused state change, progress, or
useful information gain, instead of mechanically scanning geometry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from modules.world_model.intervention_targets import (
    InterventionTarget,
    InterventionValueModel,
    WorldAnchor,
    _clamp01,
)


@dataclass(frozen=True)
class AnchorAffordance:
    anchor_ref: str
    action_modes: List[str] = field(default_factory=list)
    parameter_requirements: Dict[str, List[str]] = field(default_factory=dict)
    expected_effects: Dict[str, str] = field(default_factory=dict)
    noop_risk: float = 0.0
    confidence: float = 0.0
    target_value_score: float = 0.0
    state_change_bias: float = 0.0
    progress_bias: float = 0.0
    information_gain_bias: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_ref": self.anchor_ref,
            "action_modes": list(self.action_modes),
            "parameter_requirements": {k: list(v) for k, v in self.parameter_requirements.items()},
            "expected_effects": dict(self.expected_effects),
            "noop_risk": float(self.noop_risk),
            "confidence": float(self.confidence),
            "target_value_score": float(self.target_value_score),
            "state_change_bias": float(self.state_change_bias),
            "progress_bias": float(self.progress_bias),
            "information_gain_bias": float(self.information_gain_bias),
        }


class AffordanceModel:
    def __init__(self) -> None:
        self._value_model = InterventionValueModel()

    def infer_for_anchor(self, anchor: WorldAnchor) -> AnchorAffordance:
        action_modes: List[str] = ["inspect"]
        parameter_requirements: Dict[str, List[str]] = {"inspect": []}
        expected_effects: Dict[str, str] = {"inspect": "observation_refinement"}

        has_spatial_extent = bool(anchor.centroid() is not None or anchor.extent.get("bbox"))
        relation_degree = int(anchor.actionable_signals.get("relation_degree", 0) or 0)
        estimate = self._value_model.score_anchor(anchor)
        noop_rate = _clamp01(anchor.history.get("noop_rate", 0.0), 0.0)
        uncertainty = _clamp01(anchor.uncertainty, 0.0)
        novelty = _clamp01(anchor.novelty, 0.0)

        if has_spatial_extent and estimate.repeat_penalty < 0.80:
            action_modes.extend(["pointer_select", "pointer_activate"])
            parameter_requirements["pointer_select"] = ["x", "y"]
            parameter_requirements["pointer_activate"] = ["x", "y"]
            expected_effects["pointer_select"] = "selection_or_focus_change"
            expected_effects["pointer_activate"] = (
                "task_progress" if estimate.expected_progress >= estimate.expected_state_change else "state_change"
            )

        if relation_degree > 0 and estimate.expected_information_gain >= 0.14:
            action_modes.append("probe_relation")
            parameter_requirements["probe_relation"] = []
            expected_effects["probe_relation"] = "relational_feedback"

        if estimate.expected_progress >= 0.22:
            action_modes.append("confirm")
            parameter_requirements["confirm"] = []
            expected_effects["confirm"] = "commit_or_advance"

        if estimate.expected_information_gain >= 0.20 or novelty >= 0.4 or uncertainty >= 0.5:
            action_modes.append("probe_state_change")
            parameter_requirements["probe_state_change"] = []
            expected_effects["probe_state_change"] = "information_gain"

        deduped: List[str] = []
        for mode in action_modes:
            if mode not in deduped:
                deduped.append(mode)

        confidence = _clamp01(
            estimate.final_score * 0.58
            + estimate.expected_progress * 0.12
            + estimate.expected_state_change * 0.10
            + estimate.expected_information_gain * 0.10
            + (0.06 if has_spatial_extent else 0.0)
            + (0.06 if relation_degree > 0 else 0.0)
            - noop_rate * 0.18,
            0.0,
        )
        return AnchorAffordance(
            anchor_ref=anchor.anchor_id,
            action_modes=deduped,
            parameter_requirements=parameter_requirements,
            expected_effects=expected_effects,
            noop_risk=noop_rate,
            confidence=confidence,
            target_value_score=estimate.final_score,
            state_change_bias=estimate.expected_state_change,
            progress_bias=estimate.expected_progress,
            information_gain_bias=estimate.expected_information_gain,
        )

    def infer_for_targets(self, anchors: Sequence[WorldAnchor], targets: Sequence[InterventionTarget]) -> Dict[str, Dict[str, Any]]:
        by_anchor = {anchor.anchor_id: self.infer_for_anchor(anchor).to_dict() for anchor in anchors}
        for target in targets:
            target_entry = by_anchor.get(target.anchor_ref)
            if not isinstance(target_entry, dict):
                continue
            target_entry.setdefault("target_ids", []).append(target.target_id)
            target_entry.setdefault("target_kinds", []).append(target.target_kind)
            target_entry.setdefault("target_confidences", []).append(float(target.confidence))
            target_entry["target_value_score"] = max(
                float(target_entry.get("target_value_score", 0.0) or 0.0),
                float(target.priority_features.get("target_value_score", 0.0) or 0.0),
            )
        return by_anchor

    def score_target(self, target: InterventionTarget) -> Dict[str, float]:
        features = target.priority_features if isinstance(target.priority_features, dict) else {}
        target_value_score = _clamp01(features.get("target_value_score", target.confidence), 0.0)
        progress_bias = _clamp01(features.get("expected_progress", 0.0), 0.0)
        state_change_bias = _clamp01(features.get("expected_state_change", 0.0), 0.0)
        information_gain_bias = _clamp01(features.get("expected_information_gain", 0.0), 0.0)
        repeat_penalty = _clamp01(features.get("repeat_penalty", 0.0), 0.0)
        final_score = _clamp01(
            target_value_score * 0.52
            + progress_bias * 0.18
            + state_change_bias * 0.14
            + information_gain_bias * 0.10
            - repeat_penalty * 0.24,
            0.0,
        )
        return {
            "target_value_score": round(target_value_score, 4),
            "progress_bias": round(progress_bias, 4),
            "state_change_bias": round(state_change_bias, 4),
            "information_gain_bias": round(information_gain_bias, 4),
            "repeat_penalty": round(repeat_penalty, 4),
            "final_score": round(final_score, 4),
        }

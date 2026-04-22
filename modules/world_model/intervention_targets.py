from __future__ import annotations

"""Solver-aware intervention target proposal layer.

This module keeps the generic anchor/target abstraction, but upgrades target
proposal with:
- task-frame awareness
- object binding semantics
- goal-hypothesis preference matching
- outcome-oriented target value estimation

The output remains generic InterventionTarget objects. Environment-specific
execution stays delegated to execution compilers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


@dataclass(frozen=True)
class WorldAnchor:
    anchor_id: str
    anchor_type: str
    modality: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    extent: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    actionable_signals: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.5
    novelty: float = 0.0
    history: Dict[str, Any] = field(default_factory=dict)

    def centroid(self) -> Optional[Tuple[float, float]]:
        raw = self.attributes.get("centroid") or self.extent.get("centroid")
        if not isinstance(raw, dict):
            return None
        try:
            return float(raw.get("x")), float(raw.get("y"))
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class InterventionTarget:
    target_id: str
    anchor_ref: str
    target_kind: str
    candidate_actions: List[str] = field(default_factory=list)
    expected_effect_type: str = "probe_state_change"
    confidence: float = 0.0
    priority_features: Dict[str, Any] = field(default_factory=dict)
    execution_projection: Dict[str, Any] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "anchor_ref": self.anchor_ref,
            "target_kind": self.target_kind,
            "candidate_actions": list(self.candidate_actions),
            "expected_effect_type": self.expected_effect_type,
            "confidence": float(self.confidence),
            "priority_features": dict(self.priority_features),
            "execution_projection": dict(self.execution_projection),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ProposalContext:
    world_model_summary: Dict[str, Any]
    recent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    current_goal: str = ""
    active_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    task_frame_summary: Dict[str, Any] = field(default_factory=dict)
    object_bindings_summary: Dict[str, Any] = field(default_factory=dict)
    goal_hypotheses_summary: List[Dict[str, Any]] = field(default_factory=list)
    solver_state_summary: Dict[str, Any] = field(default_factory=dict)
    mechanism_hypotheses_summary: List[Dict[str, Any]] = field(default_factory=list)
    mechanism_control_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetValueEstimate:
    expected_state_change: float
    expected_progress: float
    expected_information_gain: float
    repeat_penalty: float
    exploration_bonus: float
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_state_change": round(self.expected_state_change, 4),
            "expected_progress": round(self.expected_progress, 4),
            "expected_information_gain": round(self.expected_information_gain, 4),
            "repeat_penalty": round(self.repeat_penalty, 4),
            "exploration_bonus": round(self.exploration_bonus, 4),
            "final_score": round(self.final_score, 4),
        }


class InterventionValueModel:
    def score_anchor(
        self,
        anchor: WorldAnchor,
        *,
        hypothesis_pressure: float = 0.0,
        preferred_target_bonus: float = 0.0,
        semantic_bonus: float = 0.0,
        role_bonus: float = 0.0,
        mode_bonus: float = 0.0,
    ) -> TargetValueEstimate:
        history = anchor.history if isinstance(anchor.history, dict) else {}
        novelty = _clamp01(anchor.novelty, 0.0)
        uncertainty = _clamp01(anchor.uncertainty, 0.0)
        relation_degree = min(6, int(anchor.actionable_signals.get("relation_degree", 0) or 0))
        relation_bonus = _clamp01(relation_degree / 6.0, 0.0)
        state_change_rate = _clamp01(history.get("state_change_rate", 0.0), 0.0)
        progress_rate = _clamp01(history.get("progress_rate", 0.0), 0.0)
        info_gain_rate = _clamp01(history.get("info_gain_rate", 0.0), 0.0)
        noop_rate = _clamp01(history.get("noop_rate", 0.0), 0.0)
        interaction_count = int(history.get("interaction_count", 0) or 0)
        repeated_recent_count = int(history.get("repeated_recent_count", 0) or 0)
        changed_pixels_score = _clamp01(_safe_float(history.get("avg_changed_pixels", 0.0), 0.0) / 64.0, 0.0)
        avg_reward = _safe_float(history.get("avg_reward", 0.0), 0.0)
        reward_score = _clamp01((avg_reward + 1.0) / 2.0, 0.5)

        expected_state_change = _clamp01(
            state_change_rate * 0.48
            + changed_pixels_score * 0.16
            + novelty * 0.08
            + uncertainty * 0.05
            + relation_bonus * 0.08
            + semantic_bonus * 0.07
            + mode_bonus * 0.08,
            0.0,
        )
        expected_progress = _clamp01(
            progress_rate * 0.50
            + reward_score * 0.12
            + relation_bonus * 0.06
            + hypothesis_pressure * 0.15
            + preferred_target_bonus * 0.10
            + role_bonus * 0.07,
            0.0,
        )
        expected_information_gain = _clamp01(
            info_gain_rate * 0.38
            + novelty * 0.18
            + uncertainty * 0.16
            + (0.14 if interaction_count == 0 else 0.0)
            + relation_bonus * 0.06
            + semantic_bonus * 0.08,
            0.0,
        )
        repeat_penalty = _clamp01(
            noop_rate * 0.52
            + min(0.26, repeated_recent_count * 0.09)
            + (0.12 if interaction_count >= 2 and progress_rate == 0.0 and state_change_rate == 0.0 else 0.0),
            0.0,
        )
        exploration_bonus = _clamp01(
            (0.22 if interaction_count == 0 else 0.0)
            + novelty * 0.12
            + uncertainty * 0.08
            + mode_bonus * 0.10,
            0.0,
        )
        final_score = _clamp01(
            expected_state_change * 0.24
            + expected_progress * 0.34
            + expected_information_gain * 0.18
            + exploration_bonus * 0.12
            + preferred_target_bonus * 0.10
            + role_bonus * 0.08
            - repeat_penalty * 0.42,
            0.0,
        )
        return TargetValueEstimate(
            expected_state_change=expected_state_change,
            expected_progress=expected_progress,
            expected_information_gain=expected_information_gain,
            repeat_penalty=repeat_penalty,
            exploration_bonus=exploration_bonus,
            final_score=final_score,
        )


def _effect_positive(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if bool(row.get("task_progressed", False)):
        return True
    return _safe_float(row.get("reward", row.get("avg_reward", 0.0)), 0.0) > 0.0


def _effect_state_changed(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if bool(row.get("state_changed", False)):
        return True
    return _safe_float(row.get("changed_pixel_count", 0.0), 0.0) > 0.0


def _effect_information_gain(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    info_gain = _safe_float(row.get("information_gain", row.get("info_gain", 0.0)), 0.0)
    if info_gain > 0.0:
        return True
    return _safe_float(row.get("changed_pixel_count", 0.0), 0.0) >= 4.0


def _interaction_history(recent_interactions: Sequence[Dict[str, Any]], anchor_id: str) -> Dict[str, Any]:
    relevant = [row for row in recent_interactions if isinstance(row, dict) and str(row.get("anchor_ref", "")) == anchor_id]
    total = len(relevant)
    state_change_count = sum(1 for row in relevant if _effect_state_changed(row))
    progress_count = sum(1 for row in relevant if _effect_positive(row))
    info_gain_count = sum(1 for row in relevant if _effect_information_gain(row))
    noop_count = max(0, total - state_change_count)
    repeated_recent_count = 0
    for row in reversed(list(recent_interactions)):
        if not isinstance(row, dict):
            continue
        if str(row.get("anchor_ref", "")) != anchor_id:
            break
        repeated_recent_count += 1
    return {
        "interaction_count": total,
        "state_change_count": state_change_count,
        "progress_count": progress_count,
        "info_gain_count": info_gain_count,
        "noop_count": noop_count,
        "repeated_recent_count": repeated_recent_count,
        "avg_changed_pixels": (sum(_safe_float(row.get("changed_pixel_count", 0.0), 0.0) for row in relevant) / float(total)) if total else 0.0,
        "avg_reward": (sum(_safe_float(row.get("reward", 0.0), 0.0) for row in relevant) / float(total)) if total else 0.0,
        "avg_information_gain": (sum(_safe_float(row.get("information_gain", row.get("info_gain", 0.0)), 0.0) for row in relevant) / float(total)) if total else 0.0,
        "state_change_rate": (state_change_count / float(total)) if total else 0.0,
        "progress_rate": (progress_count / float(total)) if total else 0.0,
        "info_gain_rate": (info_gain_count / float(total)) if total else 0.0,
        "noop_rate": (noop_count / float(total)) if total else 0.0,
    }


def _relation_index(relations: Sequence[Dict[str, Any]], anchor_id: str) -> List[Dict[str, Any]]:
    indexed: List[Dict[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if relation.get("source") == anchor_id or relation.get("target") == anchor_id:
            indexed.append(dict(relation))
    return indexed


def _object_binding_anchors(
    world_model_summary: Dict[str, Any],
    object_bindings_summary: Dict[str, Any],
    recent_interactions: Sequence[Dict[str, Any]],
) -> List[WorldAnchor]:
    modality = str(world_model_summary.get("observed_modality", world_model_summary.get("scene_type", "unknown")) or "unknown")
    relation_summary = _as_dict(world_model_summary.get("world_relation_summary", {}))
    world_relations = _as_list(world_model_summary.get("world_relations", []))
    novelty_score = _clamp01(world_model_summary.get("world_novelty_score", world_model_summary.get("novelty_score", 0.0)), 0.0)
    anchors: List[WorldAnchor] = []
    for obj in _as_list(object_bindings_summary.get("objects", [])):
        if not isinstance(obj, dict):
            continue
        anchor_id = str(obj.get("object_id", "") or "")
        bbox = _as_dict(obj.get("bbox", {}))
        centroid = _as_dict(obj.get("centroid", {}))
        semantic_candidates = [dict(item) for item in _as_list(obj.get("semantic_candidates", [])) if isinstance(item, dict)]
        role_candidates = [dict(item) for item in _as_list(obj.get("role_candidates", [])) if isinstance(item, dict)]
        history = _interaction_history(recent_interactions, anchor_id)
        localized_relations = _relation_index(world_relations, anchor_id)
        actionable_signals = {
            "relation_degree": len(localized_relations),
            "salience_score": float(obj.get("salience_score", 0.0) or 0.0),
            "actionable_score": float(obj.get("actionable_score", 0.0) or 0.0),
            "top_semantic": str(semantic_candidates[0].get("label", "") or "") if semantic_candidates else "",
            "top_role": str(role_candidates[0].get("role", "") or "") if role_candidates else "",
            "state_change_rate": round(history.get("state_change_rate", 0.0), 4),
            "progress_rate": round(history.get("progress_rate", 0.0), 4),
            "info_gain_rate": round(history.get("info_gain_rate", 0.0), 4),
            "noop_rate": round(history.get("noop_rate", 0.0), 4),
            "repeat_pressure": int(history.get("repeated_recent_count", 0) or 0),
        }
        attributes = {
            "color": obj.get("color"),
            "centroid": centroid,
            "semantic_candidates": semantic_candidates,
            "role_candidates": role_candidates,
            "geometric_features": _as_dict(obj.get("geometric_features", {})),
            "relation_summary": relation_summary,
        }
        extent = {"bbox": bbox, "centroid": centroid}
        uncertainty = _clamp01(0.30 + history.get("noop_rate", 0.0) * 0.22 + (0.14 if history.get("interaction_count", 0) == 0 else 0.0), 0.0)
        anchors.append(WorldAnchor(
            anchor_id=anchor_id,
            anchor_type="bound_object",
            modality=modality,
            attributes=attributes,
            extent=extent,
            relations=localized_relations,
            actionable_signals=actionable_signals,
            uncertainty=uncertainty,
            novelty=novelty_score,
            history=history,
        ))
    return anchors


def build_world_anchors(
    world_model_summary: Dict[str, Any],
    recent_interactions: Optional[Sequence[Dict[str, Any]]] = None,
    object_bindings_summary: Optional[Dict[str, Any]] = None,
) -> List[WorldAnchor]:
    summary = dict(world_model_summary or {})
    recent_rows = list(recent_interactions or [])
    binding_summary = _as_dict(object_bindings_summary)
    anchors = _object_binding_anchors(summary, binding_summary, recent_rows)
    if anchors:
        return anchors

    modality = str(summary.get("observed_modality", summary.get("scene_type", "unknown")) or "unknown")
    entities = _as_list(summary.get("world_entities", []))
    relations = _as_list(summary.get("world_relations", []))
    novelty_score = _clamp01(summary.get("world_novelty_score", summary.get("novelty_score", 0.0)), 0.0)
    anchors = []
    for idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        anchor_id = str(entity.get("entity_id", f"anchor_{idx}") or f"anchor_{idx}")
        localized_relations = _relation_index(relations, anchor_id)
        history = _interaction_history(recent_rows, anchor_id)
        anchors.append(WorldAnchor(
            anchor_id=anchor_id,
            anchor_type=str(entity.get("entity_type", "entity") or "entity"),
            modality=modality,
            attributes={
                "color": entity.get("color"),
                "area": entity.get("area"),
                "fill_ratio": entity.get("fill_ratio"),
                "centroid": _as_dict(entity.get("centroid", {})),
                "semantic_candidates": [],
                "role_candidates": [],
            },
            extent={"bbox": _as_dict(entity.get("bbox", {})), "centroid": _as_dict(entity.get("centroid", {}))},
            relations=localized_relations,
            actionable_signals={
                "relation_degree": len(localized_relations),
                "state_change_rate": round(history.get("state_change_rate", 0.0), 4),
                "progress_rate": round(history.get("progress_rate", 0.0), 4),
                "info_gain_rate": round(history.get("info_gain_rate", 0.0), 4),
                "noop_rate": round(history.get("noop_rate", 0.0), 4),
                "repeat_pressure": int(history.get("repeated_recent_count", 0) or 0),
            },
            uncertainty=_clamp01(0.34 + history.get("noop_rate", 0.0) * 0.18, 0.0),
            novelty=novelty_score,
            history=history,
        ))
    if anchors:
        return anchors
    return [WorldAnchor(anchor_id="scene_root", anchor_type="scene", modality=modality, attributes={"scene_summary": _as_dict(summary.get("world_scene_summary", {}))}, novelty=novelty_score, history=_interaction_history(recent_rows, "scene_root"))]


class InterventionTargetProposer:
    def __init__(self, *, max_targets: int = 8) -> None:
        self._max_targets = max(1, int(max_targets))
        self._value_model = InterventionValueModel()

    def propose(self, context: ProposalContext) -> List[InterventionTarget]:
        anchors = build_world_anchors(
            context.world_model_summary,
            recent_interactions=context.recent_interactions,
            object_bindings_summary=context.object_bindings_summary,
        )
        targets: List[InterventionTarget] = []
        for anchor in anchors:
            targets.extend(list(self._targets_for_anchor(anchor, context)))
        ranked = sorted(
            targets,
            key=lambda item: (
                -float(item.confidence),
                -float(item.priority_features.get("target_value_score", 0.0) or 0.0),
                item.target_kind,
                item.anchor_ref,
            ),
        )
        return ranked[: self._max_targets]

    def _targets_for_anchor(self, anchor: WorldAnchor, context: ProposalContext) -> Iterable[InterventionTarget]:
        features = self._priority_features(anchor, context)
        candidate_actions = self._candidate_actions(anchor, features, context.task_frame_summary)
        execution_projection = self._execution_projection(anchor)
        rationale = self._rationale(anchor, features)
        target_kind = self._target_kind(features, context)
        effect_type = self._effect_type(features)
        target_confidence = _clamp01(features["target_value_score"] * 0.78 + features["goal_hypothesis_support"] * 0.14 + features["mode_match_score"] * 0.08, 0.0)
        yield InterventionTarget(
            target_id=f"target::{anchor.anchor_id}::{target_kind}",
            anchor_ref=anchor.anchor_id,
            target_kind=target_kind,
            candidate_actions=candidate_actions,
            expected_effect_type=effect_type,
            confidence=target_confidence,
            priority_features=features,
            execution_projection=execution_projection,
            rationale=rationale,
        )
        if features["expected_progress"] >= 0.20 and "confirm" not in candidate_actions:
            yield InterventionTarget(
                target_id=f"target::{anchor.anchor_id}::confirm_followup",
                anchor_ref=anchor.anchor_id,
                target_kind="confirm_followup",
                candidate_actions=["confirm", *candidate_actions],
                expected_effect_type="state_commit",
                confidence=_clamp01(target_confidence * 0.90 + 0.05, 0.0),
                priority_features=features,
                execution_projection=execution_projection,
                rationale=[*rationale, "goal_hypothesis_supports_commit_followup"],
            )


    def _mechanism_support(self, anchor: WorldAnchor, context: ProposalContext) -> Dict[str, Any]:
        mechanisms = [row for row in _as_list(context.mechanism_hypotheses_summary) if isinstance(row, dict)]
        mechanism_control = _as_dict(context.mechanism_control_summary)
        best_family = str(mechanism_control.get("dominant_mechanism_family", "") or "")
        best_confidence = _clamp01(mechanism_control.get("dominant_mechanism_confidence", 0.0), 0.0)
        best_support = 0.0
        best_target_match = False
        preferred_action_families: List[str] = [str(x or "") for x in _as_list(mechanism_control.get("preferred_action_families", [])) if str(x or "")]
        discriminating_actions: List[str] = [str(x or "") for x in _as_list(mechanism_control.get("discriminating_actions", [])) if str(x or "")]
        for row in mechanisms:
            target_refs = [str(x or "") for x in _as_list(row.get("preferred_target_refs", [])) if str(x or "")]
            action_families = [str(x or "") for x in _as_list(row.get("preferred_action_families", [])) if str(x or "")]
            row_actions = [str(x or "") for x in _as_list(row.get("best_discriminating_actions", [])) if str(x or "")]
            confidence = _clamp01(row.get("confidence", 0.0), 0.0)
            target_match = bool(anchor.anchor_id and anchor.anchor_id in target_refs)
            support = confidence * (1.0 if target_match else 0.62)
            if support > best_support:
                best_support = support
                best_family = str(row.get("family", "") or best_family)
                best_confidence = confidence
                best_target_match = target_match
            for item in action_families:
                if item and item not in preferred_action_families:
                    preferred_action_families.append(item)
            for item in row_actions:
                if item and item not in discriminating_actions:
                    discriminating_actions.append(item)
        return {
            "mechanism_support_score": round(_clamp01(best_support, 0.0), 4),
            "mechanism_family": best_family,
            "mechanism_confidence": round(_clamp01(best_confidence, 0.0), 4),
            "mechanism_target_match": bool(best_target_match),
            "mechanism_preferred_action_families": preferred_action_families[:4],
            "mechanism_discriminating_actions": discriminating_actions[:4],
        }

    def _generic_actions_from_families(self, families: Sequence[str]) -> List[str]:
        mapped: List[str] = []
        for raw in families:
            family = str(raw or "").strip()
            if not family:
                continue
            if family == "pointer_interaction":
                for name in ("pointer_select", "pointer_activate"):
                    if name not in mapped:
                        mapped.append(name)
            elif family == "navigation_interaction":
                if "navigate_focus" not in mapped:
                    mapped.append("navigate_focus")
            elif family == "confirm_interaction":
                if "confirm" not in mapped:
                    mapped.append("confirm")
            elif family == "state_transform_interaction":
                if "probe_relation" not in mapped:
                    mapped.append("probe_relation")
        return mapped

    def _priority_features(self, anchor: WorldAnchor, context: ProposalContext) -> Dict[str, Any]:
        task_frame = _as_dict(context.task_frame_summary)
        solver_state = _as_dict(context.solver_state_summary)
        goal_hypotheses = [row for row in _as_list(context.goal_hypotheses_summary) if isinstance(row, dict)]
        semantic_candidates = [row for row in _as_list(anchor.attributes.get("semantic_candidates", [])) if isinstance(row, dict)]
        role_candidates = [row for row in _as_list(anchor.attributes.get("role_candidates", [])) if isinstance(row, dict)]
        mechanism_support = self._mechanism_support(anchor, context)
        top_semantic = str(semantic_candidates[0].get("label", "") or "") if semantic_candidates else ""
        top_role = str(role_candidates[0].get("role", "") or "") if role_candidates else ""

        dominant_mode = str(task_frame.get("dominant_interaction_mode", "") or "")
        preferred_target_refs = list(solver_state.get("preferred_target_refs", []) or [])
        preferred_target_bonus = 1.0 if anchor.anchor_id and anchor.anchor_id in preferred_target_refs else 0.0
        hypothesis_pressure = 0.0
        preferred_action_families: List[str] = []
        dominant_goal_family = str(solver_state.get("dominant_goal_family", "") or "")
        dominant_goal_confidence = _clamp01(solver_state.get("dominant_goal_confidence", 0.0), 0.0)
        for hypo in goal_hypotheses:
            preferred_refs = [str(x or "") for x in _as_list(hypo.get("preferred_target_refs", [])) if str(x or "")]
            if anchor.anchor_id in preferred_refs:
                hypothesis_pressure = max(hypothesis_pressure, _clamp01(hypo.get("confidence", 0.0), 0.0))
                preferred_action_families.extend([str(x or "") for x in _as_list(hypo.get("preferred_action_families", []))])
        semantic_bonus = 0.0
        if top_semantic in {"directional_like", "boundary_structure", "token_like"}:
            semantic_bonus += 0.10
        if top_role in {"hint_or_marker", "interactive_token", "scene_anchor"}:
            semantic_bonus += 0.06
        role_bonus = 0.10 if top_role in {"hint_or_marker", "interactive_token"} else (0.06 if top_role else 0.0)
        mode_bonus = 0.0
        if dominant_mode == "pointer_interaction":
            mode_bonus += 0.10
        elif dominant_mode == "navigation_interaction" and top_role == "scene_anchor":
            mode_bonus += 0.06
        value_estimate = self._value_model.score_anchor(
            anchor,
            hypothesis_pressure=max(hypothesis_pressure, dominant_goal_confidence * 0.4, float(mechanism_support.get("mechanism_confidence", 0.0) or 0.0) * 0.6),
            preferred_target_bonus=max(
                preferred_target_bonus * max(0.22, dominant_goal_confidence),
                1.0 if bool(mechanism_support.get("mechanism_target_match", False)) else 0.0,
            ),
            semantic_bonus=semantic_bonus,
            role_bonus=role_bonus,
            mode_bonus=mode_bonus,
        )
        action_family_match_score = 0.0
        if dominant_mode == "pointer_interaction":
            action_family_match_score = 1.0 if anchor.centroid() is not None else 0.45
        elif dominant_mode == "navigation_interaction":
            action_family_match_score = 0.70
        elif dominant_mode == "confirm_interaction":
            action_family_match_score = 0.55
        value_payload = value_estimate.to_dict()
        features = {
            "goal_hypothesis_support": round(max(hypothesis_pressure, dominant_goal_confidence * 0.45), 4),
            "dominant_goal_family": dominant_goal_family,
            "dominant_goal_confidence": round(dominant_goal_confidence, 4),
            "preferred_target_bonus": round(preferred_target_bonus, 4),
            "dominant_interaction_mode": dominant_mode,
            "mode_match_score": round(_clamp01(action_family_match_score, 0.0), 4),
            "semantic_hint": top_semantic,
            "role_hint": top_role,
            "solver_preferred_action_families": preferred_action_families[:4],
            "mechanism_family": str(mechanism_support.get("mechanism_family", "") or ""),
            "mechanism_confidence": float(mechanism_support.get("mechanism_confidence", 0.0) or 0.0),
            "mechanism_target_match": bool(mechanism_support.get("mechanism_target_match", False)),
            "mechanism_support_score": float(mechanism_support.get("mechanism_support_score", 0.0) or 0.0),
            "mechanism_preferred_action_families": list(mechanism_support.get("mechanism_preferred_action_families", []) or []),
            "mechanism_discriminating_actions": list(mechanism_support.get("mechanism_discriminating_actions", []) or []),
            **{k: v for k, v in (anchor.history if isinstance(anchor.history, dict) else {}).items() if not isinstance(v, dict)},
            **value_payload,
            "target_value_score": float(value_payload.get("final_score", 0.0) or 0.0),
        }
        return features

    def _target_kind(self, features: Dict[str, Any], context: ProposalContext) -> str:
        dominant_goal_family = str(features.get("dominant_goal_family", "") or "")
        if features["repeat_penalty"] >= 0.42 and int(features.get("interaction_count", 0) or 0) >= 2:
            return "avoid_repeated_noop"
        if dominant_goal_family == "select_or_activate_salient_structures" and features.get("preferred_target_bonus", 0.0) > 0.0:
            return "goal_aligned_activation"
        if dominant_goal_family == "reveal_hidden_state_via_probe" and features["expected_information_gain"] >= 0.24:
            return "information_gain_probe"
        if dominant_goal_family == "commit_or_confirm_world_state" and features["expected_progress"] >= 0.22:
            return "commit_likely_anchor"
        if features["expected_progress"] >= 0.30:
            return "commit_likely_anchor"
        if features["expected_state_change"] >= 0.30:
            return "state_change_probe"
        if features["expected_information_gain"] >= 0.26:
            return "information_gain_probe"
        return "explore_anchor"

    def _effect_type(self, features: Dict[str, Any]) -> str:
        if features["expected_progress"] >= 0.28:
            return "task_progress"
        if features["expected_state_change"] >= features["expected_information_gain"]:
            return "state_change"
        return "information_gain"

    def _candidate_actions(self, anchor: WorldAnchor, features: Dict[str, Any], task_frame_summary: Dict[str, Any]) -> List[str]:
        dominant_mode = str(task_frame_summary.get("dominant_interaction_mode", "") or "")
        actions: List[str] = ["inspect"]
        has_spatial_extent = bool(anchor.centroid() is not None or anchor.extent.get("bbox"))
        if dominant_mode == "pointer_interaction" and has_spatial_extent:
            if features["expected_progress"] >= 0.16:
                actions.append("pointer_activate")
            actions.append("pointer_select")
        elif dominant_mode == "navigation_interaction":
            actions.append("navigate_focus")
        if features["expected_progress"] >= 0.22:
            actions.append("confirm")
        if features["expected_information_gain"] >= 0.20:
            actions.append("probe_state_change")
        if int(anchor.actionable_signals.get("relation_degree", 0) or 0) > 0 and features["expected_information_gain"] >= 0.12:
            actions.append("probe_relation")
        for name in self._generic_actions_from_families(features.get("mechanism_preferred_action_families", [])):
            if name not in actions:
                actions.insert(1 if actions else 0, name)
        for name in self._generic_actions_from_families(features.get("mechanism_discriminating_actions", [])):
            if name not in actions:
                actions.append(name)
        deduped: List[str] = []
        for name in actions:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _execution_projection(self, anchor: WorldAnchor) -> Dict[str, Any]:
        projection: Dict[str, Any] = {"modality": anchor.modality, "anchor_type": anchor.anchor_type}
        centroid = anchor.centroid()
        if centroid is not None:
            projection["centroid"] = {"x": round(centroid[0], 3), "y": round(centroid[1], 3)}
        bbox = anchor.extent.get("bbox")
        if isinstance(bbox, dict) and bbox:
            projection["bbox"] = dict(bbox)
        return projection

    def _rationale(self, anchor: WorldAnchor, features: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        if features.get("preferred_target_bonus", 0.0) > 0.0:
            reasons.append("preferred_target_ref_match")
        if features["expected_progress"] >= 0.28:
            reasons.append("high_expected_progress")
        elif features["expected_state_change"] >= 0.30:
            reasons.append("high_expected_state_change")
        elif features["expected_information_gain"] >= 0.26:
            reasons.append("high_expected_information_gain")
        if features["repeat_penalty"] >= 0.42:
            reasons.append("repeat_penalty_high")
        if features.get("goal_hypothesis_support", 0.0) >= 0.35:
            reasons.append("goal_hypothesis_supported")
        if features.get("mechanism_support_score", 0.0) >= 0.35:
            reasons.append("mechanism_supported")
        if str(features.get("mechanism_family", "") or ""):
            reasons.append(f"mechanism:{features['mechanism_family']}")
        if str(features.get("semantic_hint", "") or ""):
            reasons.append(f"semantic:{features['semantic_hint']}")
        if str(features.get("role_hint", "") or ""):
            reasons.append(f"role:{features['role_hint']}")
        return reasons or ["generic_world_anchor"]


def summarize_intervention_targets(targets: Sequence[InterventionTarget]) -> List[Dict[str, Any]]:
    return [target.to_dict() for target in targets]

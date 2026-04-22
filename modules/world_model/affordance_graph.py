from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from modules.world_model.affordance_model import AffordanceModel
from modules.world_model.intervention_targets import (
    InterventionTargetProposer,
    ProposalContext,
    build_world_anchors,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def build_affordance_graph(
    world_model_summary: Dict[str, Any],
    *,
    recent_interactions: Optional[Sequence[Dict[str, Any]]] = None,
    task_frame_summary: Optional[Dict[str, Any]] = None,
    object_bindings_summary: Optional[Dict[str, Any]] = None,
    goal_hypotheses_summary: Optional[Sequence[Dict[str, Any]]] = None,
    solver_state_summary: Optional[Dict[str, Any]] = None,
    mechanism_hypotheses_summary: Optional[Sequence[Dict[str, Any]]] = None,
    mechanism_control_summary: Optional[Dict[str, Any]] = None,
    max_targets: int = 8,
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    proposer = InterventionTargetProposer(max_targets=max_targets)
    context = ProposalContext(
        world_model_summary=summary,
        recent_interactions=[dict(row) for row in list(recent_interactions or []) if isinstance(row, dict)],
        current_goal=str(_as_dict(summary.get("task_frame_summary", {})).get("inferred_level_goal", "")) if not isinstance(task_frame_summary, dict) else str(task_frame_summary.get("inferred_level_goal", "")),
        active_hypotheses=[dict(row) for row in _as_list(summary.get("active_hypotheses", [])) if isinstance(row, dict)],
        task_frame_summary=dict(task_frame_summary or summary.get("task_frame_summary", {})),
        object_bindings_summary=dict(object_bindings_summary or summary.get("object_bindings_summary", {})),
        goal_hypotheses_summary=[dict(row) for row in list(goal_hypotheses_summary or summary.get("goal_hypotheses_summary", [])) if isinstance(row, dict)],
        solver_state_summary=dict(solver_state_summary or summary.get("solver_state_summary", {})),
        mechanism_hypotheses_summary=[dict(row) for row in list(mechanism_hypotheses_summary or summary.get("mechanism_hypotheses", [])) if isinstance(row, dict)],
        mechanism_control_summary=dict(mechanism_control_summary or summary.get("mechanism_control_summary", {})),
    )
    targets = proposer.propose(context)
    anchors = build_world_anchors(
        summary,
        recent_interactions=context.recent_interactions,
        object_bindings_summary=context.object_bindings_summary,
    )
    affordance_model = AffordanceModel()
    affordance_map = affordance_model.infer_for_targets(anchors, targets)
    target_rows = [target.to_dict() for target in targets]
    affordance_rows = [dict(row) for row in affordance_map.values() if isinstance(row, dict)]
    expected_information_gain = 0.0
    if target_rows:
        expected_information_gain = max(
            float((row.get("priority_features", {}) if isinstance(row.get("priority_features", {}), dict) else {}).get("expected_information_gain", 0.0) or 0.0)
            for row in target_rows
        )
    return {
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "anchor_type": anchor.anchor_type,
                "modality": anchor.modality,
                "uncertainty": float(anchor.uncertainty),
                "novelty": float(anchor.novelty),
            }
            for anchor in anchors
        ],
        "affordances": affordance_rows,
        "candidate_intervention_targets": target_rows,
        "expected_information_gain": round(expected_information_gain, 4),
    }

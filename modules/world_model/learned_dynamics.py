from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.world_model.object_graph import build_object_graph
from modules.world_model.object_identity import PersistentObjectIdentityTracker


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def _normalize_bbox(raw_bbox: Any) -> Dict[str, int]:
    bbox = _as_dict(raw_bbox)
    if not bbox:
        return {}
    x_min = _as_int(bbox.get("x_min", bbox.get("col_min", 0)), 0)
    x_max = _as_int(bbox.get("x_max", bbox.get("col_max", x_min)), x_min)
    y_min = _as_int(bbox.get("y_min", bbox.get("row_min", 0)), 0)
    y_max = _as_int(bbox.get("y_max", bbox.get("row_max", y_min)), y_min)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": max(0, x_max - x_min + 1),
        "height": max(0, y_max - y_min + 1),
    }


def _extract_function_name(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return "wait"
    if action.get("kind") == "wait":
        return "wait"
    payload = _as_dict(action.get("payload", {}))
    tool_args = _as_dict(payload.get("tool_args", {}))
    return str(tool_args.get("function_name", "wait") or "wait")


def build_action_summary(
    action: Dict[str, Any],
    *,
    clicked_family: Optional[Dict[str, Any]] = None,
    goal_progress_assessment: Optional[Dict[str, Any]] = None,
    action_effect_signature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action_dict = _as_dict(action)
    meta = _as_dict(action_dict.get("_candidate_meta", {}))
    clicked = _as_dict(clicked_family)
    assessment = _as_dict(goal_progress_assessment)
    effect = _as_dict(action_effect_signature)
    anchor_ref = str(
        meta.get("anchor_ref")
        or assessment.get("clicked_anchor_ref")
        or effect.get("clicked_anchor_ref")
        or clicked.get("anchor_ref")
        or ""
    )
    object_color = (
        meta.get("object_color")
        if meta.get("object_color") is not None
        else clicked.get("color")
    )
    relation_type = str(
        meta.get("goal_progress_relation_type")
        or assessment.get("relation_type")
        or ""
    )
    target_family = str(
        meta.get("target_family")
        or clicked.get("family")
        or clicked.get("target_family")
        or ""
    )
    action_family = str(meta.get("action_family", "") or "")
    return {
        "function_name": _extract_function_name(action_dict),
        "target_anchor_ref": anchor_ref,
        "target_anchor_role": str(clicked.get("role", "") or ""),
        "target_family": target_family,
        "action_family": action_family,
        "object_color": int(object_color) if object_color is not None and str(object_color) != "" else None,
        "relation_type": relation_type,
        "controller_anchor": bool(
            meta.get("goal_progress_controller_anchor", False)
            or assessment.get("controller_signal", False)
        ),
        "controller_supported_goal_anchor": bool(
            meta.get("goal_progress_controller_supported_goal_anchor", False)
        ),
        "relation_anchor_match": bool(
            meta.get("goal_progress_relation_anchor_match", False)
        ),
        "gap_closing_relation_anchor": bool(
            meta.get("goal_progress_gap_closing_relation_anchor", False)
        ),
        "gap_closing_preferred_goal": bool(
            meta.get("goal_progress_gap_closing_preferred_goal", False)
        ),
    }


def _normalize_phase(value: Any, fallback: str = "unknown") -> str:
    phase = str(value or "").strip().lower()
    if not phase:
        return fallback
    aliases = {
        "not_finished": "running",
        "running": "running",
        "warm": "warming",
        "start": "starting",
        "solved": "committed",
        "done": "committed",
        "complete": "committed",
        "completed": "committed",
        "failed": "disrupted",
        "error": "disrupted",
    }
    return aliases.get(phase, phase)


def _reward_sign(reward: Any) -> str:
    value = _as_float(reward, 0.0)
    if value > 0.05:
        return "positive"
    if value < -0.05:
        return "negative"
    return "zero"


def _risk_type_from_transition(result: Dict[str, Any], reward: Any) -> str:
    result_dict = _as_dict(result)
    failure_reason = str(result_dict.get("failure_reason", "") or "").strip().lower()
    if "schema_failure" in failure_reason or failure_reason in {
        "illegal_click_coordinate_or_remote_rejection",
        "arc_agi3_schema_failure_remote_rejection",
    }:
        return "schema_failure"
    if _as_float(reward, 0.0) >= 0.0 and bool(result_dict.get("success", True)):
        return "opportunity_cost"
    err = _as_dict(result_dict.get("error", {}))
    err_type = str(err.get("type", "") or "").lower()
    if "resource" in err_type or "timeout" in err_type:
        return "resource_failure"
    if "bind" in err_type or "schema" in err_type:
        return "representation_failure"
    if "world" in err_type or "state" in err_type:
        return "world_model_failure"
    if "plan" in err_type:
        return "planner_failure"
    return "execution_failure"


def _extract_phase_from_observation(obs: Dict[str, Any], fallback: str = "unknown") -> str:
    observation = _as_dict(obs)
    world_state = _as_dict(observation.get("world_state", {}))
    novel_api = _as_dict(observation.get("novel_api", {}))
    raw = _as_dict(observation.get("raw", {}))
    return _normalize_phase(
        world_state.get("phase")
        or world_state.get("state")
        or novel_api.get("state")
        or raw.get("state"),
        fallback=fallback,
    )


def _top_relation_type(relation_hypotheses: Sequence[Dict[str, Any]]) -> str:
    if not isinstance(relation_hypotheses, list):
        return ""
    for row in relation_hypotheses:
        if not isinstance(row, dict):
            continue
        relation_type = str(row.get("relation_type", "") or "")
        if relation_type:
            return relation_type
    return ""


def _compact_belief_summary(world_model_summary: Dict[str, Any], belief_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = _as_dict(world_model_summary)
    belief = _as_dict(belief_summary if isinstance(belief_summary, dict) else summary)
    task_frame = _as_dict(summary.get("task_frame_summary", {}))
    inferred_goal = _as_dict(task_frame.get("inferred_level_goal", {}))
    if not inferred_goal and isinstance(summary.get("inferred_level_goal", {}), dict):
        inferred_goal = _as_dict(summary.get("inferred_level_goal", {}))
    return {
        "goal_family": str(
            inferred_goal.get("goal_family")
            or belief.get("goal_family")
            or summary.get("predicted_phase")
            or ""
        ),
        "goal_anchor_count": len(_as_list(inferred_goal.get("goal_anchor_refs", []))),
        "controller_count": len(_as_list(inferred_goal.get("controller_anchor_refs", []))),
        "relation_hypothesis_count": len(_as_list(inferred_goal.get("relation_hypotheses", []))),
        "preferred_goal_count": len(_as_list(inferred_goal.get("preferred_next_goal_anchor_refs", []))),
        "top_relation_type": _top_relation_type(_as_list(inferred_goal.get("relation_hypotheses", []))),
        "belief_total_count": _as_int(belief.get("total_beliefs", 0), 0),
        "belief_active_count": _as_int(belief.get("active_count", 0), 0),
        "belief_uncertain_count": _as_int(belief.get("uncertain_count", 0), 0),
    }


def _summarize_object_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    object_rows = [row for row in _as_list(graph.get("objects", [])) if isinstance(row, dict)]
    relation_rows = [row for row in _as_list(graph.get("relations", [])) if isinstance(row, dict)]
    intervention_targets = [str(value) for value in _as_list(graph.get("intervention_targets", [])) if str(value or "")]
    identity_summary = _as_dict(graph.get("identity_summary", {}))
    mean_actionable = (
        sum(_as_float(row.get("actionable_score", 0.0), 0.0) for row in object_rows) / float(len(object_rows))
        if object_rows
        else 0.0
    )
    colors: Dict[str, int] = defaultdict(int)
    for row in object_rows:
        color = row.get("color")
        if color is not None and str(color) != "":
            colors[str(color)] += 1
    top_colors = [name for name, _ in sorted(colors.items(), key=lambda item: (-item[1], item[0]))[:3]]
    compact_objects: List[Dict[str, Any]] = []
    for row in object_rows:
        centroid = _as_dict(row.get("centroid", {}))
        bbox = _normalize_bbox(row.get("bbox", {}))
        compact_objects.append(
            {
                "object_id": str(row.get("object_id", "") or ""),
                "persistent_object_id": str(row.get("persistent_object_id", "") or str(row.get("object_id", "") or "")),
                "object_type": str(row.get("object_type", "") or ""),
                "color": row.get("color"),
                "relation_degree": _as_int(row.get("relation_degree", 0), 0),
                "actionable_bucket": int(_clamp01(row.get("actionable_score", 0.0), 0.0) * 4.0),
                "identity_confidence": round(_as_float(row.get("identity_confidence", 0.0), 0.0), 4),
                "lineage_event": str(row.get("lineage_event", "") or ""),
                "lineage_parent_ids": [str(item) for item in _as_list(row.get("lineage_parent_ids", [])) if str(item or "")][:4],
                "track_age": _as_int(row.get("track_age", 0), 0),
                "centroid": {
                    "x": _as_int(round(_as_float(centroid.get("x", 0.0), 0.0)), 0),
                    "y": _as_int(round(_as_float(centroid.get("y", 0.0), 0.0)), 0),
                },
                "bbox": bbox,
            }
        )
    compact_relations: List[Dict[str, Any]] = []
    for row in relation_rows[:24]:
        source = str(row.get("source_persistent_object_id", row.get("source_object_id", "")) or "")
        target = str(row.get("target_persistent_object_id", row.get("target_object_id", "")) or "")
        relation_type = str(row.get("relation_type", "") or "")
        if not source or not target or not relation_type:
            continue
        compact_relations.append(
            {
                "relation_type": relation_type,
                "source_object_id": source,
                "target_object_id": target,
            }
        )
    return {
        "object_count": len(object_rows),
        "relation_count": len(relation_rows),
        "intervention_target_count": len(intervention_targets),
        "mean_actionable_score": round(mean_actionable, 4),
        "world_state_signature": str(graph.get("world_state_signature", "") or ""),
        "top_colors": top_colors,
        "objects": compact_objects,
        "relations": compact_relations,
        "identity_summary": {
            "active_track_count": _as_int(identity_summary.get("active_track_count", len(compact_objects)), len(compact_objects)),
            "new_track_count": _as_int(identity_summary.get("new_track_count", 0), 0),
            "reappeared_track_count": _as_int(identity_summary.get("reappeared_track_count", 0), 0),
            "transformed_track_count": _as_int(identity_summary.get("transformed_track_count", 0), 0),
            "split_event_count": _as_int(identity_summary.get("split_event_count", 0), 0),
            "merge_event_count": _as_int(identity_summary.get("merge_event_count", 0), 0),
        },
    }


def build_learned_dynamics_state_snapshot(
    observation: Dict[str, Any],
    *,
    world_model_summary: Optional[Dict[str, Any]] = None,
    hidden_state_summary: Optional[Dict[str, Any]] = None,
    belief_summary: Optional[Dict[str, Any]] = None,
    object_graph: Optional[Dict[str, Any]] = None,
    identity_tracker: Optional[PersistentObjectIdentityTracker] = None,
    tick: Optional[int] = None,
) -> Dict[str, Any]:
    summary = _as_dict(world_model_summary)
    hidden = _as_dict(hidden_state_summary)
    belief = _compact_belief_summary(summary, belief_summary)
    graph = (
        _as_dict(object_graph)
        if isinstance(object_graph, dict) and object_graph
        else build_object_graph(
            _as_dict(observation).get("perception", observation),
            world_model_summary=summary,
        )
    )
    if identity_tracker is not None:
        try:
            graph = identity_tracker.annotate_graph(graph, tick=_as_int(tick, 0))
        except Exception:
            graph = _as_dict(graph)
    graph_summary = _summarize_object_graph(graph)
    phase = _normalize_phase(
        hidden.get("phase")
        or hidden.get("hidden_state_phase")
        or graph.get("phases", [{}])[0].get("phase")
        or _extract_phase_from_observation(observation, fallback="unknown"),
        fallback="unknown",
    )
    hidden_summary = {
        "phase": phase,
        "phase_confidence": round(_clamp01(hidden.get("phase_confidence", hidden.get("hidden_phase_confidence", 0.0)), 0.0), 4),
        "hidden_state_depth": _as_int(hidden.get("hidden_state_depth", 0), 0),
        "drift_score": round(_clamp01(hidden.get("drift_score", hidden.get("hidden_drift_score", 0.0)), 0.0), 4),
        "uncertainty_score": round(_clamp01(hidden.get("uncertainty_score", hidden.get("hidden_uncertainty_score", 0.0)), 0.0), 4),
        "expected_next_phase": _normalize_phase(hidden.get("expected_next_phase", phase), fallback=phase),
    }
    numeric_features = {
        "object_count": graph_summary["object_count"],
        "relation_count": graph_summary["relation_count"],
        "intervention_target_count": graph_summary["intervention_target_count"],
        "mean_actionable_score": graph_summary["mean_actionable_score"],
        "phase_confidence": hidden_summary["phase_confidence"],
        "hidden_state_depth": hidden_summary["hidden_state_depth"],
        "drift_score": hidden_summary["drift_score"],
        "uncertainty_score": hidden_summary["uncertainty_score"],
        "goal_anchor_count": belief["goal_anchor_count"],
        "controller_count": belief["controller_count"],
        "relation_hypothesis_count": belief["relation_hypothesis_count"],
        "preferred_goal_count": belief["preferred_goal_count"],
        "belief_total_count": belief["belief_total_count"],
        "belief_active_count": belief["belief_active_count"],
        "belief_uncertain_count": belief["belief_uncertain_count"],
        "active_track_count": _as_int(graph_summary.get("identity_summary", {}).get("active_track_count", 0), 0),
        "new_track_count": _as_int(graph_summary.get("identity_summary", {}).get("new_track_count", 0), 0),
        "reappeared_track_count": _as_int(graph_summary.get("identity_summary", {}).get("reappeared_track_count", 0), 0),
        "transformed_track_count": _as_int(graph_summary.get("identity_summary", {}).get("transformed_track_count", 0), 0),
        "split_event_count": _as_int(graph_summary.get("identity_summary", {}).get("split_event_count", 0), 0),
        "merge_event_count": _as_int(graph_summary.get("identity_summary", {}).get("merge_event_count", 0), 0),
    }
    return {
        "phase": phase,
        "object_graph": graph_summary,
        "hidden_state": hidden_summary,
        "belief_summary": belief,
        "numeric_features": numeric_features,
    }


def summarize_object_graph_delta(before_snapshot: Dict[str, Any], after_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    before_graph = _as_dict(_as_dict(before_snapshot).get("object_graph", {}))
    after_graph = _as_dict(_as_dict(after_snapshot).get("object_graph", {}))
    return {
        "object_count_delta": _as_int(after_graph.get("object_count", 0), 0) - _as_int(before_graph.get("object_count", 0), 0),
        "relation_count_delta": _as_int(after_graph.get("relation_count", 0), 0) - _as_int(before_graph.get("relation_count", 0), 0),
        "intervention_target_delta": _as_int(after_graph.get("intervention_target_count", 0), 0) - _as_int(before_graph.get("intervention_target_count", 0), 0),
        "mean_actionable_delta": round(
            _as_float(after_graph.get("mean_actionable_score", 0.0), 0.0)
            - _as_float(before_graph.get("mean_actionable_score", 0.0), 0.0),
            4,
        ),
        "state_signature_changed": bool(
            str(before_graph.get("world_state_signature", "") or "")
            != str(after_graph.get("world_state_signature", "") or "")
        ),
    }


def _compact_object_state(row: Dict[str, Any]) -> Dict[str, Any]:
    row_dict = _as_dict(row)
    centroid = _as_dict(row_dict.get("centroid", {}))
    return {
        "persistent_object_id": str(row_dict.get("persistent_object_id", row_dict.get("object_id", "")) or ""),
        "object_type": str(row_dict.get("object_type", "") or ""),
        "color": row_dict.get("color"),
        "relation_degree": _as_int(row_dict.get("relation_degree", 0), 0),
        "actionable_bucket": _as_int(row_dict.get("actionable_bucket", 0), 0),
        "lineage_event": str(row_dict.get("lineage_event", "") or ""),
        "centroid": {
            "x": _as_int(centroid.get("x", 0), 0),
            "y": _as_int(centroid.get("y", 0), 0),
        },
        "bbox": _normalize_bbox(row_dict.get("bbox", {})),
    }


def _compact_changed_cell_signature(object_level_delta: Dict[str, Any]) -> List[Dict[str, Any]]:
    delta = _as_dict(object_level_delta)
    regions: List[Dict[str, Any]] = []
    for row in _as_list(delta.get("changed_objects", [])):
        row_dict = _as_dict(row)
        regions.append(
            {
                "persistent_object_id": str(row_dict.get("persistent_object_id", "") or ""),
                "change_kind": "changed",
                "bbox": _normalize_bbox(row_dict.get("after_bbox", row_dict.get("before_bbox", {}))),
                "before_color": row_dict.get("before_color"),
                "after_color": row_dict.get("after_color"),
            }
        )
    for row in _as_list(delta.get("added_objects", [])):
        row_dict = _as_dict(row)
        regions.append(
            {
                "persistent_object_id": str(row_dict.get("persistent_object_id", "") or ""),
                "change_kind": "added",
                "bbox": _normalize_bbox(row_dict.get("bbox", {})),
                "before_color": None,
                "after_color": row_dict.get("color"),
            }
        )
    for row in _as_list(delta.get("removed_objects", [])):
        row_dict = _as_dict(row)
        regions.append(
            {
                "persistent_object_id": str(row_dict.get("persistent_object_id", "") or ""),
                "change_kind": "removed",
                "bbox": _normalize_bbox(row_dict.get("bbox", {})),
                "before_color": row_dict.get("color"),
                "after_color": None,
            }
        )
    return regions[:8]


def _derive_next_object_state_prediction(object_level_delta: Dict[str, Any]) -> List[Dict[str, Any]]:
    delta = _as_dict(object_level_delta)
    predictions: List[Dict[str, Any]] = []
    for row in _as_list(delta.get("changed_objects", [])):
        row_dict = _as_dict(row)
        predictions.append(
            {
                "persistent_object_id": str(row_dict.get("persistent_object_id", "") or ""),
                "object_type": str(row_dict.get("after_object_type", row_dict.get("before_object_type", "")) or ""),
                "before_color": row_dict.get("before_color"),
                "after_color": row_dict.get("after_color"),
                "before_centroid": _as_dict(row_dict.get("before_centroid", {})),
                "after_centroid": _as_dict(row_dict.get("after_centroid", {})),
                "before_bbox": _normalize_bbox(row_dict.get("before_bbox", {})),
                "after_bbox": _normalize_bbox(row_dict.get("after_bbox", {})),
                "before_relation_degree": _as_int(row_dict.get("before_relation_degree", 0), 0),
                "after_relation_degree": _as_int(row_dict.get("after_relation_degree", 0), 0),
                "before_actionable_bucket": _as_int(row_dict.get("before_actionable_bucket", 0), 0),
                "after_actionable_bucket": _as_int(row_dict.get("after_actionable_bucket", 0), 0),
                "changed_fields": [str(item) for item in _as_list(row_dict.get("changed_fields", [])) if str(item or "")][:8],
                "lineage_event": str(row_dict.get("lineage_event", "") or ""),
            }
        )
    for row in _as_list(delta.get("added_objects", [])):
        row_dict = _as_dict(row)
        predictions.append(
            {
                "persistent_object_id": str(row_dict.get("persistent_object_id", "") or ""),
                "object_type": str(row_dict.get("object_type", "") or ""),
                "after_color": row_dict.get("color"),
                "after_centroid": _as_dict(row_dict.get("centroid", {})),
                "after_bbox": _normalize_bbox(row_dict.get("bbox", {})),
                "after_relation_degree": _as_int(row_dict.get("relation_degree", 0), 0),
                "after_actionable_bucket": _as_int(row_dict.get("actionable_bucket", 0), 0),
                "changed_fields": ["added"],
                "lineage_event": str(row_dict.get("lineage_event", "new") or "new"),
            }
        )
    return predictions[:8]


def _derive_next_relation_state_prediction(object_level_delta: Dict[str, Any]) -> Dict[str, Any]:
    delta = _as_dict(object_level_delta)
    return {
        "added_edges": [
            _as_dict(row)
            for row in _as_list(delta.get("added_relation_edge_refs", []))
            if isinstance(row, dict)
        ][:8],
        "removed_edges": [
            _as_dict(row)
            for row in _as_list(delta.get("removed_relation_edge_refs", []))
            if isinstance(row, dict)
        ][:8],
    }


def summarize_object_level_delta(before_snapshot: Dict[str, Any], after_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    before_graph = _as_dict(_as_dict(before_snapshot).get("object_graph", {}))
    after_graph = _as_dict(_as_dict(after_snapshot).get("object_graph", {}))
    before_objects = {
        str(row.get("persistent_object_id", row.get("object_id", "")) or ""): row
        for row in _as_list(before_graph.get("objects", []))
        if isinstance(row, dict) and str(row.get("persistent_object_id", row.get("object_id", "")) or "")
    }
    after_objects = {
        str(row.get("persistent_object_id", row.get("object_id", "")) or ""): row
        for row in _as_list(after_graph.get("objects", []))
        if isinstance(row, dict) and str(row.get("persistent_object_id", row.get("object_id", "")) or "")
    }
    before_ids = set(before_objects.keys())
    after_ids = set(after_objects.keys())
    added_ids = sorted(after_ids - before_ids)
    removed_ids = sorted(before_ids - after_ids)
    added_objects = [_compact_object_state(_as_dict(after_objects.get(object_id, {}))) for object_id in added_ids]
    removed_objects = [_compact_object_state(_as_dict(before_objects.get(object_id, {}))) for object_id in removed_ids]
    changed_ids: List[str] = []
    changed_objects: List[Dict[str, Any]] = []
    moved_count = 0
    recolored_count = 0
    relation_degree_shift_count = 0
    actionable_bucket_shift_count = 0
    reappeared_count = 0
    transformed_count = 0
    split_event_count = 0
    merge_event_count = 0
    for object_id in sorted(before_ids & after_ids):
        before_row = _as_dict(before_objects.get(object_id, {}))
        after_row = _as_dict(after_objects.get(object_id, {}))
        before_centroid = _as_dict(before_row.get("centroid", {}))
        after_centroid = _as_dict(after_row.get("centroid", {}))
        before_bbox = _as_dict(before_row.get("bbox", {}))
        after_bbox = _as_dict(after_row.get("bbox", {}))
        moved = (
            _as_int(before_centroid.get("x", 0), 0) != _as_int(after_centroid.get("x", 0), 0)
            or _as_int(before_centroid.get("y", 0), 0) != _as_int(after_centroid.get("y", 0), 0)
            or _as_int(before_bbox.get("x_min", 0), 0) != _as_int(after_bbox.get("x_min", 0), 0)
            or _as_int(before_bbox.get("y_min", 0), 0) != _as_int(after_bbox.get("y_min", 0), 0)
            or _as_int(before_bbox.get("x_max", 0), 0) != _as_int(after_bbox.get("x_max", 0), 0)
            or _as_int(before_bbox.get("y_max", 0), 0) != _as_int(after_bbox.get("y_max", 0), 0)
        )
        recolored = before_row.get("color") != after_row.get("color")
        relation_degree_shifted = _as_int(before_row.get("relation_degree", 0), 0) != _as_int(after_row.get("relation_degree", 0), 0)
        actionable_bucket_shifted = _as_int(before_row.get("actionable_bucket", 0), 0) != _as_int(after_row.get("actionable_bucket", 0), 0)
        if moved:
            moved_count += 1
        if recolored:
            recolored_count += 1
        if relation_degree_shifted:
            relation_degree_shift_count += 1
        if actionable_bucket_shifted:
            actionable_bucket_shift_count += 1
        lineage_event = str(after_row.get("lineage_event", "") or "")
        if lineage_event == "reappeared":
            reappeared_count += 1
        elif lineage_event == "transform":
            transformed_count += 1
        elif lineage_event == "split":
            split_event_count += 1
        elif lineage_event == "merge":
            merge_event_count += 1
        changed_fields: List[str] = []
        if moved:
            changed_fields.append("geometry")
        if recolored:
            changed_fields.append("color")
        if relation_degree_shifted:
            changed_fields.append("relation_degree")
        if actionable_bucket_shifted:
            changed_fields.append("actionable_bucket")
        if lineage_event in {"reappeared", "transform", "split", "merge"}:
            changed_fields.append(f"lineage:{lineage_event}")
        if changed_fields:
            changed_ids.append(object_id)
            changed_objects.append(
                {
                    "persistent_object_id": object_id,
                    "before_color": before_row.get("color"),
                    "after_color": after_row.get("color"),
                    "before_object_type": str(before_row.get("object_type", "") or ""),
                    "after_object_type": str(after_row.get("object_type", "") or ""),
                    "before_centroid": dict(before_centroid),
                    "after_centroid": dict(after_centroid),
                    "before_bbox": _normalize_bbox(before_bbox),
                    "after_bbox": _normalize_bbox(after_bbox),
                    "before_relation_degree": _as_int(before_row.get("relation_degree", 0), 0),
                    "after_relation_degree": _as_int(after_row.get("relation_degree", 0), 0),
                    "before_actionable_bucket": _as_int(before_row.get("actionable_bucket", 0), 0),
                    "after_actionable_bucket": _as_int(after_row.get("actionable_bucket", 0), 0),
                    "changed_fields": changed_fields,
                    "lineage_event": lineage_event,
                }
            )
    before_edges = {
        (
            str(row.get("relation_type", "") or ""),
            str(row.get("source_object_id", "") or ""),
            str(row.get("target_object_id", "") or ""),
        )
        for row in _as_list(before_graph.get("relations", []))
        if isinstance(row, dict)
    }
    after_edges = {
        (
            str(row.get("relation_type", "") or ""),
            str(row.get("source_object_id", "") or ""),
            str(row.get("target_object_id", "") or ""),
        )
        for row in _as_list(after_graph.get("relations", []))
        if isinstance(row, dict)
    }
    added_edges = sorted(after_edges - before_edges)
    removed_edges = sorted(before_edges - after_edges)
    return {
        "added_object_count": len(added_ids),
        "removed_object_count": len(removed_ids),
        "changed_object_count": len(changed_ids),
        "moved_object_count": moved_count,
        "recolored_object_count": recolored_count,
        "relation_degree_shift_count": relation_degree_shift_count,
        "actionable_bucket_shift_count": actionable_bucket_shift_count,
        "reappeared_object_count": reappeared_count,
        "transformed_object_count": transformed_count,
        "split_event_count": split_event_count,
        "merge_event_count": merge_event_count,
        "added_relation_edge_count": len(added_edges),
        "removed_relation_edge_count": len(removed_edges),
        "added_object_refs": added_ids[:6],
        "removed_object_refs": removed_ids[:6],
        "changed_object_refs": changed_ids[:6],
        "added_objects": added_objects[:6],
        "removed_objects": removed_objects[:6],
        "added_relation_edge_refs": [
            {"relation_type": relation_type, "source_object_id": source, "target_object_id": target}
            for relation_type, source, target in added_edges[:6]
        ],
        "removed_relation_edge_refs": [
            {"relation_type": relation_type, "source_object_id": source, "target_object_id": target}
            for relation_type, source, target in removed_edges[:6]
        ],
        "changed_objects": changed_objects[:6],
    }


def build_transition_target(
    before_snapshot: Dict[str, Any],
    after_snapshot: Dict[str, Any],
    *,
    result: Optional[Dict[str, Any]] = None,
    reward: Any = 0.0,
    information_gain: Any = 0.0,
    goal_progress_assessment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result_dict = _as_dict(result)
    assessment = _as_dict(goal_progress_assessment)
    delta = summarize_object_graph_delta(before_snapshot, after_snapshot)
    valid_state_change = bool(
        assessment.get("progressed", False)
        or _as_float(information_gain, 0.0) >= 0.1
        or delta.get("state_signature_changed", False)
        or abs(_as_int(delta.get("object_count_delta", 0), 0)) > 0
        or abs(_as_int(delta.get("relation_count_delta", 0), 0)) > 0
        or abs(_as_float(delta.get("mean_actionable_delta", 0.0), 0.0)) >= 0.05
    )
    object_level_delta = summarize_object_level_delta(before_snapshot, after_snapshot)
    valid_state_change = bool(
        valid_state_change
        or _as_int(object_level_delta.get("changed_object_count", 0), 0) > 0
        or _as_int(object_level_delta.get("added_relation_edge_count", 0), 0) > 0
        or _as_int(object_level_delta.get("removed_relation_edge_count", 0), 0) > 0
    )
    next_phase = _normalize_phase(
        _as_dict(after_snapshot).get("phase")
        or _extract_phase_from_observation(result_dict, fallback="unknown"),
        fallback="unknown",
    )
    return {
        "next_phase": next_phase,
        "reward_sign": _reward_sign(reward),
        "information_gain": round(_clamp01(information_gain, 0.0), 4),
        "risk_type": _risk_type_from_transition(result_dict, reward),
        "valid_state_change": bool(valid_state_change),
        "next_frame_signature": str(_as_dict(after_snapshot).get("object_graph", {}).get("world_state_signature", "") or ""),
        "next_object_graph_delta": delta,
        "next_object_level_delta": object_level_delta,
        "next_changed_cell_signature": _compact_changed_cell_signature(object_level_delta),
        "next_object_state_prediction": _derive_next_object_state_prediction(object_level_delta),
        "next_relation_state_prediction": _derive_next_relation_state_prediction(object_level_delta),
    }


def _object_state_accuracy(
    predicted_rows: Sequence[Dict[str, Any]],
    actual_rows: Sequence[Dict[str, Any]],
) -> float:
    predicted_by_id = {
        str(_as_dict(row).get("persistent_object_id", "") or ""): _as_dict(row)
        for row in predicted_rows
        if str(_as_dict(row).get("persistent_object_id", "") or "")
    }
    actual_by_id = {
        str(_as_dict(row).get("persistent_object_id", "") or ""): _as_dict(row)
        for row in actual_rows
        if str(_as_dict(row).get("persistent_object_id", "") or "")
    }
    shared_ids = sorted(set(predicted_by_id.keys()) & set(actual_by_id.keys()))
    if not shared_ids:
        return 1.0 if not predicted_rows and not actual_rows else 0.0
    score_total = 0.0
    for object_id in shared_ids:
        predicted = predicted_by_id[object_id]
        actual = actual_by_id[object_id]
        checks = 0.0
        hits = 0.0
        for field in ("after_color", "after_relation_degree", "after_actionable_bucket", "lineage_event"):
            if field in predicted or field in actual:
                checks += 1.0
                hits += float(predicted.get(field) == actual.get(field))
        predicted_centroid = _as_dict(predicted.get("after_centroid", {}))
        actual_centroid = _as_dict(actual.get("after_centroid", {}))
        if predicted_centroid or actual_centroid:
            checks += 1.0
            hits += float(
                _as_int(predicted_centroid.get("x", 0), 0) == _as_int(actual_centroid.get("x", 0), 0)
                and _as_int(predicted_centroid.get("y", 0), 0) == _as_int(actual_centroid.get("y", 0), 0)
            )
        score_total += hits / max(checks, 1.0)
    precision = score_total / float(max(len(predicted_rows), 1))
    recall = score_total / float(max(len(actual_rows), 1))
    if precision + recall <= 1e-6:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _relation_state_f1(predicted_state: Dict[str, Any], actual_state: Dict[str, Any]) -> float:
    def _edge_set(state: Dict[str, Any], key: str) -> set[Tuple[str, str, str]]:
        return {
            (
                str(_as_dict(row).get("relation_type", "") or ""),
                str(_as_dict(row).get("source_object_id", "") or ""),
                str(_as_dict(row).get("target_object_id", "") or ""),
            )
            for row in _as_list(_as_dict(state).get(key, []))
            if isinstance(row, dict)
        }

    predicted_edges = _edge_set(predicted_state, "added_edges") | _edge_set(predicted_state, "removed_edges")
    actual_edges = _edge_set(actual_state, "added_edges") | _edge_set(actual_state, "removed_edges")
    if not predicted_edges and not actual_edges:
        return 1.0
    if not predicted_edges or not actual_edges:
        return 0.0
    overlap = len(predicted_edges & actual_edges)
    precision = overlap / float(len(predicted_edges))
    recall = overlap / float(len(actual_edges))
    if precision + recall <= 1e-6:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def compare_transition_prediction(predicted: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    predicted_delta = _as_dict(_as_dict(predicted).get("next_object_graph_delta", {}))
    actual_delta = _as_dict(_as_dict(actual).get("next_object_graph_delta", {}))
    predicted_object_delta = _as_dict(_as_dict(predicted).get("next_object_level_delta", {}))
    actual_object_delta = _as_dict(_as_dict(actual).get("next_object_level_delta", {}))
    frame_signature_match = str(predicted.get("next_frame_signature", "")) == str(actual.get("next_frame_signature", ""))
    object_state_accuracy = _object_state_accuracy(
        _as_list(predicted.get("next_object_state_prediction", [])),
        _as_list(actual.get("next_object_state_prediction", [])),
    )
    relation_state_f1 = _relation_state_f1(
        _as_dict(predicted.get("next_relation_state_prediction", {})),
        _as_dict(actual.get("next_relation_state_prediction", {})),
    )
    info_gain_error = abs(
        _as_float(predicted.get("information_gain", 0.0), 0.0)
        - _as_float(actual.get("information_gain", 0.0), 0.0)
    )
    delta_error = (
        abs(_as_int(predicted_delta.get("object_count_delta", 0), 0) - _as_int(actual_delta.get("object_count_delta", 0), 0))
        + abs(_as_int(predicted_delta.get("relation_count_delta", 0), 0) - _as_int(actual_delta.get("relation_count_delta", 0), 0))
        + abs(_as_int(predicted_delta.get("intervention_target_delta", 0), 0) - _as_int(actual_delta.get("intervention_target_delta", 0), 0))
        + abs(_as_float(predicted_delta.get("mean_actionable_delta", 0.0), 0.0) - _as_float(actual_delta.get("mean_actionable_delta", 0.0), 0.0))
    )
    object_level_error = (
        abs(_as_int(predicted_object_delta.get("added_object_count", 0), 0) - _as_int(actual_object_delta.get("added_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("removed_object_count", 0), 0) - _as_int(actual_object_delta.get("removed_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("changed_object_count", 0), 0) - _as_int(actual_object_delta.get("changed_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("moved_object_count", 0), 0) - _as_int(actual_object_delta.get("moved_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("recolored_object_count", 0), 0) - _as_int(actual_object_delta.get("recolored_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("relation_degree_shift_count", 0), 0) - _as_int(actual_object_delta.get("relation_degree_shift_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("actionable_bucket_shift_count", 0), 0) - _as_int(actual_object_delta.get("actionable_bucket_shift_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("reappeared_object_count", 0), 0) - _as_int(actual_object_delta.get("reappeared_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("transformed_object_count", 0), 0) - _as_int(actual_object_delta.get("transformed_object_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("split_event_count", 0), 0) - _as_int(actual_object_delta.get("split_event_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("merge_event_count", 0), 0) - _as_int(actual_object_delta.get("merge_event_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("added_relation_edge_count", 0), 0) - _as_int(actual_object_delta.get("added_relation_edge_count", 0), 0))
        + abs(_as_int(predicted_object_delta.get("removed_relation_edge_count", 0), 0) - _as_int(actual_object_delta.get("removed_relation_edge_count", 0), 0))
    )
    total_error = (
        (0.25 if str(predicted.get("next_phase", "")) != str(actual.get("next_phase", "")) else 0.0)
        + (0.2 if str(predicted.get("reward_sign", "")) != str(actual.get("reward_sign", "")) else 0.0)
        + (0.2 if str(predicted.get("risk_type", "")) != str(actual.get("risk_type", "")) else 0.0)
        + (0.15 if bool(predicted.get("valid_state_change", False)) != bool(actual.get("valid_state_change", False)) else 0.0)
        + min(0.1, info_gain_error)
        + min(0.1, delta_error * 0.1)
        + min(0.1, object_level_error * 0.04)
        + (0.05 if not frame_signature_match else 0.0)
        + min(0.05, 1.0 - object_state_accuracy)
        + min(0.05, 1.0 - relation_state_f1)
    )
    return {
        "next_phase_match": str(predicted.get("next_phase", "")) == str(actual.get("next_phase", "")),
        "reward_sign_match": str(predicted.get("reward_sign", "")) == str(actual.get("reward_sign", "")),
        "risk_type_match": str(predicted.get("risk_type", "")) == str(actual.get("risk_type", "")),
        "valid_state_change_match": bool(predicted.get("valid_state_change", False)) == bool(actual.get("valid_state_change", False)),
        "frame_signature_match": frame_signature_match,
        "object_state_accuracy": round(object_state_accuracy, 4),
        "relation_state_f1": round(relation_state_f1, 4),
        "information_gain_error": round(info_gain_error, 4),
        "object_graph_delta_error": round(delta_error, 4),
        "object_level_delta_error": round(object_level_error, 4),
        "total_error": round(total_error, 4),
    }


class BucketedLearnedDynamicsModel:
    def __init__(self, buckets: Optional[Dict[str, Dict[str, Any]]] = None):
        self._buckets: Dict[str, Dict[str, Any]] = dict(buckets or {})

    @staticmethod
    def _bucket_key(
        snapshot: Dict[str, Any],
        action_summary: Dict[str, Any],
        *,
        include_goal: bool = True,
        include_relation: bool = True,
        include_target_anchor: bool = False,
    ) -> str:
        hidden = _as_dict(snapshot.get("hidden_state", {}))
        belief = _as_dict(snapshot.get("belief_summary", {}))
        parts = [
            str(action_summary.get("function_name", "wait") or "wait"),
            str(hidden.get("phase", "unknown") or "unknown"),
        ]
        parts.append(str(belief.get("goal_family", "") or "") if include_goal else "")
        parts.append(str(belief.get("top_relation_type", "") or "") if include_relation else "")
        parts.append(str(action_summary.get("target_family", "") or ""))
        parts.append(str(action_summary.get("action_family", "") or ""))
        parts.append(str(action_summary.get("relation_type", "") or ""))
        parts.append(str(action_summary.get("object_color", "") or ""))
        role_flags = [
            "ctrl" if bool(action_summary.get("controller_anchor", False)) else "",
            "ctrl_supported" if bool(action_summary.get("controller_supported_goal_anchor", False)) else "",
            "rel" if bool(action_summary.get("relation_anchor_match", False)) else "",
            "gap_rel" if bool(action_summary.get("gap_closing_relation_anchor", False)) else "",
            "gap_goal" if bool(action_summary.get("gap_closing_preferred_goal", False)) else "",
        ]
        parts.append(",".join(flag for flag in role_flags if flag))
        if include_target_anchor:
            target_anchor_ref = str(action_summary.get("target_anchor_ref", "") or "")
            if target_anchor_ref:
                parts.append(target_anchor_ref)
        return "|".join(parts)

    @classmethod
    def _bucket_specs(
        cls,
        snapshot: Dict[str, Any],
        action_summary: Dict[str, Any],
    ) -> List[Tuple[str, str, float]]:
        raw_specs = [
            ("goal_relation", cls._bucket_key(snapshot, action_summary, include_goal=True, include_relation=True, include_target_anchor=False), 1.0),
            ("goal_relation_anchor", cls._bucket_key(snapshot, action_summary, include_goal=True, include_relation=True, include_target_anchor=True), 0.92),
            ("goal_only", cls._bucket_key(snapshot, action_summary, include_goal=True, include_relation=False, include_target_anchor=False), 0.8),
            ("relation_only", cls._bucket_key(snapshot, action_summary, include_goal=False, include_relation=True, include_target_anchor=False), 0.72),
            ("coarse", cls._bucket_key(snapshot, action_summary, include_goal=False, include_relation=False, include_target_anchor=False), 0.56),
        ]
        deduped: List[Tuple[str, str, float]] = []
        seen: set[str] = set()
        for name, key, weight in raw_specs:
            if not key or key in seen:
                continue
            deduped.append((name, key, weight))
            seen.add(key)
        return deduped

    @staticmethod
    def _template_key(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps({}, sort_keys=True)

    @classmethod
    def _accumulate_template_counts(
        cls,
        bucket: Dict[str, Any],
        field: str,
        values: Any,
    ) -> None:
        counts = bucket.setdefault(field, defaultdict(int))
        if isinstance(values, list):
            for row in values:
                counts[cls._template_key(row)] += 1
            return
        if values not in ({}, [], None, ""):
            counts[cls._template_key(values)] += 1

    @staticmethod
    def _trim_template_counts(raw_counts: Dict[str, int], *, limit: int = 24) -> Dict[str, int]:
        items = [
            (str(key), _as_int(value, 0))
            for key, value in _as_dict(raw_counts).items()
            if _as_int(value, 0) > 0
        ]
        items.sort(key=lambda item: (-item[1], item[0]))
        return {key: value for key, value in items[:limit]}

    @staticmethod
    def _decode_template_rows(raw_counts: Dict[str, float], *, limit: int = 6) -> List[Any]:
        items = [
            (str(key), _as_float(value, 0.0))
            for key, value in _as_dict(raw_counts).items()
            if _as_float(value, 0.0) > 0.0
        ]
        items.sort(key=lambda item: (-item[1], item[0]))
        decoded: List[Any] = []
        for key, _ in items[:limit]:
            try:
                decoded.append(json.loads(key))
            except json.JSONDecodeError:
                continue
        return decoded

    @classmethod
    def fit(cls, samples: Sequence[Dict[str, Any]]) -> "BucketedLearnedDynamicsModel":
        buckets: Dict[str, Dict[str, Any]] = {}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            input_payload = _as_dict(sample.get("input", {}))
            target_payload = _as_dict(sample.get("target", {}))
            state_snapshot = _as_dict(input_payload.get("state_snapshot", {}))
            action_summary = _as_dict(input_payload.get("action_summary", {}))
            if not action_summary:
                action_summary = {"function_name": str(input_payload.get("function_name", "wait") or "wait")}
            for _, key, _ in cls._bucket_specs(state_snapshot, action_summary):
                bucket = buckets.setdefault(
                    key,
                    {
                        "count": 0,
                        "next_phase_counts": defaultdict(int),
                        "reward_sign_counts": defaultdict(int),
                        "risk_type_counts": defaultdict(int),
                        "valid_state_change_true": 0,
                        "information_gain_sum": 0.0,
                        "object_count_delta_sum": 0.0,
                        "relation_count_delta_sum": 0.0,
                        "intervention_target_delta_sum": 0.0,
                        "mean_actionable_delta_sum": 0.0,
                        "state_signature_changed_true": 0,
                        "added_object_count_sum": 0.0,
                        "removed_object_count_sum": 0.0,
                        "changed_object_count_sum": 0.0,
                        "moved_object_count_sum": 0.0,
                        "recolored_object_count_sum": 0.0,
                        "relation_degree_shift_count_sum": 0.0,
                        "actionable_bucket_shift_count_sum": 0.0,
                        "reappeared_object_count_sum": 0.0,
                        "transformed_object_count_sum": 0.0,
                        "split_event_count_sum": 0.0,
                        "merge_event_count_sum": 0.0,
                        "added_relation_edge_count_sum": 0.0,
                        "removed_relation_edge_count_sum": 0.0,
                        "next_frame_signature_counts": defaultdict(int),
                        "changed_cell_template_counts": defaultdict(int),
                        "object_state_template_counts": defaultdict(int),
                        "added_relation_state_template_counts": defaultdict(int),
                        "removed_relation_state_template_counts": defaultdict(int),
                    },
                )
                bucket["count"] += 1
                bucket["next_phase_counts"][str(target_payload.get("next_phase", "unknown") or "unknown")] += 1
                bucket["reward_sign_counts"][str(target_payload.get("reward_sign", "zero") or "zero")] += 1
                bucket["risk_type_counts"][str(target_payload.get("risk_type", "opportunity_cost") or "opportunity_cost")] += 1
                if bool(target_payload.get("valid_state_change", False)):
                    bucket["valid_state_change_true"] += 1
                bucket["information_gain_sum"] += _as_float(target_payload.get("information_gain", 0.0), 0.0)
                delta = _as_dict(target_payload.get("next_object_graph_delta", {}))
                bucket["object_count_delta_sum"] += _as_float(delta.get("object_count_delta", 0.0), 0.0)
                bucket["relation_count_delta_sum"] += _as_float(delta.get("relation_count_delta", 0.0), 0.0)
                bucket["intervention_target_delta_sum"] += _as_float(delta.get("intervention_target_delta", 0.0), 0.0)
                bucket["mean_actionable_delta_sum"] += _as_float(delta.get("mean_actionable_delta", 0.0), 0.0)
                if bool(delta.get("state_signature_changed", False)):
                    bucket["state_signature_changed_true"] += 1
                object_level_delta = _as_dict(target_payload.get("next_object_level_delta", {}))
                bucket["added_object_count_sum"] += _as_float(object_level_delta.get("added_object_count", 0.0), 0.0)
                bucket["removed_object_count_sum"] += _as_float(object_level_delta.get("removed_object_count", 0.0), 0.0)
                bucket["changed_object_count_sum"] += _as_float(object_level_delta.get("changed_object_count", 0.0), 0.0)
                bucket["moved_object_count_sum"] += _as_float(object_level_delta.get("moved_object_count", 0.0), 0.0)
                bucket["recolored_object_count_sum"] += _as_float(object_level_delta.get("recolored_object_count", 0.0), 0.0)
                bucket["relation_degree_shift_count_sum"] += _as_float(object_level_delta.get("relation_degree_shift_count", 0.0), 0.0)
                bucket["actionable_bucket_shift_count_sum"] += _as_float(object_level_delta.get("actionable_bucket_shift_count", 0.0), 0.0)
                bucket["reappeared_object_count_sum"] += _as_float(object_level_delta.get("reappeared_object_count", 0.0), 0.0)
                bucket["transformed_object_count_sum"] += _as_float(object_level_delta.get("transformed_object_count", 0.0), 0.0)
                bucket["split_event_count_sum"] += _as_float(object_level_delta.get("split_event_count", 0.0), 0.0)
                bucket["merge_event_count_sum"] += _as_float(object_level_delta.get("merge_event_count", 0.0), 0.0)
                bucket["added_relation_edge_count_sum"] += _as_float(object_level_delta.get("added_relation_edge_count", 0.0), 0.0)
                bucket["removed_relation_edge_count_sum"] += _as_float(object_level_delta.get("removed_relation_edge_count", 0.0), 0.0)
                bucket["next_frame_signature_counts"][str(target_payload.get("next_frame_signature", "") or "")] += 1
                cls._accumulate_template_counts(
                    bucket,
                    "changed_cell_template_counts",
                    target_payload.get("next_changed_cell_signature", []),
                )
                cls._accumulate_template_counts(
                    bucket,
                    "object_state_template_counts",
                    target_payload.get("next_object_state_prediction", []),
                )
                cls._accumulate_template_counts(
                    bucket,
                    "added_relation_state_template_counts",
                    _as_dict(target_payload.get("next_relation_state_prediction", {})).get("added_edges", []),
                )
                cls._accumulate_template_counts(
                    bucket,
                    "removed_relation_state_template_counts",
                    _as_dict(target_payload.get("next_relation_state_prediction", {})).get("removed_edges", []),
                )
        serializable: Dict[str, Dict[str, Any]] = {}
        for key, bucket in buckets.items():
            serializable[key] = {
                "count": int(bucket["count"]),
                "next_phase_counts": dict(bucket["next_phase_counts"]),
                "reward_sign_counts": dict(bucket["reward_sign_counts"]),
                "risk_type_counts": dict(bucket["risk_type_counts"]),
                "valid_state_change_true": int(bucket["valid_state_change_true"]),
                "information_gain_sum": float(bucket["information_gain_sum"]),
                "object_count_delta_sum": float(bucket["object_count_delta_sum"]),
                "relation_count_delta_sum": float(bucket["relation_count_delta_sum"]),
                "intervention_target_delta_sum": float(bucket["intervention_target_delta_sum"]),
                "mean_actionable_delta_sum": float(bucket["mean_actionable_delta_sum"]),
                "state_signature_changed_true": int(bucket["state_signature_changed_true"]),
                "added_object_count_sum": float(bucket["added_object_count_sum"]),
                "removed_object_count_sum": float(bucket["removed_object_count_sum"]),
                "changed_object_count_sum": float(bucket["changed_object_count_sum"]),
                "moved_object_count_sum": float(bucket["moved_object_count_sum"]),
                "recolored_object_count_sum": float(bucket["recolored_object_count_sum"]),
                "relation_degree_shift_count_sum": float(bucket["relation_degree_shift_count_sum"]),
                "actionable_bucket_shift_count_sum": float(bucket["actionable_bucket_shift_count_sum"]),
                "reappeared_object_count_sum": float(bucket["reappeared_object_count_sum"]),
                "transformed_object_count_sum": float(bucket["transformed_object_count_sum"]),
                "split_event_count_sum": float(bucket["split_event_count_sum"]),
                "merge_event_count_sum": float(bucket["merge_event_count_sum"]),
                "added_relation_edge_count_sum": float(bucket["added_relation_edge_count_sum"]),
                "removed_relation_edge_count_sum": float(bucket["removed_relation_edge_count_sum"]),
                "next_frame_signature_counts": dict(bucket["next_frame_signature_counts"]),
                "changed_cell_template_counts": cls._trim_template_counts(bucket["changed_cell_template_counts"]),
                "object_state_template_counts": cls._trim_template_counts(bucket["object_state_template_counts"]),
                "added_relation_state_template_counts": cls._trim_template_counts(bucket["added_relation_state_template_counts"]),
                "removed_relation_state_template_counts": cls._trim_template_counts(bucket["removed_relation_state_template_counts"]),
            }
        return cls(serializable)

    @staticmethod
    def _majority_label(raw_counts: Dict[str, int], default: str) -> Tuple[str, float]:
        counts = {str(k): int(v) for k, v in _as_dict(raw_counts).items()}
        if not counts:
            return default, 0.0
        label, support = max(counts.items(), key=lambda item: (item[1], item[0]))
        total = max(1, sum(counts.values()))
        return label, float(support) / float(total)

    @staticmethod
    def _majority_label_weighted(raw_counts: Dict[str, float], default: str) -> Tuple[str, float]:
        counts = {str(k): float(v) for k, v in _as_dict(raw_counts).items()}
        if not counts:
            return default, 0.0
        label, support = max(counts.items(), key=lambda item: (item[1], item[0]))
        total = max(1e-6, sum(counts.values()))
        return label, float(support) / float(total)

    def predict(self, state_snapshot: Dict[str, Any], action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action_summary = build_action_summary(action)
        matched_specs: List[Tuple[str, str, float, Dict[str, Any]]] = []
        for name, key, prior_weight in self._bucket_specs(state_snapshot, action_summary):
            bucket = _as_dict(self._buckets.get(key, {}))
            if bucket:
                matched_specs.append((name, key, prior_weight, bucket))
        if not matched_specs:
            return None
        weighted_total = 0.0
        max_count = 0
        weighted_next_phase_counts: Dict[str, float] = defaultdict(float)
        weighted_reward_sign_counts: Dict[str, float] = defaultdict(float)
        weighted_risk_type_counts: Dict[str, float] = defaultdict(float)
        weighted_valid_true = 0.0
        information_gain_sum = 0.0
        object_count_delta_sum = 0.0
        relation_count_delta_sum = 0.0
        intervention_target_delta_sum = 0.0
        mean_actionable_delta_sum = 0.0
        state_signature_changed_true = 0.0
        added_object_count_sum = 0.0
        removed_object_count_sum = 0.0
        changed_object_count_sum = 0.0
        moved_object_count_sum = 0.0
        recolored_object_count_sum = 0.0
        relation_degree_shift_count_sum = 0.0
        actionable_bucket_shift_count_sum = 0.0
        reappeared_object_count_sum = 0.0
        transformed_object_count_sum = 0.0
        split_event_count_sum = 0.0
        merge_event_count_sum = 0.0
        added_relation_edge_count_sum = 0.0
        removed_relation_edge_count_sum = 0.0
        weighted_frame_signature_counts: Dict[str, float] = defaultdict(float)
        weighted_changed_cell_templates: Dict[str, float] = defaultdict(float)
        weighted_object_state_templates: Dict[str, float] = defaultdict(float)
        weighted_added_relation_state_templates: Dict[str, float] = defaultdict(float)
        weighted_removed_relation_state_templates: Dict[str, float] = defaultdict(float)
        chosen_key = matched_specs[0][1]
        for _, key, prior_weight, bucket in matched_specs:
            count = max(1, _as_int(bucket.get("count", 0), 0))
            max_count = max(max_count, count)
            weighted_total += prior_weight * float(count)
            for label, label_count in _as_dict(bucket.get("next_phase_counts", {})).items():
                weighted_next_phase_counts[str(label)] += prior_weight * _as_int(label_count, 0)
            for label, label_count in _as_dict(bucket.get("reward_sign_counts", {})).items():
                weighted_reward_sign_counts[str(label)] += prior_weight * _as_int(label_count, 0)
            for label, label_count in _as_dict(bucket.get("risk_type_counts", {})).items():
                weighted_risk_type_counts[str(label)] += prior_weight * _as_int(label_count, 0)
            weighted_valid_true += prior_weight * _as_int(bucket.get("valid_state_change_true", 0), 0)
            information_gain_sum += prior_weight * _as_float(bucket.get("information_gain_sum", 0.0), 0.0)
            object_count_delta_sum += prior_weight * _as_float(bucket.get("object_count_delta_sum", 0.0), 0.0)
            relation_count_delta_sum += prior_weight * _as_float(bucket.get("relation_count_delta_sum", 0.0), 0.0)
            intervention_target_delta_sum += prior_weight * _as_float(bucket.get("intervention_target_delta_sum", 0.0), 0.0)
            mean_actionable_delta_sum += prior_weight * _as_float(bucket.get("mean_actionable_delta_sum", 0.0), 0.0)
            state_signature_changed_true += prior_weight * _as_int(bucket.get("state_signature_changed_true", 0), 0)
            added_object_count_sum += prior_weight * _as_float(bucket.get("added_object_count_sum", 0.0), 0.0)
            removed_object_count_sum += prior_weight * _as_float(bucket.get("removed_object_count_sum", 0.0), 0.0)
            changed_object_count_sum += prior_weight * _as_float(bucket.get("changed_object_count_sum", 0.0), 0.0)
            moved_object_count_sum += prior_weight * _as_float(bucket.get("moved_object_count_sum", 0.0), 0.0)
            recolored_object_count_sum += prior_weight * _as_float(bucket.get("recolored_object_count_sum", 0.0), 0.0)
            relation_degree_shift_count_sum += prior_weight * _as_float(bucket.get("relation_degree_shift_count_sum", 0.0), 0.0)
            actionable_bucket_shift_count_sum += prior_weight * _as_float(bucket.get("actionable_bucket_shift_count_sum", 0.0), 0.0)
            reappeared_object_count_sum += prior_weight * _as_float(bucket.get("reappeared_object_count_sum", 0.0), 0.0)
            transformed_object_count_sum += prior_weight * _as_float(bucket.get("transformed_object_count_sum", 0.0), 0.0)
            split_event_count_sum += prior_weight * _as_float(bucket.get("split_event_count_sum", 0.0), 0.0)
            merge_event_count_sum += prior_weight * _as_float(bucket.get("merge_event_count_sum", 0.0), 0.0)
            added_relation_edge_count_sum += prior_weight * _as_float(bucket.get("added_relation_edge_count_sum", 0.0), 0.0)
            removed_relation_edge_count_sum += prior_weight * _as_float(bucket.get("removed_relation_edge_count_sum", 0.0), 0.0)
            for label, label_count in _as_dict(bucket.get("next_frame_signature_counts", {})).items():
                weighted_frame_signature_counts[str(label)] += prior_weight * _as_int(label_count, 0)
            for template_key, template_count in _as_dict(bucket.get("changed_cell_template_counts", {})).items():
                weighted_changed_cell_templates[str(template_key)] += prior_weight * _as_int(template_count, 0)
            for template_key, template_count in _as_dict(bucket.get("object_state_template_counts", {})).items():
                weighted_object_state_templates[str(template_key)] += prior_weight * _as_int(template_count, 0)
            for template_key, template_count in _as_dict(bucket.get("added_relation_state_template_counts", {})).items():
                weighted_added_relation_state_templates[str(template_key)] += prior_weight * _as_int(template_count, 0)
            for template_key, template_count in _as_dict(bucket.get("removed_relation_state_template_counts", {})).items():
                weighted_removed_relation_state_templates[str(template_key)] += prior_weight * _as_int(template_count, 0)
        next_phase, phase_purity = self._majority_label_weighted(weighted_next_phase_counts, "unknown")
        reward_sign, reward_purity = self._majority_label_weighted(weighted_reward_sign_counts, "zero")
        risk_type, risk_purity = self._majority_label_weighted(weighted_risk_type_counts, "opportunity_cost")
        next_frame_signature, frame_purity = self._majority_label_weighted(weighted_frame_signature_counts, "")
        valid_ratio = weighted_valid_true / max(weighted_total, 1e-6)
        purity = max(phase_purity, reward_purity, risk_purity, frame_purity, valid_ratio, 1.0 - valid_ratio)
        predicted_changed_cells = self._decode_template_rows(weighted_changed_cell_templates, limit=8)
        predicted_object_states = self._decode_template_rows(weighted_object_state_templates, limit=8)
        added_relation_rows = [
            _as_dict(row)
            for row in self._decode_template_rows(weighted_added_relation_state_templates, limit=8)
            if isinstance(row, dict)
        ]
        removed_relation_rows = [
            _as_dict(row)
            for row in self._decode_template_rows(weighted_removed_relation_state_templates, limit=8)
            if isinstance(row, dict)
        ]
        confidence = _clamp01(
            0.24
            + min(0.34, max_count * 0.035)
            + min(0.18, weighted_total * 0.015)
            + purity * 0.26,
            0.0,
        )
        return {
            "source": "learned_dynamics_shadow",
            "bucket_key": chosen_key,
            "support": max_count,
            "fusion_support": round(weighted_total, 4),
            "matched_bucket_count": len(matched_specs),
            "matched_bucket_keys": [
                {"name": name, "key": key, "prior_weight": round(prior_weight, 4), "count": _as_int(bucket.get("count", 0), 0)}
                for name, key, prior_weight, bucket in matched_specs
            ],
            "action_summary": action_summary,
            "confidence": round(confidence, 4),
            "next_phase": next_phase,
            "reward_sign": reward_sign,
            "next_frame_signature": next_frame_signature,
            "information_gain": round(information_gain_sum / max(weighted_total, 1e-6), 4),
            "risk_type": risk_type,
            "valid_state_change": bool(valid_ratio >= 0.5),
            "next_object_graph_delta": {
                "object_count_delta": round(object_count_delta_sum / max(weighted_total, 1e-6), 4),
                "relation_count_delta": round(relation_count_delta_sum / max(weighted_total, 1e-6), 4),
                "intervention_target_delta": round(intervention_target_delta_sum / max(weighted_total, 1e-6), 4),
                "mean_actionable_delta": round(mean_actionable_delta_sum / max(weighted_total, 1e-6), 4),
                "state_signature_changed": bool(
                    state_signature_changed_true / max(weighted_total, 1e-6) >= 0.5
                ),
            },
            "next_object_level_delta": {
                "added_object_count": round(added_object_count_sum / max(weighted_total, 1e-6), 4),
                "removed_object_count": round(removed_object_count_sum / max(weighted_total, 1e-6), 4),
                "changed_object_count": round(changed_object_count_sum / max(weighted_total, 1e-6), 4),
                "moved_object_count": round(moved_object_count_sum / max(weighted_total, 1e-6), 4),
                "recolored_object_count": round(recolored_object_count_sum / max(weighted_total, 1e-6), 4),
                "relation_degree_shift_count": round(relation_degree_shift_count_sum / max(weighted_total, 1e-6), 4),
                "actionable_bucket_shift_count": round(actionable_bucket_shift_count_sum / max(weighted_total, 1e-6), 4),
                "reappeared_object_count": round(reappeared_object_count_sum / max(weighted_total, 1e-6), 4),
                "transformed_object_count": round(transformed_object_count_sum / max(weighted_total, 1e-6), 4),
                "split_event_count": round(split_event_count_sum / max(weighted_total, 1e-6), 4),
                "merge_event_count": round(merge_event_count_sum / max(weighted_total, 1e-6), 4),
                "added_relation_edge_count": round(added_relation_edge_count_sum / max(weighted_total, 1e-6), 4),
                "removed_relation_edge_count": round(removed_relation_edge_count_sum / max(weighted_total, 1e-6), 4),
            },
            "next_changed_cell_signature": predicted_changed_cells,
            "next_object_state_prediction": predicted_object_states,
            "next_relation_state_prediction": {
                "added_edges": added_relation_rows[:8],
                "removed_edges": removed_relation_rows[:8],
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "bucketed_learned_dynamics_shadow",
            "bucket_count": len(self._buckets),
            "buckets": self._buckets,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BucketedLearnedDynamicsModel":
        return cls(_as_dict(payload).get("buckets", {}))

    def summary(self) -> Dict[str, Any]:
        return {
            "model_type": "bucketed_learned_dynamics_shadow",
            "bucket_count": len(self._buckets),
            "max_support": max((_as_int(bucket.get("count", 0), 0) for bucket in self._buckets.values()), default=0),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> Optional["BucketedLearnedDynamicsModel"]:
        model_path = Path(path)
        if not model_path.exists():
            return None
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(payload)


def default_learned_dynamics_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime" / "models" / "learned_dynamics_shadow.json"


def _derive_hidden_state_for_trace_entry(entry: Dict[str, Any], next_entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observation = _as_dict(entry.get("observation", {}))
    outcome = _as_dict(entry.get("outcome", {}))
    task_progress = _as_dict(entry.get("task_progress", {}))
    current_phase = _extract_phase_from_observation(observation, fallback="unknown")
    next_phase = (
        _extract_phase_from_observation(_as_dict(next_entry.get("observation", {})), fallback="")
        if isinstance(next_entry, dict)
        else ""
    )
    if not next_phase:
        next_phase = _extract_phase_from_observation(outcome, fallback=current_phase)
    return {
        "phase": current_phase,
        "phase_confidence": 0.6 if current_phase != "unknown" else 0.35,
        "hidden_state_depth": 1 if bool(task_progress.get("progressed", False)) else 0,
        "uncertainty_score": 0.25 if bool(task_progress.get("progressed", False)) else 0.55,
        "drift_score": 0.1 if bool(task_progress.get("progressed", False)) else 0.35,
        "expected_next_phase": next_phase or current_phase,
    }


def extract_training_samples_from_episode_trace(episode_trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace = [row for row in _as_list(episode_trace) if isinstance(row, dict)]
    samples: List[Dict[str, Any]] = []
    identity_tracker = PersistentObjectIdentityTracker()
    snapshot_sequence: List[Dict[str, Any]] = []
    for index, entry in enumerate(trace):
        observation = _as_dict(entry.get("observation", {}))
        hidden_before = _derive_hidden_state_for_trace_entry(entry, trace[index + 1] if index + 1 < len(trace) else None)
        belief_before = {
            "inferred_level_goal": _as_dict(entry.get("inferred_level_goal", {})),
            "goal_progress_assessment": _as_dict(entry.get("goal_progress_assessment", {})),
        }
        snapshot_sequence.append(
            build_learned_dynamics_state_snapshot(
                observation,
                world_model_summary=belief_before,
                hidden_state_summary=hidden_before,
                belief_summary=belief_before,
                identity_tracker=identity_tracker,
                tick=index * 2,
            )
        )
    if trace:
        last_entry = trace[-1]
        next_observation = _as_dict(last_entry.get("outcome", {}))
        hidden_after = _derive_hidden_state_for_trace_entry({"observation": next_observation}, None)
        snapshot_sequence.append(
            build_learned_dynamics_state_snapshot(
                next_observation,
                world_model_summary={},
                hidden_state_summary=hidden_after,
                belief_summary={},
                identity_tracker=identity_tracker,
                tick=len(trace) * 2 + 1,
            )
        )
    for index, entry in enumerate(trace):
        action = _as_dict(entry.get("action", {}))
        outcome = _as_dict(entry.get("outcome", {}))
        before_snapshot = snapshot_sequence[index]
        after_snapshot = snapshot_sequence[index + 1]
        information_gain = _as_float(entry.get("information_gain", 0.0), 0.0)
        reward = _as_float(entry.get("reward", _as_dict(outcome).get("reward", 0.0)), 0.0)
        target = build_transition_target(
            before_snapshot,
            after_snapshot,
            result=outcome,
            reward=reward,
            information_gain=information_gain,
            goal_progress_assessment=_as_dict(entry.get("goal_progress_assessment", {})),
        )
        target["target_state_snapshot"] = after_snapshot
        samples.append(
            {
                "input": {
                    "tick": _as_int(entry.get("tick", index), index),
                    "function_name": _extract_function_name(action),
                    "action_summary": build_action_summary(
                        action,
                        clicked_family=_as_dict(entry.get("clicked_family", {})),
                        goal_progress_assessment=_as_dict(entry.get("goal_progress_assessment", {})),
                        action_effect_signature=_as_dict(entry.get("action_effect_signature", {})),
                    ),
                    "state_snapshot": before_snapshot,
                },
                "target": target,
                "metadata": {
                    "tick": _as_int(entry.get("tick", index), index),
                    "reward": reward,
                    "progressed": bool(_as_dict(entry.get("task_progress", {})).get("progressed", False)),
                },
            }
        )
    return samples


def extract_rollout_training_sequences_from_episode_trace(
    episode_trace: Sequence[Dict[str, Any]],
    *,
    horizons: Sequence[int] = (1, 2, 4),
) -> List[Dict[str, Any]]:
    trace = [row for row in _as_list(episode_trace) if isinstance(row, dict)]
    if not trace:
        return []
    one_step_samples = extract_training_samples_from_episode_trace(trace)
    rollout_rows: List[Dict[str, Any]] = []
    for start_index, entry in enumerate(trace):
        action_sequence: List[Dict[str, Any]] = []
        step_inputs: List[Dict[str, Any]] = []
        horizon_targets: List[Dict[str, Any]] = []
        max_horizon = min(max(horizons) if horizons else 1, len(trace) - start_index)
        for offset in range(max_horizon):
            current = trace[start_index + offset]
            action_summary = build_action_summary(
                _as_dict(current.get("action", {})),
                clicked_family=_as_dict(current.get("clicked_family", {})),
                goal_progress_assessment=_as_dict(current.get("goal_progress_assessment", {})),
                action_effect_signature=_as_dict(current.get("action_effect_signature", {})),
            )
            action_sequence.append(action_summary)
            step_inputs.append(
                {
                    "state_snapshot": dict(_as_dict(one_step_samples[start_index + offset].get("input", {})).get("state_snapshot", {})),
                    "action_summary": dict(action_summary),
                }
            )
            horizon = offset + 1
            if horizon not in horizons:
                continue
            cumulative_reward = sum(
                _as_float(trace[idx].get("reward", _as_dict(trace[idx].get("outcome", {})).get("reward", 0.0)), 0.0)
                for idx in range(start_index, start_index + horizon)
            )
            horizon_information_gain = max(
                [_as_float(trace[idx].get("information_gain", 0.0), 0.0) for idx in range(start_index, start_index + horizon)]
                or [0.0]
            )
            terminal_entry = trace[start_index + horizon - 1]
            terminal_state_snapshot = dict(
                _as_dict(one_step_samples[start_index + horizon - 1].get("target", {})).get("target_state_snapshot", {})
            )
            terminal_target = build_transition_target(
                dict(_as_dict(one_step_samples[start_index].get("input", {})).get("state_snapshot", {})),
                terminal_state_snapshot,
                result=_as_dict(terminal_entry.get("outcome", {})),
                reward=cumulative_reward,
                information_gain=horizon_information_gain,
                goal_progress_assessment=_as_dict(terminal_entry.get("goal_progress_assessment", {})),
            )
            horizon_targets.append(
                {
                    "horizon": horizon,
                    "action_sequence": [dict(item) for item in action_sequence[:horizon]],
                    "step_inputs": [dict(item) for item in step_inputs[:horizon]],
                    "step_targets": [
                        dict(_as_dict(one_step_samples[start_index + inner].get("target", {})))
                        for inner in range(horizon)
                    ],
                    "terminal_target": terminal_target,
                }
            )
        rollout_rows.append(
            {
                "input": dict(_as_dict(one_step_samples[start_index].get("input", {}))),
                "rollout_targets": horizon_targets,
                "metadata": dict(_as_dict(one_step_samples[start_index].get("metadata", {}))),
            }
        )
    return rollout_rows


def _prediction_from_action_summary(
    model: "BucketedLearnedDynamicsModel",
    state_snapshot: Dict[str, Any],
    action_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    synthetic_action = {
        "kind": "call_tool",
        "payload": {"tool_args": {"function_name": str(action_summary.get("function_name", "wait") or "wait")}},
        "_candidate_meta": {
            "anchor_ref": str(action_summary.get("target_anchor_ref", "") or ""),
            "target_family": str(action_summary.get("target_family", "") or ""),
            "action_family": str(action_summary.get("action_family", "") or ""),
            "object_color": action_summary.get("object_color"),
            "goal_progress_relation_type": str(action_summary.get("relation_type", "") or ""),
            "goal_progress_controller_anchor": bool(action_summary.get("controller_anchor", False)),
            "goal_progress_controller_supported_goal_anchor": bool(action_summary.get("controller_supported_goal_anchor", False)),
            "goal_progress_relation_anchor_match": bool(action_summary.get("relation_anchor_match", False)),
            "goal_progress_gap_closing_relation_anchor": bool(action_summary.get("gap_closing_relation_anchor", False)),
            "goal_progress_gap_closing_preferred_goal": bool(action_summary.get("gap_closing_preferred_goal", False)),
        },
    }
    return model.predict(state_snapshot, synthetic_action)


def _apply_prediction_to_snapshot(state_snapshot: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    next_snapshot = json.loads(json.dumps(state_snapshot))
    predicted_delta = _as_dict(prediction.get("next_object_graph_delta", {}))
    object_graph = _as_dict(next_snapshot.get("object_graph", {}))
    object_rows = [
        _as_dict(row)
        for row in _as_list(object_graph.get("objects", []))
        if isinstance(row, dict)
    ]
    object_by_id = {
        str(row.get("persistent_object_id", row.get("object_id", "")) or ""): row
        for row in object_rows
        if str(row.get("persistent_object_id", row.get("object_id", "")) or "")
    }
    for object_id in [
        str(item)
        for item in _as_list(_as_dict(prediction.get("next_object_level_delta", {})).get("removed_object_refs", []))
        if str(item or "")
    ]:
        row = object_by_id.pop(object_id, None)
        if row in object_rows:
            object_rows.remove(row)
    for row in _as_list(prediction.get("next_object_state_prediction", [])):
        row_dict = _as_dict(row)
        object_id = str(row_dict.get("persistent_object_id", "") or "")
        if not object_id:
            continue
        target = object_by_id.get(object_id)
        if target is None:
            target = {
                "persistent_object_id": object_id,
                "object_id": object_id,
            }
            object_rows.append(target)
            object_by_id[object_id] = target
        if "object_type" in row_dict and str(row_dict.get("object_type", "") or ""):
            target["object_type"] = str(row_dict.get("object_type", "") or "")
        if row_dict.get("after_color") is not None:
            target["color"] = row_dict.get("after_color")
        after_centroid = _as_dict(row_dict.get("after_centroid", {}))
        if after_centroid:
            target["centroid"] = {
                "x": _as_int(after_centroid.get("x", 0), 0),
                "y": _as_int(after_centroid.get("y", 0), 0),
            }
        after_bbox = _normalize_bbox(row_dict.get("after_bbox", {}))
        if after_bbox:
            target["bbox"] = after_bbox
        if "after_relation_degree" in row_dict:
            target["relation_degree"] = _as_int(row_dict.get("after_relation_degree", 0), 0)
        if "after_actionable_bucket" in row_dict:
            target["actionable_bucket"] = _as_int(row_dict.get("after_actionable_bucket", 0), 0)
        if "lineage_event" in row_dict:
            target["lineage_event"] = str(row_dict.get("lineage_event", "") or "")
    relation_rows = [
        _as_dict(row)
        for row in _as_list(object_graph.get("relations", []))
        if isinstance(row, dict)
    ]
    relation_set = {
        (
            str(row.get("relation_type", "") or ""),
            str(row.get("source_object_id", "") or ""),
            str(row.get("target_object_id", "") or ""),
        )
        for row in relation_rows
        if str(row.get("relation_type", "") or "")
        and str(row.get("source_object_id", "") or "")
        and str(row.get("target_object_id", "") or "")
    }
    relation_prediction = _as_dict(prediction.get("next_relation_state_prediction", {}))
    for row in _as_list(relation_prediction.get("removed_edges", [])):
        row_dict = _as_dict(row)
        relation_set.discard(
            (
                str(row_dict.get("relation_type", "") or ""),
                str(row_dict.get("source_object_id", "") or ""),
                str(row_dict.get("target_object_id", "") or ""),
            )
        )
    for row in _as_list(relation_prediction.get("added_edges", [])):
        row_dict = _as_dict(row)
        edge = (
            str(row_dict.get("relation_type", "") or ""),
            str(row_dict.get("source_object_id", "") or ""),
            str(row_dict.get("target_object_id", "") or ""),
        )
        if all(edge):
            relation_set.add(edge)
    relation_rows = [
        {"relation_type": relation_type, "source_object_id": source, "target_object_id": target}
        for relation_type, source, target in sorted(relation_set)
    ]
    object_graph["objects"] = object_rows
    object_graph["relations"] = relation_rows
    if prediction.get("next_frame_signature") is not None:
        object_graph["world_state_signature"] = str(prediction.get("next_frame_signature", "") or "")
    object_graph["object_count"] = max(0, _as_int(object_graph.get("object_count", 0), 0) + int(round(_as_float(predicted_delta.get("object_count_delta", 0.0), 0.0))))
    object_graph["relation_count"] = max(0, _as_int(object_graph.get("relation_count", 0), 0) + int(round(_as_float(predicted_delta.get("relation_count_delta", 0.0), 0.0))))
    object_graph["intervention_target_count"] = max(0, _as_int(object_graph.get("intervention_target_count", 0), 0) + int(round(_as_float(predicted_delta.get("intervention_target_delta", 0.0), 0.0))))
    object_graph["mean_actionable_score"] = round(
        _clamp01(_as_float(object_graph.get("mean_actionable_score", 0.0), 0.0) + _as_float(predicted_delta.get("mean_actionable_delta", 0.0), 0.0), 0.0),
        4,
    )
    object_graph["object_count"] = len(object_rows)
    object_graph["relation_count"] = len(relation_rows)
    hidden_state = _as_dict(next_snapshot.get("hidden_state", {}))
    hidden_state["phase"] = str(prediction.get("next_phase", hidden_state.get("phase", "unknown")) or hidden_state.get("phase", "unknown"))
    hidden_state["expected_next_phase"] = str(prediction.get("next_phase", hidden_state.get("expected_next_phase", hidden_state.get("phase", "unknown"))) or hidden_state.get("phase", "unknown"))
    hidden_state["phase_confidence"] = round(max(_as_float(hidden_state.get("phase_confidence", 0.0), 0.0), _as_float(prediction.get("confidence", 0.0), 0.0)), 4)
    hidden_state["predicted_changed_cell_count"] = len(_as_list(prediction.get("next_changed_cell_signature", [])))
    next_snapshot["object_graph"] = object_graph
    next_snapshot["hidden_state"] = hidden_state
    next_snapshot["phase"] = str(prediction.get("next_phase", next_snapshot.get("phase", "unknown")) or next_snapshot.get("phase", "unknown"))
    next_snapshot["numeric_features"] = {
        **_as_dict(next_snapshot.get("numeric_features", {})),
        "object_count": object_graph.get("object_count", 0),
        "relation_count": object_graph.get("relation_count", 0),
        "intervention_target_count": object_graph.get("intervention_target_count", 0),
        "mean_actionable_score": object_graph.get("mean_actionable_score", 0.0),
        "predicted_changed_cell_count": len(_as_list(prediction.get("next_changed_cell_signature", []))),
    }
    next_snapshot["predicted_changed_cell_signature"] = _as_list(prediction.get("next_changed_cell_signature", []))
    return next_snapshot


def evaluate_multistep_rollouts(
    model: "BucketedLearnedDynamicsModel",
    rollout_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    teacher_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    free_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rollout_rows:
        row_dict = _as_dict(row)
        for target_row in _as_list(row_dict.get("rollout_targets", [])):
            target_dict = _as_dict(target_row)
            horizon = _as_int(target_dict.get("horizon", 1), 1)
            step_inputs = [_as_dict(item) for item in _as_list(target_dict.get("step_inputs", [])) if isinstance(item, dict)]
            step_targets = [_as_dict(item) for item in _as_list(target_dict.get("step_targets", [])) if isinstance(item, dict)]
            terminal_target = _as_dict(target_dict.get("terminal_target", {}))
            if not step_inputs or not step_targets or not terminal_target:
                continue
            teacher_last_prediction: Dict[str, Any] = {}
            teacher_object_errors = 0.0
            teacher_relation_scores = 0.0
            teacher_object_scores = 0.0
            for step_input, step_target in zip(step_inputs, step_targets):
                prediction = _prediction_from_action_summary(model, _as_dict(step_input.get("state_snapshot", {})), _as_dict(step_input.get("action_summary", {})))
                if not prediction:
                    continue
                teacher_last_prediction = prediction
                comparison = compare_transition_prediction(prediction, step_target)
                teacher_object_errors += _as_float(comparison.get("object_level_delta_error", 0.0), 0.0)
                teacher_relation_scores += _as_float(comparison.get("relation_state_f1", 0.0), 0.0)
                teacher_object_scores += _as_float(comparison.get("object_state_accuracy", 0.0), 0.0)
            teacher_stats[horizon]["count"] += 1.0
            if teacher_last_prediction:
                teacher_stats[horizon]["phase_hits"] += float(
                    str(teacher_last_prediction.get("next_phase", "")) == str(terminal_target.get("next_phase", ""))
                )
                teacher_stats[horizon]["reward_hits"] += float(
                    str(teacher_last_prediction.get("reward_sign", "")) == str(terminal_target.get("reward_sign", ""))
                )
            teacher_stats[horizon]["object_error"] += teacher_object_errors / float(max(len(step_targets), 1))
            teacher_stats[horizon]["relation_edge_f1"] += teacher_relation_scores / float(max(len(step_targets), 1))
            teacher_stats[horizon]["object_accuracy"] += teacher_object_scores / float(max(len(step_targets), 1))

            simulated_snapshot = dict(_as_dict(step_inputs[0].get("state_snapshot", {})))
            free_last_prediction: Dict[str, Any] = {}
            for step_input in step_inputs:
                prediction = _prediction_from_action_summary(model, simulated_snapshot, _as_dict(step_input.get("action_summary", {})))
                if not prediction:
                    break
                free_last_prediction = prediction
                simulated_snapshot = _apply_prediction_to_snapshot(simulated_snapshot, prediction)
            free_stats[horizon]["count"] += 1.0
            if free_last_prediction:
                free_stats[horizon]["phase_hits"] += float(
                    str(free_last_prediction.get("next_phase", "")) == str(terminal_target.get("next_phase", ""))
                )
                free_stats[horizon]["reward_hits"] += float(
                    str(free_last_prediction.get("reward_sign", "")) == str(terminal_target.get("reward_sign", ""))
                )
                initial_graph = _as_dict(_as_dict(step_inputs[0].get("state_snapshot", {})).get("object_graph", {}))
                simulated_graph = _as_dict(simulated_snapshot.get("object_graph", {}))
                terminal_graph = _as_dict(terminal_target.get("next_object_graph_delta", {}))
                divergence = (
                    abs(
                        (_as_int(simulated_graph.get("object_count", 0), 0) - _as_int(initial_graph.get("object_count", 0), 0))
                        - _as_int(terminal_graph.get("object_count_delta", 0), 0)
                    )
                    + abs(
                        (_as_int(simulated_graph.get("relation_count", 0), 0) - _as_int(initial_graph.get("relation_count", 0), 0))
                        - _as_int(terminal_graph.get("relation_count_delta", 0), 0)
                    )
                )
                free_stats[horizon]["divergence"] += divergence
                free_stats[horizon]["diverged"] += float(divergence > 3.0)
    summary: Dict[str, Any] = {"teacher_forcing": {}, "free_running": {}}
    for horizon, stats in teacher_stats.items():
        count = max(1.0, stats.get("count", 0.0))
        summary["teacher_forcing"][str(horizon)] = {
            "k_step_phase_accuracy": round(stats.get("phase_hits", 0.0) / count, 4),
            "k_step_reward_accuracy": round(stats.get("reward_hits", 0.0) / count, 4),
            "k_step_object_delta_error": round(stats.get("object_error", 0.0) / count, 4),
            "k_step_object_accuracy": round(stats.get("object_accuracy", 0.0) / count, 4),
            "relation_edge_f1": round(stats.get("relation_edge_f1", 0.0) / count, 4),
            "relation_edge_change_f1_proxy": round(stats.get("relation_edge_f1", 0.0) / count, 4),
        }
    for horizon, stats in free_stats.items():
        count = max(1.0, stats.get("count", 0.0))
        summary["free_running"][str(horizon)] = {
            "k_step_phase_accuracy": round(stats.get("phase_hits", 0.0) / count, 4),
            "k_step_reward_accuracy": round(stats.get("reward_hits", 0.0) / count, 4),
            "long_rollout_divergence_rate": round(stats.get("diverged", 0.0) / count, 4),
            "mean_rollout_divergence": round(stats.get("divergence", 0.0) / count, 4),
        }
    return summary


def extract_training_samples_from_audit_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    audit = _as_dict(payload)
    raw_audit = _as_dict(audit.get("raw_audit", {}))
    episode_trace = _as_list(raw_audit.get("episode_trace", audit.get("episode_trace", [])))
    if not episode_trace:
        return []
    return extract_training_samples_from_episode_trace(episode_trace)

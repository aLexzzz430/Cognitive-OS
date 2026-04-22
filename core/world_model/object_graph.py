from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from core.world_model.entity_schema import EntityNode, ObjectGraphState, RelationEdge, StateSlot
from modules.world_model.object_binding import build_object_bindings
from modules.world_model.object_graph import build_object_graph as build_base_object_graph


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tokenize(*values: Any) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            out.update(_tokenize(*list(value)))
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        for raw in text.replace("::", "_").replace("-", "_").replace("/", "_").split():
            for token in raw.replace("_", " ").split():
                normalized = str(token or "").strip().lower()
                if normalized:
                    out.add(normalized)
    return out


def _action_focus(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {"anchor_ref": "", "binding_tokens": set()}
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    intervention_target = meta.get("intervention_target", {}) if isinstance(meta.get("intervention_target", {}), dict) else {}
    anchor_ref = str(intervention_target.get("anchor_ref", "") or meta.get("anchor_ref", "") or "").strip()
    binding_tokens = _tokenize(
        anchor_ref,
        intervention_target.get("target_kind", ""),
        meta.get("grounded_binding_tokens", []),
        meta.get("solver_semantic_labels", []),
        meta.get("solver_object_roles", []),
    )
    return {"anchor_ref": anchor_ref, "binding_tokens": binding_tokens}


def _scene_state(
    *,
    obs_before: Optional[Dict[str, Any]],
    obs_after: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    actual_transition: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    before = _as_dict(obs_before)
    after = _as_dict(obs_after)
    result_dict = _as_dict(result)
    transition = _as_dict(actual_transition)
    phase = str(
        after.get("phase", "")
        or transition.get("next_phase", "")
        or result_dict.get("belief_phase", "")
        or before.get("phase", "")
        or result_dict.get("status", "")
        or ""
    ).strip()
    signal_tokens = _tokenize(
        after.get("revealed_signal_token", ""),
        before.get("revealed_signal_token", ""),
        transition.get("observation_tokens", []),
    )
    counterevidence_tokens = _tokenize(
        after.get("counterevidence_token", ""),
        before.get("counterevidence_token", ""),
    )
    return {
        "phase": phase,
        "signal_tokens": sorted(signal_tokens),
        "counterevidence_tokens": sorted(counterevidence_tokens),
        "goal_revealed": bool(after.get("goal_revealed", before.get("goal_revealed", False))),
        "prerequisite_missing": bool(after.get("prerequisite_missing", before.get("prerequisite_missing", False))),
        "has_prerequisite": bool(after.get("has_prerequisite", before.get("has_prerequisite", False))),
        "recovery_required": bool(after.get("recovery_required", before.get("recovery_required", False))),
        "pending_countdown": int(after.get("pending_countdown", before.get("pending_countdown", 0)) or 0),
        "solved": bool(result_dict.get("solved", after.get("solved", before.get("solved", False)))),
        "state_changed": bool(transition.get("valid_state_change", result_dict.get("state_changed", False))),
        "observation_changed": bool(result_dict.get("observation_changed", False)),
    }


def _entity_slots(
    row: Dict[str, Any],
    *,
    scene_state: Dict[str, Any],
    intervention_targets: Sequence[str],
    focus_tokens: Set[str],
) -> List[StateSlot]:
    semantic_labels = _string_list(row.get("semantic_labels", []))
    role_labels = _string_list(row.get("role_labels", []))
    object_tokens = _tokenize(
        row.get("object_id", ""),
        semantic_labels,
        role_labels,
    )
    support_overlap = bool(object_tokens & focus_tokens)
    slots = [
        StateSlot(name="salience_score", value=round(_float(row.get("salience_score", 0.0), 0.0), 4)),
        StateSlot(name="actionable_score", value=round(_float(row.get("actionable_score", 0.0), 0.0), 4)),
        StateSlot(name="relation_degree", value=int(row.get("relation_degree", 0) or 0)),
        StateSlot(name="is_intervention_target", value=str(row.get("object_id", "") or "") in intervention_targets),
        StateSlot(name="focus_overlap", value=support_overlap),
        StateSlot(name="signal_visible", value=bool(object_tokens & set(scene_state.get("signal_tokens", [])))),
        StateSlot(name="counterevidence_visible", value=bool(object_tokens & set(scene_state.get("counterevidence_tokens", [])))),
    ]
    return slots


def build_runtime_object_graph(
    *,
    world_model_summary: Optional[Dict[str, Any]] = None,
    obs_before: Optional[Dict[str, Any]] = None,
    obs_after: Optional[Dict[str, Any]] = None,
    action: Optional[Dict[str, Any]] = None,
    actual_transition: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = _as_dict(world_model_summary)
    before = _as_dict(obs_before)
    after = _as_dict(obs_after)
    focus = _action_focus(action)
    scene_state = _scene_state(
        obs_before=before,
        obs_after=after,
        result=result,
        actual_transition=actual_transition,
    )

    graph = _as_dict(summary.get("object_graph", {}))
    if not graph:
        binding_summary = build_object_bindings(
            after if after else before,
            world_model_summary=summary,
        )
        graph = build_base_object_graph(
            after if after else before,
            world_model_summary=summary,
            object_bindings_summary=binding_summary,
        )

    intervention_targets = _string_list(graph.get("intervention_targets", []))
    objects: List[EntityNode] = []
    focus_object_ids: List[str] = []
    for row in _as_list(graph.get("objects", [])):
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("object_id", "") or "")
        if not entity_id:
            continue
        semantic_labels = _string_list(row.get("semantic_labels", []))
        role_labels = _string_list(row.get("role_labels", []))
        slots = _entity_slots(
            row,
            scene_state=scene_state,
            intervention_targets=intervention_targets,
            focus_tokens=set(focus.get("binding_tokens", set())),
        )
        object_tokens = _tokenize(entity_id, semantic_labels, role_labels)
        if focus.get("anchor_ref") and focus["anchor_ref"] == entity_id:
            focus_object_ids.append(entity_id)
        elif object_tokens & set(focus.get("binding_tokens", set())):
            focus_object_ids.append(entity_id)
        objects.append(
            EntityNode(
                entity_id=entity_id,
                entity_type=str(row.get("object_type", "entity") or "entity"),
                semantic_labels=semantic_labels,
                role_labels=role_labels,
                salience_score=round(_float(row.get("salience_score", 0.0), 0.0), 4),
                actionable_score=round(_float(row.get("actionable_score", 0.0), 0.0), 4),
                bbox=_as_dict(row.get("bbox", {})),
                centroid=_as_dict(row.get("centroid", {})),
                affordances=_string_list(before.get("visible_functions", []) or after.get("visible_functions", [])),
                state_slots=slots,
            )
        )

    relations: List[RelationEdge] = []
    for row in _as_list(graph.get("relations", [])):
        if not isinstance(row, dict):
            continue
        source = str(row.get("source_object_id", row.get("source", "")) or "")
        target = str(row.get("target_object_id", row.get("target", "")) or "")
        relation_type = str(row.get("relation_type", "") or "")
        if not source or not target or not relation_type:
            continue
        relations.append(
            RelationEdge(
                relation_id=str(row.get("relation_id", f"{relation_type}:{source}->{target}") or f"{relation_type}:{source}->{target}"),
                relation_type=relation_type,
                source_entity_id=source,
                target_entity_id=target,
                attributes={
                    key: value
                    for key, value in row.items()
                    if key not in {"relation_id", "relation_type", "source_object_id", "target_object_id", "source", "target"}
                },
            )
        )

    snapshot = ObjectGraphState(
        world_state_signature=str(
            graph.get("world_state_signature", "")
            or scene_state.get("phase", "")
            or summary.get("world_state_signature", "")
            or ""
        ),
        scene_state=scene_state,
        objects=objects,
        relations=relations,
        intervention_targets=intervention_targets,
        focus_object_ids=list(dict.fromkeys(focus_object_ids)),
    )
    return snapshot.to_dict()

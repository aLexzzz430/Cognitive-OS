from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from modules.world_model.canonical_state import summarize_value_world
from modules.world_model.object_binding import build_object_bindings


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _semantic_labels(obj: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for row in _as_list(obj.get("semantic_candidates", [])):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "") or "")
        if label and label not in labels:
            labels.append(label)
    return labels


def _role_labels(obj: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for row in _as_list(obj.get("role_candidates", [])):
        if not isinstance(row, dict):
            continue
        label = str(row.get("role", "") or "")
        if label and label not in labels:
            labels.append(label)
    return labels


def _binding_object_node(obj: Dict[str, Any]) -> Dict[str, Any]:
    node_id = str(obj.get("object_id", "") or obj.get("entity_id", "") or "")
    salience = _clamp01(obj.get("salience_score", obj.get("confidence", 0.0)), 0.0)
    actionable = _clamp01(obj.get("actionable_score", salience), salience)
    semantic_labels = _semantic_labels(obj)
    role_labels = _role_labels(obj)
    return {
        "object_id": node_id,
        "object_type": str(obj.get("object_type", obj.get("entity_type", "bound_object")) or "bound_object"),
        "salience_score": salience,
        "actionable_score": actionable,
        "bbox": _as_dict(obj.get("bbox", {})),
        "centroid": _as_dict(obj.get("centroid", {})),
        "semantic_labels": semantic_labels,
        "role_labels": role_labels,
    }


def _canonical_entity_node(entity: Dict[str, Any]) -> Dict[str, Any]:
    bbox = _as_dict(entity.get("bbox", {}))
    fill_ratio = _clamp01(entity.get("fill_ratio", 0.0), 0.0)
    area = max(1.0, float(entity.get("area", 1.0) or 1.0))
    salience = _clamp01(min(1.0, (area / 16.0) * 0.55 + fill_ratio * 0.45), 0.0)
    actionable = _clamp01(salience * 0.72 + (0.12 if entity.get("entity_type") == "connected_component" else 0.0), 0.0)
    return {
        "object_id": str(entity.get("entity_id", "") or ""),
        "object_type": str(entity.get("entity_type", "entity") or "entity"),
        "color": entity.get("color"),
        "salience_score": salience,
        "actionable_score": actionable,
        "bbox": bbox,
        "centroid": _as_dict(entity.get("centroid", {})),
        "semantic_labels": [str(entity.get("entity_type", "entity") or "entity")],
        "role_labels": [],
    }


def _relation_row(index: int, relation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    relation_type = str(relation.get("relation_type", relation.get("type", "")) or "")
    source = str(relation.get("source", relation.get("source_object_id", "")) or "")
    target = str(relation.get("target", relation.get("target_object_id", "")) or "")
    if not relation_type or not source or not target:
        return None
    row = dict(relation)
    row["relation_id"] = str(row.get("relation_id", f"rel_{index}") or f"rel_{index}")
    row["relation_type"] = relation_type
    row["source_object_id"] = source
    row["target_object_id"] = target
    return row


def build_object_graph(
    perception_summary: Any,
    *,
    world_model_summary: Optional[Dict[str, Any]] = None,
    object_bindings_summary: Optional[Dict[str, Any]] = None,
    max_objects: int = 12,
    max_relations: int = 24,
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    binding_summary = _as_dict(
        object_bindings_summary
        if isinstance(object_bindings_summary, dict)
        else summary.get("object_bindings_summary", {})
    )
    perception_payload = _as_dict(perception_summary)
    if not _as_list(binding_summary.get("objects", [])):
        try:
            binding_obs = (
                perception_payload
                if "perception" in perception_payload
                else {"perception": perception_payload}
            )
            fallback_bindings = build_object_bindings(binding_obs, world_model_summary=summary)
            if _as_list(fallback_bindings.get("objects", [])):
                binding_summary = fallback_bindings
        except Exception:
            pass
    canonical = summarize_value_world(
        perception_summary
        if perception_summary not in (None, "")
        else summary.get("perception", {})
    )

    objects: Dict[str, Dict[str, Any]] = {}
    for entity in _as_list(summary.get("world_entities", [])) or _as_list(canonical.get("world_entities", [])):
        if not isinstance(entity, dict):
            continue
        node = _canonical_entity_node(entity)
        if node["object_id"]:
            objects[node["object_id"]] = node

    for obj in _as_list(binding_summary.get("objects", [])):
        if not isinstance(obj, dict):
            continue
        node = _binding_object_node(obj)
        if not node["object_id"]:
            continue
        existing = objects.get(node["object_id"], {})
        merged = dict(existing)
        merged.update({key: value for key, value in node.items() if value not in (None, "", [], {})})
        merged["semantic_labels"] = list(dict.fromkeys([*existing.get("semantic_labels", []), *node.get("semantic_labels", [])]))
        merged["role_labels"] = list(dict.fromkeys([*existing.get("role_labels", []), *node.get("role_labels", [])]))
        merged["salience_score"] = max(float(existing.get("salience_score", 0.0) or 0.0), float(node.get("salience_score", 0.0) or 0.0))
        merged["actionable_score"] = max(float(existing.get("actionable_score", 0.0) or 0.0), float(node.get("actionable_score", 0.0) or 0.0))
        objects[node["object_id"]] = merged

    relation_rows: List[Dict[str, Any]] = []
    raw_relations = _as_list(summary.get("world_relations", [])) or _as_list(canonical.get("world_relations", []))
    for index, relation in enumerate(raw_relations[: max_relations]):
        if not isinstance(relation, dict):
            continue
        normalized = _relation_row(index, relation)
        if normalized is not None:
            relation_rows.append(normalized)

    relation_degree: Dict[str, int] = {}
    for relation in relation_rows:
        source = str(relation.get("source_object_id", "") or "")
        target = str(relation.get("target_object_id", "") or "")
        relation_degree[source] = relation_degree.get(source, 0) + 1
        relation_degree[target] = relation_degree.get(target, 0) + 1

    object_rows = sorted(
        objects.values(),
        key=lambda item: (
            -float(item.get("actionable_score", 0.0) or 0.0),
            -float(item.get("salience_score", 0.0) or 0.0),
            str(item.get("object_id", "") or ""),
        ),
    )[: max_objects]
    for row in object_rows:
        row["relation_degree"] = int(relation_degree.get(str(row.get("object_id", "") or ""), 0))

    predicted_phase = str(summary.get("predicted_phase", "") or "")
    hidden_state = _as_dict(summary.get("hidden_state", {}))
    phases = [
        {
            "phase": predicted_phase or str(hidden_state.get("phase", "exploring") or "exploring"),
            "confidence": _clamp01(summary.get("transition_confidence", hidden_state.get("phase_confidence", 0.0)), 0.0),
        }
    ]

    intervention_targets = [
        str(row.get("object_id", "") or "")
        for row in object_rows
        if float(row.get("actionable_score", 0.0) or 0.0) >= 0.34
    ][:6]

    return {
        "objects": object_rows,
        "relations": relation_rows,
        "phases": phases,
        "mechanism_families": list(dict.fromkeys(_as_list(summary.get("mechanism_families", [])))),
        "intervention_targets": intervention_targets,
        "scene_summary": _as_dict(
            summary.get(
                "world_scene_summary",
                binding_summary.get("scene_summary", canonical.get("world_scene_summary", {})),
            )
        ),
        "world_state_signature": str(summary.get("world_state_signature", canonical.get("world_state_signature", "")) or ""),
    }

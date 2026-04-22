from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_bbox(raw_bbox: Any) -> Dict[str, float]:
    bbox = _as_dict(raw_bbox)
    if not bbox:
        return {}
    x_min = _as_float(bbox.get("x_min", bbox.get("col_min", 0.0)), 0.0)
    x_max = _as_float(bbox.get("x_max", bbox.get("col_max", x_min)), x_min)
    y_min = _as_float(bbox.get("y_min", bbox.get("row_min", 0.0)), 0.0)
    y_max = _as_float(bbox.get("y_max", bbox.get("row_max", y_min)), y_min)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": max(0.0, x_max - x_min + 1.0),
        "height": max(0.0, y_max - y_min + 1.0),
    }


def _bbox_iou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    a = _normalize_bbox(left)
    b = _normalize_bbox(right)
    if not a or not b:
        return 0.0
    ix0 = max(a["x_min"], b["x_min"])
    ix1 = min(a["x_max"], b["x_max"])
    iy0 = max(a["y_min"], b["y_min"])
    iy1 = min(a["y_max"], b["y_max"])
    iw = max(0.0, ix1 - ix0 + 1.0)
    ih = max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    area_a = max(1.0, a["width"] * a["height"])
    area_b = max(1.0, b["width"] * b["height"])
    union = max(1.0, area_a + area_b - inter)
    return inter / union


def _centroid_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    a = _as_dict(left)
    b = _as_dict(right)
    dx = _as_float(a.get("x", 0.0), 0.0) - _as_float(b.get("x", 0.0), 0.0)
    dy = _as_float(a.get("y", 0.0), 0.0) - _as_float(b.get("y", 0.0), 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _safe_labels(row: Dict[str, Any], key: str, field: str) -> List[str]:
    values: List[str] = []
    for item in _as_list(row.get(key, [])):
        if not isinstance(item, dict):
            continue
        text = str(item.get(field, "") or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _object_signature(row: Dict[str, Any]) -> Dict[str, Any]:
    bbox = _normalize_bbox(row.get("bbox", {}))
    centroid = _as_dict(row.get("centroid", {}))
    semantic = _safe_labels(row, "semantic_labels", "label") or _safe_labels(row, "semantic_candidates", "label")
    roles = _safe_labels(row, "role_labels", "role") or _safe_labels(row, "role_candidates", "role")
    object_type = str(row.get("object_type", row.get("entity_type", "entity")) or "entity")
    return {
        "object_type": object_type,
        "color": row.get("color"),
        "bbox": bbox,
        "centroid": {
            "x": _as_float(centroid.get("x", 0.0), 0.0),
            "y": _as_float(centroid.get("y", 0.0), 0.0),
        },
        "area": max(1.0, _as_float(bbox.get("width", 0.0), 0.0) * _as_float(bbox.get("height", 0.0), 0.0)),
        "semantic": semantic[:4],
        "roles": roles[:3],
    }


def _object_match_score(current: Dict[str, Any], previous: Dict[str, Any], *, gap: int) -> float:
    cur = _object_signature(current)
    prev = _object_signature(previous)
    score = 0.0
    if cur["color"] == prev["color"] and cur["color"] not in (None, ""):
        score += 0.36
    elif cur["color"] not in (None, "") and prev["color"] not in (None, ""):
        score -= 0.12
    if cur["object_type"] == prev["object_type"]:
        score += 0.18
    semantic_overlap = len(set(cur["semantic"]) & set(prev["semantic"]))
    if semantic_overlap:
        score += min(0.12, semantic_overlap * 0.05)
    role_overlap = len(set(cur["roles"]) & set(prev["roles"]))
    if role_overlap:
        score += min(0.08, role_overlap * 0.04)
    iou = _bbox_iou(cur["bbox"], prev["bbox"])
    score += min(0.22, iou * 0.22)
    span = max(
        _as_float(cur["bbox"].get("width", 0.0), 0.0),
        _as_float(cur["bbox"].get("height", 0.0), 0.0),
        _as_float(prev["bbox"].get("width", 0.0), 0.0),
        _as_float(prev["bbox"].get("height", 0.0), 0.0),
        1.0,
    )
    centroid_bonus = max(0.0, 1.0 - (_centroid_distance(cur["centroid"], prev["centroid"]) / max(4.0, span * 2.2)))
    score += centroid_bonus * 0.2
    area_ratio = min(cur["area"], prev["area"]) / max(cur["area"], prev["area"], 1.0)
    score += area_ratio * 0.08
    if gap <= 1:
        score += 0.06
    elif gap <= 3:
        score += 0.03
    return max(0.0, min(1.0, score))


@dataclass
class _TrackState:
    persistent_object_id: str
    last_seen_tick: int
    age: int
    object_row: Dict[str, Any]
    lineage: List[Dict[str, Any]]


class PersistentObjectIdentityTracker:
    """Maintains a lightweight persistent identity spine over object-graph rows."""

    def __init__(self, *, match_threshold: float = 0.46, lineage_threshold: float = 0.34) -> None:
        self._match_threshold = float(match_threshold)
        self._lineage_threshold = float(lineage_threshold)
        self.reset()

    def reset(self) -> None:
        self._tracks: Dict[str, _TrackState] = {}
        self._retired_tracks: Dict[str, _TrackState] = {}
        self._next_id = 0

    def _new_track_id(self) -> str:
        track_id = f"track_{self._next_id}"
        self._next_id += 1
        return track_id

    def _candidate_pairs(self, objects: Sequence[Dict[str, Any]], tick: int) -> List[Tuple[float, int, str, int]]:
        pairs: List[Tuple[float, int, str, int]] = []
        for index, row in enumerate(objects):
            for track_id, track in self._tracks.items():
                gap = max(0, int(tick) - int(track.last_seen_tick))
                score = _object_match_score(row, track.object_row, gap=gap)
                if score >= self._lineage_threshold:
                    pairs.append((score, index, track_id, gap))
        pairs.sort(key=lambda item: (-item[0], item[3], item[2], item[1]))
        return pairs

    def annotate_graph(self, graph: Dict[str, Any], *, tick: int) -> Dict[str, Any]:
        graph_dict = _as_dict(graph)
        raw_objects = [dict(row) for row in _as_list(graph_dict.get("objects", [])) if isinstance(row, dict)]
        if not raw_objects:
            return {**graph_dict, "identity_summary": {"active_track_count": len(self._tracks), "new_track_count": 0}}

        candidate_pairs = self._candidate_pairs(raw_objects, tick)
        assigned_object_indexes: set[int] = set()
        assigned_track_ids: set[str] = set()
        assignments: Dict[int, Tuple[str, float, int]] = {}
        for score, index, track_id, gap in candidate_pairs:
            if score < self._match_threshold or index in assigned_object_indexes or track_id in assigned_track_ids:
                continue
            assignments[index] = (track_id, score, gap)
            assigned_object_indexes.add(index)
            assigned_track_ids.add(track_id)

        lineage_edges: List[Dict[str, Any]] = []
        summary = {
            "new_track_count": 0,
            "stable_track_count": 0,
            "reappeared_track_count": 0,
            "transformed_track_count": 0,
            "split_event_count": 0,
            "merge_event_count": 0,
        }

        all_candidate_lookup: Dict[int, List[Tuple[str, float, int]]] = {}
        for score, index, track_id, gap in candidate_pairs:
            all_candidate_lookup.setdefault(index, []).append((track_id, score, gap))

        annotated_objects: List[Dict[str, Any]] = []
        for index, row in enumerate(raw_objects):
            assigned = assignments.get(index)
            parent_ids: List[str] = []
            lineage_event = "new"
            confidence = 0.0
            reappearance_gap = 0
            if assigned is not None:
                track_id, confidence, gap = assigned
                track = self._tracks[track_id]
                parent_ids = [track_id]
                same_color = row.get("color") == track.object_row.get("color")
                same_type = str(row.get("object_type", "") or "") == str(track.object_row.get("object_type", "") or "")
                lineage_event = "stable" if same_color and same_type else "transform"
                if gap > 1:
                    lineage_event = "reappeared"
                    reappearance_gap = gap
                if lineage_event == "stable":
                    summary["stable_track_count"] += 1
                elif lineage_event == "reappeared":
                    summary["reappeared_track_count"] += 1
                else:
                    summary["transformed_track_count"] += 1
            else:
                candidates = all_candidate_lookup.get(index, [])
                parent_ids = [track_id for track_id, score, _ in candidates if score >= self._lineage_threshold][:2]
                confidence = candidates[0][1] if candidates else 0.0
                if len(parent_ids) >= 2:
                    lineage_event = "merge"
                    summary["merge_event_count"] += 1
                elif len(parent_ids) == 1 and parent_ids[0] in assigned_track_ids:
                    lineage_event = "split"
                    summary["split_event_count"] += 1
                else:
                    lineage_event = "new"
                    summary["new_track_count"] += 1
                track_id = self._new_track_id()
            next_track = _TrackState(
                persistent_object_id=track_id,
                last_seen_tick=int(tick),
                age=int((self._tracks.get(track_id).age if track_id in self._tracks else 0) + 1),
                object_row=dict(row),
                lineage=list(self._tracks.get(track_id).lineage if track_id in self._tracks else []),
            )
            next_track.lineage.append(
                {
                    "tick": int(tick),
                    "event": lineage_event,
                    "parent_ids": list(parent_ids),
                    "raw_object_id": str(row.get("object_id", "") or ""),
                }
            )
            next_track.object_row = dict(row)
            self._tracks[track_id] = next_track
            lineage_edges.append(
                {
                    "persistent_object_id": track_id,
                    "event": lineage_event,
                    "parent_ids": list(parent_ids),
                }
            )
            annotated = dict(row)
            annotated["persistent_object_id"] = track_id
            annotated["identity_confidence"] = round(float(confidence), 4)
            annotated["lineage_event"] = lineage_event
            annotated["lineage_parent_ids"] = list(parent_ids)
            annotated["reappearance_gap"] = int(reappearance_gap)
            annotated["track_age"] = int(next_track.age)
            annotated_objects.append(annotated)

        active_ids = {str(row.get("persistent_object_id", "") or "") for row in annotated_objects}
        for track_id in list(self._tracks.keys()):
            if track_id in active_ids:
                continue
            self._retired_tracks[track_id] = self._tracks.pop(track_id)

        annotated_relations: List[Dict[str, Any]] = []
        raw_to_persistent = {
            str(row.get("object_id", "") or ""): str(row.get("persistent_object_id", "") or "")
            for row in annotated_objects
            if str(row.get("object_id", "") or "") and str(row.get("persistent_object_id", "") or "")
        }
        for relation in _as_list(graph_dict.get("relations", [])):
            if not isinstance(relation, dict):
                continue
            normalized = dict(relation)
            source_raw = str(normalized.get("source_object_id", "") or "")
            target_raw = str(normalized.get("target_object_id", "") or "")
            if source_raw:
                normalized["source_persistent_object_id"] = raw_to_persistent.get(source_raw, "")
            if target_raw:
                normalized["target_persistent_object_id"] = raw_to_persistent.get(target_raw, "")
            annotated_relations.append(normalized)

        summary["active_track_count"] = len(active_ids)
        summary["retired_track_count"] = len(self._retired_tracks)
        return {
            **graph_dict,
            "objects": annotated_objects,
            "relations": annotated_relations,
            "identity_summary": summary,
            "lineage_edges": lineage_edges[:32],
        }

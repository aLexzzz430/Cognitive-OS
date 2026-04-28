"""Canonical world-state summarization helpers.

These helpers add a lightweight entity / relation / event spine without
hard-coding one benchmark into the rest of the world-model stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


Grid = List[List[int]]
Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _is_grid_like(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    width: Optional[int] = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _normalize_grid_payload(value: Any) -> Optional[Grid]:
    if _is_grid_like(value):
        return [list(map(int, row)) for row in value]
    if isinstance(value, list) and value and all(_is_grid_like(item) for item in value):
        return [list(map(int, row)) for row in value[0]]
    return None


def _background_color(grid: Grid) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[int(cell)] = counts.get(int(cell), 0) + 1
    if any(cell == 0 for row in grid for cell in row):
        return 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0] if counts else 0


def _bbox(points: Sequence[Point]) -> Optional[BBox]:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return (min(ys), max(ys), min(xs), max(xs))


def _bbox_area(box: BBox) -> int:
    return max(0, box[1] - box[0] + 1) * max(0, box[3] - box[2] + 1)


def _bbox_center(box: BBox) -> Tuple[float, float]:
    return ((box[2] + box[3]) / 2.0, (box[0] + box[1]) / 2.0)


def _extract_grid_entities(grid: Grid, *, max_entities: int = 12) -> List[Dict[str, Any]]:
    if not grid or not grid[0]:
        return []
    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    entities: List[Dict[str, Any]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            color = int(grid[row_idx][col_idx])
            if color == bg or (row_idx, col_idx) in seen:
                continue
            stack = [(row_idx, col_idx)]
            seen.add((row_idx, col_idx))
            cells: List[Point] = []
            while stack:
                cur_r, cur_c = stack.pop()
                cells.append((cur_c, cur_r))
                for nxt_r, nxt_c in (
                    (cur_r - 1, cur_c),
                    (cur_r + 1, cur_c),
                    (cur_r, cur_c - 1),
                    (cur_r, cur_c + 1),
                ):
                    if not (0 <= nxt_r < rows and 0 <= nxt_c < cols):
                        continue
                    if (nxt_r, nxt_c) in seen:
                        continue
                    if int(grid[nxt_r][nxt_c]) != color:
                        continue
                    seen.add((nxt_r, nxt_c))
                    stack.append((nxt_r, nxt_c))
            box = _bbox(cells)
            if box is None:
                continue
            area = len(cells)
            width = box[3] - box[2] + 1
            height = box[1] - box[0] + 1
            center_x, center_y = _bbox_center(box)
            entities.append(
                {
                    "entity_id": f"component_{len(entities)}",
                    "entity_type": "connected_component",
                    "modality": "grid",
                    "color": color,
                    "area": area,
                    "bbox": {
                        "row_min": box[0],
                        "row_max": box[1],
                        "col_min": box[2],
                        "col_max": box[3],
                        "width": width,
                        "height": height,
                    },
                    "centroid": {"x": round(center_x, 3), "y": round(center_y, 3)},
                    "fill_ratio": round(area / max(_bbox_area(box), 1), 4),
                }
            )
    entities.sort(key=lambda item: (-int(item.get("area", 0) or 0), str(item.get("entity_id", ""))))
    return entities[:max_entities]


def _pair_relation(left: Dict[str, Any], right: Dict[str, Any]) -> List[Dict[str, Any]]:
    left_box = left.get("bbox", {}) if isinstance(left.get("bbox", {}), dict) else {}
    right_box = right.get("bbox", {}) if isinstance(right.get("bbox", {}), dict) else {}
    if not left_box or not right_box:
        return []
    relations: List[Dict[str, Any]] = []
    if int(left_box.get("col_max", -1)) < int(right_box.get("col_min", -1)):
        relations.append({"relation_type": "left_of", "source": left["entity_id"], "target": right["entity_id"]})
    if int(left_box.get("row_max", -1)) < int(right_box.get("row_min", -1)):
        relations.append({"relation_type": "above", "source": left["entity_id"], "target": right["entity_id"]})
    if left_box.get("row_min") == right_box.get("row_min") and left_box.get("row_max") == right_box.get("row_max"):
        relations.append({"relation_type": "row_aligned", "source": left["entity_id"], "target": right["entity_id"]})
    if left_box.get("col_min") == right_box.get("col_min") and left_box.get("col_max") == right_box.get("col_max"):
        relations.append({"relation_type": "col_aligned", "source": left["entity_id"], "target": right["entity_id"]})
    horizontal_gap = max(
        0,
        max(
            int(right_box.get("col_min", 0)) - int(left_box.get("col_max", 0)) - 1,
            int(left_box.get("col_min", 0)) - int(right_box.get("col_max", 0)) - 1,
        ),
    )
    vertical_gap = max(
        0,
        max(
            int(right_box.get("row_min", 0)) - int(left_box.get("row_max", 0)) - 1,
            int(left_box.get("row_min", 0)) - int(right_box.get("row_max", 0)) - 1,
        ),
    )
    if horizontal_gap <= 1 or vertical_gap <= 1:
        relations.append(
            {
                "relation_type": "near",
                "source": left["entity_id"],
                "target": right["entity_id"],
                "distance_hint": min(horizontal_gap, vertical_gap),
            }
        )
    return relations


def summarize_grid_world(grid: Grid, *, max_entities: int = 12, max_relations: int = 24) -> Dict[str, Any]:
    entities = _extract_grid_entities(grid, max_entities=max_entities)
    relations: List[Dict[str, Any]] = []
    relation_counts: Dict[str, int] = {}
    for idx, left in enumerate(entities):
        for right in entities[idx + 1 :]:
            for relation in _pair_relation(left, right):
                relation_name = str(relation.get("relation_type", "") or "")
                relation_counts[relation_name] = relation_counts.get(relation_name, 0) + 1
                relations.append(relation)
                if len(relations) >= max_relations:
                    break
            if len(relations) >= max_relations:
                break
        if len(relations) >= max_relations:
            break

    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    non_background = sum(1 for row in grid for cell in row if int(cell) != bg)
    distinct_colors = sorted({int(cell) for row in grid for cell in row})
    return {
        "observed_modality": "grid",
        "scene_type": "spatial_grid",
        "world_state_signature": f"grid:{rows}x{cols}:bg={bg}:entities={len(entities)}",
        "world_scene_summary": {
            "rows": rows,
            "cols": cols,
            "background_color": bg,
            "non_background_cells": non_background,
            "distinct_colors": distinct_colors,
            "entity_count": len(entities),
            "relation_count": len(relations),
        },
        "world_entities": entities,
        "world_relations": relations,
        "world_relation_summary": relation_counts,
    }


def summarize_value_world(value: Any) -> Dict[str, Any]:
    grid = _normalize_grid_payload(value)
    if grid is not None:
        return summarize_grid_world(grid)

    if isinstance(value, dict):
        grid_child = None
        for _, child in value.items():
            if grid_child is None:
                maybe_grid = _normalize_grid_payload(child)
                if maybe_grid is not None:
                    grid_child = maybe_grid
        if grid_child is not None:
            summary = summarize_grid_world(grid_child)
            summary["container_type"] = "dict"
            summary["container_keys"] = sorted(str(key) for key in value.keys())[:16]
            return summary
        return {
            "observed_modality": "structured_dict",
            "scene_type": "record",
            "world_state_signature": f"dict:{len(value)}",
            "world_scene_summary": {"key_count": len(value), "keys": sorted(str(key) for key in value.keys())[:16]},
            "world_entities": [],
            "world_relations": [],
            "world_relation_summary": {},
        }

    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value[:8]):
            return {
                "observed_modality": "record_list",
                "scene_type": "collection",
                "world_state_signature": f"list:{len(value)}",
                "world_scene_summary": {"length": len(value), "item_type": "dict"},
                "world_entities": [],
                "world_relations": [],
                "world_relation_summary": {},
            }
        return {
            "observed_modality": "list",
            "scene_type": "collection",
            "world_state_signature": f"list:{len(value)}",
            "world_scene_summary": {"length": len(value)},
            "world_entities": [],
            "world_relations": [],
            "world_relation_summary": {},
        }

    return {
        "observed_modality": type(value).__name__,
        "scene_type": "primitive",
        "world_state_signature": f"{type(value).__name__}:{repr(value)[:32]}",
        "world_scene_summary": {"type": type(value).__name__},
        "world_entities": [],
        "world_relations": [],
        "world_relation_summary": {},
    }


def summarize_observation_world(obs: Any) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return summarize_value_world(obs)
    for key in ("frame", "grid", "observation"):
        if key not in obs:
            continue
        payload = obs.get(key)
        summary = summarize_value_world(payload)
        summary["source_key"] = key
        return summary
    return summarize_value_world(obs)


def summarize_world_transition(before_obs: Any, after_obs: Any) -> Dict[str, Any]:
    before = summarize_observation_world(before_obs)
    after = summarize_observation_world(after_obs)
    before_sig = str(before.get("world_state_signature", "") or "")
    after_sig = str(after.get("world_state_signature", "") or "")
    before_scene = before.get("world_scene_summary", {}) if isinstance(before.get("world_scene_summary", {}), dict) else {}
    after_scene = after.get("world_scene_summary", {}) if isinstance(after.get("world_scene_summary", {}), dict) else {}
    entity_count_delta = int(after_scene.get("entity_count", 0) or 0) - int(before_scene.get("entity_count", 0) or 0)
    relation_count_delta = int(after_scene.get("relation_count", 0) or 0) - int(before_scene.get("relation_count", 0) or 0)
    state_changed = before_sig != after_sig
    novelty_score = 0.0
    if state_changed:
        novelty_score += 0.45
    novelty_score += min(0.25, abs(entity_count_delta) * 0.08)
    novelty_score += min(0.20, abs(relation_count_delta) * 0.05)
    if before.get("observed_modality") != after.get("observed_modality"):
        novelty_score += 0.10
    novelty_score = max(0.0, min(1.0, novelty_score))
    return {
        "observed_modality": str(after.get("observed_modality", "") or ""),
        "before_signature": before_sig,
        "after_signature": after_sig,
        "state_changed": state_changed,
        "entity_count_delta": entity_count_delta,
        "relation_count_delta": relation_count_delta,
        "novelty_score": round(novelty_score, 4),
        "event_signature": (
            f"{before.get('observed_modality', 'unknown')}=>{after.get('observed_modality', 'unknown')}:"
            f"changed={int(state_changed)}:entities={entity_count_delta}:relations={relation_count_delta}"
        ),
    }

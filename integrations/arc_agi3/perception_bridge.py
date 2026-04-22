from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


Grid = List[List[int]]
Point = Tuple[int, int]


def _env_flag_enabled(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _coerce_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("array(") and text.endswith(")"):
            inner = text[len("array("):-1]
            dtype_marker = inner.find(", dtype=")
            if dtype_marker >= 0:
                inner = inner[:dtype_marker]
            try:
                parsed = ast.literal_eval(inner)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except Exception:
            converted = None
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
    if hasattr(value, "shape") and hasattr(value, "__iter__"):
        try:
            return list(value)
        except Exception:
            return None
    return None


def _to_int_grid(frame: Any) -> Optional[Grid]:
    raw_rows = _coerce_sequence(frame)
    if not raw_rows:
        return None
    rows: List[List[int]] = []
    for row in raw_rows:
        raw_cells = _coerce_sequence(row)
        if raw_cells is None:
            return None
        parsed_row: List[int] = []
        for cell in raw_cells:
            try:
                parsed_row.append(int(cell))
            except (TypeError, ValueError):
                return None
        rows.append(parsed_row)
    width = len(rows[0]) if rows else 0
    if width == 0:
        return None
    if any(len(row) != width for row in rows):
        return None
    return rows


def _extract_frames(obs: Dict[str, Any]) -> List[Grid]:
    raw_frames = obs.get("frame")
    if raw_frames is None:
        raw_frames = (obs.get("novel_api") or {}).get("frame") if isinstance(obs.get("novel_api"), dict) else None
    if raw_frames is None:
        return []

    direct_grid = _to_int_grid(raw_frames)
    if direct_grid is not None:
        return [direct_grid]

    raw_frame_list = _coerce_sequence(raw_frames)
    if raw_frame_list is None:
        return []

    frames: List[Grid] = []
    for item in raw_frame_list:
        grid = _to_int_grid(item)
        if grid is not None:
            frames.append(grid)
    return frames


def _bbox(points: Sequence[Point]) -> Optional[Dict[str, int]]:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "width": max(xs) - min(xs) + 1,
        "height": max(ys) - min(ys) + 1,
    }


def _background_color(grid: Grid) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return 0
    if 0 in counts:
        return 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _color_histogram(grid: Grid) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _connected_components(grid: Grid, background_color: int) -> List[Dict[str, Any]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen: set[Tuple[int, int]] = set()
    components: List[Dict[str, Any]] = []
    for y in range(rows):
        for x in range(cols):
            color = int(grid[y][x])
            if color == background_color or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells: List[Point] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if not (0 <= nx < cols and 0 <= ny < rows):
                        continue
                    if (nx, ny) in seen or int(grid[ny][nx]) != color:
                        continue
                    seen.add((nx, ny))
                    stack.append((nx, ny))
            bbox = _bbox(cells)
            if bbox is None:
                continue
            centroid_x = round(sum(px for px, _ in cells) / float(max(len(cells), 1)), 4)
            centroid_y = round(sum(py for _, py in cells) / float(max(len(cells), 1)), 4)
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": bbox,
                    "area": len(cells),
                    "centroid": {"x": centroid_x, "y": centroid_y},
                    "boundary_contact": bool(
                        bbox["x_min"] == 0
                        or bbox["y_min"] == 0
                        or bbox["x_max"] == cols - 1
                        or bbox["y_max"] == rows - 1
                    ),
                }
            )
    return components


def _clip_point(x: int, y: int, width: int, height: int) -> Point:
    return max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))


def _component_neighbor_points(component: Dict[str, Any], width: int, height: int) -> List[Point]:
    bbox = component.get("bbox", {}) if isinstance(component, dict) else {}
    if not isinstance(bbox, dict) or not bbox:
        return []
    x_min = int(bbox.get("x_min", 0) or 0)
    x_max = int(bbox.get("x_max", x_min) or x_min)
    y_min = int(bbox.get("y_min", 0) or 0)
    y_max = int(bbox.get("y_max", y_min) or y_min)
    cx = int(round((x_min + x_max) / 2.0))
    cy = int(round((y_min + y_max) / 2.0))
    candidates = [
        _clip_point(cx, max(0, y_min - 1), width, height),
        _clip_point(cx, min(height - 1, y_max + 1), width, height),
        _clip_point(max(0, x_min - 1), cy, width, height),
        _clip_point(min(width - 1, x_max + 1), cy, width, height),
    ]
    dedup: List[Point] = []
    for point in candidates:
        if point not in dedup:
            dedup.append(point)
    return dedup[:2]


def _score_component(
    component: Dict[str, Any],
    *,
    width: int,
    height: int,
    color_histogram: Dict[int, int],
    changed_pixels: Sequence[Point],
    total_pixels: int,
) -> Dict[str, Any]:
    bbox = component.get("bbox", {}) if isinstance(component, dict) else {}
    area = int(component.get("area", 0) or 0)
    color = int(component.get("color", 0) or 0)
    x_min = int(bbox.get("x_min", 0) or 0)
    x_max = int(bbox.get("x_max", x_min) or x_min)
    y_min = int(bbox.get("y_min", 0) or 0)
    y_max = int(bbox.get("y_max", y_min) or y_min)
    changed_overlap = sum(1 for x, y in changed_pixels if x_min <= x <= x_max and y_min <= y <= y_max)
    rarity = 1.0
    if color_histogram:
        rarity = 1.0 - (float(color_histogram.get(color, 0)) / float(max(total_pixels, 1)))
    area_score = min(1.0, area / float(max(4, int(total_pixels * 0.08))))
    changed_score = min(1.0, changed_overlap / float(max(area, 1))) if changed_overlap else 0.0
    boundary_penalty = 0.18 if bool(component.get("boundary_contact", False)) else 0.0
    salience = max(0.0, min(1.0, 0.34 + rarity * 0.34 + area_score * 0.18 + changed_score * 0.24 - boundary_penalty))
    actionable = max(0.0, min(1.0, salience * 0.72 + changed_score * 0.18 + (0.10 if not bool(component.get("boundary_contact", False)) else 0.0)))
    scored = dict(component)
    scored["rarity_score"] = round(rarity, 4)
    scored["changed_overlap"] = int(changed_overlap)
    scored["salience_score"] = round(salience, 4)
    scored["actionable_score"] = round(actionable, 4)
    return scored


def _component_keepalive_tags(component: Dict[str, Any], *, total_pixels: int) -> List[str]:
    bbox = component.get("bbox", {}) if isinstance(component.get("bbox", {}), dict) else {}
    area = int(component.get("area", 0) or 0)
    color = int(component.get("color", 0) or 0)
    rarity = float(component.get("rarity_score", 0.0) or 0.0)
    actionable = float(component.get("actionable_score", 0.0) or 0.0)
    changed_overlap = int(component.get("changed_overlap", 0) or 0)
    boundary_contact = bool(component.get("boundary_contact", False))
    bbox_width = int(bbox.get("width", 0) or 0)
    bbox_height = int(bbox.get("height", 0) or 0)
    small_area_limit = max(4, int(total_pixels * 0.03))
    small_object = bool(area > 0 and (area <= small_area_limit or bbox_width <= 2 or bbox_height <= 2))
    rare_object = bool(rarity >= 0.82)
    # Goal-family is a heuristic signal: changed overlap and highly actionable rare objects
    # tend to be the task-relevant anchors we should not silently prune away.
    goal_like = bool(changed_overlap > 0 or (actionable >= 0.78 and rare_object))

    tags: List[str] = []
    if color == 9 and small_object and not _env_flag_enabled("AGI_WORLD_V2_ABLATE_COLOR9_SMALL_TAG"):
        tags.append("color9_small")
    if rare_object and small_object:
        tags.append("rare_small")
    if boundary_contact and small_object:
        tags.append("boundary_touching_small")
    if boundary_contact and rare_object:
        tags.append("boundary_touching_rare")
    if goal_like:
        tags.append("goal_like")
    return tags


def _preserve_click_targets(targets: Sequence[Dict[str, Any]], *, base_limit: int = 12) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_points: set[Point] = set()

    def append_target(target: Dict[str, Any]) -> None:
        point = (int(target.get("x", 0) or 0), int(target.get("y", 0) or 0))
        if point in seen_points:
            return
        seen_points.add(point)
        selected.append(target)

    for target in list(targets[:base_limit]):
        append_target(target)

    if not any(str(item.get("role", "") or "") == "background_control" for item in selected):
        background_target = next((item for item in targets if str(item.get("role", "") or "") == "background_control"), None)
        if background_target is not None:
            append_target(background_target)

    for target in targets:
        if bool(target.get("preserve_target", False)):
            append_target(target)

    return selected


@dataclass
class _PerceptionMemory:
    previous_frames: List[Grid] = field(default_factory=list)

    def reset(self) -> None:
        self.previous_frames = []


class PerceptionBridge:
    """
    Lightweight ARC-AGI-3 perception adapter.

    It does not try to solve the game. It only normalizes raw frame observations
    into a stable summary that existing AGI_WORLD_V2 code can consume.
    """

    def __init__(self) -> None:
        self._memory = _PerceptionMemory()

    def reset(self) -> None:
        self._memory.reset()

    def observe(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        frames = _extract_frames(obs)
        previous = list(self._memory.previous_frames)
        self._memory.previous_frames = frames

        frame_summaries: List[Dict[str, Any]] = []
        active_pixels: List[Point] = []
        changed_pixels: List[Point] = []
        unique_colors = set()

        for index, grid in enumerate(frames):
            height = len(grid)
            width = len(grid[0]) if height else 0
            non_zero: List[Point] = []
            for y, row in enumerate(grid):
                for x, value in enumerate(row):
                    unique_colors.add(int(value))
                    if int(value) != 0:
                        non_zero.append((x, y))
                        active_pixels.append((x, y))
            bbox = _bbox(non_zero)
            frame_summaries.append(
                {
                    "index": index,
                    "height": height,
                    "width": width,
                    "non_zero_count": len(non_zero),
                    "bbox": bbox,
                }
            )

            if index < len(previous):
                prev = previous[index]
                prev_h = len(prev)
                prev_w = len(prev[0]) if prev_h else 0
                if prev_h == height and prev_w == width:
                    for y, row in enumerate(grid):
                        for x, value in enumerate(row):
                            if int(value) != int(prev[y][x]):
                                changed_pixels.append((x, y))

        available_actions = obs.get("available_actions", [])
        if not isinstance(available_actions, list):
            available_actions = []

        bbox_all = _bbox(active_pixels)
        bbox_changed = _bbox(changed_pixels)

        hotspot = None
        if bbox_changed:
            hotspot = {
                "x": (bbox_changed["x_min"] + bbox_changed["x_max"]) // 2,
                "y": (bbox_changed["y_min"] + bbox_changed["y_max"]) // 2,
                "source": "changed_pixels",
            }
        elif bbox_all:
            hotspot = {
                "x": (bbox_all["x_min"] + bbox_all["x_max"]) // 2,
                "y": (bbox_all["y_min"] + bbox_all["y_max"]) // 2,
                "source": "active_pixels",
            }

        primary_grid = frames[0] if frames else None
        grid_shape = {
            "height": frame_summaries[0]["height"] if frame_summaries else 0,
            "width": frame_summaries[0]["width"] if frame_summaries else 0,
        }
        background_color = 0
        color_histogram: Dict[int, int] = {}
        salient_objects: List[Dict[str, Any]] = []
        suggested_click_targets: List[Dict[str, Any]] = []
        counterfactual_points: List[Dict[str, int]] = []

        if primary_grid is not None and grid_shape["height"] > 0 and grid_shape["width"] > 0:
            background_color = _background_color(primary_grid)
            color_histogram = _color_histogram(primary_grid)
            components = _connected_components(primary_grid, background_color)
            total_pixels = grid_shape["height"] * grid_shape["width"]
            scored = [
                _score_component(
                    component,
                    width=grid_shape["width"],
                    height=grid_shape["height"],
                    color_histogram=color_histogram,
                    changed_pixels=changed_pixels,
                    total_pixels=total_pixels,
                )
                for component in components
            ]
            scored.sort(
                key=lambda item: (
                    -float(item.get("actionable_score", 0.0) or 0.0),
                    -float(item.get("salience_score", 0.0) or 0.0),
                    -int(item.get("area", 0) or 0),
                    int(item.get("color", 0) or 0),
                )
            )
            for idx, component in enumerate(scored):
                centroid = component.get("centroid", {}) if isinstance(component.get("centroid", {}), dict) else {}
                bbox = component.get("bbox", {}) if isinstance(component.get("bbox", {}), dict) else {}
                keepalive_tags = _component_keepalive_tags(component, total_pixels=total_pixels)
                salient_objects.append(
                    {
                        "object_id": f"percept_{idx}",
                        "color": int(component.get("color", 0) or 0),
                        "area": int(component.get("area", 0) or 0),
                        "bbox": dict(bbox),
                        "centroid": {"x": float(centroid.get("x", 0.0) or 0.0), "y": float(centroid.get("y", 0.0) or 0.0)},
                        "boundary_contact": bool(component.get("boundary_contact", False)),
                        "rarity_score": float(component.get("rarity_score", 0.0) or 0.0),
                        "changed_overlap": int(component.get("changed_overlap", 0) or 0),
                        "salience_score": float(component.get("salience_score", 0.0) or 0.0),
                        "actionable_score": float(component.get("actionable_score", 0.0) or 0.0),
                        "goal_like": "goal_like" in keepalive_tags,
                        "keepalive_tags": list(keepalive_tags),
                    }
                )

            seen_targets: Dict[Point, Dict[str, Any]] = {}
            def add_target(
                x: int,
                y: int,
                *,
                role: str,
                priority: float,
                reason: str,
                object_id: str = "",
                color: Optional[int] = None,
                target_family: str = "",
                probe_aliases: Optional[Sequence[str]] = None,
                keepalive_tags: Optional[Sequence[str]] = None,
            ) -> None:
                point = _clip_point(x, y, grid_shape["width"], grid_shape["height"])
                keepalive_tag_list = [str(item) for item in list(keepalive_tags or []) if str(item)]
                candidate = {
                    "x": int(point[0]),
                    "y": int(point[1]),
                    "role": str(role),
                    "priority": round(float(priority), 4),
                    "reason": str(reason),
                    "object_id": str(object_id or ""),
                    "color": int(color) if color is not None else None,
                    "target_family": str(target_family or role),
                    "probe_aliases": [str(item) for item in list(probe_aliases or []) if str(item)],
                    "keepalive_tags": keepalive_tag_list,
                    "preserve_target": bool(keepalive_tag_list),
                }
                if point in seen_targets:
                    existing = seen_targets[point]
                    existing_keepalive = [str(item) for item in list(existing.get("keepalive_tags", []) or []) if str(item)]
                    merged_keepalive: List[str] = []
                    for tag in existing_keepalive + keepalive_tag_list:
                        if tag not in merged_keepalive:
                            merged_keepalive.append(tag)
                    existing_probe_aliases = [str(item) for item in list(existing.get("probe_aliases", []) or []) if str(item)]
                    merged_probe_aliases: List[str] = []
                    for alias in existing_probe_aliases + candidate["probe_aliases"]:
                        if alias not in merged_probe_aliases:
                            merged_probe_aliases.append(alias)
                    if bool(candidate["preserve_target"]) and not bool(existing.get("preserve_target", False)):
                        for field in ("role", "reason", "object_id", "color", "target_family"):
                            existing[field] = candidate[field]
                    existing["priority"] = round(max(float(existing.get("priority", 0.0) or 0.0), float(candidate["priority"])), 4)
                    existing["probe_aliases"] = merged_probe_aliases
                    existing["keepalive_tags"] = merged_keepalive
                    existing["preserve_target"] = bool(existing.get("preserve_target", False) or candidate["preserve_target"])
                    return
                seen_targets[point] = candidate
                suggested_click_targets.append(candidate)

            if hotspot is not None:
                add_target(
                    int(hotspot["x"]),
                    int(hotspot["y"]),
                    role="changed_hotspot",
                    priority=0.98,
                    reason="center_of_changed_pixels",
                    target_family="changed_region",
                    probe_aliases=["probe_state_transition"],
                )

            for component in salient_objects:
                centroid = component.get("centroid", {}) if isinstance(component.get("centroid", {}), dict) else {}
                x = int(round(float(centroid.get("x", 0.0) or 0.0)))
                y = int(round(float(centroid.get("y", 0.0) or 0.0)))
                keepalive_tags = [str(item) for item in list(component.get("keepalive_tags", []) or []) if str(item)]
                add_target(
                    x,
                    y,
                    role="salient_object_center",
                    priority=0.91 + float(component.get("actionable_score", 0.0) or 0.0) * 0.10,
                    reason="center_of_salient_object",
                    object_id=str(component.get("object_id", "") or ""),
                    color=int(component.get("color", 0) or 0),
                    target_family="salient_object",
                    probe_aliases=["probe_high_impact_belief", "probe_state_transition"],
                    keepalive_tags=keepalive_tags,
                )
                raw_component = next((c for c in scored if int(c.get("color", 0) or 0) == int(component.get("color", 0) or 0) and c.get("centroid") == component.get("centroid")), None)
                if raw_component is not None:
                    for nx, ny in _component_neighbor_points(raw_component, grid_shape["width"], grid_shape["height"]):
                        add_target(
                            nx,
                            ny,
                            role="salient_object_neighbor",
                            priority=0.74,
                            reason="neighbor_of_salient_object",
                            object_id=str(component.get("object_id", "") or ""),
                            color=int(component.get("color", 0) or 0),
                            target_family="salient_object_neighbor",
                            probe_aliases=["probe_relation", "probe_state_transition"],
                            keepalive_tags=keepalive_tags,
                        )

            candidate_controls = [
                (0, 0),
                (grid_shape["width"] - 1, 0),
                (0, grid_shape["height"] - 1),
                (grid_shape["width"] - 1, grid_shape["height"] - 1),
                (grid_shape["width"] // 2, grid_shape["height"] // 2),
                (grid_shape["width"] // 4, grid_shape["height"] // 4),
                ((3 * grid_shape["width"]) // 4, (3 * grid_shape["height"]) // 4),
            ]
            occupied = {
                (int(point[0]), int(point[1]))
                for component in scored[:10]
                for point in list(component.get("cells", []))
            }
            for cx, cy in candidate_controls:
                point = _clip_point(cx, cy, grid_shape["width"], grid_shape["height"])
                if point in occupied:
                    continue
                counterfactual_points.append({"x": int(point[0]), "y": int(point[1])})
                add_target(
                    int(point[0]),
                    int(point[1]),
                    role="background_control",
                    priority=0.42,
                    reason="counterfactual_background_probe",
                    target_family="background_control",
                    probe_aliases=["probe_state_transition"],
                )
                if len(counterfactual_points) >= 3:
                    break

            suggested_click_targets.sort(
                key=lambda item: (
                    -float(item.get("priority", 0.0) or 0.0),
                    str(item.get("role", "") or ""),
                    int(item.get("y", 0) or 0),
                    int(item.get("x", 0) or 0),
                )
            )
            if suggested_click_targets:
                suggested_click_targets = _preserve_click_targets(suggested_click_targets, base_limit=12)

        return {
            "coordinate_type": "grid_absolute",
            "coordinate_confidence": 0.95 if frames else 0.0,
            "frame_count": len(frames),
            "frame_summaries": frame_summaries,
            "grid_shape": grid_shape,
            "unique_colors": sorted(unique_colors),
            "color_histogram": {str(key): int(value) for key, value in sorted(color_histogram.items())},
            "background_color": int(background_color),
            "active_pixel_count": len(active_pixels),
            "changed_pixel_count": len(changed_pixels),
            "active_bbox": bbox_all,
            "changed_bbox": bbox_changed,
            "suggested_hotspot": hotspot,
            "suggested_click_targets": suggested_click_targets,
            "counterfactual_points": counterfactual_points[:3],
            "salient_objects": salient_objects,
            "available_actions": [int(x) for x in available_actions if isinstance(x, int)],
            "dynamic_entities": ["grid_state"] if frames else [],
            "color_remapping_detected": False,
            "camera_motion_score": 0.0,
        }

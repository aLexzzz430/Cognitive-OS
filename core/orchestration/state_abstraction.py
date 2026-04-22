"""Lightweight state abstraction helpers for cross-task reasoning."""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple


Grid = List[List[int]]
GridKey = Tuple[Tuple[int, ...], ...]


def _grid_key(grid: Sequence[Sequence[int]]) -> GridKey:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def is_grid_like(value: Any) -> bool:
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


def _majority_color(grid: Grid) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _background_color(grid: Grid) -> int:
    if any(cell == 0 for row in grid for cell in row):
        return 0
    return _majority_color(grid)


def _bbox_of_non_background(grid: Grid, bg: int) -> Optional[Tuple[int, int, int, int]]:
    coords = [
        (r_idx, c_idx)
        for r_idx, row in enumerate(grid)
        for c_idx, cell in enumerate(row)
        if cell != bg
    ]
    if not coords:
        return None
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    return (min_r, max_r, min_c, max_c)


def _component_areas(grid: Grid, *, color_sensitive: bool) -> List[int]:
    if not grid or not grid[0]:
        return []
    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    areas: List[int] = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            start_color = grid[r_idx][c_idx]
            if (r_idx, c_idx) in seen or start_color == bg:
                continue
            stack = [(r_idx, c_idx)]
            seen.add((r_idx, c_idx))
            area = 0
            while stack:
                cur_r, cur_c = stack.pop()
                area += 1
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
                    if grid[nxt_r][nxt_c] == bg:
                        continue
                    if color_sensitive and grid[nxt_r][nxt_c] != start_color:
                        continue
                    seen.add((nxt_r, nxt_c))
                    stack.append((nxt_r, nxt_c))
            areas.append(area)
    return areas


def _extract_color_components(grid: Grid) -> List[Dict[str, Any]]:
    if not grid or not grid[0]:
        return []
    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    components: List[Dict[str, Any]] = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            color = grid[r_idx][c_idx]
            if (r_idx, c_idx) in seen or color == bg:
                continue
            stack = [(r_idx, c_idx)]
            seen.add((r_idx, c_idx))
            cells: List[Tuple[int, int]] = []
            while stack:
                cur_r, cur_c = stack.pop()
                cells.append((cur_r, cur_c))
                for nxt_r, nxt_c in (
                    (cur_r - 1, cur_c),
                    (cur_r + 1, cur_c),
                    (cur_r, cur_c - 1),
                    (cur_r, cur_c + 1),
                ):
                    if not (0 <= nxt_r < rows and 0 <= nxt_c < cols):
                        continue
                    if (nxt_r, nxt_c) in seen or grid[nxt_r][nxt_c] != color:
                        continue
                    seen.add((nxt_r, nxt_c))
                    stack.append((nxt_r, nxt_c))
            min_r = min(r for r, _ in cells)
            max_r = max(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            max_c = max(c for _, c in cells)
            components.append({
                "color": int(color),
                "area": len(cells),
                "bbox": (min_r, max_r, min_c, max_c),
            })
    return components


def _pairwise_component_stats(
    components: Sequence[Dict[str, Any]],
 ) -> Tuple[int, int, int, int]:
    row_overlap_pairs = 0
    col_overlap_pairs = 0
    strict_row_order_pairs = 0
    strict_col_order_pairs = 0
    for idx, left in enumerate(components):
        left_bbox = left.get("bbox", (0, 0, 0, 0))
        left_row_start, left_row_end = left_bbox[0], left_bbox[1]
        left_col_start, left_col_end = left_bbox[2], left_bbox[3]
        for right in components[idx + 1 :]:
            right_bbox = right.get("bbox", (0, 0, 0, 0))
            right_row_start, right_row_end = right_bbox[0], right_bbox[1]
            right_col_start, right_col_end = right_bbox[2], right_bbox[3]
            if max(left_row_start, right_row_start) <= min(left_row_end, right_row_end):
                row_overlap_pairs += 1
            if max(left_col_start, right_col_start) <= min(left_col_end, right_col_end):
                col_overlap_pairs += 1
            if left_row_end < right_row_start:
                strict_row_order_pairs += 1
            if left_col_end < right_col_start:
                strict_col_order_pairs += 1
    return (
        row_overlap_pairs,
        col_overlap_pairs,
        strict_row_order_pairs,
        strict_col_order_pairs,
    )


def _component_strip_axis(component_count: int, row_overlap_pairs: int, col_overlap_pairs: int) -> str:
    if component_count <= 1:
        return "singleton"
    pair_count = component_count * (component_count - 1) // 2
    if row_overlap_pairs == pair_count and col_overlap_pairs == 0:
        return "horizontal"
    if col_overlap_pairs == pair_count and row_overlap_pairs == 0:
        return "vertical"
    return "mixed"


def _component_uniform_gap(components: Sequence[Dict[str, Any]], strip_axis: str) -> int:
    if strip_axis not in {"horizontal", "vertical"} or len(components) <= 1:
        return -1
    if strip_axis == "horizontal":
        ordered = sorted(components, key=lambda component: (component["bbox"][2], component["bbox"][0]))
        gaps = [
            ordered[idx + 1]["bbox"][2] - ordered[idx]["bbox"][3] - 1
            for idx in range(len(ordered) - 1)
        ]
    else:
        ordered = sorted(components, key=lambda component: (component["bbox"][0], component["bbox"][2]))
        gaps = [
            ordered[idx + 1]["bbox"][0] - ordered[idx]["bbox"][1] - 1
            for idx in range(len(ordered) - 1)
        ]
    if not gaps:
        return 0
    if any(gap < 0 for gap in gaps):
        return -1
    return gaps[0] if all(gap == gaps[0] for gap in gaps[1:]) else -1


@lru_cache(maxsize=8192)
def _summarize_object_relation_graph_cached(grid_key: GridKey) -> Dict[str, Any]:
    components = _extract_color_components(grid_key)
    if not components:
        return {
            "component_count": 0,
            "row_overlap_pairs": 0,
            "col_overlap_pairs": 0,
            "strict_row_order_pairs": 0,
            "strict_col_order_pairs": 0,
            "strip_axis": "empty",
            "uniform_gap": -1,
        }
    (
        row_overlap_pairs,
        col_overlap_pairs,
        strict_row_order_pairs,
        strict_col_order_pairs,
    ) = _pairwise_component_stats(components)
    strip_axis = _component_strip_axis(len(components), row_overlap_pairs, col_overlap_pairs)
    return {
        "component_count": len(components),
        "row_overlap_pairs": row_overlap_pairs,
        "col_overlap_pairs": col_overlap_pairs,
        "strict_row_order_pairs": strict_row_order_pairs,
        "strict_col_order_pairs": strict_col_order_pairs,
        "strip_axis": strip_axis,
        "uniform_gap": _component_uniform_gap(components, strip_axis),
    }


def summarize_object_relation_graph(grid: Any) -> Dict[str, Any]:
    if not is_grid_like(grid):
        return {}
    return _summarize_object_relation_graph_cached(_grid_key(grid))


@lru_cache(maxsize=8192)
def _summarize_grid_state_cached(grid_key: GridKey) -> Dict[str, Any]:
    grid = grid_key
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    bg = _background_color(grid)
    counts: Dict[int, int] = {}
    non_background = 0
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
            if cell != bg:
                non_background += 1
    bbox = _bbox_of_non_background(grid, bg)
    component_areas = _component_areas(grid, color_sensitive=False)
    color_component_areas = _component_areas(grid, color_sensitive=True)
    object_relation_graph = _summarize_object_relation_graph_cached(grid_key)
    return {
        "rows": rows,
        "cols": cols,
        "area": rows * cols,
        "background_color": bg,
        "non_background_cells": non_background,
        "density": round(non_background / max(rows * cols, 1), 4),
        "distinct_colors": len(counts),
        "distinct_non_background_colors": len([color for color in counts if color != bg]),
        "component_count": len(component_areas),
        "color_component_count": len(color_component_areas),
        "largest_component_area": max(component_areas) if component_areas else 0,
        "largest_color_component_area": max(color_component_areas) if color_component_areas else 0,
        "bbox_rows": (bbox[1] - bbox[0] + 1) if bbox else 0,
        "bbox_cols": (bbox[3] - bbox[2] + 1) if bbox else 0,
        "color_histogram": dict(sorted(counts.items())),
        "object_relation_graph": object_relation_graph,
    }


def summarize_grid_state(grid: Any) -> Dict[str, Any]:
    if not is_grid_like(grid):
        return {}
    return _summarize_grid_state_cached(_grid_key(grid))


@lru_cache(maxsize=16384)
def _summarize_grid_transition_cached(src_key: GridKey, dst_key: GridKey) -> Dict[str, Any]:
    src_summary = _summarize_grid_state_cached(src_key)
    dst_summary = _summarize_grid_state_cached(dst_key)
    same_shape = src_summary["rows"] == dst_summary["rows"] and src_summary["cols"] == dst_summary["cols"]
    shape_relation = "same_shape"
    if not same_shape:
        if dst_summary["rows"] <= src_summary["rows"] and dst_summary["cols"] <= src_summary["cols"]:
            shape_relation = "cropped_or_shrunk"
        elif dst_summary["rows"] >= src_summary["rows"] and dst_summary["cols"] >= src_summary["cols"]:
            shape_relation = "expanded"
        else:
            shape_relation = "reshaped"
    return {
        "same_shape": same_shape,
        "shape_relation": shape_relation,
        "rows_delta": dst_summary["rows"] - src_summary["rows"],
        "cols_delta": dst_summary["cols"] - src_summary["cols"],
        "area_delta": dst_summary["area"] - src_summary["area"],
        "non_background_delta": dst_summary["non_background_cells"] - src_summary["non_background_cells"],
        "distinct_color_delta": dst_summary["distinct_colors"] - src_summary["distinct_colors"],
        "distinct_non_background_color_delta": dst_summary["distinct_non_background_colors"] - src_summary["distinct_non_background_colors"],
        "component_delta": dst_summary["component_count"] - src_summary["component_count"],
        "color_component_delta": dst_summary["color_component_count"] - src_summary["color_component_count"],
        "bbox_rows_delta": dst_summary["bbox_rows"] - src_summary["bbox_rows"],
        "bbox_cols_delta": dst_summary["bbox_cols"] - src_summary["bbox_cols"],
        "background_preserved": src_summary["background_color"] == dst_summary["background_color"],
    }


def summarize_grid_transition(src: Any, dst: Any) -> Dict[str, Any]:
    if not is_grid_like(src) or not is_grid_like(dst):
        return {}
    return _summarize_grid_transition_cached(_grid_key(src), _grid_key(dst))


def summarize_arc_transition_profile(train_examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    output_relation_summaries: List[Dict[str, Any]] = []
    for example in train_examples:
        if not isinstance(example, dict):
            continue
        transition = summarize_grid_transition(example.get("input"), example.get("output"))
        if transition:
            transitions.append(transition)
        output_summary = summarize_grid_state(example.get("output"))
        output_relations = output_summary.get("object_relation_graph", {}) if isinstance(output_summary.get("object_relation_graph", {}), dict) else {}
        if output_relations:
            output_relation_summaries.append(output_relations)
    if not transitions:
        return {}
    consistent_features: Dict[str, Any] = {}
    exemplar = transitions[0]
    for key, value in exemplar.items():
        if all(transition.get(key) == value for transition in transitions[1:]):
            consistent_features[key] = value
    consistent_output_relations: Dict[str, Any] = {}
    if output_relation_summaries:
        output_exemplar = output_relation_summaries[0]
        for key, value in output_exemplar.items():
            if all(summary.get(key) == value for summary in output_relation_summaries[1:]):
                consistent_output_relations[key] = value
    return {
        "example_count": len(transitions),
        "consistent_features": consistent_features,
        "consistent_output_relations": consistent_output_relations,
    }


def score_arc_transition_alignment(input_grid: Any, predicted_grid: Any, profile: Dict[str, Any]) -> float:
    if not profile:
        return 0.5
    transition = summarize_grid_transition(input_grid, predicted_grid)
    if not transition:
        return 0.0
    consistent = profile.get("consistent_features", {}) if isinstance(profile.get("consistent_features", {}), dict) else {}
    consistent_output_relations = profile.get("consistent_output_relations", {}) if isinstance(profile.get("consistent_output_relations", {}), dict) else {}
    if not consistent and not consistent_output_relations:
        return 0.5
    matches = 0
    total = 0
    for key, value in consistent.items():
        total += 1
        if transition.get(key) == value:
            matches += 1
    if consistent_output_relations:
        predicted_summary = summarize_grid_state(predicted_grid)
        predicted_relations = predicted_summary.get("object_relation_graph", {}) if isinstance(predicted_summary.get("object_relation_graph", {}), dict) else {}
        for key, value in consistent_output_relations.items():
            total += 1
            if predicted_relations.get(key) == value:
                matches += 1
    return matches / max(total, 1)


def summarize_value_structure(value: Any, *, _depth: int = 0) -> Dict[str, Any]:
    if _depth > 3:
        return {"type": type(value).__name__, "depth": _depth}
    if is_grid_like(value):
        summary = summarize_grid_state(value)
        return {
            "type": "grid",
            "depth": _depth + 2,
            "grid_summary": summary,
        }
    if isinstance(value, dict):
        child_depths: List[int] = []
        child_types: Dict[str, str] = {}
        for key, child in value.items():
            child_summary = summarize_value_structure(child, _depth=_depth + 1)
            child_types[str(key)] = str(child_summary.get("type", type(child).__name__))
            child_depths.append(int(child_summary.get("depth", _depth + 1) or (_depth + 1)))
        return {
            "type": "dict",
            "depth": max([_depth + 1, *child_depths]),
            "key_count": len(value),
            "keys": sorted(str(key) for key in value.keys()),
            "child_types": child_types,
        }
    if isinstance(value, list):
        child_summaries = [summarize_value_structure(child, _depth=_depth + 1) for child in value[:5]]
        child_depths = [int(summary.get("depth", _depth + 1) or (_depth + 1)) for summary in child_summaries]
        grid_like_items = sum(1 for summary in child_summaries if summary.get("type") == "grid")
        return {
            "type": "list",
            "depth": max([_depth + 1, *child_depths]) if child_depths else (_depth + 1),
            "length": len(value),
            "grid_like_items": grid_like_items,
            "item_types": [str(summary.get("type", "unknown")) for summary in child_summaries],
        }
    return {
        "type": type(value).__name__,
        "depth": _depth + 1,
    }


def summarize_action_state(action: Any) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
    kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
    function_name = str(tool_args.get("function_name", "wait") or "wait")
    kwargs_summary = summarize_value_structure(kwargs)
    grid_like_payloads = 0
    first_grid_summary: Dict[str, Any] = {}
    if is_grid_like(kwargs.get("grid")):
        grid_like_payloads = 1
        first_grid_summary = summarize_grid_state(kwargs["grid"])
    elif isinstance(kwargs.get("grids"), list):
        grids = [grid for grid in kwargs.get("grids", []) if is_grid_like(grid)]
        if grids:
            grid_like_payloads = len(grids)
            first_grid_summary = summarize_grid_state(grids[0])
    return {
        "function_name": function_name,
        "source": str(action.get("_source", "unknown") or "unknown"),
        "has_kwargs": bool(kwargs),
        "kwargs_keys": sorted(kwargs.keys()),
        "kwargs_summary": kwargs_summary,
        "max_value_depth": int(kwargs_summary.get("depth", 0) or 0),
        "grid_like_payloads": grid_like_payloads,
        "first_grid_summary": first_grid_summary,
        "object_signal": float(first_grid_summary.get("color_component_count", 0) or 0),
    }


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def summarize_cognitive_object_records(records: Any, *, limit: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(records, list):
        return []
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row = {
            "object_id": str(record.get("object_id", "") or ""),
            "object_type": str(record.get("object_type", "") or ""),
            "family": str(record.get("family", "") or ""),
            "summary": str(record.get("summary", "") or ""),
            "confidence": _clamp01(record.get("confidence", 0.0)),
            "status": str(record.get("status", "") or ""),
            "asset_status": str(record.get("asset_status", "") or ""),
            "surface_priority": _clamp01(record.get("surface_priority", 0.0)),
        }
        applicability = record.get("applicability", {})
        if isinstance(applicability, dict) and applicability:
            row["applicability"] = dict(applicability)
        failure_conditions = record.get("failure_conditions", [])
        if isinstance(failure_conditions, list) and failure_conditions:
            row["failure_conditions"] = [str(item) for item in failure_conditions[:4] if str(item or "")]
        identity_profile = record.get("identity_profile", {})
        if isinstance(identity_profile, dict) and identity_profile:
            row["identity_profile"] = dict(identity_profile)
        episode_refs = record.get("episode_refs", [])
        if isinstance(episode_refs, list) and episode_refs:
            row["episode_refs"] = [str(item) for item in episode_refs[:5] if str(item or "")]
        reuse_evidence = record.get("reuse_evidence", [])
        if isinstance(reuse_evidence, list) and reuse_evidence:
            row["reuse_evidence"] = [dict(item) for item in reuse_evidence[:4] if isinstance(item, dict)]
        if str(record.get("object_type", "") or "").strip() == "hypothesis":
            structured_payload = record.get("structured_payload", record.get("content", {}))
            structured_payload = structured_payload if isinstance(structured_payload, dict) else {}
            row["hypothesis_id"] = str(
                record.get("hypothesis_id", "")
                or record.get("object_id", "")
                or ""
            )
            row["hypothesis_type"] = str(record.get("hypothesis_type", record.get("family", "")) or "")
            row["posterior"] = _clamp01(record.get("posterior", record.get("confidence", 0.0)))
            row["status"] = str(structured_payload.get("status", row.get("status", "")) or row.get("status", ""))
            try:
                row["support_count"] = int(record.get("support_count", 0) or 0)
            except (TypeError, ValueError):
                row["support_count"] = 0
            try:
                row["contradiction_count"] = int(record.get("contradiction_count", 0) or 0)
            except (TypeError, ValueError):
                row["contradiction_count"] = 0
            row["source"] = str(record.get("source", "") or "")
            predictions = record.get("predictions", structured_payload.get("predictions", {}))
            if isinstance(predictions, dict) and predictions:
                row["predictions"] = copy.deepcopy(predictions)
                row["predicted_action_effects"] = copy.deepcopy(
                    predictions.get("predicted_action_effects", {})
                )
                row["predicted_action_effects_by_signature"] = copy.deepcopy(
                    predictions.get("predicted_action_effects_by_signature", {})
                )
                row["predicted_observation_tokens"] = [
                    str(item)
                    for item in list(predictions.get("predicted_observation_tokens", []) or [])[:6]
                    if str(item or "")
                ]
                row["predicted_phase_shift"] = str(predictions.get("predicted_phase_shift", "") or "")
                row["predicted_information_gain"] = _clamp01(
                    predictions.get("predicted_information_gain", 0.0)
                )
            falsifiers = record.get("falsifiers", structured_payload.get("falsifiers", []))
            if isinstance(falsifiers, list) and falsifiers:
                row["falsifiers"] = [str(item) for item in falsifiers[:4] if str(item or "")]
            conflicts_with = record.get("conflicts_with", structured_payload.get("conflicts_with", []))
            if isinstance(conflicts_with, list) and conflicts_with:
                row["conflicts_with"] = [str(item) for item in conflicts_with[:4] if str(item or "")]
            supporting_evidence = record.get("supporting_evidence", structured_payload.get("supporting_evidence", []))
            if isinstance(supporting_evidence, list) and supporting_evidence:
                row["supporting_evidence"] = [str(item) for item in supporting_evidence[:4] if str(item or "")]
            contradicting_evidence = record.get("contradicting_evidence", structured_payload.get("contradicting_evidence", []))
            if isinstance(contradicting_evidence, list) and contradicting_evidence:
                row["contradicting_evidence"] = [str(item) for item in contradicting_evidence[:4] if str(item or "")]
            tags = record.get("tags", structured_payload.get("tags", []))
            if isinstance(tags, list) and tags:
                row["tags"] = [str(item) for item in tags[:6] if str(item or "")]
        rows.append(row)
        if len(rows) >= max(0, int(limit)):
            break
    return rows


def summarize_evidence_queue(trace_tail: Any, *, limit: int = 6) -> List[Dict[str, Any]]:
    if not isinstance(trace_tail, list):
        return []
    queue: List[Dict[str, Any]] = []
    for entry in trace_tail[-max(0, int(limit)):]:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action_snapshot", {}) if isinstance(entry.get("action_snapshot", {}), dict) else {}
        if not action:
            action = entry.get("action", {}) if isinstance(entry.get("action", {}), dict) else {}
        outcome = entry.get("outcome", {}) if isinstance(entry.get("outcome", {}), dict) else {}
        queue.append({
            "tick": int(entry.get("tick", 0) or 0),
            "function_name": str(action.get("function_name", "") or ""),
            "reward": float(entry.get("reward", 0.0) or 0.0),
            "information_gain": _clamp01(entry.get("information_gain", 0.0)),
            "success": bool(outcome.get("success", float(entry.get("reward", 0.0) or 0.0) >= 0.0)),
            "belief_phase": str(
                outcome.get(
                    "belief_phase",
                    entry.get("observation", {}).get("belief_phase", "") if isinstance(entry.get("observation", {}), dict) else "",
                ) or ""
            ),
        })
    return queue


def summarize_uncertainty_vector(
    *,
    world_shift_risk: Any,
    retrieval_pressure: Any,
    probe_pressure: Any,
    active_hypotheses: Any,
    self_model_summary: Any,
) -> Dict[str, float]:
    hypothesis_rows = [item for item in (active_hypotheses or []) if isinstance(item, dict)]
    hypothesis_uncertainty = 0.5
    if hypothesis_rows:
        hypothesis_uncertainty = sum(
            1.0 - _clamp01(item.get("confidence", 0.5), default=0.5)
            for item in hypothesis_rows
        ) / len(hypothesis_rows)
    sm_summary = self_model_summary if isinstance(self_model_summary, dict) else {}
    self_model_uncertainty = 1.0 - _clamp01(sm_summary.get("global_reliability", 0.5), default=0.5)
    world_model_uncertainty = _clamp01(world_shift_risk, default=0.0)
    retrieval_uncertainty = _clamp01(retrieval_pressure, default=0.0)
    probe_uncertainty = _clamp01(probe_pressure, default=0.0)
    overall = (
        world_model_uncertainty
        + hypothesis_uncertainty
        + self_model_uncertainty
        + retrieval_uncertainty
        + probe_uncertainty
    ) / 5.0
    return {
        "world_model": round(world_model_uncertainty, 4),
        "hypothesis": round(hypothesis_uncertainty, 4),
        "self_model": round(self_model_uncertainty, 4),
        "retrieval": round(retrieval_uncertainty, 4),
        "probe": round(probe_uncertainty, 4),
        "overall": round(overall, 4),
    }


def summarize_goal_agenda(
    *,
    goal_stack: Any,
    continuity_snapshot: Any,
    plan_summary: Any,
) -> List[Dict[str, Any]]:
    agenda: List[Dict[str, Any]] = []
    seen = set()
    goal_stack_dict = goal_stack if isinstance(goal_stack, dict) else {}
    plan_summary_dict = plan_summary if isinstance(plan_summary, dict) else {}
    continuity_dict = continuity_snapshot if isinstance(continuity_snapshot, dict) else {}

    top_goal = continuity_dict.get("top_goal", {})
    top_goal_text = ""
    if isinstance(top_goal, dict):
        top_goal_text = str(top_goal.get("description", "") or "")
    if not top_goal_text:
        top_goal_text = str(goal_stack_dict.get("top_goal", "") or "")
    current_focus = str(goal_stack_dict.get("current_focus", "") or "")
    plan_goal = str(plan_summary_dict.get("goal", "") or "")
    subgoals = [
        str(item or "")
        for item in list(goal_stack_dict.get("subgoals", []) or [])
        if str(item or "")
    ]
    candidates = [
        ("top_goal", top_goal_text, "long_horizon"),
        ("current_focus", current_focus, "active"),
        ("plan_goal", plan_goal, "plan"),
    ]
    candidates.extend((f"subgoal_{idx}", item, "subgoal") for idx, item in enumerate(subgoals))
    for source, goal_text, horizon in candidates:
        normalized = goal_text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        agenda.append({
            "goal": normalized,
            "source": source,
            "horizon": horizon,
        })
    return agenda


def summarize_long_horizon_commitments(
    *,
    goal_stack: Any,
    continuity_snapshot: Any,
    plan_summary: Any,
    identity_state: Any,
) -> List[Dict[str, Any]]:
    commitments: List[Dict[str, Any]] = []
    agenda = summarize_goal_agenda(
        goal_stack=goal_stack,
        continuity_snapshot=continuity_snapshot,
        plan_summary=plan_summary,
    )
    for item in agenda[:3]:
        commitments.append({
            "commitment": str(item.get("goal", "") or ""),
            "source": str(item.get("source", "") or ""),
            "horizon": str(item.get("horizon", "long_horizon") or "long_horizon"),
        })
    identity = identity_state if isinstance(identity_state, dict) else {}
    identity_summary = str(identity.get("summary", "") or "")
    if identity_summary:
        commitments.append({
            "commitment": identity_summary,
            "source": "identity_state",
            "horizon": "identity_continuity",
        })
    return commitments[:4]


def summarize_workspace_budget_state(
    *,
    self_summary: Any,
    governance_context: Any,
    self_model_summary: Any,
    plan_summary: Any,
    retrieval_pressure: Any,
    probe_pressure: Any,
    uncertainty_vector: Any,
) -> Dict[str, Dict[str, Any]]:
    self_summary_dict = self_summary if isinstance(self_summary, dict) else {}
    governance_dict = governance_context if isinstance(governance_context, dict) else {}
    self_model_dict = self_model_summary if isinstance(self_model_summary, dict) else {}
    plan_summary_dict = plan_summary if isinstance(plan_summary, dict) else {}
    resource_budget = self_summary_dict.get("resource_budget", {})
    if not isinstance(resource_budget, dict):
        resource_budget = {}
    budget_state = governance_dict.get("budget_state", {})
    if not isinstance(budget_state, dict):
        budget_state = {}
    uncertainty = uncertainty_vector if isinstance(uncertainty_vector, dict) else {}
    capability_envelope = self_model_dict.get("capability_envelope", {})
    if not isinstance(capability_envelope, dict):
        capability_envelope = {}
    budget_multiplier = _clamp01(
        self_model_dict.get("budget_multiplier", capability_envelope.get("budget_multiplier", 1.0)),
        default=1.0,
    )
    compute_budget = {
        "time_budget": _clamp01(resource_budget.get("time_budget", 1.0), default=1.0),
        "energy_budget": _clamp01(resource_budget.get("energy_budget", 1.0), default=1.0),
        "compute_budget": _clamp01(resource_budget.get("compute_budget", 1.0), default=1.0),
        "resource_pressure": str(self_model_dict.get("resource_tightness", "normal") or "normal"),
        "budget_multiplier": budget_multiplier,
        "exploration_ratio_target": _clamp01(
            self_model_dict.get("exploration_ratio_target", capability_envelope.get("exploration_ratio_target", 0.5)),
            default=0.5,
        ),
    }
    compute_budget["compute_budget"] = _clamp01(
        compute_budget["compute_budget"] * budget_multiplier,
        default=compute_budget["compute_budget"],
    )
    safety_budget = {
        "risk_budget": _clamp01(resource_budget.get("risk_budget", 1.0), default=1.0),
        "energy": int(budget_state.get("energy", 100) or 100),
        "mode": str(governance_dict.get("mode", "normal") or "normal"),
        "teacher_off_escalation": bool(
            self_model_dict.get(
                "teacher_off_escalation",
                capability_envelope.get("teacher_off_escalation", False),
            )
        ),
    }
    remaining = compute_budget["compute_budget"] * max(0.0, 1.0 - (_clamp01(uncertainty.get("overall", 0.0)) * 0.25))
    deliberation_budget = {
        "remaining": round(remaining, 4),
        "retrieval_pressure": _clamp01(retrieval_pressure, default=0.0),
        "probe_pressure": _clamp01(probe_pressure, default=0.0),
        "plan_depth": max(0, int(plan_summary_dict.get("remaining_steps", 0) or 0)),
    }
    return {
        "compute_budget": compute_budget,
        "safety_budget": safety_budget,
        "deliberation_budget": deliberation_budget,
    }

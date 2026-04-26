"""Structured answer synthesis for open-ended task outputs."""

from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from modules.llm.capabilities import STRUCTURED_OUTPUT_ACTION_KWARGS
from modules.llm.gateway import ensure_llm_gateway
from core.orchestration.state_abstraction import (
    score_arc_transition_alignment,
    summarize_arc_transition_profile,
    summarize_grid_state,
)
from core.reasoning.arc_program_dsl import (
    normalize_arc_program_rows,
    serialize_arc_candidate_spec,
)
from core.reasoning.arc_output_critic import rank_arc_candidate_outputs
from core.reasoning.arc_refinement_loop import ArcRefinementConfig, run_arc_refinement_loop


Grid = List[List[int]]
ArcCandidateScore = Tuple[int, int, float, float, float, float, float, int, float]
FrozenGrid = Tuple[Tuple[int, ...], ...]
FrozenGridBatch = Tuple[FrozenGrid, ...]


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _identity(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _freeze_grid(grid: Grid) -> FrozenGrid:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def _freeze_grids(grids: Sequence[Grid]) -> FrozenGridBatch:
    return tuple(_freeze_grid(grid) for grid in grids)


def _thaw_grid(grid: FrozenGrid) -> Grid:
    return [list(row) for row in grid]


def _thaw_grids(grids: FrozenGridBatch) -> List[Grid]:
    return [_thaw_grid(grid) for grid in grids]


def _flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def _transpose(grid: Grid) -> Grid:
    if not grid or not grid[0]:
        return []
    return [list(row) for row in zip(*grid)]


def _rotate90(grid: Grid) -> Grid:
    return _flip_h(_transpose(grid))


def _rotate180(grid: Grid) -> Grid:
    return _flip_v(_flip_h(grid))


def _rotate270(grid: Grid) -> Grid:
    return _flip_v(_transpose(grid))


def _crop_nonzero_bbox(grid: Grid) -> Grid:
    coords = [
        (r_idx, c_idx)
        for r_idx, row in enumerate(grid)
        for c_idx, cell in enumerate(row)
        if cell != 0
    ]
    if not coords:
        return _identity(grid)
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    return [list(row[min_c : max_c + 1]) for row in grid[min_r : max_r + 1]]


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


def _crop_majority_bbox(grid: Grid) -> Grid:
    bg = _majority_color(grid)
    coords = [
        (r_idx, c_idx)
        for r_idx, row in enumerate(grid)
        for c_idx, cell in enumerate(row)
        if cell != bg
    ]
    if not coords:
        return _identity(grid)
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    return [list(row[min_c : max_c + 1]) for row in grid[min_r : max_r + 1]]


@lru_cache(maxsize=8192)
def _component_bboxes_cached(grid: FrozenGrid) -> Tuple[Tuple[int, int, int, int], ...]:
    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    boxes: List[Tuple[int, int, int, int]] = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            if (r_idx, c_idx) in seen or grid[r_idx][c_idx] == bg:
                continue
            stack = [(r_idx, c_idx)]
            seen.add((r_idx, c_idx))
            comp: List[Tuple[int, int]] = []
            while stack:
                cur_r, cur_c = stack.pop()
                comp.append((cur_r, cur_c))
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
                    seen.add((nxt_r, nxt_c))
                    stack.append((nxt_r, nxt_c))
            min_r = min(r for r, _ in comp)
            max_r = max(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            max_c = max(c for _, c in comp)
            boxes.append((min_r, max_r, min_c, max_c))
    return tuple(boxes)


def _component_bboxes(grid: Grid) -> List[Tuple[int, int, int, int]]:
    if not grid or not grid[0]:
        return []
    return list(_component_bboxes_cached(_freeze_grid(grid)))


@lru_cache(maxsize=8192)
def _color_component_bboxes_cached(grid: FrozenGrid) -> Tuple[Tuple[int, int, int, int], ...]:
    bg = _background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    boxes: List[Tuple[int, int, int, int]] = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            color = grid[r_idx][c_idx]
            if (r_idx, c_idx) in seen or color == bg:
                continue
            stack = [(r_idx, c_idx)]
            seen.add((r_idx, c_idx))
            comp: List[Tuple[int, int]] = []
            while stack:
                cur_r, cur_c = stack.pop()
                comp.append((cur_r, cur_c))
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
            min_r = min(r for r, _ in comp)
            max_r = max(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            max_c = max(c for _, c in comp)
            boxes.append((min_r, max_r, min_c, max_c))
    return tuple(boxes)


def _color_component_bboxes(grid: Grid) -> List[Tuple[int, int, int, int]]:
    if not grid or not grid[0]:
        return []
    return list(_color_component_bboxes_cached(_freeze_grid(grid)))


def _crop_box(grid: Grid, box: Tuple[int, int, int, int]) -> Grid:
    min_r, max_r, min_c, max_c = box
    return [list(row[min_c : max_c + 1]) for row in grid[min_r : max_r + 1]]


def _select_box_from_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    mode: str,
) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    if mode == "largest":
        return max(boxes, key=lambda box: ((box[1] - box[0] + 1) * (box[3] - box[2] + 1), -box[0], -box[2]))
    if mode == "smallest":
        return min(boxes, key=lambda box: ((box[1] - box[0] + 1) * (box[3] - box[2] + 1), box[0], box[2]))
    if mode == "topmost":
        return min(boxes, key=lambda box: (box[0], box[2], box[1], box[3]))
    if mode == "bottommost":
        return max(boxes, key=lambda box: (box[1], -box[0], -box[2]))
    if mode == "leftmost":
        return min(boxes, key=lambda box: (box[2], box[0], box[3], box[1]))
    if mode == "rightmost":
        return max(boxes, key=lambda box: (box[3], -box[2], -box[0]))
    return None


def _select_component_box(grid: Grid, mode: str) -> Optional[Tuple[int, int, int, int]]:
    return _select_box_from_boxes(_component_bboxes(grid), mode)


def _crop_component(grid: Grid, mode: str) -> Grid:
    box = _select_component_box(grid, mode)
    if box is None:
        return _identity(grid)
    return _crop_box(grid, box)


def _crop_color_component(grid: Grid, mode: str) -> Grid:
    box = _select_box_from_boxes(_color_component_bboxes(grid), mode)
    if box is None:
        return _identity(grid)
    return _crop_box(grid, box)


@lru_cache(maxsize=8192)
def _extract_color_components_cached(grid: FrozenGrid) -> Tuple[Dict[str, Any], ...]:
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
            cell_set = set(cells)
            crop: Grid = []
            for crop_r in range(min_r, max_r + 1):
                crop_row: List[int] = []
                for crop_c in range(min_c, max_c + 1):
                    crop_row.append(int(color) if (crop_r, crop_c) in cell_set else -1)
                crop.append(crop_row)
            components.append({
                "color": int(color),
                "bbox": (min_r, max_r, min_c, max_c),
                "crop": crop,
                "area": len(cells),
            })
    return tuple(components)


def _extract_color_components(grid: Grid) -> List[Dict[str, Any]]:
    if not grid or not grid[0]:
        return []
    return list(_extract_color_components_cached(_freeze_grid(grid)))


def _materialize_component_crop(crop: Grid, *, background_color: int) -> Grid:
    return [
        [int(background_color) if int(cell) == -1 else int(cell) for cell in row]
        for row in crop
    ]


def _component_center(component: Dict[str, Any]) -> Tuple[float, float]:
    min_r, max_r, min_c, max_c = component.get("bbox", (0, 0, 0, 0))
    return ((float(min_r) + float(max_r)) / 2.0, (float(min_c) + float(max_c)) / 2.0)


def _rank_anchor_marker_candidates(
    components: Sequence[Dict[str, Any]],
    output_grid: Grid,
) -> List[Dict[str, Any]]:
    output_colors = {int(cell) for row in output_grid for cell in row}
    return sorted(
        components,
        key=lambda component: (
            0 if int(component.get("color", -1)) not in output_colors else 1,
            int(component.get("area", 0) or 0),
            int(component.get("bbox", (0, 0, 0, 0))[0]),
            int(component.get("bbox", (0, 0, 0, 0))[2]),
        ),
    )


def _select_anchor_relation_components(
    components: Sequence[Dict[str, Any]],
    marker: Dict[str, Any],
    *,
    relation_mode: str,
) -> Optional[List[Dict[str, Any]]]:
    marker_min_r, marker_max_r, marker_min_c, marker_max_c = marker.get("bbox", (0, 0, 0, 0))
    marker_center_r, marker_center_c = _component_center(marker)

    def _left_candidates() -> List[Tuple[float, float, int, int, Dict[str, Any]]]:
        ranked: List[Tuple[float, float, int, int, Dict[str, Any]]] = []
        for component in components:
            min_r, max_r, min_c, max_c = component.get("bbox", (0, 0, 0, 0))
            if max_c >= marker_min_c:
                continue
            center_r, _center_c = _component_center(component)
            ranked.append((float(marker_min_c - max_c), abs(center_r - marker_center_r), min_c, min_r, component))
        return ranked

    def _right_candidates() -> List[Tuple[float, float, int, int, Dict[str, Any]]]:
        ranked: List[Tuple[float, float, int, int, Dict[str, Any]]] = []
        for component in components:
            min_r, max_r, min_c, max_c = component.get("bbox", (0, 0, 0, 0))
            if min_c <= marker_max_c:
                continue
            center_r, _center_c = _component_center(component)
            ranked.append((float(min_c - marker_max_c), abs(center_r - marker_center_r), min_r, min_c, component))
        return ranked

    def _above_candidates() -> List[Tuple[float, float, int, int, Dict[str, Any]]]:
        ranked: List[Tuple[float, float, int, int, Dict[str, Any]]] = []
        for component in components:
            min_r, max_r, min_c, max_c = component.get("bbox", (0, 0, 0, 0))
            if max_r >= marker_min_r:
                continue
            _center_r, center_c = _component_center(component)
            ranked.append((float(marker_min_r - max_r), abs(center_c - marker_center_c), min_r, min_c, component))
        return ranked

    def _below_candidates() -> List[Tuple[float, float, int, int, Dict[str, Any]]]:
        ranked: List[Tuple[float, float, int, int, Dict[str, Any]]] = []
        for component in components:
            min_r, max_r, min_c, max_c = component.get("bbox", (0, 0, 0, 0))
            if min_r <= marker_max_r:
                continue
            _center_r, center_c = _component_center(component)
            ranked.append((float(min_r - marker_max_r), abs(center_c - marker_center_c), min_r, min_c, component))
        return ranked

    left_candidates = sorted(_left_candidates())
    right_candidates = sorted(_right_candidates())
    above_candidates = sorted(_above_candidates())
    below_candidates = sorted(_below_candidates())

    if relation_mode == "select_left_of_marker":
        return [left_candidates[0][4]] if left_candidates else None
    if relation_mode == "select_right_of_marker":
        return [right_candidates[0][4]] if right_candidates else None
    if relation_mode == "select_above_marker":
        return [above_candidates[0][4]] if above_candidates else None
    if relation_mode == "select_below_marker":
        return [below_candidates[0][4]] if below_candidates else None
    if relation_mode == "compose_horizontal_neighbors":
        if not left_candidates or not right_candidates:
            return None
        return [left_candidates[0][4], right_candidates[0][4]]
    if relation_mode == "compose_vertical_neighbors":
        if not above_candidates or not below_candidates:
            return None
        return [above_candidates[0][4], below_candidates[0][4]]
    return None


def _derive_color_map(src: Grid, dst: Grid) -> Optional[Dict[int, int]]:
    if len(src) != len(dst):
        return None
    mapping: Dict[int, int] = {}
    for src_row, dst_row in zip(src, dst):
        if len(src_row) != len(dst_row):
            return None
        for src_cell, dst_cell in zip(src_row, dst_row):
            if src_cell in mapping and mapping[src_cell] != dst_cell:
                return None
            mapping[src_cell] = dst_cell
    return mapping


def _apply_color_map(grid: Grid, color_map: Dict[int, int]) -> Grid:
    return [[int(color_map.get(cell, cell)) for cell in row] for row in grid]


def _compose(op_a: Callable[[Grid], Grid], op_b: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    return lambda grid: op_b(op_a(grid))


def _extract_local_patch(grid: Grid, row_idx: int, col_idx: int, radius: int) -> Tuple[Tuple[int, ...], ...]:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    patch: List[Tuple[int, ...]] = []
    for patch_r in range(row_idx - radius, row_idx + radius + 1):
        patch_row: List[int] = []
        for patch_c in range(col_idx - radius, col_idx + radius + 1):
            if 0 <= patch_r < rows and 0 <= patch_c < cols:
                patch_row.append(grid[patch_r][patch_c])
            else:
                patch_row.append(-1)
        patch.append(tuple(patch_row))
    return tuple(patch)


def _flatten_patch(patch: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
    return tuple(cell for row in patch for cell in row)


def _normalize_patch_signature(patch: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    color_ids: Dict[int, int] = {}
    next_color_id = 0
    normalized: List[Tuple[int, ...]] = []
    for row in patch:
        normalized_row: List[int] = []
        for cell in row:
            if cell == -1:
                normalized_row.append(-1)
                continue
            if cell not in color_ids:
                color_ids[cell] = next_color_id
                next_color_id += 1
            normalized_row.append(color_ids[cell])
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _grid_similarity(predicted: Any, target: Any) -> float:
    if not _is_grid(predicted) or not _is_grid(target):
        return 0.0
    pred_rows = len(predicted)
    pred_cols = len(predicted[0]) if predicted else 0
    tgt_rows = len(target)
    tgt_cols = len(target[0]) if target else 0
    if pred_rows == 0 or pred_cols == 0 or tgt_rows == 0 or tgt_cols == 0:
        return 0.0
    overlap_rows = min(pred_rows, tgt_rows)
    overlap_cols = min(pred_cols, tgt_cols)
    overlap_area = overlap_rows * overlap_cols
    if overlap_area == 0:
        return 0.0
    matches = 0
    for r_idx in range(overlap_rows):
        for c_idx in range(overlap_cols):
            if predicted[r_idx][c_idx] == target[r_idx][c_idx]:
                matches += 1
    overlap_match = matches / overlap_area
    area_penalty = overlap_area / max(pred_rows * pred_cols, tgt_rows * tgt_cols)
    return overlap_match * area_penalty


class StructuredAnswerSynthesizer:
    """Populate action kwargs for tasks that require structured open-ended answers."""

    _ANSWER_FN_PREFIXES = ("submit", "answer", "solve")
    _LOCAL_MACHINE_ATOMIC_FUNCTIONS = frozenset(
        {
            "repo_tree",
            "repo_find",
            "repo_grep",
            "file_read",
            "file_outline",
            "file_summary",
            "note_write",
            "hypothesis_add",
            "candidate_files_set",
            "candidate_files_update",
            "investigation_status",
            "apply_patch",
            "edit_replace_range",
            "edit_insert_after",
            "create_file",
            "delete_file",
            "run_test",
            "run_lint",
            "run_typecheck",
            "run_build",
            "read_run_output",
            "read_test_failure",
        }
    )
    _ARC_KEY = "arc_task"
    _ARC_SIMULATION_MAX_DEPTH = 3
    _ARC_SIMULATION_BEAM_WIDTH = 8
    _ARC_SIMULATION_KEEP_TOP = 16
    _ARC_CANDIDATE_PROGRAM_LIMIT = 12
    _ARC_CANDIDATE_OUTPUT_LIMIT = 8
    _ARC_REFINEMENT_TOP_K = 3
    _ARC_REFINEMENT_ROUNDS = 1
    _ARC_POSITIONAL_RULE_MAX_MOD = 8
    _ARC_PANEL_TRACE_MAX_PARTIAL_RULES = 32
    _COMPONENT_TRANSFORMS: Tuple[Tuple[str, Callable[[Grid], Grid]], ...] = (
        ("identity", _identity),
        ("flip_h", _flip_h),
        ("flip_v", _flip_v),
        ("transpose", _transpose),
        ("rotate90", _rotate90),
        ("rotate180", _rotate180),
        ("rotate270", _rotate270),
    )
    _BASE_OPS: Tuple[Tuple[str, Callable[[Grid], Grid]], ...] = (
        ("identity", _identity),
        ("flip_h", _flip_h),
        ("flip_v", _flip_v),
        ("transpose", _transpose),
        ("rotate90", _rotate90),
        ("rotate180", _rotate180),
        ("rotate270", _rotate270),
        ("crop_nonzero_bbox", _crop_nonzero_bbox),
        ("crop_majority_bbox", _crop_majority_bbox),
        ("crop_largest_component", lambda grid: _crop_component(grid, "largest")),
        ("crop_smallest_component", lambda grid: _crop_component(grid, "smallest")),
        ("crop_topmost_component", lambda grid: _crop_component(grid, "topmost")),
        ("crop_bottommost_component", lambda grid: _crop_component(grid, "bottommost")),
        ("crop_leftmost_component", lambda grid: _crop_component(grid, "leftmost")),
        ("crop_rightmost_component", lambda grid: _crop_component(grid, "rightmost")),
        ("crop_largest_color_component", lambda grid: _crop_color_component(grid, "largest")),
        ("crop_smallest_color_component", lambda grid: _crop_color_component(grid, "smallest")),
        ("crop_topmost_color_component", lambda grid: _crop_color_component(grid, "topmost")),
        ("crop_bottommost_color_component", lambda grid: _crop_color_component(grid, "bottommost")),
        ("crop_leftmost_color_component", lambda grid: _crop_color_component(grid, "leftmost")),
        ("crop_rightmost_color_component", lambda grid: _crop_color_component(grid, "rightmost")),
    )

    def __init__(self) -> None:
        self._llm_draft_cache: "OrderedDict[str, Tuple[Dict[str, Any], Dict[str, Any]]]" = OrderedDict()
        self._llm_draft_cache_limit = 128

    @staticmethod
    def _scalarize_arc_candidate_score(score: ArcCandidateScore) -> float:
        exact_count, holdout_exact_count, counterexample_score, holdout_similarity, avg_similarity, holdout_transition_alignment, avg_transition_alignment, generalization_bias, complexity_penalty = score
        return round(
            (exact_count * 1.0)
            + (holdout_exact_count * 0.8)
            + (counterexample_score * 0.6)
            + (holdout_similarity * 0.5)
            + (avg_similarity * 0.45)
            + (holdout_transition_alignment * 0.35)
            + (avg_transition_alignment * 0.3)
            + (generalization_bias * 0.08)
            + (complexity_penalty * 0.2),
            6,
        )

    def enumerate_arc_candidate_programs(
        self,
        task_payload: Any,
        *,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        if not isinstance(task_payload, dict):
            return []
        train = task_payload.get("train", [])
        if not isinstance(train, list) or not train:
            return []
        transition_profile = summarize_arc_transition_profile(train)
        candidates = self._build_arc_candidate_programs(train)
        ranked = self._rank_arc_candidates(
            candidates,
            train,
            transition_profile=transition_profile,
        )
        rows = [
            self._program_row_from_candidate(candidate, score, index)
            for index, (candidate, score) in enumerate(ranked[: max(0, int(limit))])
        ]
        return normalize_arc_program_rows(rows, limit=max(0, int(limit)))

    def enumerate_arc_candidate_outputs(
        self,
        task_payload: Any,
        *,
        candidate_programs: Optional[Sequence[Dict[str, Any]]] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        if not isinstance(task_payload, dict):
            return []
        train = task_payload.get("train", [])
        test_inputs = task_payload.get("test_inputs", [])
        if not isinstance(train, list) or not train or not isinstance(test_inputs, list) or not test_inputs:
            return []
        transition_profile = summarize_arc_transition_profile(train)
        program_rows = normalize_arc_program_rows(
            list(candidate_programs or []),
            limit=max(0, int(limit)) or None,
        )
        if not program_rows:
            program_rows = self.enumerate_arc_candidate_programs(task_payload, limit=limit)
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(program_rows[: max(0, int(limit))]):
            program_name = str(row.get("name", "") or "")
            candidate = self._candidate_from_program_row(row, train)
            if candidate is None:
                continue
            predicted_outputs = self._predict_arc_candidate(candidate, test_inputs, train)
            if not predicted_outputs:
                continue
            score_key = self._score_arc_candidate(candidate, train, transition_profile=transition_profile)
            first_output = predicted_outputs[0] if predicted_outputs else []
            rows.append({
                "output_id": f"arc_output_{index}",
                "program_id": str(row.get("program_id", f"arc_program_{index}") or f"arc_program_{index}"),
                "program_name": program_name,
                "program_kind": str(row.get("kind", "") or ""),
                "program_complexity": int(row.get("complexity", 0) or 0),
                "program_tags": list(row.get("program_tags", [])) if isinstance(row.get("program_tags", []), list) else [],
                "score": self._scalarize_arc_candidate_score(score_key),
                "score_key": list(score_key),
                "predicted_outputs": predicted_outputs,
                "first_output_state": summarize_grid_state(first_output) if _is_grid(first_output) else {},
                "transition_alignment": float(score_key[6] if len(score_key) > 6 else 0.0),
            })
        return rows

    def maybe_populate_action_kwargs(
        self,
        action: Dict[str, Any],
        obs: Dict[str, Any],
        *,
        llm_client: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        function_name = str(tool_args.get("function_name", "") or "").strip()
        if not function_name:
            return action
        if not self._looks_like_structured_answer(function_name, obs):
            return action

        synthesized_kwargs, strategy_name, synthesis_meta = self._synthesize_kwargs(
            function_name,
            obs,
            llm_client=llm_client,
        )
        if not synthesized_kwargs:
            return action

        updated = deepcopy(action)
        updated_payload = updated.setdefault("payload", {})
        updated_tool_args = updated_payload.setdefault("tool_args", {})
        updated_tool_args["kwargs"] = synthesized_kwargs
        meta = updated.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            meta["structured_answer_synthesized"] = True
            meta["structured_answer_function"] = function_name
            meta["structured_answer_strategy"] = strategy_name or ""
            meta["structured_answer_kwargs_keys"] = sorted(synthesized_kwargs.keys())
            if strategy_name:
                meta["structured_answer_internal_simulation"] = strategy_name.startswith("sim:")
                meta["structured_answer_simulation_steps"] = strategy_name.count("=>") + 1 if strategy_name.startswith("sim:") else 1
            if isinstance(synthesis_meta, dict):
                solver_path = str(synthesis_meta.get("solver_path", "") or "")
                if solver_path:
                    meta["structured_answer_solver_path"] = solver_path
                meta["structured_answer_candidate_program_count"] = int(synthesis_meta.get("candidate_program_count", 0) or 0)
                meta["structured_answer_candidate_output_count"] = int(synthesis_meta.get("candidate_output_count", 0) or 0)
                meta["structured_answer_refinement_rounds"] = int(synthesis_meta.get("refinement_rounds", 0) or 0)
                meta["structured_answer_refinement_used"] = bool(synthesis_meta.get("refinement_used", False))
                meta["structured_answer_fallback_used"] = bool(synthesis_meta.get("fallback_used", False))
                selected_program = str(synthesis_meta.get("selected_program", "") or "")
                if selected_program:
                    meta["structured_answer_selected_program"] = selected_program
                if synthesis_meta.get("selected_output_score") is not None:
                    meta["structured_answer_selected_output_score"] = float(synthesis_meta.get("selected_output_score", 0.0) or 0.0)
                meta["structured_answer_llm_candidate_considered"] = bool(synthesis_meta.get("llm_candidate_considered", False))
                meta["structured_answer_llm_candidate_selected"] = bool(synthesis_meta.get("llm_candidate_selected", False))
                llm_trace = synthesis_meta.get("llm_trace", [])
                if isinstance(llm_trace, list) and llm_trace:
                    meta["structured_answer_llm_trace"] = [
                        dict(row) for row in llm_trace if isinstance(row, dict)
                    ]
                fallback_reason = str(synthesis_meta.get("fallback_reason", "") or "")
                if fallback_reason:
                    meta["structured_answer_fallback_reason"] = fallback_reason
            self._attach_state_abstraction_meta(meta, synthesized_kwargs, obs)
        return updated

    def _looks_like_structured_answer(self, function_name: str, obs: Dict[str, Any]) -> bool:
        if function_name.startswith(self._ANSWER_FN_PREFIXES):
            return True
        if function_name in {"internet_fetch", "internet_fetch_project", "mirror_exec"} or function_name in self._LOCAL_MACHINE_ATOMIC_FUNCTIONS:
            local_mirror = obs.get("local_mirror", {}) if isinstance(obs, dict) else {}
            if function_name in self._LOCAL_MACHINE_ATOMIC_FUNCTIONS:
                return isinstance(local_mirror, dict)
            return isinstance(local_mirror, dict) and bool(local_mirror.get("internet_enabled") or function_name == "mirror_exec")
        return self._ARC_KEY in obs

    def _synthesize_kwargs(
        self,
        function_name: str,
        obs: Dict[str, Any],
        *,
        llm_client: Any = None,
    ) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
        if self._ARC_KEY in obs:
            if llm_client is not None:
                kwargs, strategy_name, synthesis_meta = self._solve_arc_task(obs[self._ARC_KEY], llm_client=llm_client)
            else:
                kwargs, strategy_name, synthesis_meta = self._solve_arc_task_cached(obs[self._ARC_KEY])
            if kwargs:
                return kwargs, strategy_name, synthesis_meta
            return {}, None, synthesis_meta if isinstance(synthesis_meta, dict) else {}
        prefer_llm_kwargs = self._prefer_llm_kwargs(obs)
        llm_meta: Dict[str, Any] = {}
        if prefer_llm_kwargs and llm_client is not None:
            kwargs, llm_meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
            if kwargs:
                return kwargs, "llm_draft", llm_meta
        if self._deterministic_fallback_enabled(obs):
            fallback_kwargs = self._local_machine_fallback_kwargs(function_name, obs)
            if fallback_kwargs:
                meta: Dict[str, Any] = {"fallback_used": True}
                if isinstance(llm_meta, dict) and llm_meta.get("llm_trace"):
                    meta["llm_candidate_considered"] = bool(llm_meta.get("llm_candidate_considered", False))
                    meta["llm_candidate_selected"] = False
                    meta["llm_trace"] = list(llm_meta.get("llm_trace", []) or [])
                    meta["fallback_reason"] = "llm_draft_empty_or_invalid"
                return fallback_kwargs, "local_machine_fallback", meta
        if llm_client is not None and not prefer_llm_kwargs:
            kwargs, llm_meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
            if kwargs:
                return kwargs, "llm_draft", llm_meta
        return {}, None, {}

    @staticmethod
    def _deterministic_fallback_enabled(obs: Dict[str, Any]) -> bool:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        if not local_mirror:
            return True
        return bool(local_mirror.get("deterministic_fallback_enabled", True))

    @staticmethod
    def _prefer_llm_kwargs(obs: Dict[str, Any]) -> bool:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        return bool(local_mirror.get("prefer_llm_kwargs", False))

    def _local_machine_fallback_kwargs(self, function_name: str, obs: Dict[str, Any]) -> Dict[str, Any]:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        if not local_mirror:
            return {}
        instruction = str(
            obs.get("instruction")
            or local_mirror.get("instruction")
            or ""
        ).lower()
        if function_name == "internet_fetch" and any(token in instruction for token in ("market", "research", "trend", "competitor", "调研", "市场")):
            return {
                "url": "https://github.com/topics/ai-tools",
                "filename": "ai-tools-market-signal.html",
            }
        if function_name == "mirror_exec" and (
            "generated_product" in instruction
            or ("product" in instruction and "ai" in instruction)
            or "产品" in instruction
        ):
            return {
                "command": ["python3", "-c", self._local_ai_product_builder_script()],
                "purpose": "build",
                "target": "generated_product",
                "timeout_seconds": 90,
            }
        return {}

    @staticmethod
    def _local_ai_product_builder_script() -> str:
        return r'''
from pathlib import Path
import textwrap

root = Path("generated_product")
pkg = root / "src" / "signalbrief_ai"
tests = root / "tests"
docs = root / "docs"
scripts = root / "scripts"
for path in (pkg, tests, docs, scripts):
    path.mkdir(parents=True, exist_ok=True)

(root / ".gitignore").write_text(textwrap.dedent("""
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
dist/
build/
*.egg-info/
""").strip() + "\n", encoding="utf-8")

(root / "LICENSE").write_text(textwrap.dedent("""
MIT License

Copyright (c) 2026 Con OS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""").strip() + "\n", encoding="utf-8")

(root / "README.md").write_text(textwrap.dedent("""
# SignalBrief AI

SignalBrief AI is a local-first prompt and AI-workflow brief analyzer for small teams.
It turns a raw prompt, support ticket, or product idea into a concise quality report:
clarity score, missing context, risk flags, and concrete rewrite suggestions.

## Why this niche

AI tooling is crowded around chat frontends and agent platforms. A smaller unsolved
workflow is pre-flight quality control: teams need a lightweight way to inspect an
AI request before spending model tokens or delegating work to an agent.

SignalBrief AI is intentionally offline and dependency-light, so it can run in CI,
pre-commit hooks, or local product workflows without sending sensitive text away.

## Quick Start

```bash
python -m signalbrief_ai.cli analyze --text "Draft a launch plan for our AI tool"
python scripts/smoke_test.py
```

## Features

- Prompt clarity scoring
- Missing-context detection
- Risk and ambiguity flags
- Actionable rewrite checklist
- JSON output for CI or automation

## GitHub Description

Local-first prompt quality and AI-workflow brief analyzer for teams that want safer,
clearer AI requests before calling a model or agent.
""").strip() + "\n", encoding="utf-8")

(root / "pyproject.toml").write_text(textwrap.dedent("""
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "signalbrief-ai"
version = "0.1.0"
description = "Local-first prompt quality and AI-workflow brief analyzer."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Con OS"}]
dependencies = []

[project.scripts]
signalbrief = "signalbrief_ai.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
""").strip() + "\n", encoding="utf-8")

(pkg / "__init__.py").write_text('__all__ = ["analyze_brief"]\nfrom .core import analyze_brief\n', encoding="utf-8")

(pkg / "core.py").write_text(textwrap.dedent("""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


AMBIGUITY_TERMS = {
    "thing", "stuff", "etc", "quickly", "better", "optimize", "improve",
    "asap", "somehow", "nice", "good", "bad",
}

RISK_PATTERNS = {
    "credential_or_secret": re.compile(r"\\b(api[_-]?key|password|token|secret)\\b", re.I),
    "destructive_action": re.compile(r"\\b(delete|wipe|drop table|reset hard|destroy)\\b", re.I),
    "production_change": re.compile(r"\\b(production|prod|deploy|release)\\b", re.I),
}


@dataclass(frozen=True)
class BriefReport:
    clarity_score: int
    word_count: int
    missing_context: list[str]
    risk_flags: list[str]
    ambiguity_terms: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_brief(text: str) -> BriefReport:
    content = str(text or "").strip()
    words = re.findall(r"[A-Za-z0-9_'-]+", content)
    lowered = content.lower()
    ambiguity = sorted({term for term in AMBIGUITY_TERMS if re.search(rf"\\b{re.escape(term)}\\b", lowered)})
    risk_flags = [name for name, pattern in RISK_PATTERNS.items() if pattern.search(content)]

    missing: list[str] = []
    if len(words) < 12:
        missing.append("objective_detail")
    if not re.search(r"\\b(success|done|acceptance|metric|measure)\\b", lowered):
        missing.append("success_criteria")
    if not re.search(r"\\b(user|customer|audience|team|developer|operator)\\b", lowered):
        missing.append("target_user")
    if not re.search(r"\\b(input|data|file|source|context|constraint)\\b", lowered):
        missing.append("input_context")

    score = 100
    score -= min(35, len(missing) * 10)
    score -= min(20, len(ambiguity) * 5)
    score -= min(25, len(risk_flags) * 8)
    score = max(0, min(100, score))

    recommendations = []
    if "success_criteria" in missing:
        recommendations.append("Add explicit acceptance criteria or measurable success signals.")
    if "target_user" in missing:
        recommendations.append("Name the target user and the workflow they are trying to complete.")
    if "input_context" in missing:
        recommendations.append("List required inputs, constraints, and any data the model may or may not use.")
    if ambiguity:
        recommendations.append("Replace vague terms with concrete outcomes, examples, or constraints.")
    if risk_flags:
        recommendations.append("Add a review gate before handling secrets, production systems, or destructive actions.")
    if not recommendations:
        recommendations.append("Brief is clear enough for a first AI pass; keep evidence and test outputs attached.")

    return BriefReport(
        clarity_score=score,
        word_count=len(words),
        missing_context=missing,
        risk_flags=risk_flags,
        ambiguity_terms=ambiguity,
        recommendations=recommendations,
    )
""").strip() + "\n", encoding="utf-8")

(pkg / "cli.py").write_text(textwrap.dedent("""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import analyze_brief


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signalbrief")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze an AI prompt or workflow brief.")
    analyze.add_argument("--text", default="", help="Text to analyze.")
    analyze.add_argument("--file", default="", help="Path to a UTF-8 text file.")
    analyze.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.command == "analyze":
        text = args.text
        if args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        report = analyze_brief(text)
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"SignalBrief score: {report.clarity_score}/100")
            for item in report.recommendations:
                print(f"- {item}")
        return 0 if report.clarity_score >= 60 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
""").strip() + "\n", encoding="utf-8")

(tests / "test_core.py").write_text(textwrap.dedent("""
from signalbrief_ai import analyze_brief


def test_analyze_brief_flags_missing_context():
    report = analyze_brief("Improve this thing quickly")
    assert report.clarity_score < 80
    assert "success_criteria" in report.missing_context
    assert report.recommendations


def test_analyze_brief_accepts_concrete_prompt():
    report = analyze_brief(
        "For a developer audience, analyze this source file input and return a success metric with risks."
    )
    assert report.clarity_score >= 70
""").strip() + "\n", encoding="utf-8")

(scripts / "smoke_test.py").write_text(textwrap.dedent("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signalbrief_ai import analyze_brief

report = analyze_brief("For a support team user, summarize this ticket input and include success criteria.")
print(report.to_dict())
assert report.clarity_score >= 60
""").strip() + "\n", encoding="utf-8")

(docs / "MARKET_RESEARCH.md").write_text(textwrap.dedent("""
# Market Research Notes

Research seed captured by Con OS: public AI tools listings and repository/topic pages.

Observed pattern:
- Many AI tools focus on chat, generation, and agent execution.
- Teams still need lightweight pre-flight checks before sending prompts to models.
- Local-first review is attractive where prompts contain private product, support, or code context.

Product opportunity:
SignalBrief AI targets prompt and AI-workflow quality control before model execution.
It is small enough for CI, local scripts, and team review workflows.
""").strip() + "\n", encoding="utf-8")
'''.strip()

    def _solve_arc_task_cached(self, task_payload: Any) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
        cache = getattr(self, "_arc_task_solution_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._arc_task_solution_cache = cache
        cache_key = self._arc_task_cache_key(task_payload)
        if isinstance(cache, dict) and cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                cached_kwargs, cached_strategy_name, cached_meta = cached
                return deepcopy(cached_kwargs), cached_strategy_name, deepcopy(cached_meta)
        kwargs, strategy_name, synthesis_meta = self._solve_arc_task(task_payload)
        if isinstance(cache, dict) and cache_key:
            cache[cache_key] = (deepcopy(kwargs), strategy_name, deepcopy(synthesis_meta))
        return kwargs, strategy_name, synthesis_meta

    def _arc_task_cache_key(self, task_payload: Any) -> Optional[str]:
        if not isinstance(task_payload, dict):
            return None
        try:
            return json.dumps(task_payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None

    def _solve_arc_task(self, task_payload: Any, llm_client: Any = None) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
        if not isinstance(task_payload, dict):
            return {}, None, {}
        train = task_payload.get("train", [])
        test_inputs = task_payload.get("test_inputs", [])
        if not isinstance(train, list) or not isinstance(test_inputs, list) or not train or not test_inputs:
            return {}, None, {}
        previous_score_cache = getattr(self, "_active_arc_score_cache", None)
        previous_prediction_cache = getattr(self, "_active_arc_prediction_cache", None)
        self._active_arc_score_cache = {}
        self._active_arc_prediction_cache = {}
        try:
            try:
                search_result = self._run_arc_candidate_output_search(task_payload)
                if llm_client is not None:
                    search_result = self._merge_llm_candidate_output(
                        task_payload,
                        search_result,
                        llm_client=llm_client,
                    )
                selected_output = search_result.get("selected_output", {}) if isinstance(search_result.get("selected_output", {}), dict) else {}
                predicted_outputs = [
                    grid
                    for grid in list(selected_output.get("predicted_outputs", []) or [])
                    if _is_grid(grid)
                ]
                if predicted_outputs:
                    merged_predictions, used_memorized_match = self._merge_memorized_arc_predictions(
                        predicted_outputs,
                        test_inputs,
                        train,
                    )
                    selected_program = search_result.get("selected_program", {}) if isinstance(search_result.get("selected_program", {}), dict) else {}
                    strategy_name = str(selected_program.get("name", "") or "")
                    if used_memorized_match:
                        strategy_name = f"memorized_train_match_or_{strategy_name}" if strategy_name else "memorized_train_match"
                    return (
                        self._pack_arc_outputs(merged_predictions),
                        strategy_name,
                        {
                            "solver_path": str(search_result.get("solver_path", "candidate_output_search") or "candidate_output_search"),
                            "candidate_program_count": len(list(search_result.get("candidate_programs", []) or [])),
                            "candidate_output_count": len(list(search_result.get("candidate_outputs", []) or [])),
                            "selected_program": str(selected_program.get("name", "") or ""),
                            "selected_output_score": float(selected_output.get("critic_score", selected_output.get("score", 0.0)) or 0.0),
                            "refinement_rounds": int(search_result.get("refinement_rounds_used", 0) or 0),
                            "refinement_used": bool(search_result.get("used_refinement", False)),
                            "fallback_used": False,
                            "llm_candidate_considered": bool(search_result.get("llm_candidate_considered", False)),
                            "llm_candidate_selected": bool(search_result.get("llm_candidate_selected", False)),
                        },
                    )
            except Exception as exc:
                kwargs, strategy_name = self._solve_arc_task_baseline(task_payload)
                if kwargs:
                    return kwargs, strategy_name, {
                        "solver_path": "baseline_fallback",
                        "candidate_program_count": 0,
                        "candidate_output_count": 0,
                        "refinement_rounds": 0,
                        "refinement_used": False,
                        "fallback_used": True,
                        "fallback_reason": exc.__class__.__name__,
                    }
                return {}, None, {
                    "solver_path": "baseline_fallback",
                    "candidate_program_count": 0,
                    "candidate_output_count": 0,
                    "refinement_rounds": 0,
                    "refinement_used": False,
                    "fallback_used": True,
                    "fallback_reason": exc.__class__.__name__,
                }

            kwargs, strategy_name = self._solve_arc_task_baseline(task_payload)
            if kwargs:
                return kwargs, strategy_name, {
                    "solver_path": "baseline_fallback",
                    "candidate_program_count": 0,
                    "candidate_output_count": 0,
                    "refinement_rounds": 0,
                    "refinement_used": False,
                    "fallback_used": True,
                }
            return {}, None, {
                "solver_path": "baseline_fallback",
                "candidate_program_count": 0,
                "candidate_output_count": 0,
                "refinement_rounds": 0,
                "refinement_used": False,
                "fallback_used": True,
                "llm_candidate_considered": False,
                "llm_candidate_selected": False,
            }
        finally:
            if isinstance(previous_score_cache, dict):
                self._active_arc_score_cache = previous_score_cache
            elif hasattr(self, "_active_arc_score_cache"):
                delattr(self, "_active_arc_score_cache")
            if isinstance(previous_prediction_cache, dict):
                self._active_arc_prediction_cache = previous_prediction_cache
            elif hasattr(self, "_active_arc_prediction_cache"):
                delattr(self, "_active_arc_prediction_cache")

    def _solve_arc_task_baseline(self, task_payload: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        if not isinstance(task_payload, dict):
            return {}, None
        train = task_payload.get("train", [])
        test_inputs = task_payload.get("test_inputs", [])
        if not isinstance(train, list) or not isinstance(test_inputs, list) or not train or not test_inputs:
            return {}, None
        outputs = [example.get("output") for example in train if isinstance(example, dict)]
        if outputs and all(_is_grid(grid) for grid in outputs):
            first = outputs[0]
            if all(grid == first for grid in outputs[1:]):
                return self._pack_arc_outputs([_identity(first) for _ in test_inputs]), "constant_output"

        memorized_predictions = self._predict_exact_train_matches(test_inputs, train)
        if memorized_predictions:
            return self._pack_arc_outputs(memorized_predictions), "memorized_train_match"

        transition_profile = summarize_arc_transition_profile(train)
        candidates = self._build_arc_candidate_programs(train)
        best = self._select_best_arc_candidate(candidates, train, transition_profile=transition_profile)
        if best is not None:
            predictions = self._predict_arc_candidate(best, test_inputs, train)
            if predictions:
                merged_predictions, used_memorized_match = self._merge_memorized_arc_predictions(
                    predictions,
                    test_inputs,
                    train,
                )
                strategy_name = str(best.get("name", "") or "")
                if used_memorized_match:
                    strategy_name = f"memorized_train_match_or_{strategy_name}" if strategy_name else "memorized_train_match"
                return self._pack_arc_outputs(merged_predictions), strategy_name
        if outputs and _is_grid(outputs[0]):
            return self._pack_arc_outputs([_identity(outputs[0]) for _ in test_inputs]), "fallback_first_output"
        if test_inputs and _is_grid(test_inputs[0]):
            return self._pack_arc_outputs([_identity(grid) for grid in test_inputs if _is_grid(grid)]), "fallback_identity_test"
        return {}, None

    def _merge_memorized_arc_predictions(
        self,
        predictions: Sequence[Grid],
        test_inputs: Sequence[Any],
        train_examples: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Grid], bool]:
        merged_predictions: List[Grid] = []
        used_memorized_match = False
        for index, grid in enumerate(test_inputs):
            exact_output = self._lookup_exact_train_output(grid, train_examples) if _is_grid(grid) else None
            if exact_output is not None:
                merged_predictions.append(exact_output)
                used_memorized_match = True
                continue
            if index < len(predictions) and _is_grid(predictions[index]):
                merged_predictions.append(_identity(predictions[index]))
        return merged_predictions, used_memorized_match

    def _program_row_from_candidate(
        self,
        candidate: Dict[str, Any],
        score: ArcCandidateScore,
        index: int,
    ) -> Dict[str, Any]:
        return {
            "program_id": f"arc_program_{index}",
            "name": str(candidate.get("name", "") or ""),
            "kind": str(candidate.get("kind", "") or ""),
            "score": self._scalarize_arc_candidate_score(score),
            "score_key": list(score),
            "generalization_bias": int(candidate.get("generalization_bias", self._candidate_generalization_bias(candidate)) or 0),
            "complexity": int(candidate.get("complexity", self._candidate_complexity(candidate)) or 0),
            "terminal_only": bool(candidate.get("terminal_only", False)),
            "program_spec": serialize_arc_candidate_spec(candidate),
        }

    def _candidate_from_program_row(
        self,
        program_row: Dict[str, Any],
        train_examples: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        program_spec = program_row.get("program_spec", {}) if isinstance(program_row.get("program_spec", {}), dict) else {}
        if program_spec:
            rebuilt = self._rebuild_arc_candidate(program_spec, train_examples)
            if rebuilt is not None:
                return rebuilt
        program_name = str(program_row.get("name", "") or "")
        if not program_name:
            return None
        for candidate in self._build_arc_candidate_programs(train_examples):
            if str(candidate.get("name", "") or "") == program_name:
                return candidate
        return None

    def _candidate_from_steps(self, steps: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        clean_steps = [step for step in steps if isinstance(step, dict)]
        if not clean_steps:
            return None
        candidate = clean_steps[0]
        for step in clean_steps[1:]:
            candidate = self._compose_arc_candidate_sequence(candidate, step)
        return candidate

    def _refine_arc_program_rows(
        self,
        selected_programs: Sequence[Dict[str, Any]],
        task_payload: Dict[str, Any],
        existing_names: Set[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        train = task_payload.get("train", []) if isinstance(task_payload, dict) else []
        if not isinstance(train, list) or not train:
            return []
        atomic_candidates = self._build_arc_atomic_candidates(train)
        transition_profile = summarize_arc_transition_profile(train)
        ranked_atomic = [
            candidate
            for candidate, _ in self._rank_arc_candidates(
                atomic_candidates,
                train,
                transition_profile=transition_profile,
            )[: self._ARC_SIMULATION_BEAM_WIDTH]
        ]
        refined_candidates: List[Dict[str, Any]] = []
        for program_row in selected_programs:
            base_candidate = self._candidate_from_program_row(program_row, train)
            if base_candidate is None:
                continue
            if isinstance(base_candidate.get("steps", []), list):
                base_steps = [step for step in base_candidate.get("steps", []) if isinstance(step, dict)]
            else:
                base_steps = [base_candidate]
            if len(base_steps) > 1:
                drop_first = self._candidate_from_steps(base_steps[1:])
                if drop_first is not None:
                    refined_candidates.append(drop_first)
                drop_last = self._candidate_from_steps(base_steps[:-1])
                if drop_last is not None:
                    refined_candidates.append(drop_last)
            if bool(base_candidate.get("terminal_only", False)):
                continue
            for seed in ranked_atomic[: self._ARC_REFINEMENT_TOP_K + 1]:
                refined_candidates.append(self._compose_arc_candidate_sequence(base_candidate, seed))
                if not bool(seed.get("terminal_only", False)):
                    refined_candidates.append(self._compose_arc_candidate_sequence(seed, base_candidate))
        ranked_refined = self._rank_arc_candidates(
            self._dedupe_arc_candidates(refined_candidates),
            train,
            transition_profile=transition_profile,
        )
        rows = [
            self._program_row_from_candidate(candidate, score, index)
            for index, (candidate, score) in enumerate(ranked_refined)
            if str(candidate.get("name", "") or "") not in existing_names
        ]
        return normalize_arc_program_rows(rows, limit=max(0, int(limit)))

    def _run_arc_candidate_output_search(
        self,
        task_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_programs = self.enumerate_arc_candidate_programs(
            task_payload,
            limit=self._ARC_CANDIDATE_PROGRAM_LIMIT,
        )
        if not base_programs:
            return {}
        return run_arc_refinement_loop(
            task_payload=task_payload,
            base_programs=base_programs,
            predict_outputs=lambda program_rows: self.enumerate_arc_candidate_outputs(
                task_payload,
                candidate_programs=program_rows,
                limit=self._ARC_CANDIDATE_OUTPUT_LIMIT,
            ),
            refine_programs=lambda selected_programs, existing_names, limit: self._refine_arc_program_rows(
                selected_programs,
                task_payload,
                existing_names,
                limit,
            ),
            config=ArcRefinementConfig(
                program_limit=self._ARC_CANDIDATE_PROGRAM_LIMIT,
                output_limit=self._ARC_CANDIDATE_OUTPUT_LIMIT,
                top_k=self._ARC_REFINEMENT_TOP_K,
                rounds=self._ARC_REFINEMENT_ROUNDS,
            ),
        )

    def _arc_op_matches_training(self, op, train_examples: Sequence[Dict[str, Any]]) -> bool:
        for example in train_examples:
            if not isinstance(example, dict):
                return False
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return False
            if op(input_grid) != output_grid:
                return False
        return True

    def _candidate_ops(self) -> Sequence[Tuple[str, Callable[[Grid], Grid]]]:
        ops: List[Tuple[str, Callable[[Grid], Grid]]] = list(self._BASE_OPS)
        for left_name, left_op in self._BASE_OPS:
            for right_name, right_op in self._BASE_OPS:
                if left_name == "identity":
                    continue
                composed_name = f"{left_name}->{right_name}"
                ops.append((composed_name, _compose(left_op, right_op)))
        return ops

    def _derive_consistent_task_color_map(
        self,
        train_examples: Sequence[Dict[str, Any]],
        op: Callable[[Grid], Grid],
    ) -> Optional[Dict[int, int]]:
        merged: Dict[int, int] = {}
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            transformed = op(input_grid)
            color_map = _derive_color_map(transformed, output_grid)
            if color_map is None:
                return None
            for src_color, dst_color in color_map.items():
                if src_color in merged and merged[src_color] != dst_color:
                    return None
                merged[src_color] = dst_color
            if _apply_color_map(transformed, merged) != output_grid:
                return None
        return merged

    def _build_arc_candidate_programs(self, train_examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        atomic_candidates = self._build_arc_atomic_candidates(train_examples)
        simulated_candidates = self._build_simulated_arc_candidates(atomic_candidates, train_examples)
        return self._dedupe_arc_candidates([*atomic_candidates, *simulated_candidates])

    def _build_arc_atomic_candidates(self, train_examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        constant_output = self._derive_constant_output_grid(train_examples)
        if constant_output is not None:
            candidates.append({
                "name": "constant_output",
                "predict": lambda _grid, constant_output=constant_output: _identity(constant_output),
                "kind": "constant_output",
                "terminal_only": True,
                "_builder_kind": "constant_output",
                "_builder_params": {},
            })
        for name, op in self._candidate_ops():
            candidates.append({
                "name": f"op:{name}",
                "predict": lambda grid, op=op: op(grid),
                "kind": "direct_op",
                "_builder_kind": "direct_op",
                "_builder_params": {"op_name": name},
            })
            mapping = self._derive_consistent_task_color_map(train_examples, op)
            if mapping is not None:
                candidates.append({
                    "name": f"op_color:{name}",
                    "predict": lambda grid, op=op, mapping=mapping: _apply_color_map(op(grid), mapping),
                    "kind": "op_plus_color_map",
                    "_builder_kind": "op_plus_color_map",
                    "_builder_params": {"op_name": name},
                })
        for radius in (1, 2, 3):
            local_rule = self._derive_local_rewrite_rule(train_examples, radius)
            if local_rule is not None:
                candidates.append({
                    "name": f"local_rule_r{radius}",
                    "predict": lambda grid, radius=radius, rule=local_rule: self._apply_local_rewrite_rule(grid, radius, rule),
                    "kind": "local_rule",
                    "_builder_kind": "local_rule",
                    "_builder_params": {"radius": radius},
                })
            symbolic_rule = self._derive_symbolic_local_rewrite_rule(train_examples, radius)
            if symbolic_rule is not None:
                candidates.append({
                    "name": f"symbolic_local_rule_r{radius}",
                    "predict": lambda grid, radius=radius, rule=symbolic_rule: self._apply_symbolic_local_rewrite_rule(grid, radius, rule),
                    "kind": "symbolic_local_rule",
                    "_builder_kind": "symbolic_local_rule",
                    "_builder_params": {"radius": radius},
                })
            candidates.extend(self._build_positional_symbolic_local_candidates(train_examples, radius))
        candidates.extend(self._build_component_layout_candidates(train_examples))
        candidates.extend(self._build_anchor_relation_candidates(train_examples))
        candidates.extend(self._build_panel_trace_candidates(train_examples))
        candidates.append({
            "name": "nearest_train_output",
            "predict_with_train": True,
            "kind": "nearest_neighbor_output",
            "terminal_only": True,
            "_builder_kind": "nearest_neighbor_output",
            "_builder_params": {},
        })
        return candidates

    def _build_component_layout_candidates(
        self,
        train_examples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for axis in ("horizontal", "vertical"):
            for sort_mode in ("top_left", "left_top", "area_desc", "area_asc"):
                for transform_name, _transform in self._COMPONENT_TRANSFORMS:
                    rule = self._derive_component_layout_rule(
                        train_examples,
                        axis=axis,
                        sort_mode=sort_mode,
                        transform_name=transform_name,
                    )
                    if rule is None:
                        continue
                    candidates.append({
                        "name": f"component_layout_rule_{axis}_{sort_mode}_{transform_name}",
                        "predict": lambda grid, rule=rule: self._apply_component_layout_rule(grid, rule),
                        "kind": "component_layout_rule",
                        "_builder_kind": "component_layout_rule",
                        "_builder_params": {
                            "axis": axis,
                            "sort_mode": sort_mode,
                            "transform_name": transform_name,
                        },
                    })
        return candidates

    def _build_anchor_relation_candidates(
        self,
        train_examples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for relation_mode in (
            "select_left_of_marker",
            "select_right_of_marker",
            "select_above_marker",
            "select_below_marker",
            "compose_horizontal_neighbors",
            "compose_vertical_neighbors",
        ):
            rule = self._derive_anchor_relation_rule(train_examples, relation_mode=relation_mode)
            if rule is None:
                continue
            candidates.append({
                "name": f"anchor_relation_rule_{relation_mode}",
                "predict": lambda grid, rule=rule: self._apply_anchor_relation_rule(grid, rule),
                "kind": "anchor_relation_rule",
                "terminal_only": True,
                "_builder_kind": "anchor_relation_rule",
                "_builder_params": {"relation_mode": relation_mode},
            })
        return candidates

    def _build_panel_trace_candidates(
        self,
        train_examples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for order_mode in ("col_major", "row_major"):
            rule = self._derive_panel_trace_rule(train_examples, order_mode=order_mode)
            if rule is None:
                continue
            candidates.append({
                "name": f"panel_trace_rule_{order_mode}",
                "predict": lambda grid, rule=rule: self._apply_panel_trace_rule(grid, rule),
                "kind": "panel_trace_rule",
                "_builder_kind": "panel_trace_rule",
                "_builder_params": {"order_mode": order_mode},
            })
        return candidates

    def _derive_constant_output_grid(
        self,
        train_examples: Sequence[Dict[str, Any]],
    ) -> Optional[Grid]:
        outputs = [example.get("output") for example in train_examples if isinstance(example, dict)]
        if not outputs or not all(_is_grid(grid) for grid in outputs):
            return None
        first = outputs[0]
        if all(grid == first for grid in outputs[1:]):
            return _identity(first)
        return None

    def _build_positional_symbolic_local_candidates(
        self,
        train_examples: Sequence[Dict[str, Any]],
        radius: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for row_mod in self._candidate_position_moduli(train_examples, axis="row"):
            rule = self._derive_positional_symbolic_local_rewrite_rule(
                train_examples,
                radius,
                row_mod=row_mod,
            )
            if rule is None:
                continue
            candidates.append({
                "name": f"positional_symbolic_local_rule_r{radius}_rowmod{row_mod}",
                "predict": lambda grid, radius=radius, row_mod=row_mod, rule=rule: self._apply_positional_symbolic_local_rewrite_rule(
                    grid,
                    radius,
                    rule,
                    row_mod=row_mod,
                ),
                "kind": "positional_symbolic_local_rule",
                "_builder_kind": "positional_symbolic_local_rule",
                "_builder_params": {
                    "radius": radius,
                    "row_mod": row_mod,
                },
            })
        for col_mod in self._candidate_position_moduli(train_examples, axis="col"):
            rule = self._derive_positional_symbolic_local_rewrite_rule(
                train_examples,
                radius,
                col_mod=col_mod,
            )
            if rule is None:
                continue
            candidates.append({
                "name": f"positional_symbolic_local_rule_r{radius}_colmod{col_mod}",
                "predict": lambda grid, radius=radius, col_mod=col_mod, rule=rule: self._apply_positional_symbolic_local_rewrite_rule(
                    grid,
                    radius,
                    rule,
                    col_mod=col_mod,
                ),
                "kind": "positional_symbolic_local_rule",
                "_builder_kind": "positional_symbolic_local_rule",
                "_builder_params": {
                    "radius": radius,
                    "col_mod": col_mod,
                },
            })
        return candidates

    def _build_simulated_arc_candidates(
        self,
        atomic_candidates: Sequence[Dict[str, Any]],
        train_examples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        transition_profile = summarize_arc_transition_profile(train_examples)
        ranked_atomic = self._rank_arc_candidates(
            atomic_candidates,
            train_examples,
            transition_profile=transition_profile,
        )
        if not ranked_atomic:
            return []
        seed_candidates = [candidate for candidate, _ in ranked_atomic[: self._ARC_SIMULATION_BEAM_WIDTH]]
        retained: List[Dict[str, Any]] = [candidate for candidate, _ in ranked_atomic[: self._ARC_SIMULATION_KEEP_TOP]]
        beam: List[Dict[str, Any]] = [candidate for candidate in seed_candidates if not candidate.get("terminal_only")]
        simulated: List[Dict[str, Any]] = []
        for _depth in range(2, self._ARC_SIMULATION_MAX_DEPTH + 1):
            if not beam:
                break
            expanded: List[Dict[str, Any]] = []
            for prefix in beam[: self._ARC_SIMULATION_BEAM_WIDTH]:
                for step in seed_candidates:
                    expanded.append(self._compose_arc_candidate_sequence(prefix, step))
            ranked_expanded = self._rank_arc_candidates(
                self._dedupe_arc_candidates(expanded),
                train_examples,
                transition_profile=transition_profile,
            )
            top_expanded = [candidate for candidate, _ in ranked_expanded[: self._ARC_SIMULATION_KEEP_TOP]]
            simulated.extend(top_expanded)
            retained.extend(top_expanded)
            beam = [candidate for candidate in top_expanded if not candidate.get("terminal_only")]
        return self._dedupe_arc_candidates(simulated + retained[: self._ARC_SIMULATION_KEEP_TOP])

    def _compose_arc_candidate_sequence(self, prefix: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        prefix_steps = list(prefix.get("steps", [prefix]))
        step_steps = list(step.get("steps", [step]))
        steps = [*prefix_steps, *step_steps]
        return {
            "name": "sim:" + "=>".join(str(item.get("name", "")) for item in steps),
            "steps": steps,
            "kind": "simulated_program",
            "terminal_only": bool(steps[-1].get("terminal_only")),
            "generalization_bias": max(
                self._candidate_generalization_bias(item)
                for item in steps
            ) - max(len(steps) - 1, 0),
            "complexity": sum(self._candidate_complexity(item) for item in steps) + max(len(steps) - 1, 0),
        }

    def _dedupe_arc_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            name = str(candidate.get("name", "") or "")
            if not name or name in unique:
                continue
            unique[name] = candidate
        return list(unique.values())

    def _rank_arc_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        train_examples: Sequence[Dict[str, Any]],
        *,
        transition_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], ArcCandidateScore]]:
        ranked: List[Tuple[Dict[str, Any], ArcCandidateScore]] = []
        for candidate in candidates:
            ranked.append((candidate, self._score_arc_candidate(candidate, train_examples, transition_profile=transition_profile)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _select_best_arc_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        train_examples: Sequence[Dict[str, Any]],
        *,
        transition_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        best_candidate: Optional[Dict[str, Any]] = None
        best_key: Optional[ArcCandidateScore] = None
        for candidate in candidates:
            candidate_key = self._score_arc_candidate(candidate, train_examples, transition_profile=transition_profile)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_candidate = candidate
        return best_candidate

    def _score_arc_candidate(
        self,
        candidate: Dict[str, Any],
        train_examples: Sequence[Dict[str, Any]],
        *,
        transition_profile: Optional[Dict[str, Any]] = None,
    ) -> ArcCandidateScore:
        candidate_name = str(candidate.get("name", "") or "")
        score_cache = getattr(self, "_active_arc_score_cache", None)
        if candidate_name and isinstance(score_cache, dict) and candidate_name in score_cache:
            return score_cache[candidate_name]
        exact_count = 0
        similarity_total = 0.0
        transition_alignment_total = 0.0
        train_count = len(train_examples)
        for idx, example in enumerate(train_examples):
            if not isinstance(example, dict):
                continue
            input_grid = example.get("input")
            target_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(target_grid):
                continue
            predicted = self._predict_arc_candidate(candidate, [input_grid], train_examples, exclude_idx=idx)
            if not predicted:
                continue
            pred_grid = predicted[0]
            if pred_grid == target_grid:
                exact_count += 1
            similarity_total += _grid_similarity(pred_grid, target_grid)
            transition_alignment_total += score_arc_transition_alignment(
                input_grid,
                pred_grid,
                transition_profile or {},
            )
        avg_similarity = similarity_total / max(train_count, 1)
        avg_transition_alignment = transition_alignment_total / max(train_count, 1)
        holdout_exact_count = 0
        holdout_similarity = 0.0
        holdout_transition_alignment = 0.0
        min_exact_for_holdout = max(1, train_count - 1)
        should_run_holdout = (
            train_count < 2
            or exact_count >= min_exact_for_holdout
            or avg_similarity >= 0.98
        )
        if should_run_holdout:
            holdout_exact_count, holdout_similarity, holdout_transition_alignment = self._score_arc_candidate_holdout(
                candidate,
                train_examples,
                transition_profile=transition_profile,
            )
        counterexample_score = self._score_arc_candidate_counterexamples(candidate, train_examples)
        complexity_penalty = 1.0 / (1.0 + self._candidate_complexity(candidate))
        generalization_bias = self._candidate_generalization_bias(candidate)
        score = (
            exact_count,
            holdout_exact_count,
            counterexample_score,
            holdout_similarity,
            avg_similarity,
            holdout_transition_alignment,
            avg_transition_alignment,
            generalization_bias,
            complexity_penalty,
        )
        if candidate_name and isinstance(score_cache, dict):
            score_cache[candidate_name] = score
        return score

    def _score_arc_candidate_counterexamples(
        self,
        candidate: Dict[str, Any],
        train_examples: Sequence[Dict[str, Any]],
    ) -> float:
        if str(candidate.get("kind", "") or "") != "component_layout_rule":
            return 0.5
        total_score = 0.0
        evaluated_count = 0
        for example in train_examples:
            if not isinstance(example, dict):
                continue
            input_grid = example.get("input")
            target_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(target_grid):
                continue
            synthetic_input = self._build_component_counterexample_input(input_grid)
            if synthetic_input is None:
                continue
            predicted = self._predict_arc_candidate(candidate, [synthetic_input], train_examples)
            if not predicted:
                continue
            pred_grid = predicted[0]
            total_score += 1.0 if pred_grid == target_grid else _grid_similarity(pred_grid, target_grid)
            evaluated_count += 1
        if evaluated_count == 0:
            return 0.5
        return total_score / evaluated_count

    def _build_component_counterexample_input(self, input_grid: Grid) -> Optional[Grid]:
        components = self._sort_components(_extract_color_components(input_grid), "top_left")
        if len(components) <= 1:
            return None
        background_color = _background_color(input_grid)
        rows = sum(len(component["crop"]) for component in components) + len(components)
        cols = sum(len(component["crop"][0]) if component["crop"] else 0 for component in components) + len(components)
        synthetic = [[int(background_color) for _ in range(cols)] for _ in range(rows)]
        row_offset = 0
        col_offset = 0
        for component in components:
            crop = component["crop"]
            for r_idx, row in enumerate(crop):
                for c_idx, cell in enumerate(row):
                    if cell == -1:
                        continue
                    synthetic[row_offset + r_idx][col_offset + c_idx] = int(cell)
            row_offset += len(crop) + 1
            col_offset += (len(crop[0]) if crop else 0) + 1
        return synthetic if synthetic != input_grid else None

    def _score_arc_candidate_holdout(
        self,
        candidate: Dict[str, Any],
        train_examples: Sequence[Dict[str, Any]],
        *,
        transition_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, float, float]:
        if len(train_examples) < 2:
            return (0, 0.5, 0.5)
        exact_count = 0
        similarity_total = 0.0
        transition_alignment_total = 0.0
        evaluated_count = 0
        for idx, example in enumerate(train_examples):
            if not isinstance(example, dict):
                continue
            input_grid = example.get("input")
            target_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(target_grid):
                continue
            evaluated_count += 1
            support_examples = [
                other
                for support_idx, other in enumerate(train_examples)
                if support_idx != idx and isinstance(other, dict)
            ]
            rebuilt_candidate = self._rebuild_arc_candidate(candidate, support_examples)
            if rebuilt_candidate is None:
                continue
            predicted = self._predict_arc_candidate(rebuilt_candidate, [input_grid], support_examples)
            if not predicted:
                continue
            pred_grid = predicted[0]
            if pred_grid == target_grid:
                exact_count += 1
            similarity_total += _grid_similarity(pred_grid, target_grid)
            transition_alignment_total += score_arc_transition_alignment(
                input_grid,
                pred_grid,
                transition_profile or {},
            )
        if evaluated_count == 0:
            return (0, 0.5, 0.5)
        return (
            exact_count,
            similarity_total / evaluated_count,
            transition_alignment_total / evaluated_count,
        )

    def _rebuild_arc_candidate(
        self,
        candidate: Dict[str, Any],
        train_examples: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(candidate, dict):
            return None
        steps = [step for step in candidate.get("steps", []) if isinstance(step, dict)]
        if steps:
            rebuilt_sequence: List[Dict[str, Any]] = []
            for step in steps:
                rebuilt_step = self._rebuild_arc_candidate(step, train_examples)
                if rebuilt_step is None:
                    return None
                rebuilt_sequence.append(rebuilt_step)
            rebuilt_candidate = rebuilt_sequence[0]
            for rebuilt_step in rebuilt_sequence[1:]:
                rebuilt_candidate = self._compose_arc_candidate_sequence(rebuilt_candidate, rebuilt_step)
            return rebuilt_candidate

        builder_kind = str(candidate.get("_builder_kind", "") or "")
        builder_params = candidate.get("_builder_params", {}) if isinstance(candidate.get("_builder_params", {}), dict) else {}

        if builder_kind == "direct_op":
            op_name = str(builder_params.get("op_name", "") or "")
            op = self._candidate_op_by_name(op_name)
            if op is None:
                return None
            return {
                "name": f"op:{op_name}",
                "predict": lambda grid, op=op: op(grid),
                "kind": "direct_op",
                "_builder_kind": "direct_op",
                "_builder_params": {"op_name": op_name},
            }

        if builder_kind == "constant_output":
            constant_output = self._derive_constant_output_grid(train_examples)
            if constant_output is None:
                return None
            return {
                "name": "constant_output",
                "predict": lambda _grid, constant_output=constant_output: _identity(constant_output),
                "kind": "constant_output",
                "terminal_only": True,
                "_builder_kind": "constant_output",
                "_builder_params": {},
            }

        if builder_kind == "op_plus_color_map":
            op_name = str(builder_params.get("op_name", "") or "")
            op = self._candidate_op_by_name(op_name)
            if op is None:
                return None
            mapping = self._derive_consistent_task_color_map(train_examples, op)
            if mapping is None:
                return None
            return {
                "name": f"op_color:{op_name}",
                "predict": lambda grid, op=op, mapping=mapping: _apply_color_map(op(grid), mapping),
                "kind": "op_plus_color_map",
                "_builder_kind": "op_plus_color_map",
                "_builder_params": {"op_name": op_name},
            }

        if builder_kind == "local_rule":
            radius = int(builder_params.get("radius", 0) or 0)
            rule = self._derive_local_rewrite_rule(train_examples, radius)
            if rule is None:
                return None
            return {
                "name": f"local_rule_r{radius}",
                "predict": lambda grid, radius=radius, rule=rule: self._apply_local_rewrite_rule(grid, radius, rule),
                "kind": "local_rule",
                "_builder_kind": "local_rule",
                "_builder_params": {"radius": radius},
            }

        if builder_kind == "symbolic_local_rule":
            radius = int(builder_params.get("radius", 0) or 0)
            rule = self._derive_symbolic_local_rewrite_rule(train_examples, radius)
            if rule is None:
                return None
            return {
                "name": f"symbolic_local_rule_r{radius}",
                "predict": lambda grid, radius=radius, rule=rule: self._apply_symbolic_local_rewrite_rule(grid, radius, rule),
                "kind": "symbolic_local_rule",
                "_builder_kind": "symbolic_local_rule",
                "_builder_params": {"radius": radius},
            }

        if builder_kind == "positional_symbolic_local_rule":
            radius = int(builder_params.get("radius", 0) or 0)
            row_mod = builder_params.get("row_mod")
            col_mod = builder_params.get("col_mod")
            rule = self._derive_positional_symbolic_local_rewrite_rule(
                train_examples,
                radius,
                row_mod=int(row_mod) if row_mod is not None else None,
                col_mod=int(col_mod) if col_mod is not None else None,
            )
            if rule is None:
                return None
            if row_mod is not None:
                row_mod = int(row_mod)
                return {
                    "name": f"positional_symbolic_local_rule_r{radius}_rowmod{row_mod}",
                    "predict": lambda grid, radius=radius, row_mod=row_mod, rule=rule: self._apply_positional_symbolic_local_rewrite_rule(
                        grid,
                        radius,
                        rule,
                        row_mod=row_mod,
                    ),
                    "kind": "positional_symbolic_local_rule",
                    "_builder_kind": "positional_symbolic_local_rule",
                    "_builder_params": {"radius": radius, "row_mod": row_mod},
                }
            if col_mod is not None:
                col_mod = int(col_mod)
                return {
                    "name": f"positional_symbolic_local_rule_r{radius}_colmod{col_mod}",
                    "predict": lambda grid, radius=radius, col_mod=col_mod, rule=rule: self._apply_positional_symbolic_local_rewrite_rule(
                        grid,
                        radius,
                        rule,
                        col_mod=col_mod,
                    ),
                    "kind": "positional_symbolic_local_rule",
                    "_builder_kind": "positional_symbolic_local_rule",
                    "_builder_params": {"radius": radius, "col_mod": col_mod},
                }
            return None

        if builder_kind == "component_layout_rule":
            axis = str(builder_params.get("axis", "") or "")
            sort_mode = str(builder_params.get("sort_mode", "") or "")
            transform_name = str(builder_params.get("transform_name", "") or "")
            if not axis or not sort_mode or not transform_name:
                return None
            rule = self._derive_component_layout_rule(
                train_examples,
                axis=axis,
                sort_mode=sort_mode,
                transform_name=transform_name,
            )
            if rule is None:
                return None
            return {
                "name": f"component_layout_rule_{axis}_{sort_mode}_{transform_name}",
                "predict": lambda grid, rule=rule: self._apply_component_layout_rule(grid, rule),
                "kind": "component_layout_rule",
                "_builder_kind": "component_layout_rule",
                "_builder_params": {
                    "axis": axis,
                    "sort_mode": sort_mode,
                    "transform_name": transform_name,
                },
            }

        if builder_kind == "anchor_relation_rule":
            relation_mode = str(builder_params.get("relation_mode", "") or "")
            if not relation_mode:
                return None
            rule = self._derive_anchor_relation_rule(
                train_examples,
                relation_mode=relation_mode,
            )
            if rule is None:
                return None
            return {
                "name": f"anchor_relation_rule_{relation_mode}",
                "predict": lambda grid, rule=rule: self._apply_anchor_relation_rule(grid, rule),
                "kind": "anchor_relation_rule",
                "terminal_only": True,
                "_builder_kind": "anchor_relation_rule",
                "_builder_params": {"relation_mode": relation_mode},
            }

        if builder_kind == "panel_trace_rule":
            order_mode = str(builder_params.get("order_mode", "") or "")
            if not order_mode:
                return None
            rule = self._derive_panel_trace_rule(train_examples, order_mode=order_mode)
            if rule is None:
                return None
            return {
                "name": f"panel_trace_rule_{order_mode}",
                "predict": lambda grid, rule=rule: self._apply_panel_trace_rule(grid, rule),
                "kind": "panel_trace_rule",
                "_builder_kind": "panel_trace_rule",
                "_builder_params": {"order_mode": order_mode},
            }

        if builder_kind == "nearest_neighbor_output":
            return {
                "name": "nearest_train_output",
                "predict_with_train": True,
                "kind": "nearest_neighbor_output",
                "terminal_only": True,
                "_builder_kind": "nearest_neighbor_output",
                "_builder_params": {},
            }

        return None

    def _candidate_op_by_name(self, op_name: str) -> Optional[Callable[[Grid], Grid]]:
        for candidate_name, op in self._candidate_ops():
            if candidate_name == op_name:
                return op
        return None

    def _predict_arc_candidate(
        self,
        candidate: Dict[str, Any],
        grids: Sequence[Grid],
        train_examples: Sequence[Dict[str, Any]],
        *,
        exclude_idx: Optional[int] = None,
    ) -> List[Grid]:
        prediction_cache = getattr(self, "_active_arc_prediction_cache", None)
        cache_key: Optional[Tuple[int, FrozenGridBatch, Optional[int]]] = None
        if isinstance(prediction_cache, dict) and grids and all(_is_grid(grid) for grid in grids):
            cache_key = (id(candidate), _freeze_grids(grids), exclude_idx)
            cached = prediction_cache.get(cache_key)
            if cached is not None:
                return _thaw_grids(cached)
        outputs: List[Grid] = []
        if isinstance(candidate.get("steps"), list):
            sequence_steps = [step for step in candidate.get("steps", []) if isinstance(step, dict)]
            if not sequence_steps:
                return []
            for grid in grids:
                if not _is_grid(grid):
                    return []
                current = _identity(grid)
                for step in sequence_steps:
                    predicted_steps = self._predict_arc_candidate(
                        step,
                        [current],
                        train_examples,
                        exclude_idx=exclude_idx,
                    )
                    if not predicted_steps:
                        return []
                    current = predicted_steps[0]
                outputs.append(current)
            if cache_key is not None and isinstance(prediction_cache, dict):
                prediction_cache[cache_key] = _freeze_grids(outputs)
            return outputs
        for grid in grids:
            if not _is_grid(grid):
                return []
            if candidate.get("predict_with_train"):
                predicted = self._predict_nearest_train_output(grid, train_examples, exclude_idx=exclude_idx)
            else:
                predicted = candidate["predict"](grid)
            if not _is_grid(predicted):
                return []
            outputs.append(predicted)
        if cache_key is not None and isinstance(prediction_cache, dict):
            prediction_cache[cache_key] = _freeze_grids(outputs)
        return outputs

    def _predict_nearest_train_output(
        self,
        input_grid: Grid,
        train_examples: Sequence[Dict[str, Any]],
        *,
        exclude_idx: Optional[int] = None,
    ) -> Optional[Grid]:
        if exclude_idx is None:
            exact_output = self._lookup_exact_train_output(input_grid, train_examples)
            if exact_output is not None:
                return exact_output
        best_output: Optional[Grid] = None
        best_score = -1.0
        for idx, example in enumerate(train_examples):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            if not isinstance(example, dict):
                continue
            candidate_input = example.get("input")
            candidate_output = example.get("output")
            if not _is_grid(candidate_input) or not _is_grid(candidate_output):
                continue
            score = _grid_similarity(input_grid, candidate_input)
            if len(input_grid) == len(candidate_input) and len(input_grid[0]) == len(candidate_input[0]):
                score += 0.05
            if score > best_score:
                best_score = score
                best_output = _identity(candidate_output)
        return best_output

    def _derive_local_rewrite_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        radius: int,
    ) -> Optional[Dict[Tuple[Tuple[int, ...], ...], int]]:
        rule: Dict[Tuple[Tuple[int, ...], ...], int] = {}
        changed_any = False
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
                return None
            for row_idx, row in enumerate(input_grid):
                for col_idx, cell in enumerate(row):
                    patch = _extract_local_patch(input_grid, row_idx, col_idx, radius)
                    target = output_grid[row_idx][col_idx]
                    existing = rule.get(patch)
                    if existing is not None and existing != target:
                        return None
                    rule[patch] = target
                    if cell != target:
                        changed_any = True
        return rule if changed_any else None

    def _apply_local_rewrite_rule(
        self,
        grid: Grid,
        radius: int,
        rule: Dict[Tuple[Tuple[int, ...], ...], int],
    ) -> Grid:
        output: Grid = []
        for row_idx, row in enumerate(grid):
            output_row: List[int] = []
            for col_idx, cell in enumerate(row):
                patch = _extract_local_patch(grid, row_idx, col_idx, radius)
                output_row.append(int(rule.get(patch, cell)))
            output.append(output_row)
        return output

    def _derive_symbolic_local_rewrite_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        radius: int,
    ) -> Optional[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, int]]]:
        choices_by_signature: Dict[Tuple[Tuple[int, ...], ...], List[Tuple[str, int]]] = {}
        changed_any = False
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
                return None
            for row_idx, row in enumerate(input_grid):
                for col_idx, cell in enumerate(row):
                    patch = _extract_local_patch(input_grid, row_idx, col_idx, radius)
                    signature = _normalize_patch_signature(patch)
                    patch_values = _flatten_patch(patch)
                    target = output_grid[row_idx][col_idx]
                    candidate_actions: List[Tuple[str, int]] = [
                        ("offset", idx)
                        for idx, patch_value in enumerate(patch_values)
                        if patch_value == target
                    ]
                    candidate_actions.append(("const", target))
                    if signature in choices_by_signature:
                        existing = set(choices_by_signature[signature])
                        intersection = [action for action in candidate_actions if action in existing]
                        if not intersection:
                            return None
                        choices_by_signature[signature] = intersection
                    else:
                        choices_by_signature[signature] = list(candidate_actions)
                    if cell != target:
                        changed_any = True
        if not changed_any:
            return None
        rule: Dict[Tuple[Tuple[int, ...], ...], Tuple[str, int]] = {}
        for signature, actions in choices_by_signature.items():
            if not actions:
                return None
            rule[signature] = self._pick_symbolic_local_action(actions, radius)
        return rule

    def _derive_positional_symbolic_local_rewrite_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        radius: int,
        *,
        row_mod: Optional[int] = None,
        col_mod: Optional[int] = None,
    ) -> Optional[Dict[Tuple[Any, ...], Tuple[str, int]]]:
        choices_by_signature: Dict[Tuple[Any, ...], List[Tuple[str, int]]] = {}
        changed_any = False
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
                return None
            for row_idx, row in enumerate(input_grid):
                for col_idx, cell in enumerate(row):
                    patch = _extract_local_patch(input_grid, row_idx, col_idx, radius)
                    signature = self._local_rewrite_signature(
                        patch,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        row_mod=row_mod,
                        col_mod=col_mod,
                    )
                    patch_values = _flatten_patch(patch)
                    target = output_grid[row_idx][col_idx]
                    candidate_actions: List[Tuple[str, int]] = [
                        ("offset", idx)
                        for idx, patch_value in enumerate(patch_values)
                        if patch_value == target
                    ]
                    candidate_actions.append(("const", target))
                    if signature in choices_by_signature:
                        existing = set(choices_by_signature[signature])
                        intersection = [action for action in candidate_actions if action in existing]
                        if not intersection:
                            return None
                        choices_by_signature[signature] = intersection
                    else:
                        choices_by_signature[signature] = list(candidate_actions)
                    if cell != target:
                        changed_any = True
        if not changed_any:
            return None
        rule: Dict[Tuple[Any, ...], Tuple[str, int]] = {}
        for signature, actions in choices_by_signature.items():
            if not actions:
                return None
            rule[signature] = self._pick_symbolic_local_action(actions, radius)
        return rule

    def _pick_symbolic_local_action(
        self,
        actions: Sequence[Tuple[str, int]],
        radius: int,
    ) -> Tuple[str, int]:
        center_idx = ((2 * radius + 1) ** 2) // 2
        offset_indexes = sorted(value for kind, value in actions if kind == "offset")
        if center_idx in offset_indexes:
            return ("offset", center_idx)
        if offset_indexes:
            return ("offset", offset_indexes[0])
        const_values = sorted(value for kind, value in actions if kind == "const")
        return ("const", const_values[0] if const_values else 0)

    def _apply_symbolic_local_rewrite_rule(
        self,
        grid: Grid,
        radius: int,
        rule: Dict[Tuple[Tuple[int, ...], ...], Tuple[str, int]],
    ) -> Grid:
        output: Grid = []
        for row_idx, row in enumerate(grid):
            output_row: List[int] = []
            for col_idx, cell in enumerate(row):
                patch = _extract_local_patch(grid, row_idx, col_idx, radius)
                signature = _normalize_patch_signature(patch)
                action = rule.get(signature)
                if action is None:
                    output_row.append(cell)
                    continue
                mode, value = action
                if mode == "const":
                    output_row.append(int(value))
                    continue
                patch_values = _flatten_patch(patch)
                copied_value = patch_values[value] if 0 <= value < len(patch_values) else cell
                output_row.append(cell if copied_value == -1 else int(copied_value))
            output.append(output_row)
        return output

    def _apply_positional_symbolic_local_rewrite_rule(
        self,
        grid: Grid,
        radius: int,
        rule: Dict[Tuple[Any, ...], Tuple[str, int]],
        *,
        row_mod: Optional[int] = None,
        col_mod: Optional[int] = None,
    ) -> Grid:
        output: Grid = []
        for row_idx, row in enumerate(grid):
            output_row: List[int] = []
            for col_idx, cell in enumerate(row):
                patch = _extract_local_patch(grid, row_idx, col_idx, radius)
                signature = self._local_rewrite_signature(
                    patch,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    row_mod=row_mod,
                    col_mod=col_mod,
                )
                action = rule.get(signature)
                if action is None:
                    output_row.append(cell)
                    continue
                mode, value = action
                if mode == "const":
                    output_row.append(int(value))
                    continue
                patch_values = _flatten_patch(patch)
                copied_value = patch_values[value] if 0 <= value < len(patch_values) else cell
                output_row.append(cell if copied_value == -1 else int(copied_value))
            output.append(output_row)
        return output

    def _local_rewrite_signature(
        self,
        patch: Tuple[Tuple[int, ...], ...],
        *,
        row_idx: int,
        col_idx: int,
        row_mod: Optional[int] = None,
        col_mod: Optional[int] = None,
    ) -> Tuple[Any, ...]:
        signature: List[Any] = [_normalize_patch_signature(patch)]
        if row_mod:
            signature.append(("row_mod", row_idx % row_mod))
        if col_mod:
            signature.append(("col_mod", col_idx % col_mod))
        return tuple(signature)

    def _candidate_position_moduli(
        self,
        train_examples: Sequence[Dict[str, Any]],
        *,
        axis: str,
    ) -> List[int]:
        lengths: List[int] = []
        for example in train_examples:
            if not isinstance(example, dict):
                continue
            output_grid = example.get("output")
            if not _is_grid(output_grid):
                continue
            if axis == "row":
                lengths.append(len(output_grid))
            else:
                lengths.append(len(output_grid[0]) if output_grid else 0)
        if not lengths:
            return []
        max_mod = min(self._ARC_POSITIONAL_RULE_MAX_MOD, max(0, min(lengths) - 1))
        if max_mod < 2:
            return []
        return list(range(2, max_mod + 1))

    def _component_transform_by_name(self, transform_name: str) -> Optional[Callable[[Grid], Grid]]:
        for name, transform in self._COMPONENT_TRANSFORMS:
            if name == transform_name:
                return transform
        return None

    def _sort_components(
        self,
        components: Sequence[Dict[str, Any]],
        sort_mode: str,
    ) -> List[Dict[str, Any]]:
        if sort_mode == "top_left":
            return sorted(
                components,
                key=lambda component: (
                    component["bbox"][0],
                    component["bbox"][2],
                    component["bbox"][1],
                    component["bbox"][3],
                    component["color"],
                ),
            )
        if sort_mode == "left_top":
            return sorted(
                components,
                key=lambda component: (
                    component["bbox"][2],
                    component["bbox"][0],
                    component["bbox"][3],
                    component["bbox"][1],
                    component["color"],
                ),
            )
        if sort_mode == "area_desc":
            return sorted(
                components,
                key=lambda component: (
                    -component["area"],
                    component["bbox"][0],
                    component["bbox"][2],
                    component["color"],
                ),
            )
        if sort_mode == "area_asc":
            return sorted(
                components,
                key=lambda component: (
                    component["area"],
                    component["bbox"][0],
                    component["bbox"][2],
                    component["color"],
                ),
            )
        return list(components)

    def _infer_component_layout_gap(
        self,
        component_grids: Sequence[Grid],
        output_grid: Grid,
        *,
        axis: str,
    ) -> Optional[int]:
        if not component_grids:
            return None
        count = len(component_grids)
        heights = [len(grid) for grid in component_grids]
        widths = [len(grid[0]) if grid else 0 for grid in component_grids]
        if axis == "horizontal":
            if len(output_grid) != max(heights):
                return None
            occupied = sum(widths)
            slack = len(output_grid[0]) - occupied
        else:
            if len(output_grid[0]) != max(widths):
                return None
            occupied = sum(heights)
            slack = len(output_grid) - occupied
        if slack < 0:
            return None
        if count <= 1:
            return 0 if slack == 0 else None
        if slack % (count - 1) != 0:
            return None
        return slack // (count - 1)

    def _render_component_layout(
        self,
        component_grids: Sequence[Grid],
        *,
        axis: str,
        gap: int,
        background_color: int,
    ) -> Grid:
        if not component_grids:
            return []
        heights = [len(grid) for grid in component_grids]
        widths = [len(grid[0]) if grid else 0 for grid in component_grids]
        if axis == "horizontal":
            rows = max(heights)
            cols = sum(widths) + gap * max(len(component_grids) - 1, 0)
        else:
            rows = sum(heights) + gap * max(len(component_grids) - 1, 0)
            cols = max(widths)
        output = [[int(background_color) for _ in range(cols)] for _ in range(rows)]
        row_offset = 0
        col_offset = 0
        for grid in component_grids:
            for r_idx, row in enumerate(grid):
                for c_idx, cell in enumerate(row):
                    if cell == -1:
                        continue
                    output[row_offset + r_idx][col_offset + c_idx] = int(cell)
            if axis == "horizontal":
                col_offset += (len(grid[0]) if grid else 0) + gap
            else:
                row_offset += len(grid) + gap
        return output

    def _derive_component_layout_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        *,
        axis: str,
        sort_mode: str,
        transform_name: str,
    ) -> Optional[Dict[str, Any]]:
        transform = self._component_transform_by_name(transform_name)
        if transform is None:
            return None
        learned_gap: Optional[int] = None
        learned_background_color: Optional[int] = None
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            components = _extract_color_components(input_grid)
            if not components:
                return None
            ordered_components = self._sort_components(components, sort_mode)
            transformed_components = [transform(component["crop"]) for component in ordered_components]
            current_gap = self._infer_component_layout_gap(
                transformed_components,
                output_grid,
                axis=axis,
            )
            if current_gap is None:
                return None
            current_background_color = _background_color(output_grid)
            rendered = self._render_component_layout(
                transformed_components,
                axis=axis,
                gap=current_gap,
                background_color=current_background_color,
            )
            if rendered != output_grid:
                return None
            if learned_gap is None:
                learned_gap = current_gap
            elif learned_gap != current_gap:
                return None
            if learned_background_color is None:
                learned_background_color = current_background_color
            elif learned_background_color != current_background_color:
                return None
        if learned_gap is None or learned_background_color is None:
            return None
        return {
            "axis": axis,
            "sort_mode": sort_mode,
            "transform_name": transform_name,
            "gap": learned_gap,
            "background_color": learned_background_color,
        }

    def _apply_component_layout_rule(
        self,
        grid: Grid,
        rule: Dict[str, Any],
    ) -> Grid:
        axis = str(rule.get("axis", "") or "")
        sort_mode = str(rule.get("sort_mode", "") or "")
        transform_name = str(rule.get("transform_name", "") or "")
        gap = int(rule.get("gap", 0) or 0)
        background_color = int(rule.get("background_color", 0) or 0)
        transform = self._component_transform_by_name(transform_name)
        if axis not in {"horizontal", "vertical"} or transform is None:
            return _identity(grid)
        components = _extract_color_components(grid)
        if not components:
            return _identity(grid)
        ordered_components = self._sort_components(components, sort_mode)
        transformed_components = [transform(component["crop"]) for component in ordered_components]
        rendered = self._render_component_layout(
            transformed_components,
            axis=axis,
            gap=gap,
            background_color=background_color,
        )
        return rendered if _is_grid(rendered) else _identity(grid)

    def _derive_anchor_relation_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        *,
        relation_mode: str,
    ) -> Optional[Dict[str, Any]]:
        learned_background_color: Optional[int] = None
        learned_gap: Optional[int] = None
        learned_marker_area: Optional[int] = None
        learned_marker_color: Optional[int] = None
        saw_color_mismatch = False

        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            components = _extract_color_components(input_grid)
            if len(components) < 2:
                return None

            matched_rule: Optional[Tuple[Dict[str, Any], int, int, Optional[int]]] = None
            for marker in _rank_anchor_marker_candidates(components, output_grid):
                selected = _select_anchor_relation_components(
                    [component for component in components if component is not marker],
                    marker,
                    relation_mode=relation_mode,
                )
                if not selected:
                    continue
                background_color = _background_color(output_grid)
                rendered: Optional[Grid]
                current_gap: Optional[int] = None
                if relation_mode.startswith("compose_"):
                    axis = "horizontal" if "horizontal" in relation_mode else "vertical"
                    materialized_components = [
                        _materialize_component_crop(component["crop"], background_color=background_color)
                        for component in selected
                    ]
                    current_gap = self._infer_component_layout_gap(
                        materialized_components,
                        output_grid,
                        axis=axis,
                    )
                    if current_gap is None:
                        continue
                    rendered = self._render_component_layout(
                        materialized_components,
                        axis=axis,
                        gap=current_gap,
                        background_color=background_color,
                    )
                else:
                    rendered = _materialize_component_crop(
                        selected[0]["crop"],
                        background_color=background_color,
                    )
                if rendered != output_grid:
                    continue
                matched_rule = (
                    marker,
                    background_color,
                    int(marker.get("area", 0) or 0),
                    current_gap,
                )
                break

            if matched_rule is None:
                return None

            marker, background_color, marker_area, current_gap = matched_rule
            marker_color = int(marker.get("color", -1))
            if learned_marker_area is None:
                learned_marker_area = marker_area
            elif learned_marker_area != marker_area:
                return None
            if learned_marker_color is None:
                learned_marker_color = marker_color
            elif learned_marker_color != marker_color:
                saw_color_mismatch = True
            if relation_mode.startswith("compose_"):
                if learned_background_color is None:
                    learned_background_color = background_color
                elif learned_background_color != background_color:
                    return None
                if current_gap is None:
                    return None
                if learned_gap is None:
                    learned_gap = current_gap
                elif learned_gap != current_gap:
                    return None
            elif any(cell == -1 for row in selected[0]["crop"] for cell in row):
                if learned_background_color is None:
                    learned_background_color = background_color
                elif learned_background_color != background_color:
                    return None

        if learned_marker_area is None:
            return None
        if learned_background_color is None:
            learned_background_color = 0

        return {
            "relation_mode": relation_mode,
            "background_color": learned_background_color,
            "marker_area": learned_marker_area,
            "marker_color": None if saw_color_mismatch else learned_marker_color,
            "gap": learned_gap,
        }

    def _select_anchor_marker_for_rule(
        self,
        components: Sequence[Dict[str, Any]],
        rule: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        marker_area = int(rule.get("marker_area", 0) or 0)
        marker_color_raw = rule.get("marker_color")
        candidates = [
            component
            for component in components
            if int(component.get("area", 0) or 0) == marker_area
        ]
        if marker_color_raw is not None:
            marker_color = int(marker_color_raw)
            candidates = [
                component
                for component in candidates
                if int(component.get("color", -1)) == marker_color
            ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda component: (
                int(component.get("bbox", (0, 0, 0, 0))[0]),
                int(component.get("bbox", (0, 0, 0, 0))[2]),
                int(component.get("color", -1)),
            ),
        )

    def _apply_anchor_relation_rule(
        self,
        grid: Grid,
        rule: Dict[str, Any],
    ) -> Grid:
        relation_mode = str(rule.get("relation_mode", "") or "")
        background_color = int(rule.get("background_color", 0) or 0)
        gap = int(rule.get("gap", 0) or 0)
        components = _extract_color_components(grid)
        if len(components) < 2:
            return _identity(grid)
        marker = self._select_anchor_marker_for_rule(components, rule)
        if marker is None:
            return _identity(grid)
        selected = _select_anchor_relation_components(
            [component for component in components if component is not marker],
            marker,
            relation_mode=relation_mode,
        )
        if not selected:
            return _identity(grid)
        if relation_mode.startswith("compose_"):
            axis = "horizontal" if "horizontal" in relation_mode else "vertical"
            materialized_components = [
                _materialize_component_crop(component["crop"], background_color=background_color)
                for component in selected
            ]
            return self._render_component_layout(
                materialized_components,
                axis=axis,
                gap=gap,
                background_color=background_color,
            )
        return _materialize_component_crop(
            selected[0]["crop"],
            background_color=background_color,
        )

    def _derive_panel_trace_rule(
        self,
        train_examples: Sequence[Dict[str, Any]],
        *,
        order_mode: str,
    ) -> Optional[Dict[str, Any]]:
        separator_col: Optional[int] = None
        separator_color: Optional[int] = None
        column_bands: Optional[List[Tuple[int, int]]] = None
        tile_height: Optional[int] = None

        partial_rules: List[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]] = [{}]
        for example in train_examples:
            if not isinstance(example, dict):
                return None
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not _is_grid(input_grid) or not _is_grid(output_grid):
                return None
            separator = self._find_uniform_separator_column(input_grid)
            if separator is None:
                return None
            current_separator_col, current_separator_color = separator
            if separator_col is None:
                separator_col = current_separator_col
                separator_color = current_separator_color
            elif separator_col != current_separator_col or separator_color != current_separator_color:
                return None
            if len(output_grid[0]) != separator_col:
                return None
            layout = self._extract_panel_trace_layout(input_grid, separator_col)
            if layout is None:
                return None
            row_bands, current_column_bands = layout
            if not row_bands or not current_column_bands:
                return None
            if column_bands is None:
                column_bands = current_column_bands
            elif column_bands != current_column_bands:
                return None
            current_tile_heights = {end - start + 1 for start, end in row_bands}
            if len(current_tile_heights) != 1:
                return None
            current_tile_height = next(iter(current_tile_heights))
            if tile_height is None:
                tile_height = current_tile_height
            elif tile_height != current_tile_height:
                return None

            next_partial_rules: List[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]] = []
            for partial_rule in partial_rules:
                solutions = self._solve_panel_trace_example(
                    input_grid,
                    output_grid,
                    separator_col,
                    separator_color,
                    current_column_bands,
                    tile_height=current_tile_height,
                    order_mode=order_mode,
                    partial_rule=partial_rule,
                )
                next_partial_rules.extend(solutions)
            partial_rules = self._dedupe_panel_trace_rule_candidates(next_partial_rules)
            if not partial_rules:
                return None

        if separator_col is None or separator_color is None or column_bands is None or tile_height is None:
            return None
        if not partial_rules:
            return None
        return {
            "separator_col": separator_col,
            "separator_color": separator_color,
            "column_bands": column_bands,
            "tile_height": tile_height,
            "order_mode": order_mode,
            "pattern_ops": partial_rules[0],
        }

    def _find_uniform_separator_column(self, grid: Grid) -> Optional[Tuple[int, int]]:
        if not grid or not grid[0]:
            return None
        cols = len(grid[0])
        for col_idx in range(cols):
            column_values = {row[col_idx] for row in grid}
            if len(column_values) != 1:
                continue
            value = next(iter(column_values))
            if value != 0:
                return col_idx, int(value)
        return None

    def _extract_panel_trace_layout(
        self,
        grid: Grid,
        separator_col: int,
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        if not grid or not grid[0] or separator_col <= 0 or separator_col >= len(grid[0]):
            return None
        left_panel = [row[:separator_col] for row in grid]
        if not left_panel or not left_panel[0]:
            return None
        row_blank_mask = [all(cell == 0 for cell in row) for row in left_panel]
        col_blank_mask = [
            all(left_panel[row_idx][col_idx] == 0 for row_idx in range(len(left_panel)))
            for col_idx in range(separator_col)
        ]
        row_bands = self._segment_nonblank_ranges(row_blank_mask)
        col_bands = self._segment_nonblank_ranges(col_blank_mask)
        if not row_bands or not col_bands:
            return None
        return row_bands, col_bands

    def _segment_nonblank_ranges(self, blank_mask: Sequence[bool]) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, is_blank in enumerate(list(blank_mask) + [True]):
            if not is_blank and start is None:
                start = idx
            elif is_blank and start is not None:
                ranges.append((start, idx - 1))
                start = None
        return ranges

    def _extract_panel_trace_sequence(
        self,
        grid: Grid,
        separator_col: int,
        column_bands: Sequence[Tuple[int, int]],
        *,
        tile_height: int,
        order_mode: str,
    ) -> Optional[List[Tuple[Tuple[Tuple[int, ...], ...], int]]]:
        layout = self._extract_panel_trace_layout(grid, separator_col)
        if layout is None:
            return None
        row_bands, current_column_bands = layout
        if list(current_column_bands) != list(column_bands):
            return None
        if any((end - start + 1) != tile_height for start, end in row_bands):
            return None

        sequence: List[Tuple[Tuple[Tuple[int, ...], ...], int]] = []
        if order_mode == "col_major":
            band_pairs = [
                ((row_start, row_end), (col_start, col_end))
                for col_start, col_end in column_bands
                for row_start, row_end in row_bands
            ]
        else:
            band_pairs = [
                ((row_start, row_end), (col_start, col_end))
                for row_start, row_end in row_bands
                for col_start, col_end in column_bands
            ]

        for (row_start, row_end), (col_start, col_end) in band_pairs:
            tile = [row[col_start : col_end + 1] for row in grid[row_start : row_end + 1]]
            nonzero_colors = {cell for row in tile for cell in row if cell != 0}
            if not nonzero_colors:
                continue
            if len(nonzero_colors) != 1:
                return None
            color = int(next(iter(nonzero_colors)))
            signature = tuple(
                tuple(1 if cell != 0 else 0 for cell in row)
                for row in tile
            )
            sequence.append((signature, color))
        return sequence

    def _find_panel_trace_marker(
        self,
        grid: Grid,
        separator_col: int,
        separator_color: int,
    ) -> Optional[Tuple[int, int, int]]:
        if not grid or not grid[0] or separator_col >= len(grid[0]):
            return None
        markers = [
            (row_idx, col_idx, int(cell))
            for row_idx, row in enumerate(grid)
            for col_idx, cell in enumerate(row)
            if col_idx > separator_col and cell not in (0, separator_color)
        ]
        if len(markers) != 1:
            return None
        return markers[0]

    def _solve_panel_trace_example(
        self,
        input_grid: Grid,
        output_grid: Grid,
        separator_col: int,
        separator_color: int,
        column_bands: Sequence[Tuple[int, int]],
        *,
        tile_height: int,
        order_mode: str,
        partial_rule: Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]],
    ) -> List[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]]:
        sequence = self._extract_panel_trace_sequence(
            input_grid,
            separator_col,
            column_bands,
            tile_height=tile_height,
            order_mode=order_mode,
        )
        if sequence is None:
            return []
        marker = self._find_panel_trace_marker(input_grid, separator_col, separator_color)
        if marker is None:
            return []
        marker_row, marker_col, marker_color = marker
        start_col = marker_col - (separator_col + 1)
        if not (0 <= marker_row < len(output_grid) and 0 <= start_col < len(output_grid[0])):
            return []
        if output_grid[marker_row][start_col] != marker_color:
            return []

        target_nonzero = {
            (row_idx, col_idx)
            for row_idx, row in enumerate(output_grid)
            for col_idx, cell in enumerate(row)
            if cell != 0
        }
        if (marker_row, start_col) not in target_nonzero:
            return []
        consumed: Set[Tuple[int, int]] = {(marker_row, start_col)}
        solutions: List[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]] = []

        def backtrack(
            seq_idx: int,
            cursor: Tuple[int, int],
            used_cells: Set[Tuple[int, int]],
            rule_map: Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]],
        ) -> None:
            if len(solutions) >= self._ARC_PANEL_TRACE_MAX_PARTIAL_RULES:
                return
            if seq_idx >= len(sequence):
                if used_cells == target_nonzero:
                    solutions.append(dict(rule_map))
                return

            signature, color = sequence[seq_idx]
            assigned_op = rule_map.get(signature)
            for op, op_cells, next_cursor in self._panel_trace_candidate_ops(
                cursor,
                color,
                output_grid,
                used_cells,
            ):
                if assigned_op is not None and op != assigned_op:
                    continue
                next_rule_map = rule_map
                if assigned_op is None:
                    next_rule_map = dict(rule_map)
                    next_rule_map[signature] = op
                backtrack(
                    seq_idx + 1,
                    next_cursor,
                    used_cells | set(op_cells),
                    next_rule_map,
                )

        backtrack(0, (marker_row, start_col), consumed, dict(partial_rule))
        return solutions

    def _dedupe_panel_trace_rule_candidates(
        self,
        candidates: Sequence[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]],
    ) -> List[Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]]:
        unique: Dict[Tuple[Any, ...], Dict[Tuple[Tuple[int, ...], ...], Tuple[str, str, int]]] = {}
        for candidate in candidates:
            key = tuple(
                sorted(candidate.items(), key=lambda item: repr(item[0]))
            )
            if key in unique:
                continue
            unique[key] = candidate
            if len(unique) >= self._ARC_PANEL_TRACE_MAX_PARTIAL_RULES:
                break
        return list(unique.values())

    def _panel_trace_candidate_ops(
        self,
        cursor: Tuple[int, int],
        color: int,
        target_grid: Grid,
        consumed: Set[Tuple[int, int]],
    ) -> List[Tuple[Tuple[str, str, int], List[Tuple[int, int]], Tuple[int, int]]]:
        rows = len(target_grid)
        cols = len(target_grid[0]) if target_grid else 0
        row_idx, col_idx = cursor
        candidates: List[Tuple[Tuple[str, str, int], List[Tuple[int, int]], Tuple[int, int]]] = []

        next_row = row_idx + 1
        if next_row < rows:
            for length in range(1, col_idx + 2):
                cells = [(next_row, cur_col) for cur_col in range(col_idx - length + 1, col_idx + 1)]
                if all(
                    target_grid[cell_row][cell_col] == color and (cell_row, cell_col) not in consumed
                    for cell_row, cell_col in cells
                ):
                    candidates.append((("H", "L", length), cells, (next_row, col_idx - length + 1)))
            for length in range(1, cols - col_idx + 1):
                cells = [(next_row, cur_col) for cur_col in range(col_idx, col_idx + length)]
                if all(
                    target_grid[cell_row][cell_col] == color and (cell_row, cell_col) not in consumed
                    for cell_row, cell_col in cells
                ):
                    candidates.append((("H", "R", length), cells, (next_row, col_idx + length - 1)))

        for length in range(1, rows - row_idx):
            cells = [(cur_row, col_idx) for cur_row in range(row_idx + 1, row_idx + length + 1)]
            if all(
                target_grid[cell_row][cell_col] == color and (cell_row, cell_col) not in consumed
                for cell_row, cell_col in cells
            ):
                candidates.append((("V", "D", length), cells, (row_idx + length, col_idx)))
        return candidates

    def _apply_panel_trace_rule(
        self,
        grid: Grid,
        rule: Dict[str, Any],
    ) -> Grid:
        separator_col = int(rule.get("separator_col", 0) or 0)
        separator_color = int(rule.get("separator_color", 0) or 0)
        column_bands = rule.get("column_bands", [])
        tile_height = int(rule.get("tile_height", 0) or 0)
        order_mode = str(rule.get("order_mode", "col_major") or "col_major")
        pattern_ops = rule.get("pattern_ops", {})
        if not separator_col or not tile_height or not isinstance(pattern_ops, dict):
            return _identity(grid)
        if not grid or not grid[0] or separator_col >= len(grid[0]):
            return _identity(grid)

        sequence = self._extract_panel_trace_sequence(
            grid,
            separator_col,
            column_bands,
            tile_height=tile_height,
            order_mode=order_mode,
        )
        marker = self._find_panel_trace_marker(grid, separator_col, separator_color)
        if sequence is None or marker is None:
            return _identity(grid)
        marker_row, marker_col, marker_color = marker
        start_col = marker_col - (separator_col + 1)
        if not (0 <= marker_row < len(grid) and 0 <= start_col < separator_col):
            return _identity(grid)

        output = [[0 for _ in range(separator_col)] for _ in range(len(grid))]
        output[marker_row][start_col] = marker_color
        cursor = (marker_row, start_col)
        for signature, color in sequence:
            op = pattern_ops.get(signature)
            if not isinstance(op, tuple) or len(op) != 3:
                return _identity(grid)
            kind, direction, length = op
            length = int(length)
            row_idx, col_idx = cursor
            if kind == "H" and direction == "L":
                next_row = row_idx + 1
                start = col_idx - length + 1
                if next_row >= len(output) or start < 0:
                    return _identity(grid)
                for cur_col in range(start, col_idx + 1):
                    output[next_row][cur_col] = color
                cursor = (next_row, start)
                continue
            if kind == "H" and direction == "R":
                next_row = row_idx + 1
                end = col_idx + length - 1
                if next_row >= len(output) or end >= separator_col:
                    return _identity(grid)
                for cur_col in range(col_idx, end + 1):
                    output[next_row][cur_col] = color
                cursor = (next_row, end)
                continue
            if kind == "V" and direction == "D":
                end_row = row_idx + length
                if end_row >= len(output):
                    return _identity(grid)
                for cur_row in range(row_idx + 1, end_row + 1):
                    output[cur_row][col_idx] = color
                cursor = (end_row, col_idx)
                continue
            return _identity(grid)
        return output

    def _predict_exact_train_matches(
        self,
        grids: Sequence[Grid],
        train_examples: Sequence[Dict[str, Any]],
    ) -> Optional[List[Grid]]:
        outputs: List[Grid] = []
        for grid in grids:
            if not _is_grid(grid):
                return None
            matched_output = self._lookup_exact_train_output(grid, train_examples)
            if matched_output is None:
                return None
            outputs.append(matched_output)
        return outputs or None

    def _lookup_exact_train_output(
        self,
        input_grid: Grid,
        train_examples: Sequence[Dict[str, Any]],
    ) -> Optional[Grid]:
        for example in train_examples:
            if not isinstance(example, dict):
                continue
            candidate_input = example.get("input")
            candidate_output = example.get("output")
            if not _is_grid(candidate_input) or not _is_grid(candidate_output):
                continue
            if candidate_input == input_grid:
                return _identity(candidate_output)
        return None

    def _candidate_complexity(self, candidate: Any) -> int:
        if isinstance(candidate, dict):
            explicit = candidate.get("complexity")
            if explicit is not None:
                return int(explicit)
            name = str(candidate.get("name", "") or "")
        else:
            name = str(candidate or "")
        penalty = name.count("->")
        if "constant" in name:
            penalty += 1
        if "color" in name:
            penalty += 1
        if "component_layout_rule" in name:
            penalty += 2
        if "anchor_relation_rule" in name:
            penalty += 2
        if "panel_trace_rule" in name:
            penalty += 3
        if "positional_symbolic_local_rule" in name:
            penalty += 2
        if "symbolic_local_rule" in name:
            penalty += 2
        elif "local_rule" in name:
            penalty += 3
        if "nearest" in name:
            penalty += 2
        return penalty

    def _candidate_generalization_bias(self, candidate: Dict[str, Any]) -> int:
        explicit = candidate.get("generalization_bias")
        if explicit is not None:
            return int(explicit)
        kind = str(candidate.get("kind", "") or "")
        if kind == "component_layout_rule":
            return 4
        if kind == "anchor_relation_rule":
            return 4
        if kind == "constant_output":
            return 2
        if kind == "panel_trace_rule":
            return 3
        if kind == "positional_symbolic_local_rule":
            return 2
        if kind == "direct_op":
            return 4
        if kind == "op_plus_color_map":
            return 3
        if kind == "simulated_program":
            return 2
        if kind == "symbolic_local_rule":
            return 2
        if kind == "local_rule":
            return 1
        return 0

    def _attach_state_abstraction_meta(
        self,
        meta: Dict[str, Any],
        kwargs: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> None:
        outputs: List[Grid] = []
        if _is_grid(kwargs.get("grid")):
            outputs = [kwargs["grid"]]
        elif isinstance(kwargs.get("grids"), list):
            outputs = [grid for grid in kwargs.get("grids", []) if _is_grid(grid)]
        if not outputs:
            return
        abstraction: Dict[str, Any] = {
            "output_count": len(outputs),
            "output_shapes": [[len(grid), len(grid[0]) if grid else 0] for grid in outputs],
            "first_output_state": summarize_grid_state(outputs[0]),
        }
        task_payload = obs.get(self._ARC_KEY, {}) if isinstance(obs.get(self._ARC_KEY, {}), dict) else {}
        train_examples = task_payload.get("train", []) if isinstance(task_payload.get("train", []), list) else []
        test_inputs = task_payload.get("test_inputs", []) if isinstance(task_payload.get("test_inputs", []), list) else []
        transition_profile = summarize_arc_transition_profile(train_examples)
        if transition_profile:
            abstraction["task_transition_profile"] = transition_profile
        if test_inputs:
            alignments = [
                score_arc_transition_alignment(input_grid, output_grid, transition_profile)
                for input_grid, output_grid in zip(test_inputs, outputs)
                if _is_grid(input_grid) and _is_grid(output_grid)
            ]
            if alignments:
                abstraction["transition_alignment"] = sum(alignments) / len(alignments)
                meta["structured_answer_transition_alignment"] = abstraction["transition_alignment"]
        meta["structured_answer_state_abstraction"] = abstraction

    def _pack_arc_outputs(self, outputs: Sequence[Grid]) -> Dict[str, Any]:
        if len(outputs) == 1:
            return {"grid": outputs[0]}
        return {"grids": list(outputs)}

    def _normalize_llm_arc_submission_payload(self, kwargs: Any) -> Dict[str, Any]:
        if not isinstance(kwargs, dict):
            return {}
        if _is_grid(kwargs.get("grid")):
            return {"grid": _identity(kwargs.get("grid"))}
        grids = kwargs.get("grids")
        if isinstance(grids, list):
            clean_grids = [_identity(grid) for grid in grids if _is_grid(grid)]
            if clean_grids:
                return {"grids": clean_grids}
        return {}

    def _build_arc_llm_obs(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instruction": "Study the ARC train examples and submit the exact output grid for every test input.",
            "perception": {
                "goal": "Infer the transformation rule and submit the exact output grid(s).",
            },
            "function_signatures": {"submit_grid": {"required": []}},
            self._ARC_KEY: task_payload,
        }

    def _build_arc_llm_candidate_output(
        self,
        task_payload: Dict[str, Any],
        llm_client: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(task_payload, dict):
            return None
        train = list(task_payload.get("train", []) or [])
        test_inputs = list(task_payload.get("test_inputs", []) or [])
        if not train or not test_inputs:
            return None

        test_obs = self._build_arc_llm_obs(task_payload)
        test_kwargs = self.draft_kwargs_with_llm_only("submit_grid", test_obs, llm_client)
        test_payload = self._normalize_llm_arc_submission_payload(test_kwargs)
        test_outputs = list(test_payload.get("grids", []) or [])
        if not test_outputs and _is_grid(test_payload.get("grid")):
            test_outputs = [test_payload.get("grid")]
        test_outputs = [grid for grid in test_outputs if _is_grid(grid)]
        if len(test_outputs) != len(test_inputs):
            return None

        train_inputs = [example.get("input") for example in train if isinstance(example, dict) and _is_grid(example.get("input"))]
        train_targets = [example.get("output") for example in train if isinstance(example, dict) and _is_grid(example.get("output"))]
        if len(train_inputs) != len(train_targets) or not train_inputs:
            return None
        train_payload = {
            "train": train,
            "test_inputs": train_inputs,
        }
        train_obs = self._build_arc_llm_obs(train_payload)
        train_kwargs = self.draft_kwargs_with_llm_only("submit_grid", train_obs, llm_client)
        train_submission = self._normalize_llm_arc_submission_payload(train_kwargs)
        train_predictions = list(train_submission.get("grids", []) or [])
        if not train_predictions and _is_grid(train_submission.get("grid")):
            train_predictions = [train_submission.get("grid")]
        train_predictions = [grid for grid in train_predictions if _is_grid(grid)]
        if len(train_predictions) != len(train_targets):
            return None

        transition_profile = summarize_arc_transition_profile(train)
        exact_count = 0
        similarity_total = 0.0
        transition_alignment_total = 0.0
        for input_grid, predicted_grid, target_grid in zip(train_inputs, train_predictions, train_targets):
            if predicted_grid == target_grid:
                exact_count += 1
            similarity_total += _grid_similarity(predicted_grid, target_grid)
            transition_alignment_total += score_arc_transition_alignment(
                input_grid,
                predicted_grid,
                transition_profile,
            )
        train_count = max(len(train_targets), 1)
        avg_similarity = similarity_total / train_count
        avg_transition_alignment = transition_alignment_total / train_count
        score_key: ArcCandidateScore = (
            exact_count,
            0,
            0.5,
            avg_similarity,
            avg_similarity,
            avg_transition_alignment,
            avg_transition_alignment,
            0,
            0.5,
        )
        return {
            "output_id": "arc_output_llm_candidate",
            "program_id": "arc_program_llm_candidate",
            "program_name": "llm_candidate_draft",
            "program_kind": "llm_candidate_draft",
            "program_complexity": 1,
            "program_tags": ["llm_candidate"],
            "score": self._scalarize_arc_candidate_score(score_key),
            "score_key": list(score_key),
            "predicted_outputs": test_outputs,
            "first_output_state": summarize_grid_state(test_outputs[0]) if test_outputs and _is_grid(test_outputs[0]) else {},
            "transition_alignment": float(avg_transition_alignment),
            "solver_path": "candidate_output_search_with_llm_candidate",
        }

    def _merge_llm_candidate_output(
        self,
        task_payload: Dict[str, Any],
        search_result: Dict[str, Any],
        *,
        llm_client: Any,
    ) -> Dict[str, Any]:
        merged_result = dict(search_result or {})
        llm_row = self._build_arc_llm_candidate_output(task_payload, llm_client)
        if llm_row is None:
            merged_result["llm_candidate_considered"] = False
            merged_result["llm_candidate_selected"] = False
            return merged_result
        candidate_outputs = list(merged_result.get("candidate_outputs", []) or [])
        candidate_outputs.append(llm_row)
        ranked_outputs = rank_arc_candidate_outputs(
            task_payload,
            candidate_outputs,
            limit=self._ARC_CANDIDATE_OUTPUT_LIMIT,
        )
        merged_result["candidate_outputs"] = ranked_outputs
        merged_result["selected_output"] = dict(ranked_outputs[0]) if ranked_outputs else {}
        selected_output = merged_result.get("selected_output", {}) if isinstance(merged_result.get("selected_output", {}), dict) else {}
        llm_selected = str(selected_output.get("program_name", "") or "") == "llm_candidate_draft"
        if llm_selected:
            merged_result["selected_program"] = {
                "program_id": "arc_program_llm_candidate",
                "name": "llm_candidate_draft",
                "kind": "llm_candidate_draft",
                "score": float(selected_output.get("score", 0.0) or 0.0),
            }
            merged_result["solver_path"] = "candidate_output_search_with_llm_candidate"
        merged_result["llm_candidate_considered"] = True
        merged_result["llm_candidate_selected"] = llm_selected
        return merged_result

    def draft_kwargs_with_llm_only(
        self,
        function_name: str,
        obs: Dict[str, Any],
        llm_client: Any,
    ) -> Dict[str, Any]:
        return self._draft_with_llm(function_name, obs, llm_client)

    def _draft_with_llm(self, function_name: str, obs: Dict[str, Any], llm_client: Any) -> Dict[str, Any]:
        kwargs, _meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
        return kwargs

    def _draft_with_llm_with_trace(
        self,
        function_name: str,
        obs: Dict[str, Any],
        llm_client: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._build_llm_prompt(function_name, obs)
        system_prompt = self._build_llm_system_prompt(function_name, obs)
        cache_key = self._llm_draft_cache_key(function_name, prompt, system_prompt, llm_client)
        cached = self._llm_draft_cache_get(cache_key)
        if cached is not None:
            return cached
        gateway = ensure_llm_gateway(
            llm_client,
            route_name="structured_answer",
            capability_prefix="structured_output",
        )
        if gateway is None:
            return {}, {
                "llm_candidate_considered": False,
                "llm_trace": [{
                    "function_name": function_name,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": "",
                    "parsed_kwargs": {},
                    "error": "llm_gateway_unavailable",
                }],
            }
        local_machine_kwargs_call = (
            function_name in self._LOCAL_MACHINE_ATOMIC_FUNCTIONS
            or function_name in {"internet_fetch", "internet_fetch_project", "mirror_exec"}
        )
        max_tokens = 256 if local_machine_kwargs_call else 1024
        timeout_sec = 8.0 if local_machine_kwargs_call else None
        try:
            response = gateway.request_raw(
                STRUCTURED_OUTPUT_ACTION_KWARGS,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                system_prompt=system_prompt,
                think=False,
                timeout_sec=timeout_sec,
            )
            parsed = self._parse_llm_kwargs_response(response)
            kwargs = parsed if isinstance(parsed, dict) else {}
            meta = {
                "llm_candidate_considered": True,
                "llm_candidate_selected": bool(kwargs),
                "llm_trace": [{
                    "function_name": function_name,
                    "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
                    "route_name": "structured_answer",
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": str(response or ""),
                    "parsed_kwargs": dict(kwargs),
                    "error": "",
                }],
            }
            self._llm_draft_cache_put(cache_key, kwargs, meta)
            return kwargs, meta
        except Exception as exc:
            meta = {
                "llm_candidate_considered": True,
                "llm_candidate_selected": False,
                "llm_trace": [{
                    "function_name": function_name,
                    "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
                    "route_name": "structured_answer",
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": "",
                    "parsed_kwargs": {},
                    "error": str(exc),
                }],
            }
            self._llm_draft_cache_put(cache_key, {}, meta)
            return {}, meta

    def _llm_draft_cache_key(
        self,
        function_name: str,
        prompt: str,
        system_prompt: str,
        llm_client: Any,
    ) -> str:
        client_fields = {
            "class": type(llm_client).__name__,
            "id": id(llm_client),
        }
        for attr in ("base_url", "model", "model_name", "route_name"):
            value = getattr(llm_client, attr, None)
            if value:
                client_fields[attr] = str(value)
        payload = {
            "function_name": str(function_name or ""),
            "prompt": str(prompt or ""),
            "system_prompt": str(system_prompt or ""),
            "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
            "client": client_fields,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_draft_cache_get(self, cache_key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if not hasattr(self, "_llm_draft_cache"):
            self._llm_draft_cache = OrderedDict()
            self._llm_draft_cache_limit = 128
        cached = self._llm_draft_cache.get(cache_key)
        if cached is None:
            return None
        self._llm_draft_cache.move_to_end(cache_key)
        kwargs, meta = deepcopy(cached)
        meta["llm_cache_hit"] = True
        llm_trace = meta.get("llm_trace", [])
        if isinstance(llm_trace, list):
            for row in llm_trace:
                if isinstance(row, dict):
                    row["cache_hit"] = True
        return kwargs, meta

    def _llm_draft_cache_put(
        self,
        cache_key: str,
        kwargs: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        if not hasattr(self, "_llm_draft_cache"):
            self._llm_draft_cache = OrderedDict()
            self._llm_draft_cache_limit = 128
        self._llm_draft_cache[cache_key] = (deepcopy(kwargs), deepcopy(meta))
        self._llm_draft_cache.move_to_end(cache_key)
        while len(self._llm_draft_cache) > int(getattr(self, "_llm_draft_cache_limit", 128) or 128):
            self._llm_draft_cache.popitem(last=False)

    def _build_llm_prompt(self, function_name: str, obs: Dict[str, Any]) -> str:
        if self._ARC_KEY in obs:
            function_signatures = obs.get("function_signatures", {})
            signature = function_signatures.get(function_name, {}) if isinstance(function_signatures, dict) else {}
            visible_context = {
                "instruction": obs.get("instruction"),
                "goal": ((obs.get("perception") or {}) if isinstance(obs.get("perception"), dict) else {}).get("goal"),
                "function_name": function_name,
                "function_signature": signature,
                "arc_task": obs.get(self._ARC_KEY, {}),
            }
        else:
            local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
            function_signatures = obs.get("function_signatures", {})
            signature = function_signatures.get(function_name, {}) if isinstance(function_signatures, dict) else {}
            available_functions = sorted(str(name) for name in function_signatures.keys()) if isinstance(function_signatures, dict) else []
            visible_context = {
                "instruction": obs.get("instruction") or local_mirror.get("instruction"),
                "query": obs.get("query"),
                "perception": obs.get("perception"),
                "function_name": function_name,
                "function_signature": signature,
                "available_functions": available_functions,
                "local_mirror": self._compact_local_mirror_for_kwargs(local_mirror),
                "generation_contract": {
                    "deterministic_fallback_enabled": bool(local_mirror.get("deterministic_fallback_enabled", True)),
                    "require_llm_generation": bool(local_mirror.get("require_llm_generation", False)),
                    "require_market_evidence_reference": bool(local_mirror.get("require_market_evidence_reference", False)),
                    "require_non_template_product": bool(local_mirror.get("require_non_template_product", False)),
                },
            }
        return (
            "Fill executable kwargs for the selected function using the visible task context.\n"
            "Do not explain your reasoning.\n\n"
            f"Context:\n{json.dumps(visible_context, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _compact_local_mirror_for_kwargs(local_mirror: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(local_mirror, dict):
            return {}
        diff_summary = local_mirror.get("diff_summary", {}) if isinstance(local_mirror.get("diff_summary", {}), dict) else {}
        internet_ingress = local_mirror.get("internet_ingress", {}) if isinstance(local_mirror.get("internet_ingress", {}), dict) else {}
        investigation = local_mirror.get("investigation", {}) if isinstance(local_mirror.get("investigation", {}), dict) else {}
        artifacts = [
            {
                "artifact_id": str(row.get("artifact_id", "") or ""),
                "normalized_url": str(row.get("normalized_url", "") or ""),
                "fetch_kind": str(row.get("fetch_kind", "") or ""),
                "local_path": str(row.get("local_path", "") or ""),
                "bytes_written": int(row.get("bytes_written", 0) or 0),
            }
            for row in list(internet_ingress.get("artifacts", []) or [])[:5]
            if isinstance(row, dict)
        ]
        return {
            "instruction": str(local_mirror.get("instruction", "") or ""),
            "source_root": str(local_mirror.get("source_root", "") or ""),
            "mirror_root": str(local_mirror.get("mirror_root", "") or ""),
            "workspace_root": str(local_mirror.get("workspace_root", "") or ""),
            "control_root": str(local_mirror.get("control_root", "") or ""),
            "workspace_file_count": int(local_mirror.get("workspace_file_count", 0) or 0),
            "command_executed": bool(local_mirror.get("command_executed", False)),
            "latest_command_returncode": local_mirror.get("latest_command_returncode"),
            "diff_ref": dict(local_mirror.get("diff_ref", {}) or {}),
            "diff_summary": {
                "entry_count": int(diff_summary.get("entry_count", 0) or 0),
                "status_counts": dict(diff_summary.get("status_counts", {}) or {}),
                "examples": list(diff_summary.get("examples", []) or [])[:8],
                "examples_truncated": bool(diff_summary.get("examples_truncated", False)),
            },
            "investigation": {
                "candidate_files": list(investigation.get("candidate_files", local_mirror.get("candidate_files", [])) or [])[:20],
                "candidate_reason": str(investigation.get("candidate_reason", "") or ""),
                "last_tree": StructuredAnswerSynthesizer._compact_tree_result(investigation.get("last_tree", {}) or {}),
                "last_search": StructuredAnswerSynthesizer._compact_search_result(investigation.get("last_search", {}) or {}),
                "last_read": StructuredAnswerSynthesizer._compact_read_result(investigation.get("last_read", {}) or {}),
                "notes": list(investigation.get("notes", []) or [])[-12:],
                "hypotheses": list(investigation.get("hypotheses", []) or [])[-8:],
                "last_run_ref": str(investigation.get("last_run_ref", "") or ""),
            },
            "external_baselines": [
                {
                    "artifact_id": str(row.get("artifact_id", "") or ""),
                    "workspace_relative_path": str(row.get("workspace_relative_path", "") or ""),
                    "baseline_path": str(row.get("baseline_path", "") or ""),
                }
                for row in list(local_mirror.get("external_baselines", []) or [])[:5]
                if isinstance(row, dict)
            ],
            "internet_artifacts": artifacts,
        }

    @staticmethod
    def _compact_tree_result(last_tree: Any) -> Dict[str, Any]:
        if not isinstance(last_tree, dict):
            return {}
        entries = list(last_tree.get("entries", []) or [])
        slim_entries = []
        for row in entries[:80]:
            if not isinstance(row, dict):
                continue
            slim_entries.append({
                "path": str(row.get("path", "") or ""),
                "kind": str(row.get("kind", "") or ""),
                "depth": int(row.get("depth", 0) or 0),
            })
        return {
            "root": str(last_tree.get("root", "") or ""),
            "depth": int(last_tree.get("depth", 0) or 0),
            "entry_count": int(last_tree.get("entry_count", len(entries)) or 0),
            "entries": slim_entries,
            "entries_truncated": len(entries) > len(slim_entries),
        }

    @staticmethod
    def _compact_search_result(last_search: Any) -> Dict[str, Any]:
        if not isinstance(last_search, dict):
            return {}
        matches = list(last_search.get("matches", last_search.get("results", [])) or [])
        slim_matches = []
        for row in matches[:60]:
            if not isinstance(row, dict):
                continue
            slim_matches.append({
                "path": str(row.get("path", "") or ""),
                "line": row.get("line") if row.get("line") is not None else row.get("line_number"),
                "text": str(row.get("text", row.get("line_text", "")) or "")[:240],
            })
        compacted = {
            "query": str(last_search.get("query", "") or ""),
            "root": str(last_search.get("root", "") or ""),
            "match_count": int(last_search.get("match_count", len(matches)) or 0),
            "matches": slim_matches,
            "matches_truncated": len(matches) > len(slim_matches),
        }
        if "name_pattern" in last_search:
            compacted["name_pattern"] = str(last_search.get("name_pattern", "") or "")
        return compacted

    @staticmethod
    def _compact_read_result(last_read: Any) -> Dict[str, Any]:
        if not isinstance(last_read, dict):
            return {}
        content = str(last_read.get("content", last_read.get("text", "")) or "")
        if not content and isinstance(last_read.get("lines"), list):
            line_texts: List[str] = []
            for row in list(last_read.get("lines", []) or [])[:120]:
                if not isinstance(row, dict):
                    continue
                line_no = row.get("line", row.get("line_number", ""))
                text = str(row.get("text", row.get("content", "")) or "")
                prefix = f"{line_no}: " if line_no not in (None, "") else ""
                line_texts.append(f"{prefix}{text}")
            content = "\n".join(line_texts)
        return {
            "path": str(last_read.get("path", "") or ""),
            "start_line": last_read.get("start_line"),
            "end_line": last_read.get("end_line"),
            "line_count": last_read.get("line_count"),
            "content_excerpt": content[:4000],
            "content_truncated": len(content) > 4000,
        }

    def _build_llm_system_prompt(self, function_name: str, obs: Dict[str, Any]) -> str:
        if self._ARC_KEY in obs:
            return (
                "You are producing kwargs for an ARC task submission tool.\n"
                "Return EXACTLY one line in this format:\n"
                "SUBMISSION_JSON: {\"grid\": [[...]]}\n"
                "or\n"
                "SUBMISSION_JSON: {\"grids\": [[[...]], [[...]]]}\n"
                "Return no extra text."
            )
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        genuine_product_rules = ""
        if function_name == "mirror_exec" and (
            bool(local_mirror.get("require_llm_generation", False))
            or bool(local_mirror.get("require_market_evidence_reference", False))
            or bool(local_mirror.get("require_non_template_product", False))
        ):
            genuine_product_rules = (
                "This run requires genuine model-generated product work. Do not reproduce built-in examples or templates.\n"
                "If creating a product, choose a fresh product name and implementation from the research context.\n"
                "When market evidence is available, write a design or market note that cites at least one internet artifact id or normalized URL from local_mirror.internet_artifacts.\n"
                "Do not create SignalBrief AI or signalbrief_ai; those are banned template markers for verifier checks.\n"
            )
        return (
            "You are filling structured tool kwargs for a task-solving agent.\n"
            "Return EXACTLY one line in this format:\n"
            f"KWARGS_JSON: {{...}}\n"
            f"The JSON object must be executable kwargs for `{function_name}`.\n"
            f"{genuine_product_rules}"
            "For local-machine codebase work, prefer atomic actions: repo_tree, repo_find, repo_grep, file_read, note_write, candidate_files_set, apply_patch, edit_replace_range, run_test, run_lint, and read_run_output.\n"
            "LLM decides what to inspect or edit; the adapter decides how to execute it. Do not invent shell commands when an atomic action exists.\n"
            "For mirror_exec, use only a short emergency fallback command with purpose, target, and timeout_seconds; long python -c scripts are rejected.\n"
            "For internet_fetch_project, choose a public project URL and use source_type auto, git, or archive.\n"
            "Return no extra text."
        )

    def _parse_llm_kwargs_response(self, response: Any) -> Dict[str, Any]:
        text = str(response or "").strip()
        if not text:
            return {}
        text = self._strip_llm_fences(text)
        for prefix in ("SUBMISSION_JSON:", "KWARGS_JSON:"):
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(prefix):
                    payload = stripped[len(prefix) :].strip()
                    parsed = self._try_parse_llm_kwargs_json(payload)
                    if parsed:
                        return parsed
        parsed = self._try_parse_llm_kwargs_json(text)
        return parsed if parsed else {}

    def _try_parse_llm_kwargs_json(self, text: str) -> Dict[str, Any]:
        stripped = str(text or "").strip()
        if not stripped:
            return {}
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidate = stripped[start:end] if start >= 0 and end > start else stripped
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _strip_llm_fences(self, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

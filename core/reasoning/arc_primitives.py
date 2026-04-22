from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple


Grid = List[List[int]]


def is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def grid_area(grid: Grid) -> int:
    if not is_grid(grid):
        return 0
    return len(grid) * len(grid[0])


def majority_color(grid: Grid) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    if not counts:
        return 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def background_color(grid: Grid) -> int:
    if any(cell == 0 for row in grid for cell in row):
        return 0
    return majority_color(grid)


def grid_palette(grid: Grid, *, include_background: bool = False) -> Set[int]:
    if not is_grid(grid):
        return set()
    palette = {int(cell) for row in grid for cell in row}
    if not include_background:
        palette.discard(background_color(grid))
    return palette


def grid_similarity(predicted: Any, target: Any) -> float:
    if not is_grid(predicted) or not is_grid(target):
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
    for row_idx in range(overlap_rows):
        for col_idx in range(overlap_cols):
            if predicted[row_idx][col_idx] == target[row_idx][col_idx]:
                matches += 1
    overlap_match = matches / overlap_area
    area_penalty = overlap_area / max(pred_rows * pred_cols, tgt_rows * tgt_cols)
    return overlap_match * area_penalty


def extract_color_components(grid: Grid) -> List[Dict[str, Any]]:
    if not is_grid(grid):
        return []
    bg = background_color(grid)
    rows = len(grid)
    cols = len(grid[0])
    seen = set()
    components: List[Dict[str, Any]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            color = grid[row_idx][col_idx]
            if color == bg or (row_idx, col_idx) in seen:
                continue
            stack = [(row_idx, col_idx)]
            seen.add((row_idx, col_idx))
            cells: List[Tuple[int, int]] = []
            while stack:
                cur_row, cur_col = stack.pop()
                cells.append((cur_row, cur_col))
                for nxt_row, nxt_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if not (0 <= nxt_row < rows and 0 <= nxt_col < cols):
                        continue
                    if (nxt_row, nxt_col) in seen or grid[nxt_row][nxt_col] != color:
                        continue
                    seen.add((nxt_row, nxt_col))
                    stack.append((nxt_row, nxt_col))
            min_row = min(row for row, _ in cells)
            max_row = max(row for row, _ in cells)
            min_col = min(col for _, col in cells)
            max_col = max(col for _, col in cells)
            components.append({
                "color": int(color),
                "bbox": (min_row, max_row, min_col, max_col),
                "area": len(cells),
                "height": max_row - min_row + 1,
                "width": max_col - min_col + 1,
            })
    return components


def output_stability_score(outputs: Sequence[Grid]) -> float:
    valid_outputs = [grid for grid in outputs if is_grid(grid)]
    if not valid_outputs:
        return 0.0
    if len(valid_outputs) == 1:
        return 1.0
    shape_votes = {
        (len(grid), len(grid[0]) if grid else 0)
        for grid in valid_outputs
    }
    shape_score = 1.0 / max(len(shape_votes), 1)
    palette_scores: List[float] = []
    for left_idx, left in enumerate(valid_outputs):
        left_palette = grid_palette(left)
        for right in valid_outputs[left_idx + 1 :]:
            right_palette = grid_palette(right)
            if not left_palette and not right_palette:
                palette_scores.append(1.0)
                continue
            union = left_palette | right_palette
            palette_scores.append(len(left_palette & right_palette) / max(len(union), 1))
    palette_score = sum(palette_scores) / len(palette_scores) if palette_scores else 1.0
    return round(max(0.0, min(1.0, (shape_score * 0.6) + (palette_score * 0.4))), 6)


def object_correspondence_score(input_grid: Grid, output_grid: Grid) -> float:
    if not is_grid(input_grid) or not is_grid(output_grid):
        return 0.0
    input_components = extract_color_components(input_grid)
    output_components = extract_color_components(output_grid)
    if not input_components and not output_components:
        return 1.0
    count_score = 1.0 - (
        abs(len(input_components) - len(output_components))
        / max(len(input_components), len(output_components), 1)
    )
    input_area = sum(component["area"] for component in input_components)
    output_area = sum(component["area"] for component in output_components)
    area_score = 1.0 - (abs(input_area - output_area) / max(input_area, output_area, 1))
    input_palette = grid_palette(input_grid)
    output_palette = grid_palette(output_grid)
    if not input_palette and not output_palette:
        palette_score = 1.0
    else:
        palette_union = input_palette | output_palette
        palette_score = len(input_palette & output_palette) / max(len(palette_union), 1)
    return round(max(0.0, min(1.0, (count_score * 0.45) + (area_score * 0.30) + (palette_score * 0.25))), 6)


from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


Grid = List[List[int]]
Point = Tuple[int, int]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_grid(value: Any) -> Optional[Grid]:
    if isinstance(value, list) and value and all(isinstance(row, list) and row for row in value):
        width = None
        grid: Grid = []
        for row in value:
            if width is None:
                width = len(row)
            elif len(row) != width:
                return None
            try:
                grid.append([int(cell) for cell in row])
            except Exception:
                return None
        return grid
    return None


def _extract_grid(obs: Dict[str, Any]) -> Optional[Grid]:
    for key in ('frame', 'grid', 'observation'):
        if key in obs:
            grid = _normalize_grid(obs.get(key))
            if grid is not None:
                return grid
    raw_arc = obs.get('raw_arc_obs', {})
    if isinstance(raw_arc, dict):
        grid = _normalize_grid(raw_arc.get('frame'))
        if grid is not None:
            return grid
    return None


def _background_color(grid: Grid) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    if 0 in counts:
        return 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0] if counts else 0


def _connected_components(grid: Grid) -> List[Dict[str, Any]]:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    bg = _background_color(grid)
    seen = set()
    components: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r][c])
            if color == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: List[Point] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cc, cr))
                for nr, nc in ((cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)):
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if (nr, nc) in seen or int(grid[nr][nc]) != color:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            bbox = {
                'col_min': min(xs),
                'col_max': max(xs),
                'row_min': min(ys),
                'row_max': max(ys),
            }
            bbox['width'] = bbox['col_max'] - bbox['col_min'] + 1
            bbox['height'] = bbox['row_max'] - bbox['row_min'] + 1
            components.append({
                'color': color,
                'cells': cells,
                'bbox': bbox,
            })
    return components


def _normalize_bbox(raw_bbox: Any) -> Dict[str, Any]:
    bbox = _as_dict(raw_bbox)
    if not bbox:
        return {}
    col_min = int(bbox.get('col_min', bbox.get('x_min', 0)) or 0)
    col_max = int(bbox.get('col_max', bbox.get('x_max', col_min)) or col_min)
    row_min = int(bbox.get('row_min', bbox.get('y_min', 0)) or 0)
    row_max = int(bbox.get('row_max', bbox.get('y_max', row_min)) or row_min)
    width = int(bbox.get('width', max(0, col_max - col_min + 1)) or max(0, col_max - col_min + 1))
    height = int(bbox.get('height', max(0, row_max - row_min + 1)) or max(0, row_max - row_min + 1))
    return {
        'col_min': col_min,
        'col_max': col_max,
        'row_min': row_min,
        'row_max': row_max,
        'width': width,
        'height': height,
    }


def _symmetry_score(cells: Sequence[Point], bbox: Dict[str, Any], axis: str) -> float:
    points = set(cells)
    width = int(bbox.get('width', 0) or 0)
    height = int(bbox.get('height', 0) or 0)
    if width <= 0 or height <= 0:
        return 0.0
    matched = 0
    total = len(points)
    for x, y in points:
        if axis == 'vertical':
            mx = int(bbox['col_min'] + bbox['col_max'] - x)
            my = y
        else:
            mx = x
            my = int(bbox['row_min'] + bbox['row_max'] - y)
        if (mx, my) in points:
            matched += 1
    return round(matched / float(max(total, 1)), 4)


def _shape_semantic_candidates(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    width = float(features.get('width', 0) or 0)
    height = float(features.get('height', 0) or 0)
    fill_ratio = float(features.get('fill_ratio', 0.0) or 0.0)
    aspect_ratio = float(features.get('aspect_ratio', 1.0) or 1.0)
    edge_contact = bool(features.get('boundary_contact', False))
    vertical_sym = float(features.get('vertical_symmetry', 0.0) or 0.0)
    horizontal_sym = float(features.get('horizontal_symmetry', 0.0) or 0.0)
    area = float(features.get('area', 0) or 0)

    candidates: List[Dict[str, Any]] = []

    if area <= 4:
        candidates.append({'label': 'token_like', 'confidence': 0.72})
    if fill_ratio >= 0.70 and abs(width - height) <= 1:
        candidates.append({'label': 'block_like', 'confidence': 0.76})
    if max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6)) >= 2.0 and fill_ratio >= 0.45:
        candidates.append({'label': 'bar_like', 'confidence': 0.70})
    if max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6)) >= 1.6 and fill_ratio < 0.60 and min(vertical_sym, horizontal_sym) < 0.65:
        candidates.append({'label': 'directional_like', 'confidence': 0.62})
    if edge_contact:
        candidates.append({'label': 'boundary_structure', 'confidence': 0.64})
    if vertical_sym >= 0.8 and horizontal_sym >= 0.8:
        candidates.append({'label': 'highly_symmetric_structure', 'confidence': 0.58})
    if not candidates:
        candidates.append({'label': 'generic_object', 'confidence': 0.50})

    candidates.sort(key=lambda item: (-float(item['confidence']), item['label']))
    return candidates[:4]


def _role_candidates(features: Dict[str, Any], semantic_candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    area = float(features.get('area', 0) or 0)
    boundary_contact = bool(features.get('boundary_contact', False))
    top_semantic = str(semantic_candidates[0]['label']) if semantic_candidates else 'generic_object'
    roles: List[Dict[str, Any]] = []
    if top_semantic in {'directional_like', 'boundary_structure'}:
        roles.append({'role': 'hint_or_marker', 'confidence': 0.60})
    if area <= 6:
        roles.append({'role': 'interactive_token', 'confidence': 0.55})
    if boundary_contact:
        roles.append({'role': 'constraint_or_boundary', 'confidence': 0.58})
    if area >= 12:
        roles.append({'role': 'scene_anchor', 'confidence': 0.54})
    if not roles:
        roles.append({'role': 'generic_interactable', 'confidence': 0.45})
    roles.sort(key=lambda item: (-float(item['confidence']), item['role']))
    return roles[:3]


def _objects_from_perception(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    perception = _as_dict(obs.get('perception', {}))
    salient_objects = [item for item in _as_list(perception.get('salient_objects', [])) if isinstance(item, dict)]
    objects: List[Dict[str, Any]] = []
    for idx, item in enumerate(salient_objects):
        bbox = _normalize_bbox(item.get('bbox', {}))
        if not bbox:
            continue
        centroid = _as_dict(item.get('centroid', {}))
        width = int(bbox.get('width', 0) or 0)
        height = int(bbox.get('height', 0) or 0)
        area = int(item.get('area', width * height) or width * height)
        fill_ratio = min(1.0, area / float(max(width * height, 1)))
        features = {
            'width': width,
            'height': height,
            'area': area,
            'fill_ratio': round(fill_ratio, 4),
            'aspect_ratio': round(width / float(max(height, 1)), 4) if height else 0.0,
            'boundary_contact': bool(item.get('boundary_contact', False)),
            'vertical_symmetry': 0.0,
            'horizontal_symmetry': 0.0,
            'changed_overlap': int(item.get('changed_overlap', 0) or 0),
            'goal_like': bool(item.get('goal_like', False)),
            'rarity_score': round(_safe_float(item.get('rarity_score', 0.0), 0.0), 4),
        }
        semantic_candidates = _shape_semantic_candidates(features)
        role_candidates = _role_candidates(features, semantic_candidates)
        objects.append({
            'object_id': str(item.get('object_id', f'obj_{idx}') or f'obj_{idx}'),
            'color': int(item.get('color', 0) or 0),
            'bbox': bbox,
            'centroid': {
                'x': round(_safe_float(centroid.get('x', 0.0), 0.0), 4),
                'y': round(_safe_float(centroid.get('y', 0.0), 0.0), 4),
            },
            'geometric_features': features,
            'semantic_candidates': semantic_candidates,
            'role_candidates': role_candidates,
            'salience_score': round(_safe_float(item.get('salience_score', 0.0), 0.0), 4),
            'actionable_score': round(_safe_float(item.get('actionable_score', 0.0), 0.0), 4),
        })
    return objects


def _bbox_union(boxes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = [_normalize_bbox(box) for box in boxes if _normalize_bbox(box)]
    if not normalized:
        return {}
    col_min = min(int(box['col_min']) for box in normalized)
    col_max = max(int(box['col_max']) for box in normalized)
    row_min = min(int(box['row_min']) for box in normalized)
    row_max = max(int(box['row_max']) for box in normalized)
    return {
        'col_min': col_min,
        'col_max': col_max,
        'row_min': row_min,
        'row_max': row_max,
        'width': col_max - col_min + 1,
        'height': row_max - row_min + 1,
    }


def _edge_contacts(bbox: Dict[str, Any], grid_width: int, grid_height: int) -> List[str]:
    contacts: List[str] = []
    if not bbox or grid_width <= 0 or grid_height <= 0:
        return contacts
    if int(bbox.get('row_min', 1)) <= 0:
        contacts.append('top')
    if int(bbox.get('row_max', -1)) >= grid_height - 1:
        contacts.append('bottom')
    if int(bbox.get('col_min', 1)) <= 0:
        contacts.append('left')
    if int(bbox.get('col_max', -1)) >= grid_width - 1:
        contacts.append('right')
    return contacts


def _bbox_gap(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, int]:
    ax0, ax1 = int(a.get('col_min', 0) or 0), int(a.get('col_max', 0) or 0)
    ay0, ay1 = int(a.get('row_min', 0) or 0), int(a.get('row_max', 0) or 0)
    bx0, bx1 = int(b.get('col_min', 0) or 0), int(b.get('col_max', 0) or 0)
    by0, by1 = int(b.get('row_min', 0) or 0), int(b.get('row_max', 0) or 0)
    gap_x = max(0, max(bx0 - ax1 - 1, ax0 - bx1 - 1))
    gap_y = max(0, max(by0 - ay1 - 1, ay0 - by1 - 1))
    return gap_x, gap_y


def _cluster_objects(objects: Sequence[Dict[str, Any]], grid_width: int, grid_height: int) -> List[List[Dict[str, Any]]]:
    rows = [item for item in objects if isinstance(item, dict)]
    if len(rows) < 2:
        return []
    threshold = max(2, int(round(max(grid_width, grid_height) * 0.14))) if grid_width and grid_height else 6
    clusters: List[List[Dict[str, Any]]] = []
    visited = set()
    for start_idx, start in enumerate(rows):
        if start_idx in visited:
            continue
        stack = [start_idx]
        group: List[Dict[str, Any]] = []
        visited.add(start_idx)
        while stack:
            idx = stack.pop()
            current = rows[idx]
            group.append(current)
            current_box = _normalize_bbox(current.get('bbox', {}))
            current_centroid = _as_dict(current.get('centroid', {}))
            for other_idx, other in enumerate(rows):
                if other_idx in visited:
                    continue
                other_box = _normalize_bbox(other.get('bbox', {}))
                gap_x, gap_y = _bbox_gap(current_box, other_box)
                other_centroid = _as_dict(other.get('centroid', {}))
                dx = abs(_safe_float(current_centroid.get('x', 0.0), 0.0) - _safe_float(other_centroid.get('x', 0.0), 0.0))
                dy = abs(_safe_float(current_centroid.get('y', 0.0), 0.0) - _safe_float(other_centroid.get('y', 0.0), 0.0))
                if (gap_x <= 2 and gap_y <= 2) or max(dx, dy) <= threshold:
                    visited.add(other_idx)
                    stack.append(other_idx)
        if len(group) >= 2:
            clusters.append(group)
    return clusters


def _scene_elements_from_objects(objects: Sequence[Dict[str, Any]], grid_width: int, grid_height: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not objects:
        return [], {
            'layout_mode': 'empty',
            'scene_element_count': 0,
            'boundary_element_count': 0,
            'resource_like_element_count': 0,
            'interaction_cluster_count': 0,
        }

    elements: List[Dict[str, Any]] = []
    boundary_track_object_ids = set()
    resource_like_element_ids: List[str] = []

    def _add_element(element_type: str, role: str, confidence: float, bbox: Dict[str, Any], member_refs: Sequence[str], attributes: Dict[str, Any]) -> None:
        element_id = f'scene_elem_{len(elements)}'
        element = {
            'element_id': element_id,
            'element_type': element_type,
            'role': role,
            'confidence': round(float(confidence), 4),
            'bbox': _normalize_bbox(bbox),
            'member_refs': [str(item) for item in member_refs if str(item or '')],
            'attributes': dict(attributes),
        }
        elements.append(element)
        if role == 'resource_track_candidate':
            resource_like_element_ids.append(element_id)

    for obj in objects:
        bbox = _normalize_bbox(obj.get('bbox', {}))
        features = _as_dict(obj.get('geometric_features', {}))
        width = int(features.get('width', bbox.get('width', 0)) or bbox.get('width', 0) or 0)
        height = int(features.get('height', bbox.get('height', 0)) or bbox.get('height', 0) or 0)
        thinness = min(width, height)
        long_axis = max(width, height)
        span_ratio = max(
            width / float(max(grid_width, 1)),
            height / float(max(grid_height, 1)),
        ) if grid_width and grid_height else 0.0
        aspect_ratio = max(width / float(max(height, 1)), height / float(max(width, 1))) if width and height else 0.0
        boundary_contact = bool(features.get('boundary_contact', False))
        if boundary_contact and thinness <= 2 and span_ratio >= 0.55 and aspect_ratio >= 4.0:
            edge_contacts = _edge_contacts(bbox, grid_width, grid_height)
            role = 'resource_track_candidate' if span_ratio >= 0.8 else 'boundary_track'
            boundary_track_object_ids.add(str(obj.get('object_id', '') or ''))
            _add_element(
                'boundary_track',
                role,
                0.74 + min(0.2, span_ratio * 0.2),
                bbox,
                [str(obj.get('object_id', '') or '')],
                {
                    'edge_contacts': edge_contacts,
                    'span_ratio': round(span_ratio, 4),
                    'thinness': int(thinness),
                    'long_axis': int(long_axis),
                    'aspect_ratio': round(aspect_ratio, 4),
                },
            )

    non_track_objects = [
        obj for obj in objects
        if str(obj.get('object_id', '') or '') not in boundary_track_object_ids
    ]
    playfield_bbox = _bbox_union([obj.get('bbox', {}) for obj in non_track_objects])
    if playfield_bbox:
        _add_element(
            'playfield_region',
            'interactive_playfield',
            0.68,
            playfield_bbox,
            [str(obj.get('object_id', '') or '') for obj in non_track_objects[:8]],
            {
                'object_count': len(non_track_objects),
                'covers_boundary': bool(_edge_contacts(playfield_bbox, grid_width, grid_height)),
            },
        )

    interaction_clusters = _cluster_objects(non_track_objects, grid_width, grid_height)
    for cluster in interaction_clusters[:4]:
        cluster_bbox = _bbox_union([item.get('bbox', {}) for item in cluster])
        mean_actionable = sum(_safe_float(item.get('actionable_score', 0.0), 0.0) for item in cluster) / float(max(len(cluster), 1))
        _add_element(
            'object_cluster',
            'interaction_cluster',
            0.54 + min(0.22, mean_actionable * 0.24),
            cluster_bbox,
            [str(item.get('object_id', '') or '') for item in cluster],
            {
                'cluster_size': len(cluster),
                'mean_actionable_score': round(mean_actionable, 4),
            },
        )

    elements.sort(
        key=lambda item: (
            -float(item.get('confidence', 0.0) or 0.0),
            str(item.get('element_type', '') or ''),
            str(item.get('role', '') or ''),
        )
    )

    resource_like_count = sum(1 for item in elements if str(item.get('role', '') or '') == 'resource_track_candidate')
    boundary_count = sum(1 for item in elements if str(item.get('element_type', '') or '') == 'boundary_track')
    cluster_count = sum(1 for item in elements if str(item.get('element_type', '') or '') == 'object_cluster')
    layout_mode = 'object_field'
    if boundary_count and playfield_bbox:
        layout_mode = 'boundary_scaffold_and_playfield'
    elif boundary_count and cluster_count:
        layout_mode = 'boundary_scaffold_with_clusters'
    elif cluster_count >= 2:
        layout_mode = 'clustered_objects'
    elif len(objects) <= 2:
        layout_mode = 'sparse_objects'

    scene_summary = {
        'layout_mode': layout_mode,
        'scene_element_count': len(elements),
        'boundary_element_count': boundary_count,
        'resource_like_element_count': resource_like_count,
        'interaction_cluster_count': cluster_count,
        'resource_like_element_ids': resource_like_element_ids[:4],
        'playfield_bbox': playfield_bbox,
    }
    return elements[:8], scene_summary


def build_object_bindings(obs: Dict[str, Any], world_model_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    world_model_summary = _as_dict(world_model_summary)
    grid = _extract_grid(obs)
    objects: List[Dict[str, Any]] = []
    perception = _as_dict(obs.get('perception', {}))
    perception_grid_shape = _as_dict(perception.get('grid_shape', {}))
    grid_width = int(perception_grid_shape.get('width', 0) or 0)
    grid_height = int(perception_grid_shape.get('height', 0) or 0)

    if grid is not None:
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid else 0
        for idx, component in enumerate(_connected_components(grid)):
            cells = component['cells']
            bbox = component['bbox']
            width = int(bbox['width'])
            height = int(bbox['height'])
            area = len(cells)
            fill_ratio = round(area / float(max(width * height, 1)), 4)
            aspect_ratio = round(width / float(max(height, 1)), 4)
            boundary_contact = (
                bbox['col_min'] == 0 or bbox['row_min'] == 0 or
                bbox['col_max'] == len(grid[0]) - 1 or bbox['row_max'] == len(grid) - 1
            )
            centroid_x = round(sum(x for x, _ in cells) / float(max(area, 1)), 4)
            centroid_y = round(sum(y for _, y in cells) / float(max(area, 1)), 4)
            features = {
                'width': width,
                'height': height,
                'area': area,
                'fill_ratio': fill_ratio,
                'aspect_ratio': aspect_ratio,
                'boundary_contact': boundary_contact,
                'vertical_symmetry': _symmetry_score(cells, bbox, 'vertical'),
                'horizontal_symmetry': _symmetry_score(cells, bbox, 'horizontal'),
            }
            semantic_candidates = _shape_semantic_candidates(features)
            role_candidates = _role_candidates(features, semantic_candidates)
            salience_score = round(min(1.0, 0.25 + area * 0.03 + (0.08 if semantic_candidates and semantic_candidates[0]['label'] != 'generic_object' else 0.0)), 4)
            actionable_score = round(min(1.0, 0.20 + salience_score * 0.45 + (0.12 if not boundary_contact else 0.0)), 4)
            objects.append({
                'object_id': f'obj_{idx}',
                'color': int(component['color']),
                'bbox': bbox,
                'centroid': {'x': centroid_x, 'y': centroid_y},
                'geometric_features': features,
                'semantic_candidates': semantic_candidates,
                'role_candidates': role_candidates,
                'salience_score': salience_score,
                'actionable_score': actionable_score,
            })
    elif _as_list(perception.get('salient_objects', [])):
        objects = _objects_from_perception(obs)
    else:
        for idx, entity in enumerate(_as_list(world_model_summary.get('world_entities', []))[:8]):
            if not isinstance(entity, dict):
                continue
            bbox = _as_dict(entity.get('bbox', {}))
            area = int(entity.get('area', 0) or 0)
            width = int(bbox.get('width', 0) or 0)
            height = int(bbox.get('height', 0) or 0)
            fill_ratio = float(entity.get('fill_ratio', 0.0) or 0.0)
            features = {
                'width': width,
                'height': height,
                'area': area,
                'fill_ratio': fill_ratio,
                'aspect_ratio': round(width / float(max(height, 1)), 4) if height else 0.0,
                'boundary_contact': False,
                'vertical_symmetry': 0.0,
                'horizontal_symmetry': 0.0,
            }
            semantic_candidates = _shape_semantic_candidates(features)
            role_candidates = _role_candidates(features, semantic_candidates)
            objects.append({
                'object_id': str(entity.get('entity_id', f'obj_{idx}') or f'obj_{idx}'),
                'color': int(entity.get('color', 0) or 0),
                'bbox': bbox,
                'centroid': _as_dict(entity.get('centroid', {})),
                'geometric_features': features,
                'semantic_candidates': semantic_candidates,
                'role_candidates': role_candidates,
                'salience_score': round(min(1.0, 0.20 + area * 0.02), 4),
                'actionable_score': round(min(1.0, 0.20 + area * 0.015), 4),
            })
        if not grid_width or not grid_height:
            scene_bbox = _bbox_union([item.get('bbox', {}) for item in objects])
            grid_width = int(scene_bbox.get('col_max', 0) or 0) + 1 if scene_bbox else 0
            grid_height = int(scene_bbox.get('row_max', 0) or 0) + 1 if scene_bbox else 0

    objects.sort(key=lambda item: (-float(item.get('actionable_score', 0.0) or 0.0), -float(item.get('salience_score', 0.0) or 0.0), str(item.get('object_id', ''))))
    salient_ids = [str(item.get('object_id', '')) for item in objects[:4]]
    family_counter: Dict[str, int] = {}
    for item in objects:
        top_label = str(item.get('semantic_candidates', [{}])[0].get('label', 'generic_object'))
        family_counter[top_label] = family_counter.get(top_label, 0) + 1
    scene_elements, scene_summary = _scene_elements_from_objects(objects, grid_width, grid_height)

    return {
        'binding_mode': 'scene_and_object' if scene_elements else ('object_first' if objects else 'surface_first'),
        'object_count': len(objects),
        'objects': objects[:12],
        'salient_object_ids': salient_ids,
        'object_families': family_counter,
        'scene_element_count': len(scene_elements),
        'scene_elements': scene_elements,
        'scene_summary': scene_summary,
    }

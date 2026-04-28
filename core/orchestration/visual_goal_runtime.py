"""Visual feedback and goal-progress helpers for CoreMainLoop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.world_model.object_binding import build_object_bindings
from modules.world_model.task_frame import infer_task_frame
from core.orchestration.goal_progress_runtime import (
    derive_action_effect_signature,
    derive_goal_progress_assessment,
    recent_goal_progress_state,
)


class MainLoopVisualGoalMixin:
    def _surface_visual_feedback_fields(
        self,
        obs: Dict[str, Any],
        perception_summary: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(obs, dict) or not isinstance(perception_summary, dict):
            return

        changed_pixels = self._safe_float(perception_summary.get('changed_pixel_count'), default=None)
        if changed_pixels is not None and 'changed_pixel_count' not in obs:
            obs['changed_pixel_count'] = changed_pixels

        changed_bbox = perception_summary.get('changed_bbox')
        if isinstance(changed_bbox, dict) and changed_bbox and 'changed_bbox' not in obs:
            obs['changed_bbox'] = dict(changed_bbox)

        hotspot = perception_summary.get('suggested_hotspot')
        if isinstance(hotspot, dict) and hotspot and 'suggested_hotspot' not in obs:
            obs['suggested_hotspot'] = dict(hotspot)

        if (self._safe_float(obs.get('changed_pixel_count'), default=0.0) or 0.0) > 0.0:
            obs['observation_changed'] = bool(obs.get('observation_changed', False) or True)

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bbox_area(bbox: Any) -> float:
        if not isinstance(bbox, dict):
            return 0.0
        width = MainLoopVisualGoalMixin._safe_float(bbox.get('width'), default=0.0) or 0.0
        height = MainLoopVisualGoalMixin._safe_float(bbox.get('height'), default=0.0) or 0.0
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return float(width * height)

    def _extract_visual_feedback(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {
                'changed_pixel_count': 0.0,
                'grid_area': 0.0,
                'changed_ratio': 0.0,
                'changed_bbox_area': 0.0,
                'changed_bbox_ratio': 0.0,
                'hotspot_source': '',
            }

        perception = result.get('perception', {}) if isinstance(result.get('perception', {}), dict) else {}
        changed_pixels = self._safe_float(
            result.get('changed_pixel_count', perception.get('changed_pixel_count')),
            default=0.0,
        ) or 0.0
        changed_bbox = (
            result.get('changed_bbox')
            if isinstance(result.get('changed_bbox'), dict)
            else perception.get('changed_bbox')
        )
        changed_bbox_area = self._bbox_area(changed_bbox)
        grid_shape = perception.get('grid_shape', {}) if isinstance(perception.get('grid_shape', {}), dict) else {}
        grid_width = self._safe_float(grid_shape.get('width'), default=0.0) or 0.0
        grid_height = self._safe_float(grid_shape.get('height'), default=0.0) or 0.0
        grid_area = float(grid_width * grid_height) if grid_width > 0.0 and grid_height > 0.0 else 0.0
        changed_ratio = (changed_pixels / grid_area) if grid_area > 0.0 else 0.0
        changed_bbox_ratio = (changed_bbox_area / grid_area) if grid_area > 0.0 else 0.0
        hotspot = result.get('suggested_hotspot') if isinstance(result.get('suggested_hotspot'), dict) else perception.get('suggested_hotspot')
        hotspot_source = str((hotspot or {}).get('source', '') or '') if isinstance(hotspot, dict) else ''
        return {
            'changed_pixel_count': float(max(0.0, changed_pixels)),
            'grid_area': float(max(0.0, grid_area)),
            'changed_ratio': float(max(0.0, changed_ratio)),
            'changed_bbox_area': float(max(0.0, changed_bbox_area)),
            'changed_bbox_ratio': float(max(0.0, changed_bbox_ratio)),
            'hotspot_source': hotspot_source,
        }

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _surface_object_descriptors_from_obs(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(obs, dict):
            return []
        perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
        salient_objects = list(perception.get('salient_objects', []) or []) if isinstance(perception.get('salient_objects', []), list) else []
        descriptors: List[Dict[str, Any]] = []
        for obj in salient_objects:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {}
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
            x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
            y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
            y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
            descriptors.append({
                'anchor_ref': str(obj.get('object_id', '') or ''),
                'color': self._safe_int(obj.get('color')),
                'bbox': {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'width': int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1)),
                    'height': int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1)),
                },
                'centroid': {
                    'x': float(centroid.get('x', (x_min + x_max) / 2.0) or 0.0),
                    'y': float(centroid.get('y', (y_min + y_max) / 2.0) or 0.0),
                },
                'shape_labels': self._surface_shape_labels(obj),
                'boundary_contact': bool(obj.get('boundary_contact', False)),
                'salience_score': float(obj.get('salience_score', 0.0) or 0.0),
                'actionable_score': float(obj.get('actionable_score', 0.0) or 0.0),
            })
        return descriptors

    def _surface_shape_labels(self, obj: Dict[str, Any]) -> List[str]:
        semantic_rows = list(obj.get('semantic_candidates', []) or []) if isinstance(obj.get('semantic_candidates', []), list) else []
        labels: List[str] = []
        for row in semantic_rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get('label', '') or '').strip()
            if label and label not in labels:
                labels.append(label)
        if labels:
            return labels

        bbox = obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {}
        x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
        x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
        y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
        y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
        width = int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1))
        height = int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1))
        area = int(obj.get('area', width * height) or width * height)
        fill_ratio = float(area / float(max(width * height, 1)))
        aspect_ratio = float(width / float(max(height, 1))) if height > 0 else 1.0
        boundary_contact = bool(obj.get('boundary_contact', False))
        if area <= 4:
            labels.append('token_like')
        if width > 0 and height > 0 and fill_ratio >= 0.70 and abs(width - height) <= 1:
            labels.append('block_like')
        if width > 0 and height > 0 and max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6)) >= 2.0 and fill_ratio >= 0.45:
            labels.append('bar_like')
        if boundary_contact:
            labels.append('boundary_structure')
        if not labels:
            labels.append('generic_object')
        return labels

    def _match_click_to_descriptor(
        self,
        descriptors: List[Dict[str, Any]],
        point: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if not descriptors:
            return None
        px, py = int(point[0]), int(point[1])
        containing: List[Tuple[int, float, Dict[str, Any]]] = []
        nearest: List[Tuple[float, int, Dict[str, Any]]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            bbox = descriptor.get('bbox', {}) if isinstance(descriptor.get('bbox', {}), dict) else {}
            x_min = int(bbox.get('x_min', 0) or 0)
            x_max = int(bbox.get('x_max', x_min) or x_min)
            y_min = int(bbox.get('y_min', 0) or 0)
            y_max = int(bbox.get('y_max', y_min) or y_min)
            area = max(1, int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1)) * int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1)))
            centroid = descriptor.get('centroid', {}) if isinstance(descriptor.get('centroid', {}), dict) else {}
            distance = abs(float(centroid.get('x', 0.0) or 0.0) - float(px)) + abs(float(centroid.get('y', 0.0) or 0.0) - float(py))
            if x_min <= px <= x_max and y_min <= py <= y_max:
                containing.append((area, distance, descriptor))
            else:
                nearest.append((distance, area, descriptor))
        if containing:
            containing.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            return containing[0][2]
        if nearest:
            nearest.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            if nearest[0][0] <= 2.5:
                return nearest[0][2]
        return None

    def _family_summary_from_descriptor(
        self,
        descriptor: Optional[Dict[str, Any]],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = action.get('_candidate_meta', {}) if isinstance(getattr(action, 'get', None), type(dict.get)) and isinstance(action.get('_candidate_meta', {}), dict) else {}
        family = {
            'anchor_ref': '',
            'color': None,
            'shape_labels': [],
            'boundary_contact': False,
            'target_family': str(meta.get('target_family', '') or ''),
            'surface_click_role': str(meta.get('surface_click_role', '') or ''),
        }
        if isinstance(descriptor, dict):
            family['anchor_ref'] = str(descriptor.get('anchor_ref', '') or family['anchor_ref'])
            family['color'] = descriptor.get('color') if descriptor.get('color') is not None else family['color']
            family['shape_labels'] = [str(label) for label in list(descriptor.get('shape_labels', []) or []) if str(label)]
            family['boundary_contact'] = bool(descriptor.get('boundary_contact', family['boundary_contact']))
        if family['color'] is None and self._safe_int(meta.get('object_color')) is not None:
            family['color'] = int(self._safe_int(meta.get('object_color')) or 0)
        return family

    def _family_match_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0
        score = 0.0
        left_anchor = str(left.get('anchor_ref', '') or '')
        right_anchor = str(right.get('anchor_ref', '') or '')
        if left_anchor and right_anchor and left_anchor == right_anchor:
            score += 1.15
        left_color = self._safe_int(left.get('color'))
        right_color = self._safe_int(right.get('color'))
        if left_color is not None and right_color is not None:
            if left_color == right_color:
                score += 0.90
            else:
                score -= 0.20
        left_shapes = {str(item) for item in list(left.get('shape_labels', []) or []) if str(item)}
        right_shapes = {str(item) for item in list(right.get('shape_labels', []) or []) if str(item)}
        if left_shapes and right_shapes:
            overlap = len(left_shapes & right_shapes) / float(max(len(left_shapes | right_shapes), 1))
            score += 0.42 * overlap
        if str(left.get('target_family', '') or '') and str(left.get('target_family', '') or '') == str(right.get('target_family', '') or ''):
            score += 0.18
        if bool(left.get('boundary_contact', False)) and bool(right.get('boundary_contact', False)):
            score += 0.08
        return max(0.0, float(score))

    def _descriptor_affected_by_visual_change(
        self,
        descriptor: Dict[str, Any],
        changed_bbox: Optional[Dict[str, Any]],
        hotspot: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(descriptor, dict):
            return False
        bbox = descriptor.get('bbox', {}) if isinstance(descriptor.get('bbox', {}), dict) else {}
        x_min = int(bbox.get('x_min', 0) or 0)
        x_max = int(bbox.get('x_max', x_min) or x_min)
        y_min = int(bbox.get('y_min', 0) or 0)
        y_max = int(bbox.get('y_max', y_min) or y_min)
        if isinstance(changed_bbox, dict):
            cx_min = int(changed_bbox.get('x_min', 0) or 0)
            cx_max = int(changed_bbox.get('x_max', cx_min) or cx_min)
            cy_min = int(changed_bbox.get('y_min', 0) or 0)
            cy_max = int(changed_bbox.get('y_max', cy_min) or cy_min)
            if not (x_max < cx_min or cx_max < x_min or y_max < cy_min or cy_max < y_min):
                return True
        if isinstance(hotspot, dict):
            hx = self._safe_int(hotspot.get('x'))
            hy = self._safe_int(hotspot.get('y'))
            if hx is not None and hy is not None:
                if x_min <= hx <= x_max and y_min <= hy <= y_max:
                    return True
                centroid = descriptor.get('centroid', {}) if isinstance(descriptor.get('centroid', {}), dict) else {}
                distance = abs(float(centroid.get('x', 0.0) or 0.0) - float(hx)) + abs(float(centroid.get('y', 0.0) or 0.0) - float(hy))
                if distance <= 3.0:
                    return True
        return False

    def _progress_markers_show_positive_progress(self, progress_markers: List[Dict[str, Any]]) -> bool:
        task_progress_seen = False
        goal_stalled = False
        local_only_reaction = False
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            name = str(marker.get('name', '') or '')
            if name in {'goal_progressed', 'positive_reward'}:
                return True
            if name == 'task_progressed':
                task_progress_seen = True
            if name == 'goal_stalled':
                goal_stalled = True
            if name == 'local_only_reaction':
                local_only_reaction = True
            if name == 'terminal_reached' and bool(marker.get('success', False)):
                return True
        return bool(task_progress_seen and not goal_stalled and not local_only_reaction)

    def _derive_action_effect_signature(
        self,
        *,
        obs_before: Dict[str, Any],
        result: Dict[str, Any],
        action: Dict[str, Any],
        information_gain: float,
        progress_markers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return derive_action_effect_signature(
            self,
            obs_before=obs_before,
            result=result,
            action=action,
            information_gain=information_gain,
            progress_markers=progress_markers,
        )

    def _infer_level_goal_summary(
        self,
        *,
        obs_before: Dict[str, Any],
        world_model_summary: Optional[Dict[str, Any]] = None,
        task_frame_summary: Optional[Dict[str, Any]] = None,
        object_bindings_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = dict(world_model_summary or {})
        frame = dict(task_frame_summary or {})
        bindings = dict(object_bindings_summary or {})
        inferred = frame.get('inferred_level_goal', {}) if isinstance(frame.get('inferred_level_goal', {}), dict) else {}
        if inferred:
            return dict(inferred)
        try:
            if not bindings:
                bindings = dict(build_object_bindings(obs_before, summary) or {})
            if not frame:
                frame = dict(infer_task_frame(obs_before, summary, bindings, list(self._episode_trace[-8:])) or {})
            inferred = frame.get('inferred_level_goal', {}) if isinstance(frame.get('inferred_level_goal', {}), dict) else {}
            return dict(inferred)
        except Exception:
            return {}

    def _entry_has_local_anchor_signal(self, entry: Dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False
        if float(entry.get('information_gain', 0.0) or 0.0) >= 0.10:
            return True
        task_progress = entry.get('task_progress', {}) if isinstance(entry.get('task_progress', {}), dict) else {}
        if bool(task_progress.get('progressed', False)):
            return True
        if bool(entry.get('state_changed', False) or entry.get('observation_changed', False)):
            return True
        progress_markers = entry.get('progress_markers', []) if isinstance(entry.get('progress_markers', []), list) else []
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            if str(marker.get('name', '') or '') in {'task_progressed', 'goal_progressed', 'visual_change_detected', 'positive_reward'}:
                return True
        return False

    def _recent_goal_progress_state(
        self,
        episode_trace: List[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> Dict[str, Any]:
        return recent_goal_progress_state(self, episode_trace, limit=limit)

    def _recent_same_goal_anchor_streak(
        self,
        episode_trace: List[Dict[str, Any]],
        clicked_anchor_ref: str,
    ) -> int:
        anchor = str(clicked_anchor_ref or '').strip()
        if not anchor:
            return 0
        streak = 0
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            assessment = entry.get('goal_progress_assessment', {}) if isinstance(entry.get('goal_progress_assessment', {}), dict) else {}
            prior_anchor = str(assessment.get('clicked_anchor_ref', '') or '')
            if not prior_anchor:
                clicked_family = entry.get('clicked_family', {}) if isinstance(entry.get('clicked_family', {}), dict) else {}
                prior_anchor = str(clicked_family.get('anchor_ref', '') or '')
            if prior_anchor != anchor:
                break
            streak += 1
        return streak

    def _derive_goal_bundle_state(
        self,
        *,
        goal_summary: Dict[str, Any],
        goal_progress_assessment: Dict[str, Any],
        recent_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(goal_summary, dict) or not goal_summary:
            return {}
        assessment = goal_progress_assessment if isinstance(goal_progress_assessment, dict) else {}
        recent = recent_state if isinstance(recent_state, dict) else {}
        goal_anchor_refs = {
            str(ref or '').strip()
            for ref in list(goal_summary.get('goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        engaged_anchor_refs = set(recent.get('engaged_goal_anchor_refs', set()) or set())
        engaged_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(assessment.get('engaged_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        necessary_anchor_refs = set(recent.get('necessary_anchor_refs', set()) or set())
        clicked_anchor_ref = str(assessment.get('clicked_anchor_ref', '') or '')
        if bool(assessment.get('necessary_signal', False)) and clicked_anchor_ref:
            necessary_anchor_refs.add(clicked_anchor_ref)
        necessary_but_insufficient_anchor_refs = set(recent.get('necessary_but_insufficient_anchor_refs', set()) or set())
        if bool(assessment.get('necessary_but_insufficient', False)) and clicked_anchor_ref:
            necessary_but_insufficient_anchor_refs.add(clicked_anchor_ref)
        local_only_anchor_refs = set(recent.get('local_only_anchor_refs', set()) or set())
        if bool(assessment.get('local_only_signal', False)) and clicked_anchor_ref:
            local_only_anchor_refs.add(clicked_anchor_ref)
        controller_anchor_refs = set(recent.get('controller_anchor_refs', set()) or set())
        controller_supported_goal_anchor_refs = set(
            recent.get('controller_supported_goal_anchor_refs', set()) or set()
        )
        controller_supported_goal_colors = set(
            recent.get('controller_supported_goal_colors', set()) or set()
        )
        controller_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_colors |= {
            self._safe_int(color)
            for color in list(goal_summary.get('controller_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        controller_anchor_ref = str(assessment.get('controller_anchor_ref', '') or '')
        if bool(assessment.get('controller_effect', False)) and controller_anchor_ref:
            controller_anchor_refs.add(controller_anchor_ref)
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(assessment.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs &= goal_anchor_refs
        controller_supported_goal_colors |= {
            self._safe_int(color)
            for color in list(assessment.get('controller_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        coverage_target = int(goal_summary.get('coverage_target', 0) or 0)
        if coverage_target <= 0:
            coverage_target = max(1, len(goal_anchor_refs) or 1)
        requires_multi_anchor_coordination = bool(
            goal_summary.get('requires_multi_anchor_coordination', False)
            or coverage_target > 1
            or len(necessary_but_insufficient_anchor_refs) > 0
            or len(necessary_anchor_refs) >= 2
            or len(controller_supported_goal_anchor_refs) > 0
        )
        complementary_goal_anchor_refs = set(goal_anchor_refs - engaged_anchor_refs)
        complementary_goal_anchor_refs |= {
            ref
            for ref in controller_supported_goal_anchor_refs
            if ref and ref != controller_anchor_ref
        }
        active_combo_seed_anchor = str(recent.get('active_combo_seed_anchor', '') or '')
        active_seed_is_controller = bool(
            active_combo_seed_anchor and active_combo_seed_anchor in controller_anchor_refs
        )
        clicked_anchor_is_controller_supported = bool(
            clicked_anchor_ref and clicked_anchor_ref in controller_supported_goal_anchor_refs
        )
        if (
            bool(assessment.get('controller_effect', False))
            and bool(assessment.get('progressed', False))
            and controller_anchor_ref
            and requires_multi_anchor_coordination
        ):
            active_combo_seed_anchor = controller_anchor_ref
        if not active_combo_seed_anchor and necessary_but_insufficient_anchor_refs:
            active_combo_seed_anchor = sorted(necessary_but_insufficient_anchor_refs)[0]
        if (
            bool(assessment.get('necessary_but_insufficient', False))
            and clicked_anchor_ref
            and not (
                active_seed_is_controller
                and clicked_anchor_is_controller_supported
            )
        ):
            active_combo_seed_anchor = clicked_anchor_ref
        if active_combo_seed_anchor:
            complementary_goal_anchor_refs.discard(active_combo_seed_anchor)
        bundle_progress_fraction = min(
            1.0,
            len(engaged_anchor_refs & goal_anchor_refs) / float(max(coverage_target, 1))
        ) if goal_anchor_refs else 0.0
        coordination_pressure = 0.0
        if requires_multi_anchor_coordination:
            coordination_pressure = min(
                1.0,
                (len(necessary_but_insufficient_anchor_refs) * 0.45)
                + (len(complementary_goal_anchor_refs) * 0.18)
                + (0.16 if active_combo_seed_anchor else 0.0),
            )
        return {
            'goal_family': str(goal_summary.get('goal_family', '') or ''),
            'coverage_target': int(coverage_target),
            'requires_multi_anchor_coordination': bool(requires_multi_anchor_coordination),
            'engaged_anchor_refs': sorted(engaged_anchor_refs),
            'necessary_anchor_refs': sorted(necessary_anchor_refs),
            'necessary_but_insufficient_anchor_refs': sorted(necessary_but_insufficient_anchor_refs),
            'local_only_anchor_refs': sorted(local_only_anchor_refs),
            'controller_anchor_refs': sorted(controller_anchor_refs),
            'controller_supported_goal_anchor_refs': sorted(controller_supported_goal_anchor_refs),
            'controller_supported_goal_colors': sorted(controller_supported_goal_colors),
            'complementary_goal_anchor_refs': sorted(complementary_goal_anchor_refs)[:6],
            'active_combo_seed_anchor': active_combo_seed_anchor,
            'bundle_progress_fraction': round(float(bundle_progress_fraction), 4),
            'coordination_pressure': round(float(coordination_pressure), 4),
        }

    def _derive_goal_progress_assessment(
        self,
        *,
        goal_summary: Dict[str, Any],
        effect_trace: Dict[str, Any],
        information_gain: float,
        progress_markers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return derive_goal_progress_assessment(
            self,
            goal_summary=goal_summary,
            effect_trace=effect_trace,
            information_gain=information_gain,
            progress_markers=progress_markers,
        )

    def _build_goal_progress_markers(
        self,
        goal_progress_assessment: Dict[str, Any],
        fn_name: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(goal_progress_assessment, dict) or not goal_progress_assessment:
            return []
        markers: List[Dict[str, Any]] = []
        if bool(goal_progress_assessment.get('progressed', False)):
            markers.append({
                'name': 'goal_progressed',
                'function_name': fn_name,
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
                'goal_progress_score': float(goal_progress_assessment.get('goal_progress_score', 0.0) or 0.0),
                'goal_coverage_delta': int(goal_progress_assessment.get('goal_coverage_delta', 0) or 0),
            })
        if bool(goal_progress_assessment.get('stalled', False)):
            markers.append({
                'name': 'goal_stalled',
                'function_name': fn_name,
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'repeat_anchor_overrun': int(goal_progress_assessment.get('repeat_anchor_overrun', 0) or 0),
            })
        if bool(goal_progress_assessment.get('necessary_but_insufficient', False)):
            markers.append({
                'name': 'necessary_but_insufficient_anchor',
                'function_name': fn_name,
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
            })
        if bool(goal_progress_assessment.get('local_only_signal', False)):
            markers.append({
                'name': 'local_only_reaction',
                'function_name': fn_name,
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
            })
        return markers
    

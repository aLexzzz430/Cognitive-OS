from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SelectedActionExecutionResolver:
    """
    Materialize execution-time kwargs for selected actions before they hit the
    environment adapter.

    Current scope:
    - ACTION6: convert context-style kwargs into a single explicit x/y pair.

    The resolver is deliberately verbose in metadata so later debugging can see
    exactly which clues were considered and why one coordinate won.
    """

    def resolve(self, action: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action

        resolved = deepcopy(action)
        payload = resolved.get("payload", {}) if isinstance(resolved.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        function_name = str(tool_args.get("function_name", "") or "").strip().upper()
        if function_name != "ACTION6":
            return resolved

        kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
        kwargs = dict(kwargs)
        meta = resolved.setdefault("_candidate_meta", {})
        if not isinstance(meta, dict):
            meta = {}
            resolved["_candidate_meta"] = meta

        meta_role = str(meta.get("surface_click_role", "") or kwargs.get("role", "") or "")
        meta_target_family = str(meta.get("target_family", "") or kwargs.get("target_family", "") or "")
        meta_anchor_ref = str(meta.get("anchor_ref", "") or kwargs.get("anchor_ref", "") or "")
        meta_object_color = meta.get("object_color")
        if meta_object_color is None:
            meta_object_color = kwargs.get("object_color")
        meta_role_lower = meta_role.lower()
        meta_target_family_lower = meta_target_family.lower()
        meta_object_color_int = self._safe_int(meta_object_color)

        deferred_point = self._extract_explicit_point_from_context(kwargs, meta)
        if deferred_point is not None:
            kwargs["x"] = int(deferred_point[0])
            kwargs["y"] = int(deferred_point[1])
            tool_args["kwargs"] = kwargs
            payload["tool_args"] = tool_args
            resolved["payload"] = payload
            resolution = {
                "resolver": "SelectedActionExecutionResolver",
                "status": "resolved_from_deferred_context",
                "function_name": "ACTION6",
                "selected_point": {"x": int(deferred_point[0]), "y": int(deferred_point[1])},
                "selected_strategy": "deferred_context_point",
                "selected_reason": "used explicit deferred execution point already attached to candidate context",
                "incoming_kwargs_keys": sorted(kwargs.keys()),
                "required_keys": ["x", "y"],
                "missing_before": ["x", "y"],
                "used_clues": [{"kind": "deferred_context_point", "source": "candidate_context", "detail": {"anchor_ref": meta_anchor_ref, "role": meta_role, "target_family": meta_target_family}}],
                "candidate_points": [{"x": int(deferred_point[0]), "y": int(deferred_point[1]), "score": 1.5, "sources": ["candidate_context"]}],
            }
            meta["execution_click_parameterized"] = True
            meta["execution_kwarg_resolution"] = resolution
            meta["execution_kwarg_resolution_status"] = "resolved_from_deferred_context"
            return resolved

        x = kwargs.get("x")
        y = kwargs.get("y")
        explicit_point = (int(x), int(y)) if self._valid_coord(x) and self._valid_coord(y) else None

        candidate_points: Dict[Tuple[int, int], Dict[str, Any]] = {}
        clues: List[Dict[str, Any]] = []
        perception_sources = self._collect_perception_sources(kwargs, obs_before)
        frame_grid = self._extract_frame_grid(obs_before)
        source_summaries = [
            {
                "source": source_name,
                "keys": sorted(perception.keys())[:20],
            }
            for source_name, perception in perception_sources
        ]

        if explicit_point is not None:
            self._add_candidate_point(
                candidate_points,
                clues,
                point=explicit_point,
                score=0.74,
                clue={
                    "kind": "incoming_explicit_point",
                    "source": "selected_action",
                    "detail": {
                        "reason": "selected action already carried explicit coordinates before execution refinement",
                    },
                },
            )

        for source_name, perception in perception_sources:
            suggested_targets = perception.get("suggested_click_targets", [])
            if isinstance(suggested_targets, list):
                for index, target in enumerate(suggested_targets[:24]):
                    if not isinstance(target, dict):
                        continue
                    tx = target.get("x")
                    ty = target.get("y")
                    if not self._valid_coord(tx) or not self._valid_coord(ty):
                        continue
                    target_role = str(target.get("role", "") or "")
                    target_family = str(target.get("target_family", "") or "")
                    target_object_id = str(target.get("object_id", "") or "")
                    target_color = self._safe_int(target.get("color"))
                    priority = float(target.get("priority", 0.0) or 0.0)
                    score = 0.55 + max(0.0, min(priority, 1.0)) * 0.10
                    detail = {
                        "target_index": index,
                        "priority": round(priority, 4),
                        "role": target_role,
                        "reason": str(target.get("reason", "") or ""),
                        "target_family": target_family,
                    }
                    if meta_anchor_ref and target_object_id == meta_anchor_ref:
                        score += 0.55
                        detail["anchor_ref_match"] = True
                    if meta_role and target_role == meta_role:
                        score += 0.35
                        detail["role_match"] = True
                    if meta_target_family and target_family == meta_target_family:
                        score += 0.22
                        detail["target_family_match"] = True
                    if meta_object_color_int is not None and target_color == meta_object_color_int:
                        score += 0.12
                        detail["color_match"] = True
                    if target_role == "background_control" and (meta_anchor_ref or "salient" in meta_role_lower or "salient" in meta_target_family_lower):
                        score -= 0.30
                    self._add_candidate_point(
                        candidate_points,
                        clues,
                        point=(int(tx), int(ty)),
                        score=score,
                        clue={
                            "kind": "suggested_click_target",
                            "source": source_name,
                            "detail": detail,
                        },
                    )

            salient_objects = perception.get("salient_objects", [])
            if isinstance(salient_objects, list):
                for index, obj in enumerate(salient_objects[:24]):
                    if not isinstance(obj, dict):
                        continue
                    centroid = obj.get("centroid", {}) if isinstance(obj.get("centroid", {}), dict) else {}
                    tx = centroid.get("x")
                    ty = centroid.get("y")
                    if not self._valid_coord(tx) or not self._valid_coord(ty):
                        continue
                    object_id = str(obj.get("object_id", "") or "")
                    object_color = self._safe_int(obj.get("color"))
                    keepalive_tags = [str(item) for item in list(obj.get("keepalive_tags", []) or []) if str(item)]
                    score = 0.35
                    detail = {
                        "object_index": index,
                        "object_id": object_id,
                        "color": object_color,
                        "keepalive_tags": keepalive_tags[:8],
                    }
                    if meta_anchor_ref and object_id == meta_anchor_ref:
                        score += 0.70
                        detail["anchor_ref_match"] = True
                    if meta_object_color_int is not None and object_color == meta_object_color_int:
                        score += 0.18
                        detail["color_match"] = True
                    if "color9_small" in keepalive_tags and (meta_object_color_int == 9 or meta_anchor_ref or "goal" in meta_target_family_lower):
                        score += 0.20
                    if "boundary_touching_rare" in keepalive_tags and (meta_anchor_ref or "salient" in meta_role_lower):
                        score += 0.10
                    representative_point = self._representative_object_point(
                        frame_grid=frame_grid,
                        bbox=obj.get("bbox", {}),
                        object_color=object_color,
                        background_color=self._safe_int(perception.get("background_color")),
                    )
                    if representative_point is not None:
                        representative_score = 0.44
                        representative_detail = {
                            "object_index": index,
                            "object_id": object_id,
                            "color": object_color,
                            "bbox": self._compact_bbox(obj.get("bbox", {}) if isinstance(obj.get("bbox", {}), dict) else {}),
                            "scan_order": "top_to_bottom_then_left_to_right",
                            "derivation": "representative_visible_pixel",
                        }
                        if meta_anchor_ref and object_id == meta_anchor_ref:
                            representative_score += 0.78
                            representative_detail["anchor_ref_match"] = True
                        if meta_object_color_int is not None and object_color == meta_object_color_int:
                            representative_score += 0.16
                            representative_detail["color_match"] = True
                        if "center" in meta_role_lower or "salient" in meta_target_family_lower or "bound_object" in meta_target_family_lower:
                            representative_score += 0.12
                            representative_detail["click_legality_refinement"] = True
                        self._add_candidate_point(
                            candidate_points,
                            clues,
                            point=representative_point,
                            score=representative_score,
                            clue={
                                "kind": "salient_object_representative_pixel",
                                "source": source_name,
                                "detail": representative_detail,
                            },
                        )
                    self._add_candidate_point(
                        candidate_points,
                        clues,
                        point=(int(tx), int(ty)),
                        score=score,
                        clue={
                            "kind": "salient_object_centroid",
                            "source": source_name,
                            "detail": detail,
                        },
                    )

            changed_bbox = perception.get("changed_bbox", {}) if isinstance(perception.get("changed_bbox", {}), dict) else {}
            if changed_bbox:
                point = self._bbox_center(changed_bbox)
                if point is not None:
                    width = int(changed_bbox.get("width", 0) or 0)
                    height = int(changed_bbox.get("height", 0) or 0)
                    score = 1.05 if width == 1 and height == 1 else 1.0
                    self._add_candidate_point(
                        candidate_points,
                        clues,
                        point=point,
                        score=score,
                        clue={
                            "kind": "changed_bbox_center",
                            "source": source_name,
                            "detail": {
                                "bbox": self._compact_bbox(changed_bbox),
                                "derivation": "bbox_center",
                                "single_cell": bool(width == 1 and height == 1),
                            },
                        },
                    )

            hotspot = perception.get("suggested_hotspot", {}) if isinstance(perception.get("suggested_hotspot", {}), dict) else {}
            hx = hotspot.get("x")
            hy = hotspot.get("y")
            if self._valid_coord(hx) and self._valid_coord(hy):
                hotspot_source = str(hotspot.get("source", "") or "")
                hotspot_score = 0.92 if hotspot_source == "changed_pixels" else 0.76
                self._add_candidate_point(
                    candidate_points,
                    clues,
                    point=(int(hx), int(hy)),
                    score=hotspot_score,
                    clue={
                        "kind": "suggested_hotspot",
                        "source": source_name,
                        "detail": {
                            "hotspot_source": hotspot_source,
                        },
                    },
                )

            active_bbox = perception.get("active_bbox", {}) if isinstance(perception.get("active_bbox", {}), dict) else {}
            active_center = self._bbox_center(active_bbox)
            if active_center is not None:
                self._add_candidate_point(
                    candidate_points,
                    clues,
                    point=active_center,
                    score=0.28,
                    clue={
                        "kind": "active_bbox_center",
                        "source": source_name,
                        "detail": {
                            "bbox": self._compact_bbox(active_bbox),
                            "derivation": "bbox_center",
                        },
                    },
                )

            grid_shape = perception.get("grid_shape", {}) if isinstance(perception.get("grid_shape", {}), dict) else {}
            grid_center = self._grid_center(grid_shape)
            if grid_center is not None:
                self._add_candidate_point(
                    candidate_points,
                    clues,
                    point=grid_center,
                    score=0.08,
                    clue={
                        "kind": "grid_center",
                        "source": source_name,
                        "detail": {
                            "grid_shape": {
                                "width": int(grid_shape.get("width", 0) or 0),
                                "height": int(grid_shape.get("height", 0) or 0),
                            },
                            "derivation": "grid_midpoint",
                        },
                    },
                )

        ranked_points = sorted(
            candidate_points.values(),
            key=lambda item: (-float(item.get("score", 0.0) or 0.0), int(item.get("y", 0)), int(item.get("x", 0))),
        )

        resolution: Dict[str, Any] = {
            "resolver": "SelectedActionExecutionResolver",
            "status": "failed",
            "function_name": "ACTION6",
            "incoming_kwargs_keys": sorted(kwargs.keys()),
            "required_keys": ["x", "y"],
            "missing_before": [] if explicit_point is not None else ["x", "y"],
            "perception_sources": source_summaries,
            "used_clues": clues[:12],
            "candidate_points": ranked_points[:8],
            "selected_point": {"x": explicit_point[0], "y": explicit_point[1]} if explicit_point is not None else None,
            "selected_strategy": "",
            "selected_reason": (
                "kept explicit coordinates because no better legality-aware point was found"
                if explicit_point is not None
                else "no resolvable coordinate clues found before execution"
            ),
        }

        selected = None
        if explicit_point is not None:
            explicit_entry = candidate_points.get(explicit_point)
            selected = explicit_entry
            if ranked_points:
                best = ranked_points[0]
                if self._should_refine_explicit_point(best, explicit_entry, meta_role_lower, meta_target_family_lower):
                    selected = best
        elif ranked_points:
            selected = ranked_points[0]

        if selected is not None:
            kwargs["x"] = int(selected["x"])
            kwargs["y"] = int(selected["y"])
            tool_args["kwargs"] = kwargs
            payload["tool_args"] = tool_args
            resolved["payload"] = payload

            selected_support = list(selected.get("supporting_clues", []))
            selected_strategy = "highest_ranked_candidate_point"
            selected_status = "resolved"
            if explicit_point is not None and (int(selected["x"]), int(selected["y"])) == explicit_point:
                selected_strategy = "already_explicit"
                selected_status = "explicit_passthrough"
            elif explicit_point is not None:
                selected_strategy = "refined_explicit_to_representative_clickable_pixel"
                selected_status = "explicit_refined"
            resolution.update(
                {
                    "status": selected_status,
                    "selected_point": {"x": int(selected["x"]), "y": int(selected["y"])},
                    "selected_strategy": selected_strategy,
                    "selected_reason": (
                        f"selected ({int(selected['x'])}, {int(selected['y'])}) from "
                        f"{', '.join(selected_support) if selected_support else 'ranked clues'} "
                        f"with score {float(selected.get('score', 0.0) or 0.0):.4f}"
                    ),
                    "resolved_from": list(selected.get("sources", [])),
                }
            )
            meta["execution_click_parameterized"] = True

        meta["execution_kwarg_resolution"] = resolution
        meta["execution_kwarg_resolution_status"] = str(resolution.get("status", "failed") or "failed")
        return resolved

    def _extract_explicit_point_from_context(self, kwargs: Dict[str, Any], meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        for key in ("resolved_execution_coords", "execution_point", "selected_execution_coords"):
            value = kwargs.get(key)
            if isinstance(value, dict) and self._valid_coord(value.get("x")) and self._valid_coord(value.get("y")):
                return int(value.get("x")), int(value.get("y"))
            value = meta.get(key)
            if isinstance(value, dict) and self._valid_coord(value.get("x")) and self._valid_coord(value.get("y")):
                return int(value.get("x")), int(value.get("y"))
        return None

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _collect_perception_sources(
        self,
        kwargs: Dict[str, Any],
        obs_before: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        sources: List[Tuple[str, Dict[str, Any]]] = []
        seen: set[Tuple[str, ...]] = set()

        def _append(source_name: str, value: Any) -> None:
            if not isinstance(value, dict):
                return
            key = tuple(sorted(str(k) for k in value.keys()))
            marker = (source_name, *key)
            if marker in seen:
                return
            seen.add(marker)
            sources.append((source_name, value))

        _append("kwargs.perception", kwargs.get("perception"))
        if isinstance(obs_before, dict):
            _append("obs.perception", obs_before.get("perception"))
            world_model = obs_before.get("world_model", {})
            if isinstance(world_model, dict):
                _append("obs.world_model.perception", world_model.get("perception"))
        return sources

    def _add_candidate_point(
        self,
        candidate_points: Dict[Tuple[int, int], Dict[str, Any]],
        clues: List[Dict[str, Any]],
        *,
        point: Tuple[int, int],
        score: float,
        clue: Dict[str, Any],
    ) -> None:
        x, y = point
        entry = candidate_points.get((x, y))
        compact_clue = {
            "kind": str(clue.get("kind", "") or ""),
            "source": str(clue.get("source", "") or ""),
            "score": round(float(score or 0.0), 4),
            "point": {"x": int(x), "y": int(y)},
            "detail": clue.get("detail", {}),
        }
        clues.append(compact_clue)
        if entry is None:
            entry = {
                "x": int(x),
                "y": int(y),
                "score": round(float(score or 0.0), 4),
                "supporting_clues": [compact_clue["kind"]],
                "sources": [compact_clue["source"]],
                "details": [compact_clue["detail"]],
            }
            candidate_points[(x, y)] = entry
            return

        existing_score = float(entry.get("score", 0.0) or 0.0)
        if score > existing_score:
            entry["score"] = round(float(score or 0.0), 4)
        if compact_clue["kind"] and compact_clue["kind"] not in entry["supporting_clues"]:
            entry["supporting_clues"].append(compact_clue["kind"])
        if compact_clue["source"] and compact_clue["source"] not in entry["sources"]:
            entry["sources"].append(compact_clue["source"])
        if len(entry["details"]) < 4:
            entry["details"].append(compact_clue["detail"])

    def _bbox_center(self, bbox: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if not isinstance(bbox, dict):
            return None
        x_min = bbox.get("x_min", bbox.get("col_min"))
        x_max = bbox.get("x_max", bbox.get("col_max"))
        y_min = bbox.get("y_min", bbox.get("row_min"))
        y_max = bbox.get("y_max", bbox.get("row_max"))
        if not all(self._valid_coord(value) for value in (x_min, x_max, y_min, y_max)):
            return None
        return (
            int((int(x_min) + int(x_max)) // 2),
            int((int(y_min) + int(y_max)) // 2),
        )

    def _grid_center(self, grid_shape: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if not isinstance(grid_shape, dict):
            return None
        width = grid_shape.get("width")
        height = grid_shape.get("height")
        if not self._valid_coord(width) or not self._valid_coord(height):
            return None
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return None
        return (max(0, width // 2), max(0, height // 2))

    def _extract_frame_grid(self, obs_before: Dict[str, Any]) -> Optional[List[List[int]]]:
        if not isinstance(obs_before, dict):
            return None
        raw_frame = obs_before.get("frame")
        if not isinstance(raw_frame, list) or not raw_frame:
            return None
        first = raw_frame[0]
        if hasattr(first, "tolist"):
            try:
                first = first.tolist()
            except Exception:
                return None
        if not isinstance(first, list) or not first:
            return None

        grid: List[List[int]] = []
        for row in first:
            if hasattr(row, "tolist"):
                try:
                    row = row.tolist()
                except Exception:
                    return None
            if not isinstance(row, list):
                return None
            grid.append([int(cell) for cell in row])
        return grid or None

    def _representative_object_point(
        self,
        *,
        frame_grid: Optional[List[List[int]]],
        bbox: Any,
        object_color: Optional[int],
        background_color: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        if not isinstance(bbox, dict):
            return None
        x_min = self._safe_int(bbox.get("x_min", bbox.get("col_min")))
        x_max = self._safe_int(bbox.get("x_max", bbox.get("col_max")))
        y_min = self._safe_int(bbox.get("y_min", bbox.get("row_min")))
        y_max = self._safe_int(bbox.get("y_max", bbox.get("row_max")))
        if None in (x_min, x_max, y_min, y_max):
            return None

        if frame_grid is None:
            return int(x_min), int(y_min)

        height = len(frame_grid)
        width = len(frame_grid[0]) if height > 0 else 0
        if width <= 0:
            return None

        x_min = max(0, min(width - 1, int(x_min)))
        x_max = max(0, min(width - 1, int(x_max)))
        y_min = max(0, min(height - 1, int(y_min)))
        y_max = max(0, min(height - 1, int(y_max)))
        if x_min > x_max or y_min > y_max:
            return None

        if object_color is not None:
            for yy in range(y_min, y_max + 1):
                row = frame_grid[yy]
                for xx in range(x_min, x_max + 1):
                    if xx < len(row) and int(row[xx]) == int(object_color):
                        return xx, yy

        for yy in range(y_min, y_max + 1):
            row = frame_grid[yy]
            for xx in range(x_min, x_max + 1):
                if xx >= len(row):
                    continue
                pixel = int(row[xx])
                if background_color is not None:
                    if pixel != int(background_color):
                        return xx, yy
                elif pixel >= 0:
                    return xx, yy

        return int(x_min), int(y_min)

    def _should_refine_explicit_point(
        self,
        best: Dict[str, Any],
        explicit_entry: Optional[Dict[str, Any]],
        meta_role_lower: str,
        meta_target_family_lower: str,
    ) -> bool:
        if not isinstance(best, dict):
            return False
        if explicit_entry is None:
            return True
        best_point = (int(best.get("x", 0) or 0), int(best.get("y", 0) or 0))
        explicit_point = (int(explicit_entry.get("x", 0) or 0), int(explicit_entry.get("y", 0) or 0))
        if best_point == explicit_point:
            return False

        best_score = float(best.get("score", 0.0) or 0.0)
        explicit_score = float(explicit_entry.get("score", 0.0) or 0.0)
        best_clues = {str(item) for item in list(best.get("supporting_clues", []) or [])}
        object_center_like = (
            "center" in meta_role_lower
            or "salient_object" in meta_target_family_lower
            or "bound_object" in meta_target_family_lower
        )
        if "salient_object_representative_pixel" in best_clues and object_center_like:
            return best_score >= explicit_score + 0.05
        return best_score >= explicit_score + 0.18

    def _compact_bbox(self, bbox: Dict[str, Any]) -> Dict[str, int]:
        compact: Dict[str, int] = {}
        for key in ("x_min", "x_max", "y_min", "y_max", "width", "height"):
            if self._valid_coord(bbox.get(key)):
                compact[key] = int(bbox.get(key))
        return compact

    def _valid_coord(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and int(value) >= 0

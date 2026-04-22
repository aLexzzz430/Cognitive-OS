from __future__ import annotations

"""ARC-AGI-3 execution compiler over generic intervention targets.

This module is intentionally *adapter-side*, not part of the generic world
model. It compiles generic intervention targets into ARC-AGI-3 surface actions.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.world_model.execution_compiler import CompiledAction
from modules.world_model.intervention_targets import InterventionTarget


class ARCAGI3InterventionExecutionCompiler:
    ACTION_MAP = {
        "navigate_up": "ACTION1",
        "navigate_down": "ACTION2",
        "navigate_left": "ACTION3",
        "navigate_right": "ACTION4",
        "confirm": "ACTION5",
        "pointer_select": "ACTION6",
        "pointer_activate": "ACTION6",
        "probe_state_change": "ACTION6",
        "probe_relation": "ACTION6",
    }

    def compile(
        self,
        target: InterventionTarget,
        *,
        available_functions: Sequence[str],
        obs: Optional[Dict[str, Any]] = None,
    ) -> List[CompiledAction]:
        available = {str(name).strip() for name in available_functions if str(name).strip()}
        compiled: List[CompiledAction] = []
        coords = self._resolve_coords(target, obs)
        for action_mode in target.candidate_actions:
            action_name = self.ACTION_MAP.get(str(action_mode).strip())
            if not action_name or action_name not in available:
                continue
            kwargs: Dict[str, Any] = {}
            rationale = [f"compiled_from:{action_mode}", f"target:{target.target_id}"]
            if action_name == "ACTION6":
                if coords is None:
                    continue
                kwargs = {"x": int(coords[0]), "y": int(coords[1])}
                rationale.append("uses_centroid_or_bbox_projection")
            compiled.append(
                CompiledAction(
                    action_name=action_name,
                    kwargs=kwargs,
                    score=float(target.confidence),
                    rationale=rationale,
                )
            )
        return compiled

    def _resolve_coords(
        self,
        target: InterventionTarget,
        obs: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, int]]:
        projection = dict(target.execution_projection or {})
        centroid = projection.get("centroid")
        if isinstance(centroid, dict) and centroid.get("x") is not None and centroid.get("y") is not None:
            try:
                return int(round(float(centroid["x"]))), int(round(float(centroid["y"])))
            except (TypeError, ValueError):
                pass
        bbox = projection.get("bbox")
        if isinstance(bbox, dict):
            try:
                x = (int(bbox["col_min"]) + int(bbox["col_max"])) / 2.0
                y = (int(bbox["row_min"]) + int(bbox["row_max"])) / 2.0
                return int(round(x)), int(round(y))
            except (KeyError, TypeError, ValueError):
                pass
        perception = obs.get("perception", {}) if isinstance(obs, dict) and isinstance(obs.get("perception", {}), dict) else {}
        hotspot = perception.get("suggested_hotspot") if isinstance(perception, dict) else None
        if isinstance(hotspot, dict) and hotspot.get("x") is not None and hotspot.get("y") is not None:
            try:
                return int(round(float(hotspot["x"]))), int(round(float(hotspot["y"])))
            except (TypeError, ValueError):
                pass
        return None

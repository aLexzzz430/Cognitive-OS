from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_ARC3_ACTION_RE = re.compile(r"^ACTION\d+$", re.IGNORECASE)
_INTERNAL_ONLY_ACTIONS = {
    "inspect",
    "reflect",
    "replan",
    "retrieve",
    "deliberate",
    "probe",
}


def _failure_category(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    raw_result = entry.get("result", {}) if isinstance(entry.get("result", {}), dict) else {}
    failure_reason = str(
        entry.get("failure_reason")
        or raw_result.get("failure_reason")
        or ""
    ).strip().lower()
    if not failure_reason:
        progress_markers = entry.get("progress_markers", []) if isinstance(entry.get("progress_markers", []), list) else []
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            if str(marker.get("name", "") or "") == "failure_reason":
                failure_reason = str(marker.get("value", "") or "").strip().lower()
                if failure_reason:
                    break
    if not failure_reason or failure_reason == "none":
        return ""
    if (
        "schema_failure" in failure_reason
        or failure_reason in {"illegal_click_coordinate_or_remote_rejection", "arc_agi3_schema_failure_remote_rejection"}
        or "requires explicit x/y" in failure_reason
    ):
        return "schema_failure"
    if "resource" in failure_reason or "timeout" in failure_reason:
        return "resource_failure"
    if "illegal" in failure_reason or "adapter_error" in failure_reason:
        return "execution_failure"
    return "failure"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_arc3_visible_functions(obs_before: Optional[Dict[str, Any]]) -> Set[str]:
    visible: Set[str] = set()
    if not isinstance(obs_before, dict):
        return visible

    def _extend(raw_values: Any) -> None:
        if not isinstance(raw_values, list):
            return
        for value in raw_values:
            text = str(value or "").strip()
            if text:
                visible.add(text)

    _extend(obs_before.get("available_functions", []))
    _extend(obs_before.get("available_action_names", []))
    _extend(obs_before.get("visible_functions", []))

    novel_api = obs_before.get("novel_api", {})
    if hasattr(novel_api, "raw"):
        novel_api = novel_api.raw
    if isinstance(novel_api, dict):
        _extend(novel_api.get("available_functions", []))
        _extend(novel_api.get("visible_functions", []))
        _extend(novel_api.get("discovered_functions", []))

    world_state = obs_before.get("world_state", {})
    if isinstance(world_state, dict):
        _extend(world_state.get("active_functions", []))

    return visible


def is_arc3_external_function(function_name: str) -> bool:
    text = str(function_name or "").strip().upper()
    return bool(text) and bool(_ARC3_ACTION_RE.match(text))


def is_arc3_surface(obs_before: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obs_before, dict):
        return False
    if str(obs_before.get("type", "") or "").strip().lower() == "arc_agi3":
        return True
    world_state = obs_before.get("world_state", {})
    if isinstance(world_state, dict) and str(world_state.get("task_family", "") or "").strip().lower() == "arc_agi3":
        return True
    return False


def _extract_action_function_name(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    fn_name = str(action.get("function_name", "") or "").strip()
    if fn_name:
        return fn_name
    payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
    return str(tool_args.get("function_name", "") or "").strip()


def _extract_action_kind(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("kind", "") or "").strip().lower()


def _extract_action_xy(action: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if not isinstance(action, dict):
        return None
    raw_x = action.get("x")
    raw_y = action.get("y")
    if raw_x is None or raw_y is None:
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
        raw_x = kwargs.get("x")
        raw_y = kwargs.get("y")
    try:
        if raw_x is None or raw_y is None:
            return None
        return int(raw_x), int(raw_y)
    except (TypeError, ValueError):
        return None


def _is_internal_action(action: Dict[str, Any], function_name: str) -> bool:
    kind = _extract_action_kind(action)
    fn_name = str(function_name or "").strip().lower()
    if kind == "wait" or fn_name == "wait":
        return False
    if kind in _INTERNAL_ONLY_ACTIONS:
        return True
    return fn_name in _INTERNAL_ONLY_ACTIONS or kind == "probe"


def _entry_has_state_change(entry: Dict[str, Any]) -> bool:
    reward = float(entry.get("reward", 0.0) or 0.0)
    if reward != 0.0:
        return True
    info_gain = float(entry.get("information_gain", 0.0) or 0.0)
    if info_gain > 0.0:
        return True
    task_progress = _as_dict(entry.get("task_progress", {}))
    if bool(task_progress.get("progressed", False) or task_progress.get("solved", False)):
        return True
    goal_progress = _as_dict(entry.get("goal_progress_assessment", {}))
    if bool(goal_progress.get("progressed", False)):
        return True
    effect_signature = _as_dict(entry.get("action_effect_signature", {}))
    for key in ("changed_pixel_count", "changed_pixels", "changed_cells"):
        try:
            if float(effect_signature.get(key, 0.0) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    progress_markers = entry.get("progress_markers", []) if isinstance(entry.get("progress_markers", []), list) else []
    for marker in progress_markers:
        if not isinstance(marker, dict):
            continue
        if str(marker.get("name", "") or "") in {
            "visual_change_detected",
            "goal_progressed",
            "task_progressed",
            "positive_reward",
        }:
            return True
    return False


def summarize_arc3_action_coverage(
    episode_trace: Sequence[Dict[str, Any]],
    candidate_viability_log: Sequence[Dict[str, Any]] | None = None,
    governance_log: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    trace = list(episode_trace or [])
    if not trace:
        return {}

    arc3_entries = [
        entry for entry in trace
        if isinstance(entry, dict) and is_arc3_surface(_as_dict(entry.get("observation", {})))
    ]
    if not arc3_entries:
        return {}

    seen_available_actions: Set[str] = set()
    tried_external_counts: Dict[str, int] = {}
    action6_coordinates: Set[Tuple[int, int]] = set()
    repeated_effective_counts: Dict[str, int] = {}
    first_state_change_tick: Optional[int] = None
    first_reward_tick: Optional[int] = None
    wait_count = 0
    internal_count = 0
    no_op_count = 0
    schema_failure_count = 0
    action6_schema_failure_count = 0
    exploratory_no_effect_count = 0

    for entry in arc3_entries:
        observation = _as_dict(entry.get("observation", {}))
        seen_available_actions.update(extract_arc3_visible_functions(observation))

        action = _as_dict(entry.get("action", {}))
        tick = int(entry.get("tick", 0) or 0)
        function_name = _extract_action_function_name(action)
        effective = _entry_has_state_change(entry)
        reward = float(entry.get("reward", 0.0) or 0.0)
        failure_category = _failure_category(entry)

        if first_state_change_tick is None and effective:
            first_state_change_tick = tick
        if first_reward_tick is None and reward != 0.0:
            first_reward_tick = tick

        if failure_category == "schema_failure":
            schema_failure_count += 1
            if function_name.upper() == "ACTION6":
                action6_schema_failure_count += 1
        elif reward == 0.0 and not effective:
            no_op_count += 1
            if function_name and is_arc3_external_function(function_name):
                exploratory_no_effect_count += 1

        if not function_name:
            continue
        if function_name == "wait" or _extract_action_kind(action) == "wait":
            wait_count += 1
            continue
        if _is_internal_action(action, function_name):
            internal_count += 1
            continue
        if not is_arc3_external_function(function_name):
            continue

        tried_external_counts[function_name] = tried_external_counts.get(function_name, 0) + 1
        if function_name.upper() == "ACTION6":
            xy = _extract_action_xy(action)
            if xy is not None:
                action6_coordinates.add(xy)
        if effective:
            action_identity = function_name.upper()
            xy = _extract_action_xy(action)
            if xy is not None:
                action_identity = f"{action_identity}@{xy[0]},{xy[1]}"
            repeated_effective_counts[action_identity] = repeated_effective_counts.get(action_identity, 0) + 1

    blocked_action_count = 0
    for event in list(candidate_viability_log or []):
        if not isinstance(event, dict):
            continue
        blocked_action_count += len([
            str(item or "").strip()
            for item in list(event.get("suppressed_functions", []) or [])
            if str(item or "").strip()
        ])
        blocked_action_count += len([
            str(item or "").strip()
            for item in list(event.get("blocked_functions", []) or [])
            if str(item or "").strip()
        ])
        if str(event.get("event", "") or "") in {
            "governance_candidate_viability_failure",
            "recent_no_progress_candidates_suppressed",
        }:
            blocked_action_count += 1

    for row in list(governance_log or []):
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "") or "")
        if "block_non_low_risk" in reason or "advisory_mode_requires_wait" in reason:
            blocked_action_count += 1

    total_ticks = max(len(arc3_entries), 1)
    wait_internal_count = wait_count + internal_count
    seen_external_actions = sorted(fn for fn in seen_available_actions if is_arc3_external_function(fn))
    tried_external_actions = sorted(tried_external_counts)
    untried_external_actions = sorted(set(seen_external_actions) - set(tried_external_actions))

    return {
        "enabled": True,
        "total_ticks": len(arc3_entries),
        "seen_available_actions": sorted(seen_available_actions),
        "seen_external_actions": seen_external_actions,
        "tried_external_actions": tried_external_actions,
        "external_action_try_counts": dict(sorted(tried_external_counts.items())),
        "untried_external_actions": untried_external_actions,
        "action_coverage_ratio": round(len(tried_external_actions) / float(max(len(seen_external_actions), 1)), 4),
        "action6_coordinates_tried": [
            {"x": x, "y": y}
            for x, y in sorted(action6_coordinates)
        ],
        "action6_coordinate_count": len(action6_coordinates),
        "no_op_rate": round(no_op_count / float(total_ticks), 4),
        "wait_action_rate": round(wait_count / float(total_ticks), 4),
        "internal_action_rate": round(internal_count / float(total_ticks), 4),
        "wait_internal_action_rate": round(wait_internal_count / float(total_ticks), 4),
        "schema_failure_count": int(schema_failure_count),
        "action6_schema_failure_count": int(action6_schema_failure_count),
        "exploratory_no_effect_count": int(exploratory_no_effect_count),
        "blocked_action_count": int(blocked_action_count),
        "first_state_change_tick": first_state_change_tick,
        "first_reward_tick": first_reward_tick,
        "repeated_effective_action_count": int(sum(max(0, count - 1) for count in repeated_effective_counts.values())),
        "repeated_effective_actions": [
            {"action": action_id, "count": count}
            for action_id, count in sorted(repeated_effective_counts.items())
            if count > 1
        ],
    }

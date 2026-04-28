from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from core.objects import OBJECT_TYPE_HYPOTHESIS
from core.orchestration.action_utils import extract_action_function_name, extract_action_xy
from modules.llm.capabilities import ANALYSIS_SHADOW_REVIEW, ANALYSIS_VERIFICATION_REVIEW
from modules.llm.gateway import ensure_llm_gateway
from modules.world_model.object_binding import build_object_bindings
from modules.world_model.task_frame import infer_task_frame, validate_goal_proposal_candidates


_FAILURE_CATEGORY_ALLOWLIST = (
    "wrong_function_availability",
    "hidden_state_uncertainty",
    "goal_misidentification",
    "insufficient_probe",
    "over_probe",
    "constraint_conflict",
    "surface_mismatch",
    "unknown",
)
_MECHANISM_MARKERS = {
    "because",
    "cause",
    "causes",
    "constraint",
    "counterfactual",
    "evidence",
    "goal",
    "hidden",
    "mechanism",
    "predict",
    "progress",
    "state",
    "stabilize",
    "support",
    "transition",
    "trigger",
    "verify",
}


def _env_flag_enabled(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, *, limit: int = 320) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _text_tokens(value: Any) -> Set[str]:
    text = str(value or "").lower()
    return {token for token in re.findall(r"[a-z0-9_]{3,}", text)}


def _function_refs_from_text(value: Any) -> List[str]:
    refs: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(value or "")):
        lowered = token.strip()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        refs.append(lowered)
    return refs


def _safe_json_loads(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _is_grid(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(row, list) for row in value)
    )


def _grid_signature(grid: Any) -> Dict[str, Any]:
    if not _is_grid(grid):
        return {}
    rows = len(grid)
    cols = max((len(row) for row in grid if isinstance(row, list)), default=0)
    colors: List[Any] = []
    seen = set()
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            if cell in seen:
                continue
            seen.add(cell)
            colors.append(cell)
            if len(colors) >= 8:
                break
        if len(colors) >= 8:
            break
    return {
        "shape": [rows, cols],
        "colors": colors,
    }


def _compact_grid_examples(rows: Sequence[Any], *, limit: int) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in list(rows or [])[:limit]:
        if isinstance(row, dict):
            entry: Dict[str, Any] = {}
            if _is_grid(row.get("input")):
                entry["input"] = _grid_signature(row.get("input"))
            if _is_grid(row.get("output")):
                entry["output"] = _grid_signature(row.get("output"))
            if entry:
                compact.append(entry)
                continue
        if _is_grid(row):
            compact.append(_grid_signature(row))
    return compact


def _compact_query_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("task_id", "attempt_index", "expected_test_count", "submission_format"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    train = payload.get("train")
    if isinstance(train, list):
        compact["train_count"] = len(train)
        compact["train_examples"] = _compact_grid_examples(train, limit=3)
    test_inputs = payload.get("test_inputs")
    if isinstance(test_inputs, list):
        compact["test_count"] = len(test_inputs)
        compact["test_inputs"] = _compact_grid_examples(test_inputs, limit=2)
    return compact


def _compact_query_value(query: Any) -> Any:
    if isinstance(query, dict):
        return _compact_query_payload(query)
    text = str(query or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _safe_text(text, limit=220)
    if isinstance(payload, dict):
        return _compact_query_payload(payload)
    return _safe_text(text, limit=220)


def _compact_observation(obs_before: Dict[str, Any]) -> Dict[str, Any]:
    world_state = obs_before.get("world_state", {}) if isinstance(obs_before.get("world_state", {}), dict) else {}
    perception = obs_before.get("perception", {}) if isinstance(obs_before.get("perception", {}), dict) else {}
    novel_api = obs_before.get("novel_api", {}) if isinstance(obs_before.get("novel_api", {}), dict) else {}
    compact_perception = {}
    for key in ("goal", "coordinate_type", "text", "summary"):
        value = perception.get(key)
        if value:
            compact_perception[key] = _safe_text(value, limit=240)
    compact: Dict[str, Any] = {
        "instruction": _safe_text(obs_before.get("instruction"), limit=220),
        "perception": compact_perception,
        "world_state": {
            "state": _safe_text(world_state.get("state"), limit=80),
            "task_family": _safe_text(world_state.get("task_family"), limit=80),
            "task_id": _safe_text(world_state.get("task_id"), limit=120),
            "active_functions": list(world_state.get("active_functions", []) or [])[:12]
            if isinstance(world_state.get("active_functions", []), list)
            else [],
        },
        "visible_functions": list(novel_api.get("visible_functions", []) or [])[:12]
        if isinstance(novel_api.get("visible_functions", []), list)
        else [],
        "discovered_functions": list(novel_api.get("discovered_functions", []) or [])[:12]
        if isinstance(novel_api.get("discovered_functions", []), list)
        else [],
    }
    query_summary = _compact_query_value(obs_before.get("query"))
    if query_summary not in ("", {}, []):
        compact["query_summary"] = query_summary
    return compact


def _compact_initial_goal_object_summary(loop: Any, obs_before: Dict[str, Any]) -> Dict[str, Any]:
    descriptors = list(loop._surface_object_descriptors_from_obs(obs_before if isinstance(obs_before, dict) else {}) or [])
    top_objects = [_trim_object_descriptor(row) for row in descriptors[:5] if isinstance(row, dict)]
    color_groups: List[Dict[str, Any]] = []
    color_seen = set()
    shape_groups: List[Dict[str, Any]] = []
    shape_seen = set()
    pair_candidates: List[Dict[str, Any]] = []
    include_pair_candidates = not _env_flag_enabled("AGI_WORLD_V2_ABLATE_INITIAL_GOAL_PAIR_CANDIDATES")
    include_controller_candidates = not _env_flag_enabled("AGI_WORLD_V2_ABLATE_INITIAL_GOAL_CONTROLLER_CANDIDATES")
    include_alignment_cues = not _env_flag_enabled("AGI_WORLD_V2_ABLATE_INITIAL_GOAL_ALIGNMENT_CUES")

    def _member_stats(members: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        xs = []
        ys = []
        anchors: List[str] = []
        boundary_contacts = 0
        for row in members:
            if not isinstance(row, dict):
                continue
            centroid = row.get("centroid", {}) if isinstance(row.get("centroid", {}), dict) else {}
            xs.append(float(centroid.get("x", 0.0) or 0.0))
            ys.append(float(centroid.get("y", 0.0) or 0.0))
            anchor_ref = _safe_text(row.get("anchor_ref", ""), limit=120)
            if anchor_ref:
                anchors.append(anchor_ref)
            if bool(row.get("boundary_contact", False)):
                boundary_contacts += 1
        x_spread = round((max(xs) - min(xs)) if xs else 0.0, 4)
        y_spread = round((max(ys) - min(ys)) if ys else 0.0, 4)
        aligned_axis = "row" if ys and x_spread > y_spread else "col" if xs and y_spread > x_spread else "mixed"
        alignment_error = round(min(x_spread, y_spread), 4)
        separation = round(max(x_spread, y_spread), 4)
        stats = {
            "anchor_refs": anchors[:4],
            "x_spread": x_spread,
            "y_spread": y_spread,
            "alignment_error": alignment_error,
            "separation": separation,
            "aligned_axis_hint": aligned_axis,
            "boundary_contact_count": boundary_contacts,
        }
        if not include_alignment_cues:
            stats.pop("alignment_error", None)
            stats.pop("separation", None)
            stats.pop("aligned_axis_hint", None)
        return stats

    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        color = descriptor.get("color")
        if color not in color_seen:
            color_seen.add(color)
            members = [
                row for row in descriptors
                if isinstance(row, dict) and row.get("color") == color
            ]
            stats = _member_stats(members)
            color_groups.append(
                {
                    "color": color,
                    "count": len(members),
                    **stats,
                }
            )
            if include_pair_candidates and len(members) == 2:
                pair_candidates.append(
                    {
                        "grouping_basis": "color",
                        "grouping_value": color,
                        **stats,
                    }
                )
        labels = [
            str(label)
            for label in list(descriptor.get("shape_labels", []) or [])
            if str(label)
        ]
        if labels:
            shape = labels[0]
            if shape not in shape_seen:
                shape_seen.add(shape)
                members = [
                    row for row in descriptors
                    if isinstance(row, dict) and shape in {
                        str(label) for label in list(row.get("shape_labels", []) or []) if str(label)
                    }
                ]
                stats = _member_stats(members)
                shape_groups.append(
                    {
                        "shape": shape,
                        "count": len(members),
                        **stats,
                    }
                )

    color_groups.sort(key=lambda row: (-int(row.get("count", 0) or 0), float(row.get("alignment_error", 999.0) or 999.0), str(row.get("color", ""))))
    shape_groups.sort(key=lambda row: (-int(row.get("count", 0) or 0), float(row.get("alignment_error", 999.0) or 999.0), str(row.get("shape", ""))))
    pair_candidates.sort(key=lambda row: (float(row.get("alignment_error", 999.0) or 999.0), -float(row.get("separation", 0.0) or 0.0), str(row.get("grouping_value", ""))))

    controller_candidates: List[Dict[str, Any]] = []
    if include_controller_candidates:
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            bbox = descriptor.get("bbox", {}) if isinstance(descriptor.get("bbox", {}), dict) else {}
            width = int(bbox.get("width", 0) or 0)
            height = int(bbox.get("height", 0) or 0)
            area = max(1, width * height)
            rarity_like = (1.0 - min(1.0, area / 25.0))
            controller_score = (
                rarity_like * 0.48
                + min(1.0, float(descriptor.get("actionable_score", 0.0) or 0.0)) * 0.32
                + (0.12 if not bool(descriptor.get("boundary_contact", False)) else 0.0)
            )
            controller_candidates.append(
                {
                    "anchor_ref": _safe_text(descriptor.get("anchor_ref", ""), limit=120),
                    "color": descriptor.get("color"),
                    "shape_labels": [str(label) for label in list(descriptor.get("shape_labels", []) or [])[:3] if str(label)],
                    "controller_likelihood": round(min(1.0, controller_score), 4),
                }
            )
        controller_candidates.sort(key=lambda row: (-float(row.get("controller_likelihood", 0.0) or 0.0), str(row.get("anchor_ref", "") or "")))
    return {
        "top_objects": top_objects[:5],
        "color_groups": color_groups[:4],
        "shape_groups": shape_groups[:4],
        "pair_candidates": pair_candidates[:4],
        "controller_candidates": controller_candidates[:4],
    }


def _system_top_hypothesis(loop: Any, planner_output: Any) -> Dict[str, Any]:
    deliberation = (
        planner_output.deliberation_result
        if isinstance(getattr(planner_output, "deliberation_result", {}), dict)
        else {}
    )
    ranked = list(deliberation.get("ranked_candidate_hypothesis_objects", []) or []) if isinstance(deliberation, dict) else []
    if not ranked:
        ranked = list(deliberation.get("ranked_candidate_hypotheses", []) or []) if isinstance(deliberation, dict) else []
    if ranked and isinstance(ranked[0], dict):
        row = ranked[0]
        claim = str(row.get("summary", row.get("claim", row.get("hypothesis_id", ""))) or "")
        return {
            "claim": claim,
            "hypothesis_type": str(row.get("hypothesis_type", row.get("type", "")) or ""),
            "confidence": _clamp01(row.get("confidence", row.get("score", 0.0)), 0.0),
            "target_functions": _function_refs_from_text(claim),
        }
    tracker = getattr(loop, "_hypotheses", None)
    if tracker is None or not hasattr(tracker, "get_active"):
        return {}
    active = list(tracker.get_active() or [])
    if not active:
        return {}
    active.sort(key=lambda hyp: float(getattr(hyp, "confidence", 0.0) or 0.0), reverse=True)
    top = active[0]
    claim = str(getattr(top, "claim", "") or "")
    return {
        "claim": claim,
        "hypothesis_type": str(getattr(top, "type", "") or ""),
        "confidence": _clamp01(getattr(top, "confidence", 0.0), 0.0),
        "target_functions": _function_refs_from_text(claim),
    }


def _system_top_discriminating_action(planner_output: Any) -> Dict[str, Any]:
    deliberation = (
        planner_output.deliberation_result
        if isinstance(getattr(planner_output, "deliberation_result", {}), dict)
        else {}
    )
    ranked = list(deliberation.get("ranked_candidate_tests", []) or []) if isinstance(deliberation, dict) else []
    if not ranked or not isinstance(ranked[0], dict):
        return {}
    row = ranked[0]
    return {
        "function_name": str(row.get("function_name", row.get("target_function", "")) or ""),
        "reason": str(row.get("reason", row.get("summary", "")) or ""),
        "score": _clamp01(row.get("score", row.get("confidence", 0.0)), 0.0),
    }


def _system_selected_action(loop: Any, action_to_use: Dict[str, Any]) -> Dict[str, Any]:
    payload = action_to_use.get("payload", {}) if isinstance(action_to_use, dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
    kwargs = tool_args.get("kwargs", {}) if isinstance(tool_args, dict) else {}
    return {
        "function_name": str(extract_action_function_name(action_to_use, default="wait") or "wait"),
        "kwargs": dict(kwargs) if isinstance(kwargs, dict) else {},
    }


def _recent_failure_context(loop: Any) -> Dict[str, Any]:
    trace = list(getattr(loop, "_episode_trace", []) or [])
    if not trace:
        return {}
    entry = trace[-1] if isinstance(trace[-1], dict) else {}
    if not entry:
        return {}
    outcome = entry.get("outcome", {}) if isinstance(entry.get("outcome", {}), dict) else {}
    task_progress = entry.get("task_progress", {}) if isinstance(entry.get("task_progress", {}), dict) else {}
    assessment = entry.get("goal_progress_assessment", {}) if isinstance(entry.get("goal_progress_assessment", {}), dict) else {}
    action = entry.get("action", {}) if isinstance(entry.get("action", {}), dict) else {}
    function_name = str(extract_action_function_name(action, default="wait") or "wait")
    reward = float(entry.get("reward", 0.0) or 0.0)
    info_gain = float(entry.get("information_gain", 0.0) or 0.0)
    error_text = _safe_text(outcome.get("error", ""), limit=200)
    categories: List[str] = []
    if error_text and any(token in error_text.lower() for token in ("available", "unknown", "expected", "invalid")):
        categories.append("wrong_function_availability")
    if bool(assessment.get("necessary_but_insufficient", False)) or bool(assessment.get("local_only_signal", False)):
        categories.append("goal_misidentification")
    if function_name.lower().startswith("probe") and reward <= 0.0 and not bool(task_progress.get("progressed", False)):
        categories.append("over_probe")
    if reward <= 0.0 and info_gain < 0.12 and not bool(task_progress.get("progressed", False)):
        categories.append("insufficient_probe")
    if not categories and reward <= 0.0:
        categories.append("unknown")
    return {
        "previous_action": function_name,
        "reward": reward,
        "information_gain": info_gain,
        "error": error_text,
        "progressed": bool(task_progress.get("progressed", False)),
        "failure_categories": categories,
    }


def _candidate_function_universe(loop: Any, obs_before: Dict[str, Any], planner_output: Any, governance_output: Any) -> List[str]:
    names: List[str] = []
    seen = set()
    for fn_name in list(loop._extract_known_functions(obs_before) or []):
        if fn_name and fn_name not in seen:
            seen.add(fn_name)
            names.append(fn_name)
    planner_visible = list(getattr(planner_output, "visible_functions", []) or [])
    planner_discovered = list(getattr(planner_output, "discovered_functions", []) or [])
    for fn_name in planner_visible + planner_discovered:
        text = str(fn_name or "").strip()
        if text and text not in seen:
            seen.add(text)
            names.append(text)
    for candidate in list(getattr(governance_output, "candidate_actions", []) or []):
        fn_name = str(extract_action_function_name(candidate, default="") or "")
        if fn_name and fn_name not in seen:
            seen.add(fn_name)
            names.append(fn_name)
    return names[:12]


def _base_shadow_context(
    *,
    obs_before: Dict[str, Any],
    failure_context: Dict[str, Any],
    candidate_functions: Sequence[str],
) -> str:
    compact = _compact_observation(obs_before)
    return (
        f"Obs:{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}\n"
        f"RecentFailure:{json.dumps(failure_context, ensure_ascii=False, separators=(',', ':'))}\n"
        f"CandidateFunctions:{json.dumps(list(candidate_functions), ensure_ascii=False, separators=(',', ':'))}"
    )


def _base_initial_goal_context(
    *,
    obs_before: Dict[str, Any],
    candidate_functions: Sequence[str],
    object_summary: Dict[str, Any],
    world_model_snapshot: Dict[str, Any],
) -> str:
    compact = _compact_observation(obs_before)
    compact_before = {
        "instruction": compact.get("instruction", ""),
        "perception": compact.get("perception", {}),
        "world_state": compact.get("world_state", {}),
        "visible_functions": compact.get("visible_functions", []),
        "discovered_functions": compact.get("discovered_functions", []),
        "query_summary": compact.get("query_summary", {}),
    }
    return (
        f"InitialObservation:{json.dumps(compact_before, ensure_ascii=False, separators=(',', ':'))}\n"
        f"ObjectGroupSummary:{json.dumps(object_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"WorldModelSnapshot:{json.dumps(world_model_snapshot, ensure_ascii=False, separators=(',', ':'))}\n"
        f"CandidateFunctions:{json.dumps(list(candidate_functions), ensure_ascii=False, separators=(',', ':'))}"
    )


def _trim_snapshot_object(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    semantic_candidates = list(row.get("semantic_candidates", []) or [])
    top_semantic = semantic_candidates[0] if semantic_candidates and isinstance(semantic_candidates[0], dict) else {}
    centroid = row.get("centroid", {}) if isinstance(row.get("centroid", {}), dict) else {}
    return {
        "anchor_ref": _safe_text(row.get("object_id", ""), limit=120),
        "color": row.get("color"),
        "semantic_label": _safe_text(top_semantic.get("label", ""), limit=80),
        "center": [
            round(float(centroid.get("x", 0.0) or 0.0), 2),
            round(float(centroid.get("y", 0.0) or 0.0), 2),
        ],
        "actionable_score": round(float(row.get("actionable_score", 0.0) or 0.0), 4),
        "salience_score": round(float(row.get("salience_score", 0.0) or 0.0), 4),
    }


def _episode_trace_tail(loop: Any, limit: int = 8) -> List[Dict[str, Any]]:
    trace = [row for row in list(getattr(loop, "_episode_trace", []) or []) if isinstance(row, dict)]
    return trace[-max(0, int(limit)):]


def _build_world_model_snapshot(
    loop: Any,
    *,
    obs: Dict[str, Any],
    object_summary: Dict[str, Any],
    initial_goal_prior_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    object_bindings_summary = build_object_bindings(obs if isinstance(obs, dict) else {}, {})
    world_model_summary: Dict[str, Any] = {}
    prior_rows = [
        dict(row)
        for row in list(initial_goal_prior_rows or [])
        if isinstance(row, dict)
    ]
    if prior_rows:
        world_model_summary["initial_goal_priors"] = prior_rows[:4]
    task_frame_summary = infer_task_frame(
        obs if isinstance(obs, dict) else {},
        world_model_summary,
        object_bindings_summary,
        _episode_trace_tail(loop),
    )
    scene_objects = [
        _trim_snapshot_object(row)
        for row in list(object_bindings_summary.get("objects", []) or [])[:6]
        if isinstance(row, dict)
    ]
    return {
        "available_action_names": list(task_frame_summary.get("available_action_names", []) or [])[:8],
        "scene_objects": scene_objects,
        "groups": {
            "color_groups": list(object_summary.get("color_groups", []) or [])[:4],
            "shape_groups": list(object_summary.get("shape_groups", []) or [])[:4],
            "pair_candidates": list(object_summary.get("pair_candidates", []) or [])[:4],
            "controller_candidates": list(object_summary.get("controller_candidates", []) or [])[:4],
        },
        "goal_family_candidates": [
            dict(row)
            for row in list(task_frame_summary.get("goal_family_candidates", []) or [])[:4]
            if isinstance(row, dict)
        ],
        "inferred_level_goal": dict(task_frame_summary.get("inferred_level_goal", {}) or {}),
        "scene_summary": dict(task_frame_summary.get("scene_summary", {}) or {}),
        "task_frame_summary": task_frame_summary,
    }


def _build_world_model_proposal_candidates(
    loop: Any,
    *,
    payload: Dict[str, Any],
    candidate_functions: Sequence[str],
    source_kind: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    episode = int(getattr(loop, "_episode", 0) or 0)
    tick = int(getattr(loop, "_tick", 0) or 0)
    goal_rows = [
        dict(row)
        for row in list(payload.get("goal_hypotheses", []) or [])
        if isinstance(row, dict)
    ]
    discriminating_function = _safe_text(payload.get("discriminating_function", ""), limit=120)
    if discriminating_function and discriminating_function not in set(candidate_functions or []):
        discriminating_function = ""
    for row in goal_rows[:3]:
        goal_rank = max(1, int(row.get("goal_rank", 1) or 1))
        summary = _safe_text(row.get("goal_hypothesis", ""), limit=220)
        if not summary:
            continue
        target_group = _safe_text(row.get("target_group", ""), limit=180)
        target_relation = _safe_text(row.get("target_relation", ""), limit=120)
        completion_criterion = _safe_text(row.get("completion_criterion", ""), limit=220)
        confidence = _clamp01(row.get("confidence", 0.0), 0.0)
        rows.append(
            {
                "proposal_id": f"{source_kind}_goal_ep{episode}_tick{tick}_g{goal_rank}",
                "proposal_type": "goal",
                "summary": summary,
                "target_group": target_group,
                "target_relation": target_relation,
                "completion_criterion": completion_criterion,
                "supporting_function": discriminating_function,
                "confidence": confidence,
                "source_kind": source_kind,
                "source_episode": episode,
                "source_tick": tick,
            }
        )
        if target_relation:
            rows.append(
                {
                    "proposal_id": f"{source_kind}_relation_ep{episode}_tick{tick}_g{goal_rank}",
                    "proposal_type": "relation",
                    "summary": summary,
                    "target_group": target_group,
                    "target_relation": target_relation,
                    "completion_criterion": completion_criterion,
                    "supporting_function": discriminating_function,
                    "confidence": round(min(0.95, confidence * 0.92 + 0.03), 4),
                    "source_kind": source_kind,
                    "source_episode": episode,
                    "source_tick": tick,
                }
            )
    top_hypothesis = _safe_text(payload.get("top_hypothesis", ""), limit=220)
    mechanism_guess = _safe_text(payload.get("mechanism_guess", ""), limit=220)
    discriminating_action = _safe_text(
        ((payload.get("discriminating_action", {}) or {}).get("function_name", "")),
        limit=120,
    )
    if discriminating_action and discriminating_action not in set(candidate_functions or []):
        discriminating_action = ""
    if top_hypothesis or mechanism_guess:
        rows.append(
            {
                "proposal_id": f"{source_kind}_analysis_ep{episode}_tick{tick}",
                "proposal_type": "analysis",
                "summary": top_hypothesis or mechanism_guess,
                "target_group": "",
                "target_relation": "",
                "completion_criterion": _safe_text(payload.get("expected_visual_change", ""), limit=220),
                "supporting_function": discriminating_action,
                "confidence": _clamp01(payload.get("confidence", 0.0), 0.0),
                "source_kind": source_kind,
                "source_episode": episode,
                "source_tick": tick,
            }
        )
    return rows[:8]


def _feedback_summary_rows(feedback_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for row in list(feedback_rows or [])[:4]:
        if not isinstance(row, dict):
            continue
        summary.append(
            {
                "proposal_id": _safe_text(row.get("proposal_id", ""), limit=120),
                "proposal_type": _safe_text(row.get("proposal_type", ""), limit=80),
                "decision": _safe_text(row.get("decision", ""), limit=80),
                "wm_consistency_score": round(float(row.get("wm_consistency_score", 0.0) or 0.0), 4),
                "matched_relation": _safe_text(row.get("matched_relation", ""), limit=80),
                "predicted_goal_proximity_delta": round(float(row.get("predicted_goal_proximity_delta", 0.0) or 0.0), 4),
            }
        )
    return summary


def _current_world_model_feedback_summary(loop: Any) -> Dict[str, Any]:
    feedback_rows = [
        dict(row)
        for row in list(getattr(loop, "_llm_world_model_validation_feedback", []) or [])
        if isinstance(row, dict)
    ]
    if not feedback_rows:
        return {}
    accepted = [row for row in feedback_rows if str(row.get("decision", "") or "") == "accept_transient"]
    rejected = [row for row in feedback_rows if str(row.get("decision", "") or "") != "accept_transient"]
    return {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": _feedback_summary_rows(accepted),
        "rejected": _feedback_summary_rows(rejected),
    }


def _shadow_observation_signature(obs_before: Dict[str, Any], candidate_functions: Sequence[str]) -> str:
    compact = _compact_observation(obs_before)
    signature_payload = {
        "world_state": compact.get("world_state", {}),
        "perception": compact.get("perception", {}),
        "visible_functions": compact.get("visible_functions", []),
        "discovered_functions": compact.get("discovered_functions", []),
        "candidate_functions": list(candidate_functions or []),
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _shadow_failure_signature(failure_context: Dict[str, Any]) -> str:
    if not isinstance(failure_context, dict) or not failure_context:
        return ""
    payload = {
        "previous_action": str(failure_context.get("previous_action", "") or ""),
        "failure_categories": list(failure_context.get("failure_categories", []) or []),
        "error": str(failure_context.get("error", "") or ""),
        "progressed": bool(failure_context.get("progressed", False)),
    }
    if not payload["previous_action"] and not payload["failure_categories"] and not payload["error"]:
        return ""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _should_run_llm_shadow(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    candidate_functions: Sequence[str],
    failure_context: Dict[str, Any],
) -> Dict[str, Any]:
    observation_signature = _shadow_observation_signature(obs_before, candidate_functions)
    failure_signature = _shadow_failure_signature(failure_context)
    previous_observation_signature = str(getattr(loop, "_llm_shadow_last_observation_signature", "") or "")
    previous_failure_signature = str(getattr(loop, "_llm_shadow_last_failure_signature", "") or "")
    reasons: List[str] = []
    if int(getattr(loop, "_tick", 0) or 0) == 0:
        reasons.append("tick0")
    if previous_observation_signature and observation_signature != previous_observation_signature:
        reasons.append("observation_shift")
    if failure_signature and failure_signature != previous_failure_signature:
        reasons.append("new_failure_pattern")
    return {
        "should_run": bool(reasons),
        "reasons": reasons,
        "observation_signature": observation_signature,
        "failure_signature": failure_signature,
    }


def _shadow_reasoning_profile(loop: Any, reasons: Sequence[str]) -> Dict[str, Any]:
    tick = int(getattr(loop, "_tick", 0) or 0)
    reason_set = {str(item or "") for item in list(reasons or [])}
    if "new_failure_pattern" in reason_set:
        return {"name": "high", "think": True, "token_scale": 1.35}
    if "observation_shift" in reason_set and tick >= 2:
        return {"name": "high", "think": True, "token_scale": 1.2}
    return {"name": "low", "think": False, "token_scale": 0.7}


def _record_shadow_call(loop: Any, kind: str, route_name: str = "shadow") -> None:
    if hasattr(loop, "_record_budgeted_llm_call"):
        loop._record_budgeted_llm_call(kind, route_name=route_name)
        return
    if hasattr(loop, "_llm_advice_log") and isinstance(loop._llm_advice_log, list):
        loop._llm_advice_log.append(
            {
                "episode": loop._episode,
                "tick": loop._tick,
                "kind": kind,
                "entry": "budgeted_llm_call_intent",
                "route_name": route_name,
            }
        )


def _record_shadow_skip(loop: Any, reasons: Sequence[str]) -> None:
    if hasattr(loop, "_llm_advice_log") and isinstance(loop._llm_advice_log, list):
        loop._llm_advice_log.append(
            {
                "episode": loop._episode,
                "tick": loop._tick,
                "kind": "shadow_skip",
                "reasons": list(reasons or []),
            }
        )


def _shadow_raw_text(
    client: Any,
    prompt: str,
    *,
    capability: Any = ANALYSIS_SHADOW_REVIEW,
    max_tokens: int = 220,
    think: Optional[bool] = None,
    system_prompt: Optional[str] = None,
) -> str:
    gateway = ensure_llm_gateway(
        client,
        route_name=getattr(capability, "route_name", "shadow"),
        capability_prefix="analysis",
    )
    if gateway is None:
        return ""
    try:
        return str(
            gateway.request_raw(
                capability,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                system_prompt=system_prompt or "Return one short JSON object immediately. No prose.",
                think=think,
            )
            or ""
        )
    except Exception:
        return ""


def _shadow_complete_json(
    client: Any,
    prompt: str,
    *,
    capability: Any = ANALYSIS_SHADOW_REVIEW,
    max_tokens: int = 220,
    think: Optional[bool] = None,
) -> Dict[str, Any]:
    gateway = ensure_llm_gateway(
        client,
        route_name=getattr(capability, "route_name", "shadow"),
        capability_prefix="analysis",
    )
    if gateway is None:
        return {}
    for attempt_prompt in (
        prompt,
        prompt + "\nIf unsure, still guess. Start with { and end with }. Never return empty objects.",
    ):
        try:
            payload = gateway.request_json(
                capability,
                attempt_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                think=think,
            )
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _extract_think_body(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", str(text or ""), flags=re.DOTALL)
    if match:
        return _safe_text(match.group(1), limit=500)
    return _safe_text(text, limit=500)


def _strip_think_sections(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    return cleaned.replace("</think>", "").strip()


def _shadow_visible_response_text(text: str) -> str:
    visible = _strip_think_sections(text)
    if visible:
        return visible
    return _extract_think_body(text)


def _parse_fixed_field_lines(text: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = re.sub(r"[^A-Z0-9]+", "_", str(key or "").strip().upper()).strip("_")
        normalized_value = str(value or "").strip()
        if normalized_key and normalized_value:
            payload[normalized_key] = normalized_value
    return payload


def _parse_confidence_field(value: Any, *, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return _clamp01(match.group(0), default)
    return _inferred_confidence_from_text(text, default=default)


def _parse_list_field(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "n/a", "null"}:
        return []
    rows = [
        item.strip()
        for item in re.split(r"[,\|;/]", text)
        if item.strip()
    ]
    deduped: List[str] = []
    seen = set()
    for item in rows:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _parse_hypothesis_failure_template(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    fields = _parse_fixed_field_lines(visible)
    top_hypothesis = {
        "claim": _safe_text(fields.get("HYPOTHESIS_CLAIM", ""), limit=320),
        "hypothesis_type": _safe_text(fields.get("HYPOTHESIS_TYPE", ""), limit=80),
        "confidence": _parse_confidence_field(fields.get("HYPOTHESIS_CONFIDENCE"), default=0.0),
        "target_functions": [
            item for item in _parse_list_field(fields.get("HYPOTHESIS_TARGET_FUNCTIONS"))
            if item in set(candidate_functions or [])
        ],
        "mechanism_rationale": _safe_text(fields.get("HYPOTHESIS_REASON", visible), limit=360),
    }
    failure_categories = [
        item for item in _parse_list_field(fields.get("FAILURE_CATEGORIES"))
        if item in _FAILURE_CATEGORY_ALLOWLIST
    ]
    failure_explanation = {
        "failure_categories": failure_categories,
        "summary": _safe_text(fields.get("FAILURE_SUMMARY", ""), limit=320),
        "confidence": _parse_confidence_field(fields.get("FAILURE_CONFIDENCE"), default=0.0),
        "mechanism_rationale": _safe_text(fields.get("FAILURE_REASON", visible), limit=360),
    }
    if not top_hypothesis["claim"] and not failure_explanation["summary"] and not failure_explanation["failure_categories"]:
        return {}
    return {
        "top_hypothesis": top_hypothesis,
        "failure_explanation": failure_explanation,
    }


def _parse_action_template(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    fields = _parse_fixed_field_lines(visible)
    function_name = _safe_text(fields.get("ACTION_FUNCTION", ""), limit=120)
    if function_name not in set(candidate_functions or []):
        function_name = ""
    payload = {
        "function_name": function_name,
        "confidence": _parse_confidence_field(fields.get("ACTION_CONFIDENCE"), default=0.0),
        "why_discriminating": _safe_text(fields.get("ACTION_REASON", visible), limit=320),
        "expected_observation": _safe_text(fields.get("ACTION_EXPECTED_OBSERVATION", ""), limit=220),
    }
    if not payload["function_name"] and not payload["why_discriminating"]:
        return {}
    return payload


def _inferred_confidence_from_text(text: str, *, default: float = 0.35) -> float:
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in ("must", "clearly", "definitely", "strongly")):
        return 0.8
    if any(marker in lowered for marker in ("likely", "probably")):
        return 0.65
    if any(marker in lowered for marker in ("maybe", "could", "uncertain", "not sure")):
        return 0.35
    return default


def _mentioned_candidate_functions(text: str, candidate_functions: Sequence[str]) -> List[str]:
    lowered = str(text or "").lower()
    mentioned = [
        str(fn_name or "")
        for fn_name in candidate_functions
        if str(fn_name or "").strip() and str(fn_name or "").lower() in lowered
    ]
    if mentioned:
        return mentioned
    if len(list(candidate_functions or [])) == 1:
        lone = str(list(candidate_functions or [])[0] or "").strip()
        return [lone] if lone else []
    return []


def _heuristic_hypothesis_payload(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    think = _extract_think_body(raw_text)
    lowered = think.lower()
    hypothesis_type = "unknown"
    claim = "uncertain transformation rule"
    if any(token in lowered for token in ("extract", "subgrid", "smaller grid", "region")):
        hypothesis_type = "crop"
        claim = "extract a relevant subgrid"
    elif any(token in lowered for token in ("flip", "mirror")):
        hypothesis_type = "flip"
        claim = "flip the salient pattern"
    elif "rotate" in lowered:
        hypothesis_type = "rotate"
        claim = "rotate the salient pattern"
    elif any(token in lowered for token in ("panel", "row-major", "col-major", "trace")):
        hypothesis_type = "panel_trace"
        claim = "trace a panel ordering rule"
    elif any(token in lowered for token in ("component", "object", "connected")):
        hypothesis_type = "component_layout"
        claim = "rearrange salient components"
    elif any(token in lowered for token in ("cell", "local rule", "neighbor", "per-cell")):
        hypothesis_type = "local_rule"
        claim = "apply a local cell rule"
    elif any(token in lowered for token in ("count", "frequency", "most frequent")):
        hypothesis_type = "count_based"
        claim = "count or frequency drives output"
    elif "identity" in lowered:
        hypothesis_type = "identity"
        claim = "copy the input pattern"
    return {
        "claim": claim,
        "hypothesis_type": hypothesis_type,
        "confidence": _inferred_confidence_from_text(think),
        "target_functions": _mentioned_candidate_functions(think, candidate_functions),
        "mechanism_rationale": think,
    }


def _heuristic_action_payload(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    think = _extract_think_body(raw_text)
    mentioned = _mentioned_candidate_functions(think, candidate_functions)
    function_name = mentioned[0] if mentioned else ""
    return {
        "function_name": function_name,
        "confidence": _inferred_confidence_from_text(think, default=0.5 if function_name else 0.0),
        "why_discriminating": think,
    }


def _heuristic_failure_payload(raw_text: str) -> Dict[str, Any]:
    think = _extract_think_body(raw_text)
    lowered = think.lower()
    categories = [category for category in _FAILURE_CATEGORY_ALLOWLIST if category in lowered]
    if not categories:
        if any(token in lowered for token in ("uncertain", "ambigu", "not enough information")):
            categories = ["hidden_state_uncertainty"]
        elif any(token in lowered for token in ("wrong function", "only visible function", "available function")):
            categories = ["wrong_function_availability"]
        elif any(token in lowered for token in ("probe", "need more evidence", "insufficient evidence")):
            categories = ["insufficient_probe"]
        else:
            categories = ["unknown"]
    return {
        "failure_categories": categories[:2],
        "summary": think,
        "confidence": _inferred_confidence_from_text(think),
        "mechanism_rationale": think,
    }


def _heuristic_hypothesis_failure_payload(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    return {
        "top_hypothesis": _heuristic_hypothesis_payload(raw_text, candidate_functions),
        "failure_explanation": _heuristic_failure_payload(raw_text),
    }


def _local_availability_payload(
    candidate_functions: Sequence[str],
    *,
    obs_before: Dict[str, Any],
    actual_available: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    visible = {
        str(item or "").strip()
        for item in list(((obs_before.get("novel_api", {}) or {}).get("visible_functions", [])) or [])
        if str(item or "").strip()
    } if isinstance(obs_before.get("novel_api", {}), dict) else set()
    discovered = {
        str(item or "").strip()
        for item in list(((obs_before.get("novel_api", {}) or {}).get("discovered_functions", [])) or [])
        if str(item or "").strip()
    } if isinstance(obs_before.get("novel_api", {}), dict) else set()
    rows: List[Dict[str, Any]] = []
    for fn_name in candidate_functions:
        text = str(fn_name or "").strip()
        if not text:
            continue
        available = False
        confidence = 0.4
        reason = "shadow_local_default_unavailable"
        if text in set(actual_available or []):
            available = True
            confidence = 1.0
            reason = "runtime_executable_function"
        elif text in visible:
            available = True
            confidence = 0.75
            reason = "visible_function_surface"
        elif text in discovered:
            available = True
            confidence = 0.65
            reason = "discovered_function_surface"
        rows.append(
            {
                "function_name": text,
                "available": available,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return {
        "function_availability_judgment": rows,
        "function_availability_source": "local_runtime",
    }


def _call_shadow_llm_components(
    loop: Any,
    client: Any,
    *,
    obs_before: Dict[str, Any],
    failure_context: Dict[str, Any],
    candidate_functions: Sequence[str],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    base = _base_shadow_context(
        obs_before=obs_before,
        failure_context=failure_context,
        candidate_functions=candidate_functions,
    )
    token_scale = float(profile.get("token_scale", 1.0) or 1.0)
    think = profile.get("think")
    hypothesis_failure_tokens = max(180, int(round(360 * token_scale)))
    action_tokens = max(120, int(round(220 * token_scale)))
    actual_available = set(loop._collect_executable_function_names(obs_before))

    _record_shadow_call(loop, "shadow_hypothesis_failure")
    hypothesis_failure_raw = _shadow_raw_text(
        client,
        (
            "Shadow task: infer top current hypothesis and current likely failure mode.\n"
            "Return exactly these lines and nothing else:\n"
            "HYPOTHESIS_CLAIM: <short text>\n"
            "HYPOTHESIS_TYPE: <identity|crop|flip|rotate|panel_trace|local_rule|component_layout|count_based|unknown>\n"
            "HYPOTHESIS_CONFIDENCE: <0.0-1.0>\n"
            "HYPOTHESIS_TARGET_FUNCTIONS: <comma-separated functions or none>\n"
            "FAILURE_CATEGORIES: <comma-separated categories>\n"
            "FAILURE_SUMMARY: <short text>\n"
            "FAILURE_CONFIDENCE: <0.0-1.0>\n"
            f"Allowed failure_categories: {list(_FAILURE_CATEGORY_ALLOWLIST)}.\n"
            "Use a short transformation-family hypothesis such as identity,crop,flip,rotate,panel_trace,local_rule,component_layout,count_based,unknown.\n"
            + base
        ),
        capability=ANALYSIS_SHADOW_REVIEW,
        max_tokens=hypothesis_failure_tokens,
        think=think,
        system_prompt="Return only the requested fixed-field lines. No JSON. No bullets. No prose outside the fields.",
    )
    hypothesis_failure = _parse_hypothesis_failure_template(hypothesis_failure_raw, candidate_functions)
    if not hypothesis_failure:
        hypothesis_failure = _heuristic_hypothesis_failure_payload(
            hypothesis_failure_raw,
            candidate_functions,
        )
    top_hypothesis = hypothesis_failure.get("top_hypothesis", {}) if isinstance(hypothesis_failure.get("top_hypothesis", {}), dict) else {}
    failure_explanation = hypothesis_failure.get("failure_explanation", {}) if isinstance(hypothesis_failure.get("failure_explanation", {}), dict) else {}

    _record_shadow_call(loop, "shadow_discriminating_action")
    discriminating_action_raw = _shadow_raw_text(
        client,
        (
            "Shadow task: name the most discriminating next action.\n"
            "Return exactly these lines and nothing else:\n"
            "ACTION_FUNCTION: <one function from CandidateFunctions or none>\n"
            "ACTION_CONFIDENCE: <0.0-1.0>\n"
            "ACTION_REASON: <short text>\n"
            "ACTION_EXPECTED_OBSERVATION: <short text or none>\n"
            "Choose one function from CandidateFunctions.\n"
            + base
        ),
        capability=ANALYSIS_SHADOW_REVIEW,
        max_tokens=action_tokens,
        think=think,
        system_prompt="Return only the requested fixed-field lines. No JSON. No bullets. No prose outside the fields.",
    )
    discriminating_action = _parse_action_template(discriminating_action_raw, candidate_functions)
    if not discriminating_action:
        discriminating_action = _heuristic_action_payload(
            discriminating_action_raw,
            candidate_functions,
        )

    availability = _local_availability_payload(
        candidate_functions,
        obs_before=obs_before,
        actual_available=actual_available,
    )
    if hasattr(loop, "_llm_advice_log") and isinstance(loop._llm_advice_log, list):
        loop._llm_advice_log.append(
            {
                "episode": loop._episode,
                "tick": loop._tick,
                "kind": "shadow_function_availability_local",
                "source": "local_runtime",
                "available_functions": sorted(actual_available),
            }
        )

    return {
        "top_hypothesis": top_hypothesis,
        "discriminating_action": discriminating_action,
        "failure_explanation": failure_explanation,
        **availability,
    }


def _normalize_shadow_payload(raw: Dict[str, Any], candidate_functions: Sequence[str]) -> Dict[str, Any]:
    top = raw.get("top_hypothesis", {}) if isinstance(raw.get("top_hypothesis", {}), dict) else {}
    action = raw.get("discriminating_action", {}) if isinstance(raw.get("discriminating_action", {}), dict) else {}
    failure = raw.get("failure_explanation", {}) if isinstance(raw.get("failure_explanation", {}), dict) else {}
    availability_rows = list(raw.get("function_availability_judgment", []) or [])

    normalized_rows: List[Dict[str, Any]] = []
    seen = set()
    for row in availability_rows:
        if not isinstance(row, dict):
            continue
        function_name = str(row.get("function_name", "") or "").strip()
        if not function_name or function_name in seen:
            continue
        seen.add(function_name)
        normalized_rows.append({
            "function_name": function_name,
            "available": bool(row.get("available", False)),
            "confidence": _clamp01(row.get("confidence", 0.0), 0.0),
            "reason": _safe_text(row.get("reason", ""), limit=220),
            "surface_cue_refs": [str(item or "") for item in list(row.get("surface_cue_refs", []) or [])[:6]],
        })
    for function_name in candidate_functions:
        text = str(function_name or "").strip()
        if not text or text in seen:
            continue
        normalized_rows.append({
            "function_name": text,
            "available": False,
            "confidence": 0.0,
            "reason": "",
            "surface_cue_refs": [],
        })
        seen.add(text)

    failure_categories = [
        str(item or "").strip()
        for item in list(failure.get("failure_categories", []) or [])
        if str(item or "").strip() in _FAILURE_CATEGORY_ALLOWLIST
    ]
    return {
        "top_hypothesis": {
            "claim": _safe_text(top.get("claim", ""), limit=320),
            "hypothesis_type": _safe_text(top.get("hypothesis_type", ""), limit=80),
            "confidence": _clamp01(top.get("confidence", 0.0), 0.0),
            "target_functions": [str(item or "") for item in list(top.get("target_functions", []) or [])[:6] if str(item or "").strip()],
            "surface_cue_refs": [str(item or "") for item in list(top.get("surface_cue_refs", []) or [])[:8] if str(item or "").strip()],
            "mechanism_rationale": _safe_text(top.get("mechanism_rationale", ""), limit=360),
        },
        "discriminating_action": {
            "function_name": _safe_text(action.get("function_name", ""), limit=120),
            "kwargs": dict(action.get("kwargs", {})) if isinstance(action.get("kwargs", {}), dict) else {},
            "confidence": _clamp01(action.get("confidence", 0.0), 0.0),
            "why_discriminating": _safe_text(action.get("why_discriminating", ""), limit=320),
            "expected_observation": _safe_text(action.get("expected_observation", ""), limit=220),
            "surface_cue_refs": [str(item or "") for item in list(action.get("surface_cue_refs", []) or [])[:8] if str(item or "").strip()],
        },
        "failure_explanation": {
            "summary": _safe_text(failure.get("summary", ""), limit=320),
            "failure_categories": failure_categories,
            "confidence": _clamp01(failure.get("confidence", 0.0), 0.0),
            "surface_cue_refs": [str(item or "") for item in list(failure.get("surface_cue_refs", []) or [])[:8] if str(item or "").strip()],
            "mechanism_rationale": _safe_text(failure.get("mechanism_rationale", ""), limit=360),
        },
        "function_availability_source": _safe_text(raw.get("function_availability_source", "llm_shadow"), limit=80) or "llm_shadow",
        "function_availability_judgment": normalized_rows,
    }


def _surface_parroting_analysis(payload: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
    surface_text = " ".join(
        [
            str(obs_before.get("instruction", "") or ""),
            str((obs_before.get("perception", {}) or {}).get("goal", "") if isinstance(obs_before.get("perception", {}), dict) else ""),
            " ".join(list((obs_before.get("novel_api", {}) or {}).get("visible_functions", []) or []))
            if isinstance(obs_before.get("novel_api", {}), dict)
            else "",
            " ".join(list((obs_before.get("novel_api", {}) or {}).get("discovered_functions", []) or []))
            if isinstance(obs_before.get("novel_api", {}), dict)
            else "",
        ]
    )
    surface_tokens = _text_tokens(surface_text)
    reasoning_text = " ".join(
        [
            str(payload.get("top_hypothesis", {}).get("claim", "") or ""),
            str(payload.get("top_hypothesis", {}).get("mechanism_rationale", "") or ""),
            str(payload.get("discriminating_action", {}).get("why_discriminating", "") or ""),
            str(payload.get("failure_explanation", {}).get("summary", "") or ""),
            str(payload.get("failure_explanation", {}).get("mechanism_rationale", "") or ""),
        ]
    )
    reasoning_tokens = _text_tokens(reasoning_text)
    overlap = len(surface_tokens & reasoning_tokens)
    overlap_ratio = (overlap / max(1, len(reasoning_tokens))) if reasoning_tokens else 0.0
    mechanism_marker_count = sum(1 for token in reasoning_tokens if token in _MECHANISM_MARKERS)
    cue_refs = (
        len(list(payload.get("top_hypothesis", {}).get("surface_cue_refs", []) or []))
        + len(list(payload.get("discriminating_action", {}).get("surface_cue_refs", []) or []))
        + len(list(payload.get("failure_explanation", {}).get("surface_cue_refs", []) or []))
    )
    risk = "low"
    if cue_refs >= 2 and overlap_ratio >= 0.55 and mechanism_marker_count == 0:
        risk = "high"
    elif cue_refs >= 1 and overlap_ratio >= 0.40 and mechanism_marker_count <= 1:
        risk = "medium"
    return {
        "surface_token_overlap_ratio": round(overlap_ratio, 4),
        "mechanism_marker_count": mechanism_marker_count,
        "cue_ref_count": cue_refs,
        "risk": risk,
    }


def _hypothesis_agreement(llm_hypothesis: Dict[str, Any], system_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    llm_targets = {str(item or "").strip() for item in list(llm_hypothesis.get("target_functions", []) or []) if str(item or "").strip()}
    system_targets = {str(item or "").strip() for item in list(system_hypothesis.get("target_functions", []) or []) if str(item or "").strip()}
    exact_target_match = bool(llm_targets and system_targets and llm_targets == system_targets)
    target_overlap = len(llm_targets & system_targets) / max(1, len(llm_targets | system_targets)) if (llm_targets or system_targets) else 0.0
    claim_overlap = len(_text_tokens(llm_hypothesis.get("claim", "")) & _text_tokens(system_hypothesis.get("claim", ""))) / max(
        1,
        len(_text_tokens(llm_hypothesis.get("claim", "")) | _text_tokens(system_hypothesis.get("claim", ""))),
    ) if llm_hypothesis.get("claim") or system_hypothesis.get("claim") else 0.0
    return {
        "exact_target_match": exact_target_match,
        "target_overlap": round(target_overlap, 4),
        "claim_overlap": round(claim_overlap, 4),
        "agrees": bool(exact_target_match or target_overlap >= 0.5 or claim_overlap >= 0.45),
    }


def _action_agreement(llm_action: Dict[str, Any], system_action: Dict[str, Any], system_test_action: Dict[str, Any]) -> Dict[str, Any]:
    llm_fn = str(llm_action.get("function_name", "") or "")
    selected_fn = str(system_action.get("function_name", "") or "")
    test_fn = str(system_test_action.get("function_name", "") or "")
    kwargs_match = (
        bool(llm_fn and selected_fn and llm_fn == selected_fn)
        and dict(llm_action.get("kwargs", {}) or {}) == dict(system_action.get("kwargs", {}) or {})
    )
    return {
        "matches_selected_action": bool(llm_fn and llm_fn == selected_fn),
        "matches_selected_kwargs": bool(kwargs_match),
        "matches_system_discriminating_action": bool(llm_fn and test_fn and llm_fn == test_fn),
        "agrees": bool(llm_fn and (llm_fn == selected_fn or (test_fn and llm_fn == test_fn))),
    }


def _availability_agreement(rows: Sequence[Dict[str, Any]], actual_available: Set[str]) -> Dict[str, Any]:
    judged = 0
    correct = 0
    false_positive = 0
    false_negative = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        fn_name = str(row.get("function_name", "") or "").strip()
        if not fn_name:
            continue
        predicted = bool(row.get("available", False))
        actual = fn_name in actual_available
        judged += 1
        if predicted == actual:
            correct += 1
        elif predicted and not actual:
            false_positive += 1
        elif (not predicted) and actual:
            false_negative += 1
    accuracy = correct / max(1, judged)
    return {
        "judged_count": judged,
        "accuracy": round(accuracy, 4),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "agrees": bool(judged > 0 and accuracy >= 0.75),
    }


def _failure_explanation_agreement(llm_failure: Dict[str, Any], system_failure: Dict[str, Any]) -> Dict[str, Any]:
    llm_categories = {str(item or "").strip() for item in list(llm_failure.get("failure_categories", []) or []) if str(item or "").strip()}
    system_categories = {str(item or "").strip() for item in list(system_failure.get("failure_categories", []) or []) if str(item or "").strip()}
    overlap = len(llm_categories & system_categories) / max(1, len(llm_categories | system_categories)) if (llm_categories or system_categories) else 0.0
    return {
        "category_overlap": round(overlap, 4),
        "agrees": bool(overlap >= 0.5) if (llm_categories or system_categories) else False,
    }


def _case_strength_tags(
    *,
    llm_payload: Dict[str, Any],
    system_action: Dict[str, Any],
    action_agreement: Dict[str, Any],
    availability_agreement: Dict[str, Any],
    availability_source: str,
    result: Optional[Dict[str, Any]],
    reward: Optional[float],
    surface_parroting: Dict[str, Any],
    actual_available: Set[str],
) -> Dict[str, Any]:
    llm_tags: List[str] = []
    system_tags: List[str] = []
    selected_fn = str(system_action.get("function_name", "") or "")
    llm_fn = str(llm_payload.get("discriminating_action", {}).get("function_name", "") or "")
    selected_available = selected_fn in actual_available if selected_fn else True
    llm_available = llm_fn in actual_available if llm_fn else False
    success = bool((result or {}).get("success", False))
    positive = bool(success or float(reward or 0.0) > 0.0)

    if not positive and llm_fn and llm_fn != selected_fn and llm_available and not selected_available:
        llm_tags.append("available_alt_when_system_selected_unavailable")
    if not positive and llm_fn and llm_fn != selected_fn and llm_available and surface_parroting.get("risk") != "high":
        llm_tags.append("available_alt_when_system_failed")
    if (
        availability_source == "llm_shadow"
        and not positive
        and availability_agreement.get("accuracy", 0.0) >= 0.75
    ):
        llm_tags.append("availability_judgment_held_up_under_failure")

    if positive and action_agreement.get("matches_selected_action", False):
        system_tags.append("system_action_succeeded")
    if positive and llm_fn and llm_fn != selected_fn:
        system_tags.append("system_outperformed_shadow_action")
    if surface_parroting.get("risk") == "high":
        system_tags.append("llm_surface_cue_parroting")
    if availability_source == "llm_shadow" and availability_agreement.get("accuracy", 0.0) < 0.5:
        system_tags.append("llm_availability_judgment_weak")

    return {
        "llm_stronger_reasons": llm_tags,
        "system_stronger_reasons": system_tags,
    }


def run_llm_shadow_pre_execution(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    planner_output: Any,
    governance_output: Any,
    action_to_use: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    client = loop._resolve_llm_client("shadow") if hasattr(loop, "_resolve_llm_client") else getattr(loop, "_llm_shadow_client", None)
    if client is None:
        return None
    candidate_functions = _candidate_function_universe(loop, obs_before, planner_output, governance_output)
    failure_context = _recent_failure_context(loop)
    trigger = _should_run_llm_shadow(
        loop,
        obs_before=obs_before,
        candidate_functions=candidate_functions,
        failure_context=failure_context,
    )
    loop._llm_shadow_last_observation_signature = trigger.get("observation_signature", "")
    loop._llm_shadow_last_failure_signature = trigger.get("failure_signature", "")
    if not bool(trigger.get("should_run", False)):
        _record_shadow_skip(loop, trigger.get("reasons", []))
        return None
    profile = _shadow_reasoning_profile(loop, trigger.get("reasons", []))
    raw = _call_shadow_llm_components(
        loop,
        client,
        obs_before=obs_before,
        failure_context=failure_context,
        candidate_functions=candidate_functions,
        profile=profile,
    )
    payload = _normalize_shadow_payload(raw, candidate_functions)
    system_hypothesis = _system_top_hypothesis(loop, planner_output)
    system_action = _system_selected_action(loop, action_to_use)
    system_test_action = _system_top_discriminating_action(planner_output)
    actual_available = set(loop._collect_executable_function_names(obs_before))
    surface_parroting = _surface_parroting_analysis(payload, obs_before)
    entry = {
        "episode": loop._episode,
        "tick": loop._tick,
        "llm_mode": str(getattr(loop, "_llm_mode", "integrated") or "integrated"),
        "prompt_context": {
            "candidate_functions": list(candidate_functions),
            "recent_failure_context": failure_context,
            "trigger_reasons": list(trigger.get("reasons", []) or []),
            "reasoning_profile": dict(profile),
        },
        "shadow_output": payload,
        "system_reference": {
            "top_hypothesis": system_hypothesis,
            "selected_action": system_action,
            "top_discriminating_action": system_test_action,
            "actual_available_functions": sorted(actual_available),
        },
        "agreement": {
            "hypothesis": _hypothesis_agreement(payload.get("top_hypothesis", {}), system_hypothesis),
            "discriminating_action": _action_agreement(
                payload.get("discriminating_action", {}),
                system_action,
                system_test_action,
            ),
            "function_availability": _availability_agreement(
                payload.get("function_availability_judgment", []),
                actual_available,
            ),
            "failure_explanation": _failure_explanation_agreement(
                payload.get("failure_explanation", {}),
                failure_context,
            ),
        },
        "surface_parroting": surface_parroting,
        "outcome": {},
    }
    loop._llm_shadow_log.append(entry)
    return entry


def finalize_llm_shadow_post_execution(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    action_to_use: Dict[str, Any],
    result: Dict[str, Any],
    reward: float,
) -> None:
    if not getattr(loop, "_llm_shadow_log", None):
        return
    entry = loop._llm_shadow_log[-1]
    if not isinstance(entry, dict):
        return
    if int(entry.get("episode", -1)) != int(loop._episode) or int(entry.get("tick", -1)) != int(loop._tick):
        return
    actual_available = set(loop._collect_executable_function_names(obs_before))
    strength = _case_strength_tags(
        llm_payload=entry.get("shadow_output", {}),
        system_action=entry.get("system_reference", {}).get("selected_action", {}),
        action_agreement=entry.get("agreement", {}).get("discriminating_action", {}),
        availability_agreement=entry.get("agreement", {}).get("function_availability", {}),
        availability_source=str(entry.get("shadow_output", {}).get("function_availability_source", "") or ""),
        result=result,
        reward=reward,
        surface_parroting=entry.get("surface_parroting", {}),
        actual_available=actual_available,
    )
    entry["outcome"] = {
        "reward": float(reward or 0.0),
        "success": bool(result.get("success", False)) if isinstance(result, dict) else False,
        "terminal": bool(result.get("terminal", False) or result.get("done", False)) if isinstance(result, dict) else False,
        "selected_action_available": bool(
            str(entry.get("system_reference", {}).get("selected_action", {}).get("function_name", "") or "") in actual_available
        ) if str(entry.get("system_reference", {}).get("selected_action", {}).get("function_name", "") or "") else True,
    }
    entry["strength"] = strength


def build_llm_shadow_summary(log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in list(log or []) if isinstance(row, dict)]
    if not rows:
        return {
            "tick_count": 0,
            "hypothesis_agreement_rate": 0.0,
            "discriminating_action_agreement_rate": 0.0,
            "function_availability_accuracy": 0.0,
            "failure_explanation_agreement_rate": 0.0,
            "surface_parroting_rate": 0.0,
            "llm_stronger_cases": [],
            "system_stronger_cases": [],
        }
    hyp_agree = sum(1 for row in rows if bool(((row.get("agreement", {}) or {}).get("hypothesis", {}) or {}).get("agrees", False)))
    action_agree = sum(1 for row in rows if bool(((row.get("agreement", {}) or {}).get("discriminating_action", {}) or {}).get("agrees", False)))
    failure_agree = sum(1 for row in rows if bool(((row.get("agreement", {}) or {}).get("failure_explanation", {}) or {}).get("agrees", False)))
    availability_scores = [
        float((((row.get("agreement", {}) or {}).get("function_availability", {}) or {}).get("accuracy", 0.0) or 0.0))
        for row in rows
    ]
    availability_sources = {
        str((row.get("shadow_output", {}) or {}).get("function_availability_source", "") or "llm_shadow")
        for row in rows
    }
    parroting = sum(1 for row in rows if str(((row.get("surface_parroting", {}) or {}).get("risk", "low")) or "low") == "high")
    llm_cases = []
    system_cases = []
    for row in rows:
        strength = row.get("strength", {}) if isinstance(row.get("strength", {}), dict) else {}
        if list(strength.get("llm_stronger_reasons", []) or []):
            llm_cases.append({
                "episode": int(row.get("episode", 0) or 0),
                "tick": int(row.get("tick", 0) or 0),
                "reasons": list(strength.get("llm_stronger_reasons", []) or []),
            })
        if list(strength.get("system_stronger_reasons", []) or []):
            system_cases.append({
                "episode": int(row.get("episode", 0) or 0),
                "tick": int(row.get("tick", 0) or 0),
                "reasons": list(strength.get("system_stronger_reasons", []) or []),
            })
    return {
        "tick_count": len(rows),
        "hypothesis_agreement_rate": round(hyp_agree / len(rows), 4),
        "discriminating_action_agreement_rate": round(action_agree / len(rows), 4),
        "function_availability_mode": next(iter(availability_sources)) if len(availability_sources) == 1 else "mixed",
        "function_availability_accuracy": round(sum(availability_scores) / max(1, len(availability_scores)), 4),
        "failure_explanation_agreement_rate": round(failure_agree / len(rows), 4),
        "surface_parroting_rate": round(parroting / len(rows), 4),
        "llm_stronger_cases": llm_cases,
        "system_stronger_cases": system_cases,
    }


def _record_analyst_skip(loop: Any, reasons: Sequence[str]) -> None:
    if hasattr(loop, "_llm_advice_log") and isinstance(loop._llm_advice_log, list):
        loop._llm_advice_log.append(
            {
                "episode": loop._episode,
                "tick": loop._tick,
                "kind": "analyst_skip",
                "reasons": list(reasons or []),
            }
        )


def _post_action_candidate_universe(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    result: Dict[str, Any],
    action_to_use: Dict[str, Any],
) -> List[str]:
    names: List[str] = []
    seen = set()
    for source in (result, obs_before):
        for fn_name in list(loop._extract_known_functions(source) or []):
            text = str(fn_name or "").strip()
            if text and text not in seen:
                seen.add(text)
                names.append(text)
        novel_api = source.get("novel_api", {}) if isinstance(source.get("novel_api", {}), dict) else {}
        for fn_name in list(novel_api.get("visible_functions", []) or []) + list(novel_api.get("discovered_functions", []) or []):
            text = str(fn_name or "").strip()
            if text and text not in seen:
                seen.add(text)
                names.append(text)
    current_fn = str(extract_action_function_name(action_to_use, default="") or "").strip()
    if current_fn and current_fn not in seen:
        names.append(current_fn)
    return names[:12]


def _analyst_observation_signature(
    result: Dict[str, Any],
    candidate_functions: Sequence[str],
    visual_feedback: Dict[str, Any],
) -> str:
    compact = _compact_observation(result if isinstance(result, dict) else {})
    payload = {
        "world_state": compact.get("world_state", {}),
        "perception": compact.get("perception", {}),
        "visible_functions": compact.get("visible_functions", []),
        "discovered_functions": compact.get("discovered_functions", []),
        "candidate_functions": list(candidate_functions or []),
        "visual_feedback": {
            "changed_pixel_count": round(float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0), 4),
            "changed_ratio": round(float(visual_feedback.get("changed_ratio", 0.0) or 0.0), 4),
            "changed_bbox_ratio": round(float(visual_feedback.get("changed_bbox_ratio", 0.0) or 0.0), 4),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _should_run_llm_analyst(
    loop: Any,
    *,
    result: Dict[str, Any],
    candidate_functions: Sequence[str],
    failure_context: Dict[str, Any],
    visual_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    observation_signature = _analyst_observation_signature(result, candidate_functions, visual_feedback)
    failure_signature = _shadow_failure_signature(failure_context)
    previous_observation_signature = str(getattr(loop, "_llm_analyst_last_observation_signature", "") or "")
    previous_failure_signature = str(getattr(loop, "_llm_analyst_last_failure_signature", "") or "")
    changed_pixels = float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0)
    changed_bbox_ratio = float(visual_feedback.get("changed_bbox_ratio", 0.0) or 0.0)
    tick = int(getattr(loop, "_tick", 0) or 0)
    reasons: List[str] = []
    if tick == 0:
        reasons.append("tick0")
    if (changed_pixels > 0.0 or changed_bbox_ratio > 0.0) and observation_signature != previous_observation_signature:
        reasons.append("visual_change")
    if tick > 0 and failure_signature and failure_signature != previous_failure_signature:
        reasons.append("new_failure_pattern")
    return {
        "should_run": bool(reasons),
        "reasons": reasons,
        "observation_signature": observation_signature,
        "failure_signature": failure_signature,
    }


def _latest_trace_progress_snapshot(loop: Any) -> Dict[str, Any]:
    trace = list(getattr(loop, "_episode_trace", []) or [])
    if not trace:
        return {}
    entry = trace[-1] if isinstance(trace[-1], dict) else {}
    if not isinstance(entry, dict):
        return {}
    progress_markers = [
        marker
        for marker in list(entry.get("progress_markers", []) or [])[-4:]
        if isinstance(marker, dict)
    ]
    goal_progress = entry.get("goal_progress_assessment", {}) if isinstance(entry.get("goal_progress_assessment", {}), dict) else {}
    inferred_goal = entry.get("inferred_level_goal", {}) if isinstance(entry.get("inferred_level_goal", {}), dict) else {}
    task_progress = entry.get("task_progress", {}) if isinstance(entry.get("task_progress", {}), dict) else {}
    relation_hypotheses = [
        dict(item)
        for item in list(inferred_goal.get("relation_hypotheses", []) or [])[-2:]
        if isinstance(item, dict)
    ]
    return {
        "progress_markers": progress_markers,
        "goal_progress_assessment": {
            "progressed": bool(goal_progress.get("progressed", False)),
            "goal_family": _safe_text(goal_progress.get("goal_family", ""), limit=120),
            "progress_class": _safe_text(goal_progress.get("progress_class", ""), limit=120),
            "goal_progress_score": float(goal_progress.get("goal_progress_score", 0.0) or 0.0),
            "goal_distance_estimate": float(goal_progress.get("goal_distance_estimate", 1.0) or 1.0),
            "necessary_but_insufficient": bool(goal_progress.get("necessary_but_insufficient", False)),
            "local_only_signal": bool(goal_progress.get("local_only_signal", False)),
            "controller_anchor_ref": _safe_text(goal_progress.get("controller_anchor_ref", ""), limit=120),
            "clicked_anchor_ref": _safe_text(goal_progress.get("clicked_anchor_ref", ""), limit=120),
        },
        "task_progress": {
            "progressed": bool(task_progress.get("progressed", False)),
            "solved": bool(task_progress.get("solved", False)),
        },
        "inferred_level_goal": {
            "goal_family": _safe_text(inferred_goal.get("goal_family", ""), limit=120),
            "goal_anchor_refs": list(inferred_goal.get("goal_anchor_refs", []) or [])[:6]
            if isinstance(inferred_goal.get("goal_anchor_refs", []), list)
            else [],
            "controller_anchor_refs": list(inferred_goal.get("controller_anchor_refs", []) or [])[:6]
            if isinstance(inferred_goal.get("controller_anchor_refs", []), list)
            else [],
            "preferred_next_goal_anchor_refs": list(inferred_goal.get("preferred_next_goal_anchor_refs", []) or [])[:6]
            if isinstance(inferred_goal.get("preferred_next_goal_anchor_refs", []), list)
            else [],
            "relation_hypotheses": relation_hypotheses,
            "initial_goal_prior_relation": _safe_text(inferred_goal.get("initial_goal_prior_relation", ""), limit=80),
            "initial_goal_prior_target_group": _safe_text(inferred_goal.get("initial_goal_prior_target_group", ""), limit=160),
            "initial_goal_prior_summary": _safe_text(inferred_goal.get("initial_goal_prior_summary", ""), limit=180),
        },
    }


def _compact_world_model_goal_decision(progress_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    inferred_goal = progress_snapshot.get("inferred_level_goal", {}) if isinstance(progress_snapshot.get("inferred_level_goal", {}), dict) else {}
    relation_hypotheses = [
        dict(item)
        for item in list(inferred_goal.get("relation_hypotheses", []) or [])
        if isinstance(item, dict)
    ]
    top_relation = relation_hypotheses[0] if relation_hypotheses else {}
    return {
        "goal_family": _safe_text(inferred_goal.get("goal_family", ""), limit=120),
        "goal_anchor_refs": [
            _safe_text(item, limit=120)
            for item in list(inferred_goal.get("goal_anchor_refs", []) or [])[:4]
            if _safe_text(item, limit=120)
        ],
        "controller_anchor_refs": [
            _safe_text(item, limit=120)
            for item in list(inferred_goal.get("controller_anchor_refs", []) or [])[:4]
            if _safe_text(item, limit=120)
        ],
        "preferred_next_goal_anchor_refs": [
            _safe_text(item, limit=120)
            for item in list(inferred_goal.get("preferred_next_goal_anchor_refs", []) or [])[:4]
            if _safe_text(item, limit=120)
        ],
        "top_relation": {
            "relation_type": _safe_text(top_relation.get("relation_type", ""), limit=80),
            "target_relation": _safe_text(top_relation.get("target_relation", ""), limit=80),
            "grouping_basis": _safe_text(top_relation.get("grouping_basis", ""), limit=80),
            "grouping_value": _safe_text(top_relation.get("grouping_value", ""), limit=120),
            "member_anchor_refs": [
                _safe_text(item, limit=120)
                for item in list(top_relation.get("member_anchor_refs", []) or [])[:4]
                if _safe_text(item, limit=120)
            ],
        },
        "initial_goal_prior_relation": _safe_text(inferred_goal.get("initial_goal_prior_relation", ""), limit=80),
        "initial_goal_prior_target_group": _safe_text(inferred_goal.get("initial_goal_prior_target_group", ""), limit=160),
        "initial_goal_prior_summary": _safe_text(inferred_goal.get("initial_goal_prior_summary", ""), limit=180),
    }


def _trim_object_descriptor(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(descriptor, dict):
        return {}
    return {
        "anchor_ref": _safe_text(descriptor.get("anchor_ref", ""), limit=120),
        "color": descriptor.get("color"),
        "shape_labels": [str(label) for label in list(descriptor.get("shape_labels", []) or [])[:4] if str(label)],
        "boundary_contact": bool(descriptor.get("boundary_contact", False)),
        "salience_score": round(float(descriptor.get("salience_score", 0.0) or 0.0), 4),
        "actionable_score": round(float(descriptor.get("actionable_score", 0.0) or 0.0), 4),
    }


def _analyst_surface_object_summary(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    result: Dict[str, Any],
    action_to_use: Dict[str, Any],
) -> Dict[str, Any]:
    after_descriptors = list(loop._surface_object_descriptors_from_obs(result if isinstance(result, dict) else {}) or [])
    before_descriptors = list(loop._surface_object_descriptors_from_obs(obs_before if isinstance(obs_before, dict) else {}) or [])
    descriptors = after_descriptors or before_descriptors
    changed_bbox = (
        result.get("changed_bbox")
        if isinstance(result.get("changed_bbox"), dict)
        else ((result.get("perception", {}) or {}).get("changed_bbox"))
        if isinstance((result.get("perception", {}) or {}).get("changed_bbox"), dict)
        else None
    )
    hotspot = (
        result.get("suggested_hotspot")
        if isinstance(result.get("suggested_hotspot"), dict)
        else ((result.get("perception", {}) or {}).get("suggested_hotspot"))
        if isinstance((result.get("perception", {}) or {}).get("suggested_hotspot"), dict)
        else None
    )
    changed_rows = [
        descriptor
        for descriptor in descriptors
        if loop._descriptor_affected_by_visual_change(descriptor, changed_bbox, hotspot)
    ][:3]
    click_point = extract_action_xy(action_to_use)
    clicked_descriptor = None
    if click_point is not None:
        clicked_descriptor = loop._match_click_to_descriptor(before_descriptors or descriptors, click_point)
    clicked_anchor = str((clicked_descriptor or {}).get("anchor_ref", "") or "")
    clicked_color = (clicked_descriptor or {}).get("color")
    clicked_shapes = {
        str(label)
        for label in list((clicked_descriptor or {}).get("shape_labels", []) or [])
        if str(label)
    }
    changed_anchors = [
        str(row.get("anchor_ref", "") or "")
        for row in changed_rows
        if isinstance(row, dict) and str(row.get("anchor_ref", "") or "")
    ]
    changed_colors = sorted(
        {
            row.get("color")
            for row in changed_rows
            if isinstance(row, dict) and row.get("color") is not None
        }
    )
    changed_shape_labels = sorted(
        {
            str(label)
            for row in changed_rows
            if isinstance(row, dict)
            for label in list(row.get("shape_labels", []) or [])
            if str(label)
        }
    )
    relation_summary = {
        "same_anchor_as_clicked": bool(clicked_anchor and clicked_anchor in changed_anchors),
        "clicked_anchor_ref": clicked_anchor,
        "changed_anchor_refs": changed_anchors[:4],
        "clicked_color": clicked_color,
        "changed_colors": changed_colors[:4],
        "color_overlap_with_clicked": bool(clicked_color is not None and clicked_color in changed_colors),
        "clicked_shape_labels": sorted(clicked_shapes)[:4],
        "changed_shape_labels": changed_shape_labels[:6],
        "shape_overlap_with_clicked": bool(clicked_shapes and bool(clicked_shapes & set(changed_shape_labels))),
        "changed_object_count": len(changed_rows),
    }
    return {
        "clicked_object": _trim_object_descriptor(clicked_descriptor) if isinstance(clicked_descriptor, dict) else {},
        "changed_objects": [_trim_object_descriptor(row) for row in changed_rows],
        "relation_summary": relation_summary,
    }


def _base_analyst_context(
    *,
    obs_before: Dict[str, Any],
    result: Dict[str, Any],
    action_to_use: Dict[str, Any],
    reward: float,
    visual_feedback: Dict[str, Any],
    failure_context: Dict[str, Any],
    candidate_functions: Sequence[str],
    progress_snapshot: Dict[str, Any],
    object_summary: Dict[str, Any],
    goal_prior_summary: Sequence[Dict[str, Any]],
    proposal_feedback_summary: Dict[str, Any],
) -> str:
    action_summary = {
        "function_name": str(extract_action_function_name(action_to_use, default="wait") or "wait"),
        "xy": list(extract_action_xy(action_to_use) or []),
        "reward": float(reward or 0.0),
    }
    visual_summary = {
        "changed_pixel_count": round(float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0), 4),
        "changed_ratio": round(float(visual_feedback.get("changed_ratio", 0.0) or 0.0), 4),
        "changed_bbox_area": round(float(visual_feedback.get("changed_bbox_area", 0.0) or 0.0), 4),
        "changed_bbox_ratio": round(float(visual_feedback.get("changed_bbox_ratio", 0.0) or 0.0), 4),
        "hotspot_source": _safe_text(visual_feedback.get("hotspot_source", ""), limit=80),
        "changed_bbox": dict(result.get("changed_bbox", {}))
        if isinstance(result.get("changed_bbox", {}), dict)
        else dict((result.get("perception", {}) or {}).get("changed_bbox", {}))
        if isinstance((result.get("perception", {}) or {}).get("changed_bbox", {}), dict)
        else {},
    }
    compact_before = _compact_observation(obs_before)
    compact_after = _compact_observation(result if isinstance(result, dict) else {})
    before_summary = {
        "instruction": compact_before.get("instruction", ""),
        "perception": compact_before.get("perception", {}),
        "world_state": compact_before.get("world_state", {}),
        "visible_functions": compact_before.get("visible_functions", []),
        "discovered_functions": compact_before.get("discovered_functions", []),
    }
    after_summary = {
        "perception": compact_after.get("perception", {}),
        "world_state": compact_after.get("world_state", {}),
        "visible_functions": compact_after.get("visible_functions", []),
        "discovered_functions": compact_after.get("discovered_functions", []),
    }
    progress_markers = list(progress_snapshot.get("progress_markers", []) or [])
    compact_progress = [
        {
            "name": _safe_text(marker.get("name", ""), limit=80),
            "changed_pixel_count": marker.get("changed_pixel_count"),
            "changed_ratio": marker.get("changed_ratio"),
            "changed_bbox_ratio": marker.get("changed_bbox_ratio"),
            "source": _safe_text(marker.get("source", ""), limit=80),
        }
        for marker in progress_markers[-3:]
        if isinstance(marker, dict)
    ]
    progress_summary = {
        "goal_progress_assessment": progress_snapshot.get("goal_progress_assessment", {}),
        "task_progress": progress_snapshot.get("task_progress", {}),
        "inferred_level_goal": progress_snapshot.get("inferred_level_goal", {}),
        "progress_markers": compact_progress,
    }
    world_model_goal_decision = _compact_world_model_goal_decision(progress_snapshot)
    return (
        f"BeforeSummary:{json.dumps(before_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"AfterSummary:{json.dumps(after_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"ActionSummary:{json.dumps(action_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"VisualFeedback:{json.dumps(visual_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"ObjectChangeSummary:{json.dumps(object_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"InitialGoalPriors:{json.dumps(list(goal_prior_summary or []), ensure_ascii=False, separators=(',', ':'))}\n"
        f"WorldModelGoalDecision:{json.dumps(world_model_goal_decision, ensure_ascii=False, separators=(',', ':'))}\n"
        f"WorldModelProposalFeedback:{json.dumps(dict(proposal_feedback_summary or {}), ensure_ascii=False, separators=(',', ':'))}\n"
        f"ProgressSummary:{json.dumps(progress_summary, ensure_ascii=False, separators=(',', ':'))}\n"
        f"RecentFailure:{json.dumps(failure_context, ensure_ascii=False, separators=(',', ':'))}\n"
        f"CandidateFunctions:{json.dumps(list(candidate_functions), ensure_ascii=False, separators=(',', ':'))}"
    )


def _parse_analyst_template(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    fields = _parse_fixed_field_lines(visible)
    function_name = _safe_text(fields.get("DISCRIMINATING_ACTION", ""), limit=120)
    if function_name.lower() in {"none", "n/a", "null"}:
        function_name = ""
    if function_name and function_name not in set(candidate_functions or []):
        function_name = ""
    risk = _safe_text(fields.get("SURFACE_CUE_RISK", ""), limit=40).lower()
    if risk not in {"low", "medium", "high"}:
        risk = ""
    payload = {
        "situation_assessment": _safe_text(fields.get("SITUATION_ASSESSMENT", ""), limit=360),
        "top_hypothesis": _safe_text(fields.get("TOP_HYPOTHESIS", ""), limit=220),
        "alternative_hypothesis": _safe_text(fields.get("ALT_HYPOTHESIS", ""), limit=220),
        "mechanism_guess": _safe_text(fields.get("MECHANISM_GUESS", ""), limit=320),
        "discriminating_action": {
            "function_name": function_name,
            "confidence": _parse_confidence_field(fields.get("CONFIDENCE"), default=0.0),
        },
        "expected_visual_change": _safe_text(fields.get("EXPECTED_VISUAL_CHANGE", ""), limit=220),
        "confidence": _parse_confidence_field(fields.get("CONFIDENCE"), default=0.0),
        "surface_cue_risk": risk,
    }
    if not any(
        [
            payload["situation_assessment"],
            payload["top_hypothesis"],
            payload["mechanism_guess"],
            payload["discriminating_action"]["function_name"],
            payload["expected_visual_change"],
        ]
    ):
        return {}
    return payload


def _heuristic_analyst_payload(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    hypothesis = _heuristic_hypothesis_payload(raw_text, candidate_functions)
    action = _heuristic_action_payload(raw_text, candidate_functions)
    situation = _safe_text(visible, limit=360)
    risk = "high" if "visible" in visible.lower() and "because" not in visible.lower() else "medium" if visible else "low"
    return {
        "situation_assessment": situation,
        "top_hypothesis": _safe_text(hypothesis.get("claim", ""), limit=220),
        "alternative_hypothesis": "",
        "mechanism_guess": _safe_text(hypothesis.get("mechanism_rationale", ""), limit=320),
        "discriminating_action": {
            "function_name": _safe_text(action.get("function_name", ""), limit=120),
            "confidence": _clamp01(action.get("confidence", 0.0), 0.0),
        },
        "expected_visual_change": "",
        "confidence": _clamp01(action.get("confidence", 0.0), 0.0),
        "surface_cue_risk": risk,
    }


def _parse_initial_goal_template(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    fields = _parse_fixed_field_lines(visible)
    function_name = _safe_text(fields.get("DISCRIMINATING_FUNCTION", ""), limit=120)
    if function_name.lower() in {"none", "n/a", "null"}:
        function_name = ""
    if function_name and function_name not in set(candidate_functions or []):
        function_name = ""
    risk = _safe_text(fields.get("SURFACE_CUE_RISK", ""), limit=40).lower()
    if risk not in {"low", "medium", "high"}:
        risk = ""

    def _goal_row(prefix: str, *, fallback: bool = False) -> Dict[str, Any]:
        hypothesis_key = f"{prefix}HYPOTHESIS" if not fallback else "GOAL_HYPOTHESIS"
        target_group_key = f"{prefix}TARGET_GROUP" if not fallback else "TARGET_GROUP"
        target_relation_key = f"{prefix}TARGET_RELATION" if not fallback else "TARGET_RELATION"
        completion_key = f"{prefix}COMPLETION_CRITERION" if not fallback else "COMPLETION_CRITERION"
        evidence_for_key = f"{prefix}EVIDENCE_FOR"
        evidence_against_key = f"{prefix}EVIDENCE_AGAINST"
        confidence_key = f"{prefix}CONFIDENCE" if not fallback else "CONFIDENCE"
        row = {
            "goal_hypothesis": _safe_text(fields.get(hypothesis_key, ""), limit=220),
            "target_group": _safe_text(fields.get(target_group_key, ""), limit=180),
            "target_relation": _safe_text(fields.get(target_relation_key, ""), limit=120),
            "completion_criterion": _safe_text(fields.get(completion_key, ""), limit=220),
            "evidence_for": _safe_text(fields.get(evidence_for_key, ""), limit=220),
            "evidence_against": _safe_text(fields.get(evidence_against_key, ""), limit=220),
            "confidence": _parse_confidence_field(fields.get(confidence_key), default=0.0),
        }
        if not any(
            [
                row["goal_hypothesis"],
                row["target_group"],
                row["target_relation"],
                row["completion_criterion"],
                row["evidence_for"],
                row["evidence_against"],
            ]
        ):
            return {}
        return row

    goal_rows: List[Dict[str, Any]] = []
    for goal_idx in range(1, 4):
        row = _goal_row(f"GOAL_{goal_idx}_")
        if row:
            row["goal_rank"] = goal_idx
            goal_rows.append(row)
    if not goal_rows:
        fallback_row = _goal_row("", fallback=True)
        if fallback_row:
            fallback_row["goal_rank"] = 1
            alt_text = _safe_text(fields.get("ALT_GOAL_HYPOTHESIS", ""), limit=220)
            if alt_text and alt_text.lower() not in {"none", "n/a", "null"}:
                fallback_row["alternative_goal_hypothesis"] = alt_text
            goal_rows.append(fallback_row)

    if not goal_rows:
        return {}
    top_goal = dict(goal_rows[0])
    payload = {
        "goal_hypotheses": goal_rows[:3],
        "goal_hypothesis": top_goal.get("goal_hypothesis", ""),
        "alternative_goal_hypothesis": _safe_text(top_goal.get("alternative_goal_hypothesis", ""), limit=220),
        "target_group": top_goal.get("target_group", ""),
        "target_relation": top_goal.get("target_relation", ""),
        "completion_criterion": top_goal.get("completion_criterion", ""),
        "discriminating_function": function_name,
        "confidence": _parse_confidence_field(top_goal.get("confidence"), default=0.0),
        "surface_cue_risk": risk,
    }
    return payload


def _heuristic_initial_goal_payload(raw_text: str, candidate_functions: Sequence[str]) -> Dict[str, Any]:
    visible = _shadow_visible_response_text(raw_text)
    text = _safe_text(visible, limit=280)
    lowered = text.lower()
    relation = "unknown"
    if "align" in lowered or "alignment" in lowered:
        relation = "alignment"
    elif "pair" in lowered:
        relation = "pairing"
    elif "symmetr" in lowered:
        relation = "symmetry"
    elif "order" in lowered or "sequence" in lowered:
        relation = "ordering"
    candidate_fn = ""
    for fn_name in list(candidate_functions or []):
        if fn_name and str(fn_name) in text:
            candidate_fn = str(fn_name)
            break
    goal_row = {
        "goal_rank": 1,
        "goal_hypothesis": text,
        "target_group": "",
        "target_relation": relation,
        "completion_criterion": "",
        "evidence_for": "",
        "evidence_against": "",
        "confidence": 0.35 if text else 0.0,
    }
    return {
        "goal_hypotheses": [goal_row] if text else [],
        "goal_hypothesis": text,
        "alternative_goal_hypothesis": "",
        "target_group": "",
        "target_relation": relation,
        "completion_criterion": "",
        "discriminating_function": candidate_fn,
        "confidence": 0.35 if text else 0.0,
        "surface_cue_risk": "medium" if text else "low",
    }


def _build_initial_goal_hypothesis_candidates(
    loop: Any,
    *,
    payload: Dict[str, Any],
    object_summary: Dict[str, Any],
    candidate_functions: Sequence[str],
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    goal_rows = [
        dict(row)
        for row in list(payload.get("goal_hypotheses", []) or [])
        if isinstance(row, dict)
    ]
    if not goal_rows and any(
        [
            _safe_text(payload.get("goal_hypothesis", ""), limit=220),
            _safe_text(payload.get("target_group", ""), limit=180),
            _safe_text(payload.get("target_relation", ""), limit=120),
            _safe_text(payload.get("completion_criterion", ""), limit=220),
        ]
    ):
        goal_rows = [
            {
                "goal_rank": 1,
                "goal_hypothesis": _safe_text(payload.get("goal_hypothesis", ""), limit=220),
                "target_group": _safe_text(payload.get("target_group", ""), limit=180),
                "target_relation": _safe_text(payload.get("target_relation", ""), limit=120),
                "completion_criterion": _safe_text(payload.get("completion_criterion", ""), limit=220),
                "evidence_for": "",
                "evidence_against": "",
                "confidence": _parse_confidence_field(payload.get("confidence"), default=0.0),
                "alternative_goal_hypothesis": _safe_text(payload.get("alternative_goal_hypothesis", ""), limit=220),
            }
        ]

    def _make_candidate(
        *,
        goal_row: Dict[str, Any],
        confidence_scale: float,
        surface_priority: float,
    ) -> Optional[Dict[str, Any]]:
        text = _safe_text(goal_row.get("goal_hypothesis", ""), limit=220)
        if not text:
            return None
        goal_rank = max(1, int(goal_row.get("goal_rank", 1) or 1))
        soft_confidence = round(
            min(0.47, max(0.18, _clamp01(goal_row.get("confidence", 0.0), 0.0) * confidence_scale)),
            4,
        )
        discriminating_function = _safe_text(payload.get("discriminating_function", ""), limit=120)
        if discriminating_function and discriminating_function not in set(candidate_functions or []):
            discriminating_function = ""
        return {
            "object_id": f"initial_goal_ep{int(getattr(loop, '_episode', 0) or 0)}_goal{goal_rank}",
            "object_type": OBJECT_TYPE_HYPOTHESIS,
            "family": "llm_initial_goal_prior",
            "summary": text,
            "description": text,
            "statement": text,
            "confidence": soft_confidence,
            "hypothesis_type": "analyst_initial_goal_prior",
            "target_functions": [discriminating_function] if discriminating_function else [],
            "memory_layer": "working",
            "memory_type": "analyst_initial_goal_prior",
            "asset_status": "transient",
            "surface_priority": surface_priority,
            "source": "llm_initial_goal_analyst",
            "source_episode": int(getattr(loop, "_episode", 0) or 0),
            "source_tick": int(getattr(loop, "_tick", 0) or 0),
            "provenance": {
                "llm_mode": str(getattr(loop, "_llm_mode", "integrated") or "integrated"),
                "surface_cue_risk": _safe_text(payload.get("surface_cue_risk", ""), limit=40),
                "target_group": _safe_text(goal_row.get("target_group", ""), limit=180),
                "target_relation": _safe_text(goal_row.get("target_relation", ""), limit=120),
                "goal_rank": goal_rank,
            },
            "goal_prior_payload": {
                "target_group": _safe_text(goal_row.get("target_group", ""), limit=180),
                "target_relation": _safe_text(goal_row.get("target_relation", ""), limit=120),
                "completion_criterion": _safe_text(goal_row.get("completion_criterion", ""), limit=220),
                "alternative_goal_hypothesis": _safe_text(goal_row.get("alternative_goal_hypothesis", ""), limit=220),
                "evidence_for": _safe_text(goal_row.get("evidence_for", ""), limit=220),
                "evidence_against": _safe_text(goal_row.get("evidence_against", ""), limit=220),
                "goal_rank": goal_rank,
                "object_group_summary": dict(object_summary or {}),
            },
        }

    rows: List[Dict[str, Any]] = []
    confidence_scales = [0.62, 0.5, 0.42]
    surface_priorities = [0.28, 0.24, 0.2]
    for idx, goal_row in enumerate(goal_rows[:3]):
        candidate = _make_candidate(
            goal_row=goal_row,
            confidence_scale=confidence_scales[idx] if idx < len(confidence_scales) else 0.4,
            surface_priority=surface_priorities[idx] if idx < len(surface_priorities) else 0.18,
        )
        if candidate:
            rows.append(candidate)
    return rows[:3]


def _current_initial_goal_prior_summary(loop: Any) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(getattr(loop, "_llm_initial_goal_hypothesis_candidates", []) or [])
        if isinstance(row, dict) and int(row.get("source_episode", -1) or -1) == int(getattr(loop, "_episode", 0) or 0)
    ]
    rows.sort(key=lambda row: float(row.get("confidence", 0.0) or 0.0), reverse=True)
    summary: List[Dict[str, Any]] = []
    for row in rows[:3]:
        payload = row.get("goal_prior_payload", {}) if isinstance(row.get("goal_prior_payload", {}), dict) else {}
        summary.append(
            {
                "summary": _safe_text(row.get("summary", ""), limit=180),
                "target_group": _safe_text(payload.get("target_group", ""), limit=160),
                "target_relation": _safe_text(payload.get("target_relation", ""), limit=80),
                "completion_criterion": _safe_text(payload.get("completion_criterion", ""), limit=180),
                "goal_rank": int(payload.get("goal_rank", row.get("provenance", {}).get("goal_rank", 0)) or 0),
                "confidence": round(float(row.get("confidence", 0.0) or 0.0), 4),
            }
        )
    return summary


def _analyst_reasoning_profile(loop: Any, reasons: Sequence[str]) -> Dict[str, Any]:
    tick = int(getattr(loop, "_tick", 0) or 0)
    reason_set = {str(item or "") for item in list(reasons or [])}
    if "new_failure_pattern" in reason_set and tick >= 2:
        return {"name": "medium", "think": False, "token_scale": 0.8}
    if "visual_change" in reason_set:
        return {"name": "low", "think": False, "token_scale": 0.62}
    return {"name": "low", "think": False, "token_scale": 0.55}


def _analyst_surface_parroting(payload: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
    pseudo_payload = {
        "top_hypothesis": {
            "claim": payload.get("top_hypothesis", ""),
            "mechanism_rationale": payload.get("mechanism_guess", ""),
            "surface_cue_refs": [],
        },
        "discriminating_action": {
            "why_discriminating": payload.get("situation_assessment", ""),
            "surface_cue_refs": [],
        },
        "failure_explanation": {
            "summary": payload.get("alternative_hypothesis", ""),
            "mechanism_rationale": payload.get("expected_visual_change", ""),
            "surface_cue_refs": [],
        },
    }
    analysis = _surface_parroting_analysis(pseudo_payload, obs_before)
    explicit_risk = str(payload.get("surface_cue_risk", "") or "").lower()
    if explicit_risk in {"low", "medium", "high"}:
        analysis["risk"] = explicit_risk
    return analysis


def _analyst_metrics(
    payload: Dict[str, Any],
    *,
    actual_available: Set[str],
    visual_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    suggested_fn = str(((payload.get("discriminating_action", {}) or {}).get("function_name", "")) or "").strip()
    suggested_available = bool(suggested_fn and suggested_fn in actual_available)
    mechanism_text = " ".join(
        [
            str(payload.get("mechanism_guess", "") or ""),
            str(payload.get("top_hypothesis", "") or ""),
            str(payload.get("situation_assessment", "") or ""),
        ]
    ).lower()
    mentions_mechanism = bool(
        str(payload.get("mechanism_guess", "") or "").strip()
        or any(token in mechanism_text for token in _MECHANISM_MARKERS)
    )
    visual_focus_text = " ".join(
        [
            str(payload.get("situation_assessment", "") or ""),
            str(payload.get("expected_visual_change", "") or ""),
            str(payload.get("mechanism_guess", "") or ""),
        ]
    ).lower()
    visual_tokens = {"change", "changed", "pixel", "visual", "bbox", "region", "shift", "moved", "appear", "disappear"}
    changed_pixels = float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0)
    mentions_visual_delta = bool(changed_pixels <= 0.0 or any(token in visual_focus_text for token in visual_tokens))
    return {
        "suggested_action_available": suggested_available,
        "mentions_mechanism": mentions_mechanism,
        "mentions_visual_delta": mentions_visual_delta,
    }


def _build_analyst_hypothesis_candidates(
    loop: Any,
    *,
    payload: Dict[str, Any],
    trigger_reasons: Sequence[str],
    object_summary: Dict[str, Any],
    visual_feedback: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    claim = _safe_text(payload.get("top_hypothesis", ""), limit=220)
    mechanism_guess = _safe_text(payload.get("mechanism_guess", ""), limit=320)
    expected_visual_change = _safe_text(payload.get("expected_visual_change", ""), limit=220)
    if not any([claim, mechanism_guess, expected_visual_change]):
        return []
    suggested_action = _safe_text(
        ((payload.get("discriminating_action", {}) or {}).get("function_name", "")),
        limit=120,
    )
    clicked_object = object_summary.get("clicked_object", {}) if isinstance(object_summary, dict) else {}
    relation_summary = object_summary.get("relation_summary", {}) if isinstance(object_summary, dict) else {}
    summary = claim or mechanism_guess or expected_visual_change
    soft_confidence = round(
        min(0.49, max(0.18, _clamp01(payload.get("confidence", 0.0), 0.0) * 0.55)),
        4,
    )
    candidate = {
        "object_id": f"analyst_hyp_ep{int(getattr(loop, '_episode', 0) or 0)}_tick{int(getattr(loop, '_tick', 0) or 0)}",
        "object_type": OBJECT_TYPE_HYPOTHESIS,
        "family": "llm_analyst_relational",
        "summary": summary,
        "description": summary,
        "statement": claim or summary,
        "confidence": soft_confidence,
        "hypothesis_type": "analyst_relational_candidate",
        "expected_transition": expected_visual_change,
        "target_functions": [suggested_action] if suggested_action else [],
        "memory_layer": "working",
        "memory_type": "analyst_hypothesis_candidate",
        "asset_status": "transient",
        "surface_priority": 0.22,
        "source": "llm_analyst",
        "source_episode": int(getattr(loop, "_episode", 0) or 0),
        "source_tick": int(getattr(loop, "_tick", 0) or 0),
        "supporting_evidence": [
            {
                "kind": "visual_delta",
                "changed_pixel_count": float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0),
                "changed_bbox_ratio": float(visual_feedback.get("changed_bbox_ratio", 0.0) or 0.0),
                "trigger_reasons": list(trigger_reasons or []),
            }
        ],
        "provenance": {
            "llm_mode": str(getattr(loop, "_llm_mode", "integrated") or "integrated"),
            "trigger_reasons": list(trigger_reasons or []),
            "clicked_anchor_ref": _safe_text(clicked_object.get("anchor_ref", ""), limit=120),
            "same_anchor_as_clicked": bool(relation_summary.get("same_anchor_as_clicked", False)),
            "surface_cue_risk": _safe_text(payload.get("surface_cue_risk", ""), limit=40),
        },
        "analyst_payload": {
            "mechanism_guess": mechanism_guess,
            "alternative_hypothesis": _safe_text(payload.get("alternative_hypothesis", ""), limit=220),
            "situation_assessment": _safe_text(payload.get("situation_assessment", ""), limit=320),
            "expected_visual_change": expected_visual_change,
        },
    }
    return [candidate]


def prepare_llm_analyst_initial_goal(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
) -> None:
    client = loop._resolve_llm_client("analyst") if hasattr(loop, "_resolve_llm_client") else getattr(loop, "_llm_analyst_client", None)
    if client is None:
        return
    if int(getattr(loop, "_tick", 0) or 0) != 0:
        return
    current_episode = int(getattr(loop, "_episode", 0) or 0)
    existing_rows = [
        row
        for row in list(getattr(loop, "_llm_initial_goal_hypothesis_candidates", []) or [])
        if isinstance(row, dict) and int(row.get("source_episode", -1) or -1) == current_episode
    ]
    if existing_rows:
        return
    candidate_functions = list(loop._extract_known_functions(obs_before) or [])
    object_summary = _compact_initial_goal_object_summary(loop, obs_before)
    world_model_snapshot = _build_world_model_snapshot(
        loop,
        obs=obs_before,
        object_summary=object_summary,
    )
    base = _base_initial_goal_context(
        obs_before=obs_before,
        candidate_functions=candidate_functions,
        object_summary=object_summary,
        world_model_snapshot={
            "available_action_names": list(world_model_snapshot.get("available_action_names", []) or [])[:8],
            "groups": dict(world_model_snapshot.get("groups", {}) or {}),
            "goal_family_candidates": list(world_model_snapshot.get("goal_family_candidates", []) or [])[:4],
            "inferred_level_goal": dict(world_model_snapshot.get("inferred_level_goal", {}) or {}),
        },
    )
    analyst_tokens = 220
    _record_shadow_call(loop, "analyst_initial_goal", route_name="analyst")
    raw = _shadow_raw_text(
        client,
        (
            "Initial goal analyst task: infer top-3 tentative level objectives from the first visible state only.\n"
            "Return exactly these lines and nothing else:\n"
            "GOAL_1_HYPOTHESIS: <short tentative objective>\n"
            "GOAL_1_TARGET_GROUP: <short object-group description>\n"
            "GOAL_1_TARGET_RELATION: <alignment|pairing|symmetry|containment|ordering|count_match|state_change|unknown>\n"
            "GOAL_1_COMPLETION_CRITERION: <short observable completion condition>\n"
            "GOAL_1_EVIDENCE_FOR: <brief structural evidence>\n"
            "GOAL_1_EVIDENCE_AGAINST: <brief uncertainty or counter-evidence>\n"
            "GOAL_1_CONFIDENCE: <0.0-1.0>\n"
            "GOAL_2_HYPOTHESIS: <second tentative objective or none>\n"
            "GOAL_2_TARGET_GROUP: <short object-group description or none>\n"
            "GOAL_2_TARGET_RELATION: <alignment|pairing|symmetry|containment|ordering|count_match|state_change|unknown>\n"
            "GOAL_2_COMPLETION_CRITERION: <short observable completion condition or none>\n"
            "GOAL_2_EVIDENCE_FOR: <brief structural evidence or none>\n"
            "GOAL_2_EVIDENCE_AGAINST: <brief uncertainty or counter-evidence or none>\n"
            "GOAL_2_CONFIDENCE: <0.0-1.0>\n"
            "GOAL_3_HYPOTHESIS: <third tentative objective or none>\n"
            "GOAL_3_TARGET_GROUP: <short object-group description or none>\n"
            "GOAL_3_TARGET_RELATION: <alignment|pairing|symmetry|containment|ordering|count_match|state_change|unknown>\n"
            "GOAL_3_COMPLETION_CRITERION: <short observable completion condition or none>\n"
            "GOAL_3_EVIDENCE_FOR: <brief structural evidence or none>\n"
            "GOAL_3_EVIDENCE_AGAINST: <brief uncertainty or counter-evidence or none>\n"
            "GOAL_3_CONFIDENCE: <0.0-1.0>\n"
            "DISCRIMINATING_FUNCTION: <one function from CandidateFunctions or none>\n"
            "SURFACE_CUE_RISK: <low|medium|high>\n"
            "Ground every goal in ObjectGroupSummary. Prefer grouped objects, pair candidates, relation cues, and controller candidates over isolated salient objects.\n"
            "Prefer tentative goal descriptions and relation guesses over definitive claims.\n"
            + base
        ),
        capability=ANALYSIS_VERIFICATION_REVIEW,
        max_tokens=analyst_tokens,
        think=False,
        system_prompt="Return only the requested fixed-field lines. No JSON. No bullets. No prose outside the fields.",
    )
    payload = _parse_initial_goal_template(raw, candidate_functions)
    if not payload:
        payload = _heuristic_initial_goal_payload(raw, candidate_functions)
    rows = _build_initial_goal_hypothesis_candidates(
        loop,
        payload=payload,
        object_summary=object_summary,
        candidate_functions=candidate_functions,
    )
    world_model_snapshot = _build_world_model_snapshot(
        loop,
        obs=obs_before,
        object_summary=object_summary,
        initial_goal_prior_rows=rows,
    )
    proposal_candidates = _build_world_model_proposal_candidates(
        loop,
        payload=payload,
        candidate_functions=candidate_functions,
        source_kind="initial_goal",
    )
    validation_feedback = validate_goal_proposal_candidates(
        world_model_snapshot.get("task_frame_summary", {}),
        proposal_candidates,
    )
    loop._llm_initial_goal_hypothesis_candidates = rows
    loop._llm_world_model_snapshot = dict(world_model_snapshot)
    loop._llm_world_model_proposal_candidates = list(proposal_candidates)
    loop._llm_world_model_validation_feedback = list(validation_feedback)
    entry = {
        "entry_kind": "initial_goal_prior",
        "episode": current_episode,
        "tick": int(getattr(loop, "_tick", 0) or 0),
        "llm_mode": str(getattr(loop, "_llm_mode", "integrated") or "integrated"),
        "prompt_context": {
            "candidate_functions": list(candidate_functions),
            "object_group_summary": object_summary,
            "world_model_snapshot": {
                "available_action_names": list(world_model_snapshot.get("available_action_names", []) or [])[:8],
                "goal_family_candidates": list(world_model_snapshot.get("goal_family_candidates", []) or [])[:4],
                "inferred_level_goal": dict(world_model_snapshot.get("inferred_level_goal", {}) or {}),
            },
        },
        "analyst_output": {
            "goal_hypotheses": [
                {
                    "goal_rank": int(row.get("goal_rank", 0) or 0),
                    "goal_hypothesis": _safe_text(row.get("goal_hypothesis", ""), limit=220),
                    "target_group": _safe_text(row.get("target_group", ""), limit=180),
                    "target_relation": _safe_text(row.get("target_relation", ""), limit=120),
                    "completion_criterion": _safe_text(row.get("completion_criterion", ""), limit=220),
                    "evidence_for": _safe_text(row.get("evidence_for", ""), limit=220),
                    "evidence_against": _safe_text(row.get("evidence_against", ""), limit=220),
                    "confidence": _clamp01(row.get("confidence", 0.0), 0.0),
                }
                for row in list(payload.get("goal_hypotheses", []) or [])[:3]
                if isinstance(row, dict)
            ],
            "goal_hypothesis": _safe_text(payload.get("goal_hypothesis", ""), limit=220),
            "alternative_goal_hypothesis": _safe_text(payload.get("alternative_goal_hypothesis", ""), limit=220),
            "target_group": _safe_text(payload.get("target_group", ""), limit=180),
            "target_relation": _safe_text(payload.get("target_relation", ""), limit=120),
            "completion_criterion": _safe_text(payload.get("completion_criterion", ""), limit=220),
            "discriminating_function": _safe_text(payload.get("discriminating_function", ""), limit=120),
            "confidence": _clamp01(payload.get("confidence", 0.0), 0.0),
            "surface_cue_risk": _safe_text(payload.get("surface_cue_risk", ""), limit=40),
        },
        "world_model_proposals": list(proposal_candidates),
        "world_model_validation_feedback": _feedback_summary_rows(validation_feedback),
        "raw_response_preview": _safe_text(_shadow_visible_response_text(raw), limit=420),
        "surface_parroting": {"risk": _safe_text(payload.get("surface_cue_risk", ""), limit=40) or "medium"},
        "analyst_metrics": {
            "mentions_mechanism": bool(_safe_text(payload.get("target_relation", ""), limit=120)),
            "suggested_action_available": bool(
                _safe_text(payload.get("discriminating_function", ""), limit=120)
                and _safe_text(payload.get("discriminating_function", ""), limit=120) in set(candidate_functions or [])
            ),
            "mentions_visual_delta": False,
        },
    }
    loop._llm_analyst_log.append(entry)


def finalize_llm_analyst_post_execution(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    action_to_use: Dict[str, Any],
    result: Dict[str, Any],
    reward: float,
) -> None:
    client = loop._resolve_llm_client("analyst") if hasattr(loop, "_resolve_llm_client") else getattr(loop, "_llm_analyst_client", None)
    if client is None:
        return
    visual_feedback = loop._extract_visual_feedback(result)
    candidate_functions = _post_action_candidate_universe(
        loop,
        obs_before=obs_before,
        result=result if isinstance(result, dict) else {},
        action_to_use=action_to_use,
    )
    failure_context = _recent_failure_context(loop)
    trigger = _should_run_llm_analyst(
        loop,
        result=result if isinstance(result, dict) else {},
        candidate_functions=candidate_functions,
        failure_context=failure_context,
        visual_feedback=visual_feedback,
    )
    loop._llm_analyst_last_observation_signature = trigger.get("observation_signature", "")
    loop._llm_analyst_last_failure_signature = trigger.get("failure_signature", "")
    if not bool(trigger.get("should_run", False)):
        _record_analyst_skip(loop, trigger.get("reasons", []))
        return
    profile = _analyst_reasoning_profile(loop, trigger.get("reasons", []))
    progress_snapshot = _latest_trace_progress_snapshot(loop)
    object_summary = _analyst_surface_object_summary(
        loop,
        obs_before=obs_before,
        result=result if isinstance(result, dict) else {},
        action_to_use=action_to_use,
    )
    world_model_snapshot = _build_world_model_snapshot(
        loop,
        obs=result if isinstance(result, dict) else obs_before,
        object_summary={
            "color_groups": [],
            "shape_groups": [],
            "pair_candidates": [],
            "controller_candidates": [],
        },
        initial_goal_prior_rows=list(getattr(loop, "_llm_initial_goal_hypothesis_candidates", []) or []),
    )
    proposal_feedback_summary = _current_world_model_feedback_summary(loop)
    base = _base_analyst_context(
        obs_before=obs_before,
        result=result if isinstance(result, dict) else {},
        action_to_use=action_to_use,
        reward=reward,
        visual_feedback=visual_feedback,
        failure_context=failure_context,
        candidate_functions=candidate_functions,
        progress_snapshot=progress_snapshot,
        object_summary=object_summary,
        goal_prior_summary=_current_initial_goal_prior_summary(loop),
        proposal_feedback_summary=proposal_feedback_summary,
    )
    token_scale = float(profile.get("token_scale", 1.0) or 1.0)
    analyst_tokens = max(180, int(round(300 * token_scale)))
    _record_shadow_call(loop, "analyst_situation", route_name="analyst")
    analyst_raw = _shadow_raw_text(
        client,
        (
            "Analyst task: summarize the current post-action state from the visual delta and object-change relations, then suggest one next-step hypothesis.\n"
            "Return exactly these lines and nothing else:\n"
            "SITUATION_ASSESSMENT: <one short sentence describing the observed change>\n"
            "TOP_HYPOTHESIS: <short working hypothesis>\n"
            "ALT_HYPOTHESIS: <optional alternate or none>\n"
            "MECHANISM_GUESS: <short relational or structural explanation grounded in the observed change>\n"
            "DISCRIMINATING_ACTION: <one function from CandidateFunctions or none>\n"
            "EXPECTED_VISUAL_CHANGE: <short expectation if that action is tried>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "SURFACE_CUE_RISK: <low|medium|high>\n"
            "Use InitialGoalPriors as tentative targets to support, weaken, or refine; do not treat them as ground truth.\n"
            "Prefer neutral relational descriptions over strong causal claims unless the evidence is direct.\n"
            "Prefer concrete references to changed regions, object relations, or repeated patterns over speculative stories.\n"
            + base
        ),
        capability=ANALYSIS_VERIFICATION_REVIEW,
        max_tokens=analyst_tokens,
        think=profile.get("think"),
        system_prompt="Return only the requested fixed-field lines. No JSON. No bullets. No prose outside the fields.",
    )
    payload = _parse_analyst_template(analyst_raw, candidate_functions)
    if not payload:
        payload = _heuristic_analyst_payload(analyst_raw, candidate_functions)
    loop._llm_analyst_hypothesis_candidates = _build_analyst_hypothesis_candidates(
        loop,
        payload=payload,
        trigger_reasons=trigger.get("reasons", []),
        object_summary=object_summary,
        visual_feedback=visual_feedback,
    )
    analyst_proposal_candidates = _build_world_model_proposal_candidates(
        loop,
        payload={
            "goal_hypotheses": [],
            "top_hypothesis": payload.get("top_hypothesis", ""),
            "mechanism_guess": payload.get("mechanism_guess", ""),
            "expected_visual_change": payload.get("expected_visual_change", ""),
            "confidence": payload.get("confidence", 0.0),
            "discriminating_function": ((payload.get("discriminating_action", {}) or {}).get("function_name", "")),
        },
        candidate_functions=candidate_functions,
        source_kind="analyst",
    )
    analyst_validation_feedback = validate_goal_proposal_candidates(
        world_model_snapshot.get("task_frame_summary", {}),
        analyst_proposal_candidates,
    )
    existing_proposals = [
        dict(row)
        for row in list(getattr(loop, "_llm_world_model_proposal_candidates", []) or [])
        if isinstance(row, dict)
    ]
    existing_feedback = [
        dict(row)
        for row in list(getattr(loop, "_llm_world_model_validation_feedback", []) or [])
        if isinstance(row, dict)
    ]
    loop._llm_world_model_snapshot = dict(world_model_snapshot)
    loop._llm_world_model_proposal_candidates = (existing_proposals + analyst_proposal_candidates)[-16:]
    loop._llm_world_model_validation_feedback = (existing_feedback + analyst_validation_feedback)[-24:]
    actual_available = set(loop._collect_executable_function_names(result if isinstance(result, dict) else obs_before))
    entry = {
        "episode": loop._episode,
        "tick": loop._tick,
        "llm_mode": str(getattr(loop, "_llm_mode", "integrated") or "integrated"),
        "prompt_context": {
            "candidate_functions": list(candidate_functions),
            "recent_failure_context": failure_context,
            "trigger_reasons": list(trigger.get("reasons", []) or []),
            "reasoning_profile": dict(profile),
            "visual_feedback": {
                "changed_pixel_count": float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0),
                "changed_ratio": float(visual_feedback.get("changed_ratio", 0.0) or 0.0),
                "changed_bbox_ratio": float(visual_feedback.get("changed_bbox_ratio", 0.0) or 0.0),
            },
            "object_change_summary": object_summary,
            "world_model_proposal_feedback": proposal_feedback_summary,
        },
        "analyst_output": payload,
        "world_model_proposals": list(analyst_proposal_candidates),
        "world_model_validation_feedback": _feedback_summary_rows(analyst_validation_feedback),
        "raw_response_preview": _safe_text(_shadow_visible_response_text(analyst_raw), limit=420),
        "system_reference": {
            "selected_action": _system_selected_action(loop, action_to_use),
            "actual_available_functions": sorted(actual_available),
            "progress_snapshot": progress_snapshot,
        },
        "surface_parroting": _analyst_surface_parroting(payload, obs_before),
        "analyst_metrics": _analyst_metrics(
            payload,
            actual_available=actual_available,
            visual_feedback=visual_feedback,
        ),
        "outcome": {
            "reward": float(reward or 0.0),
            "success": bool(result.get("success", False)) if isinstance(result, dict) else False,
            "terminal": bool(result.get("terminal", False) or result.get("done", False)) if isinstance(result, dict) else False,
        },
    }
    loop._llm_analyst_log.append(entry)


def build_llm_analyst_summary(log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [
        row for row in list(log or [])
        if isinstance(row, dict) and str(row.get("entry_kind", "") or "") != "initial_goal_prior"
    ]
    if not rows:
        return {
            "tick_count": 0,
            "average_confidence": 0.0,
            "mechanism_guess_rate": 0.0,
            "discriminating_action_suggestion_rate": 0.0,
            "suggested_action_available_rate": 0.0,
            "visual_change_focus_rate": 0.0,
            "surface_parroting_rate": 0.0,
        }
    confidences = [
        float(((row.get("analyst_output", {}) or {}).get("confidence", 0.0) or 0.0))
        for row in rows
    ]
    mechanism_count = sum(
        1
        for row in rows
        if bool(((row.get("analyst_metrics", {}) or {}).get("mentions_mechanism", False)))
    )
    action_count = sum(
        1
        for row in rows
        if str(
            ((row.get("analyst_output", {}) or {}).get("discriminating_action", {}) or {}).get("function_name", "")
            or ""
        ).strip()
    )
    available_count = sum(
        1
        for row in rows
        if bool(((row.get("analyst_metrics", {}) or {}).get("suggested_action_available", False)))
    )
    visual_count = sum(
        1
        for row in rows
        if bool(((row.get("analyst_metrics", {}) or {}).get("mentions_visual_delta", False)))
    )
    parroting = sum(
        1
        for row in rows
        if str(((row.get("surface_parroting", {}) or {}).get("risk", "low")) or "low") == "high"
    )
    return {
        "tick_count": len(rows),
        "average_confidence": round(sum(confidences) / max(1, len(confidences)), 4),
        "mechanism_guess_rate": round(mechanism_count / len(rows), 4),
        "discriminating_action_suggestion_rate": round(action_count / len(rows), 4),
        "suggested_action_available_rate": round(available_count / len(rows), 4),
        "visual_change_focus_rate": round(visual_count / len(rows), 4),
        "surface_parroting_rate": round(parroting / len(rows), 4),
    }

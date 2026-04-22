from __future__ import annotations

from typing import Any, Dict, List

from core.reasoning.arc_program_dsl import normalize_arc_program_rows


def _world_model_summary(workspace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("active_beliefs_summary", "world_model_summary"):
        raw = workspace.get(key, {})
        if isinstance(raw, dict):
            return raw
    return {}


def _program_constraints(summary: Dict[str, Any]) -> Dict[str, Any]:
    raw = summary.get("program_search_constraints", {}) if isinstance(summary, dict) else {}
    return dict(raw) if isinstance(raw, dict) else {}


def _skill_tags(row: Dict[str, Any]) -> List[str]:
    blob = " ".join(
        str(row.get(key, "") or "")
        for key in ("summary", "family", "object_type")
    ).lower()
    tags: List[str] = []
    if "transform" in blob:
        tags.append("transform")
    if "layout" in blob:
        tags.append("layout")
    if "component" in blob or "object" in blob:
        tags.append("object_centric")
    if "identity" in blob:
        tags.append("identity")
    if "marker" in blob or "trace" in blob:
        tags.append("marker_guided")
    return tags


def _world_model_program_score(row: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    constraints = _program_constraints(summary)
    preferred_tags = {
        str(tag or "") for tag in list(constraints.get("preferred_program_tags", []) or [])
        if str(tag or "")
    }
    blocked_tags = {
        str(tag or "") for tag in list(constraints.get("blocked_program_tags", []) or [])
        if str(tag or "")
    }
    tags = {
        str(tag or "") for tag in list(row.get("program_tags", row.get("tags", [])) or [])
        if str(tag or "")
    }
    preferred_hits = sorted(preferred_tags.intersection(tags))
    blocked_hits = sorted(blocked_tags.intersection(tags))
    strength = max(0.0, min(1.0, float(constraints.get("constraint_strength", 0.0) or 0.0)))
    info_gain = max(0.0, min(1.0, float(summary.get("expected_information_gain", 0.0) or 0.0)))
    rollout_uncertainty = max(0.0, min(1.0, float(summary.get("rollout_uncertainty", 0.0) or 0.0)))
    score_delta = (len(preferred_hits) * 0.12 * strength) - (len(blocked_hits) * 0.16 * strength)
    if preferred_hits:
        score_delta += info_gain * 0.04
    if blocked_hits:
        score_delta -= rollout_uncertainty * 0.05
    return {
        "score_delta": round(score_delta, 6),
        "preferred_hits": preferred_hits,
        "blocked_hits": blocked_hits,
    }


def search_candidate_programs(
    *,
    workspace: Dict[str, Any],
    obs: Dict[str, Any],
    synthesizer: Any = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    world_model_summary = _world_model_summary(workspace)
    if isinstance(obs, dict) and "arc_task" in obs and synthesizer is not None:
        programs = normalize_arc_program_rows(
            synthesizer.enumerate_arc_candidate_programs(obs.get("arc_task"), limit=max(limit * 2, limit)),
            limit=None,
        )
        rescored: List[Dict[str, Any]] = []
        for item in programs:
            row = dict(item)
            support = _world_model_program_score(row, world_model_summary)
            row["world_model_score_delta"] = support["score_delta"]
            row["world_model_preferred_tag_hits"] = support["preferred_hits"]
            row["world_model_blocked_tag_hits"] = support["blocked_hits"]
            row["score"] = round(float(row.get("score", 0.0) or 0.0) + support["score_delta"], 6)
            rescored.append(row)
        rescored.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                -len(list(item.get("world_model_preferred_tag_hits", []) or [])),
                len(list(item.get("world_model_blocked_tag_hits", []) or [])),
                str(item.get("program_id", "") or ""),
            )
        )
        return rescored[: max(0, int(limit))]

    active_skills = workspace.get("active_skills", [])
    if not isinstance(active_skills, list):
        active_skills = []
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(active_skills[: max(0, int(limit))]):
        if not isinstance(row, dict):
            continue
        skill_row = {
            "program_id": str(row.get("object_id", f"skill_program_{index}") or f"skill_program_{index}"),
            "name": str(row.get("summary", row.get("family", "")) or ""),
            "kind": "skill_program",
            "score": float(row.get("confidence", 0.5) or 0.5),
            "source_object_id": str(row.get("object_id", "") or ""),
            "program_tags": _skill_tags(row),
        }
        support = _world_model_program_score(skill_row, world_model_summary)
        skill_row["world_model_score_delta"] = support["score_delta"]
        skill_row["world_model_preferred_tag_hits"] = support["preferred_hits"]
        skill_row["world_model_blocked_tag_hits"] = support["blocked_hits"]
        skill_row["score"] = round(float(skill_row.get("score", 0.0) or 0.0) + support["score_delta"], 6)
        rows.append(skill_row)
    rows.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -len(list(item.get("world_model_preferred_tag_hits", []) or [])),
            str(item.get("program_id", "") or ""),
        )
    )
    return rows

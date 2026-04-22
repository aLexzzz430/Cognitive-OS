from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.hypothesis.mechanism_posterior_updater import (
    canonical_target_family,
    extract_target_descriptor,
    infer_action_family,
    normalize_action_family,
)


def _coerce_rows(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dedupe_mechanisms(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("object_id", "") or ""),
            str(row.get("hypothesis_id", "") or ""),
            str(row.get("family", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(dict(row))
    ordered.sort(
        key=lambda item: (
            -float(item.get("posterior", item.get("confidence", 0.0)) or 0.0),
            str(item.get("object_id", item.get("hypothesis_id", "")) or ""),
        )
    )
    return ordered


def extract_mechanism_decision_context(
    decision_context: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    context = dict(decision_context or {})
    uc = context.get("unified_context") or context.get("unified_cognitive_context")
    unified = dict(uc) if isinstance(uc, dict) else {}
    raw_control = unified.get("mechanism_control_summary", context.get("mechanism_control_summary", {}))
    mechanism_control = dict(raw_control) if isinstance(raw_control, dict) else {}

    mechanisms = _dedupe_mechanisms(
        _coerce_rows(unified.get("mechanism_hypothesis_objects", []))
        + _coerce_rows(unified.get("mechanism_hypotheses_summary", []))
        + _coerce_rows(context.get("mechanism_hypothesis_objects", []))
        + _coerce_rows(context.get("mechanism_hypotheses_summary", []))
    )
    if not mechanisms:
        return mechanism_control, [], {
            "mechanism_context_source": "summary" if mechanism_control else "empty",
            "mechanism_hypothesis_count": 0,
        }

    source = "summary+hypotheses" if mechanism_control else "hypotheses"
    top = dict(mechanisms[0])
    preferred_target_refs = [
        str(item or "")
        for item in list(mechanism_control.get("preferred_target_refs", []) or [])
        if str(item or "")
    ]
    preferred_action_families = [
        normalize_action_family(item)
        for item in list(mechanism_control.get("preferred_action_families", []) or [])
        if normalize_action_family(item)
    ]
    discriminating_actions = [
        normalize_action_family(item)
        for item in list(mechanism_control.get("discriminating_actions", []) or [])
        if normalize_action_family(item)
    ]
    mechanism_families = [
        str(item or "")
        for item in list(mechanism_control.get("mechanism_families", []) or [])
        if str(item or "")
    ]
    supported_goal_anchor_refs: List[str] = []
    goal_families: List[str] = []
    ranked_mechanism_object_ids = [
        str(item or "")
        for item in list(mechanism_control.get("ranked_mechanism_object_ids", []) or [])
        if str(item or "")
    ]

    for row in mechanisms:
        content = dict(row.get("content", {})) if isinstance(row.get("content", {}), dict) else {}
        family = str(content.get("family") or row.get("family") or "")
        if family and family not in mechanism_families:
            mechanism_families.append(family)
        goal_family = str(content.get("goal_family") or row.get("goal_family") or "")
        if goal_family and goal_family not in goal_families:
            goal_families.append(goal_family)
        object_id = str(row.get("object_id", "") or "")
        if object_id and object_id not in ranked_mechanism_object_ids:
            ranked_mechanism_object_ids.append(object_id)
        for value in list(content.get("preferred_target_refs", row.get("preferred_target_refs", [])) or []):
            text = str(value or "").strip()
            if text and text not in preferred_target_refs:
                preferred_target_refs.append(text)
        for value in list(content.get("supported_goal_anchor_refs", row.get("supported_goal_anchor_refs", [])) or []):
            text = str(value or "").strip()
            if text and text not in supported_goal_anchor_refs:
                supported_goal_anchor_refs.append(text)
            if text and text not in preferred_target_refs:
                preferred_target_refs.append(text)
        for value in list(content.get("preferred_action_families", row.get("preferred_action_families", [])) or []):
            text = normalize_action_family(value)
            if text and text not in preferred_action_families:
                preferred_action_families.append(text)
        for value in list(content.get("best_discriminating_actions", row.get("best_discriminating_actions", [])) or []):
            text = normalize_action_family(value)
            if text and text not in discriminating_actions:
                discriminating_actions.append(text)

    if not mechanism_control.get("dominant_mechanism_family"):
        mechanism_control["dominant_mechanism_family"] = str(top.get("family", "") or "")
    if "dominant_mechanism_confidence" not in mechanism_control:
        mechanism_control["dominant_mechanism_confidence"] = float(
            top.get("posterior", top.get("confidence", 0.0)) or 0.0
        )
    if not mechanism_control.get("dominant_mechanism_ref"):
        mechanism_control["dominant_mechanism_ref"] = str(top.get("hypothesis_id", "") or "")
    if not mechanism_control.get("dominant_mechanism_object_id"):
        mechanism_control["dominant_mechanism_object_id"] = str(top.get("object_id", "") or "")
    if ranked_mechanism_object_ids:
        mechanism_control["ranked_mechanism_object_ids"] = ranked_mechanism_object_ids[:8]
    if preferred_target_refs:
        mechanism_control["preferred_target_refs"] = preferred_target_refs[:8]
    if preferred_action_families:
        mechanism_control["preferred_action_families"] = preferred_action_families[:6]
    if discriminating_actions:
        mechanism_control["discriminating_actions"] = discriminating_actions[:6]
    if mechanism_families:
        mechanism_control["mechanism_families"] = mechanism_families[:8]
    if supported_goal_anchor_refs:
        mechanism_control["supported_goal_anchor_refs"] = supported_goal_anchor_refs[:8]
    if goal_families:
        mechanism_control["goal_families"] = goal_families[:6]
    mechanism_control["mechanism_hypothesis_count"] = len(mechanisms)
    mechanism_control["mechanism_context_source"] = source
    if "mechanism_ready" not in mechanism_control:
        mechanism_control["mechanism_ready"] = bool(preferred_target_refs or discriminating_actions)
    unresolved = [
        str(item or "")
        for item in list(mechanism_control.get("unresolved_mechanism_dimensions", []) or [])
        if str(item or "")
    ]
    if not preferred_target_refs and "no_preferred_targets" not in unresolved:
        unresolved.append("no_preferred_targets")
    if not discriminating_actions and "no_discriminating_actions" not in unresolved:
        unresolved.append("no_discriminating_actions")
    mechanism_control["unresolved_mechanism_dimensions"] = unresolved
    diagnostics = {
        "mechanism_context_source": source,
        "mechanism_hypothesis_count": len(mechanisms),
        "dominant_mechanism_object_id": str(top.get("object_id", "") or ""),
    }
    return mechanism_control, mechanisms, diagnostics


def mechanism_hypothesis_priority(
    action: Dict[str, Any],
    mechanisms: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(action, dict) or not mechanisms:
        return {
            "score": 0.0,
            "matched_object_id": "",
            "matched_family": "",
            "anchor_match": False,
            "supported_goal_anchor_match": False,
            "multi_anchor_support": False,
            "preferred_progress_mode": "",
        }
    target_desc = extract_target_descriptor(action)
    anchor_ref = str(target_desc.get("anchor_ref", "") or "")
    target_family = canonical_target_family(target_desc.get("target_family", "") or "")
    action_family = infer_action_family(action)
    best = {
        "score": 0.0,
        "matched_object_id": "",
        "matched_family": "",
        "anchor_match": False,
        "supported_goal_anchor_match": False,
        "multi_anchor_support": False,
        "preferred_progress_mode": "",
    }
    for row in list(mechanisms or []):
        if not isinstance(row, dict):
            continue
        content = dict(row.get("content", {})) if isinstance(row.get("content", {}), dict) else {}
        posterior = float(content.get("posterior", row.get("posterior", row.get("confidence", 0.0))) or 0.0)
        preferred_refs = {
            str(item or "")
            for item in list(content.get("preferred_target_refs", row.get("preferred_target_refs", [])) or [])
            if str(item or "")
        }
        supported_goal_refs = {
            str(item or "")
            for item in list(content.get("supported_goal_anchor_refs", row.get("supported_goal_anchor_refs", [])) or [])
            if str(item or "")
        }
        preferred_families = {
            normalize_action_family(item)
            for item in list(content.get("preferred_action_families", row.get("preferred_action_families", [])) or [])
            if normalize_action_family(item)
        }
        discriminating_families = {
            normalize_action_family(item)
            for item in list(content.get("best_discriminating_actions", row.get("best_discriminating_actions", [])) or [])
            if normalize_action_family(item)
        }
        anchor_match = bool(anchor_ref and (anchor_ref in preferred_refs or anchor_ref in supported_goal_refs))
        supported_goal_anchor_match = bool(anchor_ref and anchor_ref in supported_goal_refs)
        family_match = bool(action_family and action_family in preferred_families)
        discriminator_match = bool(action_family and action_family in discriminating_families)
        multi_anchor = bool(
            content.get("requires_multi_anchor_coordination", row.get("requires_multi_anchor_coordination", False))
        )
        progress_mode = str(
            content.get("preferred_progress_mode")
            or row.get("preferred_progress_mode")
            or ""
        )
        score = posterior * (
            (0.58 if anchor_match else 0.0)
            + (0.46 if discriminator_match else 0.0)
            + (0.28 if family_match else 0.0)
            + (0.24 if supported_goal_anchor_match and multi_anchor else 0.0)
            + (0.12 if supported_goal_anchor_match and progress_mode == "expand_anchor_coverage" else 0.0)
        )
        if score <= float(best["score"]):
            continue
        best = {
            "score": round(float(score), 6),
            "matched_object_id": str(row.get("object_id", "") or ""),
            "matched_family": str(content.get("family") or row.get("family") or ""),
            "anchor_match": bool(anchor_match),
            "supported_goal_anchor_match": bool(supported_goal_anchor_match),
            "multi_anchor_support": bool(multi_anchor and supported_goal_anchor_match),
            "preferred_progress_mode": progress_mode,
        }
    if target_family and not best["matched_family"]:
        best["matched_family"] = target_family
    return best

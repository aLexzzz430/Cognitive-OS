from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.orchestration.action_utils import extract_action_xy


def derive_action_effect_signature(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    result: Dict[str, Any],
    action: Dict[str, Any],
    information_gain: float,
    progress_markers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if loop._extract_action_function_name(action, default="") != "ACTION6":
        return {}
    click_point = extract_action_xy(action)
    if click_point is None:
        return {}

    before_descriptors = loop._surface_object_descriptors_from_obs(obs_before)
    after_descriptors = loop._surface_object_descriptors_from_obs(result)
    clicked_descriptor = loop._match_click_to_descriptor(before_descriptors, click_point)
    clicked_family = loop._family_summary_from_descriptor(clicked_descriptor, action)

    result_perception = (
        result.get("perception", {})
        if isinstance(result.get("perception", {}), dict)
        else {}
    )
    changed_bbox = (
        result.get("changed_bbox")
        if isinstance(result.get("changed_bbox"), dict)
        else result_perception.get("changed_bbox")
    )
    hotspot = (
        result.get("suggested_hotspot")
        if isinstance(result.get("suggested_hotspot"), dict)
        else result_perception.get("suggested_hotspot")
    )
    visual_feedback = loop._extract_visual_feedback(result)
    changed_pixels = float(visual_feedback.get("changed_pixel_count", 0.0) or 0.0)

    affected_descriptors = [
        descriptor
        for descriptor in after_descriptors
        if loop._descriptor_affected_by_visual_change(
            descriptor,
            changed_bbox,
            hotspot if isinstance(hotspot, dict) else None,
        )
    ]
    if not affected_descriptors and isinstance(hotspot, dict):
        hotspot_point = (
            loop._safe_int(hotspot.get("x")),
            loop._safe_int(hotspot.get("y")),
        )
        if hotspot_point[0] is not None and hotspot_point[1] is not None:
            matched = loop._match_click_to_descriptor(
                after_descriptors,
                (int(hotspot_point[0]), int(hotspot_point[1])),
            )
            if matched is not None:
                affected_descriptors = [matched]

    affected_families = [
        loop._family_summary_from_descriptor(descriptor, None)
        for descriptor in affected_descriptors
        if isinstance(descriptor, dict)
    ]
    same_family_score = 0.0
    other_family_score = 0.0
    other_family_candidates: List[Tuple[float, Dict[str, Any]]] = []
    for family in affected_families:
        match_score = loop._family_match_score(clicked_family, family)
        if match_score >= 0.60:
            same_family_score = max(same_family_score, match_score)
        else:
            alt_strength = max(0.0, 1.0 - min(1.0, match_score))
            if any(
                [
                    family.get("anchor_ref"),
                    family.get("color") is not None,
                    bool(family.get("shape_labels")),
                    family.get("target_family"),
                ]
            ):
                other_family_score = max(other_family_score, alt_strength)
                other_family_candidates.append((alt_strength, family))

    positive_progress = loop._progress_markers_show_positive_progress(progress_markers)
    preference = "neutral"
    if (
        same_family_score >= other_family_score + 0.18
        and (information_gain >= 0.10 or positive_progress or changed_pixels >= 8.0)
    ):
        preference = "same_family"
    elif (
        other_family_score >= same_family_score + 0.18
        and (information_gain >= 0.08 or changed_pixels >= 4.0)
    ):
        preference = "other_family"

    supported_families: List[Dict[str, Any]] = []
    if preference == "same_family" and any(
        [
            clicked_family.get("anchor_ref"),
            clicked_family.get("color") is not None,
            bool(clicked_family.get("shape_labels")),
            clicked_family.get("target_family"),
        ]
    ):
        supported_families.append(dict(clicked_family))
    elif preference == "other_family":
        seen_other = set()
        for _strength, family in sorted(
            other_family_candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            family_key = (
                str(family.get("anchor_ref", "") or ""),
                loop._safe_int(family.get("color")),
                tuple(
                    sorted(
                        str(item)
                        for item in list(family.get("shape_labels", []) or [])
                        if str(item)
                    )
                ),
                str(family.get("target_family", "") or ""),
            )
            if family_key in seen_other:
                continue
            seen_other.add(family_key)
            supported_families.append(dict(family))
            if len(supported_families) >= 3:
                break

    return {
        "clicked_family": clicked_family,
        "effect_signature": {
            "changed_pixel_count": round(changed_pixels, 4),
            "changed_bbox": dict(changed_bbox) if isinstance(changed_bbox, dict) else {},
            "hotspot": dict(hotspot) if isinstance(hotspot, dict) else {},
            "affected_anchor_refs": [
                str(family.get("anchor_ref", "") or "")
                for family in affected_families
                if str(family.get("anchor_ref", "") or "")
            ],
            "affected_colors": [
                int(family.get("color"))
                for family in affected_families
                if family.get("color") is not None
            ],
            "affected_shape_labels": sorted(
                {
                    str(label)
                    for family in affected_families
                    for label in list(family.get("shape_labels", []) or [])
                    if str(label)
                }
            ),
        },
        "family_effect_attribution": {
            "preference": preference,
            "same_family_score": round(float(same_family_score), 4),
            "other_family_score": round(float(other_family_score), 4),
            "clicked_family": clicked_family,
            "supported_families": supported_families,
            "changed_pixel_count": round(changed_pixels, 4),
            "information_gain": round(float(information_gain), 4),
            "positive_progress": bool(positive_progress),
        },
    }


def _entry_changed_pixel_count(entry: Dict[str, Any]) -> float:
    if not isinstance(entry, dict):
        return 0.0
    progress_markers = (
        entry.get("progress_markers", [])
        if isinstance(entry.get("progress_markers", []), list)
        else []
    )
    for marker in progress_markers:
        if not isinstance(marker, dict):
            continue
        if "changed_pixel_count" in marker:
            try:
                return max(0.0, float(marker.get("changed_pixel_count", 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _normalize_anchor_stats(
    totals: Dict[str, float],
    counts: Dict[str, int],
    maxima: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for anchor in sorted(set(totals) | set(counts) | set(maxima)):
        count = int(counts.get(anchor, 0) or 0)
        total = float(totals.get(anchor, 0.0) or 0.0)
        rows[anchor] = {
            "count": count,
            "total_changed_pixels": round(total, 4),
            "average_changed_pixels": round(total / float(max(count, 1)), 4),
            "max_changed_pixels": round(float(maxima.get(anchor, 0.0) or 0.0), 4),
        }
    return rows


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def recent_goal_progress_state(
    loop: Any,
    episode_trace: List[Dict[str, Any]],
    *,
    limit: int = 12,
) -> Dict[str, Any]:
    engaged_goal_anchor_refs = set()
    stalled_anchor_refs = set()
    successful_anchor_refs = set()
    necessary_anchor_refs = set()
    necessary_but_insufficient_anchor_refs = set()
    local_only_anchor_refs = set()
    controller_anchor_refs = set()
    controller_supported_goal_anchor_refs = set()
    controller_supported_goal_colors = set()
    anchor_visual_change_totals: Dict[str, float] = {}
    anchor_visual_change_counts: Dict[str, int] = {}
    anchor_visual_change_maxima: Dict[str, float] = {}
    controller_effect_counts: Dict[str, int] = {}
    controller_progress_counts: Dict[str, int] = {}
    anchor_click_counts: Dict[str, int] = {}
    anchor_local_signal_counts: Dict[str, int] = {}
    anchor_bundle_progress_counts: Dict[str, int] = {}
    anchor_stall_counts: Dict[str, int] = {}
    last_clicked_anchor_ref = ""
    active_combo_seed_anchor = ""
    processed = 0
    for entry in reversed(list(episode_trace or [])):
        if not isinstance(entry, dict):
            continue
        processed += 1
        assessment = (
            entry.get("goal_progress_assessment", {})
            if isinstance(entry.get("goal_progress_assessment", {}), dict)
            else {}
        )
        bundle_state = (
            entry.get("goal_bundle_state", {})
            if isinstance(entry.get("goal_bundle_state", {}), dict)
            else {}
        )
        if not active_combo_seed_anchor and str(bundle_state.get("active_combo_seed_anchor", "") or ""):
            active_combo_seed_anchor = str(bundle_state.get("active_combo_seed_anchor", "") or "")
        clicked_anchor_ref = str(assessment.get("clicked_anchor_ref", "") or "")
        if not clicked_anchor_ref:
            clicked_family = (
                entry.get("clicked_family", {})
                if isinstance(entry.get("clicked_family", {}), dict)
                else {}
            )
            clicked_anchor_ref = str(clicked_family.get("anchor_ref", "") or "")
        if clicked_anchor_ref:
            anchor_click_counts[clicked_anchor_ref] = (
                anchor_click_counts.get(clicked_anchor_ref, 0) + 1
            )
            changed_pixels = _entry_changed_pixel_count(entry)
            if changed_pixels > 0.0:
                anchor_visual_change_totals[clicked_anchor_ref] = (
                    anchor_visual_change_totals.get(clicked_anchor_ref, 0.0)
                    + changed_pixels
                )
                anchor_visual_change_counts[clicked_anchor_ref] = (
                    anchor_visual_change_counts.get(clicked_anchor_ref, 0) + 1
                )
                anchor_visual_change_maxima[clicked_anchor_ref] = max(
                    float(anchor_visual_change_maxima.get(clicked_anchor_ref, 0.0) or 0.0),
                    changed_pixels,
                )
            if not last_clicked_anchor_ref:
                last_clicked_anchor_ref = clicked_anchor_ref
            if loop._entry_has_local_anchor_signal(entry):
                anchor_local_signal_counts[clicked_anchor_ref] = (
                    anchor_local_signal_counts.get(clicked_anchor_ref, 0) + 1
                )
            if bool(assessment.get("progressed", False)):
                successful_anchor_refs.add(clicked_anchor_ref)
                anchor_bundle_progress_counts[clicked_anchor_ref] = (
                    anchor_bundle_progress_counts.get(clicked_anchor_ref, 0) + 1
                )
            if bool(assessment.get("stalled", False)):
                stalled_anchor_refs.add(clicked_anchor_ref)
                anchor_stall_counts[clicked_anchor_ref] = (
                    anchor_stall_counts.get(clicked_anchor_ref, 0) + 1
                )
            if bool(assessment.get("necessary_signal", False)) or anchor_local_signal_counts.get(
                clicked_anchor_ref,
                0,
            ) > 0:
                necessary_anchor_refs.add(clicked_anchor_ref)
            if bool(assessment.get("necessary_but_insufficient", False)):
                necessary_but_insufficient_anchor_refs.add(clicked_anchor_ref)
                if not active_combo_seed_anchor:
                    active_combo_seed_anchor = clicked_anchor_ref
        if bool(assessment.get("local_only_signal", False)):
            local_only_anchor_refs.add(clicked_anchor_ref)
        if bool(assessment.get("controller_effect", False)):
            controller_anchor_ref = str(
                assessment.get("controller_anchor_ref", "") or clicked_anchor_ref
            )
            if controller_anchor_ref:
                controller_anchor_refs.add(controller_anchor_ref)
                controller_effect_counts[controller_anchor_ref] = (
                    controller_effect_counts.get(controller_anchor_ref, 0) + 1
                )
                if bool(assessment.get("progressed", False)):
                    controller_progress_counts[controller_anchor_ref] = (
                        controller_progress_counts.get(controller_anchor_ref, 0) + 1
                    )
        for ref in list(assessment.get("controller_supported_goal_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                controller_supported_goal_anchor_refs.add(text)
        for color in list(assessment.get("controller_supported_goal_colors", []) or []):
            color_int = loop._safe_int(color)
            if color_int is not None:
                controller_supported_goal_colors.add(color_int)

        for ref in list(assessment.get("engaged_goal_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                engaged_goal_anchor_refs.add(text)
        for ref in list(bundle_state.get("engaged_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                engaged_goal_anchor_refs.add(text)
        for ref in list(bundle_state.get("necessary_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                necessary_anchor_refs.add(text)
        for ref in list(bundle_state.get("necessary_but_insufficient_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                necessary_but_insufficient_anchor_refs.add(text)
                if not active_combo_seed_anchor:
                    active_combo_seed_anchor = text
        for ref in list(bundle_state.get("local_only_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                local_only_anchor_refs.add(text)
        for ref in list(bundle_state.get("controller_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                controller_anchor_refs.add(text)
        for ref in list(bundle_state.get("controller_supported_goal_anchor_refs", []) or []):
            text = str(ref or "").strip()
            if text:
                controller_supported_goal_anchor_refs.add(text)
        for color in list(bundle_state.get("controller_supported_goal_colors", []) or []):
            color_int = loop._safe_int(color)
            if color_int is not None:
                controller_supported_goal_colors.add(color_int)
        if processed >= max(1, int(limit)):
            break
    return {
        "engaged_goal_anchor_refs": engaged_goal_anchor_refs,
        "stalled_anchor_refs": stalled_anchor_refs,
        "successful_anchor_refs": successful_anchor_refs,
        "necessary_anchor_refs": necessary_anchor_refs,
        "necessary_but_insufficient_anchor_refs": necessary_but_insufficient_anchor_refs,
        "local_only_anchor_refs": local_only_anchor_refs,
        "controller_anchor_refs": controller_anchor_refs,
        "controller_supported_goal_anchor_refs": controller_supported_goal_anchor_refs,
        "controller_supported_goal_colors": controller_supported_goal_colors,
        "anchor_visual_change_stats": _normalize_anchor_stats(
            anchor_visual_change_totals,
            anchor_visual_change_counts,
            anchor_visual_change_maxima,
        ),
        "controller_effect_counts": controller_effect_counts,
        "controller_progress_counts": controller_progress_counts,
        "anchor_click_counts": anchor_click_counts,
        "anchor_local_signal_counts": anchor_local_signal_counts,
        "anchor_bundle_progress_counts": anchor_bundle_progress_counts,
        "anchor_stall_counts": anchor_stall_counts,
        "last_clicked_anchor_ref": last_clicked_anchor_ref,
        "active_combo_seed_anchor": active_combo_seed_anchor,
    }


def derive_goal_progress_assessment(
    loop: Any,
    *,
    goal_summary: Dict[str, Any],
    effect_trace: Dict[str, Any],
    information_gain: float,
    progress_markers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(goal_summary, dict) or not goal_summary:
        return {}
    clicked_family = (
        effect_trace.get("clicked_family", {})
        if isinstance(effect_trace.get("clicked_family", {}), dict)
        else {}
    )
    effect_signature = (
        effect_trace.get("effect_signature", {})
        if isinstance(effect_trace.get("effect_signature", {}), dict)
        else {}
    )
    clicked_anchor_ref = str(clicked_family.get("anchor_ref", "") or "")
    goal_anchor_refs = {
        str(ref or "").strip()
        for ref in list(goal_summary.get("goal_anchor_refs", []) or [])
        if str(ref or "").strip()
    }
    goal_anchor_colors = {
        loop._safe_int(color)
        for color in list(goal_summary.get("goal_anchor_colors", []) or [])
        if loop._safe_int(color) is not None
    }
    affected_anchor_refs = {
        str(ref or "").strip()
        for ref in list(effect_signature.get("affected_anchor_refs", []) or [])
        if str(ref or "").strip()
    }
    affected_colors = {
        loop._safe_int(color)
        for color in list(effect_signature.get("affected_colors", []) or [])
        if loop._safe_int(color) is not None
    }
    current_changed_pixels = float(effect_signature.get("changed_pixel_count", 0.0) or 0.0)
    engaged_goal_anchor_refs = set(
        ref for ref in affected_anchor_refs if ref in goal_anchor_refs
    )
    if clicked_anchor_ref and clicked_anchor_ref in goal_anchor_refs:
        engaged_goal_anchor_refs.add(clicked_anchor_ref)
    color_goal_match = bool(goal_anchor_colors and bool(affected_colors & goal_anchor_colors))
    goal_aligned_effect = bool(engaged_goal_anchor_refs or color_goal_match)
    recent_state = loop._recent_goal_progress_state(list(loop._episode_trace), limit=8)
    prior_goal_anchor_refs = set(recent_state.get("engaged_goal_anchor_refs", set()) or set())
    stalled_anchor_refs = set(recent_state.get("stalled_anchor_refs", set()) or set())
    prior_anchor_click_counts = (
        recent_state.get("anchor_click_counts", {})
        if isinstance(recent_state.get("anchor_click_counts", {}), dict)
        else {}
    )
    prior_anchor_local_signal_counts = (
        recent_state.get("anchor_local_signal_counts", {})
        if isinstance(recent_state.get("anchor_local_signal_counts", {}), dict)
        else {}
    )
    prior_anchor_bundle_progress_counts = (
        recent_state.get("anchor_bundle_progress_counts", {})
        if isinstance(recent_state.get("anchor_bundle_progress_counts", {}), dict)
        else {}
    )
    prior_anchor_visual_change_stats = (
        recent_state.get("anchor_visual_change_stats", {})
        if isinstance(recent_state.get("anchor_visual_change_stats", {}), dict)
        else {}
    )
    prior_controller_effect_counts = (
        recent_state.get("controller_effect_counts", {})
        if isinstance(recent_state.get("controller_effect_counts", {}), dict)
        else {}
    )
    prior_controller_progress_counts = (
        recent_state.get("controller_progress_counts", {})
        if isinstance(recent_state.get("controller_progress_counts", {}), dict)
        else {}
    )
    prior_necessary_but_insufficient = set(
        recent_state.get("necessary_but_insufficient_anchor_refs", set()) or set()
    )
    novel_goal_anchor_refs = engaged_goal_anchor_refs - prior_goal_anchor_refs
    coverage_target = int(goal_summary.get("coverage_target", 0) or 0)
    if coverage_target <= 0:
        coverage_target = max(1, len(goal_anchor_refs) or 1)
    requires_multi_anchor_coordination = bool(
        goal_summary.get("requires_multi_anchor_coordination", False)
        or coverage_target > 1
    )
    coverage_before = min(len(prior_goal_anchor_refs & goal_anchor_refs), coverage_target)
    coverage_after = min(
        len((prior_goal_anchor_refs | engaged_goal_anchor_refs) & goal_anchor_refs),
        coverage_target,
    )
    coverage_delta = max(0, coverage_after - coverage_before)
    repeat_anchor_patience = max(1, int(goal_summary.get("repeat_anchor_patience", 1) or 1))
    same_anchor_streak = loop._recent_same_goal_anchor_streak(
        list(loop._episode_trace),
        clicked_anchor_ref,
    )
    repeat_overrun = 0
    if clicked_anchor_ref and clicked_anchor_ref in goal_anchor_refs and coverage_delta <= 0:
        repeat_overrun = max(0, same_anchor_streak + 1 - repeat_anchor_patience)
    goal_anchor_match = 0.0
    if goal_anchor_refs:
        goal_anchor_match = len(engaged_goal_anchor_refs) / float(max(len(goal_anchor_refs), 1))
    elif color_goal_match:
        goal_anchor_match = 0.55
    novelty_score = coverage_delta / float(max(coverage_target, 1))
    coverage_fraction = coverage_after / float(max(coverage_target, 1))
    structural_scope = len(engaged_goal_anchor_refs) / float(max(min(2, coverage_target), 1))
    positive_progress = loop._progress_markers_show_positive_progress(progress_markers)
    local_signal = bool(
        float(information_gain) >= 0.10
        or positive_progress
        or bool(engaged_goal_anchor_refs)
        or bool(color_goal_match)
    )
    controller_supported_goal_anchor_refs = set()
    controller_supported_goal_colors = set()
    controller_effect = bool(
        clicked_anchor_ref
        and clicked_anchor_ref not in goal_anchor_refs
        and goal_aligned_effect
        and (positive_progress or float(information_gain) >= 0.10)
    )
    if controller_effect:
        controller_supported_goal_anchor_refs = set(engaged_goal_anchor_refs)
        controller_supported_goal_colors = {
            color for color in affected_colors if color in goal_anchor_colors
        }
        controller_effect = bool(
            controller_supported_goal_anchor_refs or controller_supported_goal_colors
        )
    controller_anchor_ref = clicked_anchor_ref if controller_effect else ""
    prior_anchor_visual_stats = (
        prior_anchor_visual_change_stats.get(clicked_anchor_ref, {})
        if clicked_anchor_ref and isinstance(prior_anchor_visual_change_stats.get(clicked_anchor_ref, {}), dict)
        else {}
    )
    prior_average_changed_pixels = float(
        prior_anchor_visual_stats.get("average_changed_pixels", 0.0) or 0.0
    )
    prior_max_changed_pixels = float(
        prior_anchor_visual_stats.get("max_changed_pixels", 0.0) or 0.0
    )
    prior_controller_effect_count = (
        int(prior_controller_effect_counts.get(clicked_anchor_ref, 0) or 0)
        if clicked_anchor_ref
        else 0
    )
    prior_controller_progress_count = (
        int(prior_controller_progress_counts.get(clicked_anchor_ref, 0) or 0)
        if clicked_anchor_ref
        else 0
    )
    local_only_signal = bool(
        clicked_anchor_ref
        and local_signal
        and not goal_aligned_effect
        and coverage_delta <= 0
    )
    prior_local_signal_count = (
        int(prior_anchor_local_signal_counts.get(clicked_anchor_ref, 0) or 0)
        if clicked_anchor_ref
        else 0
    )
    prior_bundle_progress_count = (
        int(prior_anchor_bundle_progress_counts.get(clicked_anchor_ref, 0) or 0)
        if clicked_anchor_ref
        else 0
    )
    necessary_signal = bool(clicked_anchor_ref and local_signal)
    necessary_but_insufficient = bool(
        clicked_anchor_ref
        and requires_multi_anchor_coordination
        and local_signal
        and prior_local_signal_count >= 1
        and coverage_delta <= 0
        and (
            repeat_overrun > 0
            or (
                int(prior_anchor_click_counts.get(clicked_anchor_ref, 0) or 0) >= 1
                and prior_bundle_progress_count <= 0
            )
            or clicked_anchor_ref in prior_necessary_but_insufficient
        )
    )
    visual_change_signal = _clamp01(
        max(current_changed_pixels, prior_average_changed_pixels, prior_max_changed_pixels) / 256.0
    )
    state_relevance_score = (
        min(1.0, float(information_gain) / 0.35) * 0.20
        + min(1.0, goal_anchor_match) * 0.18
        + (0.18 if positive_progress else 0.0)
        + (0.12 if goal_aligned_effect else 0.0)
        + (0.12 if color_goal_match else 0.0)
        + visual_change_signal * 0.16
        - (0.12 if local_only_signal else 0.0)
    )
    state_relevance_score = _clamp01(state_relevance_score)
    controller_goal_support_fraction = max(
        (
            len(controller_supported_goal_anchor_refs)
            / float(max(coverage_target, 1))
        )
        if controller_supported_goal_anchor_refs
        else 0.0,
        (
            len(controller_supported_goal_colors & goal_anchor_colors)
            / float(max(len(goal_anchor_colors), 1))
        )
        if goal_anchor_colors and controller_supported_goal_colors
        else 0.0,
    )
    repeated_controller_support = _clamp01((prior_controller_effect_count + (1 if controller_effect else 0)) / 3.0)
    repeated_controller_progress = _clamp01((prior_controller_progress_count + (1 if positive_progress else 0)) / 3.0)
    controller_evidence_score = 0.0
    if controller_effect:
        controller_evidence_score = (
            controller_goal_support_fraction * 0.32
            + repeated_controller_support * 0.26
            + repeated_controller_progress * 0.18
            + visual_change_signal * 0.16
            + (0.08 if goal_aligned_effect else 0.0)
        )
    controller_evidence_score = _clamp01(controller_evidence_score)
    relation_hypotheses = [
        dict(item)
        for item in list(goal_summary.get("relation_hypotheses", []) or [])
        if isinstance(item, dict)
    ]
    top_relation = relation_hypotheses[0] if relation_hypotheses else {}
    relation_member_anchor_refs = {
        str(ref or "").strip()
        for ref in list(top_relation.get("member_anchor_refs", []) or [])
        if str(ref or "").strip()
    }
    relation_member_colors = {
        loop._safe_int(color)
        for color in list(top_relation.get("member_colors", []) or [])
        if loop._safe_int(color) is not None
    }
    clicked_color = loop._safe_int(clicked_family.get("color"))
    relation_engaged_anchor_refs = set()
    if clicked_anchor_ref and clicked_anchor_ref in relation_member_anchor_refs:
        relation_engaged_anchor_refs.add(clicked_anchor_ref)
    relation_engaged_anchor_refs |= affected_anchor_refs & relation_member_anchor_refs
    relation_color_match = bool(
        (clicked_color is not None and clicked_color in relation_member_colors)
        or bool(affected_colors & relation_member_colors)
    )
    relation_support_fraction = (
        len(relation_engaged_anchor_refs) / float(max(len(relation_member_anchor_refs), 1))
        if relation_member_anchor_refs
        else 0.0
    )
    relation_progress_score = 0.0
    if top_relation:
        relation_progress_score = (
            min(1.0, relation_support_fraction) * 0.34
            + visual_change_signal * 0.22
            + (0.16 if controller_effect else 0.0)
            + (0.12 if goal_aligned_effect else 0.0)
            + (0.10 if positive_progress else 0.0)
            + (0.08 if relation_color_match else 0.0)
            - (0.06 if local_only_signal else 0.0)
        )
    relation_progress_score = _clamp01(relation_progress_score)
    structural_goal_progress = bool(coverage_delta > 0)
    combo_goal_progress = bool(
        structural_goal_progress and requires_multi_anchor_coordination
    )
    relation_goal_progress = bool(
        top_relation and relation_progress_score >= 0.42 and not local_only_signal
    )
    goal_proximity_score = (
        min(1.0, coverage_fraction) * 0.42
        + novelty_score * 0.22
        + min(1.0, structural_scope) * 0.12
        + controller_evidence_score * 0.32
        + relation_progress_score * 0.24
        + (0.08 if combo_goal_progress else 0.0)
        - min(1.0, repeat_overrun / 2.0) * 0.18
        - (0.10 if necessary_but_insufficient else 0.0)
        - (0.08 if local_only_signal else 0.0)
    )
    goal_proximity_score = _clamp01(goal_proximity_score)
    score = (
        state_relevance_score * 0.52
        + goal_proximity_score * 0.48
        - min(1.0, repeat_overrun / 2.0) * 0.08
        - (0.06 if necessary_but_insufficient else 0.0)
    )
    score = max(0.0, min(1.0, score))
    progressed = bool(
        structural_goal_progress
        or relation_goal_progress
        or (goal_proximity_score >= 0.42 and not local_only_signal)
        or (score >= 0.56 and not local_only_signal)
    )
    stalled = bool(
        goal_anchor_refs
        and coverage_delta <= 0
        and repeat_overrun > 0
        and (
            clicked_anchor_ref in goal_anchor_refs
            or clicked_anchor_ref in stalled_anchor_refs
        )
    )
    if stalled and progressed and score < 0.68:
        progressed = False
    progress_class = "neutral"
    if combo_goal_progress:
        progress_class = "combo_goal_progress"
    elif structural_goal_progress:
        progress_class = "goal_coverage_progress"
    elif relation_goal_progress:
        progress_class = "relation_goal_progress"
    elif local_only_signal:
        progress_class = "local_only_reaction"
    elif goal_aligned_effect and local_signal:
        progress_class = "goal_aligned_reaction"
    trend = (
        "advanced"
        if progressed and not stalled
        else ("stalled" if stalled else ("local_only" if local_only_signal else "neutral"))
    )
    return {
        "goal_family": str(goal_summary.get("goal_family", "") or ""),
        "goal_confidence": float(goal_summary.get("confidence", 0.0) or 0.0),
        "clicked_anchor_ref": clicked_anchor_ref,
        "engaged_goal_anchor_refs": sorted(engaged_goal_anchor_refs),
        "novel_goal_anchor_refs": sorted(novel_goal_anchor_refs),
        "goal_coverage_before": int(coverage_before),
        "goal_coverage_after": int(coverage_after),
        "goal_coverage_delta": int(coverage_delta),
        "state_relevance_score": round(float(state_relevance_score), 4),
        "goal_proximity_score": round(float(goal_proximity_score), 4),
        "controller_evidence_score": round(float(controller_evidence_score), 4),
        "goal_progress_score": round(float(score), 4),
        "goal_distance_estimate": round(max(0.0, 1.0 - max(coverage_fraction, goal_proximity_score)), 4),
        "goal_anchor_match": round(float(goal_anchor_match), 4),
        "color_goal_match": bool(color_goal_match),
        "goal_aligned_effect": bool(goal_aligned_effect),
        "positive_progress": bool(positive_progress),
        "controller_effect": bool(controller_effect),
        "controller_anchor_ref": controller_anchor_ref,
        "controller_supported_goal_anchor_refs": sorted(
            controller_supported_goal_anchor_refs
        ),
        "controller_supported_goal_colors": sorted(controller_supported_goal_colors),
        "relation_progress_score": round(float(relation_progress_score), 4),
        "relation_goal_progress": bool(relation_goal_progress),
        "relation_type": str(top_relation.get("relation_type", "") or ""),
        "relation_target": str(top_relation.get("target_relation", "") or ""),
        "relation_grouping_basis": str(top_relation.get("grouping_basis", "") or ""),
        "relation_member_anchor_refs": sorted(relation_member_anchor_refs),
        "relation_engaged_anchor_refs": sorted(relation_engaged_anchor_refs),
        "relation_color_match": bool(relation_color_match),
        "structural_goal_progress": bool(structural_goal_progress),
        "combo_goal_progress": bool(combo_goal_progress),
        "local_only_signal": bool(local_only_signal),
        "necessary_signal": bool(necessary_signal),
        "necessary_but_insufficient": bool(necessary_but_insufficient),
        "repeat_anchor_streak": int(same_anchor_streak + 1 if clicked_anchor_ref else 0),
        "repeat_anchor_overrun": int(repeat_overrun),
        "progressed": bool(progressed),
        "stalled": bool(stalled),
        "progress_class": progress_class,
        "trend": trend,
    }

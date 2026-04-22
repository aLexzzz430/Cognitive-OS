
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text_tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9_]{2,}", str(value or "").lower())


def _normalize_relation_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"aligned", "alignment", "group_alignment", "align"}:
        return "alignment"
    if text in {"pair", "pairing", "paired"}:
        return "pairing"
    if text in {"contain", "contained", "containment"}:
        return "containment"
    if text in {"order", "ordered", "ordering", "sequence"}:
        return "ordering"
    if text in {"count", "count_match", "matching_count"}:
        return "count_match"
    if text in {"symmetry", "symmetric", "mirror"}:
        return "symmetry"
    if text in {"state_change", "state_transition", "activation", "reveal"}:
        return "state_change"
    return text


def _token_overlap_ratio(left: Any, right: Any) -> float:
    left_tokens = set(_text_tokens(left))
    right_tokens = set(_text_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _action_names(obs: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ('available_functions', 'available_action_names', 'visible_functions'):
        raw = obs.get(key, [])
        if isinstance(raw, list):
            for item in raw:
                text = str(item or '').strip()
                if text and text not in names:
                    names.append(text)
    novel_api = obs.get('novel_api', {})
    if hasattr(novel_api, 'raw'):
        novel_api = novel_api.raw
    if isinstance(novel_api, dict):
        for key in ('available_functions', 'visible_functions', 'discovered_functions'):
            raw = novel_api.get(key, [])
            if isinstance(raw, list):
                for item in raw:
                    text = str(item or '').strip()
                    if text and text not in names:
                        names.append(text)
    return names


def _interaction_mode_scores(action_names: Sequence[str], perception: Dict[str, Any], world_scene: Dict[str, Any]) -> Dict[str, float]:
    names = {str(name or '').strip().upper() for name in action_names if str(name or '').strip()}
    pointer = 0.0
    navigation = 0.0
    confirm = 0.0
    transform = 0.0

    if 'ACTION6' in names or 'CLICK' in names or 'POINTER_CLICK' in names:
        pointer += 0.65
    if {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'} & names:
        navigation += 0.55
    if 'ACTION5' in names or 'CONFIRM' in names or 'SUBMIT' in names or 'INTERACT' in names:
        confirm += 0.45

    if int(world_scene.get('entity_count', 0) or 0) >= 1:
        pointer += 0.10
        navigation += 0.05
    if int(world_scene.get('relation_count', 0) or 0) >= 2:
        transform += 0.20
    if int(world_scene.get('distinct_color_count', len(world_scene.get('distinct_colors', [])) if isinstance(world_scene.get('distinct_colors', []), list) else 0) or 0) >= 4:
        transform += 0.15
    if int(world_scene.get('scene_element_count', 0) or 0) >= 2:
        pointer += 0.08
    if int(world_scene.get('resource_like_element_count', 0) or 0) >= 1:
        pointer += 0.10
    if str(world_scene.get('layout_mode', '') or '').startswith('boundary_scaffold'):
        pointer += 0.12
        navigation += 0.02

    hotspot = perception.get('suggested_hotspot')
    if isinstance(hotspot, dict) and any(hotspot.get(key) is not None for key in ('x', 'y', 'row', 'col')):
        pointer += 0.10

    return {
        'pointer_interaction': round(min(1.0, pointer), 4),
        'navigation_interaction': round(min(1.0, navigation), 4),
        'confirm_interaction': round(min(1.0, confirm), 4),
        'state_transform_interaction': round(min(1.0, transform), 4),
    }


def _goal_family_candidates(
    mode_scores: Dict[str, float],
    world_scene: Dict[str, Any],
    object_bindings_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    object_count = int(object_bindings_summary.get('object_count', 0) or 0)
    salient_count = len(_as_list(object_bindings_summary.get('salient_object_ids', [])))
    relation_count = int(world_scene.get('relation_count', 0) or 0)
    candidates: List[Dict[str, Any]] = []

    if mode_scores.get('pointer_interaction', 0.0) >= 0.35:
        confidence = min(0.95, 0.42 + mode_scores['pointer_interaction'] * 0.35 + min(0.12, salient_count * 0.04))
        candidates.append({
            'family': 'select_or_activate_salient_structures',
            'confidence': round(confidence, 4),
            'reason': 'pointer-dominant interaction with actionable objects',
        })
        candidates.append({
            'family': 'reveal_hidden_state_via_probe',
            'confidence': round(min(0.9, 0.28 + mode_scores['pointer_interaction'] * 0.22 + (0.08 if relation_count <= 1 else 0.0)), 4),
            'reason': 'pointer actions may expose hidden state or trigger mode transitions',
        })

    if mode_scores.get('navigation_interaction', 0.0) >= 0.30:
        candidates.append({
            'family': 'navigate_agent_or_focus_to_goal',
            'confidence': round(min(0.9, 0.32 + mode_scores['navigation_interaction'] * 0.36 + (0.08 if object_count >= 1 else 0.0)), 4),
            'reason': 'navigation primitives are available and likely change controllable focus',
        })

    if mode_scores.get('confirm_interaction', 0.0) >= 0.20:
        candidates.append({
            'family': 'commit_or_confirm_world_state',
            'confidence': round(min(0.85, 0.24 + mode_scores['confirm_interaction'] * 0.42), 4),
            'reason': 'confirm-like action may seal or submit an assembled state',
        })

    if object_count >= 2 or relation_count >= 2 or mode_scores.get('state_transform_interaction', 0.0) >= 0.25:
        candidates.append({
            'family': 'arrange_or_transform_object_configuration',
            'confidence': round(min(0.9, 0.25 + relation_count * 0.06 + object_count * 0.03 + mode_scores.get('state_transform_interaction', 0.0) * 0.25), 4),
            'reason': 'multiple objects/relations suggest structure-level transformation goals',
        })

    candidates.sort(key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), str(item.get('family', '') or '')))
    return candidates[:4]


def _top_objects(object_bindings_summary: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    rows = [item for item in _as_list(object_bindings_summary.get('objects', [])) if isinstance(item, dict)]
    rows.sort(
        key=lambda item: (
            -float(item.get('actionable_score', 0.0) or 0.0),
            -float(item.get('salience_score', 0.0) or 0.0),
            str(item.get('object_id', '') or ''),
        )
    )
    return rows[:max(1, int(limit))]


def _primary_semantic_label(obj: Dict[str, Any]) -> str:
    semantic_candidates = _as_list(obj.get('semantic_candidates', []))
    if semantic_candidates and isinstance(semantic_candidates[0], dict):
        return str(semantic_candidates[0].get('label', '') or '')
    return ''


def _primary_role_label(obj: Dict[str, Any]) -> str:
    role_candidates = _as_list(obj.get('role_candidates', []))
    if role_candidates and isinstance(role_candidates[0], dict):
        return str(role_candidates[0].get('role', '') or '')
    return ''


def _goal_anchor_object_score(obj: Dict[str, Any]) -> float:
    features = _as_dict(obj.get('geometric_features', {}))
    area = float(features.get('area', 0.0) or 0.0)
    rarity = float(features.get('rarity_score', 0.0) or 0.0)
    changed_overlap = float(features.get('changed_overlap', 0.0) or 0.0)
    boundary_contact = bool(features.get('boundary_contact', False))
    goal_like = bool(features.get('goal_like', False))
    actionable = float(obj.get('actionable_score', 0.0) or 0.0)
    salience = float(obj.get('salience_score', 0.0) or 0.0)
    top_semantic = _primary_semantic_label(obj)
    top_role = _primary_role_label(obj)

    score = actionable * 1.05 + salience * 0.72
    score += min(1.0, area / 32.0) * 0.32
    score += min(1.0, rarity) * 0.34
    score += min(1.0, changed_overlap / 8.0) * 0.08
    if goal_like:
        score += 0.18
    if top_role == 'scene_anchor':
        score += 0.08
    if top_semantic in {'block_like', 'token_like', 'generic_object', 'highly_symmetric_structure'}:
        score += 0.06
    if top_semantic in {'boundary_structure', 'bar_like'}:
        score -= 0.18
    if boundary_contact:
        score -= 0.16
    if boundary_contact and area <= 8:
        score -= 0.24
    return round(score, 4)


def _cluster_element_score(element: Dict[str, Any]) -> float:
    bbox = _as_dict(element.get('bbox', {}))
    attrs = _as_dict(element.get('attributes', {}))
    row_min = int(bbox.get('row_min', 0) or 0)
    col_min = int(bbox.get('col_min', 0) or 0)
    width = int(bbox.get('width', 0) or 0)
    height = int(bbox.get('height', 0) or 0)
    mean_actionable = float(attrs.get('mean_actionable_score', 0.0) or 0.0)
    cluster_size = int(attrs.get('cluster_size', 0) or 0)
    score = float(element.get('confidence', 0.0) or 0.0) * 0.9
    score += min(1.0, mean_actionable) * 0.35
    score += min(1.0, cluster_size / 8.0) * 0.08
    score += min(1.0, max(width, height) / 24.0) * 0.06
    if row_min > 0 and col_min > 0:
        score += 0.08
    if row_min == 0 or col_min == 0:
        score -= 0.08
    return round(score, 4)


def _recent_controller_support(
    episode_trace_tail: Sequence[Dict[str, Any]] | None,
) -> Dict[str, List[Any]]:
    trace_tail = [row for row in _as_list(episode_trace_tail) if isinstance(row, dict)]
    controller_anchor_refs: List[str] = []
    controller_supported_goal_anchor_refs: List[str] = []
    controller_supported_goal_colors: List[int] = []

    def _append_ref(target: List[str], value: Any) -> None:
        text = str(value or '').strip()
        if text and text not in target:
            target.append(text)

    def _append_color(target: List[int], value: Any) -> None:
        try:
            color_int = int(value)
        except Exception:
            color_int = None
        if color_int is not None and color_int not in target:
            target.append(color_int)

    for entry in reversed(trace_tail):
        assessment = _as_dict(entry.get('goal_progress_assessment', {}))
        if not assessment:
            continue
        if bool(assessment.get('controller_effect', False)):
            _append_ref(
                controller_anchor_refs,
                assessment.get('controller_anchor_ref') or assessment.get('clicked_anchor_ref'),
            )
        for ref in _as_list(assessment.get('controller_supported_goal_anchor_refs', [])):
            _append_ref(controller_supported_goal_anchor_refs, ref)
        for ref in _as_list(_as_dict(entry.get('goal_bundle_state', {})).get('controller_anchor_refs', [])):
            _append_ref(controller_anchor_refs, ref)
        for ref in _as_list(_as_dict(entry.get('goal_bundle_state', {})).get('controller_supported_goal_anchor_refs', [])):
            _append_ref(controller_supported_goal_anchor_refs, ref)
        for color in _as_list(assessment.get('controller_supported_goal_colors', [])):
            _append_color(controller_supported_goal_colors, color)
        for color in _as_list(_as_dict(entry.get('goal_bundle_state', {})).get('controller_supported_goal_colors', [])):
            _append_color(controller_supported_goal_colors, color)

    return {
        'controller_anchor_refs': controller_anchor_refs[:4],
        'controller_supported_goal_anchor_refs': controller_supported_goal_anchor_refs[:6],
        'controller_supported_goal_colors': controller_supported_goal_colors[:6],
    }


def _summarize_mechanism_priors(
    goal_family: str,
    world_model_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    summary = _as_dict(world_model_summary)
    rows = [
        dict(item)
        for item in _as_list(summary.get('mechanism_priors', []))
        if isinstance(item, dict)
    ]
    mechanism_hypothesis_rows = [
        dict(item)
        for item in _as_list(summary.get('mechanism_hypothesis_objects', summary.get('mechanism_hypotheses_summary', [])))
        if isinstance(item, dict)
    ]
    matched: List[Dict[str, Any]] = []
    seen_object_ids = set()
    for row in rows:
        content = _as_dict(row.get('content', {}))
        row_goal_family = str(
            content.get('goal_family')
            or row.get('goal_family')
            or ''
        ).strip()
        mechanism_kind = str(
            content.get('mechanism_kind')
            or row.get('mechanism_kind')
            or ''
        ).strip()
        if row_goal_family and row_goal_family != goal_family:
            continue
        if mechanism_kind and mechanism_kind != 'controller_support':
            continue
        matched.append(row)
        object_id = str(row.get('object_id', '') or '').strip()
        if object_id:
            seen_object_ids.add(object_id)

    mechanism_family_goal_map = {
        'directional_or_salience_activation': 'select_or_activate_salient_structures',
        'salience_probe': 'reveal_hidden_state_via_probe',
        'generic_probe_then_update': 'reveal_hidden_state_via_probe',
        'navigate_then_commit': 'commit_or_confirm_world_state',
        'reveal_then_commit': 'commit_or_confirm_world_state',
        'ordered_or_relational_transform': 'arrange_or_transform_object_configuration',
    }
    for row in mechanism_hypothesis_rows:
        content = _as_dict(row.get('content', {}))
        metadata = _as_dict(content.get('metadata', row.get('metadata', {})))
        explicit_goal_family = str(
            content.get('goal_family')
            or row.get('goal_family')
            or metadata.get('goal_family')
            or ''
        ).strip()
        mapped_goal_family = mechanism_family_goal_map.get(
            str(content.get('family') or row.get('family') or '').strip(),
            '',
        )
        row_goal_family = explicit_goal_family or mapped_goal_family
        if explicit_goal_family and explicit_goal_family != goal_family:
            continue
        if (
            not explicit_goal_family
            and mapped_goal_family
            and goal_family not in {'', 'generic_progress_discovery', mapped_goal_family}
        ):
            continue
        object_id = str(
            row.get('object_id')
            or row.get('hypothesis_id')
            or content.get('object_id')
            or ''
        ).strip()
        if object_id and object_id in seen_object_ids:
            continue
        matched.append(row)
        if object_id:
            seen_object_ids.add(object_id)

    if not matched:
        return {}

    preferred_progress_mode = ''
    supporting_functions: List[str] = []
    controller_anchor_refs: List[str] = []
    supported_goal_anchor_refs: List[str] = []
    supported_goal_colors: List[int] = []
    max_supported_goal_count = 0
    max_confidence = 0.0
    requires_multi_anchor_coordination = False

    for row in matched:
        content = _as_dict(row.get('content', {}))
        metadata = _as_dict(content.get('metadata', row.get('metadata', {})))
        max_confidence = max(
            max_confidence,
            float(content.get('posterior', row.get('posterior', row.get('confidence', 0.0))) or 0.0),
        )
        fallback_anchor_refs = [
            str(ref or '').strip()
            for ref in _as_list(content.get('preferred_target_refs', row.get('preferred_target_refs', [])))
            if str(ref or '').strip()
        ]
        supported_goal_count = int(
            content.get(
                'supported_goal_count',
                row.get('supported_goal_count', len(fallback_anchor_refs)),
            ) or len(fallback_anchor_refs)
        )
        if supported_goal_count > max_supported_goal_count:
            max_supported_goal_count = supported_goal_count
        mode = str(
            content.get('preferred_progress_mode')
            or row.get('preferred_progress_mode')
            or ('expand_anchor_coverage' if supported_goal_count > 1 else '')
        ).strip()
        if not preferred_progress_mode and mode:
            preferred_progress_mode = mode
        requires_multi_anchor_coordination = bool(
            requires_multi_anchor_coordination
            or content.get('requires_multi_anchor_coordination', row.get('requires_multi_anchor_coordination', False))
            or supported_goal_count > 1
        )
        supporting_function_values: List[str] = []
        for candidate in (
            content.get('action_function'),
            row.get('action_function'),
        ):
            text = str(candidate or '').strip()
            if text:
                supporting_function_values.append(text)
        supporting_function_values.extend(
            [
                str(item or '').strip()
                for item in _as_list(content.get('supporting_functions', row.get('supporting_functions', [])))
                if str(item or '').strip()
            ]
        )
        supporting_function_values.extend(
            [
                str(item or '').strip()
                for item in _as_list(metadata.get('predicted_function_names', []))
                if str(item or '').strip()
            ]
        )
        predictions = _as_dict(content.get('predictions', row.get('predictions', {})))
        predicted_action_effects = _as_dict(predictions.get('predicted_action_effects', {}))
        supporting_function_values.extend(
            [
                str(item or '').strip()
                for item in list(predicted_action_effects.keys())
                if str(item or '').strip()
            ]
        )
        for function_name in supporting_function_values:
            if function_name and function_name not in supporting_functions:
                supporting_functions.append(function_name)
        for ref in _as_list(content.get('controller_anchor_refs', row.get('controller_anchor_refs', []))):
            text = str(ref or '').strip()
            if text and text not in controller_anchor_refs:
                controller_anchor_refs.append(text)
        for ref in _as_list(
            content.get(
                'supported_goal_anchor_refs',
                row.get('supported_goal_anchor_refs', fallback_anchor_refs),
            )
        ):
            text = str(ref or '').strip()
            if text and text not in supported_goal_anchor_refs:
                supported_goal_anchor_refs.append(text)
        for color in _as_list(content.get('supported_goal_colors', row.get('supported_goal_colors', []))):
            try:
                color_int = int(color)
            except Exception:
                color_int = None
            if color_int is not None and color_int not in supported_goal_colors:
                supported_goal_colors.append(color_int)

    recommended_coverage_target = max(
        2 if requires_multi_anchor_coordination else 1,
        max_supported_goal_count,
    )

    return {
        'count': len(matched),
        'confidence': round(min(0.98, max_confidence), 4),
        'controller_support_expected': True,
        'preferred_progress_mode': preferred_progress_mode or 'expand_anchor_coverage',
        'recommended_coverage_target': int(max(1, min(4, recommended_coverage_target))),
        'requires_multi_anchor_coordination': bool(requires_multi_anchor_coordination),
        'supporting_functions': supporting_functions[:4],
        'controller_anchor_refs': controller_anchor_refs[:6],
        'supported_goal_anchor_refs': supported_goal_anchor_refs[:6],
        'supported_goal_colors': supported_goal_colors[:6],
        'object_ids': [
            str(row.get('object_id', row.get('hypothesis_id', '')) or '')
            for row in matched
            if str(row.get('object_id', row.get('hypothesis_id', '')) or '')
        ][:6],
    }


def _summarize_initial_goal_priors(
    world_model_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    summary = _as_dict(world_model_summary)
    rows = [
        dict(item)
        for item in _as_list(summary.get('initial_goal_priors', []))
        if isinstance(item, dict)
    ]
    if not rows:
        return {}

    target_relation_votes: Dict[str, float] = {}
    target_group_votes: Dict[str, float] = {}
    supporting_functions: List[str] = []
    max_confidence = 0.0
    top_summary = ''
    for row in rows:
        payload = _as_dict(row.get('goal_prior_payload', {}))
        confidence = float(row.get('confidence', 0.0) or 0.0)
        max_confidence = max(max_confidence, confidence)
        top_summary = top_summary or str(row.get('summary', '') or '')
        target_relation = str(payload.get('target_relation', '') or '').strip().lower()
        target_group = str(payload.get('target_group', '') or '').strip()
        if target_relation:
            target_relation_votes[target_relation] = target_relation_votes.get(target_relation, 0.0) + max(0.1, confidence)
        if target_group:
            target_group_votes[target_group] = target_group_votes.get(target_group, 0.0) + max(0.1, confidence)
        for fn_name in _as_list(row.get('target_functions', [])):
            text = str(fn_name or '').strip()
            if text and text not in supporting_functions:
                supporting_functions.append(text)

    top_relation = ''
    if target_relation_votes:
        top_relation = max(
            target_relation_votes.items(),
            key=lambda item: (float(item[1] or 0.0), str(item[0] or '')),
        )[0]
    top_group = ''
    if target_group_votes:
        top_group = max(
            target_group_votes.items(),
            key=lambda item: (float(item[1] or 0.0), str(item[0] or '')),
        )[0]

    return {
        'count': len(rows),
        'confidence': round(min(0.98, max_confidence), 4),
        'top_relation': top_relation,
        'top_group': top_group,
        'supporting_functions': supporting_functions[:4],
        'top_summary': top_summary,
    }


def _apply_initial_goal_prior_bias(
    goal_candidates: Sequence[Dict[str, Any]],
    initial_goal_prior_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates = [dict(row) for row in goal_candidates if isinstance(row, dict)]
    if not candidates or not initial_goal_prior_summary:
        return candidates
    top_relation = str(initial_goal_prior_summary.get('top_relation', '') or '').strip().lower()
    top_confidence = float(initial_goal_prior_summary.get('confidence', 0.0) or 0.0)
    if not top_relation or top_confidence <= 0.0:
        return candidates

    relation_goal_families = {
        'alignment',
        'pairing',
        'symmetry',
        'containment',
        'ordering',
        'count_match',
    }
    state_goal_families = {
        'state_change',
        'state_transition',
        'activation',
        'reveal',
    }
    relation_bias = min(0.09, max(0.02, top_confidence * 0.12))
    state_bias = min(0.08, max(0.02, top_confidence * 0.1))
    adjusted: List[Dict[str, Any]] = []
    for row in candidates:
        family = str(row.get('family', '') or '')
        confidence = float(row.get('confidence', 0.0) or 0.0)
        if top_relation in relation_goal_families and family == 'arrange_or_transform_object_configuration':
            confidence += relation_bias
        elif top_relation in state_goal_families and family in {'select_or_activate_salient_structures', 'reveal_hidden_state_via_probe'}:
            confidence += state_bias
        row['confidence'] = round(min(0.98, max(0.0, confidence)), 4)
        adjusted.append(row)
    adjusted.sort(key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), str(item.get('family', '') or '')))
    return adjusted[:4]


def _rank_goal_anchor_candidates(
    object_bindings_summary: Dict[str, Any],
    world_scene: Dict[str, Any],
    *,
    goal_family: str,
) -> List[Dict[str, Any]]:
    objects = [item for item in _as_list(object_bindings_summary.get('objects', [])) if isinstance(item, dict)]
    object_by_id = {
        str(item.get('object_id', '') or ''): item
        for item in objects
        if str(item.get('object_id', '') or '')
    }
    ranked: List[Dict[str, Any]] = []
    seen = set()

    def _add_object(ref: str, *, source: str) -> None:
        text = str(ref or '').strip()
        if not text or text in seen or text not in object_by_id:
            return
        seen.add(text)
        obj = object_by_id[text]
        ranked.append({
            'object_id': text,
            'source': source,
            'score': _goal_anchor_object_score(obj),
            'object': obj,
        })

    scene_elements = [item for item in _as_list(object_bindings_summary.get('scene_elements', [])) if isinstance(item, dict)]
    cluster_elements = [
        item for item in scene_elements
        if str(item.get('role', '') or '') == 'interaction_cluster'
    ]
    cluster_elements.sort(
        key=lambda item: (
            -_cluster_element_score(item),
            str(item.get('element_id', '') or ''),
        )
    )

    if goal_family in {'select_or_activate_salient_structures', 'arrange_or_transform_object_configuration'}:
        for element in cluster_elements[:2]:
            member_rows = []
            for ref in list(element.get('member_refs', []) or []):
                text = str(ref or '').strip()
                if text and text in object_by_id:
                    member_rows.append(object_by_id[text])
            member_rows.sort(
                key=lambda item: (
                    -_goal_anchor_object_score(item),
                    str(item.get('object_id', '') or ''),
                )
            )
            for row in member_rows[:3]:
                _add_object(str(row.get('object_id', '') or ''), source='interaction_cluster')

    top_objects = sorted(
        objects,
        key=lambda item: (
            -_goal_anchor_object_score(item),
            str(item.get('object_id', '') or ''),
        )
    )
    for row in top_objects[:8]:
        _add_object(str(row.get('object_id', '') or ''), source='top_object')

    ranked.sort(
        key=lambda item: (
            -float(item.get('score', 0.0) or 0.0),
            str(item.get('source', '') or ''),
            str(item.get('object_id', '') or ''),
        )
    )
    return ranked


def _infer_relation_hypotheses(
    anchor_refs: Sequence[str],
    object_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anchor_list = [
        str(ref or '').strip()
        for ref in list(anchor_refs or [])
        if str(ref or '').strip() and str(ref or '').strip() in object_by_id
    ]
    if len(anchor_list) < 2:
        return []

    hypotheses: List[Dict[str, Any]] = []
    seen_keys = set()

    def _descriptor_stats(member_refs: Sequence[str]) -> Dict[str, Any]:
        xs: List[float] = []
        ys: List[float] = []
        colors: List[int] = []
        shape_labels: List[str] = []
        for ref in member_refs:
            obj = object_by_id.get(ref, {})
            centroid = _as_dict(obj.get('centroid', {}))
            bbox = _as_dict(obj.get('bbox', {}))
            x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
            x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
            y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
            y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
            xs.append(float(centroid.get('x', (x_min + x_max) / 2.0) or 0.0))
            ys.append(float(centroid.get('y', (y_min + y_max) / 2.0) or 0.0))
            try:
                color_int = int(obj.get('color'))
            except Exception:
                color_int = None
            if color_int is not None and color_int not in colors:
                colors.append(color_int)
            label = _primary_semantic_label(obj)
            if label and label not in shape_labels:
                shape_labels.append(label)
        x_span = (max(xs) - min(xs)) if xs else 0.0
        y_span = (max(ys) - min(ys)) if ys else 0.0
        return {
            'member_colors': colors[:6],
            'member_shape_labels': shape_labels[:6],
            'axis_spread': {
                'x': round(float(x_span), 4),
                'y': round(float(y_span), 4),
            },
            'separation': round(float(max(x_span, y_span)), 4),
        }

    def _add_hypothesis(
        member_refs: Sequence[str],
        *,
        grouping_basis: str,
        grouping_value: str,
    ) -> None:
        ordered_refs = [
            ref for ref in anchor_list
            if ref in {str(item or '').strip() for item in list(member_refs or []) if str(item or '').strip()}
        ]
        if len(ordered_refs) < 2:
            return
        hypothesis_key = (grouping_basis, tuple(ordered_refs))
        if hypothesis_key in seen_keys:
            return
        seen_keys.add(hypothesis_key)
        stats = _descriptor_stats(ordered_refs)
        same_shape_bonus = 0.0
        if len(stats['member_shape_labels']) == 1 and stats['member_shape_labels']:
            same_shape_bonus = 0.06
        confidence = (
            0.34
            + min(0.18, (len(ordered_refs) - 1) * 0.08)
            + min(0.16, float(stats['separation']) / 32.0 * 0.16)
            + same_shape_bonus
        )
        if grouping_basis == 'color':
            confidence += 0.04
        hypotheses.append({
            'relation_type': 'group_alignment',
            'target_relation': 'aligned',
            'grouping_basis': str(grouping_basis or ''),
            'grouping_value': str(grouping_value or ''),
            'member_anchor_refs': ordered_refs[:4],
            'member_colors': list(stats['member_colors']),
            'member_shape_labels': list(stats['member_shape_labels']),
            'axis_spread': dict(stats['axis_spread']),
            'confidence': round(min(0.95, max(0.0, confidence)), 4),
        })

    color_groups: Dict[int, List[str]] = {}
    for ref in anchor_list:
        obj = object_by_id.get(ref, {})
        try:
            color_int = int(obj.get('color'))
        except Exception:
            color_int = None
        if color_int is None:
            continue
        color_groups.setdefault(color_int, []).append(ref)
    for color_int, refs in color_groups.items():
        if len(refs) >= 2:
            _add_hypothesis(refs, grouping_basis='color', grouping_value=str(color_int))

    shape_groups: Dict[str, List[str]] = {}
    for ref in anchor_list:
        label = _primary_semantic_label(object_by_id.get(ref, {}))
        if not label or label in {'generic_object', 'boundary_structure'}:
            continue
        shape_groups.setdefault(label, []).append(ref)
    for label, refs in shape_groups.items():
        if len(refs) >= 2:
            _add_hypothesis(refs, grouping_basis='shape', grouping_value=label)

    hypotheses.sort(
        key=lambda item: (
            0 if str(item.get('grouping_basis', '') or '') == 'color' else 1,
            -float(item.get('confidence', 0.0) or 0.0),
            -len(list(item.get('member_anchor_refs', []) or [])),
            str(item.get('grouping_value', '') or ''),
        )
    )
    return hypotheses[:4]


def _infer_level_goal(
    goal_candidates: Sequence[Dict[str, Any]],
    object_bindings_summary: Dict[str, Any],
    world_scene: Dict[str, Any],
    world_model_summary: Dict[str, Any] | None = None,
    episode_trace_tail: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    candidates = [row for row in goal_candidates if isinstance(row, dict)]
    top = candidates[0] if candidates else {}
    goal_family = str(top.get('family', '') or 'generic_progress_discovery')
    confidence = float(top.get('confidence', 0.0) or 0.0)
    objects = [item for item in _as_list(object_bindings_summary.get('objects', [])) if isinstance(item, dict)]
    object_by_id = {
        str(item.get('object_id', '') or ''): item
        for item in objects
        if str(item.get('object_id', '') or '')
    }
    scene_elements = [item for item in _as_list(object_bindings_summary.get('scene_elements', [])) if isinstance(item, dict)]

    anchor_refs: List[str] = []

    def _append_anchor(ref: Any) -> None:
        text = str(ref or '').strip()
        if text and text not in anchor_refs:
            anchor_refs.append(text)

    ranked_anchor_candidates = _rank_goal_anchor_candidates(
        object_bindings_summary,
        world_scene,
        goal_family=goal_family,
    )
    for candidate in ranked_anchor_candidates[:8]:
        _append_anchor(candidate.get('object_id'))

    if goal_family == 'navigate_agent_or_focus_to_goal':
        anchor_refs = anchor_refs[:2]
    elif goal_family == 'commit_or_confirm_world_state':
        anchor_refs = anchor_refs[:3]
    else:
        anchor_refs = anchor_refs[:4]

    anchor_colors: List[int] = []
    anchor_shape_labels: List[str] = []
    for ref in anchor_refs:
        obj = object_by_id.get(ref, {})
        color = obj.get('color')
        if color is not None:
            try:
                color_int = int(color)
            except Exception:
                color_int = None
            if color_int is not None and color_int not in anchor_colors:
                anchor_colors.append(color_int)
        for semantic in list(obj.get('semantic_candidates', []) or []):
            if not isinstance(semantic, dict):
                continue
            label = str(semantic.get('label', '') or '')
            if label and label not in anchor_shape_labels:
                anchor_shape_labels.append(label)

    coverage_target = 1
    if anchor_refs:
        if goal_family in {'select_or_activate_salient_structures', 'arrange_or_transform_object_configuration'}:
            interaction_cluster_count = int(world_scene.get('interaction_cluster_count', 0) or 0)
            if interaction_cluster_count > 0:
                coverage_target = min(len(anchor_refs), max(2, min(3, interaction_cluster_count)))
            else:
                coverage_target = min(len(anchor_refs), 2)
        elif goal_family == 'reveal_hidden_state_via_probe':
            coverage_target = min(len(anchor_refs), max(2, min(3, len(anchor_refs))))
        else:
            coverage_target = min(len(anchor_refs), 2)
    repeat_anchor_patience = 2 if goal_family in {'reveal_hidden_state_via_probe', 'commit_or_confirm_world_state'} else 1
    preferred_progress_mode = 'expand_anchor_coverage' if coverage_target > 1 else 'intensify_single_anchor'
    controller_support = _recent_controller_support(episode_trace_tail)
    mechanism_prior_summary = _summarize_mechanism_priors(goal_family, world_model_summary)
    initial_goal_prior_summary = _summarize_initial_goal_priors(world_model_summary)
    controller_anchor_refs = [
        ref for ref in list(controller_support.get('controller_anchor_refs', []) or [])
        if str(ref or '').strip()
    ]
    controller_supported_goal_anchor_refs = [
        ref
        for ref in list(controller_support.get('controller_supported_goal_anchor_refs', []) or [])
        if str(ref or '').strip() and str(ref or '').strip() in anchor_refs
    ]
    controller_supported_goal_colors = [
        color
        for color in list(controller_support.get('controller_supported_goal_colors', []) or [])
        if color in anchor_colors
    ]
    anchor_rank_index = {
        str(candidate.get('object_id', '') or ''): index
        for index, candidate in enumerate(ranked_anchor_candidates)
        if str(candidate.get('object_id', '') or '')
    }
    preferred_next_goal_anchor_refs: List[str] = []
    preferred_next_goal_colors: List[int] = []
    if controller_supported_goal_anchor_refs:
        preferred_progress_mode = 'expand_anchor_coverage'
    if mechanism_prior_summary:
        mechanism_prior_controller_refs = [
            str(ref or '').strip()
            for ref in list(mechanism_prior_summary.get('controller_anchor_refs', []) or [])
            if str(ref or '').strip()
        ]
        mechanism_prior_anchor_refs = [
            str(ref or '').strip()
            for ref in list(mechanism_prior_summary.get('supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip() and str(ref or '').strip() in anchor_refs
        ]
        mechanism_prior_colors = []
        for color in list(mechanism_prior_summary.get('supported_goal_colors', []) or []):
            try:
                color_int = int(color)
            except Exception:
                color_int = None
            if color_int is not None and color_int not in mechanism_prior_colors:
                mechanism_prior_colors.append(color_int)
        if not controller_anchor_refs and mechanism_prior_controller_refs:
            controller_anchor_refs = mechanism_prior_controller_refs
        if not controller_supported_goal_anchor_refs and mechanism_prior_anchor_refs:
            controller_supported_goal_anchor_refs = mechanism_prior_anchor_refs
        if not controller_supported_goal_colors and mechanism_prior_colors:
            controller_supported_goal_colors = mechanism_prior_colors
    if mechanism_prior_summary:
        preferred_progress_mode = str(
            mechanism_prior_summary.get('preferred_progress_mode')
            or preferred_progress_mode
        )
        coverage_target = max(
            coverage_target,
            int(mechanism_prior_summary.get('recommended_coverage_target', coverage_target) or coverage_target),
        )
        if bool(mechanism_prior_summary.get('requires_multi_anchor_coordination', False)):
            repeat_anchor_patience = max(repeat_anchor_patience, 2)
    confidence = max(
        confidence,
        min(
            0.92,
            float(mechanism_prior_summary.get('confidence', 0.0) or 0.0) * 0.72,
        ),
    )

    ranked_supported_goal_anchor_refs = sorted(
        controller_supported_goal_anchor_refs,
        key=lambda ref: (
            anchor_rank_index.get(ref, 10_000),
            anchor_refs.index(ref) if ref in anchor_refs else 10_000,
            str(ref or ''),
        ),
    )
    for ref in ranked_supported_goal_anchor_refs:
        text = str(ref or '').strip()
        if text and text not in preferred_next_goal_anchor_refs:
            preferred_next_goal_anchor_refs.append(text)
    if not preferred_next_goal_anchor_refs and controller_supported_goal_colors:
        for ref in anchor_refs:
            obj = object_by_id.get(ref, {})
            try:
                color_int = int(obj.get('color'))
            except Exception:
                color_int = None
            if color_int is None or color_int not in controller_supported_goal_colors:
                continue
            if ref not in preferred_next_goal_anchor_refs:
                preferred_next_goal_anchor_refs.append(ref)
    if not preferred_next_goal_anchor_refs and mechanism_prior_summary:
        for ref in list(mechanism_prior_summary.get('supported_goal_anchor_refs', []) or []):
            text = str(ref or '').strip()
            if text and text in anchor_refs and text not in preferred_next_goal_anchor_refs:
                preferred_next_goal_anchor_refs.append(text)
    for ref in preferred_next_goal_anchor_refs:
        obj = object_by_id.get(ref, {})
        try:
            color_int = int(obj.get('color'))
        except Exception:
            color_int = None
        if color_int is not None and color_int not in preferred_next_goal_colors:
            preferred_next_goal_colors.append(color_int)
    for color in controller_supported_goal_colors:
        if color not in preferred_next_goal_colors:
            preferred_next_goal_colors.append(color)
    relation_hypotheses = _infer_relation_hypotheses(anchor_refs, object_by_id)

    return {
        'goal_family': goal_family,
        'confidence': round(min(0.98, max(0.0, confidence if confidence > 0.0 else 0.32)), 4),
        'goal_anchor_refs': anchor_refs,
        'goal_anchor_colors': anchor_colors,
        'goal_anchor_shape_labels': anchor_shape_labels[:6],
        'controller_anchor_refs': controller_anchor_refs,
        'controller_supported_goal_anchor_refs': controller_supported_goal_anchor_refs,
        'controller_supported_goal_colors': controller_supported_goal_colors,
        'preferred_next_goal_anchor_refs': preferred_next_goal_anchor_refs[:4],
        'preferred_next_goal_colors': preferred_next_goal_colors[:4],
        'relation_hypotheses': relation_hypotheses,
        'mechanism_prior_count': int(mechanism_prior_summary.get('count', 0) or 0),
        'mechanism_prior_confidence': float(mechanism_prior_summary.get('confidence', 0.0) or 0.0),
        'mechanism_prior_object_ids': [
            str(item or '')
            for item in list(mechanism_prior_summary.get('object_ids', []) or [])
            if str(item or '')
        ],
        'mechanism_prior_supported_goal_anchor_refs': [
            str(item or '')
            for item in list(mechanism_prior_summary.get('supported_goal_anchor_refs', []) or [])
            if str(item or '')
        ],
        'mechanism_prior_controller_anchor_refs': [
            str(item or '')
            for item in list(mechanism_prior_summary.get('controller_anchor_refs', []) or [])
            if str(item or '')
        ],
        'mechanism_prior_supported_goal_colors': [
            int(item)
            for item in list(mechanism_prior_summary.get('supported_goal_colors', []) or [])
            if isinstance(item, int)
        ],
        'mechanism_prior_supporting_functions': [
            str(item or '')
            for item in list(mechanism_prior_summary.get('supporting_functions', []) or [])
            if str(item or '')
        ],
        'mechanism_prior_strategy_hints': dict(mechanism_prior_summary or {}),
        'initial_goal_prior_count': int(initial_goal_prior_summary.get('count', 0) or 0),
        'initial_goal_prior_confidence': float(initial_goal_prior_summary.get('confidence', 0.0) or 0.0),
        'initial_goal_prior_relation': str(initial_goal_prior_summary.get('top_relation', '') or ''),
        'initial_goal_prior_target_group': str(initial_goal_prior_summary.get('top_group', '') or ''),
        'initial_goal_prior_supporting_functions': [
            str(item or '')
            for item in list(initial_goal_prior_summary.get('supporting_functions', []) or [])
            if str(item or '')
        ],
        'initial_goal_prior_summary': str(initial_goal_prior_summary.get('top_summary', '') or ''),
        'coverage_target': int(max(1, coverage_target)),
        'repeat_anchor_patience': int(max(1, repeat_anchor_patience)),
        'preferred_progress_mode': preferred_progress_mode,
        'requires_multi_anchor_coordination': bool(coverage_target > 1),
        'coordination_mode': 'multi_anchor_bundle' if coverage_target > 1 else 'single_anchor_focus',
        'scene_layout_mode': str(world_scene.get('layout_mode', '') or ''),
        'interaction_cluster_count': int(world_scene.get('interaction_cluster_count', 0) or 0),
        'resource_like_element_count': int(world_scene.get('resource_like_element_count', 0) or 0),
    }


def infer_task_frame(
    obs: Dict[str, Any],
    world_model_summary: Dict[str, Any] | None = None,
    object_bindings_summary: Dict[str, Any] | None = None,
    episode_trace_tail: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    world_model_summary = _as_dict(world_model_summary)
    object_bindings_summary = _as_dict(object_bindings_summary)
    perception = _as_dict(obs.get('perception', {}))
    world_scene = _as_dict(object_bindings_summary.get('scene_summary', {})) or _as_dict(world_model_summary.get('world_scene_summary', {}))
    action_names = _action_names(obs)
    mode_scores = _interaction_mode_scores(action_names, perception, world_scene)
    dominant_mode = max(mode_scores.items(), key=lambda item: item[1])[0] if mode_scores else 'unknown'
    goal_candidates = _goal_family_candidates(mode_scores, world_scene, object_bindings_summary)
    initial_goal_prior_summary = _summarize_initial_goal_priors(world_model_summary)
    goal_candidates = _apply_initial_goal_prior_bias(goal_candidates, initial_goal_prior_summary)

    trace_tail = _as_list(episode_trace_tail)
    recent_positive = sum(1 for row in trace_tail if isinstance(row, dict) and float(row.get('reward', 0.0) or 0.0) > 0.0)
    recent_negative = sum(1 for row in trace_tail if isinstance(row, dict) and float(row.get('reward', 0.0) or 0.0) < 0.0)

    frame_type = 'interactive_control'
    if dominant_mode == 'state_transform_interaction' and not action_names:
        frame_type = 'static_transformation'
    elif dominant_mode == 'pointer_interaction' and str(world_scene.get('layout_mode', '') or '').startswith('boundary_scaffold'):
        frame_type = 'structured_pointer_world'
    elif dominant_mode == 'pointer_interaction':
        frame_type = 'pointer_driven_world'
    elif dominant_mode == 'navigation_interaction':
        frame_type = 'navigation_driven_world'

    return {
        'frame_type': frame_type,
        'dominant_interaction_mode': dominant_mode,
        'interaction_mode_scores': mode_scores,
        'available_action_names': action_names,
        'goal_family_candidates': goal_candidates,
        'inferred_level_goal': _infer_level_goal(
            goal_candidates,
            object_bindings_summary,
            world_scene,
            world_model_summary,
            trace_tail,
        ),
        'salience_policy': 'scene_then_object' if int(world_scene.get('scene_element_count', 0) or 0) > 0 else ('object_first' if int(object_bindings_summary.get('object_count', 0) or 0) > 0 else 'surface_first'),
        'scene_understanding_mode': 'scene_elements' if int(world_scene.get('scene_element_count', 0) or 0) > 0 else 'object_only',
        'scene_summary': world_scene,
        'solver_mode': 'hypothesis_driven',
        'world_state_signature': str(world_model_summary.get('world_state_signature', '') or ''),
        'recent_feedback_profile': {
            'recent_positive_events': int(recent_positive),
            'recent_negative_events': int(recent_negative),
        },
        'initial_goal_prior_summary': dict(initial_goal_prior_summary or {}),
    }


def validate_goal_proposal_candidates(
    task_frame_summary: Dict[str, Any] | None,
    proposal_candidates: Sequence[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    task_frame = _as_dict(task_frame_summary)
    proposals = [dict(row) for row in _as_list(proposal_candidates) if isinstance(row, dict)]
    if not task_frame or not proposals:
        return []

    goal_candidates = [
        dict(row) for row in _as_list(task_frame.get('goal_family_candidates', []))
        if isinstance(row, dict)
    ]
    available_action_names = {
        str(item or '').strip()
        for item in _as_list(task_frame.get('available_action_names', []))
        if str(item or '').strip()
    }
    inferred_goal = _as_dict(task_frame.get('inferred_level_goal', {}))
    preferred_goal_refs = {
        str(item or '').strip()
        for item in _as_list(inferred_goal.get('preferred_next_goal_anchor_refs', []))
        if str(item or '').strip()
    }
    controller_refs = {
        str(item or '').strip()
        for item in _as_list(inferred_goal.get('controller_anchor_refs', []))
        if str(item or '').strip()
    }
    relation_hypotheses = [
        dict(row)
        for row in _as_list(inferred_goal.get('relation_hypotheses', []))
        if isinstance(row, dict)
    ]
    top_relation = relation_hypotheses[0] if relation_hypotheses else {}
    top_relation_label = _normalize_relation_label(
        top_relation.get('target_relation') or top_relation.get('relation_type')
    )
    initial_prior_relation = _normalize_relation_label(inferred_goal.get('initial_goal_prior_relation', ''))
    initial_prior_group = str(inferred_goal.get('initial_goal_prior_target_group', '') or '')
    top_goal_family = str(inferred_goal.get('goal_family', '') or '')
    structural_goal_family_names = {
        str(row.get('family', '') or '')
        for row in goal_candidates
        if str(row.get('family', '') or '')
    }

    feedback_rows: List[Dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = str(proposal.get('proposal_id', '') or '')
        proposal_type = str(proposal.get('proposal_type', '') or '').strip().lower()
        summary = str(proposal.get('summary', '') or '')
        target_relation = _normalize_relation_label(
            proposal.get('target_relation')
            or proposal.get('relation_type')
            or ''
        )
        target_group = str(proposal.get('target_group', '') or '')
        supporting_function = str(
            proposal.get('supporting_function')
            or proposal.get('discriminating_function')
            or ''
        ).strip()
        member_anchor_refs = {
            str(item or '').strip()
            for item in _as_list(proposal.get('member_anchor_refs', []))
            if str(item or '').strip()
        }

        relation_match = 0.0
        if target_relation:
            if target_relation == top_relation_label and top_relation_label:
                relation_match = max(relation_match, 0.92)
            if target_relation == initial_prior_relation and initial_prior_relation:
                relation_match = max(relation_match, 0.78)
            for relation_row in relation_hypotheses:
                row_label = _normalize_relation_label(
                    relation_row.get('target_relation') or relation_row.get('relation_type')
                )
                if target_relation and target_relation == row_label:
                    relation_match = max(
                        relation_match,
                        min(0.88, 0.62 + float(relation_row.get('confidence', 0.0) or 0.0) * 0.28),
                    )

        group_overlap = _token_overlap_ratio(target_group, initial_prior_group)
        if not group_overlap and target_group and top_relation:
            group_overlap = max(
                group_overlap,
                _token_overlap_ratio(target_group, top_relation.get('grouping_value', '')),
            )
        if not group_overlap:
            group_overlap = _token_overlap_ratio(summary, initial_prior_group)

        object_binding_support = 0.0
        if member_anchor_refs and preferred_goal_refs:
            object_binding_support = len(member_anchor_refs & preferred_goal_refs) / max(
                1,
                len(member_anchor_refs | preferred_goal_refs),
            )
        elif member_anchor_refs:
            object_binding_support = 0.35
        elif preferred_goal_refs and _token_overlap_ratio(summary, " ".join(sorted(preferred_goal_refs))) > 0.0:
            object_binding_support = 0.28

        controller_compatibility = 0.42
        if supporting_function and supporting_function in available_action_names:
            controller_compatibility = 0.88
        elif proposal_type in {'probe', 'action'} and not supporting_function:
            controller_compatibility = 0.34
        if controller_refs:
            controller_compatibility = max(controller_compatibility, 0.58)

        goal_family_support = 0.36
        if target_relation in {'alignment', 'pairing', 'symmetry', 'containment', 'ordering', 'count_match'}:
            if 'arrange_or_transform_object_configuration' in structural_goal_family_names or top_goal_family == 'arrange_or_transform_object_configuration':
                goal_family_support = 0.88
            else:
                goal_family_support = 0.58
        elif proposal_type in {'probe', 'action'} and 'select_or_activate_salient_structures' in structural_goal_family_names:
            goal_family_support = 0.72

        wm_consistency_score = (
            0.18
            + relation_match * 0.34
            + group_overlap * 0.18
            + object_binding_support * 0.16
            + controller_compatibility * 0.08
            + goal_family_support * 0.06
        )
        relation_progress_predictability = min(
            0.96,
            relation_match * 0.58 + object_binding_support * 0.22 + group_overlap * 0.12,
        )
        predicted_goal_proximity_delta = min(
            0.32,
            0.01 + relation_match * 0.11 + object_binding_support * 0.07 + group_overlap * 0.05,
        )

        contradictions: List[str] = []
        if target_relation and relation_match < 0.25:
            contradictions.append('weak_relation_match_to_world_model')
        if target_group and group_overlap < 0.18:
            contradictions.append('weak_target_group_overlap')
        if supporting_function and supporting_function not in available_action_names:
            contradictions.append('supporting_function_not_currently_available')
        if member_anchor_refs and not (member_anchor_refs & preferred_goal_refs):
            contradictions.append('member_anchors_not_in_preferred_goal_set')

        decision = 'accept_transient' if wm_consistency_score >= 0.56 else 'reject'
        feedback_rows.append(
            {
                'proposal_id': proposal_id,
                'proposal_type': proposal_type,
                'wm_consistency_score': round(min(0.99, max(0.0, wm_consistency_score)), 4),
                'object_binding_support': round(min(0.99, max(0.0, object_binding_support)), 4),
                'relation_progress_predictability': round(min(0.99, max(0.0, relation_progress_predictability)), 4),
                'controller_compatibility': round(min(0.99, max(0.0, controller_compatibility)), 4),
                'predicted_goal_proximity_delta': round(min(0.99, max(0.0, predicted_goal_proximity_delta)), 4),
                'matched_goal_family': top_goal_family,
                'matched_relation': top_relation_label or initial_prior_relation,
                'contradictions': contradictions,
                'decision': decision,
            }
        )

    feedback_rows.sort(
        key=lambda row: (
            0 if str(row.get('decision', '') or '') == 'accept_transient' else 1,
            -float(row.get('wm_consistency_score', 0.0) or 0.0),
            str(row.get('proposal_id', '') or ''),
        )
    )
    return feedback_rows

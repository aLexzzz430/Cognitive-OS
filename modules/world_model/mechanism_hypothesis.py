
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from core.orchestration.action_utils import action_semantic_signature_key, extract_available_functions


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _string_tokens(*values: Any, limit: int = 12) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for token in _string_tokens(*list(value), limit=limit):
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
                    if len(ordered) >= limit:
                        return ordered[:limit]
            continue
        text = str(value or '').strip().lower()
        if not text:
            continue
        canonical = re.sub(r'[^a-z0-9]+', '_', text.replace('::', '_')).strip('_')
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
            if len(ordered) >= limit:
                return ordered[:limit]
        for raw in re.split(r'[^a-z0-9]+', text.replace('::', '_')):
            token = str(raw or '').strip().lower()
            if token and token not in seen:
                seen.add(token)
                ordered.append(token)
                if len(ordered) >= limit:
                    return ordered[:limit]
    return ordered[:limit]


def _normalize_action_family(value: Any) -> str:
    text = str(value or '').strip().lower()
    if not text:
        return ''
    if text in {
        'pointer_interaction',
        'confirm_interaction',
        'navigation_interaction',
        'state_transform_interaction',
        'probe_interaction',
        'wait',
    }:
        return text
    upper = text.upper()
    if upper in {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'}:
        return 'navigation_interaction'
    if upper in {'ACTION5', 'CONFIRM', 'INTERACT', 'SUBMIT', 'ENTER', 'APPLY'}:
        return 'confirm_interaction'
    if upper in {'ACTION6', 'CLICK', 'TAP', 'POINTER_CLICK', 'POINTER_SELECT', 'POINTER_ACTIVATE', 'SELECT'}:
        return 'pointer_interaction'
    if upper in {'ACTION7', 'PROBE', 'PROBE_STATE_CHANGE', 'PROBE_RELATION', 'DRAG', 'TOGGLE', 'TRANSFORM'}:
        return 'state_transform_interaction'
    if 'nav' in text or text in {'move', 'left', 'right', 'up', 'down', 'focus'}:
        return 'navigation_interaction'
    if 'confirm' in text or 'submit' in text or 'interact' in text:
        return 'confirm_interaction'
    if 'pointer' in text or 'click' in text or 'tap' in text or 'select' in text:
        return 'pointer_interaction'
    if 'probe' in text or 'transform' in text or 'toggle' in text:
        return 'state_transform_interaction'
    return text


def _visible_functions_by_family(obs: Dict[str, Any], family: str) -> List[str]:
    available = extract_available_functions(obs)
    normalized = _normalize_action_family(family)
    ranked: List[str] = []
    for fn in available:
        if _normalize_action_family(fn) == normalized and fn not in ranked:
            ranked.append(fn)
    if ranked:
        return ranked
    upper = [str(fn or '').strip().upper() for fn in available]
    if normalized == 'confirm_interaction':
        preferred = {'ACTION5', 'INTERACT', 'SUBMIT', 'CONFIRM'}
    elif normalized == 'pointer_interaction':
        preferred = {'ACTION6', 'CLICK', 'TAP', 'SELECT'}
    elif normalized == 'navigation_interaction':
        preferred = {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'}
    elif normalized == 'state_transform_interaction':
        preferred = {'ACTION7', 'PROBE', 'TOGGLE', 'TRANSFORM'}
    else:
        preferred = set()
    for raw, up in zip(available, upper):
        if up in preferred and raw not in ranked:
            ranked.append(raw)
    return ranked


def _expected_transition_for_family(family: str) -> str:
    transitions = {
        'directional_or_salience_activation': 'unresolved -> revealed',
        'salience_probe': 'unresolved -> revealed',
        'navigate_then_commit': 'unreached -> committed',
        'reveal_then_commit': 'intermediate -> committed',
        'ordered_or_relational_transform': 'misconfigured -> configured',
        'generic_probe_then_update': 'unknown -> informed',
    }
    return str(transitions.get(str(family or ''), 'unknown -> informed'))


def _phase_shift_for_rule(mechanism_family: str, action_family: str) -> str:
    normalized = _normalize_action_family(action_family)
    if normalized == 'confirm_interaction':
        return 'committed'
    if normalized == 'navigation_interaction':
        return 'stabilizing'
    if normalized == 'state_transform_interaction':
        return 'configured'
    if mechanism_family in {'directional_or_salience_activation', 'salience_probe', 'generic_probe_then_update'}:
        return 'revealed'
    if mechanism_family == 'ordered_or_relational_transform':
        return 'configured'
    return 'informed'


def _reward_sign_for_rule(action_family: str) -> str:
    normalized = _normalize_action_family(action_family)
    if normalized == 'confirm_interaction':
        return 'positive'
    return ''


def _information_gain_for_rule(action_family: str, compression_gain: float) -> float:
    normalized = _normalize_action_family(action_family)
    if normalized == 'confirm_interaction':
        return round(max(0.12, min(0.45, compression_gain * 0.45)), 4)
    if normalized == 'navigation_interaction':
        return round(max(0.18, min(0.62, compression_gain * 0.7)), 4)
    if normalized == 'state_transform_interaction':
        return round(max(0.2, min(0.68, compression_gain * 0.78)), 4)
    return round(max(0.22, min(0.72, compression_gain * 0.86)), 4)


def _effect_tokens_for_rule(
    mechanism_family: str,
    rule: Dict[str, Any],
    *,
    target_refs: Sequence[str],
) -> List[str]:
    action_family = _normalize_action_family(rule.get('action_family', ''))
    tokens = _string_tokens(
        rule.get('predicted_effect', []),
        rule.get('preconditions', []),
        mechanism_family,
        target_refs,
    )
    if action_family == 'pointer_interaction':
        tokens.extend(token for token in ['state_change', 'reveal_hidden_state'] if token not in tokens)
    elif action_family == 'navigation_interaction':
        tokens.extend(token for token in ['focus_shift', 'goal_alignment'] if token not in tokens)
    elif action_family == 'state_transform_interaction':
        tokens.extend(token for token in ['relation_progress', 'configuration_update'] if token not in tokens)
    elif action_family == 'confirm_interaction':
        tokens.extend(token for token in ['commit', 'goal_progress'] if token not in tokens)
    phase_shift = _phase_shift_for_rule(mechanism_family, action_family)
    if phase_shift and phase_shift not in tokens:
        tokens.append(phase_shift)
    return tokens[:8]


def _block_policy_for_rule(mechanism_family: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    action_family = _normalize_action_family(rule.get('action_family', ''))
    stage = {
        'pointer_interaction': 'probe',
        'navigation_interaction': 'navigate',
        'state_transform_interaction': 'transform',
        'confirm_interaction': 'commit',
        'wait': 'wait',
    }.get(action_family, 'action')
    preconditions = [str(item or '').strip().lower() for item in _as_list(rule.get('preconditions', [])) if str(item or '').strip()]

    release_on = ['progress', 'transition_signal', 'state_changed']
    soft_decay_on = ['informative_evidence', 'support_evidence']
    if action_family in {'pointer_interaction', 'navigation_interaction', 'state_transform_interaction'}:
        release_on = ['transition_signal', 'state_changed', 'support_evidence']
        soft_decay_on = ['partial_signal', 'informative_evidence', 'support_evidence']
    elif action_family == 'confirm_interaction':
        release_on = ['progress', 'transition_signal', 'state_changed']
        soft_decay_on = ['support_evidence']
    if any('after probe' in item for item in preconditions):
        release_on = ['support_evidence', *release_on]
    release_seen = set()
    soft_seen = set()
    release_on = [item for item in release_on if not (item in release_seen or release_seen.add(item))]
    soft_decay_on = [item for item in soft_decay_on if not (item in soft_seen or soft_seen.add(item))]

    return {
        'policy_stage': stage,
        'mechanism_family': str(mechanism_family or ''),
        'block_on': ['repeated_dead_end'],
        'action_target_threshold': 2,
        'target_family_threshold': 2,
        'target_family_hard_threshold': 3,
        'action_target_reason': f'{stage}_repeated_dead_end',
        'target_family_reason': f'{stage}_family_stalled',
        'target_family_hard_reason': f'{stage}_family_hard_block',
        'release_on': release_on,
        'soft_decay_on': soft_decay_on,
    }


def _predicted_action_effects(
    *,
    obs: Dict[str, Any],
    mechanism_family: str,
    rules: Sequence[Dict[str, Any]],
    target_refs: Sequence[str],
    relations: Sequence[Dict[str, Any]],
    compression_gain: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    action_effects: Dict[str, Dict[str, Any]] = {}
    action_effects_by_signature: Dict[str, Dict[str, Any]] = {}
    relation_type = str((relations[0] if relations else {}).get('type', '') or '')
    target_family = mechanism_family or 'generic_mechanism'
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action_family = _normalize_action_family(rule.get('action_family', ''))
        function_names = _visible_functions_by_family(obs, action_family)
        if not function_names and action_family:
            function_names = [action_family]
        effect_tokens = _effect_tokens_for_rule(mechanism_family, rule, target_refs=target_refs)
        payload = {
            'function_name': '',
            'action_family': action_family,
            'target_family': target_family,
            'target_kind': target_family,
            'anchor_ref': str(target_refs[0] or '') if target_refs else '',
            'relation_type': relation_type,
            'predicted_phase_shift': _phase_shift_for_rule(mechanism_family, action_family),
            'predicted_observation_tokens': effect_tokens,
            'predicted_information_gain': _information_gain_for_rule(action_family, compression_gain),
            'valid_state_change': bool(effect_tokens),
        }
        reward_sign = _reward_sign_for_rule(action_family)
        if reward_sign:
            payload['reward_sign'] = reward_sign
        for function_name in function_names[:3]:
            payload_with_function = dict(payload)
            payload_with_function['function_name'] = str(function_name or '')
            synthetic_action = {
                'kind': 'call_tool',
                'payload': {'tool_args': {'function_name': str(function_name or ''), 'kwargs': {}}},
                'target_family': payload_with_function.get('target_family', ''),
                'relation_type': payload_with_function.get('relation_type', ''),
                'anchor_ref': payload_with_function.get('anchor_ref', ''),
            }
            signature_key = action_semantic_signature_key(synthetic_action)
            action_effects.setdefault(str(function_name), dict(payload_with_function))
            action_effects_by_signature[signature_key] = dict(payload_with_function, semantic_signature=signature_key)
    return action_effects, action_effects_by_signature


def _mechanism_summary(mechanism_family: str, target_refs: Sequence[str], preferred_action_families: Sequence[str]) -> str:
    target_text = ', '.join(str(item or '') for item in list(target_refs or [])[:2] if str(item or '')) or 'candidate targets'
    action_text = ' then '.join(str(item or '') for item in list(preferred_action_families or [])[:2] if str(item or ''))
    if action_text:
        return f'{mechanism_family} expects {action_text} over {target_text}'
    return f'{mechanism_family} explains progress over {target_text}'


def _mechanism_metadata(
    *,
    mechanism_family: str,
    target_refs: Sequence[str],
    preferred_action_families: Sequence[str],
    rules: Sequence[Dict[str, Any]],
    predicted_action_effects: Dict[str, Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    roles: Dict[str, Any],
    contradicting_evidence: Sequence[Any],
    goal_hypothesis_ref: str,
) -> Dict[str, Any]:
    transition_rules = [
        {
            'action_family': _normalize_action_family(rule.get('action_family', '')),
            'preconditions': [str(item or '') for item in _as_list(rule.get('preconditions', [])) if str(item or '')],
            'predicted_effect': [str(item or '') for item in _as_list(rule.get('predicted_effect', [])) if str(item or '')],
            'block_policy': _block_policy_for_rule(mechanism_family, rule),
        }
        for rule in rules
        if isinstance(rule, dict)
    ]
    return {
        'mechanism_hypothesis': True,
        'goal_hypothesis_ref': goal_hypothesis_ref,
        'target_binding_tokens': _string_tokens(target_refs, roles, mechanism_family, limit=8),
        'preconditions': [
            str(item or '')
            for rule in rules
            if isinstance(rule, dict)
            for item in _as_list(rule.get('preconditions', []))
            if str(item or '')
        ][:8],
        'prerequisite_tokens': _string_tokens(
            [
                item
                for rule in rules
                if isinstance(rule, dict)
                for item in _as_list(rule.get('preconditions', []))
            ],
            preferred_action_families,
            limit=8,
        ),
        'counterevidence_tokens': _string_tokens(contradicting_evidence, 'no_state_change', 'misaligned_target', limit=8),
        'recovery_tokens': _string_tokens(preferred_action_families, 'probe_then_replan', 'retarget', limit=8),
        'transition_rules': transition_rules,
        'preferred_action_families': [str(item or '') for item in list(preferred_action_families or []) if str(item or '')][:4],
        'role_targets': _as_dict(roles),
        'relations': [dict(item) for item in list(relations or []) if isinstance(item, dict)][:4],
        'predicted_function_names': [str(item or '') for item in list(predicted_action_effects.keys()) if str(item or '')][:6],
    }


def _top_objects(object_bindings_summary: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    rows = [item for item in _as_list(object_bindings_summary.get('objects', [])) if isinstance(item, dict)]
    rows.sort(
        key=lambda item: (
            -float(item.get('actionable_score', 0.0) or 0.0),
            -float(item.get('salience_score', 0.0) or 0.0),
            str(item.get('object_id', '') or ''),
        )
    )
    return rows[:limit]


def _object_semantic_labels(obj: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for item in _as_list(obj.get('semantic_candidates', [])):
        if not isinstance(item, dict):
            continue
        label = str(item.get('label', '') or '')
        if label and label not in labels:
            labels.append(label)
    return labels


def _object_role_labels(obj: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for item in _as_list(obj.get('role_candidates', [])):
        if not isinstance(item, dict):
            continue
        label = str(item.get('role', '') or '')
        if label and label not in labels:
            labels.append(label)
    return labels


def _recent_outcome_summary(episode_trace_tail: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    tail = [row for row in episode_trace_tail if isinstance(row, dict)]
    if not tail:
        return {
            'negative_rate': 0.0,
            'positive_rate': 0.0,
            'state_change_rate': 0.0,
        }
    total = float(len(tail))
    negative = sum(1 for row in tail if float(row.get('reward', 0.0) or 0.0) < 0.0)
    positive = sum(1 for row in tail if float(row.get('reward', 0.0) or 0.0) > 0.0)
    state_change = sum(
        1 for row in tail
        if bool(row.get('state_changed', False) or row.get('observation_changed', False))
    )
    return {
        'negative_rate': round(negative / total, 4),
        'positive_rate': round(positive / total, 4),
        'state_change_rate': round(state_change / total, 4),
    }


def _roles_for_object(obj: Dict[str, Any]) -> List[str]:
    semantic = set(_object_semantic_labels(obj))
    roles = set(_object_role_labels(obj))
    out: List[str] = []
    if 'directional_like' in semantic or 'hint_or_marker' in roles:
        out.append('marker')
    if 'interactive_token' in roles or 'token_like' in semantic:
        out.append('target')
    if 'constraint_or_boundary' in roles or 'boundary_structure' in semantic:
        out.append('barrier')
    if 'scene_anchor' in roles:
        out.append('anchor')
    if not out:
        out.append('candidate_object')
    return out


def _mechanism_from_goal_family(
    family: str,
    confidence: float,
    top_objects: Sequence[Dict[str, Any]],
    outcome_summary: Dict[str, float],
) -> Dict[str, Any] | None:
    obj_ids = [str(obj.get('object_id', '') or '') for obj in top_objects if str(obj.get('object_id', '') or '')]
    if not obj_ids:
        return None

    marker_objects = [obj for obj in top_objects if 'marker' in _roles_for_object(obj)]
    target_objects = [obj for obj in top_objects if 'target' in _roles_for_object(obj)]
    barrier_objects = [obj for obj in top_objects if 'barrier' in _roles_for_object(obj)]

    if family in {'select_or_activate_salient_structures', 'reveal_hidden_state_via_probe'}:
        target_refs = [str(obj.get('object_id', '') or '') for obj in (target_objects or top_objects[:2])]
        marker_refs = [str(obj.get('object_id', '') or '') for obj in marker_objects[:1]]
        mechanism_family = 'directional_or_salience_activation' if marker_refs else 'salience_probe'
        return {
            'family': mechanism_family,
            'roles': {
                'marker': marker_refs,
                'target': target_refs,
            },
            'relations': [
                {'type': 'points_to_or_highlights', 'source_role': 'marker', 'target_role': 'target'}
            ] if marker_refs else [
                {'type': 'salient_for_probe', 'source_role': 'target', 'target_role': 'target'}
            ],
            'state_variables': {
                'revealed_state': False,
                'activation_progress': 0,
            },
            'rules': [
                {
                    'preconditions': ['target unresolved'],
                    'action_family': 'pointer_interaction',
                    'on_role': 'target',
                    'predicted_effect': ['state_change or reveal hidden state'],
                },
                {
                    'preconditions': ['state changed after probe'],
                    'action_family': 'confirm_interaction',
                    'predicted_effect': ['commit or mode transition'],
                },
            ],
            'predicted_success_condition': [
                'selected salient/pointed target changes world state',
                'confirm commits only after informative interaction',
            ],
            'preferred_target_refs': target_refs,
            'preferred_action_families': ['pointer_interaction', 'confirm_interaction'],
            'best_discriminating_actions': ['pointer_interaction', 'confirm_interaction'],
            'transfer_signature': ['marker_or_salience', 'probe_then_commit'],
            'compression_gain': round(min(0.95, 0.42 + 0.12 * len(target_refs)), 4),
            'base_confidence': confidence,
        }

    if family == 'navigate_agent_or_focus_to_goal':
        target_refs = [str(obj.get('object_id', '') or '') for obj in (target_objects or top_objects[:1])]
        return {
            'family': 'navigate_then_commit',
            'roles': {'goal': target_refs, 'barrier': [str(obj.get('object_id', '') or '') for obj in barrier_objects[:2]]},
            'relations': [{'type': 'reachable_after_navigation', 'source_role': 'goal', 'target_role': 'goal'}],
            'state_variables': {'focus_position_changed': False, 'goal_reached': False},
            'rules': [
                {
                    'preconditions': ['goal not reached'],
                    'action_family': 'navigation_interaction',
                    'on_role': 'goal',
                    'predicted_effect': ['focus_position_changed'],
                },
                {
                    'preconditions': ['focus aligned with goal'],
                    'action_family': 'confirm_interaction',
                    'predicted_effect': ['goal_reached or completion'],
                },
            ],
            'predicted_success_condition': ['navigation changes focus state toward target'],
            'preferred_target_refs': target_refs,
            'preferred_action_families': ['navigation_interaction', 'confirm_interaction'],
            'best_discriminating_actions': ['navigation_interaction'],
            'transfer_signature': ['reachable_goal', 'move_then_commit'],
            'compression_gain': round(min(0.9, 0.38 + 0.08 * len(target_refs)), 4),
            'base_confidence': confidence,
        }

    if family == 'commit_or_confirm_world_state':
        target_refs = [str(obj.get('object_id', '') or '') for obj in top_objects[:2]]
        return {
            'family': 'reveal_then_commit',
            'roles': {'assembly': target_refs},
            'relations': [{'type': 'requires_commit', 'source_role': 'assembly', 'target_role': 'assembly'}],
            'state_variables': {'assembly_ready': False, 'committed': False},
            'rules': [
                {
                    'preconditions': ['assembly not ready'],
                    'action_family': 'pointer_interaction',
                    'on_role': 'assembly',
                    'predicted_effect': ['assembly_ready may change'],
                },
                {
                    'preconditions': ['assembly_ready'],
                    'action_family': 'confirm_interaction',
                    'predicted_effect': ['committed=True or level complete'],
                },
            ],
            'predicted_success_condition': ['commit matters only after valid intermediate state exists'],
            'preferred_target_refs': target_refs,
            'preferred_action_families': ['pointer_interaction', 'confirm_interaction'],
            'best_discriminating_actions': ['confirm_interaction'],
            'transfer_signature': ['intermediate_state', 'explicit_commit'],
            'compression_gain': round(min(0.9, 0.34 + 0.14 * (1.0 - outcome_summary.get('negative_rate', 0.0))), 4),
            'base_confidence': confidence,
        }

    if family == 'arrange_or_transform_object_configuration':
        target_refs = [str(obj.get('object_id', '') or '') for obj in (target_objects or top_objects[:3])]
        return {
            'family': 'ordered_or_relational_transform',
            'roles': {'tokens': target_refs, 'barrier': [str(obj.get('object_id', '') or '') for obj in barrier_objects[:2]]},
            'relations': [{'type': 'ordered_or_structural_relation', 'source_role': 'tokens', 'target_role': 'tokens'}],
            'state_variables': {'relation_progress': 0, 'configuration_valid': False},
            'rules': [
                {
                    'preconditions': ['configuration invalid'],
                    'action_family': 'pointer_interaction',
                    'on_role': 'tokens',
                    'predicted_effect': ['relation_progress changes or ordering evidence appears'],
                },
                {
                    'preconditions': ['configuration partially formed'],
                    'action_family': 'state_transform_interaction',
                    'predicted_effect': ['configuration_valid may increase'],
                },
                {
                    'preconditions': ['configuration_valid'],
                    'action_family': 'confirm_interaction',
                    'predicted_effect': ['completion'],
                },
            ],
            'predicted_success_condition': ['object relations matter more than any single object'],
            'preferred_target_refs': target_refs,
            'preferred_action_families': ['pointer_interaction', 'state_transform_interaction', 'confirm_interaction'],
            'best_discriminating_actions': ['pointer_interaction', 'state_transform_interaction'],
            'transfer_signature': ['multi_object_relation', 'ordered_or_transform'],
            'compression_gain': round(min(0.92, 0.40 + 0.10 * len(target_refs)), 4),
            'base_confidence': confidence,
        }

    return None


def build_mechanism_hypotheses(
    obs: Dict[str, Any],
    task_frame_summary: Dict[str, Any],
    object_bindings_summary: Dict[str, Any],
    goal_hypotheses_summary: Sequence[Dict[str, Any]],
    episode_trace_tail: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    task_frame_summary = _as_dict(task_frame_summary)
    object_bindings_summary = _as_dict(object_bindings_summary)
    goal_hypotheses = [row for row in _as_list(goal_hypotheses_summary) if isinstance(row, dict)]
    trace_tail = [row for row in _as_list(episode_trace_tail) if isinstance(row, dict)]
    top_objects = _top_objects(object_bindings_summary, limit=5)
    outcome_summary = _recent_outcome_summary(trace_tail)
    dominant_mode = str(task_frame_summary.get('dominant_interaction_mode', '') or '')

    mechanisms: List[Dict[str, Any]] = []
    for idx, hypo in enumerate(goal_hypotheses[:5]):
        family = str(hypo.get('family', '') or '')
        base_conf = _clamp01(hypo.get('confidence', 0.0), 0.0)
        template = _mechanism_from_goal_family(family, base_conf, top_objects, outcome_summary)
        if template is None:
            continue
        preferred_target_refs = [str(x or '') for x in list(hypo.get('preferred_target_refs', []) or []) if str(x or '')]
        if preferred_target_refs:
            template['preferred_target_refs'] = preferred_target_refs
        action_families = [str(x or '') for x in list(hypo.get('preferred_action_families', []) or []) if str(x or '')]
        if action_families:
            merged_families: List[str] = []
            for item in list(template.get('preferred_action_families', []) or []) + action_families:
                if item and item not in merged_families:
                    merged_families.append(item)
            template['preferred_action_families'] = merged_families
        best_actions: List[str] = []
        for item in list(template.get('best_discriminating_actions', []) or []) + action_families:
            if item and item not in best_actions:
                best_actions.append(item)
        if not best_actions and dominant_mode:
            best_actions.append(dominant_mode)
        template['best_discriminating_actions'] = best_actions[:3]

        confidence = _clamp01(
            base_conf * 0.58
            + template.get('compression_gain', 0.0) * 0.18
            + (0.08 if dominant_mode in template.get('preferred_action_families', []) else 0.0)
            + (0.06 if outcome_summary.get('state_change_rate', 0.0) > 0.0 else 0.0)
            - outcome_summary.get('negative_rate', 0.0) * 0.10,
            0.0,
        )
        target_refs = [str(x or '') for x in list(template.get('preferred_target_refs', []) or []) if str(x or '')]
        relations = [dict(item) for item in _as_list(template.get('relations', [])) if isinstance(item, dict)]
        roles = _as_dict(template.get('roles', {}))
        rules = [dict(item) for item in _as_list(template.get('rules', [])) if isinstance(item, dict)]
        preferred_action_families = [str(x or '') for x in _as_list(template.get('preferred_action_families', [])) if str(x or '')]
        contradicting_evidence: List[Any] = []
        expected_transition = _expected_transition_for_family(str(template.get('family', '') or ''))
        predicted_action_effects, predicted_action_effects_by_signature = _predicted_action_effects(
            obs=obs,
            mechanism_family=str(template.get('family', 'generic_mechanism') or 'generic_mechanism'),
            rules=rules,
            target_refs=target_refs,
            relations=relations,
            compression_gain=float(template.get('compression_gain', 0.0) or 0.0),
        )
        metadata = _mechanism_metadata(
            mechanism_family=str(template.get('family', 'generic_mechanism') or 'generic_mechanism'),
            target_refs=target_refs,
            preferred_action_families=preferred_action_families,
            rules=rules,
            predicted_action_effects=predicted_action_effects,
            relations=relations,
            roles=roles,
            contradicting_evidence=contradicting_evidence,
            goal_hypothesis_ref=str(hypo.get('hypothesis_id', '') or ''),
        )
        predicted_observation_tokens = _string_tokens(
            expected_transition,
            template.get('predicted_success_condition', []),
            target_refs,
            [payload.get('predicted_observation_tokens', []) for payload in predicted_action_effects.values()],
            limit=10,
        )
        mechanism = {
            'hypothesis_id': f'mech_h{idx+1}',
            'object_id': f'mech_h{idx+1}',
            'type': 'mechanism_hypothesis',
            'object_type': 'hypothesis',
            'hypothesis_type': 'mechanism_hypothesis',
            'goal_family': family,
            'family': str(template.get('family', 'generic_mechanism') or 'generic_mechanism'),
            'summary': _mechanism_summary(
                str(template.get('family', 'generic_mechanism') or 'generic_mechanism'),
                target_refs,
                preferred_action_families,
            ),
            'confidence': round(confidence, 4),
            'posterior': round(confidence, 4),
            'expected_transition': expected_transition,
            'expected_information_gain': round(max(float(template.get('compression_gain', 0.0) or 0.0), float(template.get('base_confidence', 0.0) or 0.0) * 0.55), 4),
            'predicted_phase_shift': str(expected_transition.split('->')[-1].strip() if '->' in expected_transition else ''),
            'predicted_observation_tokens': predicted_observation_tokens,
            'predicted_information_gain': round(max(float(template.get('compression_gain', 0.0) or 0.0), 0.18), 4),
            'predicted_action_effects': predicted_action_effects,
            'predicted_action_effects_by_signature': predicted_action_effects_by_signature,
            'metadata': metadata,
            'source': 'world_model_mechanism',
            'status': 'active',
            'roles': roles,
            'relations': relations,
            'state_variables': _as_dict(template.get('state_variables', {})),
            'rules': rules,
            'predicted_success_condition': [str(x or '') for x in _as_list(template.get('predicted_success_condition', [])) if str(x or '')],
            'supporting_evidence': [
                f'goal_family:{family}',
                f'dominant_mode:{dominant_mode}',
                f'preferred_targets:{",".join(target_refs[:3])}' if target_refs else 'preferred_targets:none',
            ],
            'contradicting_evidence': contradicting_evidence,
            'best_discriminating_actions': [str(x or '') for x in _as_list(template.get('best_discriminating_actions', [])) if str(x or '')],
            'preferred_target_refs': target_refs,
            'supported_goal_count': len(target_refs),
            'supported_goal_anchor_refs': list(target_refs),
            'preferred_action_families': preferred_action_families,
            'preferred_progress_mode': 'expand_anchor_coverage' if len(target_refs) > 1 else 'intensify_single_anchor',
            'requires_multi_anchor_coordination': bool(len(target_refs) > 1),
            'transfer_signature': [str(x or '') for x in _as_list(template.get('transfer_signature', [])) if str(x or '')],
            'compression_gain': float(template.get('compression_gain', 0.0) or 0.0),
            'goal_hypothesis_ref': str(hypo.get('hypothesis_id', '') or ''),
            'tags': ['mechanism_hypothesis', str(template.get('family', 'generic_mechanism') or 'generic_mechanism')],
        }
        mechanisms.append(mechanism)

    if not mechanisms:
        fallback_refs = [str(obj.get('object_id', '') or '') for obj in top_objects[:2]]
        fallback_rules = [{
            'preconditions': ['goal unknown'],
            'action_family': dominant_mode or 'pointer_interaction',
            'on_role': 'candidate_object',
            'predicted_effect': ['information gain'],
        }]
        fallback_family = 'generic_probe_then_update'
        fallback_predicted_action_effects, fallback_predicted_action_effects_by_signature = _predicted_action_effects(
            obs=obs,
            mechanism_family=fallback_family,
            rules=fallback_rules,
            target_refs=fallback_refs,
            relations=[],
            compression_gain=0.24,
        )
        mechanisms.append({
            'hypothesis_id': 'mech_h1',
            'object_id': 'mech_h1',
            'type': 'mechanism_hypothesis',
            'object_type': 'hypothesis',
            'hypothesis_type': 'mechanism_hypothesis',
            'goal_family': '',
            'family': fallback_family,
            'summary': _mechanism_summary(fallback_family, fallback_refs, [dominant_mode or 'pointer_interaction']),
            'confidence': 0.32,
            'posterior': 0.32,
            'expected_transition': _expected_transition_for_family(fallback_family),
            'expected_information_gain': 0.24,
            'predicted_phase_shift': 'revealed',
            'predicted_observation_tokens': _string_tokens('information gain', fallback_refs, fallback_family, limit=8),
            'predicted_information_gain': 0.24,
            'predicted_action_effects': fallback_predicted_action_effects,
            'predicted_action_effects_by_signature': fallback_predicted_action_effects_by_signature,
            'metadata': _mechanism_metadata(
                mechanism_family=fallback_family,
                target_refs=fallback_refs,
                preferred_action_families=[dominant_mode or 'pointer_interaction'],
                rules=fallback_rules,
                predicted_action_effects=fallback_predicted_action_effects,
                relations=[],
                roles={'candidate_object': fallback_refs},
                contradicting_evidence=[],
                goal_hypothesis_ref='',
            ),
            'source': 'world_model_mechanism',
            'status': 'active',
            'roles': {'candidate_object': fallback_refs},
            'relations': [],
            'state_variables': {'world_state_changed': False},
            'rules': fallback_rules,
            'predicted_success_condition': ['probing reduces mechanism uncertainty'],
            'supporting_evidence': [f'dominant_mode:{dominant_mode or "unknown"}'],
            'contradicting_evidence': [],
            'best_discriminating_actions': [dominant_mode or 'pointer_interaction'],
            'preferred_target_refs': fallback_refs,
            'supported_goal_count': len(fallback_refs),
            'supported_goal_anchor_refs': list(fallback_refs),
            'preferred_action_families': [dominant_mode or 'pointer_interaction'],
            'preferred_progress_mode': 'expand_anchor_coverage' if len(fallback_refs) > 1 else 'intensify_single_anchor',
            'requires_multi_anchor_coordination': bool(len(fallback_refs) > 1),
            'transfer_signature': ['generic_probe', 'uncertainty_reduction'],
            'compression_gain': 0.24,
            'goal_hypothesis_ref': '',
            'tags': ['mechanism_hypothesis', fallback_family],
        })

    mechanisms.sort(key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), str(item.get('hypothesis_id', '') or '')))
    return mechanisms[:5]


def summarize_mechanism_control(mechanism_hypotheses_summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mechanisms = [row for row in mechanism_hypotheses_summary if isinstance(row, dict)]
    top = mechanisms[0] if mechanisms else {}
    preferred_target_refs: List[str] = []
    preferred_action_families: List[str] = []
    discriminating_actions: List[str] = []
    families: List[str] = []
    for row in mechanisms:
        family = str(row.get('family', '') or '')
        if family and family not in families:
            families.append(family)
        for value in _as_list(row.get('preferred_target_refs', [])):
            text = str(value or '').strip()
            if text and text not in preferred_target_refs:
                preferred_target_refs.append(text)
        for value in _as_list(row.get('preferred_action_families', [])):
            text = str(value or '').strip()
            if text and text not in preferred_action_families:
                preferred_action_families.append(text)
        for value in _as_list(row.get('best_discriminating_actions', [])):
            text = str(value or '').strip()
            if text and text not in discriminating_actions:
                discriminating_actions.append(text)

    unresolved = []
    if not mechanisms:
        unresolved.append('no_mechanism_hypotheses')
    if not preferred_target_refs:
        unresolved.append('no_preferred_targets')
    if not discriminating_actions:
        unresolved.append('no_discriminating_actions')

    return {
        'dominant_mechanism_family': str(top.get('family', '') or ''),
        'dominant_mechanism_confidence': float(top.get('confidence', 0.0) or 0.0),
        'dominant_mechanism_ref': str(top.get('hypothesis_id', '') or ''),
        'preferred_target_refs': preferred_target_refs[:5],
        'preferred_action_families': preferred_action_families[:4],
        'discriminating_actions': discriminating_actions[:4],
        'mechanism_families': families[:6],
        'mechanism_ready': not unresolved,
        'unresolved_mechanism_dimensions': unresolved,
    }

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from core.orchestration.action_utils import action_matches_blocked_name, extract_action_identity, extract_action_kind


INTERNAL_ONLY_ACTIONS = {
    'inspect',
    'reflect',
    'replan',
    'retrieve',
    'deliberate',
}

_ORDINAL_ACTION_RE = re.compile(r'^ACTION\d+$', re.IGNORECASE)


@dataclass
class NormalizationInput:
    governance_candidates: List[Dict[str, Any]]
    raw_candidates: List[Dict[str, Any]]
    selected_action: Optional[Dict[str, Any]]
    obs_before: Optional[Dict[str, Any]]
    tick: int
    episode: int
    episode_trace: Sequence[dict]


@dataclass
class NormalizationResult:
    unique_candidate_actions: List[Dict[str, Any]]
    governance_candidates_before_normalization: List[Dict[str, Any]]
    governance_candidates_after_normalization: List[Dict[str, Any]]
    skip_layer1: bool
    skip_reason: str
    viability_events: List[Dict[str, Any]]


def has_sufficient_failure_evidence(episode_trace: Sequence[dict]) -> bool:
    recent = episode_trace[-2:] if isinstance(episode_trace, list) else []
    if not recent:
        return False
    explicit_failures = 0
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        reward = float(entry.get('reward', 0.0) or 0.0)
        result = entry.get('result', {}) if isinstance(entry.get('result', {}), dict) else {}
        failure_reason = result.get('failure_reason')
        success_flag = result.get('success')
        if reward < 0.0:
            explicit_failures += 1
        elif isinstance(failure_reason, str) and failure_reason.strip() and failure_reason != 'none':
            explicit_failures += 1
        elif success_flag is False:
            explicit_failures += 1
    return explicit_failures >= 1


def _must_enforce_non_wait_viability(
    raw_candidates: List[Dict[str, Any]],
    obs_before: Optional[Dict[str, Any]],
    tick: int,
    episode_trace: Sequence[dict],
) -> bool:
    visible_functions = _extract_visible_function_set(obs_before)
    return (
        bool(raw_candidates)
        and bool(visible_functions)
        and int(tick) <= 2
        and not has_sufficient_failure_evidence(episode_trace)
    )


def _pending_wait_only_governance_state(
    obs_before: Optional[Dict[str, Any]],
    governance_candidates: Sequence[Dict[str, Any]],
) -> bool:
    if not isinstance(obs_before, dict):
        return False
    try:
        pending_count = int(obs_before.get('pending_countdown', 0) or 0)
    except (TypeError, ValueError):
        pending_count = 0
    if pending_count <= 0 and not bool(obs_before.get('delayed_resolution_pending', False)):
        return False
    candidate_names = [
        str(candidate.get('function_name') or candidate.get('action') or '').strip()
        for candidate in governance_candidates if isinstance(candidate, dict)
    ]
    if not candidate_names:
        return False
    has_wait = any(name == 'wait' for name in candidate_names)
    has_non_wait = any(name and name != 'wait' for name in candidate_names)
    return has_wait and not has_non_wait


def _must_bias_to_external_actions(
    obs_before: Optional[Dict[str, Any]],
    tick: int,
    episode_trace: Sequence[dict],
) -> bool:
    visible_functions = _extract_visible_function_set(obs_before)
    return bool(visible_functions) and int(tick) <= 2 and not has_sufficient_failure_evidence(episode_trace)


def _extract_visible_function_set(obs_before: Optional[Dict[str, Any]]) -> Set[str]:
    visible: Set[str] = set()

    if not isinstance(obs_before, dict):
        return visible

    for key in ('available_functions', 'visible_functions'):
        raw_top = obs_before.get(key, [])
        if isinstance(raw_top, list):
            visible.update(str(fn).strip() for fn in raw_top if str(fn).strip())

    obs_novel_api = obs_before.get('novel_api', {})
    if hasattr(obs_novel_api, 'raw'):
        obs_novel_api = obs_novel_api.raw

    if isinstance(obs_novel_api, dict):
        for key in ('available_functions', 'visible_functions', 'discovered_functions'):
            raw_visible = obs_novel_api.get(key, [])
            if isinstance(raw_visible, list):
                visible.update(str(fn).strip() for fn in raw_visible if str(fn).strip())

    return visible


def _extract_visible_function_list(obs_before: Optional[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []

    def _extend(raw_values: Any) -> None:
        if not isinstance(raw_values, list):
            return
        for value in raw_values:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

    if not isinstance(obs_before, dict):
        return ordered

    for key in ('available_functions', 'visible_functions'):
        _extend(obs_before.get(key, []))

    obs_novel_api = obs_before.get('novel_api', {})
    if hasattr(obs_novel_api, 'raw'):
        obs_novel_api = obs_novel_api.raw

    if isinstance(obs_novel_api, dict):
        for key in ('available_functions', 'visible_functions', 'discovered_functions'):
            _extend(obs_novel_api.get(key, []))

    return ordered


def _extract_action_name_text(
    action: Dict[str, Any],
    extract_action_function_name: Optional[Callable[[Dict[str, Any], str], str]] = None,
) -> str:
    if not isinstance(action, dict):
        return ''
    if extract_action_function_name is not None:
        fn_name = extract_action_function_name(action, '').strip()
        if fn_name:
            return fn_name
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    for value in (
        tool_args.get('function_name'),
        payload.get('function_name'),
        action.get('function_name'),
        action.get('selected_name'),
        action.get('action'),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _is_internal_only_action(
    action: Dict[str, Any],
    extract_action_function_name: Optional[Callable[[Dict[str, Any], str], str]] = None,
) -> bool:
    if not isinstance(action, dict):
        return False
    kind = str(action.get('kind', '') or '').strip().lower()
    if kind in INTERNAL_ONLY_ACTIONS:
        return True
    fn_name = _extract_action_name_text(action, extract_action_function_name).strip().lower()
    return fn_name in INTERNAL_ONLY_ACTIONS


def _is_internal_only_governance_candidate(
    candidate: Dict[str, Any],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    if _is_internal_only_action(raw_action, extract_action_function_name):
        return True
    fn_name = str(candidate.get('function_name') or '').strip().lower()
    if fn_name in INTERNAL_ONLY_ACTIONS:
        return True
    action_name = str(candidate.get('action') or '').strip().lower()
    return action_name in INTERNAL_ONLY_ACTIONS


def _candidate_action_signature(action: Dict[str, Any], extract_action_function_name: Callable[[Dict[str, Any], str], str]) -> Optional[Tuple[str, str]]:
    if not isinstance(action, dict):
        return None
    fn_name = extract_action_identity(action, include_function_fallback=True).strip() or extract_action_function_name(action, '').strip()
    if not fn_name:
        return None
    action_kind = extract_action_kind(action, default='call_tool')
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
    try:
        param_signature = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        param_signature = str(sorted((str(k), str(v)) for k, v in kwargs.items()))
    return action_kind, fn_name, param_signature


def _raw_action_control_bucket(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return 'open'
    source = str(action.get('_source', '') or '').strip().lower()
    if source in {'planner', 'self_model'}:
        return source
    return 'open'


def _governance_signature(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    fn_name = str(candidate.get('function_name') or '').strip()
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    action_kind = extract_action_kind(raw_action, default='call_tool')
    payload = raw_action.get('payload', {}) if isinstance(raw_action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
    kwargs_sig = tuple(sorted((str(k), repr(v)) for k, v in kwargs.items()))
    return (action_kind, fn_name, kwargs_sig)


def _governance_control_bucket(candidate: Dict[str, Any]) -> str:
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    source = str(raw_action.get('_source', candidate.get('intent', '')) or '').strip().lower()
    if source in {'planner', 'self_model'}:
        return source
    return 'open'


def _candidate_dedup_priority(candidate: Dict[str, Any]) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
    procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
    intent = str(candidate.get('intent') or raw_action.get('_source') or '').strip().lower()

    priority = 0.0
    if intent == 'procedure_reuse':
        priority += 1.0
    if bool(procedure.get('is_next_step', False)):
        priority += 1.5
    if bool(procedure_guidance.get('active_next_step', False)):
        priority += 1.0
    if str(procedure.get('hit_source', '') or '') == 'latent_mechanism_abstraction':
        priority += 0.75
    return priority


def _raw_action_priority(action: Dict[str, Any]) -> float:
    if not isinstance(action, dict):
        return 0.0
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
    procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
    source = str(action.get('_source', '') or '').strip().lower()

    priority = 0.0
    if source == 'procedure_reuse':
        priority += 1.0
    if bool(procedure.get('is_next_step', False)):
        priority += 1.5
    if bool(procedure_guidance.get('active_next_step', False)):
        priority += 1.0
    if str(procedure.get('hit_source', '') or '') == 'latent_mechanism_abstraction':
        priority += 0.75
    return priority


def _action_has_procedure_support(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    if str(action.get('_source', '') or '').strip().lower() == 'procedure_reuse':
        return True
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    support_sources = meta.get('support_sources', [])
    if isinstance(support_sources, list) and any(str(source).strip().lower() == 'procedure_reuse' for source in support_sources):
        return True
    procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
    return bool(procedure)


def _make_wait_governance_candidate(intent: str = 'safe_fallback') -> Dict[str, Any]:
    return {
        'action': 'wait',
        'function_name': 'wait',
        'intent': intent,
        'risk': 0.05,
        'opportunity_estimate': 0.1,
        'final_score': 0.05,
        'estimated_cost': 0.1,
        'raw_action': {'kind': 'wait', 'payload': {}},
    }


def _candidate_recent_feedback(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
    recent_feedback = meta.get('recent_action_feedback', {}) if isinstance(meta.get('recent_action_feedback', {}), dict) else {}
    if recent_feedback:
        return recent_feedback
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    raw_meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    recent_feedback = raw_meta.get('recent_action_feedback', {}) if isinstance(raw_meta.get('recent_action_feedback', {}), dict) else {}
    return recent_feedback


def _action_recent_feedback(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    recent_feedback = meta.get('recent_action_feedback', {}) if isinstance(meta.get('recent_action_feedback', {}), dict) else {}
    return recent_feedback


def _is_recent_no_progress_locked(feedback: Dict[str, Any]) -> bool:
    if not isinstance(feedback, dict):
        return False
    consecutive_no_progress = int(feedback.get('consecutive_no_progress_count', 0) or 0)
    positive_progress_count = int(feedback.get('positive_progress_count', 0) or 0)
    cooldown_recommended = bool(feedback.get('action_cooldown_recommended', False))
    return cooldown_recommended or (consecutive_no_progress >= 3 and positive_progress_count == 0)


def _blocked_function_names(*payloads: Any) -> Set[str]:
    blocked: Set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ('blocked_functions', 'blocked_action_classes'):
            values = payload.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value or '').strip()
                if text:
                    blocked.add(text)
    return blocked


def _candidate_blocked_function_names(candidate: Dict[str, Any]) -> Set[str]:
    if not isinstance(candidate, dict):
        return set()
    blocked = _blocked_function_names(candidate)
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    if isinstance(raw_action, dict):
        blocked.update(_blocked_function_names(raw_action))
        raw_meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        blocked.update(_blocked_function_names(raw_meta))
        blocked.update(_blocked_function_names(raw_meta.get('failure_strategy_profile', {}), raw_meta.get('global_failure_strategy', {})))
    meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
    blocked.update(_blocked_function_names(meta))
    blocked.update(_blocked_function_names(meta.get('failure_strategy_profile', {}), meta.get('global_failure_strategy', {})))
    return blocked


def _rank_visible_surface_functions(obs_before: Optional[Dict[str, Any]], anchor_function: str = '') -> List[str]:
    visible = _extract_visible_function_list(obs_before)
    anchor_function = str(anchor_function or '').strip()
    if len(visible) >= 2 and all(_ORDINAL_ACTION_RE.match(str(fn or '').strip()) for fn in visible):
        non_action1 = [fn for fn in visible if str(fn).strip().upper() != 'ACTION1']
        action1 = [fn for fn in visible if str(fn).strip().upper() == 'ACTION1']
        if non_action1:
            visible = non_action1 + action1
    if anchor_function and anchor_function in visible:
        visible = [anchor_function, *[fn for fn in visible if fn != anchor_function]]
    return visible


def _surface_perception_is_empty(obs_before: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(obs_before, dict):
        return False
    perception = obs_before.get('perception', {}) if isinstance(obs_before.get('perception', {}), dict) else {}
    if not perception:
        return False
    frame_count = int(perception.get('frame_count', 0) or 0)
    hotspot = perception.get('suggested_hotspot')
    active_bbox = perception.get('active_bbox')
    shape = perception.get('grid_shape', {}) if isinstance(perception.get('grid_shape', {}), dict) else {}
    width = int(shape.get('width', 0) or 0)
    height = int(shape.get('height', 0) or 0)
    return frame_count == 0 and hotspot in (None, {}) and active_bbox in (None, {}) and width == 0 and height == 0


def _surface_action_try_counts(
    episode_trace: Sequence[dict],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in episode_trace if isinstance(episode_trace, list) else []:
        if not isinstance(entry, dict):
            continue
        action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
        fn_name = _extract_action_name_text(action, extract_action_function_name).strip()
        if not fn_name:
            continue
        counts[fn_name] = counts.get(fn_name, 0) + 1
    return counts


def _ordinal_empty_surface_priority(fn_name: str) -> int:
    priority_map = {
        'ACTION5': 0,
        'ACTION2': 1,
        'ACTION3': 1,
        'ACTION4': 1,
        'ACTION7': 2,
        'ACTION1': 3,
        'ACTION6': 4,
    }
    return priority_map.get(str(fn_name or '').strip().upper(), 5)


def _make_surface_recovery_action(function_name: str) -> Dict[str, Any]:
    return {
        'kind': 'call_tool',
        'payload': {
            'tool_name': 'call_hidden_function',
            'tool_args': {
                'function_name': function_name,
                'kwargs': {},
            },
        },
        '_source': 'surface_diversity_recovery',
        '_candidate_meta': {
            'surface_diversity_recovery': True,
            'support_sources': ['visible_surface_recovery'],
        },
    }


def _synthesize_visible_surface_candidates(
    existing_candidates: Sequence[Dict[str, Any]],
    obs_before: Optional[Dict[str, Any]],
    episode_trace: Sequence[dict],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
    *,
    anchor_function: str = '',
    anchor_only: bool = False,
    blocked_functions: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    blocked = set(blocked_functions or set())
    existing_functions = {
        str(candidate.get('function_name') or '').strip()
        for candidate in existing_candidates
        if isinstance(candidate, dict)
    }
    existing_open_functions = {
        str(candidate.get('function_name') or '').strip()
        for candidate in existing_candidates
        if isinstance(candidate, dict)
        and _governance_control_bucket(candidate) == 'open'
        and str(candidate.get('function_name') or '').strip()
    }
    ranked_visible = _rank_visible_surface_functions(obs_before, anchor_function=anchor_function)
    if anchor_only and anchor_function and anchor_function in ranked_visible:
        ranked_visible = [anchor_function]
    tried_counts = _surface_action_try_counts(episode_trace, extract_action_function_name)
    empty_surface = _surface_perception_is_empty(obs_before)
    ordinal_surface = bool(ranked_visible) and all(_ORDINAL_ACTION_RE.match(str(fn or '').strip()) for fn in ranked_visible)
    indexed_visible = list(enumerate(ranked_visible))
    indexed_visible.sort(
        key=lambda item: (
            int(tried_counts.get(str(item[1]).strip(), 0)),
            _ordinal_empty_surface_priority(item[1]) if ordinal_surface and empty_surface else (1 if str(item[1]).strip().upper() == 'ACTION6' and empty_surface else 0),
            item[0],
        )
    )
    synthesized: List[Dict[str, Any]] = []
    for order_index, (_, fn_name) in enumerate(indexed_visible):
        if not fn_name or fn_name == 'wait' or fn_name in blocked:
            continue
        allow_anchor_duplicate = (
            anchor_only
            and fn_name == anchor_function
            and fn_name not in existing_open_functions
        )
        if fn_name in existing_functions and not allow_anchor_duplicate:
            continue
        raw_action = _make_surface_recovery_action(fn_name)
        tried_count = int(tried_counts.get(str(fn_name).strip(), 0))
        base_opportunity = 0.72
        base_risk = 0.24
        fn_upper = str(fn_name).strip().upper()
        if ordinal_surface and empty_surface:
            if fn_upper == 'ACTION5':
                base_opportunity = 0.84
                base_risk = 0.18
            elif fn_upper in {'ACTION2', 'ACTION3', 'ACTION4'}:
                base_opportunity = 0.76
                base_risk = 0.23
            elif fn_upper == 'ACTION7':
                base_opportunity = 0.66
                base_risk = 0.24
            elif fn_upper == 'ACTION1':
                base_opportunity = 0.62
                base_risk = 0.25
            elif fn_upper == 'ACTION6':
                base_opportunity = 0.48
                base_risk = 0.30
        opportunity = max(0.18, min(0.92, base_opportunity - (tried_count * 0.08) - (order_index * 0.015)))
        risk = max(0.05, min(0.85, base_risk + (tried_count * 0.035)))
        synthesized.append({
            'action': fn_name,
            'function_name': fn_name,
            'intent': 'surface_diversity_recovery',
            'risk': risk,
            'opportunity_estimate': opportunity,
            'final_score': opportunity - risk,
            'estimated_cost': 1.0,
            'raw_action': raw_action,
        })
    return synthesized


def _governance_wait_baseline_allowed(raw_candidates: List[Dict[str, Any]], selected_action: Optional[Dict[str, Any]]) -> bool:
    for candidate in [*raw_candidates, selected_action]:
        if not isinstance(candidate, dict):
            continue
        meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
        for key in ('forbid_wait_baseline', 'disable_wait_baseline', 'planner_forbid_wait_baseline'):
            if bool(meta.get(key)):
                return False
    return True


def _contains_exploration_candidate(raw_candidates: List[Dict[str, Any]], extract_action_function_name: Callable[[Dict[str, Any], str], str]) -> bool:
    for candidate in raw_candidates if isinstance(raw_candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        fn_name = extract_action_function_name(candidate, '').strip()
        kind = extract_action_kind(candidate, default='call_tool')
        if kind == 'probe' or fn_name == 'probe':
            return True
    return False


def _is_synthetic_deliberation_probe_action(
    action: Dict[str, Any],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> bool:
    if not isinstance(action, dict):
        return False
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    source = str(action.get('_source', '') or '').strip().lower()
    if source == 'deliberation_probe':
        return True
    fn_name = extract_action_function_name(action, '').strip().lower()
    action_kind = extract_action_kind(action, default='call_tool').strip().lower()
    return bool(meta.get('deliberation_injected', False) and (action_kind == 'probe' or 'probe' in fn_name))


def _is_safe_selected_baseline_action(
    action: Dict[str, Any],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> bool:
    if not isinstance(action, dict):
        return False
    if _is_synthetic_deliberation_probe_action(action, extract_action_function_name):
        return False
    fn_name = extract_action_function_name(action, '').strip().lower()
    action_kind = extract_action_kind(action, default='call_tool').strip().lower()
    if not fn_name or fn_name == 'wait':
        return False
    if action_kind == 'inspect' or fn_name == 'inspect':
        return True
    return any(token in fn_name for token in ('inspect', 'verify', 'check', 'test')) and 'probe' not in fn_name


def _contains_action_signature(
    actions: Sequence[Dict[str, Any]],
    target_action: Dict[str, Any],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> bool:
    target_sig = _candidate_action_signature(target_action, extract_action_function_name)
    if target_sig is None:
        return False
    target_sig = (*target_sig, _raw_action_control_bucket(target_action))
    for action in actions:
        if not isinstance(action, dict):
            continue
        sig = _candidate_action_signature(action, extract_action_function_name)
        if sig is None:
            continue
        if (*sig, _raw_action_control_bucket(action)) == target_sig:
            return True
    return False


def _recover_selected_action_from_probe_singleton(
    unique_candidate_actions: Sequence[Dict[str, Any]],
    selected_action: Optional[Dict[str, Any]],
    obs_before: Optional[Dict[str, Any]],
    extract_action_function_name: Callable[[Dict[str, Any], str], str],
) -> Optional[Dict[str, Any]]:
    if len(_extract_visible_function_set(obs_before)) != 0:
        return None
    if len(unique_candidate_actions) != 1:
        return None
    only_action = unique_candidate_actions[0] if isinstance(unique_candidate_actions[0], dict) else {}
    if not _is_synthetic_deliberation_probe_action(only_action, extract_action_function_name):
        return None
    if not isinstance(selected_action, dict):
        return None
    if not _is_safe_selected_baseline_action(selected_action, extract_action_function_name):
        return None
    if _is_recent_no_progress_locked(_action_recent_feedback(selected_action)):
        return None
    if action_matches_blocked_name(selected_action, _candidate_blocked_function_names({'raw_action': selected_action})):
        return None
    if _contains_action_signature(unique_candidate_actions, selected_action, extract_action_function_name):
        return None
    return dict(selected_action)


def _recover_viable_non_wait_from_raw(raw_candidates: List[Dict[str, Any]], obs_before: Optional[Dict[str, Any]], extract_action_function_name: Callable[[Dict[str, Any], str], str]) -> Optional[Dict[str, Any]]:
    visible_functions = _extract_visible_function_set(obs_before)
    viable_candidates: List[Dict[str, Any]] = []
    for raw_action in raw_candidates if isinstance(raw_candidates, list) else []:
        if not isinstance(raw_action, dict):
            continue
        if _is_internal_only_action(raw_action, extract_action_function_name):
            continue
        fn_name = extract_action_function_name(raw_action, '').strip()
        if not fn_name or fn_name == 'wait':
            continue
        if visible_functions and fn_name not in visible_functions:
            continue
        if _is_recent_no_progress_locked(_action_recent_feedback(raw_action)):
            continue
        if action_matches_blocked_name(raw_action, _candidate_blocked_function_names({'raw_action': raw_action})):
            continue
        viable_candidates.append(raw_action)

    if not viable_candidates:
        return None

    viable_candidates.sort(
        key=lambda action: (
            1 if _action_has_procedure_support(action) else 0,
            _raw_action_priority(action),
        ),
        reverse=True,
    )
    raw_action = viable_candidates[0]
    fn_name = extract_action_function_name(raw_action, '').strip()
    return {
            'action': fn_name,
            'function_name': fn_name,
            'intent': 'viability_recovered_from_raw',
            'risk': float(raw_action.get('risk', 0.2)),
            'opportunity_estimate': float(raw_action.get('opportunity_estimate', 0.8)),
            'final_score': float(raw_action.get('final_score', 0.6)),
            'estimated_cost': float(raw_action.get('estimated_cost', 1.0)),
            'raw_action': raw_action,
        }


def _singleton_skip_guard_reason(unique_candidate_actions: List[Dict[str, Any]], obs_before: Optional[Dict[str, Any]]) -> str:
    if len(unique_candidate_actions) != 1:
        return ''
    only_action = unique_candidate_actions[0] if isinstance(unique_candidate_actions[0], dict) else {}
    only_kind = extract_action_kind(only_action, default='call_tool')
    only_source = str(only_action.get('_source', '') or '').strip().lower()
    only_meta = only_action.get('_candidate_meta', {}) if isinstance(only_action.get('_candidate_meta', {}), dict) else {}
    only_fn = _extract_action_name_text(only_action).strip()
    visible_function_count = len(_extract_visible_function_set(obs_before))
    recent_feedback = only_meta.get('recent_action_feedback', {}) if isinstance(only_meta.get('recent_action_feedback', {}), dict) else {}
    consecutive_no_progress = int(
        recent_feedback.get(
            'consecutive_no_progress_count',
            only_meta.get('consecutive_no_progress_count', 0),
        ) or 0
    )
    action_cooldown_recommended = bool(
        recent_feedback.get('action_cooldown_recommended', only_meta.get('action_cooldown_recommended', False))
    )

    if only_kind == 'wait' or only_fn == 'wait':
        return 'only_wait_candidate'
    if action_cooldown_recommended or consecutive_no_progress >= 2:
        return 'singleton_candidate_recent_no_progress'
    if visible_function_count >= 2:
        return 'singleton_candidate_but_multiple_visible_functions'
    if only_source in {'base_generation', 'self_model'}:
        return 'singleton_candidate_from_default_source'
    return ''


def normalize_candidates(input_obj: NormalizationInput, extract_action_function_name: Callable[[Dict[str, Any], str], str]) -> NormalizationResult:
    unique_candidate_actions: List[Dict[str, Any]] = []
    seen_action: Dict[Tuple[Any, ...], int] = {}
    for action in input_obj.raw_candidates if isinstance(input_obj.raw_candidates, list) else []:
        sig = _candidate_action_signature(action, extract_action_function_name)
        if sig is None:
            unique_candidate_actions.append(action)
            continue
        sig = (*sig, _raw_action_control_bucket(action))
        existing_idx = seen_action.get(sig)
        if existing_idx is not None:
            if _raw_action_priority(action) > _raw_action_priority(unique_candidate_actions[existing_idx]):
                unique_candidate_actions[existing_idx] = action
            continue
        seen_action[sig] = len(unique_candidate_actions)
        unique_candidate_actions.append(action)

    recovered_selected_action = _recover_selected_action_from_probe_singleton(
        unique_candidate_actions,
        input_obj.selected_action,
        input_obj.obs_before,
        extract_action_function_name,
    )
    if recovered_selected_action is not None:
        unique_candidate_actions.append(recovered_selected_action)

    singleton_guard_reason = _singleton_skip_guard_reason(unique_candidate_actions, input_obj.obs_before)
    skip_layer1 = len(unique_candidate_actions) < 2 and not singleton_guard_reason
    skip_reason = ''
    if skip_layer1:
        skip_reason = f"unique_candidates_lt_2:{len(unique_candidate_actions)}"
    elif singleton_guard_reason:
        skip_reason = f"singleton_guard:{singleton_guard_reason}"

    governance_candidates_before_normalization = [dict(c) for c in input_obj.governance_candidates if isinstance(c, dict)]
    pending_wait_only_governance = _pending_wait_only_governance_state(
        input_obj.obs_before,
        governance_candidates_before_normalization,
    )

    deduped: List[Dict[str, Any]] = []
    seen_signatures: Dict[Tuple[Any, ...], int] = {}
    for candidate in governance_candidates_before_normalization:
        fn_name = str(candidate.get('function_name') or '').strip()
        if not fn_name:
            continue
        candidate['function_name'] = fn_name
        candidate['action'] = str(candidate.get('action') or fn_name).strip() or fn_name
        sig = (*_governance_signature(candidate), _governance_control_bucket(candidate))
        existing_idx = seen_signatures.get(sig)
        if existing_idx is not None:
            if _candidate_dedup_priority(candidate) > _candidate_dedup_priority(deduped[existing_idx]):
                deduped[existing_idx] = candidate
            continue
        seen_signatures[sig] = len(deduped)
        deduped.append(candidate)

    if not deduped:
        fallback = _make_wait_governance_candidate(intent='constraint_empty_fallback')
        fallback['_normalization_source'] = 'constraint_empty_fallback'
        deduped = [fallback]

    visible_functions = _extract_visible_function_set(input_obj.obs_before)
    visible_function_list = _extract_visible_function_list(input_obj.obs_before)
    cooldown_locked_functions: Set[str] = set()
    blocked_functions: Set[str] = set()
    for candidate in [*deduped, *input_obj.raw_candidates]:
        if not isinstance(candidate, dict):
            continue
        candidate_fn = _extract_action_name_text(candidate, extract_action_function_name).strip()
        if candidate_fn and _is_recent_no_progress_locked(
            _candidate_recent_feedback(candidate) or _action_recent_feedback(candidate),
        ):
            cooldown_locked_functions.add(candidate_fn)
        candidate_blocked_functions = _candidate_blocked_function_names(candidate)
        blocked_functions.update(candidate_blocked_functions)

    if cooldown_locked_functions:
        cooled_candidates = [
            candidate
            for candidate in deduped
            if str(candidate.get('function_name') or '').strip() in cooldown_locked_functions
        ]
        removable_functions = {
            str(candidate.get('function_name') or '').strip()
            for candidate in cooled_candidates
            if str(candidate.get('function_name') or '').strip()
            and str(candidate.get('function_name') or '').strip() != 'wait'
            and (
                len(visible_functions - {str(candidate.get('function_name') or '').strip()}) >= 1
                or any(
                    str(other.get('function_name') or '').strip() not in cooldown_locked_functions
                    and str(other.get('function_name') or '').strip() != 'wait'
                    for other in deduped
                    if isinstance(other, dict)
                )
            )
        }
        if removable_functions:
            deduped = [
                candidate
                for candidate in deduped
                if str(candidate.get('function_name') or '').strip() not in removable_functions
            ]
            viability_events = [{
                'tick': input_obj.tick,
                'episode': input_obj.episode,
                'event': 'recent_no_progress_candidates_suppressed',
                'suppressed_functions': sorted(removable_functions),
                'visible_functions': list(visible_function_list),
            }]
        else:
            viability_events = []
    else:
        viability_events = []

    if recovered_selected_action is not None:
        viability_events.append({
            'tick': input_obj.tick,
            'episode': input_obj.episode,
            'event': 'selected_action_recovered_from_probe_singleton',
            'recovered_function': _extract_action_name_text(recovered_selected_action, extract_action_function_name).strip(),
            'visible_functions': list(visible_function_list),
        })
    if pending_wait_only_governance:
        viability_events.append({
            'tick': input_obj.tick,
            'episode': input_obj.episode,
            'event': 'pending_wait_governance_preserved',
            'visible_functions': list(visible_function_list),
        })

    has_wait = any(str(c.get('function_name') or '').strip() == 'wait' for c in deduped)
    non_wait_count = sum(1 for c in deduped if str(c.get('function_name') or '').strip() != 'wait')
    no_visible_functions = len(visible_functions) == 0
    has_exploration_candidate = _contains_exploration_candidate(input_obj.raw_candidates, extract_action_function_name)
    has_failure_feedback = has_sufficient_failure_evidence(input_obj.episode_trace)
    suppress_wait_competition = no_visible_functions and has_exploration_candidate and not has_failure_feedback
    if (
        non_wait_count == 1
        and not has_wait
        and _governance_wait_baseline_allowed(input_obj.raw_candidates, input_obj.selected_action)
        and not suppress_wait_competition
    ):
        deduped.append(_make_wait_governance_candidate(intent='safe_baseline_wait'))
    if suppress_wait_competition:
        deduped = [
            candidate
            for candidate in deduped
            if str(candidate.get('function_name') or '').strip() != 'wait'
        ]
        if not deduped:
            recovered = _recover_viable_non_wait_from_raw(
                input_obj.raw_candidates,
                input_obj.obs_before,
                extract_action_function_name,
            )
            if recovered is not None:
                deduped.append(recovered)

    if singleton_guard_reason:
        viability_events.append({
            'tick': input_obj.tick,
            'episode': input_obj.episode,
            'event': 'singleton_skip_layer1_blocked',
            'reason': singleton_guard_reason,
            'raw_candidate_count': len(input_obj.raw_candidates) if isinstance(input_obj.raw_candidates, list) else 0,
            'visible_functions': sorted(visible_functions),
        })
    must_enforce_non_wait = (
        not pending_wait_only_governance
        and _must_enforce_non_wait_viability(
            raw_candidates=input_obj.raw_candidates,
            obs_before=input_obj.obs_before,
            tick=input_obj.tick,
            episode_trace=input_obj.episode_trace,
        )
    )

    if must_enforce_non_wait and not any(str(c.get('function_name') or '').strip() != 'wait' for c in deduped):
        recovered = _recover_viable_non_wait_from_raw(input_obj.raw_candidates, input_obj.obs_before, extract_action_function_name)
        if recovered is not None:
            recovered['viability_recovered_from_raw'] = True
            deduped.append(recovered)
            for candidate in deduped:
                if str(candidate.get('function_name') or '').strip() != 'wait':
                    candidate['viability_recovered_from_raw'] = True
                    break
        else:
            viability_events.append({
                'tick': input_obj.tick,
                'episode': input_obj.episode,
                'event': 'governance_candidate_viability_failure',
                'reason': 'missing_non_wait_after_normalization',
                'raw_candidate_count': len(input_obj.raw_candidates) if isinstance(input_obj.raw_candidates, list) else 0,
                'visible_functions': sorted(visible_functions),
            })

    current_non_wait_functions = {
        str(candidate.get('function_name') or '').strip()
        for candidate in deduped
        if isinstance(candidate, dict) and str(candidate.get('function_name') or '').strip() not in {'', 'wait'}
    }
    recovered_procedure_candidate = None
    if not current_non_wait_functions and not pending_wait_only_governance:
        recovered_candidate = _recover_viable_non_wait_from_raw(
            input_obj.raw_candidates,
            input_obj.obs_before,
            extract_action_function_name,
        )
        if (
            isinstance(recovered_candidate, dict)
            and _action_has_procedure_support(recovered_candidate.get('raw_action', {}))
        ):
            recovered_candidate['viability_recovered_from_raw'] = True
            recovered_candidate['_normalization_source'] = 'procedure_backed_recovery'
            deduped.append(recovered_candidate)
            recovered_procedure_candidate = recovered_candidate
            current_non_wait_functions.add(str(recovered_candidate.get('function_name') or '').strip())
            viability_events.append({
                'tick': input_obj.tick,
                'episode': input_obj.episode,
                'event': 'procedure_backed_candidate_recovered_from_raw',
                'recovered_function': str(recovered_candidate.get('function_name') or '').strip(),
                'visible_functions': list(visible_function_list),
            })
    surface_anchor_only = (
        not cooldown_locked_functions
        and not has_failure_feedback
        and (
            bool(singleton_guard_reason)
            or (len(visible_functions) == 1 and len(current_non_wait_functions) == 1)
        )
    )
    visible_surface_candidates = [] if pending_wait_only_governance else _synthesize_visible_surface_candidates(
        deduped,
        input_obj.obs_before,
        input_obj.episode_trace,
        extract_action_function_name,
        anchor_function=str(
            _extract_action_name_text(input_obj.selected_action or {}, extract_action_function_name)
            or next(
                (
                    str(candidate.get('function_name') or '').strip()
                    for candidate in input_obj.governance_candidates
                    if isinstance(candidate, dict) and str(candidate.get('function_name') or '').strip() in cooldown_locked_functions
                ),
                '',
            )
        ),
        anchor_only=surface_anchor_only,
        blocked_functions=blocked_functions.union(cooldown_locked_functions),
    )
    if visible_surface_candidates and (
        cooldown_locked_functions
        or has_failure_feedback
        or (not current_non_wait_functions and recovered_procedure_candidate is None)
        or surface_anchor_only
    ):
        deduped.extend(visible_surface_candidates)
        viability_events.append({
            'tick': input_obj.tick,
            'episode': input_obj.episode,
            'event': 'surface_diversity_recovery_candidates_added',
            'added_functions': [
                str(candidate.get('function_name') or '').strip()
                for candidate in visible_surface_candidates
            ],
            'blocked_functions': sorted(blocked_functions.union(cooldown_locked_functions)),
        })

    valid = [c for c in deduped if isinstance(c, dict) and str(c.get('function_name') or '').strip()]
    if not valid:
        fallback = _make_wait_governance_candidate(intent='constraint_empty_fallback')
        fallback['_normalization_source'] = 'constraint_empty_fallback'
        valid = [fallback]

    if _must_bias_to_external_actions(
        obs_before=input_obj.obs_before,
        tick=input_obj.tick,
        episode_trace=input_obj.episode_trace,
    ):
        external_valid = [
            candidate
            for candidate in valid
            if not _is_internal_only_governance_candidate(candidate, extract_action_function_name)
        ]
        if external_valid:
            suppressed = [
                str(candidate.get('function_name') or candidate.get('action') or '')
                for candidate in valid
                if _is_internal_only_governance_candidate(candidate, extract_action_function_name)
            ]
            if suppressed:
                viability_events.append({
                    'tick': input_obj.tick,
                    'episode': input_obj.episode,
                    'event': 'internal_only_candidates_suppressed',
                    'reason': 'early_surface_action_bias',
                    'suppressed_actions': suppressed,
                    'visible_functions': sorted(visible_functions),
                })
                valid = external_valid

    return NormalizationResult(
        unique_candidate_actions=unique_candidate_actions,
        governance_candidates_before_normalization=governance_candidates_before_normalization,
        governance_candidates_after_normalization=valid,
        skip_layer1=skip_layer1,
        skip_reason=skip_reason,
        viability_events=viability_events,
    )

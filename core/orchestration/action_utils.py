"""Pure action/context helpers used by CoreMainLoop orchestration."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


_ORDINAL_ACTION_RE = re.compile(r"^ACTION(\d+)$", re.IGNORECASE)


def extract_action_function_name(action: Optional[Dict[str, Any]], default: str = "wait") -> str:
    if not isinstance(action, dict):
        return default
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    candidates = [
        tool_args.get('function_name'),
        payload.get('function_name'),
        action.get('function_name'),
        (action.get('tool_args', {}) or {}).get('function_name') if isinstance(action.get('tool_args', {}), dict) else None,
        action.get('selected_name'),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    if action.get('kind') == 'wait':
        return 'wait'
    if action.get('kind') == 'inspect':
        return 'inspect'
    return default


def extract_action_kind(action: Optional[Dict[str, Any]], default: str = "call_tool") -> str:
    """Return canonical action kind shared by governance/scoring/trace pipelines."""
    if not isinstance(action, dict):
        return default
    function_name = extract_action_function_name(action, default='').strip()
    raw_kind = str(action.get('kind', '') or '').strip().lower()
    if function_name == 'wait' or raw_kind == 'wait':
        return 'wait'
    if function_name == 'inspect' or raw_kind == 'inspect':
        return 'inspect'
    if function_name == 'probe' or raw_kind == 'probe':
        return 'probe'
    return 'call_tool'


def repair_action_function_name(action: Optional[Dict[str, Any]], selected_name: str) -> Dict[str, Any]:
    """Ensure governance-selected action carries a callable function_name for trace/logging."""
    if not isinstance(action, dict):
        return {'kind': 'wait', 'payload': {}}
    chosen = str(selected_name or '').strip()
    if not chosen or chosen == 'wait':
        return action
    if extract_action_function_name(action, default=''):
        return action
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    patched_action = dict(action)
    patched_tool_args = dict(tool_args)
    patched_tool_args['function_name'] = chosen
    patched_payload = dict(payload)
    patched_payload['tool_args'] = patched_tool_args
    patched_action['payload'] = patched_payload
    return patched_action


def candidate_counts(candidates: Any) -> Tuple[int, int]:
    if not isinstance(candidates, list):
        return 0, 0
    total = len(candidates)
    non_wait = 0
    for candidate in candidates:
        fn_name = extract_action_function_name(candidate, default='wait')
        if fn_name != 'wait':
            non_wait += 1
    return total, non_wait


def candidate_dedupe_signature(action: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(action, dict):
        return None
    action_identity = extract_action_identity(action, include_function_fallback=True).strip()
    if not action_identity:
        return None
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    try:
        param_signature = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        param_signature = str(sorted((str(k), str(v)) for k, v in kwargs.items()))
    return action_identity, param_signature


def extract_action_xy(action: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not isinstance(action, dict):
        return None
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    x = kwargs.get('x') if isinstance(kwargs, dict) else None
    y = kwargs.get('y') if isinstance(kwargs, dict) else None
    if _valid_coord(x) and _valid_coord(y):
        return int(x), int(y)

    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    resolution = meta.get('execution_kwarg_resolution', {}) if isinstance(meta.get('execution_kwarg_resolution', {}), dict) else {}
    selected_point = resolution.get('selected_point', {}) if isinstance(resolution.get('selected_point', {}), dict) else {}
    rx = selected_point.get('x')
    ry = selected_point.get('y')
    if _valid_coord(rx) and _valid_coord(ry):
        return int(rx), int(ry)
    return None


def extract_action_click_identity(action: Optional[Dict[str, Any]]) -> str:
    fn_name = extract_action_function_name(action, default='').strip().upper()
    if fn_name != 'ACTION6':
        return ''
    xy = extract_action_xy(action)
    if xy is None:
        return ''
    return f'ACTION6@{xy[0]},{xy[1]}'


def extract_action_identity(action: Optional[Dict[str, Any]], *, include_function_fallback: bool = True) -> str:
    fn_name = extract_action_function_name(action, default='').strip()
    if not fn_name:
        return ''
    click_identity = extract_action_click_identity(action)
    if click_identity:
        return click_identity
    if fn_name.upper() == 'ACTION6' and not include_function_fallback:
        return ''
    return fn_name


def extract_action_signature_kwargs(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Backfill rollout-side semantic fields into kwargs so identity stays stable across pipelines."""
    if not isinstance(action, dict):
        return {}
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    merged = dict(kwargs)
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    intervention_target = meta.get('intervention_target', {}) if isinstance(meta.get('intervention_target', {}), dict) else {}
    resolution = meta.get('execution_kwarg_resolution', {}) if isinstance(meta.get('execution_kwarg_resolution', {}), dict) else {}
    selected_point = resolution.get('selected_point', {}) if isinstance(resolution.get('selected_point', {}), dict) else {}

    backfills = {
        'x': _normalize_signature_number(
            _first_signature_value(
                kwargs.get('x'),
                tool_args.get('x'),
                payload.get('x'),
                action.get('x'),
                meta.get('x'),
                selected_point.get('x'),
            )
        ),
        'y': _normalize_signature_number(
            _first_signature_value(
                kwargs.get('y'),
                tool_args.get('y'),
                payload.get('y'),
                action.get('y'),
                meta.get('y'),
                selected_point.get('y'),
            )
        ),
        'target_family': _first_signature_value(
            kwargs.get('target_family'),
            tool_args.get('target_family'),
            payload.get('target_family'),
            action.get('target_family'),
            meta.get('target_family'),
            intervention_target.get('target_family'),
            intervention_target.get('target_kind'),
        ),
        'relation_type': _first_signature_value(
            kwargs.get('relation_type'),
            tool_args.get('relation_type'),
            payload.get('relation_type'),
            action.get('relation_type'),
            meta.get('relation_type'),
            meta.get('goal_progress_relation_type'),
        ),
        'anchor_ref': _first_signature_value(
            kwargs.get('anchor_ref'),
            tool_args.get('anchor_ref'),
            payload.get('anchor_ref'),
            action.get('anchor_ref'),
            meta.get('anchor_ref'),
            intervention_target.get('anchor_ref'),
        ),
    }
    for key, value in backfills.items():
        if _signature_value_present(value):
            merged[key] = value
    return merged


def action_semantic_signature(action: Optional[Dict[str, Any]]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    fn_name = extract_action_function_name(action, default='')
    kwargs = extract_action_signature_kwargs(action)
    normalized_kwargs = tuple(
        sorted((str(key), _canonical_signature_value(value)) for key, value in kwargs.items())
    )
    return str(fn_name or '').strip(), normalized_kwargs


def serialize_action_semantic_signature(signature: Tuple[str, Tuple[Tuple[str, str], ...]]) -> str:
    fn_name, normalized_kwargs = signature
    return json.dumps(
        {
            "function_name": str(fn_name or '').strip(),
            "signature": [[str(key), str(value)] for key, value in list(normalized_kwargs or ())],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def action_semantic_signature_key(action: Optional[Dict[str, Any]]) -> str:
    return serialize_action_semantic_signature(action_semantic_signature(action))


def action_matches_blocked_name(action: Optional[Dict[str, Any]], blocked_names: Any) -> bool:
    blocked_pool = blocked_names if isinstance(blocked_names, (list, set, tuple)) else []
    blocked = {str(value or '').strip() for value in blocked_pool if str(value or '').strip()}
    if not blocked:
        return False
    click_identity = extract_action_click_identity(action)
    if click_identity:
        return click_identity in blocked
    fn_name = extract_action_function_name(action, default='').strip()
    return bool(fn_name and fn_name in blocked)


def _valid_coord(value: Any) -> bool:
    return isinstance(value, (int, float)) and int(value) >= 0


def _first_signature_value(*values: Any) -> Any:
    for value in values:
        if _signature_value_present(value):
            return value
    return None


def _signature_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _normalize_signature_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonical_signature_value(value: Any) -> str:
    normalized = _normalize_signature_number(value)
    try:
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(normalized)


def extract_known_functions(obs: Dict[str, Any]) -> List[str]:
    api_raw = obs.get('novel_api', {})
    if hasattr(api_raw, 'raw'):
        api_raw = api_raw.raw
    discovered = list(api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else [])
    visible = list(api_raw.get('visible_functions', []) if isinstance(api_raw, dict) else [])
    known = []
    for fn in discovered + visible:
        if fn and fn not in known:
            known.append(fn)
    return known


def _ordinal_action_bias_reorder(functions: List[str]) -> List[str]:
    """Break deterministic ACTION1-first bias for generic ordinal action surfaces.

    ARC-like environments often expose ACTION1, ACTION2, ... in a stable order.
    Returning that order unchanged makes any available[0]-style generator collapse
    onto ACTION1. We preserve surface order for non-ordinal actions, but when the
    entire frontier is just ACTION<n> names we move ACTION1 to the end so cold-start
    behavior does not hard-code a single default winner.
    """
    if len(functions) < 2:
        return list(functions)
    if not all(_ORDINAL_ACTION_RE.match(str(fn or '').strip()) for fn in functions):
        return list(functions)
    non_action1 = [fn for fn in functions if str(fn).strip().upper() != 'ACTION1']
    action1 = [fn for fn in functions if str(fn).strip().upper() == 'ACTION1']
    if not non_action1:
        return list(functions)
    return non_action1 + action1


def extract_available_functions(obs: Dict[str, Any]) -> List[str]:
    """Extract available function names from observation surface."""
    api_raw = obs.get('novel_api', {})
    if hasattr(api_raw, 'raw'):
        api_raw = api_raw.raw

    ordered: List[str] = []

    def _append_many(values: Any) -> None:
        if isinstance(values, dict):
            values = list(values.keys())
        if not isinstance(values, list):
            return
        for item in values:
            if isinstance(item, dict):
                fn_name = item.get('name') or item.get('function_name')
            else:
                fn_name = item
            if not fn_name:
                continue
            fn_name = str(fn_name)
            if fn_name not in ordered:
                ordered.append(fn_name)

    if isinstance(api_raw, dict):
        _append_many(api_raw.get('visible_functions', []))
        _append_many(api_raw.get('discovered_functions', []))
        _append_many(api_raw.get('function_aliases', {}))

    _append_many(obs.get('backend_functions', {}))
    _append_many(obs.get('function_signatures', {}))
    _append_many(obs.get('available_functions', []))
    return _ordinal_action_bias_reorder(ordered)


def build_kwargs_from_context(
    fn: str,
    *,
    obs: Dict[str, Any],
    continuity_snapshot: Optional[Dict[str, Any]],
    tick: int,
    episode: int,
    plan_step_intent: Any,
    plan_step_target: Any,
    plan_revision_count: int,
    recent_reward: float,
) -> Dict[str, Any]:
    """Build function kwargs from observation/plan/history context instead of static templates."""
    retrieved = []
    if isinstance(obs.get('retrieved_objects'), list):
        retrieved = [obj for obj in obs.get('retrieved_objects', []) if isinstance(obj, dict)][:3]
    perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
    return {
        'function_name_hint': fn,
        'tick': tick,
        'episode': episode,
        'plan_step_intent': plan_step_intent,
        'plan_step_target': plan_step_target,
        'goal_id': getattr((continuity_snapshot or {}).get('top_goal'), 'goal_id', None),
        'recent_reward': float(recent_reward),
        'retrieved_objects': retrieved,
        'perception': perception,
        'plan_revision_count': plan_revision_count,
    }

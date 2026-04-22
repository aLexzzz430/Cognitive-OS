"""Utilities for audit logging and JSON-safe payload normalization."""

from __future__ import annotations

import hashlib
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, Optional


StateWriter = Callable[[str, Dict[str, Any]], None]


def cooldown_ready(current_tick: int, last_tick: int, cooldown_ticks: int) -> bool:
    return (current_tick - last_tick) >= max(0, cooldown_ticks)


def compute_observation_signature(obs_before: Dict[str, Any]) -> str:
    api_raw = obs_before.get('novel_api', {})
    if hasattr(api_raw, 'raw'):
        api_raw = api_raw.raw
    discovered = list(api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else [])
    visible = list(api_raw.get('visible_functions', []) if isinstance(api_raw, dict) else [])
    world_state = obs_before.get('world_state', {}) if isinstance(obs_before.get('world_state'), dict) else {}
    signature_payload = {
        'state': world_state.get('state', obs_before.get('state', '')),
        'levels_completed': world_state.get('levels_completed', obs_before.get('levels_completed', 0)),
        'win_levels': world_state.get('win_levels', obs_before.get('win_levels', 0)),
        'discovered': discovered,
        'visible': visible,
        'backend': sorted(list(obs_before.get('backend_functions', {}).keys())) if isinstance(obs_before.get('backend_functions'), dict) else [],
    }
    return hashlib.md5(repr(signature_payload).encode()).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert runtime objects to JSON-safe structures for audit logs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        safe_dict: Dict[str, Any] = {}
        for key, item in value.items():
            safe_dict[str(key)] = json_safe(item)
        return safe_dict
    if isinstance(value, (list, tuple, set, deque)):
        return [json_safe(item) for item in value]
    if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
        try:
            return json_safe(value.to_dict())
        except Exception:
            return str(value)
    if hasattr(value, '__dict__'):
        try:
            return json_safe(vars(value))
        except Exception:
            return str(value)
    return str(value)


def record_budgeted_llm_call(
    kind: str,
    *,
    llm_client: Optional[Any],
    episode: int,
    tick: int,
    llm_calls_this_tick: int,
    state_writer: StateWriter,
) -> int:
    if llm_client is None:
        return llm_calls_this_tick
    next_calls = llm_calls_this_tick + 1
    state_writer('llm_advice_log', {
        'tick': tick,
        'episode': episode,
        'kind': kind,
        'entry': 'budgeted_llm_call',
    })
    return next_calls


def record_llm_tick_summary(
    *,
    episode: int,
    tick: int,
    llm_calls_this_tick: int,
    state_writer: StateWriter,
) -> None:
    state_writer('llm_calls_per_tick', {
        'episode': episode,
        'tick': tick,
        'count': llm_calls_this_tick,
    })


def record_continuity_tick(
    continuity_snapshot: Dict[str, Any],
    *,
    episode: int,
    tick: int,
    state_writer: StateWriter,
) -> None:
    next_task = continuity_snapshot.get('next_task')
    top_goal = continuity_snapshot.get('top_goal')
    state_writer('continuity_log', {
        'tick': tick,
        'episode': episode,
        'active_goal_count': continuity_snapshot.get('active_goal_count', 0),
        'running_experiments': continuity_snapshot.get('running_experiments', 0),
        'next_task': getattr(next_task, 'task_id', None),
        'top_goal': getattr(top_goal, 'goal_id', None),
    })

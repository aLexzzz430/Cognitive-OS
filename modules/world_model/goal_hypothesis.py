
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _top_objects(object_bindings_summary: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
    rows = [item for item in _as_list(object_bindings_summary.get('objects', [])) if isinstance(item, dict)]
    rows.sort(key=lambda item: (-float(item.get('actionable_score', 0.0) or 0.0), -float(item.get('salience_score', 0.0) or 0.0), str(item.get('object_id', ''))))
    return rows[:limit]


def build_goal_hypotheses(
    obs: Dict[str, Any],
    task_frame_summary: Dict[str, Any],
    object_bindings_summary: Dict[str, Any],
    episode_trace_tail: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    task_frame_summary = _as_dict(task_frame_summary)
    object_bindings_summary = _as_dict(object_bindings_summary)
    trace_tail = [row for row in _as_list(episode_trace_tail) if isinstance(row, dict)]
    candidates = [row for row in _as_list(task_frame_summary.get('goal_family_candidates', [])) if isinstance(row, dict)]
    top_objects = _top_objects(object_bindings_summary, limit=4)
    top_object_ids = [str(item.get('object_id', '')) for item in top_objects if str(item.get('object_id', ''))]
    recent_negative = sum(1 for row in trace_tail if float(row.get('reward', 0.0) or 0.0) < 0.0)
    hypotheses: List[Dict[str, Any]] = []

    family_to_template = {
        'select_or_activate_salient_structures': (
            'Goal likely requires selecting or activating one of the salient scene structures.',
            'probe most actionable object to test whether state changes cluster around salient structures',
            ['pointer_interaction', 'confirm_interaction'],
        ),
        'reveal_hidden_state_via_probe': (
            'Goal may require probing uncertain regions to reveal hidden state or unlock a mode switch.',
            'click or inspect distinctive structures that maximize information gain rather than immediate reward',
            ['pointer_interaction', 'probe'],
        ),
        'navigate_agent_or_focus_to_goal': (
            'Goal likely depends on moving focus/agent toward a target-bearing region before confirmation.',
            'use navigation to test whether controllable focus can be repositioned toward salient objects',
            ['navigation_interaction', 'confirm_interaction'],
        ),
        'commit_or_confirm_world_state': (
            'Goal may require assembling a valid intermediate state and then explicitly confirming it.',
            'test whether confirmation changes the phase only after one or more object interactions',
            ['confirm_interaction'],
        ),
        'arrange_or_transform_object_configuration': (
            'Goal likely depends on changing relations among scene objects rather than touching a single point.',
            'prefer actions that can alter object alignment, activation order, or scene configuration',
            ['navigation_interaction', 'pointer_interaction', 'state_transform_interaction'],
        ),
    }

    for idx, candidate in enumerate(candidates[:5]):
        family = str(candidate.get('family', '') or '')
        template = family_to_template.get(family)
        if not template:
            continue
        statement, test_description, action_families = template
        confidence = float(candidate.get('confidence', 0.0) or 0.0)
        if recent_negative >= 2 and family == 'reveal_hidden_state_via_probe':
            confidence += 0.05
        preferred_target_refs = top_object_ids[:2] if family != 'navigate_agent_or_focus_to_goal' else top_object_ids[:1]
        hypotheses.append({
            'hypothesis_id': f'goal_h{idx+1}',
            'family': family,
            'statement': statement,
            'confidence': round(min(0.98, confidence), 4),
            'preferred_action_families': action_families,
            'preferred_target_refs': preferred_target_refs,
            'discriminating_test': test_description,
            'supporting_signals': [
                str(task_frame_summary.get('dominant_interaction_mode', '') or ''),
                family,
            ] + top_object_ids[:2],
        })

    if not hypotheses:
        hypotheses.append({
            'hypothesis_id': 'goal_h1',
            'family': 'generic_progress_discovery',
            'statement': 'Goal is unknown; prioritize actions that maximize information gain about controllable structures.',
            'confidence': 0.35,
            'preferred_action_families': [str(task_frame_summary.get('dominant_interaction_mode', '') or 'pointer_interaction')],
            'preferred_target_refs': top_object_ids[:2],
            'discriminating_test': 'compare whether interacting with salient structures changes scene state or reveals new affordances',
            'supporting_signals': [str(task_frame_summary.get('frame_type', '') or '')],
        })

    hypotheses.sort(key=lambda item: (-float(item.get('confidence', 0.0) or 0.0), str(item.get('hypothesis_id', ''))))
    return hypotheses[:4]


def summarize_solver_state(
    task_frame_summary: Dict[str, Any],
    object_bindings_summary: Dict[str, Any],
    goal_hypotheses_summary: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    hypotheses = [row for row in goal_hypotheses_summary if isinstance(row, dict)]
    top = hypotheses[0] if hypotheses else {}
    unresolved_dimensions: List[str] = []
    if not _as_list(object_bindings_summary.get('salient_object_ids', [])):
        unresolved_dimensions.append('no_salient_objects')
    if not hypotheses:
        unresolved_dimensions.append('no_goal_hypotheses')
    if str(task_frame_summary.get('dominant_interaction_mode', '') or '') in {'unknown', ''}:
        unresolved_dimensions.append('interaction_mode_unclear')

    return {
        'dominant_goal_family': str(top.get('family', '') or ''),
        'dominant_goal_confidence': float(top.get('confidence', 0.0) or 0.0),
        'preferred_target_refs': list(top.get('preferred_target_refs', []) or []),
        'unresolved_dimensions': unresolved_dimensions,
        'solver_ready': not unresolved_dimensions,
    }

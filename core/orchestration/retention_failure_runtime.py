from __future__ import annotations

from typing import Any, Dict, List, Optional


_GOVERNANCE_ENTRY_MARKERS = {
    'counterfactual_adoption_metric',
    'prediction_high_error_retrieval_pressure',
    'prediction_replan_hint',
}

_PLANNER_TARGET_SWITCH_EVENTS = {
    'wm_branch_salvage_replan',
    'wm_belief_branch_replan',
    'wm_rollout_value_replan',
    'wm_value_drop_replan',
}


def latest_governance_entry_for_tick(
    governance_log: List[Any],
    *,
    episode: int,
    tick: int,
) -> Dict[str, Any]:
    for row in reversed(list(governance_log or [])[-40:]):
        if not isinstance(row, dict):
            continue
        if int(row.get('episode', -1)) != int(episode) or int(row.get('tick', -1)) != int(tick):
            continue
        if (
            row.get('reason')
            or row.get('selected_name')
            or row.get('selected')
            or row.get('entry') in _GOVERNANCE_ENTRY_MARKERS
        ):
            return row
    return {}


def latest_plan_lookahead_telemetry(
    *,
    last_planner_runtime_payload: Any,
    planner_runtime_log: List[Any],
    episode: int,
    tick: int,
) -> Dict[str, Any]:
    payload = last_planner_runtime_payload if isinstance(last_planner_runtime_payload, dict) else {}
    if (
        isinstance(payload, dict)
        and int(payload.get('episode', -1)) == int(episode)
        and int(payload.get('tick', -1)) == int(tick)
    ):
        telemetry = payload.get('telemetry', {}) if isinstance(payload.get('telemetry', {}), dict) else {}
        lookahead = telemetry.get('plan_lookahead', {}) if isinstance(telemetry.get('plan_lookahead', {}), dict) else {}
        if lookahead:
            return dict(lookahead)

    for row in reversed(list(planner_runtime_log or [])[-20:]):
        if not isinstance(row, dict):
            continue
        if int(row.get('episode', -1)) != int(episode) or int(row.get('tick', -1)) != int(tick):
            continue
        telemetry = row.get('telemetry', {}) if isinstance(row.get('telemetry', {}), dict) else {}
        lookahead = telemetry.get('plan_lookahead', {}) if isinstance(telemetry.get('plan_lookahead', {}), dict) else {}
        if lookahead:
            return dict(lookahead)
    return {}


def classify_retention_failure(
    *,
    reward: float,
    prediction_mismatch: float,
    task_family: str,
    phase: str,
    observation_mode: str,
    resource_band: str,
    action_name: str,
    lookahead: Optional[Dict[str, Any]] = None,
    governance_entry: Optional[Dict[str, Any]] = None,
    action_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lookahead = dict(lookahead or {})
    governance_entry = dict(governance_entry or {})
    meta = dict(action_meta or {})
    mismatch = max(0.0, min(1.0, float(prediction_mismatch or 0.0)))
    forced_replan_events = [
        str(event or '')
        for event in list(lookahead.get('forced_replan_events', []) or [])
        if str(event or '')
    ]
    forced_event_set = set(forced_replan_events)
    persistence_ratio = max(0.0, min(1.0, float(lookahead.get('rollout_branch_persistence_ratio', 0.0) or 0.0)))
    governance_reason = str(governance_entry.get('reason', '') or '')
    raw_counterfactual_confidence = meta.get('counterfactual_confidence', 0.0)
    if isinstance(raw_counterfactual_confidence, (int, float)):
        counterfactual_confidence = max(0.0, min(1.0, float(raw_counterfactual_confidence)))
    else:
        counterfactual_confidence = {
            'high': 0.9,
            'medium': 0.6,
            'low': 0.3,
        }.get(str(raw_counterfactual_confidence or '').strip().lower(), 0.0)
    counterfactual_advantage = bool(meta.get('counterfactual_advantage', False))
    resolved_action_name = str(action_name or 'wait')

    failure_type = ''
    severity = 0.0
    if (
        (not counterfactual_advantage and counterfactual_confidence >= 0.72)
        or 'counterfactual_oppose' in governance_reason
    ) and (float(reward) < 0.0 or mismatch >= 0.45):
        failure_type = 'governance_overrule_misfire'
        severity = max(mismatch, 0.35 + counterfactual_confidence * 0.55)
    elif (
        'wm_branch_persistence_replan' in forced_event_set
        or persistence_ratio <= 0.30
        or str(lookahead.get('rollout_final_phase', '') or '') == 'disrupted'
    ):
        failure_type = 'branch_persistence_collapse'
        severity = max(mismatch * 0.55, 1.0 - persistence_ratio)
        if str(lookahead.get('rollout_final_phase', '') or '') == 'disrupted':
            severity = max(severity, 0.68)
    elif forced_event_set.intersection(_PLANNER_TARGET_SWITCH_EVENTS):
        failure_type = 'planner_target_switch'
        severity = max(0.38, mismatch * 0.60 + min(0.3, len(forced_event_set) * 0.08))
    elif mismatch >= 0.35 or 'prediction_high_error_retrieval_pressure' == str(governance_entry.get('entry', '') or ''):
        failure_type = 'prediction_drift'
        severity = mismatch

    severity = max(0.0, min(1.0, severity))
    if not failure_type or (float(reward) >= 0.0 and mismatch < 0.35 and not forced_event_set):
        return {'failure_type': '', 'severity': 0.0, 'context': {}}

    base_context_key = (
        f"task_family={str(task_family or 'unknown')}|phase={str(phase or 'unknown')}"
        f"|observation_mode={str(observation_mode or 'unknown')}|resource_band={str(resource_band or 'normal')}"
    )
    strategy_mode_hint = 'recover'
    if failure_type in {'prediction_drift', 'governance_overrule_misfire'}:
        strategy_mode_hint = 'verify'
    elif failure_type == 'planner_target_switch':
        strategy_mode_hint = 'recover'

    context = {
        'context_key': f"{base_context_key}|failure={failure_type}",
        'base_context_key': base_context_key,
        'failure_type': failure_type,
        'severity': severity,
        'strategy_mode_hint': strategy_mode_hint,
        'branch_budget_hint': 2 if failure_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0,
        'verification_budget_hint': 2 if failure_type in {'prediction_drift', 'governance_overrule_misfire'} else (1 if failure_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0),
        'forced_replan_events': list(forced_replan_events),
        'rollout_branch_persistence_ratio': persistence_ratio,
        'rollout_branch_id': str(lookahead.get('rollout_branch_id', '') or ''),
        'rollout_branch_target_phase': str(lookahead.get('rollout_branch_target_phase', '') or ''),
        'rollout_final_phase': str(lookahead.get('rollout_final_phase', '') or ''),
        'governance_reason': governance_reason,
        'selected_name': str(governance_entry.get('selected_name', governance_entry.get('selected', resolved_action_name)) or resolved_action_name),
        'counterfactual_confidence': counterfactual_confidence,
        'counterfactual_advantage': counterfactual_advantage,
    }
    return {'failure_type': failure_type, 'severity': severity, 'context': context}

from __future__ import annotations

from typing import Any, Dict, List

from modules.world_model.protocol import WorldModelControlProtocol

from core.learning import aggregate_learning_updates


def refresh_learning_policy_snapshot(loop: Any) -> None:
    objects = []
    for obj in loop._shared_store.iter_objects(limit=500):
        if not isinstance(obj, dict):
            continue
        if str(obj.get('memory_type', '')) != 'learning_update':
            continue
        if obj.get('status') == 'invalidated':
            continue
        objects.append(obj)
    loop._learning_policy_snapshot = aggregate_learning_updates(objects)
    if hasattr(loop, '_reliability_tracker') and hasattr(loop._reliability_tracker, 'synchronize_failure_preference_learning'):
        loop._reliability_tracker.synchronize_failure_preference_learning(
            loop._learning_policy_snapshot.get('failure_preference_policy', {})
            if isinstance(loop._learning_policy_snapshot.get('failure_preference_policy', {}), dict)
            else {}
        )


def clamp_learning_signal(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return max(minimum, min(maximum, float(default)))


def learning_merge_ordered_lists(*values: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in values:
        pool = value if isinstance(value, list) else []
        for item in pool:
            text = str(item or '').strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def is_learning_verification_function(fn_name: str) -> bool:
    name = str(fn_name or '').strip().lower()
    return bool(name) and any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test'))


def learning_resource_band(decision_context: Dict[str, Any]) -> str:
    if not isinstance(decision_context, dict):
        return 'normal'
    explicit = str(decision_context.get('resource_band', '') or '').strip().lower()
    if explicit in {'tight', 'normal'}:
        return explicit
    pressure = str(decision_context.get('resource_pressure', '') or '').strip().lower()
    if pressure in {'high', 'tight', 'critical'}:
        return 'tight'
    if pressure in {'normal', 'low', 'relaxed'}:
        return 'normal'
    self_model_summary = decision_context.get('self_model_summary', {}) if isinstance(decision_context.get('self_model_summary', {}), dict) else {}
    fallback = str(self_model_summary.get('resource_tightness', '') or '').strip().lower()
    if fallback in {'high', 'tight', 'critical'}:
        return 'tight'
    return 'normal'


def learning_world_model_competition_profile(
    decision_context: Dict[str, Any],
    *,
    candidate_function_universe: List[str],
) -> Dict[str, Any]:
    wm_control = WorldModelControlProtocol.from_context(decision_context if isinstance(decision_context, dict) else {})
    latent_branches = list(wm_control.latent_branches or [])
    dominant_branch_id = str(wm_control.dominant_branch_id or '').strip()
    dominant_branch: Dict[str, Any] = {}
    for row in latent_branches:
        if not isinstance(row, dict):
            continue
        if dominant_branch_id and str(row.get('branch_id', '') or '').strip() == dominant_branch_id:
            dominant_branch = dict(row)
            break
    if not dominant_branch and latent_branches and isinstance(latent_branches[0], dict):
        dominant_branch = dict(latent_branches[0])

    dominant_branch_id = str(dominant_branch.get('branch_id', dominant_branch_id) or dominant_branch_id or '')
    dominant_branch_confidence = clamp_learning_signal(dominant_branch.get('confidence', 0.0), 0.0, 1.0, 0.0)
    required_probes = learning_merge_ordered_lists(list(wm_control.required_probes or []))
    dominant_anchor_functions = [
        fn_name
        for fn_name in learning_merge_ordered_lists(
            dominant_branch.get('anchor_functions', []),
            dominant_branch.get('anchored_functions', []),
        )
        if fn_name in candidate_function_universe
    ]
    dominant_risky_functions = [
        fn_name
        for fn_name in learning_merge_ordered_lists(dominant_branch.get('risky_functions', []))
        if fn_name in candidate_function_universe
    ]
    probe_pressure = min(1.0, len(required_probes) / 3.0)
    latent_instability = clamp_learning_signal(
        (1.0 - dominant_branch_confidence) * 0.38
        + clamp_learning_signal(wm_control.hidden_drift_score, 0.0, 1.0, 0.0) * 0.28
        + clamp_learning_signal(wm_control.hidden_uncertainty_score, 0.0, 1.0, 0.0) * 0.22
        + clamp_learning_signal(wm_control.state_shift_risk, 0.0, 1.0, 0.0) * 0.12,
        0.0,
        1.0,
        0.0,
    )
    probe_pressure_active = (
        probe_pressure >= 0.34
        and (
            clamp_learning_signal(wm_control.control_trust, 0.0, 1.0, 0.5) <= 0.52
            or clamp_learning_signal(wm_control.transition_confidence, 0.0, 1.0, 0.5) <= 0.48
            or clamp_learning_signal(wm_control.hidden_drift_score, 0.0, 1.0, 0.0) >= 0.55
            or clamp_learning_signal(wm_control.hidden_uncertainty_score, 0.0, 1.0, 0.0) >= 0.62
            or latent_instability >= 0.58
            or clamp_learning_signal(wm_control.state_shift_risk, 0.0, 1.0, 0.0) >= 0.58
        )
    )
    return {
        'required_probes': required_probes,
        'probe_pressure': float(probe_pressure),
        'probe_pressure_active': bool(probe_pressure_active),
        'latent_instability': float(latent_instability),
        'dominant_branch_id': dominant_branch_id,
        'dominant_anchor_functions': dominant_anchor_functions,
        'dominant_risky_functions': dominant_risky_functions,
    }


def merge_learned_failure_strategy_profile(
    existing: Dict[str, Any],
    learned: Dict[str, Any],
    *,
    competition: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(existing) if isinstance(existing, dict) else {}
    if not isinstance(learned, dict) or not learned:
        return merged

    merged['strategy_mode_hint'] = str(
        learned.get('strategy_mode_hint', merged.get('strategy_mode_hint', 'balanced')) or merged.get('strategy_mode_hint', 'balanced')
    )
    merged['branch_budget_hint'] = max(
        int(merged.get('branch_budget_hint', 0) or 0),
        int(learned.get('branch_budget_hint', 0) or 0),
        2 if float(competition.get('latent_instability', 0.0) or 0.0) >= 0.58 else 0,
    )
    merged['verification_budget_hint'] = max(
        int(merged.get('verification_budget_hint', 0) or 0),
        int(learned.get('verification_budget_hint', 0) or 0),
        1 if bool(competition.get('probe_pressure_active', False)) else 0,
    )
    merged['safe_fallback_class'] = str(
        learned.get('safe_fallback_class', merged.get('safe_fallback_class', 'wait')) or merged.get('safe_fallback_class', 'wait')
    )
    merged['preferred_verification_functions'] = learning_merge_ordered_lists(
        competition.get('required_probes', []) if bool(competition.get('probe_pressure_active', False)) else [],
        [
            fn_name
            for fn_name in list(competition.get('dominant_anchor_functions', []) or [])
            if is_learning_verification_function(fn_name)
        ],
        learned.get('preferred_verification_functions', []),
        merged.get('preferred_verification_functions', []),
    )
    merged['preferred_fallback_functions'] = learning_merge_ordered_lists(
        learned.get('preferred_fallback_functions', []),
        competition.get('dominant_anchor_functions', []),
        merged.get('preferred_fallback_functions', []),
    )
    merged['blocked_action_classes'] = learning_merge_ordered_lists(
        learned.get('blocked_action_classes', []),
        competition.get('dominant_risky_functions', []) if float(competition.get('latent_instability', 0.0) or 0.0) >= 0.55 else [],
        merged.get('blocked_action_classes', []),
    )
    merged['required_probes'] = learning_merge_ordered_lists(
        learned.get('required_probes', []),
        competition.get('required_probes', []),
    )
    merged['dominant_anchor_functions'] = learning_merge_ordered_lists(
        learned.get('dominant_anchor_functions', []),
        competition.get('dominant_anchor_functions', []),
    )
    merged['dominant_risky_functions'] = learning_merge_ordered_lists(
        learned.get('dominant_risky_functions', []),
        competition.get('dominant_risky_functions', []),
    )
    merged['probe_pressure'] = max(
        float(merged.get('probe_pressure', 0.0) or 0.0),
        float(learned.get('probe_pressure', 0.0) or 0.0),
        float(competition.get('probe_pressure', 0.0) or 0.0),
    )
    merged['latent_instability'] = max(
        float(merged.get('latent_instability', 0.0) or 0.0),
        float(learned.get('latent_instability', 0.0) or 0.0),
        float(competition.get('latent_instability', 0.0) or 0.0),
    )
    merged['dominant_branch_id'] = str(
        learned.get('dominant_branch_id', competition.get('dominant_branch_id', merged.get('dominant_branch_id', ''))) or merged.get('dominant_branch_id', '')
    )
    merged['source_action'] = str(learned.get('source_action', merged.get('source_action', '')) or merged.get('source_action', ''))
    return merged

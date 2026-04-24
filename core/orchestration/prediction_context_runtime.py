from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def build_self_model_prediction_summary(
    *,
    causal_ablation: Any = None,
    self_model_facade: Any = None,
    reliability_tracker: Any = None,
    resource_state: Any = None,
) -> Dict[str, Any]:
    include_high_level_state = bool(
        getattr(causal_ablation, 'enable_high_level_self_model', True)
    )
    if self_model_facade is not None and hasattr(self_model_facade, 'build_prediction_summary'):
        return self_model_facade.build_prediction_summary(
            resource_state=resource_state,
            include_high_level_state=include_high_level_state,
        )

    reliability_by_function: Dict[str, Any] = {}
    if reliability_tracker is not None and hasattr(reliability_tracker, 'get_reliability_by_action_type'):
        reliability_by_function = dict(reliability_tracker.get_reliability_by_action_type())

    failure_profile: List[Any] = []
    if reliability_tracker is not None and hasattr(reliability_tracker, 'get_recent_failure_profile'):
        failure_profile = list(reliability_tracker.get_recent_failure_profile(limit=8))

    budget_band = 'normal'
    if resource_state is not None and hasattr(resource_state, 'budget_band'):
        budget_band = str(resource_state.budget_band() or 'normal')

    recovery_availability = 0.5
    if reliability_tracker is not None and hasattr(reliability_tracker, 'get_overall_recovery_success_rate'):
        recovery_availability = float(reliability_tracker.get_overall_recovery_success_rate())
    recovery_availability = _clamp01(recovery_availability, default=0.5)

    global_reliability = 0.5
    if reliability_by_function:
        global_reliability = sum(float(v) for v in reliability_by_function.values()) / len(reliability_by_function)
    global_reliability = _clamp01(global_reliability, default=0.5)

    if include_high_level_state:
        self_model_state = {
            'capabilities_by_domain': {},
            'capabilities_by_condition': {},
            'known_failure_modes': failure_profile,
            'fragile_regions': [],
            'recovered_regions': [],
            'external_dependencies': ['unknown'],
            'identity_markers': {'agent_id': 'unknown', 'arm_mode': 'unknown'},
            'continuity_confidence': recovery_availability,
            'value_commitments_summary': 'unknown',
            'provenance': {
                'external_dependencies': 'default',
                'identity_markers': 'default',
                'value_commitments_summary': 'default',
                'continuity_confidence': 'inferred',
            },
        }
    else:
        self_model_state = {
            'capabilities_by_domain': {},
            'capabilities_by_condition': {},
        }

    return {
        'self_model_state': self_model_state,
        'reliability_subscores': {
            'reliability_by_function': reliability_by_function,
            'global_reliability': global_reliability,
            'recovery_availability': recovery_availability,
        },
        'reliability_by_function': reliability_by_function,
        'recent_failure_modes': failure_profile,
        'resource_tightness': budget_band,
        'budget_tight': bool(
            resource_state is not None
            and hasattr(resource_state, 'is_tight_budget')
            and resource_state.is_tight_budget()
        ),
        'high_level_state_included': include_high_level_state,
        'global_reliability': global_reliability,
        'recovery_availability': recovery_availability,
        'capability_confidence': 0.6,
        'continuity_confidence': recovery_availability,
    }


def build_recovery_prediction_context(
    *,
    pending_recovery_probe: Any = None,
    pending_replan: Any = None,
    recovery_log: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    recovery_rows = list(recovery_log or [])
    last_recovery_diagnosis = (
        recovery_rows[-1]
        if recovery_rows and isinstance(recovery_rows[-1], dict)
        else {}
    )
    return {
        'pending_recovery_probe': bool(pending_recovery_probe),
        'pending_replan': bool(pending_replan),
        'last_recovery_diagnosis': last_recovery_diagnosis,
    }

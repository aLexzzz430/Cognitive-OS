from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import inspect
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.cognition.unified_context import UnifiedCognitiveContext
from core.objects import ALL_COGNITIVE_OBJECT_TYPES
from core.main_loop_components import TickContextFrame
from core.orchestration.context_builder import UnifiedContextBuilder, UnifiedContextInput
from core.orchestration.state_abstraction import summarize_action_state, summarize_value_structure
from modules.world_model.affordance_graph import build_affordance_graph
from modules.world_model.mechanism_graph import build_mechanism_graph
from modules.world_model.object_graph import build_object_graph
from modules.world_model.rollout import build_rollout_support
from modules.world_model.test_proposal import propose_discriminating_tests

TRANSITION_PRIOR_KEY_ALIASES: Dict[str, str] = {
    'fn': 'function_name',
    'function': 'function_name',
    'name': 'function_name',
    'args_bucket': 'arg_bucket',
    'argument_bucket': 'arg_bucket',
    'phase': 'belief_phase',
    'world_model_phase': 'belief_phase',
}


def canonicalize_transition_prior_key(raw_key: Any, *, default_phase: str = 'exploring') -> Dict[str, str]:
    """Map legacy transition-prior key payloads to canonical key fields.

    Supported legacy forms:
    - dict aliases, e.g. `{"fn": "...", "args_bucket": "...", "phase": "..."}`
    - signature string, e.g. `"compute_stats|default|stabilizing"`
    - function-only string, e.g. `"compute_stats"`
    """
    function_name = ''
    arg_bucket = 'default'
    phase_default = str(default_phase or 'exploring')
    belief_phase = phase_default
    if isinstance(raw_key, dict):
        normalized = {}
        for key, value in raw_key.items():
            canonical = TRANSITION_PRIOR_KEY_ALIASES.get(str(key), str(key))
            normalized[canonical] = value
        function_name = str(normalized.get('function_name', '') or '')
        arg_bucket = str(normalized.get('arg_bucket', 'default') or 'default')
        belief_phase = str(normalized.get('belief_phase', phase_default) or phase_default)
    elif isinstance(raw_key, str) and '|' in raw_key:
        fn, bucket, phase = (raw_key.split('|') + ['default', phase_default])[:3]
        function_name = str(fn or '')
        arg_bucket = str(bucket or 'default')
        belief_phase = str(phase or phase_default)
    elif isinstance(raw_key, str):
        function_name = raw_key
    return {
        'function_name': function_name,
        'arg_bucket': arg_bucket,
        'belief_phase': belief_phase,
    }


def _safe_optional_attr(value: Any, name: str) -> Any:
    """Read optional provider attributes without triggering __getattr__ side effects."""
    try:
        static_attr = inspect.getattr_static(value, name)
    except (AttributeError, TypeError):
        return None
    if static_attr is None:
        return None
    try:
        return object.__getattribute__(value, name)
    except AttributeError:
        return None


def _safe_optional_callable(value: Any, name: str) -> Any:
    attr = _safe_optional_attr(value, name)
    return attr if callable(attr) else None


class ContextProvider(Protocol):
    def beliefs(self) -> Dict[str, Any]:
        ...

    def episode_trace(self) -> List[Dict[str, Any]]:
        ...

    def plan_snapshot(self) -> Dict[str, Any]:
        ...

    def meta_control_snapshot(
        self,
        episode: int,
        tick: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    def hypotheses_snapshot(self, limit: int = 3) -> List[Dict[str, Any]]:
        ...

    def self_model_summary(self) -> Dict[str, Any]:
        ...

    def hidden_state_summary(self) -> Dict[str, Any]:
        ...

    def extraction_function_name(self, action: Any, default: str = '') -> str:
        ...

    def retrieval_should_query(self) -> bool:
        ...

    def workspace_state(self) -> Dict[str, Any]:
        ...

    def cognitive_object_records(self, object_type: str, limit: int = 8) -> List[Dict[str, Any]]:
        ...


@dataclass(frozen=True)
class LegacyContextRuntimeInput:
    episode: int
    tick: int
    episode_reward: float
    episode_trace: List[Dict[str, Any]]
    recovery_log: List[Dict[str, Any]]
    prediction_enabled: bool
    predictor_trust: Dict[str, Any]
    procedure_enabled: bool
    policy_profile: Dict[str, Any]
    representation_profile: Dict[str, Any]
    unified_context_mode: str


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / max(1, len(values))


def _program_search_constraints(
    *,
    object_graph: Dict[str, Any],
    mechanism_graph: Dict[str, Any],
    required_probes: List[str],
    hidden_uncertainty_score: float,
) -> Dict[str, Any]:
    preferred_tags: List[str] = []
    blocked_tags: List[str] = []
    scene_summary = object_graph.get('scene_summary', {}) if isinstance(object_graph.get('scene_summary', {}), dict) else {}
    entity_count = int(scene_summary.get('entity_count', 0) or 0)
    relation_count = int(scene_summary.get('relation_count', 0) or 0)
    objects = object_graph.get('objects', []) if isinstance(object_graph.get('objects', []), list) else []
    role_blob = " ".join(
        label
        for row in objects
        if isinstance(row, dict)
        for label in list(row.get('role_labels', []) or [])
        if isinstance(label, str)
    ).lower()
    if entity_count >= 2:
        preferred_tags.extend(['object_centric', 'layout'])
    if relation_count > 0:
        preferred_tags.append('component_reorder')
    if 'marker' in role_blob or 'hint' in role_blob:
        preferred_tags.append('marker_guided')
    if hidden_uncertainty_score >= 0.55 or required_probes:
        preferred_tags.extend(['conditional_rule', 'composed'])
    if entity_count <= 1 and relation_count == 0:
        preferred_tags.extend(['identity', 'transform'])
    dominant_family = str(mechanism_graph.get('dominant_mechanism_family', '') or '')
    if 'commit' in dominant_family:
        preferred_tags.append('composed')
    if 'instability' in dominant_family:
        blocked_tags.append('constant')
    return {
        'preferred_program_tags': list(dict.fromkeys(preferred_tags)),
        'blocked_program_tags': list(dict.fromkeys(blocked_tags)),
        'constraint_strength': round(max(0.25, min(0.85, 0.35 + hidden_uncertainty_score * 0.35)), 4),
    }


def _enrich_world_model_summary_with_priors(
    world_model_summary: Dict[str, Any],
    transition_priors: Dict[str, Any],
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    rollout_support = build_rollout_support(
        world_model_summary=summary,
        transition_priors=transition_priors,
        candidate_intervention_targets=summary.get('candidate_intervention_targets', []),
        mechanism_hypotheses=summary.get('mechanism_hypotheses', []),
    )
    summary['predicted_transitions'] = [
        dict(row) for row in rollout_support.get('predicted_transitions', []) if isinstance(row, dict)
    ]
    summary['counterfactual_contrasts'] = [
        dict(row) for row in rollout_support.get('counterfactual_contrasts', []) if isinstance(row, dict)
    ]
    summary['rollout_uncertainty'] = float(rollout_support.get('rollout_uncertainty', summary.get('rollout_uncertainty', 0.5)) or 0.5)
    summary['transition_prior_signature'] = str(rollout_support.get('transition_prior_signature', '') or '')
    expected_information_gain = float(summary.get('expected_information_gain', 0.0) or 0.0)
    for row in list(summary.get('predicted_transitions', []) or []):
        if not isinstance(row, dict):
            continue
        expected_information_gain = max(expected_information_gain, float(row.get('expected_information_gain', 0.0) or 0.0))
    summary['expected_information_gain'] = round(expected_information_gain, 4)
    control_hints = dict(summary.get('control_hints', {}) or {})
    control_hints['predicted_transitions'] = [dict(row) for row in summary.get('predicted_transitions', []) if isinstance(row, dict)]
    control_hints['counterfactual_contrasts'] = [dict(row) for row in summary.get('counterfactual_contrasts', []) if isinstance(row, dict)]
    control_hints['expected_information_gain'] = float(summary.get('expected_information_gain', 0.0) or 0.0)
    control_hints['rollout_uncertainty'] = float(summary.get('rollout_uncertainty', 0.5) or 0.5)
    summary['control_hints'] = control_hints
    return summary


def _normalize_phase_label(phase: Any) -> str:
    phase_str = str(phase or '').strip().lower()
    if not phase_str:
        return ''
    aliases = {
        'explore': 'exploring',
        'exploration': 'exploring',
        'stabilize': 'stabilizing',
        'stable': 'stabilizing',
        'stability': 'stabilizing',
        'commit': 'committed',
        'committing': 'committed',
        'complete': 'committed',
        'completed': 'committed',
        'solve': 'committed',
        'solved': 'committed',
        'fail': 'disrupted',
        'failed': 'disrupted',
        'failure': 'disrupted',
        'error': 'disrupted',
        'drift': 'disrupted',
        'shift': 'disrupted',
    }
    return aliases.get(phase_str, phase_str)


def _provider_hidden_state_summary(provider: ContextProvider) -> Dict[str, Any]:
    hidden_state_summary = _safe_optional_callable(provider, 'hidden_state_summary')
    if hidden_state_summary is None:
        return {}
    raw = hidden_state_summary()
    if not isinstance(raw, dict):
        return {}
    phase_confidence = _clamp01(raw.get('phase_confidence', 0.0))
    drift_score = _clamp01(raw.get('drift_score', 0.0))
    stability_score = _clamp01(raw.get('stability_score', 0.0))
    uncertainty_score = _clamp01(raw.get('uncertainty_score', max(0.0, 1.0 - phase_confidence)))
    transition_memory = raw.get('transition_memory', {}) if isinstance(raw.get('transition_memory', {}), dict) else {}
    latent_branches = _normalize_latent_branches(raw.get('latent_branches', transition_memory.get('latent_branches', [])))
    dominant_branch_id = str(
        raw.get(
            'dominant_branch_id',
            transition_memory.get('dominant_branch_id', latent_branches[0].get('branch_id', '') if latent_branches else ''),
        ) or ''
    )
    return {
        'episode': int(raw.get('episode', 0) or 0),
        'tick': int(raw.get('tick', 0) or 0),
        'phase': _normalize_phase_label(raw.get('phase', '')) or 'exploring',
        'phase_confidence': phase_confidence,
        'hidden_state_depth': max(0, int(raw.get('hidden_state_depth', 0) or 0)),
        'drift_score': drift_score,
        'stability_score': stability_score,
        'uncertainty_score': uncertainty_score,
        'explicit_observation_phase': _normalize_phase_label(raw.get('explicit_observation_phase', '')),
        'last_function_name': str(raw.get('last_function_name', '') or ''),
        'recent_phase_path': [
            _normalize_phase_label(value) or str(value or '')
            for value in list(raw.get('recent_phase_path', []) or [])[-4:]
            if str(value or '')
        ],
        'focus_functions': [
            str(value) for value in list(raw.get('focus_functions', []) or [])[:3]
            if str(value or '')
        ],
        'latent_signature': str(raw.get('latent_signature', '') or ''),
        'dominant_branch_id': dominant_branch_id,
        'latent_branches': latent_branches,
        'expected_next_phase': _normalize_phase_label(raw.get('expected_next_phase', transition_memory.get('expected_next_phase', ''))),
        'expected_next_phase_confidence': _clamp01(raw.get('expected_next_phase_confidence', transition_memory.get('expected_next_phase_confidence', 0.0))),
        'transition_entropy': _clamp01(raw.get('transition_entropy', transition_memory.get('phase_transition_entropy', 1.0))),
        'rollout_uncertainty': _clamp01(raw.get('rollout_uncertainty', transition_memory.get('rollout_uncertainty', uncertainty_score))),
        'transition_prior_signature': str(raw.get('transition_prior_signature', transition_memory.get('transition_prior_signature', '')) or ''),
        'transition_memory': {
            'current_phase': _normalize_phase_label(transition_memory.get('current_phase', raw.get('phase', ''))) or 'exploring',
            'expected_next_phase': _normalize_phase_label(transition_memory.get('expected_next_phase', raw.get('expected_next_phase', ''))),
            'expected_next_phase_confidence': _clamp01(transition_memory.get('expected_next_phase_confidence', raw.get('expected_next_phase_confidence', 0.0))),
            'phase_transition_entropy': _clamp01(transition_memory.get('phase_transition_entropy', raw.get('transition_entropy', 1.0))),
            'dominant_transitions': [
                dict(item) for item in list(transition_memory.get('dominant_transitions', []) or [])[:4]
                if isinstance(item, dict)
            ],
            'stabilizing_functions': [
                dict(item) for item in list(transition_memory.get('stabilizing_functions', []) or [])[:4]
                if isinstance(item, dict)
            ],
            'risky_functions': [
                dict(item) for item in list(transition_memory.get('risky_functions', []) or [])[:4]
                if isinstance(item, dict)
            ],
            'phase_function_scores': {
                str(name): dict(payload)
                for name, payload in (transition_memory.get('phase_function_scores', {}) or {}).items()
                if isinstance(payload, dict)
            },
            'dominant_branch_id': dominant_branch_id,
            'latent_branches': latent_branches,
            'rollout_uncertainty': _clamp01(transition_memory.get('rollout_uncertainty', raw.get('rollout_uncertainty', uncertainty_score))),
            'transition_prior_signature': str(transition_memory.get('transition_prior_signature', raw.get('transition_prior_signature', '')) or ''),
        },
        'update_count': max(0, int(raw.get('update_count', 0) or 0)),
    }


def _transition_memory_function_names(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    names: List[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get('function_name', '') or '')
        else:
            name = str(item or '')
        if name and name not in names:
            names.append(name)
    return names


def _parse_learning_context_key(raw_key: Any) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    text = str(raw_key or '').strip()
    if not text:
        return parsed
    for chunk in text.split('|'):
        if '=' not in chunk:
            continue
        key, value = chunk.split('=', 1)
        key = str(key or '').strip()
        value = str(value or '').strip()
        if key:
            parsed[key] = value
    return parsed


def _provider_learning_policy_snapshot(provider: ContextProvider) -> Dict[str, Any]:
    learning_policy_snapshot = _safe_optional_callable(provider, 'learning_policy_snapshot')
    if learning_policy_snapshot is not None:
        raw = learning_policy_snapshot()
        if isinstance(raw, dict):
            return dict(raw)
    loop = _safe_optional_attr(provider, 'loop')
    raw = getattr(loop, '_learning_policy_snapshot', {}) if loop is not None else {}
    return dict(raw) if isinstance(raw, dict) else {}


def _provider_workspace_state(provider: ContextProvider) -> Dict[str, Any]:
    workspace_state = _safe_optional_callable(provider, 'workspace_state')
    if workspace_state is not None:
        raw = workspace_state()
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _provider_cognitive_object_records(provider: ContextProvider, *, limit: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    cognitive_object_records = _safe_optional_callable(provider, 'cognitive_object_records')
    if cognitive_object_records is None:
        return {}
    records: Dict[str, List[Dict[str, Any]]] = {}
    for object_type in ALL_COGNITIVE_OBJECT_TYPES:
        rows = cognitive_object_records(object_type=object_type, limit=limit)
        if isinstance(rows, list):
            records[object_type] = [dict(item) for item in rows if isinstance(item, dict)]
    return records


def _retention_learning_adjustments(
    provider: ContextProvider,
    *,
    task_family: str,
    default_phase: str,
    hidden_phase: str,
    expected_next_phase: str,
    latent_branches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    learning_snapshot = _provider_learning_policy_snapshot(provider)
    retention_policy = learning_snapshot.get('retention_failure_policy', {}) if isinstance(learning_snapshot.get('retention_failure_policy', {}), dict) else {}
    phase_candidates = {
        _normalize_phase_label(default_phase),
        _normalize_phase_label(hidden_phase),
        _normalize_phase_label(expected_next_phase),
    }
    phase_candidates.discard('')
    branch_index = {
        str(branch.get('branch_id', '') or ''): dict(branch)
        for branch in latent_branches
        if isinstance(branch, dict) and str(branch.get('branch_id', '') or '')
    }
    adjustments: Dict[str, Any] = {
        'function': {},
        'phase': {},
        'world_shift_risk_delta': 0.0,
        'transition_confidence_delta': 0.0,
        'required_probes': [],
        'blocked_functions': [],
        'matched_contexts': [],
    }

    def _ensure_metric_bucket(pool: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
        return pool.setdefault(
            str(key),
            {
                'reward_delta': 0.0,
                'risk_delta': 0.0,
                'info_gain_delta': 0.0,
                'constraint_delta': 0.0,
                'confidence_delta': 0.0,
            },
        )

    def _apply_delta(pool: Dict[str, Dict[str, float]], key: str, *, reward: float = 0.0, risk: float = 0.0, info_gain: float = 0.0, constraint: float = 0.0, confidence: float = 0.0) -> None:
        if not key:
            return
        bucket = _ensure_metric_bucket(pool, key)
        bucket['reward_delta'] += float(reward)
        bucket['risk_delta'] += float(risk)
        bucket['info_gain_delta'] += float(info_gain)
        bucket['constraint_delta'] += float(constraint)
        bucket['confidence_delta'] += float(confidence)

    for context_key, raw_bucket in retention_policy.items():
        bucket = raw_bucket if isinstance(raw_bucket, dict) else {}
        parsed_key = _parse_learning_context_key(bucket.get('base_context_key', context_key))
        bucket_task_family = str(parsed_key.get('task_family', '') or '').strip().lower()
        current_task_family = str(task_family or '').strip().lower()
        if current_task_family and bucket_task_family and bucket_task_family not in {current_task_family, 'unknown'}:
            continue
        bucket_phase = _normalize_phase_label(parsed_key.get('phase', ''))
        branch_target_phase = _normalize_phase_label(bucket.get('rollout_branch_target_phase', ''))
        if phase_candidates and branch_target_phase and branch_target_phase not in phase_candidates:
            continue

        failure_type = str(bucket.get('dominant_failure_type', parsed_key.get('failure', '')) or '').strip().lower()
        if not failure_type:
            continue
        severity = _clamp01(max(float(bucket.get('severity', 0.0) or 0.0), abs(float(bucket.get('delta', 0.0) or 0.0))))
        confidence = _clamp01(float(bucket.get('confidence', 0.0) or 0.0))
        if severity <= 0.0 or confidence <= 0.0:
            continue
        weight = max(0.08, severity * max(0.25, confidence))
        selected_name = str(bucket.get('selected_name', '') or '').strip()
        relevant_phase = branch_target_phase or _normalize_phase_label(expected_next_phase) or _normalize_phase_label(hidden_phase) or _normalize_phase_label(default_phase) or 'exploring'
        branch_id = str(bucket.get('rollout_branch_id', '') or '').strip()
        branch = branch_index.get(branch_id, {})
        anchor_functions = [str(name) for name in list(branch.get('anchor_functions', []) or []) if str(name or '')]
        risky_functions = [str(name) for name in list(branch.get('risky_functions', []) or []) if str(name or '')]

        adjustments['matched_contexts'].append(
            {
                'context_key': str(context_key),
                'failure_type': failure_type,
                'phase': relevant_phase,
                'selected_name': selected_name,
                'severity': round(severity, 4),
                'confidence': round(confidence, 4),
            }
        )

        if failure_type == 'prediction_drift':
            _apply_delta(adjustments['phase'], relevant_phase, risk=0.10 * weight, info_gain=0.10 * weight, constraint=0.03 * weight, confidence=-0.08 * weight)
            if selected_name:
                _apply_delta(adjustments['function'], selected_name, risk=0.14 * weight, info_gain=0.12 * weight, constraint=0.04 * weight, confidence=-0.10 * weight)
            adjustments['world_shift_risk_delta'] += 0.06 * weight
            adjustments['transition_confidence_delta'] -= 0.08 * weight
            adjustments['required_probes'].extend(['probe_hidden_state_transition', 'probe_phase_alignment'])
        elif failure_type == 'branch_persistence_collapse':
            _apply_delta(adjustments['phase'], relevant_phase, reward=-(0.08 * weight), risk=0.12 * weight, info_gain=0.08 * weight, constraint=0.06 * weight, confidence=-0.10 * weight)
            if selected_name:
                _apply_delta(adjustments['function'], selected_name, reward=-(0.12 * weight), risk=0.16 * weight, info_gain=0.06 * weight, constraint=0.09 * weight, confidence=-0.14 * weight)
            for fn_name in anchor_functions:
                _apply_delta(adjustments['function'], fn_name, reward=-(0.05 * weight), risk=0.08 * weight, info_gain=0.05 * weight, confidence=-0.08 * weight)
            for fn_name in risky_functions:
                _apply_delta(adjustments['function'], fn_name, risk=0.14 * weight, constraint=0.08 * weight, confidence=-0.06 * weight)
                adjustments['blocked_functions'].append(fn_name)
            adjustments['world_shift_risk_delta'] += 0.08 * weight
            adjustments['transition_confidence_delta'] -= 0.10 * weight
            adjustments['required_probes'].append('probe_latent_branch')
        elif failure_type == 'planner_target_switch':
            _apply_delta(adjustments['phase'], relevant_phase, reward=-(0.06 * weight), risk=0.08 * weight, info_gain=0.10 * weight, constraint=0.05 * weight, confidence=-0.06 * weight)
            if selected_name:
                _apply_delta(adjustments['function'], selected_name, reward=-(0.10 * weight), risk=0.10 * weight, info_gain=0.08 * weight, constraint=0.06 * weight, confidence=-0.08 * weight)
            adjustments['world_shift_risk_delta'] += 0.05 * weight
            adjustments['transition_confidence_delta'] -= 0.06 * weight
            adjustments['required_probes'].append('probe_transition_frontier')
        elif failure_type == 'governance_overrule_misfire':
            _apply_delta(adjustments['phase'], relevant_phase, risk=0.10 * weight, info_gain=0.09 * weight, constraint=0.07 * weight, confidence=-0.07 * weight)
            if selected_name:
                _apply_delta(adjustments['function'], selected_name, reward=-(0.07 * weight), risk=0.14 * weight, info_gain=0.09 * weight, constraint=0.10 * weight, confidence=-0.08 * weight)
                if severity >= 0.72:
                    adjustments['blocked_functions'].append(selected_name)
            adjustments['world_shift_risk_delta'] += 0.04 * weight
            adjustments['transition_confidence_delta'] -= 0.07 * weight
            adjustments['required_probes'].append('probe_governance_conflict')

    adjustments['required_probes'] = list(dict.fromkeys(str(item) for item in adjustments['required_probes'] if str(item or '')))
    adjustments['blocked_functions'] = list(dict.fromkeys(str(item) for item in adjustments['blocked_functions'] if str(item or '')))
    adjustments['world_shift_risk_delta'] = max(-0.25, min(0.25, float(adjustments['world_shift_risk_delta'] or 0.0)))
    adjustments['transition_confidence_delta'] = max(-0.25, min(0.25, float(adjustments['transition_confidence_delta'] or 0.0)))
    adjustments['matched_contexts'] = adjustments['matched_contexts'][:6]
    return adjustments


def _normalize_latent_branches(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    branches: List[Dict[str, Any]] = []
    for item in items[:4]:
        if not isinstance(item, dict):
            continue
        branches.append({
            'branch_id': str(item.get('branch_id', '') or ''),
            'current_phase': _normalize_phase_label(item.get('current_phase', '')) or 'exploring',
            'target_phase': _normalize_phase_label(item.get('target_phase', '')) or 'exploring',
            'confidence': _clamp01(item.get('confidence', 0.0)),
            'support': _clamp01(item.get('support', 0.0)),
            'transition_score': _clamp01(item.get('transition_score', 0.0)),
            'success_rate': _clamp01(item.get('success_rate', 0.0)),
            'avg_reward': float(item.get('avg_reward', 0.0) or 0.0),
            'avg_depth_gain': _clamp01(item.get('avg_depth_gain', 0.0)),
            'uncertainty_pressure': _clamp01(item.get('uncertainty_pressure', 0.0)),
            'anchor_functions': [
                str(value) for value in list(item.get('anchor_functions', []) or [])[:4]
                if str(value or '')
            ],
            'risky_functions': [
                str(value) for value in list(item.get('risky_functions', []) or [])[:4]
                if str(value or '')
            ],
            'latent_signature': str(item.get('latent_signature', '') or ''),
        })
    return branches


def _rank_counter(counter: Counter[str], *, limit: int = 3) -> List[Dict[str, Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [
        {'name': name, 'count': int(count)}
        for name, count in ranked[:limit]
    ]


def _summary_has_grid_signal(summary: Dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    summary_type = str(summary.get('type', '') or '')
    if summary_type == 'grid':
        return True
    if summary_type == 'list':
        return int(summary.get('grid_like_items', 0) or 0) > 0
    if summary_type == 'dict':
        child_types = summary.get('child_types', {})
        if isinstance(child_types, dict):
            return any(str(value or '') == 'grid' for value in child_types.values())
    return False


def _structure_transition_delta(before_summary: Dict[str, Any], after_summary: Dict[str, Any]) -> float:
    if not before_summary or not after_summary:
        return 0.0
    before_depth = float(before_summary.get('depth', 0.0) or 0.0)
    after_depth = float(after_summary.get('depth', 0.0) or 0.0)
    depth_delta = abs(after_depth - before_depth) / max(before_depth, after_depth, 1.0)
    type_delta = 1.0 if before_summary.get('type') != after_summary.get('type') else 0.0
    key_delta = 0.0
    if str(before_summary.get('type', '') or '') == 'dict' and str(after_summary.get('type', '') or '') == 'dict':
        before_keys = set(before_summary.get('keys', []) or [])
        after_keys = set(after_summary.get('keys', []) or [])
        union = len(before_keys | after_keys)
        if union:
            key_delta = len(before_keys ^ after_keys) / union
    return _clamp01(depth_delta * 0.45 + type_delta * 0.35 + key_delta * 0.2)


def _estimate_world_model_dynamics(
    provider: ContextProvider,
    perception_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trace = provider.episode_trace()[-20:]
    beliefs = _safe_optional_callable(provider, 'beliefs')
    belief_snapshot = beliefs() if beliefs is not None else {}
    active_beliefs = list(belief_snapshot.get('active', [])) if isinstance(belief_snapshot, dict) else []
    avg_confidence = _mean([
        float(getattr(belief, 'confidence', 0.0) or 0.0)
        for belief in active_beliefs
    ])
    avg_uncertainty = _mean([
        float(getattr(belief, 'uncertainty', max(0.0, 1.0 - float(getattr(belief, 'confidence', 0.0) or 0.0))) or max(0.0, 1.0 - float(getattr(belief, 'confidence', 0.0) or 0.0)))
        for belief in active_beliefs
    ])
    if not active_beliefs:
        avg_uncertainty = 1.0

    reward_values: List[float] = []
    info_gain_values: List[float] = []
    reward_sign_switches = 0.0
    reward_sign_steps = 0.0
    previous_sign = 0
    action_depths: List[float] = []
    object_signals: List[float] = []
    observation_depths: List[float] = []
    outcome_depths: List[float] = []
    transition_deltas: List[float] = []
    fn_counts: Counter[str] = Counter()
    fn_failures: Counter[str] = Counter()
    fn_successes: Counter[str] = Counter()
    fn_reward_sums: Dict[str, float] = {}
    fn_step_counts: Dict[str, float] = {}
    observation_types: Counter[str] = Counter()
    outcome_types: Counter[str] = Counter()
    grid_payload_total = 0.0
    observation_grid_count = 0.0
    outcome_grid_count = 0.0
    failure_total = 0.0
    success_total = 0.0

    for row in trace:
        if not isinstance(row, dict):
            continue
        reward = float(row.get('reward', 0.0) or 0.0)
        reward_values.append(reward)
        sign = 1 if reward > 0.05 else (-1 if reward < -0.05 else 0)
        if sign != 0:
            if previous_sign != 0:
                reward_sign_steps += 1.0
                if sign != previous_sign:
                    reward_sign_switches += 1.0
            previous_sign = sign

        info_gain_raw = row.get('information_gain', 0.0)
        if isinstance(info_gain_raw, dict):
            info_gain_raw = info_gain_raw.get('value', 0.0)
        info_gain_values.append(_clamp01(info_gain_raw))

        action = row.get('action', {}) if isinstance(row.get('action', {}), dict) else {}
        fn = provider.extraction_function_name(action, default='')
        if fn and fn != 'wait':
            fn_counts[fn] += 1
            fn_reward_sums[fn] = float(fn_reward_sums.get(fn, 0.0) or 0.0) + reward
            fn_step_counts[fn] = float(fn_step_counts.get(fn, 0.0) or 0.0) + 1.0

        outcome = row.get('outcome', {}) if isinstance(row.get('outcome', {}), dict) else {}
        failed = bool(outcome.get('success') is False or reward < 0.0 or outcome.get('error'))
        if fn and fn != 'wait':
            if failed:
                fn_failures[fn] += 1
            else:
                fn_successes[fn] += 1
        if failed:
            failure_total += 1.0
        else:
            success_total += 1.0

        action_summary = summarize_action_state(action)
        if action_summary:
            action_depths.append(float(action_summary.get('max_value_depth', 0.0) or 0.0))
            object_signals.append(float(action_summary.get('object_signal', 0.0) or 0.0))
            grid_payload_total += float(action_summary.get('grid_like_payloads', 0.0) or 0.0)

        observation_summary = summarize_value_structure(row.get('observation', {}))
        outcome_summary = summarize_value_structure(outcome)
        if observation_summary:
            observation_depths.append(float(observation_summary.get('depth', 0.0) or 0.0))
            observation_types[str(observation_summary.get('type', 'unknown') or 'unknown')] += 1
            if _summary_has_grid_signal(observation_summary):
                observation_grid_count += 1.0
        if outcome_summary:
            outcome_depths.append(float(outcome_summary.get('depth', 0.0) or 0.0))
            outcome_types[str(outcome_summary.get('type', 'unknown') or 'unknown')] += 1
            if _summary_has_grid_signal(outcome_summary):
                outcome_grid_count += 1.0
        transition_deltas.append(_structure_transition_delta(observation_summary, outcome_summary))

    transition_count = len(trace)
    sample_factor = _clamp01(transition_count / 6.0)
    failure_rate = failure_total / max(1.0, success_total + failure_total)
    success_rate = success_total / max(1.0, success_total + failure_total)
    avg_reward = _mean(reward_values)
    reward_diffs = [
        abs(reward_values[idx] - reward_values[idx - 1])
        for idx in range(1, len(reward_values))
    ]
    reward_volatility = _clamp01(_mean(reward_diffs))
    reward_switch_rate = reward_sign_switches / max(1.0, reward_sign_steps)
    avg_info_gain = _mean(info_gain_values)
    structure_instability = _mean(transition_deltas)
    avg_action_depth = _mean(action_depths)
    avg_object_signal = _mean(object_signals)
    grid_payload_rate = _clamp01(grid_payload_total / max(1.0, transition_count))
    observation_complexity = _mean(observation_depths)
    outcome_complexity = _mean(outcome_depths)

    transition_confidence = _clamp01(
        avg_confidence * 0.35
        + success_rate * 0.2
        + (1.0 - reward_volatility) * 0.15
        + (1.0 - structure_instability) * 0.15
        + avg_info_gain * 0.05
        + sample_factor * 0.1
    )
    shift_risk = _clamp01(
        failure_rate * 0.3
        + reward_volatility * 0.2
        + reward_switch_rate * 0.15
        + structure_instability * 0.2
        + (1.0 - avg_confidence) * 0.15
    )

    predicted_phase = 'exploring'
    if transition_confidence >= 0.72 and shift_risk <= 0.35:
        predicted_phase = 'committed'
    elif transition_confidence >= 0.45 and shift_risk <= 0.65:
        predicted_phase = 'stabilizing'

    blocked_functions: List[str] = []
    risky_functions: List[Dict[str, Any]] = []
    stabilizing_functions: List[Dict[str, Any]] = []
    for fn_name, count in fn_counts.items():
        failures = int(fn_failures.get(fn_name, 0) or 0)
        successes = int(fn_successes.get(fn_name, 0) or 0)
        avg_fn_reward = float(fn_reward_sums.get(fn_name, 0.0) or 0.0) / max(float(fn_step_counts.get(fn_name, 1.0) or 1.0), 1.0)
        failure_ratio = failures / max(count, 1)
        entry = {
            'function_name': fn_name,
            'count': int(count),
            'failure_ratio': round(failure_ratio, 4),
            'avg_reward': round(avg_fn_reward, 4),
        }
        if failures >= 1 and (failure_ratio >= 0.5 or avg_fn_reward < 0.0):
            risky_functions.append(entry)
        if successes >= 1 and avg_fn_reward >= 0.0:
            stabilizing_functions.append(entry)
        if count >= 2 and failures >= 2 and failure_ratio >= 0.67 and avg_fn_reward < 0.0:
            blocked_functions.append(fn_name)

    preferred_action_classes = ['probe', 'inspect', 'reversible']
    if predicted_phase == 'committed' and shift_risk <= 0.35:
        preferred_action_classes = ['execute', 'reversible', 'transform']
    elif predicted_phase == 'stabilizing' and shift_risk <= 0.55:
        preferred_action_classes = ['reversible', 'probe', 'transform']

    required_probes: List[str] = []
    if transition_confidence < 0.45 or failure_rate >= 0.5:
        required_probes.append('probe_state_transition')
    if shift_risk >= 0.6:
        required_probes.append('probe_before_commit')
    if avg_confidence < 0.55:
        required_probes.append('probe_high_impact_belief')

    hard_constraints: List[str] = []
    if shift_risk >= 0.68 or predicted_phase == 'exploring':
        hard_constraints.append('no_shutdown')

    perception_structure = summarize_value_structure(perception_summary or {})
    return {
        'trace_count': int(transition_count),
        'avg_confidence': round(avg_confidence, 4),
        'avg_uncertainty': round(avg_uncertainty, 4),
        'avg_reward': round(avg_reward, 4),
        'avg_info_gain': round(avg_info_gain, 4),
        'reward_volatility': round(reward_volatility, 4),
        'reward_switch_rate': round(reward_switch_rate, 4),
        'failure_rate': round(failure_rate, 4),
        'success_rate': round(success_rate, 4),
        'predicted_phase': predicted_phase,
        'transition_confidence': round(transition_confidence, 4),
        'shift_risk': round(shift_risk, 4),
        'blocked_functions': sorted(set(blocked_functions)),
        'preferred_action_classes': preferred_action_classes,
        'hard_constraints': hard_constraints,
        'required_probes': required_probes,
        'state_abstraction': {
            'perception_structure': perception_structure,
            'avg_action_depth': round(avg_action_depth, 4),
            'avg_object_signal': round(avg_object_signal, 4),
            'grid_payload_rate': round(grid_payload_rate, 4),
            'observation_complexity': round(observation_complexity, 4),
            'outcome_complexity': round(outcome_complexity, 4),
            'observation_grid_rate': round(observation_grid_count / max(1.0, transition_count), 4),
            'outcome_grid_rate': round(outcome_grid_count / max(1.0, transition_count), 4),
            'dominant_observation_types': _rank_counter(observation_types, limit=3),
            'dominant_outcome_types': _rank_counter(outcome_types, limit=3),
        },
        'state_dynamics': {
            'trace_count': int(transition_count),
            'reward_volatility': round(reward_volatility, 4),
            'reward_switch_rate': round(reward_switch_rate, 4),
            'failure_rate': round(failure_rate, 4),
            'success_rate': round(success_rate, 4),
            'structure_instability': round(structure_instability, 4),
            'dominant_functions': _rank_counter(fn_counts, limit=4),
            'risky_functions': sorted(risky_functions, key=lambda item: (-float(item['failure_ratio']), float(item['avg_reward']), item['function_name']))[:4],
            'stabilizing_functions': sorted(stabilizing_functions, key=lambda item: (-float(item['avg_reward']), -int(item['count']), item['function_name']))[:4],
        },
    }


def build_world_model_context(provider: ContextProvider, perception_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a JSON-safe advisory summary for decision-making."""
    belief_snapshot = provider.beliefs()
    active = list(belief_snapshot.get('active', [])) if isinstance(belief_snapshot, dict) else []
    established_count = int(belief_snapshot.get('established_count', 0) or 0) if isinstance(belief_snapshot, dict) else 0
    beliefs = {
        belief.variable_name: {
            'posterior': belief.posterior,
            'confidence': belief.confidence,
            'status': belief.status.value,
            'source_hypothesis_id': belief.source_hypothesis_id,
            'evidence_chain': list(belief.evidence_ids[-5:]),
            'counter_evidence_chain': [
                value.value for value in belief.hypothesized_values
                if value.value != belief.posterior and value.evidence_ids
            ][:3],
            'invalidation_conditions': ['new_conflicting_evidence', 'sustained_negative_reward'],
        }
        for belief in active
    }
    high_value_beliefs = [
        {
            'variable': belief.variable_name,
            'posterior': belief.posterior,
            'confidence': belief.confidence,
            'impact_scope': 'planner+decision' if belief.confidence >= 0.6 else 'decision_only',
        }
        for belief in active
        if belief.confidence >= 0.55
    ]
    dynamics = _estimate_world_model_dynamics(provider, perception_summary)
    hidden_state = _provider_hidden_state_summary(provider)
    hidden_updates = int(hidden_state.get('update_count', 0) or 0)
    hidden_phase = str(hidden_state.get('phase', '') or '')
    hidden_phase_confidence = float(hidden_state.get('phase_confidence', 0.0) or 0.0)
    hidden_state_depth = int(hidden_state.get('hidden_state_depth', 0) or 0)
    hidden_drift_score = float(hidden_state.get('drift_score', 0.0) or 0.0)
    hidden_uncertainty_score = float(hidden_state.get('uncertainty_score', 0.0) or 0.0)
    transition_memory = hidden_state.get('transition_memory', {}) if isinstance(hidden_state.get('transition_memory', {}), dict) else {}
    latent_branches = _normalize_latent_branches(hidden_state.get('latent_branches', transition_memory.get('latent_branches', [])))
    dominant_branch_id = str(
        hidden_state.get(
            'dominant_branch_id',
            transition_memory.get('dominant_branch_id', latent_branches[0].get('branch_id', '') if latent_branches else ''),
        ) or ''
    )
    dominant_branch = {}
    for branch in latent_branches:
        if str(branch.get('branch_id', '') or '') == dominant_branch_id:
            dominant_branch = dict(branch)
            break
    if not dominant_branch and latent_branches:
        dominant_branch = dict(latent_branches[0])
    dominant_branch_target_phase = str(dominant_branch.get('target_phase', '') or '')
    dominant_branch_confidence = float(dominant_branch.get('confidence', 0.0) or 0.0)
    dominant_branch_risky_functions = [
        str(value) for value in list(dominant_branch.get('risky_functions', []) or [])[:4]
        if str(value or '')
    ]
    expected_next_phase = str(hidden_state.get('expected_next_phase', transition_memory.get('expected_next_phase', '')) or '')
    expected_next_phase_confidence = float(hidden_state.get('expected_next_phase_confidence', transition_memory.get('expected_next_phase_confidence', 0.0)) or 0.0)
    transition_entropy = float(hidden_state.get('transition_entropy', transition_memory.get('phase_transition_entropy', 1.0)) or 1.0)
    stabilizing_transition_functions = _transition_memory_function_names(transition_memory.get('stabilizing_functions', []))
    risky_transition_functions = _transition_memory_function_names(transition_memory.get('risky_functions', []))
    predicted_phase = str(dynamics.get('predicted_phase', '') or '')
    transition_confidence = float(dynamics.get('transition_confidence', 0.0) or 0.0)
    shift_risk = float(dynamics.get('shift_risk', 0.0) or 0.0)
    if hidden_phase and hidden_updates > 0 and (
        hidden_phase_confidence >= max(0.25, transition_confidence - 0.05)
        or hidden_state_depth > 0
    ):
        predicted_phase = hidden_phase
    if expected_next_phase and hidden_updates > 0 and expected_next_phase_confidence >= max(0.42, transition_confidence - 0.04):
        predicted_phase = expected_next_phase
    if not predicted_phase:
        predicted_phase = hidden_phase or 'exploring'
    if hidden_updates > 0:
        transition_confidence = _clamp01(
            (transition_confidence * 0.56)
            + (hidden_phase_confidence * 0.24)
            + (expected_next_phase_confidence * 0.20)
        )
        shift_risk = _clamp01(max(shift_risk, hidden_drift_score * 0.82))
    if hidden_updates > 0 and expected_next_phase_confidence < 0.45:
        shift_risk = _clamp01(max(shift_risk, 0.18 + transition_entropy * 0.34))
    uncertain_high_impact_beliefs = [
        entry['variable']
        for entry in high_value_beliefs
        if float(entry.get('confidence', 0.0) or 0.0) < 0.7
    ]
    required_probes = list(dynamics.get('required_probes', []) or [])
    if uncertain_high_impact_beliefs:
        required_probes.append('probe_high_impact_belief')
    if hidden_updates > 0 and (hidden_uncertainty_score >= 0.72 or hidden_drift_score >= 0.6):
        required_probes.append('probe_hidden_state_transition')
    if hidden_updates > 0 and hidden_phase in {'exploring', 'disrupted'} and hidden_state_depth >= 2:
        required_probes.append('probe_phase_alignment')
    if hidden_updates > 1 and transition_entropy >= 0.62:
        required_probes.append('probe_transition_frontier')
    if hidden_updates > 1 and dominant_branch_confidence >= 0.45 and dominant_branch_target_phase in {'exploring', 'disrupted'}:
        required_probes.append('probe_latent_branch')
    blocked_functions = list(dynamics.get('blocked_functions', []) or [])
    for fn_name in risky_transition_functions:
        if fn_name not in blocked_functions and hidden_updates > 1 and expected_next_phase_confidence >= 0.35:
            blocked_functions.append(fn_name)
    for fn_name in dominant_branch_risky_functions:
        if fn_name not in blocked_functions and hidden_updates > 1 and dominant_branch_confidence >= 0.52:
            blocked_functions.append(fn_name)
    required_probes = list(dict.fromkeys(required_probes))
    trace_tail = provider.episode_trace()[-8:]
    base_summary = {
        'beliefs': beliefs,
        'active_count': len(active),
        'established_count': established_count,
        'high_value_beliefs': high_value_beliefs[:6],
        'perception': dict(perception_summary or {}),
        'uncertain_high_impact_beliefs': uncertain_high_impact_beliefs[:4],
        'hidden_state': hidden_state,
        'dominant_branch_id': dominant_branch_id,
        'latent_branches': latent_branches,
        'state_abstraction': dict(dynamics.get('state_abstraction', {})),
        'state_dynamics': {
            **dict(dynamics.get('state_dynamics', {})),
            'expected_next_phase': expected_next_phase,
            'expected_next_phase_confidence': expected_next_phase_confidence,
            'phase_transition_entropy': transition_entropy,
            'stabilizing_functions': stabilizing_transition_functions,
            'risky_functions': risky_transition_functions,
            'dominant_branch_id': dominant_branch_id,
            'latent_branches': latent_branches,
            'transition_memory': transition_memory,
        },
        'predicted_phase': predicted_phase,
        'transition_confidence': transition_confidence,
        'shift_risk': shift_risk,
        'expected_next_phase': expected_next_phase,
        'expected_next_phase_confidence': expected_next_phase_confidence,
        'phase_transition_entropy': transition_entropy,
        'blocked_functions': blocked_functions,
        'preferred_action_classes': list(dynamics.get('preferred_action_classes', []) or []),
        'hard_constraints': list(dynamics.get('hard_constraints', []) or []),
        'required_probes': required_probes,
        'control_hints': {
            'predicted_phase': predicted_phase,
            'transition_confidence': transition_confidence,
            'state_shift_risk': shift_risk,
            'expected_next_phase': expected_next_phase,
            'expected_next_phase_confidence': expected_next_phase_confidence,
            'phase_transition_entropy': transition_entropy,
            'dominant_branch_id': dominant_branch_id,
            'latent_branches': latent_branches,
            'blocked_functions': blocked_functions,
            'preferred_action_classes': list(dynamics.get('preferred_action_classes', []) or []),
            'hard_constraints': list(dynamics.get('hard_constraints', []) or []),
            'required_probes': required_probes,
            'hidden_state': hidden_state,
            'transition_memory': transition_memory,
            'stabilizing_functions': stabilizing_transition_functions,
            'risky_functions': risky_transition_functions,
        },
    }
    object_graph = build_object_graph(
        perception_summary or {},
        world_model_summary=base_summary,
    )
    mechanism_graph = build_mechanism_graph(
        base_summary,
        object_graph=object_graph,
        recent_trace=trace_tail,
        limit=4,
    )
    mechanism_control_summary = {
        'dominant_mechanism_family': mechanism_graph.get('dominant_mechanism_family', ''),
        'dominant_mechanism_confidence': mechanism_graph.get('dominant_mechanism_confidence', 0.0),
        'preferred_action_families': mechanism_graph.get('preferred_action_families', []),
        'discriminating_actions': mechanism_graph.get('discriminating_actions', []),
    }
    affordance_graph = build_affordance_graph(
        {
            **base_summary,
            'world_scene_summary': dict(object_graph.get('scene_summary', {})),
            'mechanism_hypotheses': mechanism_graph.get('mechanism_hypotheses', []),
            'mechanism_control_summary': mechanism_control_summary,
        },
        recent_interactions=trace_tail,
        mechanism_hypotheses_summary=mechanism_graph.get('mechanism_hypotheses', []),
        mechanism_control_summary=mechanism_control_summary,
    )
    relation_summary = {
        str(row.get('relation_type', '') or ''): sum(
            1
            for item in object_graph.get('relations', [])
            if isinstance(item, dict) and str(item.get('relation_type', '') or '') == str(row.get('relation_type', '') or '')
        )
        for row in object_graph.get('relations', [])
        if isinstance(row, dict) and str(row.get('relation_type', '') or '')
    }
    enriched_summary = {
        **base_summary,
        'world_scene_summary': dict(object_graph.get('scene_summary', {})),
        'world_state_signature': str(object_graph.get('world_state_signature', base_summary.get('world_state_signature', '')) or ''),
        'world_entities': [dict(row) for row in object_graph.get('objects', []) if isinstance(row, dict)],
        'world_relations': [dict(row) for row in object_graph.get('relations', []) if isinstance(row, dict)],
        'world_relation_summary': relation_summary,
        'object_graph': object_graph,
        'affordance_graph': [dict(row) for row in affordance_graph.get('affordances', []) if isinstance(row, dict)],
        'candidate_intervention_targets': [dict(row) for row in affordance_graph.get('candidate_intervention_targets', []) if isinstance(row, dict)],
        'mechanism_hypotheses': [dict(row) for row in mechanism_graph.get('mechanism_hypotheses', []) if isinstance(row, dict)],
        'mechanism_families': list(mechanism_graph.get('mechanism_families', []) or []),
        'mechanism_control_summary': mechanism_control_summary,
    }
    discriminating_tests = propose_discriminating_tests(
        {
            **enriched_summary,
            'required_probes': required_probes,
        },
        available_functions=[*stabilizing_transition_functions, *risky_transition_functions, *blocked_functions],
        limit=4,
    )
    expected_information_gain = max(
        float(affordance_graph.get('expected_information_gain', 0.0) or 0.0),
        max(
            (float(row.get('info_gain_estimate', 0.0) or 0.0) for row in discriminating_tests if isinstance(row, dict)),
            default=0.0,
        ),
    )
    enriched_summary['discriminating_tests'] = discriminating_tests
    enriched_summary['expected_information_gain'] = round(expected_information_gain, 4)
    enriched_summary['rollout_uncertainty'] = round(
        _clamp01(hidden_uncertainty_score * 0.58 + transition_entropy * 0.24 + shift_risk * 0.18),
        4,
    )
    enriched_summary['program_search_constraints'] = _program_search_constraints(
        object_graph=object_graph,
        mechanism_graph=mechanism_graph,
        required_probes=required_probes,
        hidden_uncertainty_score=hidden_uncertainty_score,
    )
    control_hints = dict(enriched_summary.get('control_hints', {}) or {})
    control_hints['mechanism_hypotheses'] = [dict(row) for row in enriched_summary.get('mechanism_hypotheses', []) if isinstance(row, dict)]
    control_hints['candidate_intervention_targets'] = [dict(row) for row in enriched_summary.get('candidate_intervention_targets', []) if isinstance(row, dict)]
    control_hints['discriminating_tests'] = [dict(row) for row in discriminating_tests if isinstance(row, dict)]
    control_hints['expected_information_gain'] = float(enriched_summary.get('expected_information_gain', 0.0) or 0.0)
    control_hints['rollout_uncertainty'] = float(enriched_summary.get('rollout_uncertainty', 0.5) or 0.5)
    enriched_summary['control_hints'] = control_hints
    return enriched_summary


def build_world_model_transition_priors(
    provider: ContextProvider,
    perception_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build transition priors in canonical `wm_transition_priors_v2` schema.

    Canonical schema:
    - `__schema_version`: fixed string `wm_transition_priors_v2`.
    - `__key_fields`: `["function_name", "arg_bucket", "belief_phase"]`.
    - `__by_signature`: map of signature-id -> entry where:
      - `key.function_name` (required, str)
      - `key.arg_bucket` (optional, default: `"default"`)
      - `key.belief_phase` (optional, default inferred from beliefs; fallback `"exploring"`)
      - `metrics.<metric>.value|confidence|sample_count` for metric names:
        `long_horizon_reward`, `predicted_risk`, `reversibility`, `info_gain`
      - `metrics.constraint_violation` and `cold_start` boolean.
      - flattened legacy metric fields are preserved on the same entry.
    - `__cold_start_prior`: canonical fallback entry used when no signature evidence exists.
    - `__legacy_by_function`: function-keyed aggregate for older callers.
    - top-level function keys are also preserved for strict legacy consumers.
    """

    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def _normalize_phase(phase: Any) -> str:
        phase_str = str(phase or '').strip().lower()
        if not phase_str:
            return 'exploring'
        phase_alias = {
            'explore': 'exploring',
            'stabilize': 'stabilizing',
            'stable': 'stabilizing',
            'commit': 'committed',
            'committing': 'committed',
        }
        return phase_alias.get(phase_str, phase_str)

    def _infer_belief_phase(row: Dict[str, Any], fallback: str = 'exploring') -> str:
        phase_candidates = [
            row.get('belief_phase'),
            row.get('world_model_phase'),
            ((row.get('outcome', {}) if isinstance(row.get('outcome', {}), dict) else {}).get('belief_phase')),
            ((row.get('observation', {}) if isinstance(row.get('observation', {}), dict) else {}).get('belief_phase')),
        ]
        for candidate in phase_candidates:
            normalized = _normalize_phase(candidate)
            if normalized and normalized != 'exploring':
                return normalized
        return _normalize_phase(fallback)

    def _bucketize_args(kwargs: Dict[str, Any]) -> str:
        if not kwargs:
            return 'default'
        size = len(kwargs)
        numeric_count = sum(1 for value in kwargs.values() if isinstance(value, (int, float)) and not isinstance(value, bool))
        collection_count = sum(1 for value in kwargs.values() if isinstance(value, (list, tuple, set, dict)))
        if size >= 4 or collection_count >= 2:
            return 'complex'
        if numeric_count >= 2:
            return 'numeric_heavy'
        if size == 1:
            return 'single_arg'
        return 'structured'

    def _metric_template(value: float, confidence: float, sample_count: float) -> Dict[str, float]:
        return {
            'value': float(value),
            'confidence': _clamp(confidence, 0.0, 1.0),
            'sample_count': max(0.0, float(sample_count)),
        }

    def _flatten_metric_schema(metric_payload: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(base or {})
        for metric_name in ('long_horizon_reward', 'predicted_risk', 'reversibility', 'info_gain'):
            metric_obj = metric_payload.get(metric_name, {})
            metric_obj = metric_obj if isinstance(metric_obj, dict) else {}
            payload[metric_name] = float(metric_obj.get('value', 0.0) or 0.0)
            payload[f'{metric_name}_confidence'] = _clamp(float(metric_obj.get('confidence', 0.0) or 0.0), 0.0, 1.0)
            payload[f'{metric_name}_sample_count'] = max(0.0, float(metric_obj.get('sample_count', 0.0) or 0.0))
        payload['constraint_violation'] = _clamp(float(metric_payload.get('constraint_violation', 0.0) or 0.0), 0.0, 1.0)
        return payload

    def _domain_task_family_template(domain_tag: str, task_family: str) -> Dict[str, Any]:
        domain_key = str(domain_tag or 'generic').strip().lower()
        task_key = str(task_family or '').strip().lower()
        is_data_or_tool = any(token in f'{domain_key}/{task_key}' for token in ('table', 'sql', 'analytics', 'data', 'api'))
        base_reward = 0.08 if is_data_or_tool else 0.03
        base_risk = 0.32 if is_data_or_tool else 0.25
        base_rev = 0.56 if is_data_or_tool else 0.48
        base_info = 0.18 if is_data_or_tool else 0.12
        return {
            'long_horizon_reward': _metric_template(base_reward, confidence=0.28, sample_count=0.0),
            'predicted_risk': _metric_template(base_risk, confidence=0.32, sample_count=0.0),
            'reversibility': _metric_template(base_rev, confidence=0.35, sample_count=0.0),
            'info_gain': _metric_template(base_info, confidence=0.26, sample_count=0.0),
            'constraint_violation': 0.08 if is_data_or_tool else 0.05,
            'cold_start': True,
            'domain_template': {'domain': domain_key, 'task_family': task_key or 'general'},
        }

    def _canonical_transition_prior_key(raw_key: Any, *, default_phase: str) -> Dict[str, str]:
        normalized = canonicalize_transition_prior_key(raw_key, default_phase=default_phase)
        normalized['belief_phase'] = _normalize_phase(normalized.get('belief_phase'))
        return normalized

    trace = provider.episode_trace()[-20:]
    loop = _safe_optional_attr(provider, 'loop')
    plan_snapshot_fn = _safe_optional_callable(provider, 'plan_snapshot')
    plan_snapshot = plan_snapshot_fn() if plan_snapshot_fn is not None else {}
    task_family = str((plan_snapshot.get('plan_summary', {}) if isinstance(plan_snapshot.get('plan_summary', {}), dict) else {}).get('task_family', '') or '')
    domain_hint = ''
    if isinstance(perception_summary, dict):
        domain_hint = str(perception_summary.get('domain') or perception_summary.get('scene_type') or '')
    beliefs = _safe_optional_callable(provider, 'beliefs')
    belief_snapshot = beliefs() if beliefs is not None else {}
    active_beliefs = list(belief_snapshot.get('active', [])) if isinstance(belief_snapshot, dict) else []
    avg_confidence = 0.0
    avg_uncertainty = 1.0
    if active_beliefs:
        confs = [float(getattr(belief, 'confidence', 0.0) or 0.0) for belief in active_beliefs]
        uncs = [float(getattr(belief, 'uncertainty', max(0.0, 1.0 - conf)) or max(0.0, 1.0 - conf)) for belief, conf in zip(active_beliefs, confs)]
        avg_confidence = sum(confs) / max(1, len(confs))
        avg_uncertainty = sum(uncs) / max(1, len(uncs))
    world_dynamics = _estimate_world_model_dynamics(provider, perception_summary)
    hidden_state = _provider_hidden_state_summary(provider)
    hidden_phase = _normalize_phase_label(hidden_state.get('phase', ''))
    hidden_phase_confidence = _clamp(float(hidden_state.get('phase_confidence', 0.0) or 0.0), 0.0, 1.0)
    hidden_drift_score = _clamp(float(hidden_state.get('drift_score', 0.0) or 0.0), 0.0, 1.0)
    hidden_updates = int(hidden_state.get('update_count', 0) or 0)
    transition_memory = hidden_state.get('transition_memory', {}) if isinstance(hidden_state.get('transition_memory', {}), dict) else {}
    latent_branches = _normalize_latent_branches(hidden_state.get('latent_branches', transition_memory.get('latent_branches', [])))
    dominant_branch_id = str(
        hidden_state.get(
            'dominant_branch_id',
            transition_memory.get('dominant_branch_id', latent_branches[0].get('branch_id', '') if latent_branches else ''),
        ) or ''
    )
    expected_next_phase = _normalize_phase_label(hidden_state.get('expected_next_phase', transition_memory.get('expected_next_phase', '')))
    expected_next_phase_confidence = _clamp(float(hidden_state.get('expected_next_phase_confidence', transition_memory.get('expected_next_phase_confidence', 0.0)) or 0.0), 0.0, 1.0)
    phase_transition_entropy = _clamp(float(hidden_state.get('transition_entropy', transition_memory.get('phase_transition_entropy', 1.0)) or 1.0), 0.0, 1.0)
    phase_function_scores = transition_memory.get('phase_function_scores', {}) if isinstance(transition_memory.get('phase_function_scores', {}), dict) else {}
    stabilizing_transition_functions = _transition_memory_function_names(transition_memory.get('stabilizing_functions', []))
    risky_transition_functions = _transition_memory_function_names(transition_memory.get('risky_functions', []))
    default_phase = str(world_dynamics.get('predicted_phase', '') or ('committed' if avg_confidence >= 0.72 else ('stabilizing' if avg_confidence >= 0.45 else 'exploring')))
    if hidden_phase and hidden_updates > 0 and hidden_phase_confidence >= 0.45:
        default_phase = hidden_phase
    retention_learning = _retention_learning_adjustments(
        provider,
        task_family=task_family,
        default_phase=default_phase,
        hidden_phase=hidden_phase,
        expected_next_phase=expected_next_phase,
        latent_branches=latent_branches,
    )

    signature_stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in trace:
        if not isinstance(row, dict):
            continue
        action = row.get('action', {})
        fn = provider.extraction_function_name(action, default='')
        if not fn or fn == 'wait':
            continue
        action_snapshot = row.get('action_snapshot', {}) if isinstance(row.get('action_snapshot', {}), dict) else {}
        kwargs = action_snapshot.get('kwargs', {}) if isinstance(action_snapshot.get('kwargs', {}), dict) else {}
        arg_bucket = _bucketize_args(kwargs)
        belief_phase = _infer_belief_phase(row, fallback=default_phase)
        signature = (fn, arg_bucket, belief_phase)
        stat = signature_stats.setdefault(
            signature,
            {
                'count': 0.0,
                'reward_sum': 0.0,
                'failure_count': 0.0,
                'belief_conf_delta_sum': 0.0,
                'belief_unc_delta_sum': 0.0,
                'probe_total': 0.0,
                'probe_distinguish': 0.0,
                'recovery_triggered': 0.0,
                'recovery_success': 0.0,
                'info_gain_sum': 0.0,
                'action_depth_sum': 0.0,
                'object_signal_sum': 0.0,
                'grid_payload_total': 0.0,
                'observation_depth_sum': 0.0,
                'outcome_depth_sum': 0.0,
                'state_transition_delta_sum': 0.0,
                'state_change_count': 0.0,
            },
        )
        stat['count'] += 1.0
        reward = float(row.get('reward', 0.0) or 0.0)
        stat['reward_sum'] += reward
        outcome = row.get('outcome', {}) if isinstance(row.get('outcome', {}), dict) else {}
        failed = bool(outcome.get('success') is False or reward < 0.0 or outcome.get('error'))
        if failed:
            stat['failure_count'] += 1.0
        info_gain_raw = row.get('information_gain', 0.0)
        if isinstance(info_gain_raw, dict):
            info_gain_raw = info_gain_raw.get('value', 0.0)
        stat['info_gain_sum'] += float(info_gain_raw or 0.0)

        action_summary = summarize_action_state(action)
        if action_summary:
            stat['action_depth_sum'] += float(action_summary.get('max_value_depth', 0.0) or 0.0)
            stat['object_signal_sum'] += float(action_summary.get('object_signal', 0.0) or 0.0)
            stat['grid_payload_total'] += float(action_summary.get('grid_like_payloads', 0.0) or 0.0)
        observation_summary = summarize_value_structure(row.get('observation', {}))
        outcome_summary = summarize_value_structure(outcome)
        stat['observation_depth_sum'] += float(observation_summary.get('depth', 0.0) or 0.0) if observation_summary else 0.0
        stat['outcome_depth_sum'] += float(outcome_summary.get('depth', 0.0) or 0.0) if outcome_summary else 0.0
        transition_delta = _structure_transition_delta(observation_summary, outcome_summary)
        stat['state_transition_delta_sum'] += transition_delta
        if transition_delta >= 0.45:
            stat['state_change_count'] += 1.0

        belief_before = row.get('belief_before', {}) if isinstance(row.get('belief_before', {}), dict) else {}
        belief_after = row.get('belief_after', {}) if isinstance(row.get('belief_after', {}), dict) else {}
        conf_before = float(belief_before.get('confidence', avg_confidence) or avg_confidence)
        conf_after = float(belief_after.get('confidence', conf_before) or conf_before)
        unc_before = float(belief_before.get('uncertainty', avg_uncertainty) or avg_uncertainty)
        unc_after = float(belief_after.get('uncertainty', unc_before) or unc_before)
        stat['belief_conf_delta_sum'] += (conf_after - conf_before)
        stat['belief_unc_delta_sum'] += (unc_before - unc_after)

        probe_result = row.get('probe_result', None)
        if isinstance(probe_result, dict):
            stat['probe_total'] += 1.0
            if bool(probe_result.get('disambiguated') or probe_result.get('distinguished_hypothesis') or probe_result.get('result')):
                stat['probe_distinguish'] += 1.0
        recovery_result = row.get('recovery_result', None)
        if isinstance(recovery_result, dict):
            stat['recovery_triggered'] += 1.0
            if bool(recovery_result.get('success') or recovery_result.get('recovery_success')):
                stat['recovery_success'] += 1.0

    # Supplemental probe/recovery evidence from loop logs (if available).
    llm_log = list(getattr(loop, '_llm_advice_log', []) or []) if loop is not None else []
    recent_probe_notes = [entry for entry in llm_log[-20:] if isinstance(entry, dict) and entry.get('kind') == 'probe_result']
    probe_boost = 0.1 if recent_probe_notes else 0.0
    recovery_log = list(getattr(loop, '_recovery_log', []) or []) if loop is not None else []
    recent_recovery_events = [entry for entry in recovery_log[-10:] if isinstance(entry, dict)]

    by_signature: Dict[str, Dict[str, Any]] = {}
    by_function_aggregate: Dict[str, Dict[str, float]] = {}
    for signature, stat in signature_stats.items():
        fn, arg_bucket, belief_phase = signature
        count = max(1.0, stat['count'])
        avg_reward = stat['reward_sum'] / count
        failure_ratio = stat['failure_count'] / count
        probe_ratio = stat['probe_distinguish'] / max(1.0, stat['probe_total']) if stat['probe_total'] > 0 else 0.0
        recovery_success = stat['recovery_success'] / max(1.0, stat['recovery_triggered']) if stat['recovery_triggered'] > 0 else 0.0
        belief_shift = (stat['belief_conf_delta_sum'] + stat['belief_unc_delta_sum']) / count
        avg_transition_delta = stat['state_transition_delta_sum'] / count
        state_change_rate = stat['state_change_count'] / count
        avg_action_depth = stat['action_depth_sum'] / count
        avg_object_signal = stat['object_signal_sum'] / count
        grid_payload_rate = _clamp(stat['grid_payload_total'] / count, 0.0, 1.0)
        complexity_pressure = _clamp(
            (avg_action_depth / 5.0) * 0.45
            + avg_transition_delta * 0.35
            + _clamp(avg_object_signal / 6.0, 0.0, 1.0) * 0.2,
            0.0,
            1.0,
        )
        info_gain = (stat['info_gain_sum'] / count) + probe_ratio * 0.35 + probe_boost + avg_transition_delta * 0.1 + grid_payload_rate * 0.06

        metric_schema = {
            'long_horizon_reward': _metric_template(
                _clamp(avg_reward + 0.15 * belief_shift - avg_transition_delta * 0.05 + _clamp(avg_object_signal / 10.0, 0.0, 0.12), -1.0, 1.0),
                confidence=min(1.0, 0.25 + 0.12 * count),
                sample_count=count,
            ),
            'predicted_risk': _metric_template(
                _clamp(failure_ratio + (1.0 - recovery_success) * 0.2 + avg_transition_delta * 0.12 + complexity_pressure * 0.08, 0.0, 1.0),
                confidence=min(1.0, 0.22 + 0.1 * count),
                sample_count=count,
            ),
            'reversibility': _metric_template(
                _clamp(
                    (0.6 if fn in {'compute_stats', 'filter_by_predicate', 'probe'} else 0.4)
                    + recovery_success * 0.2
                    - failure_ratio * 0.15
                    - avg_transition_delta * 0.08
                    + grid_payload_rate * 0.04,
                    0.0,
                    1.0,
                ),
                confidence=min(1.0, 0.2 + 0.09 * count),
                sample_count=count,
            ),
            'info_gain': _metric_template(
                _clamp(info_gain + state_change_rate * 0.08, 0.0, 1.0),
                confidence=min(1.0, 0.2 + 0.1 * max(stat['probe_total'], count * 0.5)),
                sample_count=max(stat['probe_total'], count),
            ),
            'constraint_violation': _clamp(failure_ratio * 0.7 + complexity_pressure * 0.15, 0.0, 1.0),
            'cold_start': False,
        }
        sig_key = f'{fn}|{arg_bucket}|{belief_phase}'
        entry = {
            'key': _canonical_transition_prior_key(
                {'function_name': fn, 'arg_bucket': arg_bucket, 'belief_phase': belief_phase},
                default_phase=default_phase,
            ),
            'metrics': metric_schema,
            'state_model': {
                'avg_action_depth': round(avg_action_depth, 4),
                'avg_object_signal': round(avg_object_signal, 4),
                'grid_payload_rate': round(grid_payload_rate, 4),
                'avg_observation_depth': round(stat['observation_depth_sum'] / count, 4),
                'avg_outcome_depth': round(stat['outcome_depth_sum'] / count, 4),
                'avg_transition_delta': round(avg_transition_delta, 4),
                'state_change_rate': round(state_change_rate, 4),
            },
        }
        entry.update(_flatten_metric_schema(metric_schema, base={'cold_start': False}))
        by_signature[sig_key] = entry
        fn_acc = by_function_aggregate.setdefault(fn, {'count': 0.0, 'long_horizon_reward': 0.0, 'predicted_risk': 0.0, 'reversibility': 0.0, 'info_gain': 0.0, 'constraint_violation': 0.0})
        fn_acc['count'] += 1.0
        for metric in ('long_horizon_reward', 'predicted_risk', 'reversibility', 'info_gain', 'constraint_violation'):
            fn_acc[metric] += float(entry.get(metric, 0.0) or 0.0)

    perception = perception_summary if isinstance(perception_summary, dict) else {}
    high_motion = float(perception.get('camera_motion_score', 0.0) or 0.0) >= 0.6
    for sig_key, entry in by_signature.items():
        if high_motion and entry.get('key', {}).get('function_name') in {'join_tables', 'aggregate_group'}:
            entry['predicted_risk'] = max(float(entry.get('predicted_risk', 0.0) or 0.0), 0.65)
            entry['constraint_violation'] = max(float(entry.get('constraint_violation', 0.0) or 0.0), 0.45)
            metrics = entry.get('metrics', {})
            if isinstance(metrics, dict):
                if isinstance(metrics.get('predicted_risk'), dict):
                    metrics['predicted_risk']['value'] = float(entry['predicted_risk'])
                metrics['constraint_violation'] = float(entry['constraint_violation'])
            by_signature[sig_key] = entry

    if recent_recovery_events and by_signature:
        for entry in by_signature.values():
            entry['predicted_risk'] = _clamp(float(entry.get('predicted_risk', 0.0) or 0.0) + 0.05, 0.0, 1.0)
            metrics = entry.get('metrics', {})
            if isinstance(metrics, dict) and isinstance(metrics.get('predicted_risk'), dict):
                metrics['predicted_risk']['value'] = float(entry['predicted_risk'])

    if by_signature:
        for entry in by_signature.values():
            key = entry.get('key', {}) if isinstance(entry.get('key', {}), dict) else {}
            fn_name = str(key.get('function_name', '') or '')
            belief_phase = _normalize_phase(str(key.get('belief_phase', default_phase) or default_phase))
            phase_adjustment = retention_learning['phase'].get(belief_phase, {}) if isinstance(retention_learning.get('phase', {}), dict) else {}
            function_adjustment = retention_learning['function'].get(fn_name, {}) if isinstance(retention_learning.get('function', {}), dict) else {}
            total_reward_delta = float(phase_adjustment.get('reward_delta', 0.0) or 0.0) + float(function_adjustment.get('reward_delta', 0.0) or 0.0)
            total_risk_delta = float(phase_adjustment.get('risk_delta', 0.0) or 0.0) + float(function_adjustment.get('risk_delta', 0.0) or 0.0)
            total_info_delta = float(phase_adjustment.get('info_gain_delta', 0.0) or 0.0) + float(function_adjustment.get('info_gain_delta', 0.0) or 0.0)
            total_constraint_delta = float(phase_adjustment.get('constraint_delta', 0.0) or 0.0) + float(function_adjustment.get('constraint_delta', 0.0) or 0.0)
            total_confidence_delta = float(phase_adjustment.get('confidence_delta', 0.0) or 0.0) + float(function_adjustment.get('confidence_delta', 0.0) or 0.0)
            if max(abs(total_reward_delta), abs(total_risk_delta), abs(total_info_delta), abs(total_constraint_delta), abs(total_confidence_delta)) <= 1e-6:
                continue
            entry['long_horizon_reward'] = _clamp(float(entry.get('long_horizon_reward', 0.0) or 0.0) + total_reward_delta, -1.0, 1.0)
            entry['predicted_risk'] = _clamp(float(entry.get('predicted_risk', 0.0) or 0.0) + total_risk_delta, 0.0, 1.0)
            entry['info_gain'] = _clamp(float(entry.get('info_gain', 0.0) or 0.0) + total_info_delta, 0.0, 1.0)
            entry['constraint_violation'] = _clamp(float(entry.get('constraint_violation', 0.0) or 0.0) + total_constraint_delta, 0.0, 1.0)
            entry['retention_learning_bias'] = {
                'phase_reward_delta': round(float(phase_adjustment.get('reward_delta', 0.0) or 0.0), 4),
                'phase_risk_delta': round(float(phase_adjustment.get('risk_delta', 0.0) or 0.0), 4),
                'function_reward_delta': round(float(function_adjustment.get('reward_delta', 0.0) or 0.0), 4),
                'function_risk_delta': round(float(function_adjustment.get('risk_delta', 0.0) or 0.0), 4),
            }
            metrics = entry.get('metrics', {}) if isinstance(entry.get('metrics', {}), dict) else {}
            for metric_name, field_name in (
                ('long_horizon_reward', 'long_horizon_reward'),
                ('predicted_risk', 'predicted_risk'),
                ('info_gain', 'info_gain'),
            ):
                metric = metrics.get(metric_name, {}) if isinstance(metrics.get(metric_name, {}), dict) else {}
                metric['value'] = float(entry.get(field_name, 0.0) or 0.0)
                metric['confidence'] = _clamp(float(metric.get('confidence', 0.0) or 0.0) + total_confidence_delta, 0.0, 1.0)
                metrics[metric_name] = metric
            metrics['constraint_violation'] = float(entry.get('constraint_violation', 0.0) or 0.0)
            entry['metrics'] = metrics

    cold_start_template = _domain_task_family_template(domain_hint, task_family)
    world_predicted_phase = hidden_phase if hidden_phase and hidden_updates > 0 and hidden_phase_confidence >= 0.45 else str(world_dynamics.get('predicted_phase', '') or default_phase)
    if expected_next_phase and expected_next_phase_confidence >= max(0.42, hidden_phase_confidence - 0.03):
        world_predicted_phase = expected_next_phase
    priors: Dict[str, Any] = {
        '__schema_version': 'wm_transition_priors_v2',
        '__key_fields': ['function_name', 'arg_bucket', 'belief_phase'],
        '__by_signature': by_signature,
        '__world_dynamics': {
            'predicted_phase': world_predicted_phase,
            'transition_confidence': _clamp(
                (
                    (
                        float(world_dynamics.get('transition_confidence', 0.0) or 0.0) * 0.56
                        + hidden_phase_confidence * 0.24
                        + expected_next_phase_confidence * 0.20
                    ) if hidden_updates > 0 else float(world_dynamics.get('transition_confidence', 0.0) or 0.0)
                ) + float(retention_learning.get('transition_confidence_delta', 0.0) or 0.0),
                0.0,
                1.0,
            ),
            'state_shift_risk': _clamp(
                max(
                    float(world_dynamics.get('shift_risk', 0.0) or 0.0),
                    hidden_drift_score * 0.82 if hidden_updates > 0 else 0.0,
                    (0.15 + phase_transition_entropy * 0.32) if hidden_updates > 1 and expected_next_phase_confidence < 0.45 else 0.0,
                ) + float(retention_learning.get('world_shift_risk_delta', 0.0) or 0.0),
                0.0,
                1.0,
            ),
            'reward_volatility': float(world_dynamics.get('reward_volatility', 0.0) or 0.0),
            'failure_rate': float(world_dynamics.get('failure_rate', 0.0) or 0.0),
            'blocked_functions': list(dict.fromkeys(list(world_dynamics.get('blocked_functions', []) or []) + list(risky_transition_functions) + list(retention_learning.get('blocked_functions', []) or []))),
            'preferred_action_classes': list(world_dynamics.get('preferred_action_classes', []) or []),
            'required_probes': list(dict.fromkeys(list(world_dynamics.get('required_probes', []) or []) + (['probe_transition_frontier'] if phase_transition_entropy >= 0.62 else []) + list(retention_learning.get('required_probes', []) or []))),
            'state_abstraction': dict(world_dynamics.get('state_abstraction', {})),
            'hidden_state': hidden_state,
            'dominant_branch_id': dominant_branch_id,
            'latent_branches': latent_branches,
            'expected_next_phase': expected_next_phase,
            'expected_next_phase_confidence': expected_next_phase_confidence,
            'phase_transition_entropy': phase_transition_entropy,
            'transition_memory': transition_memory,
            'stabilizing_functions': stabilizing_transition_functions,
            'risky_functions': risky_transition_functions,
        },
        '__cold_start_prior': _flatten_metric_schema(
            cold_start_template,
            base={
                'cold_start': True,
                'key': _canonical_transition_prior_key(
                    {'function_name': '*', 'arg_bucket': '*', 'belief_phase': default_phase},
                    default_phase=default_phase,
                ),
                'metrics': cold_start_template,
            },
        ),
        '__legacy_by_function': {},
        '__retention_learning': {
            'matched_contexts': list(retention_learning.get('matched_contexts', []) or []),
            'phase_adjustments': {
                str(name): {metric: round(float(value or 0.0), 4) for metric, value in payload.items()}
                for name, payload in (retention_learning.get('phase', {}) or {}).items()
                if isinstance(payload, dict)
            },
            'function_adjustments': {
                str(name): {metric: round(float(value or 0.0), 4) for metric, value in payload.items()}
                for name, payload in (retention_learning.get('function', {}) or {}).items()
                if isinstance(payload, dict)
            },
        },
    }

    current_phase_for_memory = hidden_phase or default_phase
    for fn_name, payload in phase_function_scores.items():
        if not isinstance(payload, dict):
            continue
        support = min(1.0, float(payload.get('support', 0.0) or 0.0) / 3.0)
        stabilizing_score = _clamp(float(payload.get('stabilizing_score', 0.0) or 0.0), 0.0, 1.0)
        risk_score = _clamp(float(payload.get('risk_score', 0.0) or 0.0), 0.0, 1.0)
        depth_gain = _clamp(float(payload.get('avg_depth_gain', 0.0) or 0.0), 0.0, 1.0)
        affinity = (stabilizing_score - risk_score) * max(0.25, expected_next_phase_confidence or hidden_phase_confidence) * max(0.2, support)

        legacy_entry = priors.setdefault(
            str(fn_name),
            {
                'long_horizon_reward': cold_start_template['long_horizon_reward']['value'],
                'predicted_risk': cold_start_template['predicted_risk']['value'],
                'reversibility': cold_start_template['reversibility']['value'],
                'constraint_violation': float(cold_start_template.get('constraint_violation', 0.0) or 0.0),
                'info_gain': cold_start_template['info_gain']['value'],
            },
        )
        legacy_entry['long_horizon_reward'] = _clamp(float(legacy_entry.get('long_horizon_reward', 0.0) or 0.0) + affinity * 0.28 + depth_gain * 0.06 * support, -1.0, 1.0)
        legacy_entry['predicted_risk'] = _clamp(float(legacy_entry.get('predicted_risk', 0.0) or 0.0) - affinity * 0.24 + risk_score * 0.10 * support, 0.0, 1.0)
        legacy_entry['info_gain'] = _clamp(float(legacy_entry.get('info_gain', 0.0) or 0.0) + depth_gain * 0.08 * support + (0.03 * support if stabilizing_score < 0.45 else 0.0), 0.0, 1.0)
        legacy_entry['constraint_violation'] = _clamp(float(legacy_entry.get('constraint_violation', 0.0) or 0.0) + risk_score * 0.08 * support - stabilizing_score * 0.04 * support, 0.0, 1.0)
        legacy_entry['transition_affinity'] = round(affinity, 4)
        legacy_entry['prior_confidence'] = max(float(legacy_entry.get('prior_confidence', 0.0) or 0.0), round(max(expected_next_phase_confidence, hidden_phase_confidence) * support, 4))
        priors['__legacy_by_function'][str(fn_name)] = dict(legacy_entry)

        for entry in by_signature.values():
            key = entry.get('key', {}) if isinstance(entry.get('key', {}), dict) else {}
            if str(key.get('function_name', '') or '') != str(fn_name):
                continue
            if str(key.get('belief_phase', '') or '') not in {current_phase_for_memory, default_phase}:
                continue
            entry['long_horizon_reward'] = _clamp(float(entry.get('long_horizon_reward', 0.0) or 0.0) + affinity * 0.22, -1.0, 1.0)
            entry['predicted_risk'] = _clamp(float(entry.get('predicted_risk', 0.0) or 0.0) - affinity * 0.18 + risk_score * 0.07 * support, 0.0, 1.0)
            entry['info_gain'] = _clamp(float(entry.get('info_gain', 0.0) or 0.0) + depth_gain * 0.06 * support, 0.0, 1.0)
            entry['constraint_violation'] = _clamp(float(entry.get('constraint_violation', 0.0) or 0.0) + risk_score * 0.05 * support - stabilizing_score * 0.03 * support, 0.0, 1.0)
            entry['transition_affinity'] = round(affinity, 4)
            metrics = entry.get('metrics', {}) if isinstance(entry.get('metrics', {}), dict) else {}
            for metric_name in ('long_horizon_reward', 'predicted_risk', 'info_gain'):
                metric = metrics.get(metric_name, {}) if isinstance(metrics.get(metric_name, {}), dict) else {}
                metric['value'] = float(entry.get(metric_name, 0.0) or 0.0)
                metrics[metric_name] = metric
            metrics['constraint_violation'] = float(entry.get('constraint_violation', 0.0) or 0.0)
            entry['metrics'] = metrics

    aggregated_entries: Dict[str, Dict[str, float]] = {}
    for entry in by_signature.values():
        key = entry.get('key', {}) if isinstance(entry.get('key', {}), dict) else {}
        fn_name = str(key.get('function_name', '') or '')
        if not fn_name:
            continue
        bucket = aggregated_entries.setdefault(
            fn_name,
            {
                'count': 0.0,
                'long_horizon_reward': 0.0,
                'predicted_risk': 0.0,
                'reversibility': 0.0,
                'info_gain': 0.0,
                'constraint_violation': 0.0,
            },
        )
        bucket['count'] += 1.0
        for metric_name in ('long_horizon_reward', 'predicted_risk', 'reversibility', 'info_gain', 'constraint_violation'):
            bucket[metric_name] += float(entry.get(metric_name, 0.0) or 0.0)

    for fn, agg in aggregated_entries.items():
        denom = max(1.0, float(agg.get('count', 1.0) or 1.0))
        existing_legacy = priors.get(fn, {}) if isinstance(priors.get(fn, {}), dict) else {}
        legacy_entry = {
            'long_horizon_reward': _clamp(agg['long_horizon_reward'] / denom, -1.0, 1.0),
            'predicted_risk': _clamp(agg['predicted_risk'] / denom, 0.0, 1.0),
            'reversibility': _clamp(agg['reversibility'] / denom, 0.0, 1.0),
            'info_gain': _clamp(agg['info_gain'] / denom, 0.0, 1.0),
            'constraint_violation': _clamp(agg['constraint_violation'] / denom, 0.0, 1.0),
        }
        if existing_legacy:
            legacy_entry['long_horizon_reward'] = _clamp(
                legacy_entry['long_horizon_reward'] * 0.84 + float(existing_legacy.get('long_horizon_reward', 0.0) or 0.0) * 0.16,
                -1.0,
                1.0,
            )
            legacy_entry['predicted_risk'] = _clamp(
                legacy_entry['predicted_risk'] * 0.86 + float(existing_legacy.get('predicted_risk', 0.0) or 0.0) * 0.14,
                0.0,
                1.0,
            )
            legacy_entry['info_gain'] = _clamp(
                legacy_entry['info_gain'] * 0.88 + float(existing_legacy.get('info_gain', 0.0) or 0.0) * 0.12,
                0.0,
                1.0,
            )
            legacy_entry['constraint_violation'] = _clamp(
                legacy_entry['constraint_violation'] * 0.90 + float(existing_legacy.get('constraint_violation', 0.0) or 0.0) * 0.10,
                0.0,
                1.0,
            )
            if 'transition_affinity' in existing_legacy:
                legacy_entry['transition_affinity'] = float(existing_legacy.get('transition_affinity', 0.0) or 0.0)
            if 'prior_confidence' in existing_legacy:
                legacy_entry['prior_confidence'] = float(existing_legacy.get('prior_confidence', 0.0) or 0.0)
        # Backward compatibility: keep function-key readable fields at top-level.
        priors[fn] = legacy_entry
        priors['__legacy_by_function'][fn] = dict(legacy_entry)

    belief_snapshot = provider.beliefs()
    active_beliefs = list(belief_snapshot.get('active', [])) if isinstance(belief_snapshot, dict) else []
    for belief in active_beliefs:
        variable_name = str(getattr(belief, 'variable_name', '') or '')
        if not variable_name.startswith('mechanism_prior_'):
            continue
        fn_name = variable_name.replace('mechanism_prior_', '', 1)
        if not fn_name:
            continue
        confidence = max(0.0, min(1.0, float(getattr(belief, 'confidence', 0.0) or 0.0)))
        posterior = str(getattr(belief, 'posterior', '') or '')
        p = priors.setdefault(
            fn_name,
            {
                'long_horizon_reward': 0.0,
                'predicted_risk': 0.35,
                'reversibility': 0.35,
                'constraint_violation': 0.0,
                'info_gain': 0.1,
                'prior_confidence': 0.5,
            },
        )
        if posterior == 'supported_transition':
            p['long_horizon_reward'] = min(1.0, float(p.get('long_horizon_reward', 0.0)) + 0.15 * confidence)
            p['predicted_risk'] = max(0.0, float(p.get('predicted_risk', 0.35)) - 0.25 * confidence)
        elif posterior == 'refuted_transition':
            p['long_horizon_reward'] = max(-1.0, float(p.get('long_horizon_reward', 0.0)) - 0.2 * confidence)
            p['predicted_risk'] = min(1.0, float(p.get('predicted_risk', 0.35)) + 0.3 * confidence)
        p['info_gain'] = min(1.0, float(p.get('info_gain', 0.1)) + 0.05 * confidence)
        p['prior_confidence'] = max(float(p.get('prior_confidence', 0.0) or 0.0), confidence)
        if isinstance(priors.get('__legacy_by_function'), dict):
            priors['__legacy_by_function'][fn_name] = dict(p)

    return priors


def build_unified_cognitive_context(
    provider: ContextProvider,
    episode: int,
    tick: int,
    obs: Optional[Dict[str, Any]],
    continuity_snapshot: Optional[Dict[str, Any]] = None,
    *,
    unified_enabled: bool = True,
    ablation_mode: str = 'stripped',
    world_model_summary: Optional[Dict[str, Any]] = None,
    self_model_summary: Optional[Dict[str, Any]] = None,
    recent_failures: Optional[int] = None,
    world_shift_risk: Optional[float] = None,
) -> UnifiedCognitiveContext:
    obs_safe = obs if isinstance(obs, dict) else {}
    perception = obs_safe.get('perception', {}) if isinstance(obs_safe.get('perception', {}), dict) else {}
    if world_model_summary is None:
        wm_summary = build_world_model_context(provider, perception)
        wm_transition_priors = build_world_model_transition_priors(provider, perception)
        wm_summary = _enrich_world_model_summary_with_priors(wm_summary, wm_transition_priors)
    else:
        wm_summary = world_model_summary
    sm_summary = self_model_summary or provider.self_model_summary()
    control_snapshot = provider.meta_control_snapshot(episode, tick)
    retrieval_pressure = float((control_snapshot or {}).get('retrieval_pressure', 0.0) or 0.0)
    probe_pressure = float((control_snapshot or {}).get('probe_bias', 0.0) or 0.0)
    plan_snapshot = provider.plan_snapshot()
    active_hypotheses = provider.hypotheses_snapshot(limit=3)
    workspace_state = _provider_workspace_state(provider)
    cognitive_object_records = _provider_cognitive_object_records(provider, limit=8)
    return UnifiedContextBuilder.build(
        UnifiedContextInput(
            unified_enabled=unified_enabled,
            unified_ablation_mode=ablation_mode,
            ablation_mode_validated=None,
            obs=obs,
            continuity_snapshot=continuity_snapshot,
            world_model_summary=wm_summary,
            self_model_summary=sm_summary,
            plan_summary=plan_snapshot.get('plan_summary', {}),
            current_task=str(plan_snapshot.get('current_task', '') or ''),
            active_hypotheses=active_hypotheses,
            episode_trace_tail=provider.episode_trace()[-5:],
            retrieval_should_query=provider.retrieval_should_query(),
            probe_pressure=probe_pressure,
            retrieval_pressure=retrieval_pressure,
            recent_failures=recent_failures,
            world_shift_risk=world_shift_risk,
            workspace_state=workspace_state,
            cognitive_object_records=cognitive_object_records,
        )
    )


def build_tick_context(
    provider: ContextProvider,
    episode: int,
    tick: int,
    obs_before: Optional[Dict[str, Any]],
    continuity_snapshot: Optional[Dict[str, Any]],
    *,
    unified_enabled: bool = True,
    ablation_mode: str = 'stripped',
) -> TickContextFrame:
    """Build tick-local context exactly once, then pass through downstream stages."""
    obs_safe = obs_before if isinstance(obs_before, dict) else {}
    perception_summary = (
        obs_safe.get('perception', {})
        if isinstance(obs_safe.get('perception', {}), dict)
        else {}
    )
    meta_control_snapshot = provider.meta_control_snapshot(
        episode,
        tick,
        context={'phase': 'tick_context_frame'},
    )
    world_model_transition_priors = build_world_model_transition_priors(provider, perception_summary)
    world_model_summary = _enrich_world_model_summary_with_priors(
        build_world_model_context(provider, perception_summary),
        world_model_transition_priors,
    )
    self_model_summary = provider.self_model_summary()
    unified_context = build_unified_cognitive_context(
        provider,
        episode=episode,
        tick=tick,
        obs=obs_safe,
        continuity_snapshot=continuity_snapshot,
        unified_enabled=unified_enabled,
        ablation_mode=ablation_mode,
        world_model_summary=world_model_summary,
        self_model_summary=self_model_summary,
    )
    frame = TickContextFrame(
        episode=episode,
        tick=tick,
        perception_summary=dict(perception_summary),
        meta_control_snapshot=meta_control_snapshot,
        world_model_summary=world_model_summary,
        world_model_transition_priors=world_model_transition_priors,
        self_model_summary=self_model_summary,
        unified_context=unified_context,
    )
    return frame


def build_legacy_decision_context(
    loop,
    *,
    frame: TickContextFrame,
    unified_context: Optional[UnifiedCognitiveContext],
    obs_before: Optional[Dict[str, Any]],
    continuity_snapshot: Optional[Dict[str, Any]],
    surfaced: List[Any],
    plan_tick_meta: Dict[str, Any],
    recent_failures: int,
    runtime_input: Optional[LegacyContextRuntimeInput] = None,
) -> Dict[str, Any]:
    """Backward-compat bridge (deprecated): unified context -> legacy decision context schema."""
    runtime = runtime_input or _legacy_runtime_input_from_loop(loop, plan_tick_meta)
    unified_payload = unified_context.to_dict() if isinstance(unified_context, UnifiedCognitiveContext) else {}
    self_model_summary = unified_payload.get('self_model_summary', {}) if isinstance(unified_payload.get('self_model_summary', {}), dict) else {}
    plan_state = getattr(loop, '_plan_state', None)
    fallback_has_plan = bool(getattr(plan_state, 'has_plan', False))
    fallback_step_intent = ''
    if plan_state is not None and hasattr(plan_state, 'get_intent_for_step'):
        fallback_step_intent = str(plan_state.get_intent_for_step() or '')
    if 'plan_state_summary' in unified_payload:
        plan_summary = unified_payload.get('plan_state_summary')
    else:
        plan_summary = {'has_plan': fallback_has_plan}
    step_intent = unified_payload.get('current_task', fallback_step_intent)
    if 'active_beliefs_summary' in unified_payload:
        world_model_summary = unified_payload.get('active_beliefs_summary')
    else:
        world_model_summary = frame.world_model_summary
    legacy_context = {
        'episode': runtime.episode,
        'tick': runtime.tick,
        'continuity_snapshot': continuity_snapshot,
        'surfaced': surfaced[:3] if surfaced else [],
        'reward_trend': 'positive' if runtime.episode_reward > 0 else ('negative' if runtime.episode_trace else 'neutral'),
        'recovery_pending': bool(runtime.recovery_log and 'recovery' in str(runtime.recovery_log[-1:])),
        'recent_failures': recent_failures,
        'plan_summary': plan_summary if isinstance(plan_summary, dict) else {'has_plan': fallback_has_plan},
        'step_intent': str(step_intent or ''),
        'perception_summary': (obs_before or {}).get('perception', {}) if isinstance((obs_before or {}).get('perception', {}), dict) else {},
        'world_model_summary': world_model_summary if isinstance(world_model_summary, dict) else {},
        'world_model_transition_priors': frame.world_model_transition_priors,
        'policy_profile': runtime.policy_profile,
        'representation_profile': runtime.representation_profile,
        'prediction_enabled': runtime.prediction_enabled,
        'predictor_trust': runtime.predictor_trust,
        'procedure_enabled': runtime.procedure_enabled,
        'self_model_reliability': float(self_model_summary.get('global_reliability', 0.5) or 0.5),
        'recovery_availability': float(self_model_summary.get('recovery_availability', 0.5) or 0.5),
        'resource_pressure': unified_payload.get('resource_pressure', str(self_model_summary.get('resource_tightness', 'normal') or 'normal')),
        'recent_failure_profile': unified_payload.get('recent_failure_profile', list(self_model_summary.get('recent_failure_modes', [])) if isinstance(self_model_summary.get('recent_failure_modes', []), list) else []),
        'self_model_summary': self_model_summary,
        'unified_cognitive_context': unified_payload,
        'legacy_context_deprecated': True,
        'unified_context_ablation_mode': runtime.unified_context_mode,
    }
    alignment_report = check_context_field_alignment(legacy_context, unified_payload)
    loop._last_context_alignment_report = alignment_report
    return legacy_context


def _legacy_runtime_input_from_loop(loop: Any, plan_tick_meta: Dict[str, Any]) -> LegacyContextRuntimeInput:
    prediction_enabled = bool(getattr(loop, '_prediction_enabled', False))
    predictor_trust = loop._prediction_registry.get_predictor_trust() if prediction_enabled else {}
    unified_context_mode = loop._ablation_flags_snapshot().get('unified_context_mode', 'full') if hasattr(loop, '_ablation_flags_snapshot') else 'full'
    return LegacyContextRuntimeInput(
        episode=int(getattr(loop, '_episode', 0)),
        tick=int(getattr(loop, '_tick', 0)),
        episode_reward=float(getattr(loop, '_episode_reward', 0.0) or 0.0),
        episode_trace=list(getattr(loop, '_episode_trace', []) or []),
        recovery_log=list(getattr(loop, '_recovery_log', []) or []),
        prediction_enabled=prediction_enabled,
        predictor_trust=predictor_trust if isinstance(predictor_trust, dict) else {},
        procedure_enabled=bool(getattr(loop, '_procedure_enabled', False)),
        policy_profile=plan_tick_meta.get('policy_profile', loop._get_policy_profile()) if isinstance(plan_tick_meta, dict) else loop._get_policy_profile(),
        representation_profile=plan_tick_meta.get('representation_profile', loop._get_representation_profile()) if isinstance(plan_tick_meta, dict) else loop._get_representation_profile(),
        unified_context_mode=str(unified_context_mode or 'full'),
    )


def check_context_field_alignment(legacy_context: Dict[str, Any], unified_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate bridge-field compatibility consumed by planner/governance/testing pipelines."""
    legacy_required = {
        'plan_summary',
        'step_intent',
        'world_model_summary',
        'world_model_transition_priors',
        'policy_profile',
        'representation_profile',
        'unified_cognitive_context',
    }
    unified_expected = {
        'plan_state_summary': 'plan_summary',
        'current_task': 'step_intent',
        'active_beliefs_summary': 'world_model_summary',
    }

    missing_legacy = sorted([key for key in legacy_required if key not in legacy_context])
    missing_unified = sorted([key for key in unified_expected if key not in unified_payload])

    if 'plan_summary' in missing_legacy and isinstance(unified_payload.get('plan_state_summary'), dict):
        legacy_context['plan_summary'] = dict(unified_payload.get('plan_state_summary', {}))
        missing_legacy.remove('plan_summary')
    if 'step_intent' in missing_legacy:
        legacy_context['step_intent'] = str(unified_payload.get('current_task', '') or '')
        if legacy_context['step_intent']:
            missing_legacy.remove('step_intent')
    if 'world_model_summary' in missing_legacy and isinstance(unified_payload.get('active_beliefs_summary'), dict):
        legacy_context['world_model_summary'] = dict(unified_payload.get('active_beliefs_summary', {}))
        missing_legacy.remove('world_model_summary')

    return {
        'ok': not missing_legacy,
        'missing_legacy_fields': missing_legacy,
        'missing_unified_fields': missing_unified,
        'checked_mappings': unified_expected,
    }

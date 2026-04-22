"""Unified world-model control protocol.

Separates belief summary from decision/planning constraints and now exposes
a minimal canonical world-state spine (modality / entities / relations / events)
so downstream modules can reason over more than phase-only summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _prediction_trust_score(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return _clamp01(raw, 0.5)
    if isinstance(raw, str):
        raw = {'ensemble': raw}
    if not isinstance(raw, dict):
        return 0.5
    label_scores = {
        'high': 0.85,
        'medium': 0.60,
        'low': 0.30,
    }
    scores = [
        label_scores[str(value or '').strip().lower()]
        for value in raw.values()
        if str(value or '').strip().lower() in label_scores
    ]
    if not scores:
        return 0.5
    return _clamp01(sum(scores) / float(len(scores)), 0.5)


def _recent_prediction_error(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return _clamp01(raw, 0.0)
    if not isinstance(raw, list):
        return 0.0
    errors: List[float] = []
    for item in raw[-5:]:
        if isinstance(item, dict):
            errors.append(_clamp01(item.get('total_error', 0.0), 0.0))
        else:
            errors.append(_clamp01(item, 0.0))
    if not errors:
        return 0.0
    return _clamp01(sum(errors) / float(len(errors)), 0.0)


def _as_function_names(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    names: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get('function_name', '') or '')
        else:
            name = str(item or '')
        if name and name not in names:
            names.append(name)
    return names


def _as_latent_branches(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    branches: List[Dict[str, Any]] = []
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        anchor_functions = _as_function_names(item.get('anchor_functions', []))
        risky_functions = _as_function_names(item.get('risky_functions', []))
        branches.append({
            'branch_id': str(item.get('branch_id', '') or ''),
            'current_phase': str(item.get('current_phase', '') or ''),
            'target_phase': str(item.get('target_phase', '') or ''),
            'confidence': _clamp01(item.get('confidence', 0.0), 0.0),
            'support': _clamp01(item.get('support', 0.0), 0.0),
            'transition_score': _clamp01(item.get('transition_score', 0.0), 0.0),
            'success_rate': _clamp01(item.get('success_rate', 0.0), 0.0),
            'avg_reward': float(item.get('avg_reward', 0.0) or 0.0),
            'avg_depth_gain': _clamp01(item.get('avg_depth_gain', 0.0), 0.0),
            'uncertainty_pressure': _clamp01(item.get('uncertainty_pressure', 0.0), 0.0),
            'anchor_functions': anchor_functions,
            'risky_functions': risky_functions,
            'latent_signature': str(item.get('latent_signature', '') or ''),
        })
    return branches


def _as_rows(raw: Any, *, limit: int = 6) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in raw[:limit]:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


@dataclass
class WorldModelControlProtocol:
    blocked_functions: List[str] = field(default_factory=list)
    preferred_action_classes: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    predicted_phase: str = ""
    transition_confidence: float = 0.0
    state_shift_risk: float = 0.0
    required_probes: List[str] = field(default_factory=list)

    hidden_state_phase: str = ""
    hidden_phase_confidence: float = 0.0
    hidden_state_depth: int = 0
    hidden_drift_score: float = 0.0
    hidden_uncertainty_score: float = 0.0
    hidden_focus_functions: List[str] = field(default_factory=list)
    hidden_latent_signature: str = ""
    dominant_branch_id: str = ""
    latent_branches: List[Dict[str, Any]] = field(default_factory=list)
    expected_next_phase: str = ""
    expected_next_phase_confidence: float = 0.0
    phase_transition_entropy: float = 1.0
    stabilizing_focus_functions: List[str] = field(default_factory=list)
    risky_focus_functions: List[str] = field(default_factory=list)

    observed_modality: str = ""
    world_state_signature: str = ""
    world_entity_count: int = 0
    world_relation_count: int = 0
    last_event_signature: str = ""
    world_novelty_score: float = 0.0
    predicted_transitions: List[Dict[str, Any]] = field(default_factory=list)
    mechanism_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    discriminating_tests: List[Dict[str, Any]] = field(default_factory=list)
    candidate_intervention_targets: List[Dict[str, Any]] = field(default_factory=list)
    counterfactual_contrasts: List[Dict[str, Any]] = field(default_factory=list)
    expected_information_gain: float = 0.0
    rollout_uncertainty: float = 0.5

    prediction_trust_score: float = 0.5
    recent_prediction_error: float = 0.0
    control_trust: float = 0.5

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "WorldModelControlProtocol":
        def _as_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if str(item or '')]

        def _as_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def _pick(*values: Any) -> Any:
            for value in values:
                if value not in (None, '', []):
                    return value
            return None

        def _merge_lists(*values: Any) -> List[str]:
            merged: List[str] = []
            for value in values:
                for item in _as_list(value):
                    if item not in merged:
                        merged.append(item)
            return merged

        raw = context.get('world_model_control', {})
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raw = {}

        summary = context.get('world_model_summary', {})
        if not isinstance(summary, dict):
            summary = {}
        summary_hints = _as_dict(summary.get('control_hints', {}))
        priors = context.get('world_model_transition_priors', {})
        if not isinstance(priors, dict):
            priors = {}
        dynamics = _as_dict(priors.get('__world_dynamics', {}))
        world_model_constraints = _as_dict(context.get('world_model_constraints', {}))
        global_failure_strategy = _as_dict(context.get('global_failure_strategy', {}))
        failure_strategy_profile = _as_dict(context.get('failure_strategy_profile', {}))
        hidden_state = _as_dict(_pick(
            raw.get('hidden_state'),
            summary.get('hidden_state'),
            summary_hints.get('hidden_state'),
            dynamics.get('hidden_state'),
        ))
        hidden_transition_memory = _as_dict(_pick(
            raw.get('transition_memory'),
            summary.get('transition_memory'),
            summary_hints.get('transition_memory'),
            hidden_state.get('transition_memory'),
            dynamics.get('transition_memory'),
        ))
        latent_branches = _as_latent_branches(_pick(
            raw.get('latent_branches'),
            summary.get('latent_branches'),
            summary_hints.get('latent_branches'),
            hidden_state.get('latent_branches'),
            hidden_transition_memory.get('latent_branches'),
            dynamics.get('latent_branches'),
            [],
        ))
        dominant_branch_id = str(_pick(
            raw.get('dominant_branch_id'),
            summary.get('dominant_branch_id'),
            summary_hints.get('dominant_branch_id'),
            hidden_state.get('dominant_branch_id'),
            hidden_transition_memory.get('dominant_branch_id'),
            dynamics.get('dominant_branch_id'),
            latent_branches[0].get('branch_id') if latent_branches else '',
            '',
        ) or '')

        blocked = _merge_lists(
            raw.get('blocked_functions'),
            raw.get('blocked_action_classes'),
            summary.get('blocked_functions'),
            summary.get('blocked_action_classes'),
            summary_hints.get('blocked_functions'),
            summary_hints.get('blocked_action_classes'),
            dynamics.get('blocked_functions'),
            dynamics.get('blocked_action_classes'),
            world_model_constraints.get('blocked_functions'),
            world_model_constraints.get('blocked_action_classes'),
            context.get('blocked_functions'),
            context.get('blocked_action_classes'),
            global_failure_strategy.get('blocked_action_classes'),
            failure_strategy_profile.get('blocked_action_classes'),
        )

        preferred_action_classes = _as_list(_pick(
            raw.get('preferred_action_classes'),
            summary.get('preferred_action_classes'),
            summary_hints.get('preferred_action_classes'),
            dynamics.get('preferred_action_classes'),
        ))
        hard_constraints = _as_list(_pick(
            raw.get('hard_constraints'),
            summary.get('hard_constraints'),
            summary_hints.get('hard_constraints'),
        ))
        required_probes = _as_list(_pick(
            raw.get('required_probes'),
            summary.get('required_probes'),
            summary_hints.get('required_probes'),
            dynamics.get('required_probes'),
        ))

        predicted_phase = str(_pick(
            raw.get('predicted_phase'),
            summary.get('predicted_phase'),
            summary_hints.get('predicted_phase'),
            dynamics.get('predicted_phase'),
            '',
        ) or '')
        transition_confidence = float(_pick(
            raw.get('transition_confidence'),
            summary.get('transition_confidence'),
            summary_hints.get('transition_confidence'),
            dynamics.get('transition_confidence'),
            0.0,
        ) or 0.0)
        shift_risk = float(_pick(
            raw.get('state_shift_risk'),
            summary.get('shift_risk'),
            summary_hints.get('state_shift_risk'),
            dynamics.get('state_shift_risk'),
            0.0,
        ) or 0.0)
        hidden_state_phase = str(_pick(
            raw.get('hidden_state_phase'),
            hidden_state.get('phase'),
            '',
        ) or '')
        hidden_phase_confidence = max(0.0, min(1.0, _as_float(_pick(
            raw.get('hidden_phase_confidence'),
            hidden_state.get('phase_confidence'),
            0.0,
        ))))
        hidden_state_depth = max(0, _as_int(_pick(
            raw.get('hidden_state_depth'),
            hidden_state.get('hidden_state_depth'),
            0,
        )))
        hidden_drift_score = max(0.0, min(1.0, _as_float(_pick(
            raw.get('hidden_drift_score'),
            hidden_state.get('drift_score'),
            0.0,
        ))))
        hidden_uncertainty_score = max(0.0, min(1.0, _as_float(_pick(
            raw.get('hidden_uncertainty_score'),
            hidden_state.get('uncertainty_score'),
            0.0,
        ))))
        hidden_focus_functions = _as_list(_pick(
            raw.get('hidden_focus_functions'),
            hidden_state.get('focus_functions'),
            [],
        ))
        hidden_latent_signature = str(_pick(
            raw.get('hidden_latent_signature'),
            hidden_state.get('latent_signature'),
            '',
        ) or '')
        expected_next_phase = str(_pick(
            raw.get('expected_next_phase'),
            summary.get('expected_next_phase'),
            summary_hints.get('expected_next_phase'),
            hidden_state.get('expected_next_phase'),
            hidden_transition_memory.get('expected_next_phase'),
            dynamics.get('expected_next_phase'),
            '',
        ) or '')
        expected_next_phase_confidence = max(0.0, min(1.0, _as_float(_pick(
            raw.get('expected_next_phase_confidence'),
            summary.get('expected_next_phase_confidence'),
            summary_hints.get('expected_next_phase_confidence'),
            hidden_state.get('expected_next_phase_confidence'),
            hidden_transition_memory.get('expected_next_phase_confidence'),
            dynamics.get('expected_next_phase_confidence'),
            0.0,
        ))))
        phase_transition_entropy = max(0.0, min(1.0, _as_float(_pick(
            raw.get('phase_transition_entropy'),
            summary.get('phase_transition_entropy'),
            summary_hints.get('phase_transition_entropy'),
            hidden_state.get('transition_entropy'),
            hidden_transition_memory.get('phase_transition_entropy'),
            dynamics.get('phase_transition_entropy'),
            1.0,
        ), 1.0)))
        stabilizing_focus_functions = _as_function_names(_pick(
            raw.get('stabilizing_functions'),
            summary.get('stabilizing_functions'),
            summary_hints.get('stabilizing_functions'),
            hidden_transition_memory.get('stabilizing_functions'),
            dynamics.get('stabilizing_functions'),
            [],
        ))
        risky_focus_functions = _as_function_names(_pick(
            raw.get('risky_functions'),
            summary.get('risky_functions'),
            summary_hints.get('risky_functions'),
            hidden_transition_memory.get('risky_functions'),
            dynamics.get('risky_functions'),
            [],
        ))

        observed_modality = str(_pick(
            raw.get('observed_modality'),
            summary.get('observed_modality'),
            summary_hints.get('observed_modality'),
            hidden_state.get('observed_modality'),
            hidden_transition_memory.get('observed_modality'),
            dynamics.get('observed_modality'),
            '',
        ) or '')
        world_state_signature = str(_pick(
            raw.get('world_state_signature'),
            summary.get('world_state_signature'),
            summary_hints.get('world_state_signature'),
            hidden_state.get('world_state_signature'),
            hidden_transition_memory.get('world_state_signature'),
            dynamics.get('world_state_signature'),
            '',
        ) or '')
        world_entity_count = max(0, _as_int(_pick(
            raw.get('world_entity_count'),
            summary.get('world_entity_count'),
            summary_hints.get('world_entity_count'),
            hidden_state.get('world_entity_count'),
            hidden_transition_memory.get('world_entity_count'),
            dynamics.get('world_entity_count'),
            0,
        )))
        world_relation_count = max(0, _as_int(_pick(
            raw.get('world_relation_count'),
            summary.get('world_relation_count'),
            summary_hints.get('world_relation_count'),
            hidden_state.get('world_relation_count'),
            hidden_transition_memory.get('world_relation_count'),
            dynamics.get('world_relation_count'),
            0,
        )))
        last_event_signature = str(_pick(
            raw.get('last_event_signature'),
            summary.get('last_event_signature'),
            summary_hints.get('last_event_signature'),
            hidden_state.get('last_event_signature'),
            hidden_transition_memory.get('last_event_signature'),
            dynamics.get('last_event_signature'),
            '',
        ) or '')
        world_novelty_score = _clamp01(_pick(
            raw.get('world_novelty_score'),
            summary.get('world_novelty_score'),
            summary_hints.get('world_novelty_score'),
            hidden_state.get('novelty_score'),
            hidden_transition_memory.get('novelty_score'),
            dynamics.get('world_novelty_score'),
            0.0,
        ), 0.0)
        predicted_transitions = _as_rows(_pick(
            raw.get('predicted_transitions'),
            summary.get('predicted_transitions'),
            summary_hints.get('predicted_transitions'),
            [],
        ))
        mechanism_hypotheses = _as_rows(_pick(
            raw.get('mechanism_hypotheses'),
            summary.get('mechanism_hypotheses'),
            summary_hints.get('mechanism_hypotheses'),
            [],
        ))
        discriminating_tests = _as_rows(_pick(
            raw.get('discriminating_tests'),
            summary.get('discriminating_tests'),
            summary_hints.get('discriminating_tests'),
            [],
        ))
        candidate_intervention_targets = _as_rows(_pick(
            raw.get('candidate_intervention_targets'),
            summary.get('candidate_intervention_targets'),
            summary_hints.get('candidate_intervention_targets'),
            [],
        ))
        counterfactual_contrasts = _as_rows(_pick(
            raw.get('counterfactual_contrasts'),
            summary.get('counterfactual_contrasts'),
            summary_hints.get('counterfactual_contrasts'),
            [],
        ))
        expected_information_gain = _clamp01(_pick(
            raw.get('expected_information_gain'),
            summary.get('expected_information_gain'),
            summary_hints.get('expected_information_gain'),
            0.0,
        ), 0.0)
        rollout_uncertainty = _clamp01(_pick(
            raw.get('rollout_uncertainty'),
            summary.get('rollout_uncertainty'),
            summary_hints.get('rollout_uncertainty'),
            0.5,
        ), 0.5)

        prediction_trust_score = _clamp01(_pick(
            raw.get('prediction_trust_score'),
            summary.get('prediction_trust_score'),
            summary_hints.get('prediction_trust_score'),
            _prediction_trust_score(context.get('predictor_trust', {})),
            0.5,
        ), 0.5)
        recent_prediction_error = _clamp01(_pick(
            raw.get('recent_prediction_error'),
            summary.get('recent_prediction_error'),
            summary_hints.get('recent_prediction_error'),
            _recent_prediction_error(context.get('prediction_error_tail', [])),
            0.0,
        ), 0.0)

        derived_control_trust = _clamp01(
            (transition_confidence * 0.34)
            + (hidden_phase_confidence * 0.18)
            + (prediction_trust_score * 0.22)
            + ((1.0 - recent_prediction_error) * 0.10)
            + (max((float(item.get('confidence', 0.0) or 0.0) for item in latent_branches), default=0.0) * 0.06)
            + (min(0.08, hidden_state_depth * 0.02) if hidden_state_phase in {'stabilizing', 'committed'} else 0.0)
            + (world_novelty_score * 0.06)
            - (hidden_drift_score * 0.08 if hidden_state_phase in {'exploring', 'disrupted'} else 0.0)
            - (abs(transition_confidence - hidden_phase_confidence) * 0.08),
            0.5,
        )
        control_trust = _clamp01(_pick(
            raw.get('control_trust'),
            summary.get('control_trust'),
            summary_hints.get('control_trust'),
            derived_control_trust,
            0.5,
        ), 0.5)

        return cls(
            blocked_functions=blocked,
            preferred_action_classes=preferred_action_classes,
            hard_constraints=hard_constraints,
            predicted_phase=predicted_phase,
            transition_confidence=max(0.0, min(1.0, transition_confidence)),
            state_shift_risk=max(0.0, min(1.0, shift_risk)),
            required_probes=required_probes,
            hidden_state_phase=hidden_state_phase,
            hidden_phase_confidence=hidden_phase_confidence,
            hidden_state_depth=hidden_state_depth,
            hidden_drift_score=hidden_drift_score,
            hidden_uncertainty_score=hidden_uncertainty_score,
            hidden_focus_functions=hidden_focus_functions,
            hidden_latent_signature=hidden_latent_signature,
            dominant_branch_id=dominant_branch_id,
            latent_branches=latent_branches,
            expected_next_phase=expected_next_phase,
            expected_next_phase_confidence=expected_next_phase_confidence,
            phase_transition_entropy=phase_transition_entropy,
            stabilizing_focus_functions=stabilizing_focus_functions,
            risky_focus_functions=risky_focus_functions,
            observed_modality=observed_modality,
            world_state_signature=world_state_signature,
            world_entity_count=world_entity_count,
            world_relation_count=world_relation_count,
            last_event_signature=last_event_signature,
            world_novelty_score=world_novelty_score,
            predicted_transitions=predicted_transitions,
            mechanism_hypotheses=mechanism_hypotheses,
            discriminating_tests=discriminating_tests,
            candidate_intervention_targets=candidate_intervention_targets,
            counterfactual_contrasts=counterfactual_contrasts,
            expected_information_gain=expected_information_gain,
            rollout_uncertainty=rollout_uncertainty,
            prediction_trust_score=prediction_trust_score,
            recent_prediction_error=recent_prediction_error,
            control_trust=control_trust,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'blocked_functions': list(self.blocked_functions),
            'preferred_action_classes': list(self.preferred_action_classes),
            'hard_constraints': list(self.hard_constraints),
            'predicted_phase': self.predicted_phase,
            'transition_confidence': self.transition_confidence,
            'state_shift_risk': self.state_shift_risk,
            'required_probes': list(self.required_probes),
            'hidden_state_phase': self.hidden_state_phase,
            'hidden_phase_confidence': self.hidden_phase_confidence,
            'hidden_state_depth': self.hidden_state_depth,
            'hidden_drift_score': self.hidden_drift_score,
            'hidden_uncertainty_score': self.hidden_uncertainty_score,
            'hidden_focus_functions': list(self.hidden_focus_functions),
            'hidden_latent_signature': self.hidden_latent_signature,
            'dominant_branch_id': self.dominant_branch_id,
            'latent_branches': [dict(item) for item in self.latent_branches if isinstance(item, dict)],
            'expected_next_phase': self.expected_next_phase,
            'expected_next_phase_confidence': self.expected_next_phase_confidence,
            'phase_transition_entropy': self.phase_transition_entropy,
            'stabilizing_focus_functions': list(self.stabilizing_focus_functions),
            'risky_focus_functions': list(self.risky_focus_functions),
            'observed_modality': self.observed_modality,
            'world_state_signature': self.world_state_signature,
            'world_entity_count': self.world_entity_count,
            'world_relation_count': self.world_relation_count,
            'last_event_signature': self.last_event_signature,
            'world_novelty_score': self.world_novelty_score,
            'predicted_transitions': [dict(item) for item in self.predicted_transitions if isinstance(item, dict)],
            'mechanism_hypotheses': [dict(item) for item in self.mechanism_hypotheses if isinstance(item, dict)],
            'discriminating_tests': [dict(item) for item in self.discriminating_tests if isinstance(item, dict)],
            'candidate_intervention_targets': [dict(item) for item in self.candidate_intervention_targets if isinstance(item, dict)],
            'counterfactual_contrasts': [dict(item) for item in self.counterfactual_contrasts if isinstance(item, dict)],
            'expected_information_gain': self.expected_information_gain,
            'rollout_uncertainty': self.rollout_uncertainty,
            'prediction_trust_score': self.prediction_trust_score,
            'recent_prediction_error': self.recent_prediction_error,
            'control_trust': self.control_trust,
        }

"""
decision/value_model.py

Value scoring model.
"""

from __future__ import annotations
from typing import Any, Dict

from decision.utility_schema import DecisionCandidate, ValueScore, CandidateSource
from decision.value_alignment import ValueAlignmentPolicy


class ValueModel:
    def __init__(self):
        self._alignment_policy = ValueAlignmentPolicy()
        self._known_functions = {
            'compute_stats': 0.6,
            'filter_by_predicate': 0.7,
            'join_tables': 0.8,
            'aggregate_group': 0.75,
            'array_transform': 0.65,
        }

    def _perception_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        summary = context.get('perception_summary', {})
        return summary if isinstance(summary, dict) else {}

    def _world_model_beliefs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        summary = context.get('world_model_summary', {})
        if not isinstance(summary, dict):
            return {}
        beliefs = summary.get('beliefs', {})
        return beliefs if isinstance(beliefs, dict) else {}

    def _world_model_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        summary = context.get('world_model_summary', {})
        return summary if isinstance(summary, dict) else {}

    def _unified_context_payload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        uc = context.get('unified_context') or context.get('unified_cognitive_context')
        return dict(uc) if isinstance(uc, dict) else {}

    def _task_frame_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        uc = self._unified_context_payload(context)
        summary = uc.get('task_frame_summary', context.get('task_frame_summary', {}))
        return dict(summary) if isinstance(summary, dict) else {}

    def _object_bindings_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        uc = self._unified_context_payload(context)
        summary = uc.get('object_bindings_summary', context.get('object_bindings_summary', {}))
        return dict(summary) if isinstance(summary, dict) else {}

    def _goal_hypotheses_summary(self, context: Dict[str, Any]) -> list[dict]:
        uc = self._unified_context_payload(context)
        hypotheses = uc.get('goal_hypotheses_summary', context.get('goal_hypotheses_summary', []))
        return [dict(item) for item in hypotheses if isinstance(item, dict)] if isinstance(hypotheses, list) else []

    def _solver_state_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        uc = self._unified_context_payload(context)
        summary = uc.get('solver_state_summary', context.get('solver_state_summary', {}))
        return dict(summary) if isinstance(summary, dict) else {}

    def _mechanism_hypotheses_summary(self, context: Dict[str, Any]) -> list[dict]:
        uc = self._unified_context_payload(context)
        mechanisms = uc.get('mechanism_hypotheses_summary', context.get('mechanism_hypotheses_summary', []))
        return [dict(item) for item in mechanisms if isinstance(item, dict)] if isinstance(mechanisms, list) else []

    def _mechanism_control_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        uc = self._unified_context_payload(context)
        summary = uc.get('mechanism_control_summary', context.get('mechanism_control_summary', {}))
        return dict(summary) if isinstance(summary, dict) else {}

    def _candidate_action_family(self, candidate: DecisionCandidate) -> str:
        fn = str(candidate.function_name or '').upper()
        if fn in {'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'}:
            return 'navigation_interaction'
        if fn == 'ACTION5' or fn in {'CONFIRM', 'INTERACT', 'SUBMIT'}:
            return 'confirm_interaction'
        if fn == 'ACTION6' or fn in {'CLICK', 'POINTER_CLICK', 'TAP'}:
            return 'pointer_interaction'
        if candidate.is_wait:
            return 'wait'
        return 'state_transform_interaction'

    def _solver_guidance_for_candidate(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> Dict[str, Any]:
        task_frame = self._task_frame_summary(context)
        object_bindings = self._object_bindings_summary(context)
        goal_hypotheses = self._goal_hypotheses_summary(context)
        solver_state = self._solver_state_summary(context)
        meta = self._candidate_meta(candidate)
        intervention_meta = self._intervention_meta(candidate)
        action_family = self._candidate_action_family(candidate)
        dominant_mode = str(task_frame.get('dominant_interaction_mode', '') or '')
        anchor_ref = ''
        intervention_target = meta.get('intervention_target', {}) if isinstance(meta.get('intervention_target', {}), dict) else {}
        if intervention_target:
            anchor_ref = str(intervention_target.get('anchor_ref', '') or '')
        if not anchor_ref:
            anchor_ref = str(meta.get('anchor_ref', '') or '')
        preferred_target_refs = [str(x or '') for x in list(solver_state.get('preferred_target_refs', []) or []) if str(x or '')]
        preferred_target_match = bool(anchor_ref and anchor_ref in preferred_target_refs)
        preferred_action_families: list[str] = []
        hypothesis_match_conf = 0.0
        for hypo in goal_hypotheses:
            target_refs = [str(x or '') for x in list(hypo.get('preferred_target_refs', []) or []) if str(x or '')]
            action_families = [str(x or '') for x in list(hypo.get('preferred_action_families', []) or []) if str(x or '')]
            if preferred_target_match and anchor_ref in target_refs:
                hypothesis_match_conf = max(hypothesis_match_conf, float(hypo.get('confidence', 0.0) or 0.0))
                preferred_action_families.extend(action_families)
            elif action_family and action_family in action_families:
                hypothesis_match_conf = max(hypothesis_match_conf, float(hypo.get('confidence', 0.0) or 0.0) * 0.75)
                preferred_action_families.extend(action_families)
        mode_match = bool(dominant_mode and action_family == dominant_mode)
        bound_object = {}
        for obj in list(object_bindings.get('objects', []) or []):
            if isinstance(obj, dict) and str(obj.get('object_id', '') or '') == anchor_ref:
                bound_object = obj
                break
        semantic_labels = [str(item.get('label', '') or '') for item in list(bound_object.get('semantic_candidates', []) or []) if isinstance(item, dict) and str(item.get('label', '') or '')]
        role_labels = [str(item.get('role', '') or '') for item in list(bound_object.get('role_candidates', []) or []) if isinstance(item, dict) and str(item.get('role', '') or '')]
        unexplored_anchor = bool(anchor_ref) and int(intervention_meta.get('interaction_count', 0) or 0) == 0
        semantic_bonus = 0.0
        if 'directional_like' in semantic_labels or 'token_like' in semantic_labels:
            semantic_bonus += 0.12
        if 'hint_or_marker' in role_labels or 'interactive_token' in role_labels:
            semantic_bonus += 0.10
        if dominant_mode == 'pointer_interaction' and semantic_labels:
            semantic_bonus += 0.08
        solver_value_signal = min(1.0, max(0.0,
            (0.28 if mode_match else 0.0)
            + (0.24 if preferred_target_match else 0.0)
            + (0.30 * min(1.0, hypothesis_match_conf))
            + semantic_bonus
            + (0.10 if unexplored_anchor else 0.0)
            - (0.20 * float(intervention_meta.get('repeat_penalty', 0.0) or 0.0))
        ))
        guidance = {
            'action_family': action_family,
            'dominant_mode': dominant_mode,
            'mode_match': mode_match,
            'anchor_ref': anchor_ref,
            'preferred_target_match': preferred_target_match,
            'hypothesis_match_confidence': round(min(1.0, hypothesis_match_conf), 4),
            'semantic_labels': semantic_labels[:4],
            'role_labels': role_labels[:4],
            'unexplored_anchor': unexplored_anchor,
            'solver_value_signal': round(solver_value_signal, 4),
            'dominant_goal_family': str(solver_state.get('dominant_goal_family', '') or ''),
            'dominant_goal_confidence': float(solver_state.get('dominant_goal_confidence', 0.0) or 0.0),
        }
        if isinstance(candidate.action, dict):
            merged = dict(meta)
            merged['solver_guidance'] = guidance
            candidate.action['_candidate_meta'] = merged
        return guidance


    def _mechanism_guidance_for_candidate(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> Dict[str, Any]:
        mechanisms = self._mechanism_hypotheses_summary(context)
        mechanism_control = self._mechanism_control_summary(context)
        meta = self._candidate_meta(candidate)
        action_family = self._candidate_action_family(candidate)
        anchor_ref = ''
        intervention_target = meta.get('intervention_target', {}) if isinstance(meta.get('intervention_target', {}), dict) else {}
        if intervention_target:
            anchor_ref = str(intervention_target.get('anchor_ref', '') or '')
        if not anchor_ref:
            anchor_ref = str(meta.get('anchor_ref', '') or '')
        preferred_refs = [str(x or '') for x in list(mechanism_control.get('preferred_target_refs', []) or []) if str(x or '')]
        control_families = [str(x or '') for x in list(mechanism_control.get('preferred_action_families', []) or []) if str(x or '')]
        discriminating_actions = [str(x or '') for x in list(mechanism_control.get('discriminating_actions', []) or []) if str(x or '')]
        best_support = 0.0
        best_family = str(mechanism_control.get('dominant_mechanism_family', '') or '')
        best_confidence = float(mechanism_control.get('dominant_mechanism_confidence', 0.0) or 0.0)
        target_match = bool(anchor_ref and anchor_ref in preferred_refs)
        for mechanism in mechanisms:
            target_refs = [str(x or '') for x in list(mechanism.get('preferred_target_refs', []) or []) if str(x or '')]
            action_families = [str(x or '') for x in list(mechanism.get('preferred_action_families', []) or []) if str(x or '')]
            confidence = float(mechanism.get('confidence', 0.0) or 0.0)
            mechanism_target_match = bool(anchor_ref and anchor_ref in target_refs)
            support = confidence * (1.0 if mechanism_target_match else (0.74 if action_family in action_families else 0.42))
            if support > best_support:
                best_support = support
                best_family = str(mechanism.get('family', '') or best_family)
                best_confidence = confidence
                target_match = mechanism_target_match or target_match
            for item in action_families:
                if item and item not in control_families:
                    control_families.append(item)
            for item in list(mechanism.get('best_discriminating_actions', []) or []):
                item = str(item or '')
                if item and item not in discriminating_actions:
                    discriminating_actions.append(item)
        action_family_match = bool(action_family and action_family in control_families)
        mechanism_signal = self._clamp(
            (0.34 if target_match else 0.0)
            + (0.26 if action_family_match else 0.0)
            + (0.26 * self._clamp(best_confidence))
            + (0.14 if action_family and action_family in discriminating_actions else 0.0)
        )
        guidance = {
            'mechanism_family': best_family,
            'mechanism_confidence': round(self._clamp(best_confidence), 4),
            'mechanism_signal': round(mechanism_signal, 4),
            'mechanism_target_match': bool(target_match),
            'mechanism_action_family_match': bool(action_family_match),
            'mechanism_discriminating_actions': discriminating_actions[:4],
        }
        if isinstance(candidate.action, dict):
            merged = dict(meta)
            merged['mechanism_guidance'] = guidance
            candidate.action['_candidate_meta'] = merged
        return guidance

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _candidate_meta(self, candidate: DecisionCandidate) -> Dict[str, Any]:
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        return meta if isinstance(meta, dict) else {}

    def _intervention_meta(self, candidate: DecisionCandidate) -> Dict[str, Any]:
        meta = self._candidate_meta(candidate)
        if not meta:
            return {}
        if 'target_value_score' in meta:
            return meta
        intervention = meta.get('intervention', {})
        if isinstance(intervention, dict):
            merged = dict(intervention)
            for key in (
                'target_value_score',
                'expected_state_change',
                'expected_progress',
                'expected_information_gain',
                'repeat_penalty',
                'exploration_bonus',
            ):
                if key in meta and key not in merged:
                    merged[key] = meta[key]
            return merged
        return meta

    def _score_intervention_value(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        meta = self._intervention_meta(candidate)
        guidance = self._solver_guidance_for_candidate(candidate, context)
        mechanism_guidance = self._mechanism_guidance_for_candidate(candidate, context)
        if not meta and not guidance.get('solver_value_signal'):
            return 0.0
        direct = meta.get('target_value_score')
        if isinstance(direct, (int, float)):
            base = float(direct)
        else:
            expected_state_change = float(meta.get('expected_state_change', 0.0) or 0.0)
            expected_progress = float(meta.get('expected_progress', 0.0) or 0.0)
            expected_information_gain = float(meta.get('expected_information_gain', 0.0) or 0.0)
            repeat_penalty = float(meta.get('repeat_penalty', 0.0) or 0.0)
            exploration_bonus = float(meta.get('exploration_bonus', 0.0) or 0.0)
            base = (
                0.34 * expected_state_change
                + 0.30 * expected_progress
                + 0.16 * expected_information_gain
                + 0.10 * exploration_bonus
                - 0.28 * repeat_penalty
            )
        base += 0.24 * float(guidance.get('solver_value_signal', 0.0) or 0.0)
        base += 0.18 * float(mechanism_guidance.get('mechanism_signal', 0.0) or 0.0)
        if candidate.source == CandidateSource.INTERVENTION:
            base += 0.05
        return self._clamp(base)


    def _score_mechanism_value(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        guidance = self._mechanism_guidance_for_candidate(candidate, context)
        solver_guidance = self._solver_guidance_for_candidate(candidate, context)
        intervention_meta = self._intervention_meta(candidate)
        meta = self._candidate_meta(candidate)
        mechanism_control = self._mechanism_control_summary(context)
        control_mode = str(mechanism_control.get('control_mode', '') or '')
        role = str(meta.get('role', '') or '').strip().lower()
        mode_alignment = float(meta.get('mechanism_mode_alignment', 0.0) or 0.0)
        binding_score = float(meta.get('mechanism_binding_score', 0.0) or 0.0)
        binding_margin = float(meta.get('mechanism_binding_margin', 0.0) or 0.0)
        binding_specificity = float(meta.get('mechanism_binding_specificity', 0.0) or 0.0)
        evidence_strength = float(meta.get('mechanism_binding_evidence_strength', 0.0) or 0.0)
        contradiction_penalty = float(meta.get('mechanism_contradiction_penalty', 0.0) or 0.0)
        actionable = bool(meta.get('mechanism_binding_actionable', False))
        discriminating_candidate = bool(meta.get('runtime_discriminating_candidate', False) or candidate.is_probe)
        release_ready = bool(meta.get('mechanism_release_ready', False))
        wait_ready = bool(meta.get('mechanism_wait_ready', False))
        prerequisite_ready = bool(meta.get('mechanism_prerequisite_ready', False))
        recovery_ready = bool(meta.get('mechanism_recovery_ready', False))
        score = 0.0
        score += 0.46 * float(guidance.get('mechanism_signal', 0.0) or 0.0)
        score += 0.18 * float(solver_guidance.get('solver_value_signal', 0.0) or 0.0)
        if guidance.get('mechanism_target_match'):
            score += 0.14
        if guidance.get('mechanism_action_family_match'):
            score += 0.10
        score += 0.06 * float(intervention_meta.get('expected_information_gain', 0.0) or 0.0)
        score -= 0.12 * float(intervention_meta.get('repeat_penalty', 0.0) or 0.0)
        if control_mode == 'exploit':
            if actionable:
                score = max(
                    score,
                    0.64
                    + (0.18 * self._clamp(binding_score))
                    + (0.14 * self._clamp(binding_margin + binding_specificity))
                    + (0.08 * self._clamp(evidence_strength)),
                )
            elif role == 'commit' and contradiction_penalty < 0.35:
                score = max(score, 0.42 + (0.16 * self._clamp(binding_score)) + (0.08 * self._clamp(binding_specificity)))
        elif control_mode == 'discriminate':
            if discriminating_candidate:
                score = max(
                    score,
                    0.52
                    + (0.12 * self._clamp(mode_alignment))
                    + (0.10 * float(intervention_meta.get('expected_information_gain', 0.0) or 0.0))
                    + (0.10 * self._clamp(binding_specificity)),
                )
            elif role == 'commit' and not actionable:
                score = min(score, 0.34 + (0.10 * self._clamp(binding_score)))
        elif control_mode == 'wait' and candidate.is_wait:
            score = max(score, 0.78)
        elif control_mode == 'recover' and role == 'recovery':
            score = max(score, 0.76)
        elif control_mode == 'prepare' and role in {'prerequisite', 'prepare'}:
            score = max(score, 0.74)
        if wait_ready and candidate.is_wait:
            score = max(score, 0.82)
        if prerequisite_ready and role in {'prerequisite', 'prepare'}:
            score = max(score, 0.82)
        if recovery_ready and role == 'recovery':
            score = max(score, 0.82)
        if role == 'commit' and not release_ready and not actionable:
            score = min(score, 0.18)
        if contradiction_penalty >= 0.45 and role == 'commit':
            score = min(score, 0.12)
        score += 0.10 * self._clamp(mode_alignment)
        return self._clamp(score)

    def _wait_context_profile(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._candidate_meta(candidate)
        world_model_summary = self._world_model_summary(context)
        plan_summary = context.get('plan_summary', {}) if isinstance(context.get('plan_summary', {}), dict) else {}
        state_dynamics = world_model_summary.get('state_dynamics', {}) if isinstance(world_model_summary.get('state_dynamics', {}), dict) else {}

        tick = int(context.get('tick', 0) or 0)
        recent_failures = int(context.get('recent_failures', 0) or 0)
        reward_trend = str(context.get('reward_trend', 'neutral') or 'neutral').lower()
        predicted_phase = str(world_model_summary.get('predicted_phase', '') or '').lower()

        required_probes = [str(item) for item in list(world_model_summary.get('required_probes', []) or []) if str(item or '')]
        preferred_action_classes = [str(item) for item in list(world_model_summary.get('preferred_action_classes', []) or []) if str(item or '')]
        hard_constraints = [str(item) for item in list(world_model_summary.get('hard_constraints', []) or []) if str(item or '')]

        explicit_wait_justification = any(bool(meta.get(key)) for key in (
            'wait_justified', 'pending_delayed_effect', 'cooldown_wait', 'time_dependent_wait'
        ))
        planner_requested_wait = bool(meta.get('planner_matches_step')) and str(meta.get('planner_step_intent', '') or '').lower() == 'wait'
        no_function_surface = bool(meta.get('no_function_surface'))
        filtered_invalid = bool(meta.get('filtered_invalid_call_candidate'))
        injected_no_viable = str(meta.get('wait_injection_reason', '') or '') == 'no_viable_non_wait'
        trace_count = int(state_dynamics.get('trace_count', 0) or 0)

        early_exploration = tick <= 2 and recent_failures <= 0
        exploration_pressure = early_exploration and (
            bool(required_probes) or bool(preferred_action_classes) or bool(plan_summary.get('has_plan')) or
            predicted_phase in {'exploring', 'stabilizing'} or trace_count == 0
        )

        unjustified_wait = not (explicit_wait_justification or planner_requested_wait or no_function_surface)
        suppress_wait = exploration_pressure and unjustified_wait and reward_trend != 'positive'
        soft_penalty = (not suppress_wait and unjustified_wait and (early_exploration or filtered_invalid or injected_no_viable))

        profile = {
            'tick': tick,
            'recent_failures': recent_failures,
            'reward_trend': reward_trend,
            'predicted_phase': predicted_phase,
            'required_probes': required_probes[:4],
            'preferred_action_classes': preferred_action_classes[:4],
            'hard_constraints': hard_constraints[:4],
            'planner_requested_wait': planner_requested_wait,
            'explicit_wait_justification': explicit_wait_justification,
            'no_function_surface': no_function_surface,
            'filtered_invalid_call_candidate': filtered_invalid,
            'wait_injection_reason': str(meta.get('wait_injection_reason', '') or ''),
            'suppress_wait': suppress_wait,
            'soft_penalty': soft_penalty,
        }

        if isinstance(candidate.action, dict):
            merged_meta = dict(meta)
            merged_meta['wait_gate_profile'] = profile
            candidate.action['_candidate_meta'] = merged_meta

        return profile

    def _adaptive_weights(self, context: Dict[str, Any], candidate: DecisionCandidate) -> Dict[str, float]:
        weights = {
            'utility': 0.4,
            'novelty': 0.2,
            'confidence': 0.2,
            'goal_alignment': 0.2,
            'intervention': 0.0,
            'mechanism': 0.0,
        }
        reward_trend = str(context.get('reward_trend', 'neutral') or 'neutral').lower()
        if reward_trend == 'negative':
            weights['confidence'] += 0.08
            weights['goal_alignment'] += 0.05
            weights['novelty'] -= 0.07
            weights['utility'] -= 0.06
        elif reward_trend == 'positive':
            weights['utility'] += 0.05
            weights['novelty'] += 0.04
            weights['confidence'] -= 0.05
            weights['goal_alignment'] -= 0.04

        uncertainty_signals = context.get('uncertainty_signals', {})
        if isinstance(uncertainty_signals, dict):
            epistemic = float(uncertainty_signals.get('epistemic', 0.0) or 0.0)
            if epistemic > 0.6:
                weights['confidence'] += 0.05
                weights['utility'] -= 0.03

        meta = self._candidate_meta(candidate)
        if meta.get('counterfactual_advantage'):
            weights['utility'] += 0.04
            weights['confidence'] += 0.03
            weights['novelty'] -= 0.03

        intervention_meta = self._intervention_meta(candidate)
        if candidate.source == CandidateSource.INTERVENTION or intervention_meta.get('target_value_score') is not None:
            weights['intervention'] = 0.24
            weights['utility'] = max(0.12, weights['utility'] - 0.10)
            weights['novelty'] = max(0.08, weights['novelty'] - 0.04)
            weights['confidence'] = max(0.10, weights['confidence'] - 0.04)
            weights['goal_alignment'] = max(0.10, weights['goal_alignment'] - 0.10)

        mechanism_value = self._score_mechanism_value(candidate, context)
        if mechanism_value > 0.0:
            weights['mechanism'] = 0.22 if mechanism_value >= 0.35 else 0.14
            weights['utility'] = max(0.10, weights['utility'] - 0.06)
            weights['novelty'] = max(0.08, weights['novelty'] - 0.02)
        for key in list(weights.keys()):
            weights[key] = max(0.0 if key in {'intervention', 'mechanism'} else 0.05, float(weights[key]))
        return weights

    def score(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> ValueScore:
        utility = self._score_utility(candidate, context)
        novelty = self._score_novelty(candidate, context)
        confidence = self._score_confidence(candidate, context)
        goal_alignment = self._score_goal_alignment(candidate, context)
        intervention_value = self._score_intervention_value(candidate, context)
        mechanism_value = self._score_mechanism_value(candidate, context)
        return ValueScore(
            utility=utility,
            novelty=novelty,
            confidence=confidence,
            goal_alignment=goal_alignment,
            intervention_value=intervention_value,
            mechanism_value=mechanism_value,
            weights=self._adaptive_weights(context, candidate),
        )

    def _score_utility(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        meta = self._candidate_meta(candidate)
        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                return 0.02
            if wait_profile['soft_penalty']:
                return 0.08
            return 0.15

        fn = candidate.function_name
        if fn in self._known_functions:
            utility = self._known_functions[fn]
        else:
            utility = 0.5

        perception = self._perception_summary(context)
        beliefs = self._world_model_beliefs(context)

        if candidate.source == CandidateSource.PLANNER and meta.get('planner_matches_step'):
            utility = min(1.0, utility + 0.12)
        if candidate.source == CandidateSource.SELF_MODEL:
            utility = min(1.0, utility + float(meta.get('capability_score', 0.0)) * 0.25)
        if candidate.source == CandidateSource.HISTORY_REUSE:
            utility = min(1.0, utility + max(0.0, float(meta.get('history_reward', 0.0))) * 0.1)
        if candidate.source == CandidateSource.INTERVENTION:
            utility = min(1.0, utility + self._score_intervention_value(candidate, context) * 0.20 + self._score_mechanism_value(candidate, context) * 0.16)
        if candidate.source in (CandidateSource.PLANNER, CandidateSource.RETRIEVAL) and len(perception.get('dynamic_entities', [])) > 0:
            utility = min(1.0, utility + 0.08)
        if candidate.source == CandidateSource.SELF_MODEL and perception.get('coordinate_type') == 'camera_relative' and float(perception.get('coordinate_confidence', 0.0) or 0.0) >= 0.55:
            utility = min(1.0, utility + 0.12)
        if beliefs.get('observation_camera_motion', {}).get('posterior') == 'high_motion' and candidate.source == CandidateSource.SELF_MODEL:
            utility = min(1.0, utility + 0.08)
        if perception.get('color_remapping_detected') and candidate.source in (CandidateSource.HISTORY_REUSE, CandidateSource.RETRIEVAL):
            utility = max(0.0, utility - 0.08)

        cf_conf = str(meta.get('counterfactual_confidence', '')).lower()
        if meta.get('counterfactual_advantage'):
            if cf_conf == 'high':
                utility = min(1.0, utility + 0.08)
            elif cf_conf == 'medium':
                utility = min(1.0, utility + 0.04)
        elif cf_conf == 'low' and candidate.source in (CandidateSource.PLANNER, CandidateSource.SELF_MODEL):
            utility = max(0.0, utility - 0.03)
        learned_dynamics_bonus = float(meta.get('learned_dynamics_governance_bonus', 0.0) or 0.0)
        if bool(meta.get('learned_dynamics_routing_active', False)) and learned_dynamics_bonus != 0.0:
            utility = self._clamp(utility + learned_dynamics_bonus)

        guidance = self._solver_guidance_for_candidate(candidate, context)
        mechanism_guidance = self._mechanism_guidance_for_candidate(candidate, context)
        utility = utility + 0.18 * float(guidance.get('solver_value_signal', 0.0) or 0.0)
        if guidance.get('preferred_target_match'):
            utility += 0.08
        if guidance.get('mode_match'):
            utility += 0.06
        return self._clamp(utility)

    def _score_novelty(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        meta = self._candidate_meta(candidate)

        if candidate.source in (CandidateSource.SKILL_REWRITE, CandidateSource.LLM_REWRITE):
            novelty = 0.7
        elif candidate.source == CandidateSource.ARM_EVALUATION:
            novelty = 0.6
        elif candidate.source == CandidateSource.PLANNER:
            novelty = 0.55
        elif candidate.source == CandidateSource.SELF_MODEL:
            novelty = 0.45
        elif candidate.source == CandidateSource.HISTORY_REUSE:
            novelty = 0.35
        elif candidate.source == CandidateSource.RECOVERY:
            novelty = 0.4
        elif candidate.source == CandidateSource.INTERVENTION:
            novelty = 0.68
        elif candidate.source == CandidateSource.WAIT_FALLBACK:
            novelty = 0.1
        else:
            novelty = 0.5

        if candidate.surfaced_from:
            novelty = min(1.0, novelty + 0.1 * len(candidate.surfaced_from))
        if meta.get('recent_negative_feedback') and candidate.source == CandidateSource.SELF_MODEL:
            novelty = min(1.0, novelty + 0.05)

        intervention_meta = self._intervention_meta(candidate)
        exploration_bonus = float(intervention_meta.get('exploration_bonus', 0.0) or 0.0)
        repeat_penalty = float(intervention_meta.get('repeat_penalty', 0.0) or 0.0)
        novelty = novelty + (0.10 * exploration_bonus) - (0.10 * repeat_penalty)

        guidance = self._solver_guidance_for_candidate(candidate, context)
        mechanism_guidance = self._mechanism_guidance_for_candidate(candidate, context)
        if guidance.get('unexplored_anchor'):
            novelty += 0.10
        if guidance.get('preferred_target_match'):
            novelty += 0.04
        return self._clamp(novelty)

    def _score_confidence(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        if candidate.surfaced_from:
            confidence = 0.6
        else:
            confidence = 0.4

        if candidate.is_wait:
            wait_profile = self._wait_context_profile(candidate, context)
            if wait_profile['suppress_wait']:
                confidence = 0.22
            elif wait_profile['soft_penalty']:
                confidence = 0.55
            else:
                confidence = 0.9

        meta = self._candidate_meta(candidate)
        perception = self._perception_summary(context)
        beliefs = self._world_model_beliefs(context)

        if candidate.source == CandidateSource.PLANNER and meta.get('planner_matches_step'):
            confidence = max(confidence, 0.72)
        if candidate.source == CandidateSource.SELF_MODEL:
            confidence = max(confidence, float(meta.get('self_model_confidence', 0.0)))
        if candidate.source == CandidateSource.HISTORY_REUSE and float(meta.get('history_reward', 0.0)) > 0.0:
            confidence = max(confidence, 0.7)
        if candidate.source == CandidateSource.INTERVENTION:
            confidence = max(confidence, 0.62 + 0.22 * self._score_intervention_value(candidate, context) + 0.18 * self._score_mechanism_value(candidate, context))
        if perception.get('coordinate_type') == 'camera_relative' and float(perception.get('coordinate_confidence', 0.0) or 0.0) >= 0.55:
            if candidate.source in (CandidateSource.SELF_MODEL, CandidateSource.WAIT_FALLBACK):
                confidence += 0.12
            elif candidate.source == CandidateSource.HISTORY_REUSE:
                confidence -= 0.08
        if beliefs.get('observation_camera_motion', {}).get('posterior') == 'high_motion':
            if candidate.source == CandidateSource.SELF_MODEL:
                confidence += 0.08
            elif candidate.source == CandidateSource.HISTORY_REUSE:
                confidence -= 0.05
        if perception.get('color_remapping_detected') and candidate.source == CandidateSource.RETRIEVAL:
            confidence -= 0.08
        learned_dynamics_bonus = float(meta.get('learned_dynamics_governance_bonus', 0.0) or 0.0)
        if bool(meta.get('learned_dynamics_routing_active', False)) and learned_dynamics_bonus != 0.0:
            confidence += max(-0.05, min(0.05, learned_dynamics_bonus * 0.6))

        guidance = self._solver_guidance_for_candidate(candidate, context)
        mechanism_guidance = self._mechanism_guidance_for_candidate(candidate, context)
        confidence += 0.16 * float(guidance.get('hypothesis_match_confidence', 0.0) or 0.0)
        if guidance.get('mode_match'):
            confidence += 0.05
        if guidance.get('preferred_target_match'):
            confidence += 0.08
        return self._clamp(confidence)

    def _score_goal_alignment(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        alignment = 0.5
        continuity = context.get('continuity_snapshot', {})
        top_goal = continuity.get('top_goal')
        meta = self._candidate_meta(candidate)
        goal_assessment = self._alignment_policy.assess_goal_health(top_goal, context) if top_goal else None

        if top_goal:
            goal_id = goal_assessment.goal_id if goal_assessment else ''
            goal_kind = ''
            if isinstance(top_goal, dict):
                goal_kind = str(top_goal.get('goal_type') or top_goal.get('kind') or '')
            else:
                goal_kind = str(getattr(top_goal, 'goal_type', '') or getattr(top_goal, 'kind', '') or '')
            goal_signal = f"{goal_id} {goal_kind}".lower()

            if 'explore' in goal_signal:
                if candidate.function_name in ('join_tables', 'aggregate_group'):
                    alignment = 0.8
                elif candidate.function_name == 'compute_stats':
                    alignment = 0.4
            elif 'exploit' in goal_signal or 'confirm' in goal_signal:
                if candidate.function_name == 'compute_stats':
                    alignment = 0.8
                elif candidate.function_name in ('join_tables', 'aggregate_group'):
                    alignment = 0.4
            elif 'test' in goal_signal:
                alignment = 0.6

        perception = self._perception_summary(context)
        beliefs = self._world_model_beliefs(context)
        if candidate.source == CandidateSource.PLANNER and meta.get('planner_matches_step'):
            alignment = max(alignment, 0.9)
        if candidate.source == CandidateSource.SELF_MODEL and meta.get('recent_negative_feedback'):
            alignment = max(alignment, 0.7)
        if candidate.source == CandidateSource.INTERVENTION:
            alignment = max(alignment, 0.55 + 0.24 * self._score_intervention_value(candidate, context) + 0.22 * self._score_mechanism_value(candidate, context))
        if candidate.source == CandidateSource.WAIT_FALLBACK:
            wait_profile = self._wait_context_profile(candidate, context) if candidate.is_wait else None
            if wait_profile and wait_profile['suppress_wait']:
                alignment = min(alignment, 0.12)
            elif wait_profile and wait_profile['soft_penalty']:
                alignment = min(alignment, 0.24)
            else:
                alignment = min(alignment, 0.35)
        if len(perception.get('dynamic_entities', [])) > 0 and candidate.source in (CandidateSource.PLANNER, CandidateSource.RETRIEVAL):
            alignment = max(alignment, 0.72)
        if perception.get('coordinate_type') == 'camera_relative' and float(perception.get('coordinate_confidence', 0.0) or 0.0) >= 0.55 and candidate.source in (CandidateSource.SELF_MODEL, CandidateSource.WAIT_FALLBACK):
            alignment = max(alignment, 0.82)
        if beliefs.get('observation_coordinate_type', {}).get('posterior') == 'camera_relative':
            if candidate.source == CandidateSource.SELF_MODEL:
                alignment = max(alignment, 0.84)
            elif candidate.source == CandidateSource.HISTORY_REUSE:
                alignment = min(alignment, 0.45)
        learned_dynamics_bonus = float(meta.get('learned_dynamics_governance_bonus', 0.0) or 0.0)
        if bool(meta.get('learned_dynamics_routing_active', False)) and learned_dynamics_bonus != 0.0:
            alignment = self._clamp(alignment + learned_dynamics_bonus * 0.9)

        guidance = self._solver_guidance_for_candidate(candidate, context)
        mechanism_guidance = self._mechanism_guidance_for_candidate(candidate, context)
        alignment = max(alignment, 0.28 + 0.52 * float(guidance.get('solver_value_signal', 0.0) or 0.0))
        if guidance.get('preferred_target_match'):
            alignment = max(alignment, 0.72)
        if guidance.get('mode_match'):
            alignment = max(alignment, 0.64)
        if float(guidance.get('hypothesis_match_confidence', 0.0) or 0.0) > 0.0:
            alignment = max(alignment, 0.34 + 0.50 * float(guidance.get('hypothesis_match_confidence', 0.0) or 0.0))

        reward_trend = context.get('reward_trend', 'neutral')
        if reward_trend == 'positive':
            if alignment < 0.5:
                alignment = min(1.0, alignment + 0.1)
        elif reward_trend == 'negative':
            if alignment > 0.5:
                alignment = max(0.0, alignment - 0.1)

        if goal_assessment:
            alignment = self._alignment_policy.adjust_alignment(alignment, goal_assessment, meta)

        return self._clamp(alignment)

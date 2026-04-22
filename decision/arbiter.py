"""
decision/arbiter.py

Decision arbiter.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import uuid

from decision.utility_schema import (
    DecisionCandidate,
    DecisionScore,
    DecisionOutcome,
    CandidateSource,
)
from decision.value_model import ValueModel
from decision.risk_model import RiskModel
from decision.world_model_policy import WorldModelPolicy
from modules.world_model.protocol import WorldModelControlProtocol
from modules.world_model.mechanism_runtime import mechanism_obs_state
from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.action_utils import extract_action_function_name, extract_action_kind
from core.orchestration.commit_candidate_guard import is_probe_like


class DecisionArbiter:
    def __init__(self):
        self._value_model = ValueModel()
        self._risk_model = RiskModel()
        self._decision_count = 0
        self._world_model_policy = WorldModelPolicy()

    def _canonical_context(self, context: Dict[str, Any]) -> UnifiedCognitiveContext:
        from core.cognition.unified_context import CANONICAL_CONTEXT_VERSION

        if isinstance(context, UnifiedCognitiveContext):
            return context
        if not isinstance(context, dict):
            return UnifiedCognitiveContext()
        uc = context.get('unified_context') or context.get('unified_cognitive_context')
        if isinstance(uc, UnifiedCognitiveContext):
            return uc
        if isinstance(uc, dict):
            schema = uc.get('schema_version', '')
            if schema != CANONICAL_CONTEXT_VERSION or not schema:
                context.setdefault('contract_warnings', []).append('schema_version_mismatch_or_missing')
            return UnifiedCognitiveContext.from_parts(
                schema_version=uc.get('schema_version'),
                current_goal=uc.get('current_goal'),
                current_task=uc.get('current_task'),
                active_beliefs_summary=uc.get('active_beliefs_summary'),
                active_hypotheses_summary=uc.get('active_hypotheses_summary'),
                plan_state_summary=uc.get('plan_state_summary'),
                self_model_summary=uc.get('self_model_summary'),
                recent_failure_profile=uc.get('recent_failure_profile'),
                recent_progress_markers=uc.get('recent_progress_markers'),
                retrieval_pressure=uc.get('retrieval_pressure'),
                retrieval_triggered=uc.get('retrieval_triggered'),
                probe_pressure=uc.get('probe_pressure'),
                resource_pressure=uc.get('resource_pressure'),
                world_shift_risk=uc.get('world_shift_risk'),
                task_frame_summary=uc.get('task_frame_summary'),
                object_bindings_summary=uc.get('object_bindings_summary'),
                goal_hypotheses_summary=uc.get('goal_hypotheses_summary'),
                solver_state_summary=uc.get('solver_state_summary'),
                mechanism_hypotheses_summary=uc.get('mechanism_hypotheses_summary'),
                mechanism_control_summary=uc.get('mechanism_control_summary'),
                surfaced_representations=uc.get('surfaced_representations'),
                competing_hypotheses=uc.get('competing_hypotheses'),
                candidate_tests=uc.get('candidate_tests'),
                active_skills=uc.get('active_skills'),
                transfer_candidates=uc.get('transfer_candidates'),
                identity_state=uc.get('identity_state'),
                autobiographical_state=uc.get('autobiographical_state'),
                candidate_programs=uc.get('candidate_programs'),
                candidate_outputs=uc.get('candidate_outputs'),
                deliberation_budget=uc.get('deliberation_budget'),
                deliberation_mode=uc.get('deliberation_mode'),
                uncertainty_vector=uc.get('uncertainty_vector'),
                evidence_queue=uc.get('evidence_queue'),
                workspace_provenance=uc.get('workspace_provenance'),
                safety_budget=uc.get('safety_budget'),
                compute_budget=uc.get('compute_budget'),
                goal_agenda=uc.get('goal_agenda'),
                long_horizon_commitments=uc.get('long_horizon_commitments'),
            )
        context.setdefault('contract_warnings', []).append('schema_version_mismatch_or_missing')
        context.setdefault('contract_warnings', []).append('context.missing_fields:unified_cognitive_context')
        return UnifiedCognitiveContext.from_parts(
            self_model_summary=context.get('self_model_summary', {}),
            plan_state_summary=context.get('plan_state_summary', {}),
            task_frame_summary=context.get('task_frame_summary', {}),
            object_bindings_summary=context.get('object_bindings_summary', {}),
            goal_hypotheses_summary=context.get('goal_hypotheses_summary', []),
            solver_state_summary=context.get('solver_state_summary', {}),
            mechanism_hypotheses_summary=context.get('mechanism_hypotheses_summary', []),
            mechanism_control_summary=context.get('mechanism_control_summary', {}),
        )

    @property
    def decision_count(self) -> int:
        return self._decision_count

    def decide(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> DecisionOutcome:
        self._decision_count += 1
        if not candidates:
            return self._decide_no_candidates(context)
        decision_candidates = self._convert_candidates(candidates, context)
        scores = self._score_all(decision_candidates, context)
        return self._select_best(decision_candidates, scores, context)

    def _convert_candidates(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[DecisionCandidate]:
        decision_candidates: List[DecisionCandidate] = []
        for i, action_dict in enumerate(candidates):
            fn = extract_action_function_name(action_dict, default='') if isinstance(action_dict, dict) else ''
            kind = extract_action_kind(action_dict, default='call_tool') if isinstance(action_dict, dict) else 'call_tool'
            is_wait = kind == 'wait'
            is_probe = kind == 'probe'
            raw_source = action_dict.get('_source', 'base_generation') if isinstance(action_dict, dict) else 'base_generation'
            source = CandidateSource.BASE_GENERATION
            if raw_source == 'retrieval':
                source = CandidateSource.RETRIEVAL
            elif raw_source == 'arm_evaluation':
                source = CandidateSource.ARM_EVALUATION
            elif raw_source == 'base_generation':
                source = CandidateSource.BASE_GENERATION
            elif raw_source == 'planner':
                source = CandidateSource.PLANNER
            elif raw_source == 'self_model':
                source = CandidateSource.SELF_MODEL
            elif raw_source == 'history_reuse':
                source = CandidateSource.HISTORY_REUSE
            elif raw_source == 'procedure_reuse':
                source = CandidateSource.PROCEDURE_REUSE
            elif raw_source == 'wait_fallback':
                source = CandidateSource.WAIT_FALLBACK
            elif raw_source in {
                'intervention_compiler',
                'intervention_execution_compiler',
                'intervention_target',
                'intervention',
                'affordance_intervention',
            }:
                source = CandidateSource.INTERVENTION

            candidate_meta = action_dict.get('_candidate_meta', {}) if isinstance(action_dict, dict) else {}
            surfaced_from = list(candidate_meta.get('surfaced_from', [])) if isinstance(candidate_meta, dict) else []
            if isinstance(candidate_meta, dict):
                for key in ('intervention_target_ids', 'anchor_ids'):
                    values = candidate_meta.get(key, [])
                    if isinstance(values, list):
                        for value in values:
                            text = str(value or '').strip()
                            if text and text not in surfaced_from:
                                surfaced_from.append(text)

            candidate = DecisionCandidate(
                action=action_dict,
                candidate_id=f"cand_{self._decision_count}_{i}_{uuid.uuid4().hex[:4]}",
                source=source,
                surfaced_from=surfaced_from,
                function_name=fn,
                action_kind=kind,
                is_wait=is_wait,
                is_probe=is_probe,
                episode=context.get('episode', 0),
                tick=context.get('tick', 0),
            )
            decision_candidates.append(candidate)
        return decision_candidates

    def _score_all(self, candidates: List[DecisionCandidate], context: Dict[str, Any]) -> List[DecisionScore]:
        scores = []
        unified_context = self._canonical_context(context)
        plan_criticality = float(unified_context.plan_state_summary.get('criticality', 0.5) or 0.5)
        self_model_reliability = float(unified_context.self_model_summary.get('global_reliability', 0.5) or 0.5)
        recovery_availability = float(unified_context.self_model_summary.get('recovery_availability', 0.5) or 0.5)
        self._annotate_mechanism_control_candidates(candidates, context)

        for candidate in candidates:
            value_score = self._value_model.score(candidate, context)
            risk_score = self._risk_model.score(candidate, context)
            prediction_score, prediction_confidence = self._compute_prediction_signal(candidate, context)
            procedure_bonus = self._compute_procedure_bonus(candidate, context)
            learning_bonus = self._compute_learning_bonus(candidate, context)

            wm_protocol = self._wm_protocol(context)
            score = DecisionScore(
                candidate_id=candidate.candidate_id,
                value_score=value_score,
                risk_score=risk_score,
                deliberation_bonus=self._compute_deliberation_bonus(candidate, context) + procedure_bonus,
                learning_bonus=learning_bonus,
                prediction_score=prediction_score,
                prediction_confidence=prediction_confidence,
                world_model_penalty=self._compute_world_model_penalty(candidate, context),
                plan_criticality=plan_criticality,
                shift_risk=float(unified_context.world_shift_risk or wm_protocol.state_shift_risk),
                self_model_reliability=self_model_reliability,
                recovery_availability=recovery_availability,
            )
            scores.append(score)
        return scores

    def _compute_learning_bonus(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        learning_policy = context.get('learning_policy', {}) if isinstance(context.get('learning_policy', {}), dict) else {}
        if not bool(learning_policy.get('learning_enabled', False)):
            return 0.0
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        if not isinstance(meta, dict):
            return 0.0
        if any(key in meta for key in ('learning_bias','failure_preference_learning_bias','retention_learning_bonus','world_model_learning_bias')):
            learning_bonus = float(meta.get('learning_bias', 0.0) or 0.0)
        else:
            selector_bonus = float(meta.get('selector_bias', meta.get('learning_bias', 0.0)) or 0.0)
            agenda_bonus = float(meta.get('agenda_prior', 0.0) or 0.0) * 0.25
            learning_bonus = selector_bonus + agenda_bonus
        recovery_bonus = 0.0
        if bool(context.get('recovery_pending', False)):
            shortcuts = learning_policy.get('recovery_shortcut', {}) if isinstance(learning_policy.get('recovery_shortcut', {}), dict) else {}
            for payload in shortcuts.values():
                if not isinstance(payload, dict):
                    continue
                recovery_bonus = max(recovery_bonus, float(payload.get('delta', 0.0) or 0.0) * 0.3)
        return max(-0.45, min(0.45, learning_bonus + recovery_bonus))

    def _compute_procedure_bonus(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        if not bool(context.get('procedure_enabled', False)):
            return 0.0
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        proc = meta.get('procedure', {}) if isinstance(meta, dict) else {}
        if not isinstance(proc, dict):
            return 0.0
        bonus = float(proc.get('procedure_bonus', 0.0) or 0.0)
        if bool(proc.get('is_next_step', False)):
            bonus += 0.05
        failure_rate = float(proc.get('failure_rate', 0.0) or 0.0)
        success_rate = float(proc.get('success_rate', 0.0) or 0.0)
        if failure_rate >= 0.60:
            bonus -= 0.12
        elif failure_rate >= 0.35:
            bonus -= 0.06
        if success_rate <= 0.30:
            bonus -= 0.05
        return max(-0.2, min(0.25, bonus))

    def _compute_prediction_signal(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> tuple[float, float]:
        if not bool(context.get('prediction_enabled', False)):
            return 0.0, 0.0
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        pred = meta.get('prediction', {}) if isinstance(meta, dict) else {}
        if not isinstance(pred, dict):
            return 0.0, 0.0
        trust_map = context.get('predictor_trust', {})
        low_trust = isinstance(trust_map, dict) and any(v == 'low' for v in trust_map.values())
        success = float((pred.get('success', {}) or {}).get('value', 0.0) or 0.0)
        info_gain = float((pred.get('information_gain', {}) or {}).get('value', 0.0) or 0.0)
        reward_sign = str((pred.get('reward_sign', {}) or {}).get('value', 'zero'))
        risk_type = str((pred.get('risk_type', {}) or {}).get('value', 'execution_failure'))
        conf = float(pred.get('overall_confidence', 0.5) or 0.5)
        reward_bonus = 0.08 if reward_sign == 'positive' else (0.03 if reward_sign == 'zero' else -0.04)
        risk_penalty = 0.12 if risk_type in {'schema_failure', 'resource_failure', 'execution_failure'} else 0.05
        trust_penalty = 0.06 if low_trust else 0.0
        score = (0.18 * success) + (0.14 * info_gain) + reward_bonus - risk_penalty - trust_penalty
        return score, max(0.25, min(1.0, conf))

    def _wm_protocol(self, context: Dict[str, Any]) -> WorldModelControlProtocol:
        return WorldModelControlProtocol.from_context(context)

    def _mechanism_control_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        uc = context.get('unified_context') or context.get('unified_cognitive_context')
        if isinstance(uc, dict):
            summary = uc.get('mechanism_control_summary', context.get('mechanism_control_summary', {}))
            return dict(summary) if isinstance(summary, dict) else {}
        summary = context.get('mechanism_control_summary', {})
        return dict(summary) if isinstance(summary, dict) else {}

    def _mechanism_candidate_role(self, candidate: Optional[DecisionCandidate]) -> str:
        if candidate is None or not isinstance(candidate.action, dict):
            return ''
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action.get('_candidate_meta', {}), dict) else {}
        role = str(meta.get('role', '') or '').strip().lower()
        if role:
            return role
        if candidate.is_wait:
            return 'wait'
        if candidate.is_probe or is_probe_like(candidate.function_name, kind=candidate.action_kind):
            return 'discriminate'
        return 'commit'

    def _mechanism_obs_state(self, context: Dict[str, Any], mechanism_control: Dict[str, Any]) -> Dict[str, bool]:
        obs_before = context.get('obs_before', {})
        obs = dict(obs_before) if isinstance(obs_before, dict) else {}
        return mechanism_obs_state(obs, mechanism_control)

    def _mechanism_commit_actionable(
        self,
        *,
        role: str,
        profile: Dict[str, Any],
        score_margin: float,
        mechanism_control: Dict[str, Any],
    ) -> bool:
        if role != 'commit':
            return False
        commitment_trust = float(mechanism_control.get('commitment_trust', 0.0) or 0.0)
        commitment_revoked = bool(mechanism_control.get('commitment_revoked', False))
        if bool(profile.get('revoked_match', False)):
            return False
        if float(profile.get('contradiction_penalty', 0.0) or 0.0) >= 0.45:
            return False
        if bool(profile.get('active_match', False)) and not commitment_revoked and commitment_trust >= 0.58:
            return True
        return bool(
            float(profile.get('evidence_strength', 0.0) or 0.0) >= 0.85
            and float(profile.get('specificity', 0.0) or 0.0) >= 0.45
            and float(score_margin) >= 0.18
        )

    def _mechanism_mode_alignment(
        self,
        candidate: DecisionCandidate,
        *,
        role: str,
        profile: Dict[str, Any],
        actionable: bool,
        mechanism_control: Dict[str, Any],
        obs_state: Dict[str, bool],
    ) -> float:
        control_mode = str(mechanism_control.get('control_mode', '') or '')
        probe_like = bool(candidate.is_probe or is_probe_like(candidate.function_name, kind=candidate.action_kind))
        if role == 'recovery' and bool(obs_state.get('recovery_ready', False)):
            return 1.0
        if role in {'prerequisite', 'prepare'} and bool(obs_state.get('prerequisite_ready', False)):
            return 1.0
        if (candidate.is_wait or candidate.function_name == 'wait') and bool(obs_state.get('wait_ready', False)):
            return 1.0
        if control_mode == 'discriminate':
            if role == 'discriminate' or probe_like:
                return 1.0
            if actionable:
                return 0.45
            return 0.0
        if control_mode == 'exploit':
            if actionable:
                return 1.0
            if role == 'commit' and bool(profile.get('active_match', False)):
                return 0.72
            return 0.0
        if control_mode == 'wait':
            return 1.0 if candidate.is_wait or candidate.function_name == 'wait' else 0.0
        if control_mode == 'recover':
            return 1.0 if role == 'recovery' else 0.0
        if control_mode == 'prepare':
            return 1.0 if role in {'prerequisite', 'prepare'} else 0.0
        return 0.0

    def _annotate_mechanism_control_candidates(self, candidates: List[DecisionCandidate], context: Dict[str, Any]) -> None:
        from modules.hypothesis.mechanism_posterior_updater import binding_token_frequency, candidate_binding_signal

        mechanism_control = self._mechanism_control_summary(context)
        if not mechanism_control:
            return
        obs_before = context.get('obs_before', {})
        obs_snapshot = dict(obs_before) if isinstance(obs_before, dict) else {}
        obs_state = self._mechanism_obs_state(context, mechanism_control)
        actions = [candidate.action for candidate in candidates if isinstance(candidate.action, dict)]
        token_frequency = binding_token_frequency(actions)
        profiles_by_id: Dict[str, Dict[str, Any]] = {}
        commit_scores: Dict[str, float] = {}
        roles_by_id: Dict[str, str] = {}

        for candidate in candidates:
            if not isinstance(candidate.action, dict):
                continue
            role = self._mechanism_candidate_role(candidate)
            roles_by_id[candidate.candidate_id] = role
            profile = candidate_binding_signal(
                candidate.action,
                obs_before=obs_snapshot,
                mechanism_control=mechanism_control,
                token_frequency=token_frequency,
            )
            profiles_by_id[candidate.candidate_id] = profile
            if role == 'commit':
                commit_scores[candidate.candidate_id] = float(profile.get('score', 0.0) or 0.0)

        for candidate in candidates:
            if not isinstance(candidate.action, dict):
                continue
            meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action.get('_candidate_meta', {}), dict) else {}
            profile = dict(profiles_by_id.get(candidate.candidate_id, {}))
            role = roles_by_id.get(candidate.candidate_id, self._mechanism_candidate_role(candidate))
            own_score = float(profile.get('score', 0.0) or 0.0)
            runner_up = 0.0
            if role == 'commit':
                runner_up = max(
                    [score for cand_id, score in commit_scores.items() if cand_id != candidate.candidate_id],
                    default=0.0,
                )
            score_margin = own_score - runner_up if role == 'commit' else 0.0
            actionable = self._mechanism_commit_actionable(
                role=role,
                profile=profile,
                score_margin=score_margin,
                mechanism_control=mechanism_control,
            )
            mode_alignment = self._mechanism_mode_alignment(
                candidate,
                role=role,
                profile=profile,
                actionable=actionable,
                mechanism_control=mechanism_control,
                obs_state=obs_state,
            )
            merged_meta = dict(meta)
            merged_meta['runtime_discriminating_candidate'] = bool(
                merged_meta.get('runtime_discriminating_candidate', False)
                or role == 'discriminate'
                or candidate.is_probe
                or is_probe_like(candidate.function_name, kind=candidate.action_kind)
            )
            merged_meta['mechanism_binding_score'] = round(float(profile.get('score', 0.0) or 0.0), 4)
            merged_meta['mechanism_binding_margin'] = round(float(score_margin), 4)
            merged_meta['mechanism_binding_specificity'] = round(float(profile.get('specificity', 0.0) or 0.0), 4)
            merged_meta['mechanism_binding_evidence_strength'] = round(float(profile.get('evidence_strength', 0.0) or 0.0), 4)
            merged_meta['mechanism_contradiction_penalty'] = round(float(profile.get('contradiction_penalty', 0.0) or 0.0), 4)
            merged_meta['mechanism_binding_actionable'] = bool(actionable)
            merged_meta['mechanism_mode_alignment'] = round(float(mode_alignment), 4)
            merged_meta['mechanism_release_ready'] = bool(obs_state.get('release_ready', False))
            merged_meta['mechanism_wait_ready'] = bool(obs_state.get('wait_ready', False))
            merged_meta['mechanism_prerequisite_ready'] = bool(obs_state.get('prerequisite_ready', False))
            merged_meta['mechanism_recovery_ready'] = bool(obs_state.get('recovery_ready', False))
            candidate.action['_candidate_meta'] = merged_meta

    def _compute_deliberation_bonus(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        rollout_map = context.get('rollout_predictions', {})
        if not isinstance(rollout_map, dict):
            rollout_map = {}

        key = candidate.function_name or candidate.candidate_id
        pred = rollout_map.get(key, {})
        if not isinstance(pred, dict) or not pred:
            wm_projection = self._world_model_policy.project_candidate(candidate, context)
            pred = {
                'short_reward': 0.0,
                'long_reward': wm_projection.get('long_horizon_reward', 0.0),
                'info_gain': wm_projection.get('info_gain', 0.0),
                'reversibility': wm_projection.get('reversibility', 0.0),
                'risk': wm_projection.get('predicted_risk', 0.0),
            }

        short_reward = float(pred.get('short_reward', 0.0))
        long_reward = float(pred.get('long_reward', 0.0))
        info_gain = float(pred.get('info_gain', 0.0))
        reversibility = float(pred.get('reversibility', 0.0))
        rollout_risk = float(pred.get('risk', 0.0))

        lam = float(context.get('deliberation_long_weight', 0.6))
        alpha = float(context.get('deliberation_risk_weight', 0.4))
        beta = float(context.get('deliberation_info_weight', 0.2))
        gamma = float(context.get('deliberation_reversibility_weight', 0.2))
        bonus = short_reward + lam * long_reward - alpha * rollout_risk + beta * info_gain + gamma * reversibility

        depth = int(context.get('deliberation_depth', 1) or 1)
        scaled = bonus * min(1.0, max(0.2, depth / 5.0))
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action, dict) else {}
        if isinstance(meta, dict):
            engine_rank = int(meta.get('deliberation_engine_rank', 0) or 0)
            engine_score = float(meta.get('deliberation_engine_score', 0.0) or 0.0)
            if engine_rank > 0:
                scaled += max(0.0, 0.10 - ((engine_rank - 1) * 0.02))
            if engine_score:
                scaled += max(-0.15, min(0.15, (engine_score - 0.5) * 0.30))
        if bool(context.get('probe_before_commit', False)):
            scaled += 0.20 if candidate.is_probe else -0.08
        return max(-0.5, min(0.5, scaled))

    def _compute_world_model_penalty(self, candidate: DecisionCandidate, context: Dict[str, Any]) -> float:
        constraints = context.get('world_model_constraints', {})
        if not isinstance(constraints, dict):
            constraints = {}
        protocol = self._wm_protocol(context)

        fn_penalties = constraints.get('function_penalties', {})
        if isinstance(fn_penalties, dict) and candidate.function_name in fn_penalties:
            return max(0.0, float(fn_penalties[candidate.function_name]))

        blocked_functions = set(constraints.get('blocked_functions', []) or [])
        blocked_functions.update(protocol.blocked_functions or [])
        if candidate.function_name in blocked_functions:
            return 0.6

        wm_projection = self._world_model_policy.project_candidate(candidate, context)
        violation_prob = float(wm_projection.get('constraint_violation_prob', 0.0))
        threshold = float(context.get('world_model_hard_constraint_threshold', 0.8))
        if violation_prob >= threshold:
            return 0.8
        return max(0.0, violation_prob * 0.5)

    def _select_best(self, candidates: List[DecisionCandidate], scores: List[DecisionScore], context: Dict[str, Any]) -> DecisionOutcome:
        if not scores:
            return self._decide_no_candidates(context)
        candidate_by_id = {c.candidate_id: c for c in candidates}
        sorted_scores = sorted(
            scores,
            key=lambda s: self._selection_sort_key(candidate_by_id.get(s.candidate_id), s, context),
            reverse=True,
        )
        best_score = sorted_scores[0]
        best_candidate = None
        for c in candidates:
            if c.candidate_id == best_score.candidate_id:
                best_candidate = c
                break
        rejected_ids = [s.candidate_id for s in sorted_scores[1:]]
        primary_reason = self._build_reason(best_candidate, best_score, context)
        return DecisionOutcome(
            selected_candidate=best_candidate,
            selected_score=best_score,
            all_scores=sorted_scores,
            rejected_ids=rejected_ids,
            primary_reason=primary_reason,
            secondary_reasons=self._build_secondary_reasons(best_score),
            score_breakdowns=self._build_score_breakdowns(candidates, scores),
            execute_as=best_candidate.action_kind if best_candidate else 'wait',
        )

    def _selection_sort_key(
        self,
        candidate: Optional[DecisionCandidate],
        score: DecisionScore,
        context: Dict[str, Any],
    ) -> tuple[float, ...]:
        return (
            *self._mechanism_control_priority(candidate, context),
            *self._learned_dynamics_selective_priority(candidate),
            float(score.final_score),
        )

    def _mechanism_control_priority(self, candidate: Optional[DecisionCandidate], context: Dict[str, Any]) -> tuple[float, float, float, float, float]:
        if candidate is None or not isinstance(candidate.action, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        mechanism_control = self._mechanism_control_summary(context)
        if not mechanism_control:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        meta = candidate.action.get('_candidate_meta', {}) if isinstance(candidate.action.get('_candidate_meta', {}), dict) else {}
        control_mode = str(mechanism_control.get('control_mode', '') or '')
        role = self._mechanism_candidate_role(candidate)
        mode_alignment = float(meta.get('mechanism_mode_alignment', 0.0) or 0.0)
        actionable = 1.0 if bool(meta.get('mechanism_binding_actionable', False)) else 0.0
        binding_score = float(meta.get('mechanism_binding_score', 0.0) or 0.0)
        binding_margin = float(meta.get('mechanism_binding_margin', 0.0) or 0.0)
        expected_info = float(meta.get('expected_information_gain', 0.0) or 0.0)
        probe_like = bool(candidate.is_probe or is_probe_like(candidate.function_name, kind=candidate.action_kind))
        release_ready = bool(meta.get('mechanism_release_ready', False))
        wait_ready = bool(meta.get('mechanism_wait_ready', False))
        prerequisite_ready = bool(meta.get('mechanism_prerequisite_ready', False))
        recovery_ready = bool(meta.get('mechanism_recovery_ready', False))
        tier = 0.0
        if recovery_ready:
            tier = 5.0 if role == 'recovery' else (-2.0 if role == 'commit' else -0.5)
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        if prerequisite_ready:
            tier = 5.0 if role in {'prerequisite', 'prepare'} else (-2.0 if role == 'commit' else -0.5)
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        if wait_ready:
            tier = 5.0 if candidate.is_wait or candidate.function_name == 'wait' else (-1.5 if role == 'commit' else -0.5)
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        if control_mode == 'discriminate':
            if role == 'discriminate' or probe_like:
                tier = 4.0
            elif actionable:
                tier = 1.0
            else:
                tier = -1.0
            return (tier, mode_alignment, expected_info, binding_score, binding_margin)
        if control_mode == 'exploit':
            if role == 'commit' and not release_ready and not actionable:
                tier = -2.0
                return (tier, mode_alignment, actionable, binding_margin, binding_score)
            if actionable:
                tier = 4.0
            elif role == 'commit' and binding_score >= 0.35:
                tier = 2.0
            elif probe_like:
                tier = -0.5
            return (tier, mode_alignment, actionable, binding_margin, binding_score)
        if control_mode == 'wait':
            tier = 4.0 if candidate.is_wait or candidate.function_name == 'wait' else -0.5
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        if control_mode == 'recover':
            tier = 4.0 if role == 'recovery' else -0.5
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        if control_mode == 'prepare':
            tier = 4.0 if role in {'prerequisite', 'prepare'} else -0.5
            return (tier, mode_alignment, binding_score, binding_margin, expected_info)
        return (0.0, mode_alignment, binding_score, binding_margin, expected_info)

    def _learned_dynamics_selective_priority(self, candidate: Optional[DecisionCandidate]) -> tuple[float, float, float, float, float]:
        if candidate is None or not isinstance(candidate.action, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        meta = candidate.action.get('_candidate_meta', {})
        if not isinstance(meta, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        if str(meta.get('learned_dynamics_deployment_mode', '') or '') != 'selective_routing':
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        if not bool(meta.get('learned_dynamics_routing_active', False)):
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        prediction = meta.get('learned_dynamics_prediction', {})
        if not isinstance(prediction, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        reward_sign = str(prediction.get('reward_sign', '') or '')
        risk_type = str(prediction.get('risk_type', '') or '')
        information_gain = float(prediction.get('information_gain', 0.0) or 0.0)
        confidence = float(meta.get('learned_dynamics_confidence', 0.0) or 0.0)
        governance_bonus = float(meta.get('learned_dynamics_governance_bonus', 0.0) or 0.0)
        valid_state_change = bool(prediction.get('valid_state_change', False))
        promotion_signal = bool(meta.get('learned_dynamics_promotion_signal', False))
        veto_signal = bool(meta.get('learned_dynamics_veto_signal', False))

        if veto_signal:
            tier = -1.0
        elif promotion_signal or reward_sign == 'positive':
            tier = 3.0
        elif valid_state_change and information_gain >= 0.12:
            tier = 2.0
        else:
            tier = 1.0

        safe_risk = 0.0 if risk_type in {'schema_failure', 'resource_failure', 'execution_failure'} else 1.0
        return (
            tier,
            safe_risk,
            information_gain,
            confidence,
            governance_bonus,
        )

    def _build_score_breakdowns(self, candidates: List[DecisionCandidate], scores: List[DecisionScore]) -> List[Dict[str, Any]]:
        candidate_by_id = {c.candidate_id: c for c in candidates}
        rows: List[Dict[str, Any]] = []
        for score in sorted(scores, key=lambda x: x.final_score, reverse=True):
            cand = candidate_by_id.get(score.candidate_id)
            if not cand:
                continue
            kwargs_repr = ''
            if isinstance(cand.action, dict):
                payload = cand.action.get('payload', {}) if isinstance(cand.action.get('payload'), dict) else {}
                tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
                kwargs_repr = repr(tool_args.get('kwargs', {}))
            rows.append({
                'candidate_id': cand.candidate_id,
                'source': cand.source.value,
                'function_name': cand.function_name,
                'kwargs_repr': kwargs_repr,
                'final_score': score.final_score,
                'raw_final_score': score.raw_final_score,
                'intervention_value': score.value_score.intervention_value,
                'breakdown': score.breakdown.to_dict(),
            })
        return rows

    def _build_reason(self, candidate: Optional[DecisionCandidate], score: Optional[DecisionScore], context: Dict[str, Any]) -> str:
        if not candidate or not score:
            return "no viable candidate"

        reasons = []
        value = score.value_score
        if value.utility > 0.7:
            reasons.append(f"high utility ({value.utility:.2f})")
        if value.novelty > 0.6:
            reasons.append(f"novel ({value.novelty:.2f})")
        if value.goal_alignment > 0.7:
            reasons.append(f"goal-aligned ({value.goal_alignment:.2f})")
        if value.intervention_value > 0.05:
            reasons.append(f"intervention ({value.intervention_value:.2f})")

        risk = score.risk_score
        if risk.level == "low":
            reasons.append(f"low risk ({risk.failure_likelihood:.2f})")
        elif risk.level == "high":
            reasons.append(f"high risk ({risk.failure_likelihood:.2f})")

        if score.deliberation_bonus > 0.05:
            reasons.append(f"rollout+ ({score.deliberation_bonus:.2f})")
        elif score.deliberation_bonus < -0.05:
            reasons.append(f"rollout- ({score.deliberation_bonus:.2f})")
        if score.learning_bonus > 0.05:
            reasons.append(f"learning+ ({score.learning_bonus:.2f})")
        elif score.learning_bonus < -0.05:
            reasons.append(f"learning- ({score.learning_bonus:.2f})")
        if score.world_model_penalty > 0.05:
            reasons.append(f"wm-constraint ({score.world_model_penalty:.2f})")
        if candidate.function_name:
            reasons.append(f"fn={candidate.function_name}")
        if candidate.source == CandidateSource.INTERVENTION:
            reasons.append("src=intervention")
        if not reasons:
            reasons.append(f"score={score.final_score:.2f}")
        return "; ".join(reasons)

    def _build_secondary_reasons(self, score: DecisionScore) -> List[str]:
        reasons = []
        if score.value_score.confidence < 0.5:
            reasons.append(f"low confidence ({score.value_score.confidence:.2f})")
        if score.risk_score.uncertainty > 0.6:
            reasons.append(f"high uncertainty ({score.risk_score.uncertainty:.2f})")
        return reasons

    def _decide_no_candidates(self, context: Dict[str, Any]) -> DecisionOutcome:
        return DecisionOutcome(
            selected_candidate=None,
            selected_score=None,
            all_scores=[],
            rejected_ids=[],
            primary_reason="no candidates available",
            secondary_reasons=[],
            execute_as="wait",
        )

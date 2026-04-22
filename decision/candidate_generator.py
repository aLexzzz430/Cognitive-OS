"""
decision/candidate_generator.py

Priority 2: multi-source candidate generation.

This module turns planner / retrieval / self-model / history / base fallback
signals into a unified list of action candidates for the DecisionArbiter.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.orchestration.action_utils import (
    extract_action_identity,
    extract_action_function_name,
    extract_action_kind,
    extract_available_functions as extract_surface_available_functions,
)
from core.orchestration.arc3_action_coverage import is_arc3_surface
from core.adapter_registry import build_optional_adapter
from decision.governance_candidate_adapter import _governance_wait_baseline_allowed
from decision.latent_transfer_organs import LatentTransferOrganSystem

try:
    from modules.world_model.affordance_model import AffordanceModel
    from modules.world_model.intervention_targets import (
        InterventionTargetProposer,
        ProposalContext,
        build_world_anchors,
    )
except Exception:  # pragma: no cover - optional integration
    AffordanceModel = None
    InterventionTargetProposer = None
    ProposalContext = None
    build_world_anchors = None

try:
    from modules.world_model.canonical_state import summarize_observation_world
except Exception:  # pragma: no cover - optional integration
    summarize_observation_world = None


try:
    from modules.world_model.task_frame import infer_task_frame
    from modules.world_model.object_binding import build_object_bindings
    from modules.world_model.goal_hypothesis import build_goal_hypotheses, summarize_solver_state
    from modules.world_model.mechanism_hypothesis import build_mechanism_hypotheses, summarize_mechanism_control
except Exception:  # pragma: no cover - optional integration
    infer_task_frame = None
    build_object_bindings = None
    build_goal_hypotheses = None
    summarize_solver_state = None
    build_mechanism_hypotheses = None
    summarize_mechanism_control = None


_ACTION_NAMESPACE_ALIASES = {
    'up': 'ACTION1',
    'move_up': 'ACTION1',
    'down': 'ACTION2',
    'move_down': 'ACTION2',
    'left': 'ACTION3',
    'move_left': 'ACTION3',
    'right': 'ACTION4',
    'move_right': 'ACTION4',
    'space': 'ACTION5',
    'press_space': 'ACTION5',
    'confirm': 'ACTION5',
    'interact': 'ACTION5',
    'submit': 'ACTION5',
    'click': 'ACTION6',
    'pointer_click': 'ACTION6',
    'tap': 'ACTION6',
}


def _procedure_text_tokens(text: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", str(text or "").lower())
        if token and len(token) > 1
    ]


def _extract_candidate_function_name(obj: dict) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    payload = obj.get('payload', {})
    payload_tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    fn = payload_tool_args.get('function_name', '')
    if fn:
        return fn
    content = obj.get('content', {})
    if isinstance(content, dict) and content.get('type') == 'representation':
        return None
    tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
    fn = tool_args.get('function_name', '')
    if fn:
        return fn
    fn = content.get('function_name', '') if isinstance(content, dict) else ''
    if fn:
        return fn
    fn = obj.get('function_name', '')
    return fn or None


def _canonicalize_function_name(function_name: Any) -> str:
    text = str(function_name or '').strip()
    if not text:
        return ''
    upper_text = text.upper()
    if upper_text.startswith('ACTION'):
        return upper_text
    lowered = text.lower()
    return _ACTION_NAMESPACE_ALIASES.get(lowered, text)


class CandidateGenerator:
    def __init__(
        self,
        strict_signature_mode: bool = True,
        best_effort_mode: bool = False,
    ):
        self._strict_signature_mode = bool(strict_signature_mode)
        self._best_effort_mode = bool(best_effort_mode)
        self._latent_transfer_organs = LatentTransferOrganSystem()
        self._intervention_target_proposer = InterventionTargetProposer(max_targets=8) if InterventionTargetProposer else None
        self._affordance_model = AffordanceModel() if AffordanceModel else None
        self._arc_intervention_execution_compiler = build_optional_adapter(
            "arc_agi3.intervention_execution_compiler",
        )

    def generate(
        self,
        obs: Dict[str, Any],
        surfaced: Sequence[Any],
        continuity_snapshot: Optional[Dict[str, Any]],
        base_action: Optional[Dict[str, Any]],
        arm_action: Optional[Dict[str, Any]],
        plan_state: Any,
        capability_profile: Any,
        reliability_tracker: Any,
        episode_trace: Sequence[Dict[str, Any]],
        perception_summary: Optional[Dict[str, Any]] = None,
        world_model_summary: Optional[Dict[str, Any]] = None,
        procedure_objects: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        env_available_functions = self._extract_env_available_functions(obs)
        inferred_functions = self._extract_inferred_functions(surfaced, base_action, arm_action)
        available_functions = self._merge_function_sources(env_available_functions, inferred_functions)
        candidates: List[Dict[str, Any]] = []

        for candidate in self._build_surface_action_candidates(
            obs=obs,
            env_available_functions=env_available_functions,
            episode_trace=episode_trace,
            world_model_summary=world_model_summary or {},
        ):
            self._append_candidate(candidates, candidate)

        for candidate in self._build_intervention_candidates(
            obs=obs,
            available_functions=available_functions,
            continuity_snapshot=continuity_snapshot,
            episode_trace=episode_trace,
            world_model_summary=world_model_summary or {},
        ):
            self._append_candidate(candidates, candidate)

        for candidate in self._build_retrieval_candidates(surfaced, obs, plan_state, episode_trace, world_model_summary):
            self._append_candidate(candidates, candidate)

        planner_candidate = self._build_planner_candidate(plan_state, env_available_functions, obs, episode_trace, world_model_summary)
        self._append_candidate(candidates, planner_candidate)

        self_model_candidate = self._build_self_model_candidate(
            available_functions=available_functions,
            capability_profile=capability_profile,
            reliability_tracker=reliability_tracker,
            continuity_snapshot=continuity_snapshot,
            episode_trace=episode_trace,
            perception_summary=perception_summary,
            world_model_summary=world_model_summary,
            obs=obs,
        )
        self._append_candidate(candidates, self_model_candidate)

        history_candidate = self._build_history_candidate(episode_trace, available_functions)
        self._append_candidate(candidates, history_candidate)
        for proc_candidate in self._build_procedure_candidates(
            obs=obs,
            available_functions=available_functions,
            episode_trace=episode_trace,
            procedure_objects=procedure_objects or [],
        ):
            self._append_candidate(candidates, proc_candidate)
        for latent_candidate in self._build_latent_mechanism_candidates(
            obs=obs,
            available_functions=available_functions,
            episode_trace=episode_trace,
            procedure_objects=procedure_objects or [],
        ):
            self._append_candidate(candidates, latent_candidate)

        if base_action:
            self._append_candidate(candidates, self._with_source(base_action, 'base_generation'))

        if arm_action:
            self._append_candidate(candidates, self._with_source(arm_action, 'arm_evaluation'))

        has_viable_non_wait_candidate = self._has_viable_non_wait_candidate(candidates)
        if (
            not has_viable_non_wait_candidate
            and _governance_wait_baseline_allowed(candidates, None)
            and not (is_arc3_surface(obs) and bool(env_available_functions))
        ):
            wait_candidate = self._build_wait_candidate()
            wait_meta = wait_candidate.get('_candidate_meta', {})
            if not env_available_functions:
                wait_meta['no_function_surface'] = True
            else:
                wait_meta['visible_functions_but_no_viable_candidate'] = list(env_available_functions)
            wait_meta['wait_injection_reason'] = 'no_viable_non_wait'
            wait_candidate['_candidate_meta'] = wait_meta
            self._append_candidate(candidates, wait_candidate)
        self._record_candidate_trace(
            obs,
            phase='before_sanitize',
            candidate_count=len(candidates),
            filter_reason_counts={},
        )
        sanitized, sanitize_stats = self._sanitize_candidates_for_arbiter(candidates, obs)
        self._record_candidate_trace(
            obs,
            phase='after_sanitize',
            candidate_count=len(sanitized),
            filter_reason_counts=sanitize_stats.get('filter_reason_counts', {}),
        )
        self._annotate_recent_action_feedback(sanitized, episode_trace)
        return sanitized

    def _build_intervention_candidates(
        self,
        obs: Dict[str, Any],
        available_functions: Sequence[str],
        continuity_snapshot: Optional[Dict[str, Any]],
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self._intervention_target_proposer is None or ProposalContext is None:
            return []
        compiler = self._select_intervention_execution_compiler(obs, available_functions)
        if compiler is None:
            return []

        intervention_world_summary = self._build_intervention_world_summary(obs, world_model_summary)
        recent_interactions = self._recent_intervention_interactions(episode_trace)
        current_goal = ''
        if isinstance(continuity_snapshot, dict):
            top_goal = continuity_snapshot.get('top_goal')
            if top_goal is not None:
                current_goal = str(getattr(top_goal, 'description', '') or '')
        if not current_goal:
            current_goal = str(obs.get('goal') or obs.get('task') or obs.get('instruction') or '')

        context = ProposalContext(
            world_model_summary=intervention_world_summary,
            recent_interactions=recent_interactions,
            current_goal=current_goal,
            active_hypotheses=list(world_model_summary.get('active_hypotheses', [])) if isinstance(world_model_summary.get('active_hypotheses', []), list) else [],
            task_frame_summary=dict(intervention_world_summary.get('task_frame_summary', {}) or {}),
            object_bindings_summary=dict(intervention_world_summary.get('object_bindings_summary', {}) or {}),
            goal_hypotheses_summary=list(intervention_world_summary.get('goal_hypotheses_summary', []) or []),
            solver_state_summary=dict(intervention_world_summary.get('solver_state_summary', {}) or {}),
            mechanism_hypotheses_summary=list(intervention_world_summary.get('mechanism_hypotheses_summary', []) or []),
            mechanism_control_summary=dict(intervention_world_summary.get('mechanism_control_summary', {}) or {}),
        )
        targets = self._intervention_target_proposer.propose(context)
        if not targets:
            return []

        anchors = build_world_anchors(intervention_world_summary, recent_interactions=recent_interactions) if build_world_anchors else []
        affordance_by_anchor = self._affordance_model.infer_for_targets(anchors, targets) if self._affordance_model and anchors else {}

        compiled_candidates: List[Dict[str, Any]] = []
        recent_failed_points = {
            tuple(item.get('point', ()))
            for item in self._extract_recent_click_feedback(episode_trace)
            if bool(item.get('failed')) and isinstance(item.get('point'), tuple)
        }
        for target in targets:
            compiled_actions = compiler.compile(
                target,
                available_functions=available_functions,
                obs=obs,
            )
            for compiled in compiled_actions:
                action = compiled.to_action_dict()
                if _canonicalize_function_name(self._function_name_from_action(action)) == 'ACTION6':
                    payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
                    tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
                    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
                    try:
                        point = (int(kwargs.get('x')), int(kwargs.get('y')))
                    except (TypeError, ValueError):
                        point = None
                    if point is not None and point in recent_failed_points:
                        continue
                action['_source'] = 'intervention_compiler'
                meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
                meta['intervention_target'] = target.to_dict()
                meta['intervention_anchor_affordance'] = affordance_by_anchor.get(target.anchor_ref, {}) if isinstance(affordance_by_anchor, dict) else {}
                meta['intervention_world_signature'] = str(intervention_world_summary.get('world_state_signature', '') or '')
                meta['intervention_expected_effect_type'] = str(target.expected_effect_type or '')
                meta['intervention_compiler'] = compiler.__class__.__name__
                priority_features = target.priority_features if isinstance(target.priority_features, dict) else {}
                task_frame_summary = intervention_world_summary.get('task_frame_summary', {}) if isinstance(intervention_world_summary.get('task_frame_summary', {}), dict) else {}
                solver_state_summary = intervention_world_summary.get('solver_state_summary', {}) if isinstance(intervention_world_summary.get('solver_state_summary', {}), dict) else {}
                object_bindings_summary = intervention_world_summary.get('object_bindings_summary', {}) if isinstance(intervention_world_summary.get('object_bindings_summary', {}), dict) else {}
                objects = object_bindings_summary.get('objects', []) if isinstance(object_bindings_summary.get('objects', []), list) else []
                bound_object = next((obj for obj in objects if isinstance(obj, dict) and str(obj.get('object_id', '') or '') == str(target.anchor_ref or '')), {})
                semantic_candidates = bound_object.get('semantic_candidates', []) if isinstance(bound_object, dict) and isinstance(bound_object.get('semantic_candidates', []), list) else []
                role_candidates = bound_object.get('role_candidates', []) if isinstance(bound_object, dict) and isinstance(bound_object.get('role_candidates', []), list) else []
                meta['target_value_score'] = float(priority_features.get('target_value_score', priority_features.get('final_score', 0.0)) or 0.0)
                meta['expected_state_change'] = float(priority_features.get('expected_state_change', 0.0) or 0.0)
                meta['expected_progress'] = float(priority_features.get('expected_progress', 0.0) or 0.0)
                meta['expected_information_gain'] = float(priority_features.get('expected_information_gain', 0.0) or 0.0)
                meta['repeat_penalty'] = float(priority_features.get('repeat_penalty', 0.0) or 0.0)
                meta['exploration_bonus'] = float(priority_features.get('exploration_bonus', 0.0) or 0.0)
                meta['solver_goal_family'] = str(solver_state_summary.get('dominant_goal_family', '') or priority_features.get('dominant_goal_family', '') or '')
                meta['solver_goal_confidence'] = float(solver_state_summary.get('dominant_goal_confidence', 0.0) or priority_features.get('dominant_goal_confidence', 0.0) or 0.0)
                meta['solver_dominant_interaction_mode'] = str(task_frame_summary.get('dominant_interaction_mode', '') or priority_features.get('dominant_interaction_mode', '') or '')
                meta['solver_preferred_target_refs'] = list(solver_state_summary.get('preferred_target_refs', []) or [])
                meta['solver_target_match'] = bool(str(target.anchor_ref or '') in set(str(x or '') for x in list(solver_state_summary.get('preferred_target_refs', []) or [])))
                meta['solver_semantic_labels'] = [str(item.get('label', '') or '') for item in semantic_candidates if isinstance(item, dict) and str(item.get('label', '') or '')]
                mechanism_control_summary = intervention_world_summary.get('mechanism_control_summary', {}) if isinstance(intervention_world_summary.get('mechanism_control_summary', {}), dict) else {}
                meta['solver_object_roles'] = [str(item.get('role', '') or '') for item in role_candidates if isinstance(item, dict) and str(item.get('role', '') or '')]
                meta['solver_mechanism_family'] = str(mechanism_control_summary.get('dominant_mechanism_family', '') or priority_features.get('mechanism_family', '') or '')
                meta['solver_mechanism_confidence'] = float(mechanism_control_summary.get('dominant_mechanism_confidence', 0.0) or priority_features.get('mechanism_confidence', 0.0) or 0.0)
                meta['solver_discriminating_actions'] = list(mechanism_control_summary.get('discriminating_actions', []) or priority_features.get('mechanism_discriminating_actions', []) or [])
                mechanism_pref_refs = set(str(x or '') for x in list(mechanism_control_summary.get('preferred_target_refs', []) or []) if str(x or ''))
                meta['solver_mechanism_target_match'] = bool(str(target.anchor_ref or '') in mechanism_pref_refs or priority_features.get('mechanism_target_match', False))
                meta['mechanism_guidance'] = {
                    'mechanism_family': meta['solver_mechanism_family'],
                    'mechanism_confidence': meta['solver_mechanism_confidence'],
                    'discriminating_actions': list(meta['solver_discriminating_actions']),
                    'target_match': bool(meta['solver_mechanism_target_match']),
                }
                support_sources = list(meta.get('support_sources', [])) if isinstance(meta.get('support_sources', []), list) else []
                if 'intervention_compiler' not in support_sources:
                    support_sources.append('intervention_compiler')
                meta['support_sources'] = support_sources
                action['_candidate_meta'] = meta
                compiled_candidates.append(action)

        if isinstance(obs, dict):
            obs.setdefault('intervention_trace', [])
            if isinstance(obs.get('intervention_trace'), list):
                obs['intervention_trace'].append({
                    'world_state_signature': str(intervention_world_summary.get('world_state_signature', '') or ''),
                    'target_count': len(targets),
                    'compiled_candidate_count': len(compiled_candidates),
                    'compiled_functions': [
                        _canonicalize_function_name(self._function_name_from_action(candidate))
                        for candidate in compiled_candidates
                        if _canonicalize_function_name(self._function_name_from_action(candidate))
                    ],
                })

        return compiled_candidates

    def _select_intervention_execution_compiler(
        self,
        obs: Dict[str, Any],
        available_functions: Sequence[str],
    ) -> Any:
        available = {
            _canonicalize_function_name(fn)
            for fn in available_functions
            if _canonicalize_function_name(fn)
        }
        obs_type = str(obs.get('type', '') or '').strip().lower() if isinstance(obs, dict) else ''
        if self._arc_intervention_execution_compiler is not None and (obs_type == 'arc_agi3' or any(fn.startswith('ACTION') for fn in available)):
            return self._arc_intervention_execution_compiler
        return None

    def _build_intervention_world_summary(
        self,
        obs: Dict[str, Any],
        world_model_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = dict(world_model_summary or {})
        derived: Dict[str, Any] = {}
        if summarize_observation_world is not None:
            try:
                derived = dict(summarize_observation_world(obs) or {})
            except Exception:
                derived = {}
        merged = dict(derived)
        merged.update(summary)
        perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
        if perception and not isinstance(merged.get('perception', {}), dict):
            merged['perception'] = perception
        if 'world_state_signature' not in merged:
            merged['world_state_signature'] = str(obs.get('state') or obs.get('type') or 'unknown')
        object_bindings_summary = merged.get('object_bindings_summary', {}) if isinstance(merged.get('object_bindings_summary', {}), dict) else {}
        if not object_bindings_summary and build_object_bindings is not None:
            try:
                object_bindings_summary = dict(build_object_bindings(obs, merged) or {})
            except Exception:
                object_bindings_summary = {}
        task_frame_summary = merged.get('task_frame_summary', {}) if isinstance(merged.get('task_frame_summary', {}), dict) else {}
        if not task_frame_summary and infer_task_frame is not None:
            try:
                task_frame_summary = dict(infer_task_frame(obs, merged, object_bindings_summary, []) or {})
            except Exception:
                task_frame_summary = {}
        goal_hypotheses_summary = merged.get('goal_hypotheses_summary', []) if isinstance(merged.get('goal_hypotheses_summary', []), list) else []
        if not goal_hypotheses_summary and build_goal_hypotheses is not None:
            try:
                goal_hypotheses_summary = list(build_goal_hypotheses(obs, task_frame_summary, object_bindings_summary, []) or [])
            except Exception:
                goal_hypotheses_summary = []
        solver_state_summary = merged.get('solver_state_summary', {}) if isinstance(merged.get('solver_state_summary', {}), dict) else {}
        if not solver_state_summary and summarize_solver_state is not None:
            try:
                solver_state_summary = dict(summarize_solver_state(task_frame_summary, object_bindings_summary, goal_hypotheses_summary) or {})
            except Exception:
                solver_state_summary = {}
        mechanism_hypotheses_summary = merged.get('mechanism_hypotheses_summary', []) if isinstance(merged.get('mechanism_hypotheses_summary', []), list) else []
        if not mechanism_hypotheses_summary and build_mechanism_hypotheses is not None:
            try:
                mechanism_hypotheses_summary = list(build_mechanism_hypotheses(obs, task_frame_summary, object_bindings_summary, goal_hypotheses_summary, []) or [])
            except Exception:
                mechanism_hypotheses_summary = []
        mechanism_control_summary = merged.get('mechanism_control_summary', {}) if isinstance(merged.get('mechanism_control_summary', {}), dict) else {}
        if not mechanism_control_summary and summarize_mechanism_control is not None:
            try:
                mechanism_control_summary = dict(summarize_mechanism_control(mechanism_hypotheses_summary) or {})
            except Exception:
                mechanism_control_summary = {}
        if object_bindings_summary:
            merged['object_bindings_summary'] = object_bindings_summary
        if task_frame_summary:
            merged['task_frame_summary'] = task_frame_summary
        if goal_hypotheses_summary:
            merged['goal_hypotheses_summary'] = goal_hypotheses_summary
        if solver_state_summary:
            merged['solver_state_summary'] = solver_state_summary
        if mechanism_hypotheses_summary:
            merged['mechanism_hypotheses_summary'] = mechanism_hypotheses_summary
        if mechanism_control_summary:
            merged['mechanism_control_summary'] = mechanism_control_summary
        return merged

    def _recent_intervention_interactions(
        self,
        episode_trace: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in list(episode_trace or [])[-16:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
            intervention_target = meta.get('intervention_target', {}) if isinstance(meta.get('intervention_target', {}), dict) else {}
            rows.append({
                'anchor_ref': str(intervention_target.get('anchor_ref', '') or ''),
                'action_name': _canonicalize_function_name(tool_args.get('function_name', '')),
                'x': kwargs.get('x'),
                'y': kwargs.get('y'),
                'state_changed': bool(entry.get('state_changed', False) or entry.get('observation_changed', False)),
                'task_progressed': self._entry_has_positive_progress(entry),
                'reward': float(entry.get('reward', 0.0) or 0.0),
            })
        return rows

    def _build_procedure_candidates(
        self,
        obs: Dict[str, Any],
        available_functions: Sequence[str],
        episode_trace: Sequence[Dict[str, Any]],
        procedure_objects: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not procedure_objects:
            return []
        task_text = ' '.join(
            str(x).strip().lower() for x in (
                obs.get('task'),
                obs.get('goal'),
                obs.get('instruction'),
                obs.get('query'),
                (obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}).get('goal'),
                (obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}).get('summary'),
                (obs.get('world_state', {}) if isinstance(obs.get('world_state', {}), dict) else {}).get('task_family'),
                (obs.get('world_state', {}) if isinstance(obs.get('world_state', {}), dict) else {}).get('phase'),
            ) if str(x).strip()
        )
        task_tokens = set(_procedure_text_tokens(task_text))
        completed: List[str] = []
        for row in episode_trace[-6:]:
            fn = self._function_name_from_action(row.get('action', {}) if isinstance(row, dict) else {})
            if fn and fn != 'wait':
                completed.append(fn)

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for obj in procedure_objects:
            if not isinstance(obj, dict):
                continue
            content = obj.get('content', {})
            if not isinstance(content, dict):
                continue
            chain = content.get('action_chain', [])
            if not isinstance(chain, list) or len(chain) < 2:
                continue
            normalized_chain = [_canonicalize_function_name(fn) for fn in chain]
            if available_functions and not any(fn in available_functions for fn in normalized_chain):
                continue
            task_signature = str(content.get('task_signature', '')).lower()
            object_text = ' '.join(
                str(x).strip().lower()
                for x in (
                    content.get('task_signature', ''),
                    content.get('mechanism_summary', ''),
                    ' '.join(str(tag).strip() for tag in list(obj.get('retrieval_tags', []) or [])),
                    ' '.join(normalized_chain),
                    ' '.join(str(fn).strip() for fn in list(content.get('source_surface_functions', []) or [])),
                    ' '.join(str(fn).strip() for fn in list(content.get('target_surface_functions', []) or [])),
                )
                if str(x).strip()
            )
            object_tokens = set(_procedure_text_tokens(object_text))
            success_rate = float(content.get('success_rate', obj.get('confidence', 0.5)) or 0.0)
            failure_rate = float(content.get('failure_rate', max(0.0, 1.0 - success_rate)) or 0.0)
            text_match = 0.22 if task_signature and (task_signature in task_text or task_text in task_signature) else 0.0
            token_overlap = (
                float(len(task_tokens & object_tokens)) / float(max(1, len(task_tokens)))
                if task_tokens and object_tokens else 0.0
            )
            completion_match = 0.2 if completed and any(fn in completed for fn in normalized_chain[:-1]) else 0.0
            visible_overlap = (
                float(len(set(available_functions) & set(normalized_chain))) / float(max(1, len(set(available_functions))))
                if available_functions else 0.0
            )
            score = (
                (0.55 * success_rate)
                - (0.35 * failure_rate)
                + text_match
                + (token_overlap * 0.22)
                + (visible_overlap * 0.12)
                + completion_match
            )
            ranked.append((score, obj))

        candidates: List[Dict[str, Any]] = []
        for _score, obj in sorted(ranked, key=lambda x: x[0], reverse=True)[:2]:
            content = obj.get('content', {})
            chain = [_canonicalize_function_name(fn) for fn in list(content.get('action_chain', []))]
            if not chain or not available_functions:
                continue
            next_step_idx = 0
            for idx, fn in enumerate(chain):
                if fn in completed:
                    next_step_idx = min(len(chain) - 1, idx + 1)
            next_fn = chain[next_step_idx]
            if next_fn not in available_functions:
                continue
            candidate = self._make_call_action(next_fn, obs, None, episode_trace, None)
            candidate['_source'] = 'procedure_reuse'
            meta = candidate.get('_candidate_meta', {})
            meta['procedure'] = {
                'object_id': obj.get('object_id', ''),
                'task_signature': content.get('task_signature', ''),
                'action_chain': chain,
                'hit_source': 'object_store_procedure',
                'is_next_step': next_step_idx > 0,
                'success_rate': float(content.get('success_rate', obj.get('confidence', 0.5)) or 0.0),
                'failure_rate': float(content.get('failure_rate', 0.0) or 0.0),
                'procedure_bonus': float(content.get('procedure_bonus', 0.08) or 0.08),
            }
            candidate['_candidate_meta'] = meta
            candidates.append(candidate)
        return candidates

    def _build_latent_mechanism_candidates(
        self,
        obs: Dict[str, Any],
        available_functions: Sequence[str],
        episode_trace: Sequence[Dict[str, Any]],
        procedure_objects: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._latent_transfer_organs.build_candidates(
            obs=obs,
            available_functions=available_functions,
            episode_trace=episode_trace,
            procedure_objects=procedure_objects,
            make_call_action=self._make_call_action,
            function_name_from_action=self._function_name_from_action,
        )

    def _extract_env_available_functions(
        self,
        obs: Dict[str, Any],
    ) -> List[str]:
        functions: List[str] = []
        for fn in extract_surface_available_functions(obs):
            normalized_fn = _canonicalize_function_name(fn)
            if normalized_fn and normalized_fn not in functions:
                functions.append(normalized_fn)

        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw

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
                normalized_fn = _canonicalize_function_name(fn_name)
                if normalized_fn and normalized_fn not in functions:
                    functions.append(normalized_fn)

        if isinstance(api_raw, dict):
            _append_many(api_raw.get('available_functions', []))
            _append_many(api_raw.get('visible_functions', []))
            _append_many(api_raw.get('discovered_functions', []))

        _append_many(obs.get('available_functions', []))
        return functions

    def _extract_inferred_functions(
        self,
        surfaced: Sequence[Any],
        base_action: Optional[Dict[str, Any]],
        arm_action: Optional[Dict[str, Any]],
    ) -> List[str]:
        functions: List[str] = []

        for candidate in surfaced:
            fn = _canonicalize_function_name(_extract_candidate_function_name(getattr(candidate, 'object', {}) or {}))
            if fn and fn not in functions:
                functions.append(fn)

        for action in (base_action, arm_action):
            fn = _canonicalize_function_name(self._function_name_from_action(action))
            if fn and fn not in functions and fn != 'wait':
                functions.append(fn)

        return functions

    def _merge_function_sources(self, env_available_functions: Sequence[str], inferred_functions: Sequence[str]) -> List[str]:
        merged: List[str] = []
        for fn in list(env_available_functions) + list(inferred_functions):
            normalized_fn = _canonicalize_function_name(fn)
            if normalized_fn and normalized_fn not in merged:
                merged.append(normalized_fn)
        return merged

    def _build_retrieval_candidates(
        self,
        surfaced: Sequence[Any],
        obs: Dict[str, Any],
        plan_state: Any,
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for candidate in surfaced[:5]:
            obj = getattr(candidate, 'object', None) or {}
            fn = _canonicalize_function_name(_extract_candidate_function_name(obj))
            if not fn:
                continue
            score = None
            for attr_name in ('relevance_score', 'retrieval_score', 'score'):
                value = getattr(candidate, attr_name, None)
                if isinstance(value, (int, float)):
                    score = float(value)
                    break
            action = self._make_call_action(fn, obs, plan_state, episode_trace, world_model_summary, retrieved_obj=obj)
            action['_source'] = 'retrieval'
            base_meta = action.get('_candidate_meta', {})
            base_meta.update({
                'surfaced_from': [getattr(candidate, 'object_id', '')],
                'retrieval_score': score,
            })
            action['_candidate_meta'] = base_meta
            candidates.append(action)
        return candidates

    def _build_planner_candidate(self, plan_state: Any, env_available_functions: Sequence[str], obs: Dict[str, Any], episode_trace: Sequence[Dict[str, Any]], world_model_summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if plan_state is None or not getattr(plan_state, 'has_plan', False):
            return None

        target_function = _canonicalize_function_name(plan_state.get_target_function_for_step())
        step_intent = str(plan_state.get_intent_for_step() or '')
        if step_intent == 'wait':
            candidate = self._build_wait_candidate()
            candidate['_candidate_meta'] = {
                'planner_step_intent': step_intent,
                'planner_matches_step': True,
            }
            return self._with_source(candidate, 'planner')

        if not target_function or target_function == 'combine':
            return None
        planner_target_visible = target_function in env_available_functions
        if not planner_target_visible:
            raw_target = str(plan_state.get_target_function_for_step() or '')
            alt_target = _canonicalize_function_name(raw_target)
            if alt_target in env_available_functions:
                target_function = alt_target
                planner_target_visible = True
        if not planner_target_visible:
            return None

        candidate = self._make_call_action(target_function, obs, plan_state, episode_trace, world_model_summary)
        candidate['_source'] = 'planner'
        base_meta = candidate.get('_candidate_meta', {})
        base_meta.update({
            'planner_step_intent': step_intent,
            'planner_matches_step': True,
            'planner_target_function': target_function,
            'planner_target_visible': planner_target_visible,
            'planner_target_source': 'env_surface',
        })
        candidate['_candidate_meta'] = base_meta
        return candidate

    def _build_self_model_candidate(
        self,
        available_functions: Sequence[str],
        capability_profile: Any,
        reliability_tracker: Any,
        continuity_snapshot: Optional[Dict[str, Any]],
        episode_trace: Sequence[Dict[str, Any]],
        perception_summary: Optional[Dict[str, Any]],
        world_model_summary: Optional[Dict[str, Any]],
        obs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if capability_profile is None and reliability_tracker is None:
            fallback_functions = [fn for fn in available_functions if isinstance(fn, str) and fn and fn != 'wait']
            if not fallback_functions:
                return None

            tried_functions: List[str] = []
            for entry in reversed(list(episode_trace or [])):
                action = entry.get('action', {}) if isinstance(entry, dict) else {}
                fn = _canonicalize_function_name(self._function_name_from_action(action))
                if fn and fn not in tried_functions:
                    tried_functions.append(fn)

            untried_functions = [fn for fn in fallback_functions if fn not in tried_functions]
            best_fn = untried_functions[0] if untried_functions else fallback_functions[0]

            candidate = self._make_call_action(best_fn, obs, None, episode_trace, world_model_summary)
            candidate['_source'] = 'self_model'
            base_meta = candidate.get('_candidate_meta', {})
            base_meta.update({
                'capability_score': 0.0,
                'self_model_confidence': 0.0,
                'recent_negative_feedback': False,
                'perception_guard': False,
                'camera_relative_context': False,
                'cold_start_fallback': True,
                'selection_reason': 'no_capability_profile',
                'fallback_pool_size': len(fallback_functions),
                'recently_tried_count': len(tried_functions),
                'selected_untried_function': best_fn in untried_functions,
            })
            candidate['_candidate_meta'] = base_meta
            return candidate

        perception_summary = perception_summary or {}
        world_model_summary = world_model_summary or {}
        capability_context = {
            'task_family': continuity_snapshot.get('task_family', 'unknown') if isinstance(continuity_snapshot, dict) else 'unknown',
            'phase': (
                getattr(continuity_snapshot.get('top_goal'), 'goal_id', 'unknown')
                if isinstance(continuity_snapshot, dict) and continuity_snapshot.get('top_goal') is not None
                else 'unknown'
            ),
            'observation_mode': perception_summary.get('coordinate_type', 'unknown'),
            'resource_band': continuity_snapshot.get('resource_band', 'normal') if isinstance(continuity_snapshot, dict) else 'normal',
        }

        best_fn = None
        best_score = 0.0
        best_confidence = 0.0
        for fn in available_functions:
            if not fn or fn == 'wait':
                continue
            if capability_profile is not None and hasattr(capability_profile, 'get_capability_for_context'):
                cap = capability_profile.get_capability_for_context(fn, capability_context)
            else:
                cap = capability_profile.get_capability(fn) if capability_profile is not None and hasattr(capability_profile, 'get_capability') else None
            rel = reliability_tracker.get_action_type_reliability(fn) if reliability_tracker is not None and hasattr(reliability_tracker, 'get_action_type_reliability') else None
            success_rate = cap.success_rate if cap is not None else 0.0
            confidence = cap.confidence if cap is not None else 0.0
            reliability = rel.reliability_score if rel is not None else 0.0
            score = success_rate * 0.55 + confidence * 0.30 + reliability * 0.15
            if score > best_score:
                best_score = score
                best_confidence = min(1.0, confidence * 0.7 + reliability * 0.3)
                best_fn = fn

        recent_negative = any(float(entry.get('reward', 0.0) or 0.0) < 0.0 for entry in episode_trace[-3:])
        camera_relative = (
            perception_summary.get('coordinate_type') == 'camera_relative'
            and float(perception_summary.get('coordinate_confidence', 0.0) or 0.0) >= 0.55
        )
        belief_map = world_model_summary.get('beliefs', {}) if isinstance(world_model_summary, dict) else {}
        high_motion_belief = belief_map.get('observation_camera_motion', {}).get('posterior') == 'high_motion'
        needs_conservative_candidate = recent_negative or camera_relative or high_motion_belief

        if best_fn is None or (best_score <= 0.0 and not needs_conservative_candidate):
            return None

        candidate = self._make_call_action(best_fn, obs, None, episode_trace, world_model_summary)
        candidate['_source'] = 'self_model'
        base_meta = candidate.get('_candidate_meta', {})
        base_meta.update({
            'capability_score': best_score,
            'self_model_confidence': best_confidence,
            'recent_negative_feedback': recent_negative,
            'perception_guard': needs_conservative_candidate,
            'camera_relative_context': camera_relative,
            'capability_context': capability_context,
        })
        candidate['_candidate_meta'] = base_meta
        return candidate

    def _build_history_candidate(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        available_functions: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        for entry in reversed(list(episode_trace)):
            reward = float(entry.get('reward', 0.0) or 0.0)
            if reward <= 0.0:
                continue
            action = entry.get('action', {})
            fn = _canonicalize_function_name(self._function_name_from_action(action))
            if not fn or fn == 'wait':
                continue
            if available_functions and fn not in available_functions:
                continue
            candidate = self._with_source(action, 'history_reuse')
            meta = candidate.get('_candidate_meta', {})
            meta.update({
                'history_reward': reward,
                'history_tick': entry.get('tick'),
            })
            candidate['_candidate_meta'] = meta
            return candidate
        return None

    def _build_wait_candidate(self) -> Dict[str, Any]:
        return {
            'kind': 'wait',
            'payload': {'tool_args': {}},
            '_source': 'wait_fallback',
            '_candidate_meta': {'planner_matches_step': False},
        }

    def _build_surface_action_candidates(
        self,
        obs: Dict[str, Any],
        env_available_functions: Sequence[str],
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tried_counts: Dict[str, int] = {}
        for entry in episode_trace[-8:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            fn_name = _canonicalize_function_name(self._function_name_from_action(action))
            if fn_name:
                tried_counts[fn_name] = tried_counts.get(fn_name, 0) + 1

        ranked_functions = sorted(
            [_canonicalize_function_name(fn) for fn in env_available_functions if _canonicalize_function_name(fn) and _canonicalize_function_name(fn) != 'wait'],
            key=lambda fn: (
                int(tried_counts.get(fn, 0)),
                0 if fn == 'ACTION6' else 1,
                fn,
            ),
        )
        recovered: List[Dict[str, Any]] = []
        for fn in ranked_functions[:3]:
            if fn == 'ACTION6':
                click_specs = self._surface_click_specs(obs, episode_trace, world_model_summary)
                if not click_specs:
                    click_specs = [{
                        'kwargs': self._default_click_kwargs(obs, episode_trace, world_model_summary),
                        'role': 'default_click_fallback',
                        'reason': 'surface_generation_default_click',
                        'target_family': 'default_click',
                        'action_family': 'probe_state_transition',
                        'probe_aliases': ['probe_state_transition'],
                        'priority': 0.15,
                    }]
                for index, spec in enumerate(click_specs):
                    candidate = self._make_call_action(fn, obs, None, episode_trace, world_model_summary, kwargs=dict(spec.get('kwargs', {}) or {}))
                    candidate['_source'] = 'surface_generation'
                    meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
                    meta['surface_generation'] = True
                    meta['surface_visible_functions'] = list(env_available_functions)
                    meta['surface_try_count'] = int(tried_counts.get(fn, 0))
                    meta['surface_click_candidate'] = True
                    meta['surface_click_rank'] = int(index)
                    meta['surface_click_role'] = str(spec.get('role', '') or '')
                    meta['surface_click_reason'] = str(spec.get('reason', '') or '')
                    meta['target_family'] = str(spec.get('target_family', '') or '')
                    meta['action_family'] = str(spec.get('action_family', 'probe_state_transition') or 'probe_state_transition')
                    meta['probe_candidate'] = bool(spec.get('probe_candidate', True))
                    meta['discriminating_candidate'] = bool(spec.get('discriminating_candidate', False))
                    meta['world_model_probe_aliases'] = [str(item) for item in list(spec.get('probe_aliases', []) or []) if str(item)]
                    meta['surface_click_priority'] = float(spec.get('priority', 0.0) or 0.0)
                    meta['explicit_perception_target'] = bool(spec.get('explicit_perception_target', False))
                    meta['surface_diversity_bonus'] = float(spec.get('surface_diversity_bonus', 0.0) or 0.0)
                    meta['surface_diversity_novel_color'] = bool(spec.get('surface_diversity_novel_color', False))
                    meta['surface_diversity_novel_shape'] = bool(spec.get('surface_diversity_novel_shape', False))
                    meta['surface_diversity_repeat_penalty'] = float(spec.get('surface_diversity_repeat_penalty', 0.0) or 0.0)
                    meta['family_effect_bonus'] = float(spec.get('family_effect_bonus', 0.0) or 0.0)
                    meta['family_effect_preference'] = str(spec.get('family_effect_preference', '') or '')
                    meta['family_effect_clicked_family_match'] = float(spec.get('family_effect_clicked_family_match', 0.0) or 0.0)
                    meta['family_effect_supported_family_match'] = float(spec.get('family_effect_supported_family_match', 0.0) or 0.0)
                    meta['family_effect_same_family_bias_applied'] = bool(spec.get('family_effect_same_family_bias_applied', False))
                    meta['family_effect_other_family_bias_applied'] = bool(spec.get('family_effect_other_family_bias_applied', False))
                    meta['goal_progress_bonus'] = float(spec.get('goal_progress_bonus', 0.0) or 0.0)
                    meta['goal_progress_goal_anchor_match'] = bool(spec.get('goal_progress_goal_anchor_match', False))
                    meta['goal_progress_goal_color_match'] = bool(spec.get('goal_progress_goal_color_match', False))
                    meta['goal_progress_novel_goal_anchor'] = bool(spec.get('goal_progress_novel_goal_anchor', False))
                    meta['goal_progress_stalled_goal_anchor'] = bool(spec.get('goal_progress_stalled_goal_anchor', False))
                    meta['goal_progress_local_only_anchor'] = bool(spec.get('goal_progress_local_only_anchor', False))
                    meta['goal_progress_controller_anchor'] = bool(spec.get('goal_progress_controller_anchor', False))
                    meta['goal_progress_controller_anchor_evidence'] = float(
                        spec.get('goal_progress_controller_anchor_evidence', 0.0) or 0.0
                    )
                    meta['goal_progress_anchor_state_relevance'] = float(
                        spec.get('goal_progress_anchor_state_relevance', 0.0) or 0.0
                    )
                    meta['goal_progress_anchor_goal_proximity'] = float(
                        spec.get('goal_progress_anchor_goal_proximity', 0.0) or 0.0
                    )
                    meta['goal_progress_preferred_next_goal_rank'] = int(
                        spec.get('goal_progress_preferred_next_goal_rank', 0) or 0
                    )
                    meta['goal_progress_preferred_next_goal_color_match'] = bool(
                        spec.get('goal_progress_preferred_next_goal_color_match', False)
                    )
                    meta['goal_progress_gap_closing_preferred_goal'] = bool(
                        spec.get('goal_progress_gap_closing_preferred_goal', False)
                    )
                    meta['goal_progress_gap_closing_preferred_goal_rank'] = int(
                        spec.get('goal_progress_gap_closing_preferred_goal_rank', 0) or 0
                    )
                    meta['goal_progress_relation_anchor_match'] = bool(
                        spec.get('goal_progress_relation_anchor_match', False)
                    )
                    meta['goal_progress_relation_type'] = str(
                        spec.get('goal_progress_relation_type', '') or ''
                    )
                    meta['goal_progress_relation_target'] = str(
                        spec.get('goal_progress_relation_target', '') or ''
                    )
                    meta['goal_progress_relation_anchor_progress'] = float(
                        spec.get('goal_progress_relation_anchor_progress', 0.0) or 0.0
                    )
                    meta['goal_progress_gap_closing_relation_anchor'] = bool(
                        spec.get('goal_progress_gap_closing_relation_anchor', False)
                    )
                    meta['goal_progress_gap_closing_relation_anchor_rank'] = int(
                        spec.get('goal_progress_gap_closing_relation_anchor_rank', 0) or 0
                    )
                    meta['goal_progress_controller_supported_goal_anchor'] = bool(
                        spec.get('goal_progress_controller_supported_goal_anchor', False)
                    )
                    meta['goal_bundle_bonus'] = float(spec.get('goal_bundle_bonus', 0.0) or 0.0)
                    meta['goal_bundle_is_combo_complement'] = bool(spec.get('goal_bundle_is_combo_complement', False))
                    meta['goal_bundle_is_necessary_but_insufficient'] = bool(spec.get('goal_bundle_is_necessary_but_insufficient', False))
                    meta['goal_bundle_is_distinct_from_combo_seed'] = bool(spec.get('goal_bundle_is_distinct_from_combo_seed', False))
                    if spec.get('anchor_ref'):
                        meta['anchor_ref'] = str(spec.get('anchor_ref'))
                        meta['intervention_target'] = {
                            'anchor_ref': str(spec.get('anchor_ref')),
                            'target_kind': str(spec.get('target_family', spec.get('role', 'surface_click')) or 'surface_click'),
                        }
                    if spec.get('object_color') is not None:
                        meta['object_color'] = int(spec.get('object_color'))
                    if spec.get('world_model_required_probe_match') is not None:
                        meta['world_model_required_probe_match'] = bool(spec.get('world_model_required_probe_match'))
                    candidate['_candidate_meta'] = meta
                    recovered.append(candidate)
                continue

            candidate = self._make_call_action(fn, obs, None, episode_trace, world_model_summary, kwargs=None)
            candidate['_source'] = 'surface_generation'
            meta = candidate.get('_candidate_meta', {})
            meta['surface_generation'] = True
            meta['surface_visible_functions'] = list(env_available_functions)
            meta['surface_try_count'] = int(tried_counts.get(fn, 0))
            candidate['_candidate_meta'] = meta
            recovered.append(candidate)
        return recovered

    def _extract_recent_click_points(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        *,
        limit: int = 6,
    ) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            if _canonicalize_function_name(self._function_name_from_action(action)) != 'ACTION6':
                continue
            payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            if kwargs.get('x') is None or kwargs.get('y') is None:
                continue
            try:
                point = (int(kwargs.get('x')), int(kwargs.get('y')))
            except (TypeError, ValueError):
                continue
            if point not in points:
                points.append(point)
            if len(points) >= max(1, int(limit)):
                break
        return points


    def _extract_recent_click_feedback(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        feedback: List[Dict[str, Any]] = []
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            if _canonicalize_function_name(self._function_name_from_action(action)) != 'ACTION6':
                continue
            payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            if kwargs.get('x') is None or kwargs.get('y') is None:
                continue
            try:
                point = (int(kwargs.get('x')), int(kwargs.get('y')))
            except (TypeError, ValueError):
                continue
            reward = float(entry.get('reward', 0.0) or 0.0)
            info_gain = float(entry.get('information_gain', 0.0) or 0.0)
            positive = self._entry_has_positive_progress(entry)
            failure_reason = self._entry_failure_reason(entry)
            schema_failure = self._is_schema_failure_reason(failure_reason)
            failed = bool(not positive and reward <= 0.0 and info_gain <= 0.12 and not schema_failure)
            feedback.append({
                'point': point,
                'reward': reward,
                'information_gain': info_gain,
                'positive': bool(positive),
                'failed': failed,
                'schema_failure': bool(schema_failure),
                'failure_reason': failure_reason,
                'tick': int(entry.get('tick', -1) or -1),
            })
            if len(feedback) >= max(1, int(limit)):
                break
        return feedback

    def _normalized_object_bbox(self, raw_bbox: Dict[str, Any]) -> Dict[str, int]:
        bbox = raw_bbox if isinstance(raw_bbox, dict) else {}
        x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
        x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
        y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
        y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
        width = int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1))
        height = int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1))
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': max(1, width),
            'height': max(1, height),
        }

    def _surface_object_shape_labels(self, obj: Dict[str, Any]) -> List[str]:
        semantic_rows = list(obj.get('semantic_candidates', []) or []) if isinstance(obj.get('semantic_candidates', []), list) else []
        labels: List[str] = []
        for row in semantic_rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get('label', '') or '').strip()
            if label and label not in labels:
                labels.append(label)
        if labels:
            return labels

        bbox = self._normalized_object_bbox(obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {})
        features = obj.get('geometric_features', {}) if isinstance(obj.get('geometric_features', {}), dict) else {}
        width = int(features.get('width', bbox.get('width', 0)) or bbox.get('width', 0) or 0)
        height = int(features.get('height', bbox.get('height', 0)) or bbox.get('height', 0) or 0)
        area = int(obj.get('area', features.get('area', width * height)) or 0)
        fill_ratio = float(features.get('fill_ratio', area / float(max(width * height, 1))) or 0.0)
        aspect_ratio = float(features.get('aspect_ratio', width / float(max(height, 1))) or 1.0)
        boundary_contact = bool(obj.get('boundary_contact', features.get('boundary_contact', False)))

        if area <= 4:
            labels.append('token_like')
        if width > 0 and height > 0 and fill_ratio >= 0.70 and abs(width - height) <= 1:
            labels.append('block_like')
        if width > 0 and height > 0 and max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6)) >= 2.0 and fill_ratio >= 0.45:
            labels.append('bar_like')
        if boundary_contact:
            labels.append('boundary_structure')
        if not labels:
            labels.append('generic_object')
        return labels

    def _surface_object_descriptors(
        self,
        perception: Dict[str, Any],
        bindings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        descriptors: List[Dict[str, Any]] = []
        seen_keys = set()
        salient_objects = list(perception.get('salient_objects', []) or []) if isinstance(perception.get('salient_objects', []), list) else []
        for obj in salient_objects:
            if not isinstance(obj, dict):
                continue
            bbox = self._normalized_object_bbox(obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {})
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            anchor_ref = str(obj.get('object_id', '') or '')
            color = obj.get('color')
            key = (
                anchor_ref,
                int(color) if color is not None else None,
                bbox.get('x_min', 0),
                bbox.get('y_min', 0),
                bbox.get('x_max', 0),
                bbox.get('y_max', 0),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            descriptors.append({
                'anchor_ref': anchor_ref,
                'color': int(color) if color is not None else None,
                'bbox': bbox,
                'centroid': {
                    'x': float(centroid.get('x', (bbox['x_min'] + bbox['x_max']) / 2.0) or 0.0),
                    'y': float(centroid.get('y', (bbox['y_min'] + bbox['y_max']) / 2.0) or 0.0),
                },
                'shape_labels': self._surface_object_shape_labels(obj),
                'boundary_contact': bool(obj.get('boundary_contact', False)),
            })

        binding_objects = list(bindings.get('objects', []) or []) if isinstance(bindings.get('objects', []), list) else []
        for obj in binding_objects:
            if not isinstance(obj, dict):
                continue
            bbox = self._normalized_object_bbox(obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {})
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            anchor_ref = str(obj.get('object_id', '') or '')
            color = obj.get('color')
            key = (
                anchor_ref,
                int(color) if color is not None else None,
                bbox.get('x_min', 0),
                bbox.get('y_min', 0),
                bbox.get('x_max', 0),
                bbox.get('y_max', 0),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            descriptors.append({
                'anchor_ref': anchor_ref,
                'color': int(color) if color is not None else None,
                'bbox': bbox,
                'centroid': {
                    'x': float(centroid.get('x', (bbox['x_min'] + bbox['x_max']) / 2.0) or 0.0),
                    'y': float(centroid.get('y', (bbox['y_min'] + bbox['y_max']) / 2.0) or 0.0),
                },
                'shape_labels': self._surface_object_shape_labels(obj),
                'boundary_contact': bool((obj.get('geometric_features', {}) or {}).get('boundary_contact', False)),
            })
        return descriptors

    def _match_click_to_surface_descriptor(
        self,
        descriptors: Sequence[Dict[str, Any]],
        point: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if not descriptors:
            return None
        px, py = int(point[0]), int(point[1])
        containing: List[Tuple[int, float, Dict[str, Any]]] = []
        nearest: List[Tuple[float, int, Dict[str, Any]]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            bbox = descriptor.get('bbox', {}) if isinstance(descriptor.get('bbox', {}), dict) else {}
            x_min = int(bbox.get('x_min', 0) or 0)
            x_max = int(bbox.get('x_max', x_min) or x_min)
            y_min = int(bbox.get('y_min', 0) or 0)
            y_max = int(bbox.get('y_max', y_min) or y_min)
            width = int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1))
            height = int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1))
            area = max(1, width * height)
            centroid = descriptor.get('centroid', {}) if isinstance(descriptor.get('centroid', {}), dict) else {}
            cx = float(centroid.get('x', (x_min + x_max) / 2.0) or 0.0)
            cy = float(centroid.get('y', (y_min + y_max) / 2.0) or 0.0)
            distance = abs(cx - float(px)) + abs(cy - float(py))
            if x_min <= px <= x_max and y_min <= py <= y_max:
                containing.append((area, distance, descriptor))
            else:
                nearest.append((distance, area, descriptor))
        if containing:
            containing.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            return containing[0][2]
        if nearest and nearest[0][0] <= 2.5:
            nearest.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            return nearest[0][2]
        return None

    def _extract_recent_click_object_coverage(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        *,
        limit: int = 8,
    ) -> Dict[str, Any]:
        covered_colors = set()
        covered_shapes = set()
        covered_anchor_refs = set()
        matched_click_count = 0
        action6_click_count = 0
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            if _canonicalize_function_name(self._function_name_from_action(action)) != 'ACTION6':
                continue
            action6_click_count += 1
            payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            if kwargs.get('x') is None or kwargs.get('y') is None:
                continue
            try:
                point = (int(kwargs.get('x')), int(kwargs.get('y')))
            except (TypeError, ValueError):
                continue
            observation = entry.get('observation', {}) if isinstance(entry.get('observation', {}), dict) else {}
            perception = observation.get('perception', {}) if isinstance(observation.get('perception', {}), dict) else {}
            if not perception:
                continue
            bindings: Dict[str, Any] = {}
            if build_object_bindings is not None:
                try:
                    bindings = dict(build_object_bindings(observation, {}) or {})
                except Exception:
                    bindings = {}
            descriptors = self._surface_object_descriptors(perception, bindings)
            matched = self._match_click_to_surface_descriptor(descriptors, point)
            if not matched:
                continue
            matched_click_count += 1
            if matched.get('color') is not None:
                covered_colors.add(int(matched.get('color')))
            for label in list(matched.get('shape_labels', []) or []):
                if str(label):
                    covered_shapes.add(str(label))
            if str(matched.get('anchor_ref', '') or ''):
                covered_anchor_refs.add(str(matched.get('anchor_ref')))
            if matched_click_count >= max(1, int(limit)):
                break
        return {
            'covered_colors': covered_colors,
            'covered_shapes': covered_shapes,
            'covered_anchor_refs': covered_anchor_refs,
            'matched_click_count': matched_click_count,
            'action6_click_count': action6_click_count,
        }

    def _surface_diversity_priority_adjustment(
        self,
        *,
        descriptor: Optional[Dict[str, Any]],
        role: str,
        action6_click_count: int,
        covered_colors: set,
        covered_shapes: set,
        covered_anchor_refs: set,
        uncovered_available_colors: set,
        uncovered_available_shapes: set,
    ) -> Dict[str, Any]:
        normalized_role = str(role or '').strip().lower()
        if action6_click_count >= 5:
            return {'bonus': 0.0, 'novel_color': False, 'novel_shape': False, 'repeat_penalty': 0.0}
        if not descriptor or not isinstance(descriptor, dict):
            return {'bonus': 0.0, 'novel_color': False, 'novel_shape': False, 'repeat_penalty': 0.0}
        if 'neighbor' in normalized_role or normalized_role in {
            'changed_hotspot',
            'suggested_hotspot',
            'failed_click_neighbor_search',
            'hotspot_neighbor_search',
            'counterfactual_background',
            'background_control',
            'grid_center',
        }:
            return {'bonus': 0.0, 'novel_color': False, 'novel_shape': False, 'repeat_penalty': 0.0}

        bonus = 0.0
        repeat_penalty = 0.0
        color = descriptor.get('color')
        shape_labels = [str(label) for label in list(descriptor.get('shape_labels', []) or []) if str(label)]
        novel_color = bool(color is not None and color not in covered_colors)
        novel_shape = bool(shape_labels and any(label not in covered_shapes for label in shape_labels))
        anchor_ref = str(descriptor.get('anchor_ref', '') or '')

        if uncovered_available_colors and color is not None:
            if novel_color:
                bonus += 0.055
            else:
                repeat_penalty += 0.03
        if uncovered_available_shapes and shape_labels:
            if novel_shape:
                bonus += 0.04
            else:
                repeat_penalty += 0.02
        if anchor_ref and anchor_ref in covered_anchor_refs:
            repeat_penalty += 0.02

        net_bonus = max(-0.08, min(0.10, bonus - repeat_penalty))
        return {
            'bonus': round(net_bonus, 4),
            'novel_color': novel_color,
            'novel_shape': novel_shape,
            'repeat_penalty': round(repeat_penalty, 4),
        }

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _family_summary_from_descriptor(
        self,
        descriptor: Optional[Dict[str, Any]],
        *,
        role: str,
        target_family: str,
        anchor_ref: str = '',
        object_color: Any = None,
    ) -> Dict[str, Any]:
        family = {
            'anchor_ref': str(anchor_ref or ''),
            'color': self._safe_int(object_color),
            'shape_labels': [],
            'boundary_contact': False,
            'target_family': str(target_family or ''),
            'surface_click_role': str(role or ''),
        }
        if isinstance(descriptor, dict):
            if str(descriptor.get('anchor_ref', '') or ''):
                family['anchor_ref'] = str(descriptor.get('anchor_ref', '') or family['anchor_ref'])
            if descriptor.get('color') is not None:
                family['color'] = self._safe_int(descriptor.get('color'))
            family['shape_labels'] = [str(label) for label in list(descriptor.get('shape_labels', []) or []) if str(label)]
            family['boundary_contact'] = bool(descriptor.get('boundary_contact', family['boundary_contact']))
        return family

    def _family_match_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0
        score = 0.0
        left_anchor = str(left.get('anchor_ref', '') or '')
        right_anchor = str(right.get('anchor_ref', '') or '')
        if left_anchor and right_anchor and left_anchor == right_anchor:
            score += 1.15

        left_color = self._safe_int(left.get('color'))
        right_color = self._safe_int(right.get('color'))
        if left_color is not None and right_color is not None:
            if left_color == right_color:
                score += 0.90
            else:
                score -= 0.20

        left_shapes = {str(item) for item in list(left.get('shape_labels', []) or []) if str(item)}
        right_shapes = {str(item) for item in list(right.get('shape_labels', []) or []) if str(item)}
        if left_shapes and right_shapes:
            overlap = len(left_shapes & right_shapes) / float(max(len(left_shapes | right_shapes), 1))
            score += 0.42 * overlap

        if str(left.get('target_family', '') or '') and str(left.get('target_family', '') or '') == str(right.get('target_family', '') or ''):
            score += 0.18
        if bool(left.get('boundary_contact', False)) and bool(right.get('boundary_contact', False)):
            score += 0.08
        return max(0.0, float(score))

    def _extract_recent_family_effect_feedback(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        feedback: List[Dict[str, Any]] = []
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            if _canonicalize_function_name(self._function_name_from_action(action)) != 'ACTION6':
                continue
            attribution = entry.get('family_effect_attribution', {}) if isinstance(entry.get('family_effect_attribution', {}), dict) else {}
            preference = str(attribution.get('preference', '') or '').strip().lower()
            if preference not in {'same_family', 'other_family'}:
                continue
            information_gain = float(attribution.get('information_gain', entry.get('information_gain', 0.0)) or 0.0)
            changed_pixels = float(attribution.get('changed_pixel_count', 0.0) or 0.0)
            positive_progress = bool(attribution.get('positive_progress', False) or self._entry_has_positive_progress(entry))
            if not positive_progress and information_gain < 0.08 and changed_pixels < 4.0:
                continue
            clicked_family = attribution.get('clicked_family', entry.get('clicked_family', {}))
            clicked_family = clicked_family if isinstance(clicked_family, dict) else {}
            supported_families = [
                item
                for item in list(attribution.get('supported_families', []) or [])
                if isinstance(item, dict)
            ]
            feedback.append({
                'preference': preference,
                'clicked_family': clicked_family,
                'supported_families': supported_families,
                'information_gain': information_gain,
                'changed_pixel_count': changed_pixels,
                'positive_progress': positive_progress,
                'tick': int(entry.get('tick', -1) or -1),
            })
            if len(feedback) >= max(1, int(limit)):
                break
        return feedback

    def _family_effect_priority_adjustment(
        self,
        *,
        descriptor: Optional[Dict[str, Any]],
        role: str,
        target_family: str,
        anchor_ref: str,
        object_color: Any,
        recent_feedback: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_role = str(role or '').strip().lower()
        if not recent_feedback:
            return {
                'bonus': 0.0,
                'preference': '',
                'clicked_family_match': 0.0,
                'supported_family_match': 0.0,
                'same_family_bias_applied': False,
                'other_family_bias_applied': False,
            }
        if 'neighbor' in normalized_role or normalized_role in {
            'changed_hotspot',
            'suggested_hotspot',
            'failed_click_neighbor_search',
            'hotspot_neighbor_search',
            'counterfactual_background',
            'background_control',
            'grid_center',
        }:
            return {
                'bonus': 0.0,
                'preference': '',
                'clicked_family_match': 0.0,
                'supported_family_match': 0.0,
                'same_family_bias_applied': False,
                'other_family_bias_applied': False,
            }

        family = self._family_summary_from_descriptor(
            descriptor,
            role=role,
            target_family=target_family,
            anchor_ref=anchor_ref,
            object_color=object_color,
        )
        if not any([
            family.get('anchor_ref'),
            family.get('color') is not None,
            bool(family.get('shape_labels')),
            family.get('target_family'),
        ]):
            return {
                'bonus': 0.0,
                'preference': '',
                'clicked_family_match': 0.0,
                'supported_family_match': 0.0,
                'same_family_bias_applied': False,
                'other_family_bias_applied': False,
            }

        total_bonus = 0.0
        best_clicked_match = 0.0
        best_supported_match = 0.0
        strongest_preference = ''
        same_family_bias_applied = False
        other_family_bias_applied = False

        for index, feedback in enumerate(recent_feedback):
            if not isinstance(feedback, dict):
                continue
            preference = str(feedback.get('preference', '') or '')
            if preference not in {'same_family', 'other_family'}:
                continue
            clicked_family = feedback.get('clicked_family', {}) if isinstance(feedback.get('clicked_family', {}), dict) else {}
            supported_families = [
                item for item in list(feedback.get('supported_families', []) or []) if isinstance(item, dict)
            ]
            clicked_match = self._family_match_score(family, clicked_family)
            supported_match = max((self._family_match_score(family, item) for item in supported_families), default=0.0)
            best_clicked_match = max(best_clicked_match, clicked_match)
            best_supported_match = max(best_supported_match, supported_match)
            recency_weight = max(0.45, 1.0 - index * 0.18)
            signal_weight = max(
                0.65 if bool(feedback.get('positive_progress', False)) else 0.45,
                min(1.0, float(feedback.get('information_gain', 0.0) or 0.0) / 0.35) if float(feedback.get('information_gain', 0.0) or 0.0) > 0.0 else 0.0,
                min(1.0, float(feedback.get('changed_pixel_count', 0.0) or 0.0) / 32.0) if float(feedback.get('changed_pixel_count', 0.0) or 0.0) > 0.0 else 0.0,
            )
            if preference == 'same_family':
                if clicked_match >= 0.60:
                    total_bonus += recency_weight * signal_weight * 0.09 * min(1.0, clicked_match)
                    strongest_preference = preference
                    same_family_bias_applied = True
            elif preference == 'other_family':
                if supported_match >= 0.60:
                    total_bonus += recency_weight * signal_weight * 0.10 * min(1.0, supported_match)
                    strongest_preference = preference
                    other_family_bias_applied = True
                if clicked_match >= 0.60:
                    total_bonus -= recency_weight * signal_weight * 0.08 * min(1.0, clicked_match)
                    strongest_preference = preference
                    other_family_bias_applied = True

        return {
            'bonus': round(max(-0.12, min(0.12, total_bonus)), 4),
            'preference': strongest_preference,
            'clicked_family_match': round(float(best_clicked_match), 4),
            'supported_family_match': round(float(best_supported_match), 4),
            'same_family_bias_applied': same_family_bias_applied,
            'other_family_bias_applied': other_family_bias_applied,
        }

    def _extract_recent_goal_progress_feedback(
        self,
        episode_trace: Sequence[Dict[str, Any]],
        *,
        limit: int = 8,
    ) -> Dict[str, Any]:
        engaged_goal_anchor_refs = set()
        engaged_goal_colors = set()
        stalled_anchor_refs = set()
        successful_anchor_refs = set()
        necessary_anchor_refs = set()
        necessary_but_insufficient_anchor_refs = set()
        local_only_anchor_refs = set()
        complementary_goal_anchor_refs = set()
        controller_anchor_refs = set()
        controller_supported_goal_anchor_refs = set()
        controller_evidence_by_anchor: Dict[str, float] = {}
        state_relevance_by_anchor: Dict[str, float] = {}
        goal_proximity_by_anchor: Dict[str, float] = {}
        relation_engaged_anchor_refs = set()
        relation_goal_progress_anchor_refs = set()
        relation_progress_by_anchor: Dict[str, float] = {}
        latest_relation_summary: Dict[str, Any] = {}
        active_combo_seed_anchor = ''
        requires_multi_anchor_coordination = False
        latest_goal_summary: Dict[str, Any] = {}
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            if not latest_goal_summary:
                goal_summary = entry.get('inferred_level_goal', {}) if isinstance(entry.get('inferred_level_goal', {}), dict) else {}
                if goal_summary:
                    latest_goal_summary = dict(goal_summary)
            assessment = entry.get('goal_progress_assessment', {}) if isinstance(entry.get('goal_progress_assessment', {}), dict) else {}
            bundle_state = entry.get('goal_bundle_state', {}) if isinstance(entry.get('goal_bundle_state', {}), dict) else {}
            for ref in list(assessment.get('engaged_goal_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    engaged_goal_anchor_refs.add(text)
            for ref in list(bundle_state.get('engaged_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    engaged_goal_anchor_refs.add(text)
            for color in list((entry.get('inferred_level_goal', {}) if isinstance(entry.get('inferred_level_goal', {}), dict) else {}).get('goal_anchor_colors', []) or []):
                color_int = self._safe_int(color)
                if color_int is not None:
                    engaged_goal_colors.add(color_int)
            clicked_anchor_ref = str(assessment.get('clicked_anchor_ref', '') or '')
            controller_anchor_ref = str(
                assessment.get('controller_anchor_ref', '') or clicked_anchor_ref
            )
            state_relevance_score = float(assessment.get('state_relevance_score', 0.0) or 0.0)
            goal_proximity_score = float(assessment.get('goal_proximity_score', 0.0) or 0.0)
            controller_evidence_score = float(assessment.get('controller_evidence_score', 0.0) or 0.0)
            relation_progress_score = float(assessment.get('relation_progress_score', 0.0) or 0.0)
            if clicked_anchor_ref:
                state_relevance_by_anchor[clicked_anchor_ref] = max(
                    float(state_relevance_by_anchor.get(clicked_anchor_ref, 0.0) or 0.0),
                    state_relevance_score,
                )
                goal_proximity_by_anchor[clicked_anchor_ref] = max(
                    float(goal_proximity_by_anchor.get(clicked_anchor_ref, 0.0) or 0.0),
                    goal_proximity_score,
                )
            if bool(assessment.get('stalled', False)) and clicked_anchor_ref:
                stalled_anchor_refs.add(clicked_anchor_ref)
            if bool(assessment.get('progressed', False)) and clicked_anchor_ref:
                successful_anchor_refs.add(clicked_anchor_ref)
            if bool(assessment.get('controller_effect', False)):
                if controller_anchor_ref:
                    controller_anchor_refs.add(controller_anchor_ref)
                    controller_evidence_by_anchor[controller_anchor_ref] = max(
                        float(controller_evidence_by_anchor.get(controller_anchor_ref, 0.0) or 0.0),
                        controller_evidence_score,
                    )
                    goal_proximity_by_anchor[controller_anchor_ref] = max(
                        float(goal_proximity_by_anchor.get(controller_anchor_ref, 0.0) or 0.0),
                        goal_proximity_score,
                    )
            relation_type = str(assessment.get('relation_type', '') or '').strip()
            relation_target = str(assessment.get('relation_target', '') or '').strip()
            relation_grouping_basis = str(assessment.get('relation_grouping_basis', '') or '').strip()
            relation_member_anchor_refs = [
                str(ref or '').strip()
                for ref in list(assessment.get('relation_member_anchor_refs', []) or [])
                if str(ref or '').strip()
            ]
            relation_engaged_refs = [
                str(ref or '').strip()
                for ref in list(assessment.get('relation_engaged_anchor_refs', []) or [])
                if str(ref or '').strip()
            ]
            if relation_type and relation_member_anchor_refs and not latest_relation_summary:
                latest_relation_summary = {
                    'relation_type': relation_type,
                    'relation_target': relation_target,
                    'relation_grouping_basis': relation_grouping_basis,
                    'relation_member_anchor_refs': relation_member_anchor_refs,
                }
            for ref in relation_engaged_refs:
                relation_engaged_anchor_refs.add(ref)
                relation_progress_by_anchor[ref] = max(
                    float(relation_progress_by_anchor.get(ref, 0.0) or 0.0),
                    relation_progress_score,
                )
                if bool(assessment.get('relation_goal_progress', False)):
                    relation_goal_progress_anchor_refs.add(ref)
            if bool(assessment.get('necessary_signal', False)) and clicked_anchor_ref:
                necessary_anchor_refs.add(clicked_anchor_ref)
            if bool(assessment.get('necessary_but_insufficient', False)) and clicked_anchor_ref:
                necessary_but_insufficient_anchor_refs.add(clicked_anchor_ref)
                if not active_combo_seed_anchor:
                    active_combo_seed_anchor = clicked_anchor_ref
            if bool(assessment.get('local_only_signal', False)) and clicked_anchor_ref:
                local_only_anchor_refs.add(clicked_anchor_ref)
            for ref in list(bundle_state.get('necessary_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    necessary_anchor_refs.add(text)
            for ref in list(bundle_state.get('necessary_but_insufficient_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    necessary_but_insufficient_anchor_refs.add(text)
                    if not active_combo_seed_anchor:
                        active_combo_seed_anchor = text
            for ref in list(bundle_state.get('local_only_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    local_only_anchor_refs.add(text)
            for ref in list(bundle_state.get('complementary_goal_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    complementary_goal_anchor_refs.add(text)
            for ref in list(assessment.get('controller_supported_goal_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    controller_supported_goal_anchor_refs.add(text)
            for ref in list(bundle_state.get('controller_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    controller_anchor_refs.add(text)
            for ref in list(bundle_state.get('controller_supported_goal_anchor_refs', []) or []):
                text = str(ref or '').strip()
                if text:
                    controller_supported_goal_anchor_refs.add(text)
            if bool(bundle_state.get('requires_multi_anchor_coordination', False)):
                requires_multi_anchor_coordination = True
            if not active_combo_seed_anchor and str(bundle_state.get('active_combo_seed_anchor', '') or ''):
                active_combo_seed_anchor = str(bundle_state.get('active_combo_seed_anchor', '') or '')
            if len(engaged_goal_anchor_refs) >= max(1, int(limit)):
                break
        return {
            'engaged_goal_anchor_refs': engaged_goal_anchor_refs,
            'engaged_goal_colors': engaged_goal_colors,
            'stalled_anchor_refs': stalled_anchor_refs,
            'successful_anchor_refs': successful_anchor_refs,
            'necessary_anchor_refs': necessary_anchor_refs,
            'necessary_but_insufficient_anchor_refs': necessary_but_insufficient_anchor_refs,
            'local_only_anchor_refs': local_only_anchor_refs,
            'complementary_goal_anchor_refs': complementary_goal_anchor_refs,
            'controller_anchor_refs': controller_anchor_refs,
            'controller_supported_goal_anchor_refs': controller_supported_goal_anchor_refs,
            'controller_evidence_by_anchor': controller_evidence_by_anchor,
            'state_relevance_by_anchor': state_relevance_by_anchor,
            'goal_proximity_by_anchor': goal_proximity_by_anchor,
            'relation_engaged_anchor_refs': relation_engaged_anchor_refs,
            'relation_goal_progress_anchor_refs': relation_goal_progress_anchor_refs,
            'relation_progress_by_anchor': relation_progress_by_anchor,
            'latest_relation_summary': latest_relation_summary,
            'active_combo_seed_anchor': active_combo_seed_anchor,
            'requires_multi_anchor_coordination': bool(requires_multi_anchor_coordination or bool(necessary_but_insufficient_anchor_refs)),
            'latest_goal_summary': latest_goal_summary,
        }

    def _goal_bundle_priority_adjustment(
        self,
        *,
        descriptor: Optional[Dict[str, Any]],
        role: str,
        anchor_ref: str,
        recent_goal_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_role = str(role or '').strip().lower()
        if 'neighbor' in normalized_role or normalized_role in {
            'changed_hotspot',
            'suggested_hotspot',
            'failed_click_neighbor_search',
            'hotspot_neighbor_search',
            'counterfactual_background',
            'background_control',
            'grid_center',
        }:
            return {
                'bonus': 0.0,
                'is_combo_complement': False,
                'is_necessary_but_insufficient': False,
                'is_distinct_from_combo_seed': False,
            }
        if not bool(recent_goal_feedback.get('requires_multi_anchor_coordination', False)):
            return {
                'bonus': 0.0,
                'is_combo_complement': False,
                'is_necessary_but_insufficient': False,
                'is_distinct_from_combo_seed': False,
            }

        descriptor_anchor = str(descriptor.get('anchor_ref', '') or '') if isinstance(descriptor, dict) else ''
        anchor = str(anchor_ref or descriptor_anchor or '').strip()
        if not anchor:
            return {
                'bonus': 0.0,
                'is_combo_complement': False,
                'is_necessary_but_insufficient': False,
                'is_distinct_from_combo_seed': False,
            }

        necessary_but_insufficient = set(recent_goal_feedback.get('necessary_but_insufficient_anchor_refs', set()) or set())
        engaged_goal_anchor_refs = set(recent_goal_feedback.get('engaged_goal_anchor_refs', set()) or set())
        complementary_goal_anchor_refs = set(recent_goal_feedback.get('complementary_goal_anchor_refs', set()) or set())
        controller_supported_goal_anchor_refs = set(
            recent_goal_feedback.get('controller_supported_goal_anchor_refs', set()) or set()
        )
        active_combo_seed_anchor = str(recent_goal_feedback.get('active_combo_seed_anchor', '') or '')
        latest_goal_summary = recent_goal_feedback.get('latest_goal_summary', {}) if isinstance(recent_goal_feedback.get('latest_goal_summary', {}), dict) else {}
        goal_anchor_refs = {
            str(ref or '').strip()
            for ref in list(latest_goal_summary.get('goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(latest_goal_summary.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        if not active_combo_seed_anchor:
            goal_summary_controller_refs = [
                str(ref or '').strip()
                for ref in list(latest_goal_summary.get('controller_anchor_refs', []) or [])
                if str(ref or '').strip()
            ]
            if goal_summary_controller_refs:
                active_combo_seed_anchor = goal_summary_controller_refs[0]

        if not active_combo_seed_anchor and not necessary_but_insufficient:
            return {
                'bonus': 0.0,
                'is_combo_complement': False,
                'is_necessary_but_insufficient': False,
                'is_distinct_from_combo_seed': False,
            }

        if goal_anchor_refs and anchor not in goal_anchor_refs and anchor not in necessary_but_insufficient:
            return {
                'bonus': 0.0,
                'is_combo_complement': False,
                'is_necessary_but_insufficient': False,
                'is_distinct_from_combo_seed': False,
            }

        is_necessary_but_insufficient = bool(anchor in necessary_but_insufficient)
        explicit_complement = bool(
            anchor in complementary_goal_anchor_refs
            or anchor in controller_supported_goal_anchor_refs
        )
        is_combo_complement = bool(
            active_combo_seed_anchor
            and anchor != active_combo_seed_anchor
            and anchor not in necessary_but_insufficient
            and (
                explicit_complement
                or (
                    anchor not in engaged_goal_anchor_refs
                    and bool(goal_anchor_refs)
                    and anchor in goal_anchor_refs
                )
            )
        )
        is_distinct_from_combo_seed = bool(active_combo_seed_anchor and anchor != active_combo_seed_anchor)

        bonus = 0.0
        if is_necessary_but_insufficient:
            bonus -= 0.16
        elif is_combo_complement:
            bonus += 0.12

        return {
            'bonus': round(max(-0.18, min(0.18, bonus)), 4),
            'is_combo_complement': is_combo_complement,
            'is_necessary_but_insufficient': is_necessary_but_insufficient,
            'is_distinct_from_combo_seed': is_distinct_from_combo_seed,
        }

    def _goal_progress_priority_adjustment(
        self,
        *,
        descriptor: Optional[Dict[str, Any]],
        role: str,
        anchor_ref: str,
        object_color: Any,
        action6_click_count: int,
        task_frame_summary: Dict[str, Any],
        recent_goal_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_role = str(role or '').strip().lower()
        if 'neighbor' in normalized_role or normalized_role in {
            'changed_hotspot',
            'suggested_hotspot',
            'failed_click_neighbor_search',
            'hotspot_neighbor_search',
            'counterfactual_background',
            'background_control',
            'grid_center',
        }:
            return {
                'bonus': 0.0,
                'goal_anchor_match': False,
                'goal_color_match': False,
                'novel_goal_anchor': False,
                'stalled_goal_anchor': False,
                'local_only_anchor': False,
            }

        goal_summary = (
            task_frame_summary.get('inferred_level_goal', {})
            if isinstance(task_frame_summary.get('inferred_level_goal', {}), dict)
            else {}
        )
        if not goal_summary and isinstance(recent_goal_feedback.get('latest_goal_summary', {}), dict):
            goal_summary = dict(recent_goal_feedback.get('latest_goal_summary', {}) or {})
        if not goal_summary:
            return {
                'bonus': 0.0,
                'goal_anchor_match': False,
                'goal_color_match': False,
                'novel_goal_anchor': False,
                'stalled_goal_anchor': False,
                'local_only_anchor': False,
            }

        goal_anchor_refs = {
            str(ref or '').strip()
            for ref in list(goal_summary.get('goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        goal_anchor_colors = {
            self._safe_int(color)
            for color in list(goal_summary.get('goal_anchor_colors', []) or [])
            if self._safe_int(color) is not None
        }
        controller_supported_goal_colors = {
            self._safe_int(color)
            for color in list(goal_summary.get('controller_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        confidence = float(goal_summary.get('confidence', 0.0) or 0.0)
        mechanism_prior_confidence = float(goal_summary.get('mechanism_prior_confidence', 0.0) or 0.0)
        mechanism_prior_hints = (
            goal_summary.get('mechanism_prior_strategy_hints', {})
            if isinstance(goal_summary.get('mechanism_prior_strategy_hints', {}), dict)
            else {}
        )
        mechanism_prior_supported_goal_colors = {
            self._safe_int(color)
            for color in list(goal_summary.get('mechanism_prior_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        mechanism_prior_supported_goal_colors |= {
            self._safe_int(color)
            for color in list(mechanism_prior_hints.get('supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        mechanism_prior_supported_goal_anchor_refs = {
            str(ref or '').strip()
            for ref in list(goal_summary.get('mechanism_prior_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        preferred_next_goal_anchor_refs = [
            str(ref or '').strip()
            for ref in list(goal_summary.get('preferred_next_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        ]
        preferred_next_goal_colors = {
            self._safe_int(color)
            for color in list(goal_summary.get('preferred_next_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        latest_relation_summary = (
            recent_goal_feedback.get('latest_relation_summary', {})
            if isinstance(recent_goal_feedback.get('latest_relation_summary', {}), dict)
            else {}
        )
        relation_hypotheses = [
            dict(item)
            for item in list(goal_summary.get('relation_hypotheses', []) or [])
            if isinstance(item, dict)
        ]
        if not relation_hypotheses and latest_relation_summary:
            relation_hypotheses = [dict(latest_relation_summary)]
        top_relation = relation_hypotheses[0] if relation_hypotheses else {}
        relation_type = str(top_relation.get('relation_type', '') or '')
        relation_target = str(
            top_relation.get('target_relation', top_relation.get('relation_target', '')) or ''
        )
        relation_member_anchor_refs = [
            str(ref or '').strip()
            for ref in list(
                top_relation.get(
                    'member_anchor_refs',
                    top_relation.get('relation_member_anchor_refs', []),
                ) or []
            )
            if str(ref or '').strip()
        ]
        relation_confidence = float(top_relation.get('confidence', 0.0) or 0.0)
        mechanism_prior_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(mechanism_prior_hints.get('supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        mechanism_prior_supporting_functions = {
            _canonicalize_function_name(item)
            for item in list(goal_summary.get('mechanism_prior_supporting_functions', []) or [])
            if _canonicalize_function_name(item)
        }
        mechanism_prior_supporting_functions |= {
            _canonicalize_function_name(item)
            for item in list(mechanism_prior_hints.get('supporting_functions', []) or [])
            if _canonicalize_function_name(item)
        }
        mechanism_prior_expand = bool(
            mechanism_prior_hints.get('controller_support_expected', False)
            and str(mechanism_prior_hints.get('preferred_progress_mode', '') or '') == 'expand_anchor_coverage'
        )
        mechanism_prior_function_match = bool(
            not mechanism_prior_supporting_functions
            or 'ACTION6' in mechanism_prior_supporting_functions
        )
        activation_click_floor = 6
        if confidence >= 0.82 and len(goal_anchor_refs) >= 2:
            activation_click_floor = 2
        elif confidence >= 0.72 and len(goal_anchor_refs) >= 2:
            activation_click_floor = 4
        if mechanism_prior_expand and mechanism_prior_confidence >= 0.6 and len(goal_anchor_refs) >= 2:
            activation_click_floor = max(2, activation_click_floor - 2)
        if int(action6_click_count or 0) < activation_click_floor:
            return {
                'bonus': 0.0,
                'goal_anchor_match': False,
                'goal_color_match': False,
                'novel_goal_anchor': False,
                'stalled_goal_anchor': False,
                'local_only_anchor': False,
            }
        preferred_progress_mode = str(goal_summary.get('preferred_progress_mode', '') or '')
        descriptor_anchor = str(descriptor.get('anchor_ref', '') or '') if isinstance(descriptor, dict) else ''
        anchor = str(anchor_ref or descriptor_anchor or '').strip()
        color = self._safe_int(object_color if object_color is not None else (descriptor.get('color') if isinstance(descriptor, dict) else None))
        engaged_goal_anchor_refs = set(recent_goal_feedback.get('engaged_goal_anchor_refs', set()) or set())
        stalled_anchor_refs = set(recent_goal_feedback.get('stalled_anchor_refs', set()) or set())
        successful_anchor_refs = set(recent_goal_feedback.get('successful_anchor_refs', set()) or set())
        engaged_goal_colors = set(recent_goal_feedback.get('engaged_goal_colors', set()) or set())
        local_only_anchor_refs = set(recent_goal_feedback.get('local_only_anchor_refs', set()) or set())
        controller_anchor_refs = set(recent_goal_feedback.get('controller_anchor_refs', set()) or set())
        controller_supported_goal_anchor_refs = set(
            recent_goal_feedback.get('controller_supported_goal_anchor_refs', set()) or set()
        )
        controller_evidence_by_anchor = (
            recent_goal_feedback.get('controller_evidence_by_anchor', {})
            if isinstance(recent_goal_feedback.get('controller_evidence_by_anchor', {}), dict)
            else {}
        )
        state_relevance_by_anchor = (
            recent_goal_feedback.get('state_relevance_by_anchor', {})
            if isinstance(recent_goal_feedback.get('state_relevance_by_anchor', {}), dict)
            else {}
        )
        goal_proximity_by_anchor = (
            recent_goal_feedback.get('goal_proximity_by_anchor', {})
            if isinstance(recent_goal_feedback.get('goal_proximity_by_anchor', {}), dict)
            else {}
        )
        relation_engaged_anchor_refs = set(
            recent_goal_feedback.get('relation_engaged_anchor_refs', set()) or set()
        )
        relation_goal_progress_anchor_refs = set(
            recent_goal_feedback.get('relation_goal_progress_anchor_refs', set()) or set()
        )
        relation_progress_by_anchor = (
            recent_goal_feedback.get('relation_progress_by_anchor', {})
            if isinstance(recent_goal_feedback.get('relation_progress_by_anchor', {}), dict)
            else {}
        )
        controller_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        unengaged_goal_anchor_refs = set(goal_anchor_refs) - set(engaged_goal_anchor_refs)
        current_goal_evidence = bool((engaged_goal_anchor_refs | stalled_anchor_refs | successful_anchor_refs) & goal_anchor_refs)

        goal_anchor_match = bool(anchor and anchor in goal_anchor_refs)
        goal_color_match = bool(color is not None and color in goal_anchor_colors)
        novel_goal_anchor = bool(goal_anchor_match and anchor not in engaged_goal_anchor_refs)
        stalled_goal_anchor = bool(goal_anchor_match and anchor in stalled_anchor_refs)
        local_only_anchor = bool(anchor and anchor in local_only_anchor_refs)
        controller_anchor = bool(anchor and anchor in controller_anchor_refs)
        controller_supported_goal_anchor = bool(
            anchor and anchor in controller_supported_goal_anchor_refs
        )
        controller_anchor_evidence = float(controller_evidence_by_anchor.get(anchor, 0.0) or 0.0) if anchor else 0.0
        anchor_state_relevance = float(state_relevance_by_anchor.get(anchor, 0.0) or 0.0) if anchor else 0.0
        anchor_goal_proximity = float(goal_proximity_by_anchor.get(anchor, 0.0) or 0.0) if anchor else 0.0
        preferred_next_goal_rank = (
            preferred_next_goal_anchor_refs.index(anchor) + 1
            if anchor and anchor in preferred_next_goal_anchor_refs
            else 0
        )
        missing_preferred_goal_anchor_refs = [
            ref
            for ref in preferred_next_goal_anchor_refs
            if ref and ref not in successful_anchor_refs
        ]
        gap_closing_preferred_goal_ref = (
            missing_preferred_goal_anchor_refs[0]
            if missing_preferred_goal_anchor_refs
            else ''
        )
        gap_closing_preferred_goal_rank = (
            preferred_next_goal_anchor_refs.index(gap_closing_preferred_goal_ref) + 1
            if gap_closing_preferred_goal_ref
            else 0
        )
        gap_closing_preferred_goal = bool(
            anchor and gap_closing_preferred_goal_ref and anchor == gap_closing_preferred_goal_ref
        )
        preferred_next_goal_color_match = bool(
            color is not None and color in preferred_next_goal_colors
        )
        relation_anchor_match = bool(anchor and anchor in relation_member_anchor_refs)
        relation_anchor_progress = (
            float(relation_progress_by_anchor.get(anchor, 0.0) or 0.0)
            if anchor
            else 0.0
        )
        pending_relation_anchor_refs = [
            ref
            for ref in relation_member_anchor_refs
            if ref and ref not in successful_anchor_refs
        ]
        gap_closing_relation_anchor_ref = (
            pending_relation_anchor_refs[0]
            if pending_relation_anchor_refs
            else ''
        )
        gap_closing_relation_anchor_rank = (
            relation_member_anchor_refs.index(gap_closing_relation_anchor_ref) + 1
            if gap_closing_relation_anchor_ref
            else 0
        )
        gap_closing_relation_anchor = bool(
            anchor and gap_closing_relation_anchor_ref and anchor == gap_closing_relation_anchor_ref
        )
        mechanism_prior_anchor_match = bool(
            anchor and anchor in mechanism_prior_supported_goal_anchor_refs
        )
        mechanism_prior_color_match = bool(
            color is not None and color in mechanism_prior_supported_goal_colors
        )

        bonus = 0.0
        if goal_anchor_match:
            bonus += 0.02 + min(0.035, confidence * 0.035)
        elif goal_color_match and color not in engaged_goal_colors:
            bonus += 0.008 + min(0.018, confidence * 0.02)
        elif (
            goal_anchor_refs
            and confidence >= 0.84
            and preferred_progress_mode == 'expand_anchor_coverage'
            and not controller_anchor
        ):
            bonus -= 0.012 + min(0.014, confidence * 0.015)

        if current_goal_evidence:
            if local_only_anchor:
                bonus -= 0.09 + min(0.05, confidence * 0.06)
            elif novel_goal_anchor:
                bonus += 0.075 + min(0.075, confidence * 0.08)
            elif stalled_goal_anchor:
                bonus -= 0.095 + min(0.055, confidence * 0.06)
            elif controller_anchor and anchor in successful_anchor_refs:
                bonus += 0.006 + min(0.012, confidence * 0.012)
                bonus += min(0.045, controller_anchor_evidence * 0.05)
            elif goal_anchor_match and preferred_progress_mode == 'intensify_single_anchor' and anchor in successful_anchor_refs:
                bonus += 0.02 + min(0.03, confidence * 0.03)
            elif goal_anchor_match and preferred_progress_mode == 'expand_anchor_coverage' and unengaged_goal_anchor_refs:
                bonus -= 0.04 + min(0.03, confidence * 0.03)
            elif goal_color_match and color not in engaged_goal_colors:
                bonus += 0.02 + min(0.025, confidence * 0.03)
            if controller_supported_goal_anchor:
                bonus += min(0.05, anchor_goal_proximity * 0.05)
            if preferred_next_goal_rank:
                bonus += max(0.012, 0.04 - (preferred_next_goal_rank - 1) * 0.011)
                if gap_closing_preferred_goal:
                    bonus += 0.028 + min(0.03, confidence * 0.03)
                elif (
                    gap_closing_preferred_goal_ref
                    and preferred_next_goal_rank > gap_closing_preferred_goal_rank
                ):
                    bonus -= 0.012 + min(0.015, confidence * 0.016)
            elif preferred_next_goal_color_match and controller_supported_goal_colors:
                bonus += 0.008 + min(0.012, confidence * 0.012)
            elif (
                preferred_progress_mode == 'expand_anchor_coverage'
                and unengaged_goal_anchor_refs
                and not controller_anchor
            ):
                bonus -= 0.05 + min(0.04, confidence * 0.05)
            if relation_anchor_match:
                bonus += 0.008 + min(0.014, relation_confidence * 0.02)
                bonus += min(0.045, relation_anchor_progress * 0.045)
                if gap_closing_relation_anchor:
                    bonus += 0.018 + min(0.024, relation_confidence * 0.03)
                elif (
                    gap_closing_relation_anchor_ref
                    and gap_closing_relation_anchor_rank
                    and anchor
                    and anchor in relation_member_anchor_refs
                    and relation_member_anchor_refs.index(anchor) + 1 > gap_closing_relation_anchor_rank
                ):
                    bonus -= 0.008 + min(0.012, relation_confidence * 0.014)
            elif (
                relation_member_anchor_refs
                and relation_goal_progress_anchor_refs
                and not controller_anchor
                and not goal_anchor_match
            ):
                bonus -= 0.008 + min(0.012, relation_confidence * 0.014)
            if (
                mechanism_prior_expand
                and mechanism_prior_function_match
                and goal_anchor_match
                and anchor in unengaged_goal_anchor_refs
            ):
                bonus += 0.014 + min(0.02, mechanism_prior_confidence * 0.025)
        else:
            if local_only_anchor:
                bonus -= 0.04 + min(0.025, confidence * 0.03)
            elif novel_goal_anchor:
                bonus += 0.014 + min(0.02, confidence * 0.025)
            elif controller_anchor and anchor in successful_anchor_refs:
                bonus += 0.004 + min(0.008, confidence * 0.008)
                bonus += min(0.04, controller_anchor_evidence * 0.045)
            elif goal_color_match and color not in engaged_goal_colors:
                bonus += 0.01 + min(0.012, confidence * 0.02)
            if controller_supported_goal_anchor:
                bonus += min(0.045, anchor_goal_proximity * 0.045)
            if preferred_next_goal_rank:
                bonus += max(0.01, 0.034 - (preferred_next_goal_rank - 1) * 0.01)
                if gap_closing_preferred_goal:
                    bonus += 0.024 + min(0.026, confidence * 0.026)
                elif (
                    gap_closing_preferred_goal_ref
                    and preferred_next_goal_rank > gap_closing_preferred_goal_rank
                ):
                    bonus -= 0.01 + min(0.012, confidence * 0.014)
            elif preferred_next_goal_color_match and controller_supported_goal_colors:
                bonus += 0.006 + min(0.01, confidence * 0.01)
            if relation_anchor_match:
                bonus += 0.006 + min(0.012, relation_confidence * 0.016)
                bonus += min(0.04, relation_anchor_progress * 0.04)
                if gap_closing_relation_anchor:
                    bonus += 0.016 + min(0.02, relation_confidence * 0.024)
                elif (
                    gap_closing_relation_anchor_ref
                    and gap_closing_relation_anchor_rank
                    and anchor
                    and anchor in relation_member_anchor_refs
                    and relation_member_anchor_refs.index(anchor) + 1 > gap_closing_relation_anchor_rank
                ):
                    bonus -= 0.006 + min(0.01, relation_confidence * 0.012)
            elif (
                relation_member_anchor_refs
                and relation_goal_progress_anchor_refs
                and not controller_anchor
                and not goal_anchor_match
            ):
                bonus -= 0.006 + min(0.01, relation_confidence * 0.012)
            if mechanism_prior_expand:
                if mechanism_prior_function_match and (goal_anchor_match or mechanism_prior_anchor_match):
                    bonus += 0.012 + min(0.018, mechanism_prior_confidence * 0.02)
                elif mechanism_prior_function_match and mechanism_prior_color_match:
                    bonus += 0.016 + min(0.022, mechanism_prior_confidence * 0.024)
                elif not controller_anchor:
                    bonus -= 0.008 + min(0.012, mechanism_prior_confidence * 0.015)
            if (
                mechanism_prior_expand
                and mechanism_prior_function_match
                and mechanism_prior_supported_goal_colors
                and not mechanism_prior_color_match
                and not goal_anchor_match
                and not controller_anchor
            ):
                bonus -= 0.01 + min(0.014, mechanism_prior_confidence * 0.016)

        return {
            'bonus': round(max(-0.18, min(0.18, bonus)), 4),
            'goal_anchor_match': goal_anchor_match,
            'goal_color_match': goal_color_match,
            'novel_goal_anchor': novel_goal_anchor,
            'stalled_goal_anchor': stalled_goal_anchor,
            'local_only_anchor': local_only_anchor,
            'controller_anchor': controller_anchor,
            'controller_supported_goal_anchor': controller_supported_goal_anchor,
            'controller_anchor_evidence': round(float(controller_anchor_evidence), 4),
            'anchor_state_relevance': round(float(anchor_state_relevance), 4),
            'anchor_goal_proximity': round(float(anchor_goal_proximity), 4),
            'preferred_next_goal_rank': int(preferred_next_goal_rank),
            'preferred_next_goal_color_match': preferred_next_goal_color_match,
            'gap_closing_preferred_goal': gap_closing_preferred_goal,
            'gap_closing_preferred_goal_rank': int(gap_closing_preferred_goal_rank),
            'relation_anchor_match': relation_anchor_match,
            'relation_type': relation_type,
            'relation_target': relation_target,
            'relation_anchor_progress': round(float(relation_anchor_progress), 4),
            'gap_closing_relation_anchor': gap_closing_relation_anchor,
            'gap_closing_relation_anchor_rank': int(gap_closing_relation_anchor_rank),
            'mechanism_prior_anchor_match': mechanism_prior_anchor_match,
            'mechanism_prior_color_match': mechanism_prior_color_match,
            'mechanism_prior_function_match': mechanism_prior_function_match,
        }

    def _expand_click_neighbors(
        self,
        point: Tuple[int, int],
        *,
        width: int,
        height: int,
        radii: Sequence[int] = (1, 2, 4),
    ) -> List[Tuple[int, int]]:
        if width <= 0 or height <= 0:
            return []
        x0, y0 = int(point[0]), int(point[1])
        neighbors: List[Tuple[int, int]] = []
        for radius in radii:
            r = max(1, int(radius))
            ring = [
                (x0 + r, y0),
                (x0 - r, y0),
                (x0, y0 + r),
                (x0, y0 - r),
                (x0 + r, y0 + r),
                (x0 + r, y0 - r),
                (x0 - r, y0 + r),
                (x0 - r, y0 - r),
            ]
            for x, y in ring:
                x = max(0, min(width - 1, int(x)))
                y = max(0, min(height - 1, int(y)))
                point_xy = (x, y)
                if point_xy == (x0, y0):
                    continue
                if point_xy not in neighbors:
                    neighbors.append(point_xy)
        return neighbors

    def _surface_click_specs(
        self,
        obs: Dict[str, Any],
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
        grid_shape = perception.get('grid_shape', {}) if isinstance(perception.get('grid_shape', {}), dict) else {}
        width = int(grid_shape.get('width', 0) or 0)
        height = int(grid_shape.get('height', 0) or 0)
        recent_clicks = self._extract_recent_click_points(episode_trace)
        recent_click_set = set(recent_clicks)
        recent_click_feedback = self._extract_recent_click_feedback(episode_trace)
        failed_click_feedback = [item for item in recent_click_feedback if bool(item.get('failed'))]
        schema_failed_click_feedback = [item for item in recent_click_feedback if bool(item.get('schema_failure'))]
        successful_click_feedback = [item for item in recent_click_feedback if bool(item.get('positive'))]
        failed_click_points = [tuple(item.get('point', ())) for item in failed_click_feedback if isinstance(item.get('point'), tuple)]
        schema_failed_click_points = [tuple(item.get('point', ())) for item in schema_failed_click_feedback if isinstance(item.get('point'), tuple)]
        exact_failed_point_set = set(failed_click_points[:3])
        exact_schema_failed_point_set = set(schema_failed_click_points[:3])
        wm_hints = world_model_summary.get('action_kwargs_hints', {}) if isinstance(world_model_summary, dict) else {}
        click_hints = wm_hints.get('ACTION6', {}) if isinstance(wm_hints.get('ACTION6', {}), dict) else {}
        mechanism_control = world_model_summary.get('mechanism_control_summary', {}) if isinstance(world_model_summary.get('mechanism_control_summary', {}), dict) else {}
        required_probes = [str(item) for item in list(mechanism_control.get('required_probes', []) or []) if str(item)]
        preferred_target_refs = [str(item) for item in list(mechanism_control.get('preferred_target_refs', []) or []) if str(item)]
        preferred_target_ref_set = set(preferred_target_refs)
        bindings = world_model_summary.get('object_bindings_summary', {}) if isinstance(world_model_summary.get('object_bindings_summary', {}), dict) else {}
        if not bindings and build_object_bindings is not None:
            try:
                bindings = dict(build_object_bindings(obs, world_model_summary) or {})
            except Exception:
                bindings = {}
        task_frame_summary = world_model_summary.get('task_frame_summary', {}) if isinstance(world_model_summary.get('task_frame_summary', {}), dict) else {}
        if not task_frame_summary and infer_task_frame is not None:
            try:
                task_frame_summary = dict(infer_task_frame(obs, world_model_summary, bindings, list(episode_trace or [])[-8:]) or {})
            except Exception:
                task_frame_summary = {}
        current_descriptors = self._surface_object_descriptors(perception, bindings)
        descriptor_by_anchor = {
            str(item.get('anchor_ref', '') or ''): item
            for item in current_descriptors
            if str(item.get('anchor_ref', '') or '')
        }
        coverage_state = self._extract_recent_click_object_coverage(episode_trace)
        recent_family_feedback = self._extract_recent_family_effect_feedback(episode_trace)
        covered_colors = set(coverage_state.get('covered_colors', set()) or set())
        covered_shapes = set(coverage_state.get('covered_shapes', set()) or set())
        covered_anchor_refs = set(coverage_state.get('covered_anchor_refs', set()) or set())
        action6_click_count = int(coverage_state.get('action6_click_count', 0) or 0)
        recent_goal_feedback = self._extract_recent_goal_progress_feedback(episode_trace)
        available_colors = {
            int(item.get('color'))
            for item in current_descriptors
            if isinstance(item, dict) and item.get('color') is not None
        }
        available_shapes = {
            str(label)
            for item in current_descriptors
            for label in list(item.get('shape_labels', []) or [])
            if isinstance(item, dict) and str(label)
        }
        uncovered_available_colors = available_colors - covered_colors
        uncovered_available_shapes = available_shapes - covered_shapes

        specs: List[Dict[str, Any]] = []
        seen_points: set[Tuple[int, int]] = set()
        explicit_perception_targets = [item for item in list(perception.get('suggested_click_targets', []) or []) if isinstance(item, dict)]
        has_salient_center_target = any(
            str(item.get('role', '') or '') == 'salient_object_center'
            or str(item.get('target_family', '') or '') == 'salient_object'
            for item in explicit_perception_targets
        )
        early_single_failed_click = bool(
            len(recent_click_feedback) == 1
            and len(failed_click_feedback) == 1
            and not successful_click_feedback
            and int(failed_click_feedback[0].get('tick', -1) or -1) <= 0
        )
        prefer_salient_over_failed_neighbor = bool(has_salient_center_target and early_single_failed_click)

        def add_spec(
            x: Any,
            y: Any,
            *,
            role: str,
            reason: str,
            priority: float,
            target_family: str,
            action_family: str = 'probe_state_transition',
            anchor_ref: str = '',
            object_color: Any = None,
            probe_aliases: Optional[Sequence[str]] = None,
            probe_candidate: bool = True,
            discriminating_candidate: bool = False,
        ) -> None:
            try:
                px = int(x)
                py = int(y)
            except (TypeError, ValueError):
                return
            if width > 0:
                px = max(0, min(width - 1, px))
            if height > 0:
                py = max(0, min(height - 1, py))
            point = (px, py)
            if point in seen_points:
                return
            if point in exact_failed_point_set and role not in {'failed_click_retry', 'failed_click_neighbor_search'}:
                return
            if point in exact_schema_failed_point_set and role not in {'schema_failure_neighbor_search'}:
                return
            seen_points.add(point)
            probe_alias_list = [str(item) for item in list(probe_aliases or []) if str(item)]
            if action_family and action_family not in probe_alias_list:
                probe_alias_list.append(str(action_family))
            adjusted_priority = float(priority)
            if point in recent_click_set:
                adjusted_priority -= 0.22
            if point in exact_failed_point_set:
                adjusted_priority -= 0.45
            if point in exact_schema_failed_point_set:
                adjusted_priority -= 0.75
            required_match = bool(set(probe_alias_list) & set(required_probes)) if required_probes else False
            if required_match:
                adjusted_priority += 0.18
            if anchor_ref and anchor_ref in preferred_target_ref_set:
                adjusted_priority += 0.12
            descriptor = descriptor_by_anchor.get(str(anchor_ref or '')) if anchor_ref else None
            if descriptor is None:
                descriptor = self._match_click_to_surface_descriptor(current_descriptors, point)
            coverage_adjustment = self._surface_diversity_priority_adjustment(
                descriptor=descriptor,
                role=role,
                action6_click_count=action6_click_count,
                covered_colors=covered_colors,
                covered_shapes=covered_shapes,
                covered_anchor_refs=covered_anchor_refs,
                uncovered_available_colors=uncovered_available_colors,
                uncovered_available_shapes=uncovered_available_shapes,
            )
            adjusted_priority += float(coverage_adjustment.get('bonus', 0.0) or 0.0)
            family_effect_adjustment = self._family_effect_priority_adjustment(
                descriptor=descriptor,
                role=role,
                target_family=target_family,
                anchor_ref=str(anchor_ref or ''),
                object_color=object_color,
                recent_feedback=recent_family_feedback,
            )
            adjusted_priority += float(family_effect_adjustment.get('bonus', 0.0) or 0.0)
            goal_progress_adjustment = self._goal_progress_priority_adjustment(
                descriptor=descriptor,
                role=role,
                anchor_ref=str(anchor_ref or ''),
                object_color=object_color,
                action6_click_count=action6_click_count,
                task_frame_summary=task_frame_summary,
                recent_goal_feedback=recent_goal_feedback,
            )
            adjusted_priority += float(goal_progress_adjustment.get('bonus', 0.0) or 0.0)
            goal_bundle_adjustment = self._goal_bundle_priority_adjustment(
                descriptor=descriptor,
                role=role,
                anchor_ref=str(anchor_ref or ''),
                recent_goal_feedback=recent_goal_feedback,
            )
            adjusted_priority += float(goal_bundle_adjustment.get('bonus', 0.0) or 0.0)
            specs.append({
                'kwargs': {'x': px, 'y': py},
                'role': str(role),
                'reason': str(reason),
                'priority': round(adjusted_priority, 4),
                'target_family': str(target_family),
                'action_family': str(action_family or 'probe_state_transition'),
                'anchor_ref': str(anchor_ref or ''),
                'object_color': int(object_color) if object_color is not None else None,
                'probe_aliases': probe_alias_list,
                'probe_candidate': bool(probe_candidate),
                'discriminating_candidate': bool(discriminating_candidate or required_match),
                'world_model_required_probe_match': bool(required_match),
                'surface_diversity_bonus': float(coverage_adjustment.get('bonus', 0.0) or 0.0),
                'surface_diversity_novel_color': bool(coverage_adjustment.get('novel_color', False)),
                'surface_diversity_novel_shape': bool(coverage_adjustment.get('novel_shape', False)),
                'surface_diversity_repeat_penalty': float(coverage_adjustment.get('repeat_penalty', 0.0) or 0.0),
                'family_effect_bonus': float(family_effect_adjustment.get('bonus', 0.0) or 0.0),
                'family_effect_preference': str(family_effect_adjustment.get('preference', '') or ''),
                'family_effect_clicked_family_match': float(family_effect_adjustment.get('clicked_family_match', 0.0) or 0.0),
                'family_effect_supported_family_match': float(family_effect_adjustment.get('supported_family_match', 0.0) or 0.0),
                'family_effect_same_family_bias_applied': bool(family_effect_adjustment.get('same_family_bias_applied', False)),
                'family_effect_other_family_bias_applied': bool(family_effect_adjustment.get('other_family_bias_applied', False)),
                'goal_progress_bonus': float(goal_progress_adjustment.get('bonus', 0.0) or 0.0),
                'goal_progress_goal_anchor_match': bool(goal_progress_adjustment.get('goal_anchor_match', False)),
                'goal_progress_goal_color_match': bool(goal_progress_adjustment.get('goal_color_match', False)),
                'goal_progress_novel_goal_anchor': bool(goal_progress_adjustment.get('novel_goal_anchor', False)),
                'goal_progress_stalled_goal_anchor': bool(goal_progress_adjustment.get('stalled_goal_anchor', False)),
                'goal_progress_local_only_anchor': bool(goal_progress_adjustment.get('local_only_anchor', False)),
                'goal_progress_controller_anchor': bool(goal_progress_adjustment.get('controller_anchor', False)),
                'goal_progress_controller_anchor_evidence': float(
                    goal_progress_adjustment.get('controller_anchor_evidence', 0.0) or 0.0
                ),
                'goal_progress_anchor_state_relevance': float(
                    goal_progress_adjustment.get('anchor_state_relevance', 0.0) or 0.0
                ),
                'goal_progress_anchor_goal_proximity': float(
                    goal_progress_adjustment.get('anchor_goal_proximity', 0.0) or 0.0
                ),
                'goal_progress_preferred_next_goal_rank': int(
                    goal_progress_adjustment.get('preferred_next_goal_rank', 0) or 0
                ),
                'goal_progress_preferred_next_goal_color_match': bool(
                    goal_progress_adjustment.get('preferred_next_goal_color_match', False)
                ),
                'goal_progress_gap_closing_preferred_goal': bool(
                    goal_progress_adjustment.get('gap_closing_preferred_goal', False)
                ),
                'goal_progress_gap_closing_preferred_goal_rank': int(
                    goal_progress_adjustment.get('gap_closing_preferred_goal_rank', 0) or 0
                ),
                'goal_progress_relation_anchor_match': bool(
                    goal_progress_adjustment.get('relation_anchor_match', False)
                ),
                'goal_progress_relation_type': str(
                    goal_progress_adjustment.get('relation_type', '') or ''
                ),
                'goal_progress_relation_target': str(
                    goal_progress_adjustment.get('relation_target', '') or ''
                ),
                'goal_progress_relation_anchor_progress': float(
                    goal_progress_adjustment.get('relation_anchor_progress', 0.0) or 0.0
                ),
                'goal_progress_gap_closing_relation_anchor': bool(
                    goal_progress_adjustment.get('gap_closing_relation_anchor', False)
                ),
                'goal_progress_gap_closing_relation_anchor_rank': int(
                    goal_progress_adjustment.get('gap_closing_relation_anchor_rank', 0) or 0
                ),
                'goal_progress_controller_supported_goal_anchor': bool(
                    goal_progress_adjustment.get('controller_supported_goal_anchor', False)
                ),
                'goal_progress_mechanism_prior_anchor_match': bool(
                    goal_progress_adjustment.get('mechanism_prior_anchor_match', False)
                ),
                'goal_progress_mechanism_prior_color_match': bool(
                    goal_progress_adjustment.get('mechanism_prior_color_match', False)
                ),
                'goal_progress_mechanism_prior_function_match': bool(
                    goal_progress_adjustment.get('mechanism_prior_function_match', False)
                ),
                'goal_bundle_bonus': float(goal_bundle_adjustment.get('bonus', 0.0) or 0.0),
                'goal_bundle_is_combo_complement': bool(goal_bundle_adjustment.get('is_combo_complement', False)),
                'goal_bundle_is_necessary_but_insufficient': bool(goal_bundle_adjustment.get('is_necessary_but_insufficient', False)),
                'goal_bundle_is_distinct_from_combo_seed': bool(goal_bundle_adjustment.get('is_distinct_from_combo_seed', False)),
            })

        if click_hints.get('x') is not None and click_hints.get('y') is not None:
            add_spec(
                click_hints.get('x'),
                click_hints.get('y'),
                role='world_model_hint',
                reason='world_model_action_kwargs_hint',
                priority=0.93,
                target_family='world_model_hint',
                action_family='probe_state_transition',
                probe_aliases=['probe_state_transition'],
            )

        if failed_click_feedback:
            for failure_rank, item in enumerate(failed_click_feedback[:2]):
                point = item.get('point') if isinstance(item, dict) else None
                if not isinstance(point, tuple) or len(point) != 2:
                    continue
                for ring_rank, (nx, ny) in enumerate(self._expand_click_neighbors(point, width=width, height=height, radii=(1, 2, 4))[:10]):
                    failed_neighbor_priority = (
                        0.60 - failure_rank * 0.035 - ring_rank * 0.012
                        if prefer_salient_over_failed_neighbor
                        else 0.76 - failure_rank * 0.045 - ring_rank * 0.012
                    )
                    add_spec(
                        nx,
                        ny,
                        role='failed_click_neighbor_search',
                        reason='local_search_after_click_failure',
                        priority=failed_neighbor_priority,
                        target_family='failed_click_frontier',
                        action_family='probe_state_transition',
                        probe_aliases=['probe_state_transition', 'probe_high_impact_belief'],
                        discriminating_candidate=True,
                    )

        if schema_failed_click_feedback:
            for failure_rank, item in enumerate(schema_failed_click_feedback[:2]):
                point = item.get('point') if isinstance(item, dict) else None
                if not isinstance(point, tuple) or len(point) != 2:
                    continue
                for ring_rank, (nx, ny) in enumerate(self._expand_click_neighbors(point, width=width, height=height, radii=(1, 2, 3))[:12]):
                    add_spec(
                        nx,
                        ny,
                        role='schema_failure_neighbor_search',
                        reason='recover_from_schema_failed_click',
                        priority=0.82 - failure_rank * 0.04 - ring_rank * 0.014,
                        target_family='schema_failure_frontier',
                        action_family='probe_state_transition',
                        probe_aliases=['probe_state_transition', 'probe_relation'],
                        discriminating_candidate=True,
                    )

        for target in list(perception.get('suggested_click_targets', []) or []):
            if not isinstance(target, dict):
                continue
            before_count = len(specs)
            add_spec(
                target.get('x'),
                target.get('y'),
                role=str(target.get('role', 'perception_target') or 'perception_target'),
                reason=str(target.get('reason', 'perception_click_target') or 'perception_click_target'),
                priority=float(target.get('priority', 0.7) or 0.7),
                target_family=str(target.get('target_family', target.get('role', 'perception_target')) or 'perception_target'),
                action_family='probe_high_impact_belief' if str(target.get('role', '') or '').startswith('salient_') else 'probe_state_transition',
                anchor_ref=str(target.get('object_id', '') or ''),
                object_color=target.get('color'),
                probe_aliases=list(target.get('probe_aliases', []) or []),
                discriminating_candidate='salient' in str(target.get('role', '') or ''),
            )
            if len(specs) > before_count:
                specs[-1]['explicit_perception_target'] = True

        salient_objects = list(perception.get('salient_objects', []) or []) if isinstance(perception.get('salient_objects', []), list) else []
        salient_objects = [obj for obj in salient_objects if isinstance(obj, dict)]
        salient_area_limit = max(4, int(width * height * 0.03)) if width > 0 and height > 0 else 4
        for obj in salient_objects:
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            keepalive_tags = {str(item) for item in list(obj.get('keepalive_tags', []) or []) if str(item)}
            area = int(obj.get('area', 0) or 0)
            color = int(obj.get('color', 0) or 0)
            rarity = float(obj.get('rarity_score', 0.0) or 0.0)
            actionable = float(obj.get('actionable_score', 0.0) or 0.0)
            salience = float(obj.get('salience_score', 0.0) or 0.0)
            changed_overlap = int(obj.get('changed_overlap', 0) or 0)
            boundary_contact = bool(obj.get('boundary_contact', False))
            goal_like = bool(obj.get('goal_like', False) or ('goal_like' in keepalive_tags))
            high_value = bool(
                goal_like
                or 'color9_small' in keepalive_tags
                or 'boundary_touching_rare' in keepalive_tags
                or (color == 9 and area > 0 and area <= salient_area_limit)
                or (boundary_contact and rarity >= 0.82)
                or changed_overlap > 0
                or actionable >= 0.82
                or salience >= 0.88
            )
            if not high_value:
                continue
            add_spec(
                centroid.get('x'),
                centroid.get('y'),
                role='salient_object_centroid_backfill',
                reason='backfill_missing_salient_centroid',
                priority=0.8 + actionable * 0.12 + (0.05 if goal_like else 0.0),
                target_family='salient_object_backfill',
                action_family='probe_high_impact_belief' if (goal_like or keepalive_tags) else 'probe_state_transition',
                anchor_ref=str(obj.get('object_id', '') or ''),
                object_color=obj.get('color'),
                probe_aliases=['probe_state_transition', 'probe_high_impact_belief'],
                discriminating_candidate=bool(goal_like or changed_overlap > 0 or keepalive_tags),
            )

        objects = list(bindings.get('objects', []) or []) if isinstance(bindings.get('objects', []), list) else []
        objects = [obj for obj in objects if isinstance(obj, dict)]
        objects.sort(
            key=lambda item: (
                -float(item.get('actionable_score', 0.0) or 0.0),
                -float(item.get('salience_score', 0.0) or 0.0),
                str(item.get('object_id', '') or ''),
            )
        )
        for obj in objects[:6]:
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            bbox = obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {}
            anchor_ref = str(obj.get('object_id', '') or '')
            color = obj.get('color')
            add_spec(
                centroid.get('x'),
                centroid.get('y'),
                role='bound_object_center',
                reason='object_binding_actionable_center',
                priority=0.76 + float(obj.get('actionable_score', 0.0) or 0.0) * 0.18,
                target_family='bound_object',
                action_family='probe_high_impact_belief' if anchor_ref in preferred_target_ref_set else 'probe_state_transition',
                anchor_ref=anchor_ref,
                object_color=color,
                probe_aliases=['probe_state_transition', 'probe_high_impact_belief'],
                discriminating_candidate=anchor_ref in preferred_target_ref_set,
            )
            if bbox:
                x_min = int(bbox.get('col_min', bbox.get('x_min', 0)) or 0)
                x_max = int(bbox.get('col_max', bbox.get('x_max', x_min)) or x_min)
                y_min = int(bbox.get('row_min', bbox.get('y_min', 0)) or 0)
                y_max = int(bbox.get('row_max', bbox.get('y_max', y_min)) or y_min)
                cx = int(round((x_min + x_max) / 2.0))
                cy = int(round((y_min + y_max) / 2.0))
                neighbor_points = [
                    (cx, max(0, y_min - 1)),
                    (cx, min(max(height - 1, 0), y_max + 1)),
                    (max(0, x_min - 1), cy),
                    (min(max(width - 1, 0), x_max + 1), cy),
                ]
                for nx, ny in neighbor_points[:2]:
                    add_spec(
                        nx,
                        ny,
                        role='bound_object_neighbor',
                        reason='object_binding_neighbor_probe',
                        priority=0.62,
                        target_family='bound_object_neighbor',
                        action_family='probe_relation',
                        anchor_ref=anchor_ref,
                        object_color=color,
                        probe_aliases=['probe_relation', 'probe_state_transition'],
                        discriminating_candidate=anchor_ref in preferred_target_ref_set,
                    )

        hotspot = perception.get('suggested_hotspot', {}) if isinstance(perception.get('suggested_hotspot', {}), dict) else {}
        for x_key, y_key in (('x', 'y'), ('col', 'row')):
            if hotspot.get(x_key) is not None and hotspot.get(y_key) is not None:
                hotspot_point = (int(hotspot.get(x_key)), int(hotspot.get(y_key)))
                add_spec(
                    hotspot.get(x_key),
                    hotspot.get(y_key),
                    role='suggested_hotspot',
                    reason=str(hotspot.get('source', 'perception_hotspot') or 'perception_hotspot'),
                    priority=0.74,
                    target_family='hotspot',
                    action_family='probe_state_transition',
                    probe_aliases=['probe_state_transition'],
                )
                for ring_rank, (nx, ny) in enumerate(self._expand_click_neighbors(hotspot_point, width=width, height=height, radii=(1, 2))[:8]):
                    add_spec(
                        nx,
                        ny,
                        role='hotspot_neighbor_search',
                        reason='local_search_around_hotspot',
                        priority=0.69 - ring_rank * 0.02,
                        target_family='hotspot_neighbor',
                        action_family='probe_relation',
                        probe_aliases=['probe_relation', 'probe_state_transition'],
                    )
                break

        for point in list(perception.get('counterfactual_points', []) or []):
            if not isinstance(point, dict):
                continue
            add_spec(
                point.get('x'),
                point.get('y'),
                role='counterfactual_background',
                reason='counterfactual_background_probe',
                priority=0.38,
                target_family='background_control',
                action_family='probe_state_transition',
                probe_aliases=['probe_state_transition'],
            )

        if width > 0 and height > 0:
            add_spec(
                int(max(0, (width - 1) / 2)),
                int(max(0, (height - 1) / 2)),
                role='grid_center',
                reason='grid_center_fallback',
                priority=0.18,
                target_family='grid_center',
                action_family='probe_state_transition',
                probe_aliases=['probe_state_transition'],
                probe_candidate=False,
            )
            corner_points = [
                (0, 0),
                (max(width - 1, 0), 0),
                (0, max(height - 1, 0)),
                (max(width - 1, 0), max(height - 1, 0)),
            ]
            for corner_rank, (cx, cy) in enumerate(corner_points):
                add_spec(
                    cx,
                    cy,
                    role='grid_corner',
                    reason='grid_corner_sweep',
                    priority=0.24 - corner_rank * 0.01,
                    target_family='grid_corner',
                    action_family='probe_state_transition',
                    probe_aliases=['probe_state_transition'],
                    probe_candidate=False,
                )

        specs.sort(
            key=lambda item: (
                -float(item.get('priority', 0.0) or 0.0),
                not bool(item.get('world_model_required_probe_match', False)),
                str(item.get('role', '') or ''),
                int(item.get('kwargs', {}).get('y', 0) or 0),
                int(item.get('kwargs', {}).get('x', 0) or 0),
            )
        )
        top_specs = list(specs[:16])
        selected_points = {
            (
                int(item.get('kwargs', {}).get('x', 0) or 0),
                int(item.get('kwargs', {}).get('y', 0) or 0),
            )
            for item in top_specs
            if isinstance(item.get('kwargs', {}), dict)
        }

        def append_preserved_spec(spec: Dict[str, Any]) -> None:
            kwargs = spec.get('kwargs', {}) if isinstance(spec.get('kwargs', {}), dict) else {}
            point = (
                int(kwargs.get('x', 0) or 0),
                int(kwargs.get('y', 0) or 0),
            )
            if point in selected_points:
                return
            selected_points.add(point)
            top_specs.append(spec)

        for spec in specs:
            if bool(spec.get('explicit_perception_target', False)):
                append_preserved_spec(spec)
        action6_coverage_window = action6_click_count < 8
        if action6_coverage_window:
            preserve_role_groups = (
                ('salient_object_center', 'salient_object_centroid_backfill', 'bound_object_center'),
                ('bound_object_neighbor', 'hotspot_neighbor_search', 'failed_click_neighbor_search', 'schema_failure_neighbor_search'),
                ('suggested_hotspot',),
                ('background_control', 'counterfactual_background'),
                ('grid_center', 'grid_corner'),
            )
            for role_group in preserve_role_groups:
                role_spec = next(
                    (item for item in specs if str(item.get('role', '') or '') in role_group),
                    None,
                )
                if role_spec is not None:
                    append_preserved_spec(role_spec)
        if top_specs and not any(str(item.get('role', '') or '') in {'background_control', 'counterfactual_background'} for item in top_specs):
            background_spec = next((item for item in specs if str(item.get('role', '') or '') in {'background_control', 'counterfactual_background'}), None)
            if background_spec is not None:
                append_preserved_spec(background_spec)
        if (
            failed_click_feedback
            and not prefer_salient_over_failed_neighbor
            and top_specs
            and not any(str(item.get('role', '') or '') == 'failed_click_neighbor_search' for item in top_specs)
        ):
            frontier_spec = next((item for item in specs if str(item.get('role', '') or '') == 'failed_click_neighbor_search'), None)
            if frontier_spec is not None:
                append_preserved_spec(frontier_spec)
        return top_specs

    def _default_click_kwargs(
        self,
        obs: Dict[str, Any],
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        specs = self._surface_click_specs(obs, episode_trace, world_model_summary)
        if specs:
            top = specs[0].get('kwargs', {}) if isinstance(specs[0].get('kwargs', {}), dict) else {}
            if top.get('x') is not None and top.get('y') is not None:
                return {'x': int(top.get('x')), 'y': int(top.get('y'))}

        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
            if _canonicalize_function_name(self._function_name_from_action(action)) != 'ACTION6':
                continue
            payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
            if 'x' in kwargs and 'y' in kwargs and kwargs.get('x') is not None and kwargs.get('y') is not None:
                return {'x': kwargs.get('x'), 'y': kwargs.get('y')}

        perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
        active_bbox = perception.get('active_bbox', {}) if isinstance(perception.get('active_bbox', {}), dict) else {}
        if active_bbox:
            row_min = int(active_bbox.get('row_min', active_bbox.get('y_min', 0)) or 0)
            row_max = int(active_bbox.get('row_max', active_bbox.get('y_max', row_min)) or row_min)
            col_min = int(active_bbox.get('col_min', active_bbox.get('x_min', 0)) or 0)
            col_max = int(active_bbox.get('col_max', active_bbox.get('x_max', col_min)) or col_min)
            return {'x': int((col_min + col_max) / 2), 'y': int((row_min + row_max) / 2)}

        grid_shape = perception.get('grid_shape', {}) if isinstance(perception.get('grid_shape', {}), dict) else {}
        width = int(grid_shape.get('width', 0) or 0)
        height = int(grid_shape.get('height', 0) or 0)
        if width > 0 and height > 0:
            return {'x': int(max(0, (width - 1) / 2)), 'y': int(max(0, (height - 1) / 2))}
        return {'x': 0, 'y': 0}

    def _make_call_action(
        self,
        function_name: str,
        obs: Optional[Dict[str, Any]] = None,
        plan_state: Any = None,
        episode_trace: Optional[Sequence[Dict[str, Any]]] = None,
        world_model_summary: Optional[Dict[str, Any]] = None,
        retrieved_obj: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        function_name = _canonicalize_function_name(function_name)
        resolved_kwargs, sources, required_keys, resolved_keys, missing_keys = self._resolve_kwargs(
            function_name=function_name,
            obs=obs or {},
            plan_state=plan_state,
            episode_trace=episode_trace or [],
            world_model_summary=world_model_summary or {},
            retrieved_obj=retrieved_obj or {},
            explicit_kwargs=kwargs,
        )

        signatures = obs.get('function_signatures', {}) if isinstance(obs, dict) else {}
        signatures = signatures if isinstance(signatures, dict) else {}
        signature = signatures.get(function_name)
        known_signature = function_name in signatures
        signature_allows_empty_kwargs = bool(known_signature and not required_keys and not missing_keys)
        candidate_meta = {
            'param_sources': sources,
            'underspecified_candidate': bool((not resolved_kwargs) and not signature_allows_empty_kwargs),
            'signature_known': known_signature,
            'signature_allows_empty_kwargs': signature_allows_empty_kwargs,
            'required_kwargs': required_keys,
            'resolved_required_kwargs': resolved_keys,
            'missing_required_kwargs': missing_keys,
        }
        if not known_signature:
            candidate_meta['kwargs_status'] = 'pending_completion'
            candidate_meta['risk_tags'] = ['unknown_function_signature']
        elif missing_keys:
            candidate_meta['kwargs_status'] = 'insufficient_required_kwargs'
            candidate_meta['risk_tags'] = [f'missing_required_kwarg:{key}' for key in missing_keys]
            candidate_meta['executable'] = False
            candidate_meta['non_executable_reason'] = 'missing_required_kwargs'
            return {
                'kind': 'non_executable_call',
                'function_name': function_name,
                'kwargs': deepcopy(resolved_kwargs),
                'x': resolved_kwargs.get('x'),
                'y': resolved_kwargs.get('y'),
                'payload': {
                    'tool_name': 'call_hidden_function',
                    'tool_args': {
                        'function_name': function_name,
                        'kwargs': deepcopy(resolved_kwargs),
                    },
                },
                '_candidate_meta': candidate_meta,
            }
        elif signature_allows_empty_kwargs:
            candidate_meta['kwargs_status'] = 'ready_zero_arg'
        else:
            candidate_meta['kwargs_status'] = 'ready'
        if function_name == 'ACTION6' and ('x' not in resolved_kwargs or 'y' not in resolved_kwargs):
            deferred_point = None
            if isinstance(resolved_kwargs.get('resolved_execution_coords'), dict):
                coord_payload = resolved_kwargs.get('resolved_execution_coords', {})
                if coord_payload.get('x') is not None and coord_payload.get('y') is not None:
                    deferred_point = {'x': int(coord_payload.get('x')), 'y': int(coord_payload.get('y'))}
            elif (
                isinstance(resolved_kwargs.get('execution_point'), dict)
                and resolved_kwargs.get('execution_point', {}).get('x') is not None
                and resolved_kwargs.get('execution_point', {}).get('y') is not None
            ):
                coord_payload = resolved_kwargs.get('execution_point', {})
                deferred_point = {'x': int(coord_payload.get('x')), 'y': int(coord_payload.get('y'))}

            if deferred_point is not None:
                resolved_kwargs.setdefault('x', deferred_point['x'])
                resolved_kwargs.setdefault('y', deferred_point['y'])
                if 'from_deferred_execution_point' not in sources:
                    sources.append('from_deferred_execution_point')
                candidate_meta['param_sources'] = sources
                candidate_meta['click_parameterized'] = True
                candidate_meta['underspecified_candidate'] = False
                candidate_meta['kwargs_status'] = 'ready_click'
                candidate_meta['execution_contract_status'] = 'explicit_from_deferred_context'
            else:
                candidate_meta['param_sources'] = sources
                candidate_meta['click_parameterized'] = False
                candidate_meta['underspecified_candidate'] = True
                candidate_meta['kwargs_status'] = 'missing_explicit_click_coords'
                candidate_meta['execution_contract_status'] = 'missing_explicit_click_coords'
                candidate_meta['risk_tags'] = list(dict.fromkeys(list(candidate_meta.get('risk_tags', []) or []) + [
                    'missing_required_kwarg:x',
                    'missing_required_kwarg:y',
                    'action6_requires_explicit_execution_coords',
                ]))
                candidate_meta['executable'] = False
                candidate_meta['non_executable_reason'] = 'missing_explicit_click_coords'
                return {
                    'kind': 'non_executable_call',
                    'function_name': function_name,
                    'kwargs': deepcopy(resolved_kwargs),
                    'x': resolved_kwargs.get('x'),
                    'y': resolved_kwargs.get('y'),
                    'payload': {
                        'tool_name': 'call_hidden_function',
                        'tool_args': {
                            'function_name': function_name,
                            'kwargs': deepcopy(resolved_kwargs),
                        },
                    },
                    '_candidate_meta': candidate_meta,
                }

        return {
            'kind': 'call_tool',
            'function_name': function_name,
            'kwargs': deepcopy(resolved_kwargs),
            'x': resolved_kwargs.get('x'),
            'y': resolved_kwargs.get('y'),
            'payload': {
                'tool_name': 'call_hidden_function',
                'tool_args': {
                    'function_name': function_name,
                    'kwargs': deepcopy(resolved_kwargs),
                },
            },
            '_candidate_meta': candidate_meta,
        }

    def _resolve_kwargs(
        self,
        function_name: str,
        obs: Dict[str, Any],
        plan_state: Any,
        episode_trace: Sequence[Dict[str, Any]],
        world_model_summary: Dict[str, Any],
        retrieved_obj: Dict[str, Any],
        explicit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[str], List[str], List[str], List[str]]:
        kwargs: Dict[str, Any] = {}
        sources: List[str] = []

        signatures = obs.get('function_signatures', {}) if isinstance(obs, dict) else {}
        signature = signatures.get(function_name) if isinstance(signatures, dict) else None
        required_keys = self._required_kwargs_from_signature(signature)

        if isinstance(explicit_kwargs, dict) and explicit_kwargs:
            kwargs.update(explicit_kwargs)
            sources.append('from_explicit_kwargs')

        if plan_state is not None and getattr(plan_state, 'has_plan', False):
            step = getattr(plan_state, 'current_step', None)
            if step is not None and isinstance(getattr(step, 'constraints', None), dict):
                step_constraints = step.constraints.get('tool_kwargs', {})
                if isinstance(step_constraints, dict) and step_constraints:
                    for k, v in step_constraints.items():
                        kwargs.setdefault(k, v)
                    sources.append('from_plan_constraints')

        if isinstance(retrieved_obj, dict):
            content = retrieved_obj.get('content', {})
            if isinstance(content, dict):
                tool_args = content.get('tool_args', {})
                if isinstance(tool_args, dict):
                    retrieved_kwargs = tool_args.get('kwargs', {})
                    if isinstance(retrieved_kwargs, dict) and retrieved_kwargs:
                        for k, v in retrieved_kwargs.items():
                            kwargs.setdefault(k, v)
                        sources.append('from_retrieved_object')

        if episode_trace:
            for entry in reversed(list(episode_trace)):
                if float(entry.get('reward', 0.0) or 0.0) <= 0.0:
                    continue
                action = entry.get('action', {}) if isinstance(entry, dict) else {}
                if _canonicalize_function_name(self._function_name_from_action(action)) != function_name:
                    continue
                payload = action.get('payload', {}) if isinstance(action, dict) else {}
                tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
                hist_kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
                if isinstance(hist_kwargs, dict) and hist_kwargs:
                    for k, v in hist_kwargs.items():
                        kwargs.setdefault(k, v)
                    sources.append('from_history_pattern')
                    break

        wm_ctx = world_model_summary.get('action_kwargs_hints', {}) if isinstance(world_model_summary, dict) else {}
        if isinstance(wm_ctx, dict):
            hints = wm_ctx.get(function_name, {})
            if isinstance(hints, dict) and hints:
                for k, v in hints.items():
                    kwargs.setdefault(k, v)
                sources.append('from_world_model_context')

        if function_name == 'ACTION6' and ('x' not in kwargs or 'y' not in kwargs):
            click_kwargs = self._default_click_kwargs(obs, episode_trace, world_model_summary)
            for k, v in click_kwargs.items():
                kwargs.setdefault(k, v)
            sources.append('from_default_click_synthesis')

        resolved_required_keys = [key for key in required_keys if key in kwargs and kwargs.get(key) is not None]
        missing_required_keys = [key for key in required_keys if key not in kwargs or kwargs.get(key) is None]

        dedup_sources: List[str] = []
        for src in sources:
            if src not in dedup_sources:
                dedup_sources.append(src)

        return kwargs, dedup_sources, required_keys, resolved_required_keys, missing_required_keys

    def _required_kwargs_from_signature(self, signature: Any) -> List[str]:
        if not isinstance(signature, dict):
            return []

        required_keys: List[str] = []

        required = signature.get('required')
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key and key not in required_keys:
                    required_keys.append(key)

        params = signature.get('parameters')
        props: Dict[str, Any] = {}
        if isinstance(params, dict):
            params_required = params.get('required')
            if isinstance(params_required, list):
                for key in params_required:
                    if isinstance(key, str) and key and key not in required_keys:
                        required_keys.append(key)

            maybe_props = params.get('properties')
            if isinstance(maybe_props, dict):
                props = maybe_props
                for key, spec in props.items():
                    if not isinstance(key, str) or not key or key in required_keys:
                        continue
                    if isinstance(spec, dict) and spec.get('required') is True:
                        required_keys.append(key)

        return required_keys

    def _append_candidate(self, candidates: List[Dict[str, Any]], candidate: Optional[Dict[str, Any]]) -> None:
        if not candidate:
            return
        semantic_signature = self._candidate_semantic_signature(candidate)
        for existing in candidates:
            if self._candidate_semantic_signature(existing) == semantic_signature:
                self._merge_candidate(existing, candidate)
                return
        meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
        source = str(candidate.get('_source', 'base_generation') or 'base_generation')
        support_sources = list(meta.get('support_sources', [])) if isinstance(meta.get('support_sources', []), list) else []
        if source not in support_sources:
            support_sources.append(source)
        meta['support_sources'] = support_sources
        candidate['_candidate_meta'] = meta
        candidates.append(candidate)

    def _has_viable_non_wait_candidate(self, candidates: Sequence[Dict[str, Any]]) -> bool:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            fn_name = _canonicalize_function_name(extract_action_function_name(candidate, default='').strip())
            action_kind = extract_action_kind(candidate, default='call_tool')
            if fn_name == 'wait' or action_kind == 'wait':
                continue
            meta = candidate.get('_candidate_meta', {})
            if not isinstance(meta, dict):
                meta = {}
            if meta.get('executable') is False:
                continue
            return True
        return False

    def _sanitize_candidates_for_arbiter(
        self,
        candidates: List[Dict[str, Any]],
        obs: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        signatures = obs.get('function_signatures', {}) if isinstance(obs, dict) else {}
        signatures = signatures if isinstance(signatures, dict) else {}
        filter_reason_counts: Dict[str, int] = {}
        for candidate in candidates:
            if not self._strict_signature_mode:
                self._append_candidate(sanitized, candidate)
                continue
            is_valid, risk_tags = self._validate_call_candidate(candidate, signatures)
            if is_valid:
                self._append_candidate(sanitized, candidate)
                continue
            if self._best_effort_mode and candidate.get('kind') == 'call_tool':
                candidate_copy = deepcopy(candidate)
                meta = candidate_copy.get('_candidate_meta', {})
                existing = list(meta.get('risk_tags', [])) if isinstance(meta.get('risk_tags', []), list) else []
                for tag in risk_tags:
                    if tag not in existing:
                        existing.append(tag)
                meta['risk_tags'] = existing
                meta['best_effort_passthrough'] = True
                candidate_copy['_candidate_meta'] = meta
                self._append_candidate(sanitized, candidate_copy)
                continue
            invalid_call_reason = '|'.join(risk_tags) if risk_tags else 'invalid_call_candidate'
            invalid_meta = {
                'invalid_source': candidate.get('_source', 'unknown'),
                'missing_required_fields': [tag.split(':', 1)[1] for tag in risk_tags if tag.startswith('missing_required_kwarg:')],
                'invalid_candidate_summary': {
                    'kind': candidate.get('kind'),
                    'function_name': self._function_name_from_action(candidate),
                },
            }
            filter_reason_counts[invalid_call_reason] = filter_reason_counts.get(invalid_call_reason, 0) + 1
            meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
            fn_name = _canonicalize_function_name(self._function_name_from_action(candidate))
            visible_functions = set(self._extract_env_available_functions(obs))
            if fn_name and fn_name in visible_functions and fn_name != 'wait':
                candidate_copy = deepcopy(candidate)
                copy_meta = candidate_copy.get('_candidate_meta', {}) if isinstance(candidate_copy.get('_candidate_meta', {}), dict) else {}
                copy_meta['best_effort_passthrough'] = True
                copy_meta['sanitize_recovered_from_visible_surface'] = True
                copy_meta['sanitize_invalid_call_reason'] = invalid_call_reason
                candidate_copy['_candidate_meta'] = copy_meta
                self._append_candidate(sanitized, candidate_copy)
                continue
            wait = self._build_wait_candidate()
            wait_meta = wait.get('_candidate_meta', {})
            wait_meta.update({
                'filtered_invalid_call_candidate': True,
                'filtered_source': candidate.get('_source', 'unknown'),
                'invalid_call_reason': invalid_call_reason,
                'invalid_meta': invalid_meta,
                'candidate_meta': meta,
            })
            wait['_candidate_meta'] = wait_meta
            self._append_candidate(sanitized, wait)
        if (
            not any(self._function_name_from_action(candidate) == 'wait' for candidate in sanitized)
            and not self._has_viable_non_wait_candidate(sanitized)
            and _governance_wait_baseline_allowed(sanitized, None)
            and not (is_arc3_surface(obs) and bool(self._extract_env_available_functions(obs)))
        ):
            wait = self._build_wait_candidate()
            wait_meta = wait.get('_candidate_meta', {})
            wait_meta['wait_injection_reason'] = 'no_viable_non_wait'
            wait['_candidate_meta'] = wait_meta
            self._append_candidate(sanitized, wait)
        return sanitized, {'filter_reason_counts': filter_reason_counts}

    def _validate_call_candidate(self, candidate: Dict[str, Any], signatures: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not isinstance(candidate, dict):
            return False, ['invalid_candidate_shape']
        if candidate.get('kind') != 'call_tool':
            return True, []
        payload = candidate.get('payload', {}) if isinstance(candidate.get('payload'), dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        function_name = _canonicalize_function_name(tool_args.get('function_name', '') if isinstance(tool_args, dict) else '')
        if not isinstance(function_name, str) or not function_name.strip():
            return False, ['missing_function_name']

        kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
        if not isinstance(kwargs, dict):
            return False, ['invalid_kwargs_shape']

        if function_name == 'ACTION6' and kwargs.get('x') is not None and kwargs.get('y') is not None:
            return True, []

        required_keys = self._required_kwargs_from_signature(signatures.get(function_name))
        risk_tags: List[str] = []
        for key in required_keys:
            if key not in kwargs or kwargs.get(key) is None:
                risk_tags.append(f'missing_required_kwarg:{key}')
        if risk_tags:
            return False, risk_tags
        return True, []

    def _record_candidate_trace(
        self,
        obs: Dict[str, Any],
        phase: str,
        candidate_count: int,
        filter_reason_counts: Dict[str, int],
    ) -> None:
        if not isinstance(obs, dict):
            return
        trace = obs.get('candidate_trace')
        if not isinstance(trace, list):
            trace = []
            obs['candidate_trace'] = trace
        trace.append(
            {
                'phase': phase,
                'candidate_count': int(candidate_count),
                'filter_reason_counts': dict(filter_reason_counts or {}),
            }
        )

    def _candidate_signature(self, candidate: Dict[str, Any]) -> Tuple[str, str, str]:
        fn = extract_action_identity(candidate, include_function_fallback=True) or self._function_name_from_action(candidate)
        source = str(candidate.get('_source', 'base_generation'))
        payload = candidate.get('payload', {}) if isinstance(candidate, dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        kwargs_repr = repr(tool_args.get('kwargs', {}))
        return source, fn, kwargs_repr

    def _canonical_semantic_kwargs(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(candidate, dict):
            return {}
        payload = candidate.get('payload', {}) if isinstance(candidate, dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) and isinstance(tool_args.get('kwargs', {}), dict) else {}
        if not isinstance(kwargs, dict):
            return {}

        context_only_keys = {
            'function_name_hint',
            'tick',
            'episode',
            'plan_step_intent',
            'plan_step_target',
            'goal_id',
            'recent_reward',
            'retrieved_objects',
            'perception',
            'plan_revision_count',
        }
        return {
            key: value
            for key, value in kwargs.items()
            if key not in context_only_keys
        }

    def _candidate_semantic_signature(self, candidate: Dict[str, Any]) -> Tuple[str, str, str]:
        if not isinstance(candidate, dict):
            return ('wait', 'wait', '{}')
        action_kind = extract_action_kind(candidate, default='call_tool')
        if action_kind == 'wait':
            return ('wait', 'wait', '{}')
        fn = extract_action_identity(candidate, include_function_fallback=True) or _canonicalize_function_name(self._function_name_from_action(candidate))
        kwargs_repr = repr(self._canonical_semantic_kwargs(candidate))
        return action_kind, fn, kwargs_repr

    def _source_priority(self, source: str) -> int:
        order = {
            'planner': 80,
            'procedure_reuse': 75,
            'intervention_compiler': 74,
            'latent_mechanism': 72,
            'history_reuse': 68,
            'retrieval': 64,
            'self_model': 60,
            'arm_evaluation': 55,
            'base_generation': 50,
            'surface_generation': 65,
            'wait_fallback': 10,
        }
        return int(order.get(str(source or ''), 40))

    def _merge_candidate(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        existing_meta = existing.get('_candidate_meta', {}) if isinstance(existing.get('_candidate_meta', {}), dict) else {}
        incoming_meta = incoming.get('_candidate_meta', {}) if isinstance(incoming.get('_candidate_meta', {}), dict) else {}

        existing_source = str(existing.get('_source', 'base_generation') or 'base_generation')
        incoming_source = str(incoming.get('_source', 'base_generation') or 'base_generation')

        support_sources: List[str] = []
        for value in list(existing_meta.get('support_sources', [])) + [existing_source] + list(incoming_meta.get('support_sources', [])) + [incoming_source]:
            source_name = str(value or '').strip()
            if source_name and source_name not in support_sources:
                support_sources.append(source_name)
        existing_meta['support_sources'] = support_sources
        existing_meta['support_source_count'] = len(support_sources)

        surfaced_from: List[str] = []
        for value in list(existing_meta.get('surfaced_from', [])) + list(incoming_meta.get('surfaced_from', [])):
            object_id = str(value or '').strip()
            if object_id and object_id not in surfaced_from:
                surfaced_from.append(object_id)
        if surfaced_from:
            existing_meta['surfaced_from'] = surfaced_from

        for key in (
            'planner_matches_step',
            'planner_target_visible',
            'recent_negative_feedback',
            'perception_guard',
            'camera_relative_context',
            'structured_answer_synthesized',
            'structured_answer_internal_simulation',
            'counterfactual_advantage',
            'surface_generation',
            'surface_click_candidate',
            'click_parameterized',
        ):
            if bool(incoming_meta.get(key)):
                existing_meta[key] = True

        numeric_max_keys = (
            'retrieval_score',
            'history_reward',
            'capability_score',
            'self_model_confidence',
            'counterfactual_delta',
            'world_model_probe_pressure',
            'world_model_latent_instability',
        )
        for key in numeric_max_keys:
            existing_value = existing_meta.get(key)
            incoming_value = incoming_meta.get(key)
            if isinstance(existing_value, (int, float)) and isinstance(incoming_value, (int, float)):
                existing_meta[key] = max(float(existing_value), float(incoming_value))
            elif incoming_value is not None and not isinstance(existing_value, (int, float)):
                existing_meta[key] = incoming_value

        for key in (
            'procedure',
            'history_tick',
            'planner_step_intent',
            'planner_target_function',
            'planner_target_source',
            'capability_context',
            'belief_branch_guidance',
            'hidden_state_guidance',
            'failure_strategy_profile',
            'failure_preference_guidance',
            'surface_visible_functions',
        ):
            if key not in existing_meta and key in incoming_meta:
                existing_meta[key] = incoming_meta[key]

        if self._source_priority(incoming_source) > self._source_priority(existing_source):
            existing['_source'] = incoming_source
            existing['kind'] = incoming.get('kind', existing.get('kind'))
            existing['payload'] = deepcopy(incoming.get('payload', existing.get('payload', {})))

        existing['_candidate_meta'] = existing_meta

    def _entry_has_positive_progress(self, entry: Dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False
        task_progress = entry.get('task_progress', {}) if isinstance(entry.get('task_progress', {}), dict) else {}
        if bool(task_progress.get('progressed', False)):
            return True
        if bool(task_progress.get('solved', False)):
            return True
        if float(entry.get('reward', 0.0) or 0.0) > 0.0:
            return True
        progress_markers = entry.get('progress_markers', []) if isinstance(entry.get('progress_markers', []), list) else []
        task_progress_seen = False
        goal_stalled = False
        local_only_reaction = False
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            name = str(marker.get('name', '') or '')
            if name in {'goal_progressed', 'positive_reward'}:
                return True
            if name == 'task_progressed':
                task_progress_seen = True
            if name == 'goal_stalled':
                goal_stalled = True
            if name == 'local_only_reaction':
                local_only_reaction = True
            if name == 'terminal_reached' and bool(marker.get('success', False)):
                return True
        return bool(task_progress_seen and not goal_stalled and not local_only_reaction)

    def _entry_failure_reason(self, entry: Dict[str, Any]) -> str:
        if not isinstance(entry, dict):
            return ''
        result = entry.get('result', {}) if isinstance(entry.get('result', {}), dict) else {}
        failure_reason = str(
            entry.get('failure_reason')
            or result.get('failure_reason')
            or ''
        ).strip()
        if failure_reason:
            return failure_reason
        progress_markers = entry.get('progress_markers', []) if isinstance(entry.get('progress_markers', []), list) else []
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            if str(marker.get('name', '') or '') == 'failure_reason':
                failure_reason = str(marker.get('value', '') or '').strip()
                if failure_reason:
                    return failure_reason
        return ''

    def _is_schema_failure_reason(self, failure_reason: Any) -> bool:
        text = str(failure_reason or '').strip().lower()
        if not text or text == 'none':
            return False
        return (
            'schema_failure' in text
            or text in {'illegal_click_coordinate_or_remote_rejection', 'arc_agi3_schema_failure_remote_rejection'}
            or 'requires explicit x/y' in text
            or ('bad request' in text and 'action6' in text)
        )

    def _annotate_recent_action_feedback(
        self,
        candidates: Sequence[Dict[str, Any]],
        episode_trace: Sequence[Dict[str, Any]],
        window: int = 6,
    ) -> None:
        recent_entries = [entry for entry in list(episode_trace or [])[-max(1, int(window)): ] if isinstance(entry, dict)]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            fn_name = _canonicalize_function_name(self._function_name_from_action(candidate))
            if not fn_name:
                continue
            same_function_entries = []
            for entry in recent_entries:
                action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
                if _canonicalize_function_name(self._function_name_from_action(action)) != fn_name:
                    continue
                same_function_entries.append(entry)

            same_action_count = len(same_function_entries)
            positive_progress_count = sum(1 for entry in same_function_entries if self._entry_has_positive_progress(entry))
            no_progress_count = max(0, same_action_count - positive_progress_count)

            consecutive_no_progress_count = 0
            for entry in reversed(recent_entries):
                action = entry.get('action', {}) if isinstance(entry.get('action', {}), dict) else {}
                entry_fn = _canonicalize_function_name(self._function_name_from_action(action))
                if entry_fn != fn_name:
                    continue
                if self._entry_has_positive_progress(entry):
                    break
                consecutive_no_progress_count += 1

            meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
            recent_feedback = {
                'recent_same_action_count': same_action_count,
                'positive_progress_count': positive_progress_count,
                'no_progress_count': no_progress_count,
                'consecutive_no_progress_count': consecutive_no_progress_count,
                'recent_no_progress_ratio': (float(no_progress_count) / float(same_action_count)) if same_action_count else 0.0,
                'action_cooldown_recommended': bool(consecutive_no_progress_count >= 3 and positive_progress_count == 0),
            }
            meta['recent_action_feedback'] = recent_feedback
            meta['recent_same_action_count'] = same_action_count
            meta['recent_same_action_positive_progress_count'] = positive_progress_count
            meta['recent_same_action_no_progress_count'] = no_progress_count
            meta['consecutive_no_progress_count'] = consecutive_no_progress_count
            meta['recent_no_progress_ratio'] = recent_feedback['recent_no_progress_ratio']
            meta['action_cooldown_recommended'] = recent_feedback['action_cooldown_recommended']
            candidate['_candidate_meta'] = meta

    def _function_name_from_action(self, action: Optional[Dict[str, Any]]) -> str:
        if not isinstance(action, dict):
            return ''
        if action.get('kind') == 'wait':
            return 'wait'
        payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        return _canonicalize_function_name(tool_args.get('function_name', ''))

    def _with_source(self, action: Dict[str, Any], source: str) -> Dict[str, Any]:
        cloned = deepcopy(action)
        cloned['_source'] = source
        cloned.setdefault('_candidate_meta', {})
        payload = cloned.get('payload', {}) if isinstance(cloned.get('payload', {}), dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        function_name = _canonicalize_function_name(tool_args.get('function_name', ''))
        kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
        if function_name:
            tool_args['function_name'] = function_name
            payload['tool_args'] = tool_args
            cloned['payload'] = payload
            cloned['function_name'] = function_name
        if isinstance(kwargs, dict):
            cloned['kwargs'] = deepcopy(kwargs)
            if kwargs.get('x') is not None:
                cloned['x'] = kwargs.get('x')
            if kwargs.get('y') is not None:
                cloned['y'] = kwargs.get('y')
        return cloned

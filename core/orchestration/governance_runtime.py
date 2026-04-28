from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from core.main_loop_components import (
    CAPABILITY_ADVISORY,
    CAPABILITY_CONSTRAINED_CONTROL,
    CAPABILITY_PRIMARY_CONTROL,
    LOW_RISK_CONTROL_FUNCTIONS,
    TickContextFrame,
)
from core.orchestration.action_utils import (
    action_matches_blocked_name,
    extract_action_function_name,
    extract_action_kind,
    extract_action_xy,
    extract_available_functions,
)
from core.orchestration.commit_candidate_guard import (
    select_high_confidence_commit_candidate,
    should_override_selected_action_with_commit_guard,
)
from core.orchestration.state_abstraction import summarize_action_state
from decision.governance_candidate_adapter import (
    NormalizationInput,
    has_sufficient_failure_evidence,
    normalize_candidates,
)
from modules.world_model.protocol import WorldModelControlProtocol
from core.orchestration.governance_state import GovernanceState, GovernanceStatePatch

_WM_LONG_REWARD_WEIGHT = 0.45
_WM_RISK_WEIGHT = 0.55
_WM_REVERSIBILITY_WEIGHT = 0.25
_WM_INFO_GAIN_WEIGHT = 0.20
_WM_HIGH_RISK_THRESHOLD = 0.75
_WM_LOW_REVERSIBILITY_THRESHOLD = 0.30
_WM_HIGH_INFO_GAIN_THRESHOLD = 0.65
_WM_HIGH_UNCERTAINTY_THRESHOLD = 0.60
_WM_HIGH_DRIFT_THRESHOLD = 0.68
_WM_PROTECTED_ACTIVE_STEP_CONFIDENCE = 0.78
_WM_LOW_TRUST_PROTECTION_THRESHOLD = 0.72


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


class ReliabilityPort(Protocol):
    def build_global_failure_strategy(self, *, short_term_pressure: float) -> Any:
        ...


class CounterfactualPort(Protocol):
    def simulate_action_difference(self, state_slice: Any, action_a: Dict[str, Any], action_b: Dict[str, Any], *, context: Dict[str, Any]) -> Any:
        ...


class GovernanceLogPort(Protocol):
    def append_governance(self, entry: Dict[str, Any]) -> None:
        ...

    def append_candidate_viability(self, entry: Dict[str, Any]) -> None:
        ...


class OrganCapabilityPort(Protocol):
    def get_capability(self, organ: str, state: GovernanceState) -> str:
        ...


@dataclass
class GovernanceDecisionTrace:
    tick: int
    episode: int
    selected: str
    selected_name: str
    reason: str
    mode: str
    risk: float
    opportunity: float
    selection_score: float
    hard_constraints: List[str]
    soft_constraints: List[str]
    arm_meta: Dict[str, Any]
    goal_id: Optional[str]
    task_id: Optional[str]
    risk_modifier: float
    opportunity_modifier: float
    adjusted_risk: float
    adjusted_opportunity: float
    is_exploration_goal: bool
    global_failure_strategy: Dict[str, Any]
    meta_control_snapshot_id: str
    meta_control_inputs_hash: str
    governance_candidate_count: int
    governance_raw_candidate_count: int
    governance_unique_candidate_count: int
    governance_skip_layer1: bool
    governance_skip_reason: Optional[str]
    governance_selected_from_candidates: bool
    selected_candidate_index: Optional[int]
    selected_organ: str
    selected_organ_capability: str
    viability_recovered_from_raw: bool
    ablation_flags: Dict[str, Any]
    selected_world_model_competition: Dict[str, Any]
    organ_control_decisions: List[Dict[str, Any]]
    contract_version: str = 'v1'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tick': self.tick,
            'episode': self.episode,
            'selected': self.selected,
            'selected_name': self.selected_name,
            'reason': self.reason,
            'mode': self.mode,
            'risk': self.risk,
            'opportunity': self.opportunity,
            'selection_score': self.selection_score,
            'hard_constraints': list(self.hard_constraints),
            'soft_constraints': list(self.soft_constraints),
            'arm_meta': dict(self.arm_meta),
            'goal_id': self.goal_id,
            'task_id': self.task_id,
            'risk_modifier': self.risk_modifier,
            'opportunity_modifier': self.opportunity_modifier,
            'adjusted_risk': self.adjusted_risk,
            'adjusted_opportunity': self.adjusted_opportunity,
            'is_exploration_goal': self.is_exploration_goal,
            'global_failure_strategy': dict(self.global_failure_strategy),
            'meta_control_snapshot_id': self.meta_control_snapshot_id,
            'meta_control_inputs_hash': self.meta_control_inputs_hash,
            'governance_candidate_count': self.governance_candidate_count,
            'governance_raw_candidate_count': self.governance_raw_candidate_count,
            'governance_unique_candidate_count': self.governance_unique_candidate_count,
            'governance_skip_layer1': self.governance_skip_layer1,
            'governance_skip_reason': self.governance_skip_reason,
            'governance_selected_from_candidates': self.governance_selected_from_candidates,
            'selected_candidate_index': self.selected_candidate_index,
            'selected_organ': self.selected_organ,
            'selected_organ_capability': self.selected_organ_capability,
            'viability_recovered_from_raw': self.viability_recovered_from_raw,
            'ablation_flags': dict(self.ablation_flags),
            'selected_world_model_competition': dict(self.selected_world_model_competition),
            'organ_control_decisions': list(self.organ_control_decisions),
            'contract_version': self.contract_version,
        }


def _ordered_unique_strings(*values: Any) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            continue
        for item in items:
            text = str(item or '').strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
    return ordered


def _extract_current_visible_functions(obs_before: Optional[Dict[str, Any]]) -> Set[str]:
    visible: List[str] = []

    def _extend(values: Any) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            text = str(value or '').strip()
            if text and text not in visible:
                visible.append(text)

    if not isinstance(obs_before, dict):
        return set()

    _extend(obs_before.get('visible_functions', []))
    novel_api = obs_before.get('novel_api', {})
    if hasattr(novel_api, 'raw'):
        novel_api = novel_api.raw
    if isinstance(novel_api, dict):
        _extend(novel_api.get('visible_functions', []))

    if visible:
        return set(visible)

    _extend(obs_before.get('available_functions', []))
    if isinstance(novel_api, dict):
        _extend(novel_api.get('available_functions', []))
    return set(visible)


def _candidate_function_name(candidate: Dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return ''
    return str(candidate.get('function_name') or candidate.get('action') or '').strip()


def _selected_world_model_competition_summary(
    selected_candidate: Optional[Dict[str, Any]],
    governance_candidates: Sequence[Dict[str, Any]],
    *,
    selected_action_name: str,
) -> Dict[str, Any]:
    selected_raw_action = selected_candidate.get('raw_action', {}) if isinstance(selected_candidate, dict) else {}
    selected_meta = (
        selected_raw_action.get('_candidate_meta', {})
        if isinstance(selected_raw_action, dict) and isinstance(selected_raw_action.get('_candidate_meta', {}), dict)
        else {}
    )
    selected_function_name = (
        _candidate_function_name(selected_candidate)
        if isinstance(selected_candidate, dict)
        else str(selected_action_name or 'wait').strip()
    ) or str(selected_action_name or 'wait').strip() or 'wait'

    required_probes = _ordered_unique_strings(selected_meta.get('world_model_required_probes', []))
    dominant_anchor_functions = _ordered_unique_strings(selected_meta.get('world_model_anchor_functions', []))
    dominant_risky_functions = _ordered_unique_strings(selected_meta.get('world_model_risky_functions', []))
    required_probe_available_count = 0
    anchor_candidate_count = 0
    risky_candidate_count = 0

    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        fn_name = _candidate_function_name(candidate)
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        required_probes = _ordered_unique_strings(required_probes, meta.get('world_model_required_probes', []))
        dominant_anchor_functions = _ordered_unique_strings(dominant_anchor_functions, meta.get('world_model_anchor_functions', []))
        dominant_risky_functions = _ordered_unique_strings(dominant_risky_functions, meta.get('world_model_risky_functions', []))
        if bool(meta.get('world_model_required_probe_match', False)) or (fn_name and fn_name in required_probes):
            required_probe_available_count += 1
        if bool(meta.get('world_model_anchor_match', False)) or (fn_name and fn_name in dominant_anchor_functions):
            anchor_candidate_count += 1
        if bool(meta.get('world_model_risky_match', False)) or (fn_name and fn_name in dominant_risky_functions):
            risky_candidate_count += 1

    selected_required_probe_match = bool(
        selected_meta.get('world_model_required_probe_match', False)
        or (selected_function_name and selected_function_name in required_probes)
    )
    selected_anchor_match = bool(
        selected_meta.get('world_model_anchor_match', False)
        or (selected_function_name and selected_function_name in dominant_anchor_functions)
    )
    selected_risky_match = bool(
        selected_meta.get('world_model_risky_match', False)
        or (selected_function_name and selected_function_name in dominant_risky_functions)
    )
    probe_pressure = _clamp01(selected_meta.get('world_model_probe_pressure', 0.0), 0.0)
    latent_instability = _clamp01(selected_meta.get('world_model_latent_instability', 0.0), 0.0)
    dominant_branch_id = str(selected_meta.get('world_model_dominant_branch_id', '') or '').strip()
    competition_active = bool(
        probe_pressure >= 0.34
        or latent_instability >= 0.55
        or required_probe_available_count > 0
        or risky_candidate_count > 0
    )

    return {
        'selected_action': selected_function_name,
        'competition_active': competition_active,
        'required_probes': list(required_probes),
        'probe_pressure': float(probe_pressure),
        'latent_instability': float(latent_instability),
        'dominant_branch_id': dominant_branch_id,
        'selected_required_probe_match': selected_required_probe_match,
        'selected_anchor_match': selected_anchor_match,
        'selected_risky_match': selected_risky_match,
        'preserved_required_probe': bool(required_probe_available_count > 0 and selected_required_probe_match),
        'preserved_branch_anchor': bool(anchor_candidate_count > 0 and selected_anchor_match),
        'avoided_risky_branch_action': bool(risky_candidate_count > 0 and not selected_risky_match),
        'required_probe_available_count': int(required_probe_available_count),
        'anchor_candidate_count': int(anchor_candidate_count),
        'risky_candidate_count': int(risky_candidate_count),
        'failure_preference_learning_bias': float(selected_meta.get('failure_preference_learning_bias', 0.0) or 0.0),
        'retention_learning_bonus': float(selected_meta.get('retention_learning_bonus', 0.0) or 0.0),
        'world_model_learning_bias': float(selected_meta.get('world_model_learning_bias', 0.0) or 0.0),
        'learning_bias': float(selected_meta.get('learning_bias', 0.0) or 0.0),
    }


def govern_action(
    loop,
    action: Dict[str, Any],
    candidate_actions: List[Dict[str, Any]],
    continuity_snapshot: Dict[str, Any],
    frame: TickContextFrame,
    meta_control_state: Optional[Dict[str, Any]] = None,
    reliability_port: Optional[ReliabilityPort] = None,
    counterfactual_port: Optional[CounterfactualPort] = None,
    governance_log_port: Optional[GovernanceLogPort] = None,
    organ_capability_port: Optional[OrganCapabilityPort] = None,
    governance_state: Optional[GovernanceState] = None,
) -> Dict[str, Any]:
    contract_version = 'v1'
    meta_control_state = meta_control_state or {}
    arm_meta = meta_control_state.get('arm_meta', {}) if isinstance(meta_control_state.get('arm_meta', {}), dict) else {}
    decision_outcome = meta_control_state.get('decision_outcome')
    obs_before = meta_control_state.get('obs_before') if isinstance(meta_control_state.get('obs_before'), dict) else {}
    available_functions = extract_available_functions(obs_before)
    visible_functions = _extract_current_visible_functions(obs_before)
    current_surface_functions = list(visible_functions) if visible_functions else list(available_functions)
    plan_target_function = ''
    plan_state = getattr(loop, '_plan_state', None)
    current_step = getattr(plan_state, 'current_step', None) if plan_state is not None else None
    if current_step is not None:
        plan_target_function = str(getattr(current_step, 'target_function', '') or '').strip()

    top_goal = continuity_snapshot.get('top_goal')
    next_task = continuity_snapshot.get('next_task')

    is_exploration_goal = False
    is_testing_task = False
    goal_risk_modifier = 0.0
    task_opportunity_modifier = 0.0

    if top_goal:
        goal_id = getattr(top_goal, 'goal_id', '') or ''
        if 'explore' in goal_id.lower():
            is_exploration_goal = True
            goal_risk_modifier = 0.15
        elif 'exploit' in goal_id.lower() or 'confirm' in goal_id.lower():
            goal_risk_modifier = -0.10
        elif 'test' in goal_id.lower():
            goal_risk_modifier = 0.05

    if next_task:
        task_id = getattr(next_task, 'task_id', '') or ''
        if 'test' in task_id.lower() or 'probe' in task_id.lower():
            is_testing_task = True
            task_opportunity_modifier = 0.10
        elif 'explore' in task_id.lower():
            task_opportunity_modifier = 0.05
        elif 'consolidate' in task_id.lower() or 'commit' in task_id.lower():
            task_opportunity_modifier = -0.05

    selected_action_kind = extract_action_kind(action, default='call_tool')
    is_wait = selected_action_kind == 'wait'
    base_risk = 0.05 if is_wait else 0.2
    base_opportunity = 0.1 if is_wait else 0.8
    adjusted_risk = max(0.0, min(1.0, base_risk + goal_risk_modifier))
    adjusted_opportunity = max(0.0, min(1.0, base_opportunity + task_opportunity_modifier))

    candidate_actions = candidate_actions if isinstance(candidate_actions, list) else []
    normalization_result = normalize_candidates(
        NormalizationInput(
            governance_candidates=[],
            raw_candidates=candidate_actions,
            selected_action=action,
            obs_before=obs_before,
            tick=loop._tick,
            episode=loop._episode,
            episode_trace=loop._episode_trace if isinstance(loop._episode_trace, list) else [],
        ),
        extract_action_function_name=loop._extract_action_function_name,
    )
    unique_candidate_actions = normalization_result.unique_candidate_actions
    raw_candidate_count = len(candidate_actions)
    unique_candidate_count = len(unique_candidate_actions)
    skip_layer1 = normalization_result.skip_layer1
    skip_reason = normalization_result.skip_reason
    governance_candidates: List[Dict[str, Any]] = []
    selected_fn_hint = loop._extract_action_function_name(action, default='')

    for raw_action in unique_candidate_actions:
        if not isinstance(raw_action, dict):
            continue
        fn_name = loop._extract_action_function_name(raw_action, default='')
        if not fn_name and selected_fn_hint and selected_fn_hint != 'wait':
            raw_action = loop._repair_action_function_name(raw_action, selected_fn_hint)
            fn_name = loop._extract_action_function_name(raw_action, default='')
        if not fn_name:
            continue

        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        action_kind = extract_action_kind(raw_action, default='call_tool')
        is_wait_candidate = action_kind == 'wait'
        risk = meta.get('risk', raw_action.get('risk', 0.05 if is_wait_candidate else adjusted_risk))
        opportunity = meta.get(
            'opportunity_estimate',
            meta.get('opportunity', raw_action.get('opportunity_estimate', 0.1 if is_wait_candidate else adjusted_opportunity)),
        )
        final_score = meta.get('final_score', raw_action.get('final_score', float(opportunity) - float(risk)))
        estimated_cost = raw_action.get('estimated_cost', 0.1 if is_wait_candidate else 1.0)

        governance_candidates.append({
            'action': fn_name,
            'function_name': fn_name,
            'intent': raw_action.get('_source', 'main_loop_candidate'),
            'action_kind': action_kind,
            'risk': float(risk),
            'opportunity_estimate': float(opportunity),
            'final_score': float(final_score),
            'estimated_cost': float(estimated_cost),
            'raw_action': raw_action,
            'goal_id': getattr(top_goal, 'goal_id', None),
            'task_id': getattr(next_task, 'task_id', None),
            'is_exploration_goal': is_exploration_goal,
            'is_testing_task': is_testing_task,
        })
    _apply_active_procedure_guidance(unique_candidate_actions)
    governance_candidates = _normalize_governance_candidates(governance_candidates)

    normalization_result = normalize_candidates(
        NormalizationInput(
            governance_candidates=governance_candidates,
            raw_candidates=candidate_actions,
            selected_action=action,
            obs_before=obs_before,
            tick=loop._tick,
            episode=loop._episode,
            episode_trace=loop._episode_trace if isinstance(loop._episode_trace, list) else [],
        ),
        extract_action_function_name=loop._extract_action_function_name,
    )
    governance_candidates_before_normalization = normalization_result.governance_candidates_before_normalization
    governance_candidates = normalization_result.governance_candidates_after_normalization
    for viability_event in normalization_result.viability_events:
        if governance_log_port:
            governance_log_port.append_candidate_viability(dict(viability_event))

    hard_constraints = ['no_shutdown']
    soft_constraints: List[str] = []
    budget_state = {'energy': 100.0}
    governance_controls = loop._meta_control.for_planner_replan(episode=loop._episode, tick=loop._tick, context={'gate': 'governance'})
    policy_profile = governance_controls.get('policy_profile', {}) if isinstance(governance_controls.get('policy_profile', {}), dict) else {}
    global_failure_strategy = reliability_port.build_global_failure_strategy(
        short_term_pressure=min(1.0, sum(1 for entry in loop._episode_trace[-5:] if float(entry.get('reward', 0.0) or 0.0) < 0.0) / 5.0)
    ) if reliability_port else None
    executable_candidate_functions = {
        str(candidate.get('function_name') or '').strip()
        for candidate in governance_candidates
        if isinstance(candidate, dict) and str(candidate.get('function_name') or '').strip() not in {'', 'wait'}
    }
    blocked_actions = [
        str(blocked_action or '').strip()
        for blocked_action in (global_failure_strategy.blocked_action_classes if global_failure_strategy else [])
        if str(blocked_action or '').strip()
    ]
    hard_blockable_actions = {
        blocked_action
        for blocked_action in blocked_actions
        if blocked_action in executable_candidate_functions
    }
    would_exhaust_non_wait_candidates = bool(executable_candidate_functions) and hard_blockable_actions >= executable_candidate_functions
    if would_exhaust_non_wait_candidates and 'preserve_last_executable_action' not in soft_constraints:
        soft_constraints.append('preserve_last_executable_action')
    for blocked_action in blocked_actions:
        if would_exhaust_non_wait_candidates and blocked_action in executable_candidate_functions:
            continue
        constraint = f"no_{blocked_action}"
        if constraint not in hard_constraints:
            hard_constraints.append(constraint)
    confidence = max(0.1, 1.0 - loop._hypotheses.entropy())
    uncertainty = loop._hypotheses.entropy()
    prediction_error_tail = []
    if getattr(loop, '_prediction_enabled', False) and hasattr(loop, '_prediction_registry'):
        recent_errors = loop._prediction_registry.get_recent_errors(5)
        prediction_error_tail = [
            item.to_dict() if hasattr(item, 'to_dict') else dict(item)
            for item in recent_errors
            if hasattr(item, 'to_dict') or isinstance(item, dict)
        ]
    world_model_control = WorldModelControlProtocol.from_context({
        'world_model_summary': frame.world_model_summary,
        'world_model_transition_priors': frame.world_model_transition_priors,
        'predictor_trust': loop._prediction_registry.get_predictor_trust() if getattr(loop, '_prediction_enabled', False) and hasattr(loop, '_prediction_registry') else {},
        'prediction_error_tail': prediction_error_tail,
    })
    world_model_control = _sanitize_world_model_control_for_candidates(
        governance_candidates,
        world_model_control,
    )
    transition_priors = frame.world_model_transition_priors if isinstance(frame.world_model_transition_priors, dict) else {}
    _apply_world_model_scoring(
        governance_candidates,
        transition_priors=transition_priors,
        world_model_control=world_model_control,
    )
    _apply_dynamic_world_model_constraints(
        governance_candidates,
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        uncertainty=uncertainty,
        world_model_control=world_model_control,
    )
    _apply_deliberation_scoring(governance_candidates, raw_candidates=candidate_actions)
    _apply_active_procedure_arbitration(
        governance_candidates,
        world_model_control=world_model_control,
        soft_constraints=soft_constraints,
    )
    _apply_hidden_state_scoring(
        governance_candidates,
        world_model_control=world_model_control,
        soft_constraints=soft_constraints,
    )
    _apply_belief_branch_scoring(
        governance_candidates,
        current_step=loop._plan_state.current_step if getattr(loop, '_plan_state', None) and loop._plan_state.has_plan else None,
        world_model_control=world_model_control,
        soft_constraints=soft_constraints,
    )
    _apply_meta_control_scoring(
        governance_candidates,
        policy_profile=policy_profile,
        soft_constraints=soft_constraints,
    )
    _apply_failure_preference_scoring(
        governance_candidates,
        global_failure_strategy=global_failure_strategy,
        world_model_control=world_model_control,
        soft_constraints=soft_constraints,
    )
    state_slice = loop._build_state_slice()
    counterfactual_cfg = _resolve_counterfactual_policy(meta_control_state)
    cf_outcome = None
    candidate_counterfactual_map: Dict[int, Dict[str, Any]] = {}
    if len(state_slice.established_beliefs) >= 1:
        # Governance counterfactual scoring is intentionally planner-candidate bounded.
        # Mechanism matching stays in world-model/probe paths and must not expand or bias
        # governance candidate sets via mechanism candidate injection.
        top_k = max(1, int(counterfactual_cfg.get('top_k', 2) or 2))
        top_candidates = sorted(
            [(idx, candidate) for idx, candidate in enumerate(governance_candidates) if isinstance(candidate, dict)],
            key=lambda item: float(item[1].get('final_score', 0.0) or 0.0),
            reverse=True,
        )[:top_k]
        for idx, candidate in top_candidates:
            candidate_cf = compare_candidate_counterfactuals(
                loop=loop,
                counterfactual_port=counterfactual_port,
                state_slice=state_slice,
                governance_candidates=governance_candidates,
                candidate_index=idx,
                context={
                    'goal_id': getattr(top_goal, 'goal_id', None),
                    'task_id': getattr(next_task, 'task_id', None),
                    'state_features': getattr(state_slice, 'state_features', {}),
                    'transition_priors': transition_priors,
                },
            )
            if candidate_cf:
                candidate_counterfactual_map[idx] = candidate_cf
                _attach_counterfactual_to_candidate(candidate, candidate_cf)
                if governance_log_port:
                    governance_log_port.append_governance({
                        'tick': loop._tick,
                        'episode': loop._episode,
                        'entry': 'counterfactual_candidate_compare',
                        'candidate_index': idx,
                        'candidate_action': candidate.get('function_name') or candidate.get('action') or 'wait',
                        'preferred_action': candidate_cf.get('preferred_action', ''),
                        'confidence': candidate_cf.get('confidence', 0.0),
                        'reasoning': candidate_cf.get('reasoning', ''),
                        'estimated_delta': candidate_cf.get('estimated_delta', 0.0),
                        'comparisons': list(candidate_cf.get('comparisons', [])),
                    })
        if top_candidates:
            first_idx = top_candidates[0][0]
            first_advice = candidate_counterfactual_map.get(first_idx, {})
            cf_outcome = first_advice.get('outcome_obj')
        _apply_counterfactual_penalties(governance_candidates, candidate_counterfactual_map, counterfactual_cfg)

    action_candidate_index = next((idx for idx, candidate in enumerate(governance_candidates) if candidate.get('function_name') != 'wait'), None)

    gov_result = None
    selected_action_name = 'wait'
    selected_candidate_index = None
    selected_from_candidates = False

    if skip_layer1:
        if action_candidate_index is not None:
            selected_candidate_index = action_candidate_index
            selected_action_name = governance_candidates[action_candidate_index].get('action') or 'wait'
            selected_from_candidates = True
            skip_reason = skip_reason or 'unique_candidates_lt_2_prefer_non_wait'
        elif governance_candidates:
            selected_candidate_index = 0
            selected_action_name = governance_candidates[0].get('action') or 'wait'
            selected_from_candidates = True
            skip_reason = skip_reason or 'unique_candidates_lt_2_only_wait'
    else:
        gov_result = loop._governance.evaluate(
            candidates=governance_candidates,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            budget_state=budget_state,
            confidence=confidence,
            uncertainty=uncertainty,
            state_mgr=loop._state_mgr,
            world_model_control=world_model_control,
            policy_profile=policy_profile,
        )
        selected_action_name = gov_result.selected_action.get('action')
        selected_candidate_index = _resolve_selected_candidate_index(
            governance_candidates,
            gov_result.selected_action,
            selected_action_name,
        )
        selected_from_candidates = selected_candidate_index is not None
        if skip_layer1 and not selected_from_candidates and action_candidate_index is not None:
            selected_candidate_index = action_candidate_index
            selected_action_name = governance_candidates[action_candidate_index].get('action') or 'wait'
            selected_from_candidates = True
            skip_reason = skip_reason or 'skip_layer1_guard_prevent_wait_fallback'
        if not selected_from_candidates:
            selected_action_name = 'wait'

    selection_reason = f"skip_layer1:{skip_reason or 'unique_candidates_lt_2'}" if skip_layer1 else str(gov_result.selection_reason if gov_result else '')
    selected_candidate = governance_candidates[selected_candidate_index] if selected_from_candidates and selected_candidate_index is not None else None
    selected_raw_action = selected_candidate.get('raw_action', {}) if isinstance(selected_candidate, dict) else {}
    selected_source = str(selected_raw_action.get('_source', '') or '')
    selected_organ = _source_to_organ(selected_source)
    selected_capability = _capability_for_organ(organ_capability_port, governance_state, selected_organ)
    selected_function_name = (
        loop._extract_action_function_name(selected_raw_action, default=selected_action_name)
        if isinstance(selected_raw_action, dict)
        else selected_action_name
    )
    selected_is_active_procedure_step = _is_active_procedure_step(selected_raw_action)
    if (
        selected_organ
        and selected_capability == CAPABILITY_CONSTRAINED_CONTROL
        and not _is_low_risk_control_action(selected_function_name)
        and not selected_is_active_procedure_step
    ):
        if _allow_constrained_control_gap_closing_candidate(
            selected_candidate,
            visible_functions=visible_functions,
        ):
            selected_meta = selected_raw_action.get('_candidate_meta', {}) if isinstance(selected_raw_action.get('_candidate_meta', {}), dict) else {}
            if bool(selected_meta.get('safe_execution_preferred', False)):
                selection_reason = f"{selection_reason}|capability_guard:safe_execution_preferred"
            else:
                selection_reason = f"{selection_reason}|capability_guard:multi_source_gap_closing_allowed"
        else:
            fallback_idx = _find_capability_guard_fallback_candidate(
                governance_candidates,
                blocked_candidate_index=selected_candidate_index,
                organ_capability_port=organ_capability_port,
                governance_state=governance_state,
                visible_functions=visible_functions,
            )
            if fallback_idx is not None:
                selected_candidate_index = fallback_idx
                selected_action_name = governance_candidates[fallback_idx].get('action') or 'wait'
                selected_from_candidates = True
                selection_reason = f"{selection_reason}|capability_guard:block_non_low_risk->fallback_candidate"
            else:
                selected_action_name = 'wait'
                selected_candidate_index = None
                selected_from_candidates = False
                selection_reason = f"{selection_reason}|capability_guard:block_non_low_risk"
    elif selected_organ and selected_capability == CAPABILITY_ADVISORY and selected_from_candidates:
        fallback_idx = _find_capability_guard_fallback_candidate(
            governance_candidates,
            blocked_candidate_index=selected_candidate_index,
            organ_capability_port=organ_capability_port,
            governance_state=governance_state,
            visible_functions=visible_functions,
        )
        if fallback_idx is not None:
            selected_candidate_index = fallback_idx
            selected_action_name = governance_candidates[fallback_idx].get('action') or 'wait'
            selected_from_candidates = True
            selection_reason = f"{selection_reason}|capability_guard:advisory_mode_requires_wait->fallback_candidate"
        else:
            selected_action_name = 'wait'
            selected_candidate_index = None
            selected_from_candidates = False
            selection_reason = f"{selection_reason}|capability_guard:advisory_mode_requires_wait"

    counterfactual_selected_advice = candidate_counterfactual_map.get(selected_candidate_index or -1, {})
    if selected_from_candidates and counterfactual_selected_advice:
        preferred_action = str(counterfactual_selected_advice.get('preferred_action') or '')
        cf_confidence = float(counterfactual_selected_advice.get('confidence', 0.0) or 0.0)
        high_threshold = float(counterfactual_cfg.get('high_confidence_threshold', 0.75) or 0.75)
        if preferred_action and preferred_action != selected_action_name and cf_confidence >= high_threshold:
            mode = str(counterfactual_cfg.get('opposition_policy', 'hard_override') or 'hard_override')
            preferred_idx = next((idx for idx, candidate in enumerate(governance_candidates) if candidate.get('function_name') == preferred_action), None)
            if mode == 'hard_override' and preferred_idx is not None:
                selected_candidate_index = preferred_idx
                selected_action_name = preferred_action
                selected_from_candidates = True
                selection_reason = f"{selection_reason}|counterfactual_override:high_confidence"
            else:
                selection_reason = f"{selection_reason}|counterfactual_oppose:downscored"

    guarded_candidate_index, _guarded_action, guarded_reason = select_high_confidence_commit_candidate(
        [
            candidate.get('raw_action', {}) if isinstance(candidate, dict) else {}
            for candidate in governance_candidates
        ],
        available_functions=current_surface_functions,
        plan_target_function=plan_target_function,
    )
    selected_raw_action = (
        governance_candidates[selected_candidate_index].get('raw_action', {})
        if selected_from_candidates and selected_candidate_index is not None
        else {'kind': 'wait', 'payload': {}}
    )
    mechanism_control_summary = (
        frame.world_model_summary.get('mechanism_control_summary', {})
        if isinstance(getattr(frame, 'world_model_summary', {}), dict)
        and isinstance(frame.world_model_summary.get('mechanism_control_summary', {}), dict)
        else {}
    )
    if (
        guarded_candidate_index is not None
        and guarded_candidate_index < len(governance_candidates)
        and guarded_candidate_index != selected_candidate_index
        and should_override_selected_action_with_commit_guard(
            selected_raw_action,
            guarded_reason,
            guarded_action=_guarded_action,
            available_functions=current_surface_functions,
            plan_target_function=plan_target_function,
            mechanism_control_summary=mechanism_control_summary,
        )
    ):
        selected_candidate_index = guarded_candidate_index
        selected_action_name = governance_candidates[guarded_candidate_index].get('action') or 'wait'
        selected_from_candidates = True
        selection_reason = f"{selection_reason}|commit_guard_override:{guarded_reason}"

    selected_function_name = (
        _candidate_function_name(governance_candidates[selected_candidate_index])
        if selected_from_candidates and selected_candidate_index is not None
        else str(selected_action_name or 'wait').strip()
    )
    visible_candidate_index = _best_visible_surface_candidate_index(
        governance_candidates,
        visible_functions=visible_functions,
        organ_capability_port=organ_capability_port,
        governance_state=governance_state,
    )
    if (
        visible_candidate_index is not None
        and selected_function_name not in visible_functions
        and visible_candidate_index != selected_candidate_index
    ):
        selected_candidate_index = visible_candidate_index
        selected_action_name = governance_candidates[visible_candidate_index].get('action') or 'wait'
        selected_from_candidates = True
        selection_reason = f"{selection_reason}|visible_surface_priority"

    selected_action_baseline_index = _resolve_selected_candidate_index(
        governance_candidates,
        action,
        loop._extract_action_function_name(action, default=''),
    )
    if (
        selected_from_candidates
        and selected_candidate_index is not None
        and selected_action_baseline_index is not None
        and selected_action_baseline_index != selected_candidate_index
        and not visible_functions
        and not has_sufficient_failure_evidence(loop._episode_trace if isinstance(loop._episode_trace, list) else [])
        and _is_synthetic_deliberation_probe_candidate(governance_candidates[selected_candidate_index])
        and _is_safe_selected_baseline_candidate(governance_candidates[selected_action_baseline_index])
    ):
        selected_candidate_index = selected_action_baseline_index
        selected_action_name = governance_candidates[selected_action_baseline_index].get('action') or 'wait'
        selected_from_candidates = True
        selection_reason = f"{selection_reason}|selected_action_baseline_priority"

    governance_mode = 'skip_layer1' if skip_layer1 else str(gov_result.new_mode if gov_result else 'normal')
    selected_candidate = governance_candidates[selected_candidate_index] if selected_from_candidates and selected_candidate_index is not None else None

    organ_control_decisions = _audit_organ_control_decisions(
        loop=loop,
        organ_capability_port=organ_capability_port,
        governance_state=governance_state,
        decision_outcome=decision_outcome,
        governance_candidates=governance_candidates,
        selected_action_name=selected_action_name,
        selection_reason=selection_reason,
        counterfactual_outcome=cf_outcome,
    )
    selected_world_model_competition = _selected_world_model_competition_summary(
        selected_candidate,
        governance_candidates,
        selected_action_name=selected_action_name,
    )
    trace = GovernanceDecisionTrace(
        tick=loop._tick,
        episode=loop._episode,
        selected=selected_action_name,
        selected_name=selected_action_name,
        reason=selection_reason,
        mode=governance_mode,
        risk=governance_candidates[selected_candidate_index]['risk'] if selected_from_candidates and selected_candidate_index is not None else 0.05,
        opportunity=governance_candidates[selected_candidate_index]['opportunity_estimate'] if selected_from_candidates and selected_candidate_index is not None else 0.1,
        selection_score=governance_candidates[selected_candidate_index]['final_score'] if selected_from_candidates and selected_candidate_index is not None else 0.05,
        hard_constraints=list(hard_constraints),
        soft_constraints=list(soft_constraints),
        arm_meta=arm_meta,
        goal_id=getattr(top_goal, 'goal_id', None),
        task_id=getattr(next_task, 'task_id', None),
        risk_modifier=goal_risk_modifier,
        opportunity_modifier=task_opportunity_modifier,
        adjusted_risk=adjusted_risk,
        adjusted_opportunity=adjusted_opportunity,
        is_exploration_goal=is_exploration_goal,
        global_failure_strategy=global_failure_strategy.to_dict() if global_failure_strategy else {},
        meta_control_snapshot_id=str(governance_controls.get('meta_control_snapshot_id', '')),
        meta_control_inputs_hash=str(governance_controls.get('meta_control_inputs_hash', '')),
        governance_candidate_count=len(governance_candidates),
        governance_raw_candidate_count=raw_candidate_count,
        governance_unique_candidate_count=unique_candidate_count,
        governance_skip_layer1=skip_layer1,
        governance_skip_reason=skip_reason,
        governance_selected_from_candidates=selected_from_candidates,
        selected_candidate_index=selected_candidate_index,
        selected_organ=_source_to_organ(str(selected_candidate.get('raw_action', {}).get('_source', '') or '')) if isinstance(selected_candidate, dict) else '',
        selected_organ_capability=_capability_for_organ(organ_capability_port, governance_state, _source_to_organ(str(selected_candidate.get('raw_action', {}).get('_source', '') or ''))) if isinstance(selected_candidate, dict) else CAPABILITY_ADVISORY,
        viability_recovered_from_raw=any(bool(c.get('viability_recovered_from_raw')) for c in governance_candidates if isinstance(c, dict)),
        ablation_flags=loop._ablation_flags_snapshot(),
        selected_world_model_competition=selected_world_model_competition,
        organ_control_decisions=organ_control_decisions,
        contract_version=contract_version,
    )
    audit_payload = trace.to_dict()
    if governance_log_port:
        governance_log_port.append_governance(audit_payload)
        if selected_from_candidates:
            advice = counterfactual_selected_advice if isinstance(counterfactual_selected_advice, dict) else {}
            suggested = str(advice.get('preferred_action') or '')
            governance_log_port.append_governance({
                'tick': loop._tick,
                'episode': loop._episode,
                'entry': 'counterfactual_adoption_metric',
                'suggested_action': suggested,
                'selected_action': selected_action_name,
                'suggestion_present': bool(suggested),
                'adopted': bool(suggested) and suggested == selected_action_name,
            })

    selected_action = governance_candidates[selected_candidate_index]['raw_action'] if selected_from_candidates and selected_candidate_index is not None else {'kind': 'wait', 'payload': {}}
    selected_risk = governance_candidates[selected_candidate_index]['risk'] if selected_from_candidates and selected_candidate_index is not None else 0.05
    selected_opportunity = governance_candidates[selected_candidate_index]['opportunity_estimate'] if selected_from_candidates and selected_candidate_index is not None else 0.1
    selected_score = governance_candidates[selected_candidate_index]['final_score'] if selected_from_candidates and selected_candidate_index is not None else 0.05

    return {
        'selected_action': selected_action,
        'selected_name': selected_action_name,
        'reason': selection_reason,
        'mode': governance_mode,
        'meta_control_snapshot_id': governance_controls.get('meta_control_snapshot_id', ''),
        'meta_control_inputs_hash': governance_controls.get('meta_control_inputs_hash', ''),
        'risk': selected_risk,
        'opportunity': selected_opportunity,
        'selection_score': selected_score,
        'hard_constraints': list(hard_constraints),
        'soft_constraints': list(soft_constraints),
        'governance_candidate_count': len(governance_candidates),
        'governance_raw_candidate_count': raw_candidate_count,
        'governance_unique_candidate_count': unique_candidate_count,
        'governance_skip_layer1': skip_layer1,
        'governance_skip_reason': skip_reason,
        'governance_selected_from_candidates': selected_from_candidates,
        'selected_candidate_index': selected_candidate_index,
        'selected_organ': audit_payload.get('selected_organ', ''),
        'selected_organ_capability': audit_payload.get('selected_organ_capability', CAPABILITY_ADVISORY),
        'organ_control_decisions': organ_control_decisions,
        'organ_capability_flags': dict(governance_state.organ_capability_flags) if governance_state else {},
        'formal_write_guard': 'proposal_validator_required',
        'viability_recovered_from_raw': trace.viability_recovered_from_raw,
        'ablation_flags': loop._ablation_flags_snapshot(),
        'governance_candidates_before_normalization': governance_candidates_before_normalization,
        'governance_candidates_after_normalization': governance_candidates,
        'audit_payload': audit_payload,
        'contract_version': contract_version,
    }


def resolve_organ_control_outcome(
    *,
    reward: float,
    result: Dict[str, Any],
    governance_state: GovernanceState,
    pending_organ_control_audit: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not pending_organ_control_audit:
        return {'state_patch': GovernanceStatePatch(organ_failure_streaks={}, organ_capability_flags={}), 'governance_events': [], 'audit_entries': []}
    step_success = bool((result or {}).get('success', True)) and float(reward) >= 0.0
    outcome_label = 'positive' if step_success else 'negative'
    state_patch = GovernanceStatePatch(organ_failure_streaks={}, organ_capability_flags={})
    governance_events: List[Dict[str, Any]] = []
    audit_entries: List[Dict[str, Any]] = []
    while pending_organ_control_audit:
        entry = pending_organ_control_audit.pop(0)
        organ = str(entry.get('organ') or '')
        adopted = bool(entry.get('adopted', False))
        entry['expected_consequence'] = outcome_label
        if adopted and organ in governance_state.organ_failure_streaks:
            current_streak = int(state_patch.organ_failure_streaks.get(organ, governance_state.organ_failure_streaks.get(organ, 0)) or 0)
            if step_success:
                state_patch.organ_failure_streaks[organ] = 0
            else:
                next_streak = current_streak + 1
                state_patch.organ_failure_streaks[organ] = next_streak
                if next_streak >= governance_state.organ_failure_threshold:
                    state_patch.organ_capability_flags[organ] = CAPABILITY_ADVISORY
                    governance_events.append({
                        'entry': 'organ_capability_downgrade',
                        'organ': organ,
                        'new_capability': CAPABILITY_ADVISORY,
                        'reason': 'consecutive_failures_threshold',
                        'failure_streak': next_streak,
                    })
        audit_entries.append(dict(entry))
    return {'state_patch': state_patch, 'governance_events': governance_events, 'audit_entries': audit_entries}


def _source_to_organ(source_name: str) -> str:
    source_name = str(source_name or '').strip().lower()
    if source_name == 'planner':
        return 'planner'
    if source_name == 'self_model':
        return 'self_model'
    return ''


def _capability_for_organ(organ_capability_port: Optional[OrganCapabilityPort], governance_state: Optional[GovernanceState], organ: str) -> str:
    if not governance_state or not organ_capability_port:
        return CAPABILITY_ADVISORY
    return organ_capability_port.get_capability(organ, governance_state)


def _candidate_capability_block_reason(
    candidate: Optional[Dict[str, Any]],
    *,
    organ_capability_port: Optional[OrganCapabilityPort],
    governance_state: Optional[GovernanceState],
) -> str:
    if not isinstance(candidate, dict):
        return ''
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    organ = _source_to_organ(str(raw_action.get('_source', '') or ''))
    if not organ:
        return ''
    capability = _capability_for_organ(organ_capability_port, governance_state, organ)
    function_name = _candidate_function_name(candidate)
    if capability == CAPABILITY_ADVISORY:
        return 'advisory_mode_requires_wait'
    if capability == CAPABILITY_CONSTRAINED_CONTROL:
        if not _is_low_risk_control_action(function_name) and not _is_active_procedure_step(raw_action):
            return 'block_non_low_risk'
    return ''


def _has_positive_reward(episode_trace: Sequence[Dict[str, Any]]) -> bool:
    for entry in episode_trace if isinstance(episode_trace, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            if float(entry.get('reward', 0.0) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _allow_constrained_control_gap_closing_candidate(
    candidate: Optional[Dict[str, Any]],
    *,
    visible_functions: Optional[Set[str]] = None,
) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    function_name = _candidate_function_name(candidate)
    if not function_name or not visible_functions or function_name not in visible_functions:
        return False
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    if bool(meta.get('safe_execution_preferred', False)):
        return True
    support_sources = {
        str(item or '').strip().lower()
        for item in list(meta.get('support_sources', []) or [])
        if str(item or '').strip()
    }
    if int(meta.get('support_source_count', 0) or 0) < 2:
        return False
    if not any(not _source_to_organ(source_name) for source_name in support_sources):
        return False
    if not bool(meta.get('goal_progress_controller_supported_goal_anchor', False)):
        return False
    if bool(meta.get('goal_progress_gap_closing_preferred_goal', False)):
        return True
    return int(meta.get('goal_progress_preferred_next_goal_rank', 0) or 0) == 1


def _find_capability_guard_fallback_candidate(
    governance_candidates: Sequence[Dict[str, Any]],
    *,
    blocked_candidate_index: Optional[int],
    organ_capability_port: Optional[OrganCapabilityPort],
    governance_state: Optional[GovernanceState],
    visible_functions: Optional[Set[str]] = None,
) -> Optional[int]:
    if blocked_candidate_index is None:
        return None
    if blocked_candidate_index < 0 or blocked_candidate_index >= len(governance_candidates):
        return None
    blocked_candidate = governance_candidates[blocked_candidate_index]
    blocked_raw = blocked_candidate.get('raw_action', {}) if isinstance(blocked_candidate, dict) else {}
    blocked_signature = _raw_action_signature(blocked_raw)
    blocked_function_name = _candidate_function_name(blocked_candidate)

    same_signature: List[int] = []
    same_function: List[int] = []
    any_allowed: List[int] = []
    visible_same_signature: List[int] = []
    visible_same_function: List[int] = []
    visible_any_allowed: List[int] = []
    for idx, candidate in enumerate(governance_candidates):
        if idx == blocked_candidate_index or not isinstance(candidate, dict):
            continue
        if _candidate_capability_block_reason(
            candidate,
            organ_capability_port=organ_capability_port,
            governance_state=governance_state,
        ):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        function_name = _candidate_function_name(candidate)
        is_visible = bool(visible_functions and function_name in visible_functions)
        any_allowed.append(idx)
        if _raw_action_signature(raw_action) == blocked_signature:
            same_signature.append(idx)
            if is_visible:
                visible_same_signature.append(idx)
        elif blocked_function_name and _candidate_function_name(candidate) == blocked_function_name:
            same_function.append(idx)
            if is_visible:
                visible_same_function.append(idx)
        if is_visible:
            visible_any_allowed.append(idx)

    def _best_index(indexes: Sequence[int]) -> Optional[int]:
        if not indexes:
            return None
        return max(
            indexes,
            key=lambda idx: float(governance_candidates[idx].get('final_score', 0.0) or 0.0),
        )

    return (
        _best_index(visible_same_signature)
        or _best_index(visible_same_function)
        or _best_index(visible_any_allowed)
        or _best_index(same_signature)
        or _best_index(same_function)
        or _best_index(any_allowed)
    )


def _best_visible_surface_candidate_index(
    governance_candidates: Sequence[Dict[str, Any]],
    *,
    visible_functions: Set[str],
    organ_capability_port: Optional[OrganCapabilityPort],
    governance_state: Optional[GovernanceState],
) -> Optional[int]:
    if not visible_functions:
        return None
    visible_indexes: List[int] = []
    for idx, candidate in enumerate(governance_candidates):
        if not isinstance(candidate, dict):
            continue
        function_name = _candidate_function_name(candidate)
        if not function_name or function_name == 'wait' or function_name not in visible_functions:
            continue
        if _candidate_capability_block_reason(
            candidate,
            organ_capability_port=organ_capability_port,
            governance_state=governance_state,
        ):
            continue
        visible_indexes.append(idx)
    if not visible_indexes:
        return None
    return max(
        visible_indexes,
        key=lambda idx: float(governance_candidates[idx].get('final_score', 0.0) or 0.0),
    )


def _is_low_risk_control_action(function_name: str) -> bool:
    fn_name = str(function_name or 'wait').strip().lower()
    return fn_name in LOW_RISK_CONTROL_FUNCTIONS


def _is_active_procedure_step(raw_action: Any) -> bool:
    if not isinstance(raw_action, dict):
        return False
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
    procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
    return bool(procedure.get('is_next_step', False) or procedure_guidance.get('active_next_step', False))


def _active_procedure_strength(raw_action: Any) -> float:
    if not isinstance(raw_action, dict):
        return 0.0
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    procedure = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
    procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
    if not bool(procedure.get('is_next_step', False) or procedure_guidance.get('active_next_step', False)):
        return 0.0

    mapping_confidence = _clamp01(procedure.get('mapping_confidence', 0.0), 0.0)
    family_binding_confidence = _clamp01(procedure.get('family_binding_confidence', 0.0), 0.0)
    alignment_strength = _clamp01(procedure_guidance.get('alignment_strength', 0.0), 0.0)
    procedure_bonus = _clamp01(float(procedure.get('procedure_bonus', 0.0) or 0.0) * 4.0, 0.0)
    hit_source = str(procedure.get('hit_source', '') or '').strip()

    if hit_source == 'latent_mechanism_abstraction':
        latent_strength = (mapping_confidence * 0.65) + (family_binding_confidence * 0.35)
        return _clamp01(max(latent_strength, alignment_strength, procedure_bonus), 0.0)
    return _clamp01(max(mapping_confidence, alignment_strength, procedure_bonus), 0.0)


def _candidate_active_procedure_strength(candidate: Dict[str, Any]) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    return _active_procedure_strength(raw_action)


def _should_protect_active_procedure_candidate(
    candidate: Dict[str, Any],
    *,
    world_model_control: Optional[WorldModelControlProtocol],
) -> bool:
    wm_control = world_model_control or WorldModelControlProtocol()
    return (
        _candidate_active_procedure_strength(candidate) >= _WM_PROTECTED_ACTIVE_STEP_CONFIDENCE
        and float(wm_control.control_trust or 0.5) < _WM_LOW_TRUST_PROTECTION_THRESHOLD
    )


def _world_model_influence_factor(world_model_control: Optional[WorldModelControlProtocol]) -> float:
    wm_control = world_model_control or WorldModelControlProtocol()
    return 0.25 + (_clamp01(wm_control.control_trust, 0.5) * 0.75)


def _sanitize_world_model_control_for_candidates(
    candidates: Sequence[Dict[str, Any]],
    world_model_control: Optional[WorldModelControlProtocol],
) -> WorldModelControlProtocol:
    wm_control = world_model_control or WorldModelControlProtocol()
    protected_functions = {
        str(candidate.get('function_name') or candidate.get('action') or '')
        for candidate in candidates
        if isinstance(candidate, dict) and _should_protect_active_procedure_candidate(candidate, world_model_control=wm_control)
    }
    sole_actionable_function = _sole_grounded_non_wait_function(candidates)
    if sole_actionable_function and (
        sole_actionable_function in set(wm_control.blocked_functions or [])
        or f'no_{sole_actionable_function}' in set(str(item or '') for item in list(wm_control.hard_constraints or []))
    ):
        protected_functions.add(sole_actionable_function)
    protected_functions.discard('')
    if not protected_functions:
        return wm_control

    filtered_blocked = [
        fn_name for fn_name in list(wm_control.blocked_functions or [])
        if fn_name not in protected_functions
    ]
    filtered_hard_constraints = [
        constraint for constraint in list(wm_control.hard_constraints or [])
        if not (str(constraint or '').startswith('no_') and str(constraint or '')[3:] in protected_functions)
    ]
    if (
        filtered_blocked == list(wm_control.blocked_functions or [])
        and filtered_hard_constraints == list(wm_control.hard_constraints or [])
    ):
        return wm_control
    return replace(
        wm_control,
        blocked_functions=filtered_blocked,
        hard_constraints=filtered_hard_constraints,
    )


def _sole_grounded_non_wait_function(candidates: Sequence[Dict[str, Any]]) -> str:
    non_wait_functions = {
        str(candidate.get('function_name') or candidate.get('action') or '').strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get('function_name') or candidate.get('action') or '').strip() not in {'', 'wait'}
    }
    if len(non_wait_functions) != 1:
        return ''
    function_name = next(iter(non_wait_functions))
    matching_candidates = [
        candidate for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get('function_name') or candidate.get('action') or '').strip() == function_name
    ]
    if not matching_candidates:
        return ''
    if any(_candidate_has_grounded_execution_target(candidate) for candidate in matching_candidates):
        return function_name
    return ''


def _candidate_has_grounded_execution_target(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    if extract_action_kind(raw_action, default='call_tool') == 'wait':
        return False
    if extract_action_xy(raw_action) is not None:
        return True
    payload = raw_action.get('payload', {}) if isinstance(raw_action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    if kwargs:
        return True
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    if bool(meta.get('surface_click_candidate', False) or meta.get('explicit_perception_target', False)):
        return True
    return False


def _raw_action_signature(raw_action: Any) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    if not isinstance(raw_action, dict):
        return ('wait', 'wait', ())
    action_kind = extract_action_kind(raw_action, default='call_tool')
    function_name = extract_action_function_name(raw_action, default='wait')
    if action_kind == 'wait':
        return ('wait', 'wait', ())
    payload = raw_action.get('payload', {}) if isinstance(raw_action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    kwargs_sig = tuple(sorted((str(key), repr(value)) for key, value in kwargs.items()))
    return (action_kind, function_name, kwargs_sig)


def _resolve_selected_candidate_index(
    governance_candidates: Sequence[Dict[str, Any]],
    selected_action: Any,
    selected_action_name: str,
) -> Optional[int]:
    selected_signature = _raw_action_signature(selected_action)
    exact_signature_matches: List[int] = []
    name_matches: List[int] = []

    for idx, candidate in enumerate(governance_candidates):
        if not isinstance(candidate, dict):
            continue
        if selected_action is candidate:
            return idx
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        if selected_action is raw_action:
            return idx
        if _raw_action_signature(raw_action) == selected_signature:
            exact_signature_matches.append(idx)
        if selected_action_name and selected_action_name == str(candidate.get('action') or ''):
            name_matches.append(idx)

    if len(exact_signature_matches) == 1:
        return exact_signature_matches[0]
    if len(exact_signature_matches) > 1:
        return max(
            exact_signature_matches,
            key=lambda idx: float(governance_candidates[idx].get('final_score', 0.0) or 0.0),
        )
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        return max(
            name_matches,
            key=lambda idx: float(governance_candidates[idx].get('final_score', 0.0) or 0.0),
        )
    return None


def _is_synthetic_deliberation_probe_candidate(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    source = str(raw_action.get('_source', '') or '').strip().lower()
    action_kind = str(candidate.get('action_kind') or extract_action_kind(raw_action, default='call_tool') or '').strip().lower()
    fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip().lower()
    if source == 'deliberation_probe':
        return True
    return bool(meta.get('deliberation_injected', False) and (action_kind == 'probe' or 'probe' in fn_name))


def _is_safe_selected_baseline_candidate(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    if _is_synthetic_deliberation_probe_candidate(candidate):
        return False
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    action_kind = str(candidate.get('action_kind') or extract_action_kind(raw_action, default='call_tool') or '').strip().lower()
    fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip().lower()
    if not fn_name or fn_name == 'wait':
        return False
    if action_kind == 'inspect' or fn_name == 'inspect':
        return True
    return any(token in fn_name for token in ('inspect', 'verify', 'check', 'test')) and 'probe' not in fn_name


def _normalize_governance_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        c = dict(candidate)
        c['action'] = str(c.get('action') or c.get('function_name') or 'wait')
        c['function_name'] = str(c.get('function_name') or c.get('action') or 'wait')
        c['risk'] = max(0.0, min(1.0, float(c.get('risk', 0.05) or 0.05)))
        c['opportunity_estimate'] = max(0.0, min(1.0, float(c.get('opportunity_estimate', 0.1) or 0.1)))
        c['final_score'] = float(c.get('final_score', c['opportunity_estimate'] - c['risk']) or 0.0)
        c['estimated_cost'] = max(0.0, float(c.get('estimated_cost', 1.0) or 1.0))
        normalized.append(c)
    return normalized


def _apply_world_model_scoring(
    candidates: List[Dict[str, Any]],
    *,
    transition_priors: Dict[str, Any],
    world_model_control: Optional[WorldModelControlProtocol] = None,
) -> None:
    priors = transition_priors if isinstance(transition_priors, dict) else {}
    wm_control = world_model_control or WorldModelControlProtocol()
    influence_factor = _world_model_influence_factor(wm_control)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '')
        prior = priors.get(fn_name, {}) if isinstance(priors.get(fn_name, {}), dict) else {}
        wm_long_reward = max(0.0, min(1.0, float(prior.get('long_horizon_reward', 0.0) or 0.0)))
        wm_risk = max(0.0, min(1.0, float(prior.get('predicted_risk', candidate.get('risk', 0.0)) or 0.0)))
        wm_reversibility = max(0.0, min(1.0, float(prior.get('reversibility', 0.5) or 0.5)))
        wm_info_gain = max(0.0, min(1.0, float(prior.get('info_gain', 0.0) or 0.0)))
        if fn_name and fn_name in set(wm_control.hidden_focus_functions or []):
            wm_long_reward = min(1.0, wm_long_reward + min(0.12, wm_control.hidden_state_depth * 0.03))
        if wm_control.hidden_state_phase == 'disrupted' and fn_name not in {'wait', 'probe'}:
            wm_risk = min(1.0, wm_risk + wm_control.hidden_drift_score * 0.18)

        wm_score_delta = (
            wm_long_reward * _WM_LONG_REWARD_WEIGHT
            - wm_risk * _WM_RISK_WEIGHT
            + wm_reversibility * _WM_REVERSIBILITY_WEIGHT
            + wm_info_gain * _WM_INFO_GAIN_WEIGHT
        )
        wm_score_delta *= influence_factor
        if _should_protect_active_procedure_candidate(candidate, world_model_control=wm_control) and wm_score_delta < 0.0:
            wm_score_delta *= 0.15
        baseline_score = float(candidate.get('opportunity_estimate', 0.0) or 0.0) - float(candidate.get('risk', 0.0) or 0.0)
        final_score = baseline_score + wm_score_delta

        candidate['wm_long_reward'] = wm_long_reward
        candidate['wm_risk'] = wm_risk
        candidate['wm_reversibility'] = wm_reversibility
        candidate['wm_info_gain'] = wm_info_gain
        candidate['wm_score_delta'] = wm_score_delta
        candidate['final_score'] = float(final_score)


def _apply_dynamic_world_model_constraints(
    candidates: List[Dict[str, Any]],
    *,
    hard_constraints: List[str],
    soft_constraints: List[str],
    uncertainty: float,
    world_model_control: Optional[WorldModelControlProtocol] = None,
) -> None:
    wm_control = world_model_control or WorldModelControlProtocol()
    hidden_focus = set(wm_control.hidden_focus_functions or [])
    control_trust = _clamp01(wm_control.control_trust, 0.5)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '')
        wm_risk = float(candidate.get('wm_risk', 0.0) or 0.0)
        wm_reversibility = float(candidate.get('wm_reversibility', 1.0) or 1.0)
        wm_info_gain = float(candidate.get('wm_info_gain', 0.0) or 0.0)
        probe_like = _is_probe_like_candidate(candidate)
        protected_active_step = _should_protect_active_procedure_candidate(candidate, world_model_control=wm_control)

        if (
            wm_risk >= _WM_HIGH_RISK_THRESHOLD
            and wm_reversibility <= _WM_LOW_REVERSIBILITY_THRESHOLD
            and fn_name
            and control_trust >= 0.62
            and not protected_active_step
        ):
            hard = f"no_{fn_name}"
            if hard not in hard_constraints:
                hard_constraints.append(hard)

        if wm_info_gain >= _WM_HIGH_INFO_GAIN_THRESHOLD and uncertainty >= _WM_HIGH_UNCERTAINTY_THRESHOLD:
            soft = "prefer_probe"
            if soft not in soft_constraints:
                soft_constraints.append(soft)

        if (
            wm_control.hidden_drift_score >= _WM_HIGH_DRIFT_THRESHOLD
            and not probe_like
            and fn_name
            and fn_name not in hidden_focus
            and fn_name != 'wait'
            and control_trust >= 0.68
            and not protected_active_step
        ):
            hard = f"no_{fn_name}"
            if hard not in hard_constraints:
                hard_constraints.append(hard)

        if wm_control.hidden_state_phase in {'exploring', 'disrupted'} and probe_like:
            soft = "prefer_probe_hidden_state"
            if soft not in soft_constraints:
                soft_constraints.append(soft)

        if hidden_focus and fn_name in hidden_focus and wm_control.hidden_state_phase in {'stabilizing', 'committed'}:
            soft = "prefer_hidden_focus_function"
            if soft not in soft_constraints:
                soft_constraints.append(soft)

        if protected_active_step:
            soft = "preserve_active_procedure_under_low_trust_world_model"
            if soft not in soft_constraints:
                soft_constraints.append(soft)


def _apply_deliberation_scoring(
    governance_candidates: List[Dict[str, Any]],
    *,
    raw_candidates: Sequence[Dict[str, Any]],
) -> None:
    _apply_active_procedure_guidance(raw_candidates)
    support_map = _build_signature_support_map(raw_candidates)
    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        support = support_map.get(_action_signature(raw_action), set())
        profile = _compute_deliberation_profile(raw_action, support)

        risk = max(0.0, min(1.0, float(candidate.get('risk', 0.05) or 0.05)))
        opportunity = max(0.0, min(1.0, float(candidate.get('opportunity_estimate', 0.1) or 0.1)))
        wm_delta = float(candidate.get('wm_score_delta', 0.0) or 0.0)
        deliberation_score = float(profile.get('score', 0.5) or 0.5)
        ambiguity_penalty = float(profile.get('ambiguity_penalty', 0.0) or 0.0)
        clarity = float(profile.get('executable_clarity', 0.5) or 0.5)
        support_strength = float(profile.get('consensus_support', 0.0) or 0.0)
        prediction_alignment = float(profile.get('prediction_alignment', 0.5) or 0.5)
        counterfactual_alignment = float(profile.get('counterfactual_alignment', 0.5) or 0.5)
        reasoning_depth = float(profile.get('reasoning_depth', 0.0) or 0.0)
        state_model_clarity = float(profile.get('state_model_clarity', 0.0) or 0.0)
        state_transition_alignment = float(profile.get('state_transition_alignment', 0.5) or 0.5)
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
        procedure_alignment_strength = max(0.0, min(1.0, float(procedure_guidance.get('alignment_strength', 0.0) or 0.0)))

        risk_adjustment = (
            (ambiguity_penalty * 0.18)
            + ((1.0 - clarity) * 0.06)
            + ((1.0 - state_model_clarity) * 0.05)
            - (support_strength * 0.05)
            - (prediction_alignment * 0.04)
            - (counterfactual_alignment * 0.03)
            - (state_transition_alignment * 0.04)
        )
        opportunity_adjustment = (
            (clarity * 0.08)
            + (support_strength * 0.10)
            + (prediction_alignment * 0.10)
            + (counterfactual_alignment * 0.08)
            + (reasoning_depth * 0.08)
            + (state_model_clarity * 0.08)
            + (state_transition_alignment * 0.10)
            - (ambiguity_penalty * 0.12)
        )
        deliberation_bonus = (deliberation_score - 0.5) * 0.40

        if bool(procedure_guidance.get('active_next_step')):
            risk_adjustment -= procedure_alignment_strength * 0.08
            opportunity_adjustment += procedure_alignment_strength * 0.10
            deliberation_bonus += procedure_alignment_strength * 0.05
        elif bool(procedure_guidance.get('conflicts_active_procedure')):
            risk_adjustment += procedure_alignment_strength * 0.10
            opportunity_adjustment -= procedure_alignment_strength * 0.08
            deliberation_bonus -= procedure_alignment_strength * 0.04

        candidate['risk'] = max(0.0, min(1.0, risk + risk_adjustment))
        candidate['opportunity_estimate'] = max(0.0, min(1.0, opportunity + opportunity_adjustment))
        candidate['deliberation'] = dict(profile)
        candidate['deliberation_bonus'] = deliberation_bonus
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + wm_delta
            + deliberation_bonus
        )

        if isinstance(raw_action, dict):
            meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
            meta['deliberation'] = dict(profile)
            meta['deliberation_bonus'] = deliberation_bonus
            raw_action['_candidate_meta'] = meta
            candidate['raw_action'] = raw_action


def _apply_active_procedure_arbitration(
    governance_candidates: List[Dict[str, Any]],
    *,
    world_model_control: Optional[WorldModelControlProtocol],
    soft_constraints: Optional[List[str]] = None,
) -> None:
    wm_control = world_model_control or WorldModelControlProtocol()
    control_trust = _clamp01(wm_control.control_trust, 0.5)
    hidden_drift = _clamp01(getattr(wm_control, 'hidden_drift_score', 0.0), 0.0)
    hidden_uncertainty = _clamp01(getattr(wm_control, 'hidden_uncertainty_score', 0.0), 0.0)
    state_shift_risk = _clamp01(getattr(wm_control, 'state_shift_risk', 0.0), 0.0)
    required_probes = list(dict.fromkeys(
        str(item) for item in list(getattr(wm_control, 'required_probes', []) or []) if str(item or '')
    ))
    probe_pressure = min(1.0, len(required_probes) / 3.0)
    latent_rows = _world_model_latent_branch_rows(wm_control)
    dominant_branch_id = str(getattr(wm_control, 'dominant_branch_id', '') or '').strip()
    dominant_branch: Dict[str, Any] = {}
    for row in latent_rows:
        if dominant_branch_id and str(row.get('branch_id', '') or '') == dominant_branch_id:
            dominant_branch = dict(row)
            break
    if not dominant_branch and latent_rows:
        dominant_branch = dict(latent_rows[0])
    dominant_anchor_functions = set(dominant_branch.get('anchor_functions', []) or [])
    dominant_risky_functions = set(dominant_branch.get('risky_functions', []) or [])
    dominant_branch_id = str(dominant_branch.get('branch_id', dominant_branch_id) or '').strip()
    dominant_branch_confidence = _clamp01(dominant_branch.get('confidence', 0.0), 0.0)
    latent_instability = _clamp01(
        (1.0 - dominant_branch_confidence) * 0.38
        + hidden_drift * 0.28
        + hidden_uncertainty * 0.22
        + state_shift_risk * 0.12,
        0.0,
    )
    soft_constraints = soft_constraints if isinstance(soft_constraints, list) else []

    active_strength = max(
        (_candidate_active_procedure_strength(candidate) for candidate in governance_candidates if isinstance(candidate, dict)),
        default=0.0,
    )
    if active_strength < 0.72:
        return

    arbitration_pressure = _clamp01(
        max(0.0, 0.78 - control_trust)
        + hidden_drift * 0.16
        + hidden_uncertainty * 0.14
        + state_shift_risk * 0.12
        + probe_pressure * 0.10
        + latent_instability * 0.12,
        0.0,
    )
    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        procedure_guidance = meta.get('procedure_guidance', {}) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
        prediction = meta.get('prediction', {}) if isinstance(meta.get('prediction', {}), dict) else {}
        synthetic_support = str(meta.get('synthetic_support', '') or '').strip()
        counterfactual_advantage = bool(meta.get('counterfactual_advantage', False))
        prediction_source = str(prediction.get('source', '') or '').strip()
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip()
        probe_like = _is_probe_like_candidate(candidate)
        verification_like = _is_verification_like_candidate(candidate)
        branch_anchor_match = bool(fn_name and fn_name in dominant_anchor_functions)
        branch_risky_match = bool(fn_name and fn_name in dominant_risky_functions)
        probe_aligned = bool(probe_pressure >= 0.34 and (probe_like or verification_like))

        delta = 0.0
        if bool(procedure_guidance.get('active_next_step')):
            delta += 0.14 + (active_strength * 0.16) + (arbitration_pressure * 0.18)
            if branch_anchor_match:
                delta += 0.04 + latent_instability * 0.05
            if probe_aligned:
                delta += 0.03 + probe_pressure * 0.05
            if 'prefer_active_procedure_arbitration' not in soft_constraints:
                soft_constraints.append('prefer_active_procedure_arbitration')
        elif bool(procedure_guidance.get('conflicts_active_procedure')):
            delta -= 0.18 + (active_strength * 0.18) + (arbitration_pressure * 0.24)
            if branch_risky_match:
                delta -= 0.05 + max(hidden_drift, latent_instability) * 0.08
            if branch_anchor_match:
                delta += 0.04 + probe_pressure * 0.04
            if probe_aligned:
                delta += 0.05 + probe_pressure * 0.08
                if 'allow_probe_under_active_procedure_arbitration' not in soft_constraints:
                    soft_constraints.append('allow_probe_under_active_procedure_arbitration')
            if synthetic_support:
                delta -= 0.06
            if counterfactual_advantage:
                delta -= 0.05
            if prediction_source:
                delta -= 0.05

        if delta == 0.0:
            continue

        risk = float(candidate.get('risk', 0.05) or 0.05)
        opportunity = float(candidate.get('opportunity_estimate', 0.1) or 0.1)
        if delta >= 0.0:
            opportunity = max(0.0, min(1.0, opportunity + delta * 0.42))
            risk = max(0.0, min(1.0, risk - delta * 0.22))
        else:
            penalty = abs(delta)
            opportunity = max(0.0, min(1.0, opportunity - penalty * 0.30))
            risk = max(0.0, min(1.0, risk + penalty * 0.34))

        candidate['risk'] = risk
        candidate['opportunity_estimate'] = opportunity
        candidate['active_procedure_arbitration_delta'] = float(delta)
        candidate['active_procedure_arbitration'] = {
            'active_strength': float(active_strength),
            'control_trust': float(control_trust),
            'required_probe_count': len(required_probes),
            'probe_pressure': float(probe_pressure),
            'hidden_drift_score': float(hidden_drift),
            'hidden_uncertainty_score': float(hidden_uncertainty),
            'state_shift_risk': float(state_shift_risk),
            'latent_branch_id': dominant_branch_id,
            'latent_branch_confidence': float(dominant_branch_confidence),
            'latent_branch_instability': float(latent_instability),
            'branch_anchor_match': bool(branch_anchor_match),
            'branch_risky_match': bool(branch_risky_match),
            'probe_aligned': bool(probe_aligned),
            'prediction_source': prediction_source,
            'synthetic_support': synthetic_support,
            'counterfactual_advantage': counterfactual_advantage,
        }
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + float(candidate.get('wm_score_delta', 0.0) or 0.0)
            + float(candidate.get('deliberation_bonus', 0.0) or 0.0)
        )

        if isinstance(raw_action, dict):
            meta['active_procedure_arbitration_delta'] = float(delta)
            meta['active_procedure_arbitration'] = dict(candidate['active_procedure_arbitration'])
            raw_action['_candidate_meta'] = meta
            candidate['raw_action'] = raw_action


def _apply_meta_control_scoring(
    governance_candidates: List[Dict[str, Any]],
    *,
    policy_profile: Dict[str, Any],
    soft_constraints: Optional[List[str]] = None,
) -> None:
    profile = policy_profile if isinstance(policy_profile, dict) else {}
    strategy_mode = str(profile.get('strategy_mode', 'balanced') or 'balanced')
    verification_bias = max(0.0, min(1.0, float(profile.get('verification_bias', 0.5) or 0.5)))
    risk_tolerance = max(0.0, min(1.0, float(profile.get('risk_tolerance', 0.5) or 0.5)))
    recovery_bias = max(0.0, min(1.0, float(profile.get('recovery_bias', 0.5) or 0.5)))
    stability_bias = max(0.0, min(1.0, float(profile.get('stability_bias', 0.5) or 0.5)))
    soft_constraints = soft_constraints if isinstance(soft_constraints, list) else []

    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '')
        probe_like = _is_probe_like_candidate(candidate)
        verification_like = _is_verification_like_candidate(candidate)
        wait_like = str(candidate.get('action_kind') or '') == 'wait' or fn_name == 'wait'
        execution_like = not probe_like and not verification_like and not wait_like
        delta = 0.0

        if strategy_mode == 'recover':
            if probe_like or wait_like or verification_like:
                delta += 0.10 + recovery_bias * 0.12
            if execution_like:
                delta -= 0.08 + (1.0 - risk_tolerance) * 0.08
            if 'prefer_recovery_meta' not in soft_constraints:
                soft_constraints.append('prefer_recovery_meta')
        elif strategy_mode == 'verify':
            if probe_like or verification_like:
                delta += 0.08 + verification_bias * 0.12
            if execution_like:
                delta -= 0.05 + verification_bias * 0.06
            if 'prefer_verification_meta' not in soft_constraints:
                soft_constraints.append('prefer_verification_meta')
        elif strategy_mode == 'explore':
            if probe_like:
                delta += 0.05 + verification_bias * 0.06
            if execution_like and risk_tolerance >= 0.58:
                delta += 0.04 + (risk_tolerance - 0.5) * 0.08
            if 'prefer_exploration_meta' not in soft_constraints:
                soft_constraints.append('prefer_exploration_meta')
        elif strategy_mode == 'exploit':
            if execution_like:
                delta += 0.08 + risk_tolerance * 0.10 + stability_bias * 0.04
            if probe_like or verification_like:
                delta -= 0.05 + verification_bias * 0.04
            if 'prefer_execution_meta' not in soft_constraints:
                soft_constraints.append('prefer_execution_meta')
        else:
            if verification_bias >= 0.66 and (probe_like or verification_like):
                delta += 0.04
            if recovery_bias >= 0.66 and wait_like:
                delta += 0.03

        risk = float(candidate.get('risk', 0.05) or 0.05)
        opportunity = float(candidate.get('opportunity_estimate', 0.1) or 0.1)
        if delta >= 0.0:
            opportunity = max(0.0, min(1.0, opportunity + delta * 0.42))
            risk = max(0.0, min(1.0, risk - delta * 0.22))
        else:
            penalty = abs(delta)
            opportunity = max(0.0, min(1.0, opportunity - penalty * 0.28))
            risk = max(0.0, min(1.0, risk + penalty * 0.30))

        candidate['risk'] = risk
        candidate['opportunity_estimate'] = opportunity
        candidate['meta_control_delta'] = float(delta)
        candidate['meta_control_guidance'] = {
            'strategy_mode': strategy_mode,
            'verification_bias': verification_bias,
            'risk_tolerance': risk_tolerance,
            'recovery_bias': recovery_bias,
            'stability_bias': stability_bias,
        }
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + float(candidate.get('wm_score_delta', 0.0) or 0.0)
            + float(candidate.get('deliberation_bonus', 0.0) or 0.0)
        )

        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        if isinstance(raw_action, dict):
            meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
            meta['meta_control_guidance'] = dict(candidate['meta_control_guidance'])
            meta['meta_control_delta'] = float(delta)
            raw_action['_candidate_meta'] = meta
            candidate['raw_action'] = raw_action


def _failure_strategy_payload(strategy: Any) -> Dict[str, Any]:
    if isinstance(strategy, dict):
        return dict(strategy)
    if hasattr(strategy, 'to_dict'):
        try:
            payload = strategy.to_dict()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _failure_preference_class(candidate: Dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return 'wait'
    fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip().lower()
    action_kind = str(candidate.get('action_kind') or '').strip().lower()
    if action_kind == 'wait' or fn_name == 'wait':
        return 'wait'
    if _is_verification_like_candidate(candidate):
        return 'probe'
    if any(token in fn_name for token in ('stat', 'count', 'measure', 'summary', 'score', 'analy', 'calc', 'compute')):
        return 'compute_stats'
    return 'action'


def _apply_failure_preference_scoring(
    governance_candidates: List[Dict[str, Any]],
    *,
    global_failure_strategy: Any,
    world_model_control: Optional[WorldModelControlProtocol] = None,
    soft_constraints: Optional[List[str]] = None,
) -> None:
    global_profile = _failure_strategy_payload(global_failure_strategy)
    global_mode = str(global_profile.get('strategy_mode_hint', 'balanced') or 'balanced')
    global_branch_budget = max(0, int(global_profile.get('branch_budget_hint', 0) or 0))
    global_verification_budget = max(0, int(global_profile.get('verification_budget_hint', 0) or 0))
    global_safe_fallback_class = str(global_profile.get('safe_fallback_class', '') or '')
    global_blocked = set(global_profile.get('blocked_action_classes', [])) if isinstance(global_profile.get('blocked_action_classes', []), list) else set()
    global_preferred_verification = list(global_profile.get('preferred_verification_functions', [])) if isinstance(global_profile.get('preferred_verification_functions', []), list) else []
    global_preferred_fallback = list(global_profile.get('preferred_fallback_functions', [])) if isinstance(global_profile.get('preferred_fallback_functions', []), list) else []
    wm_control = world_model_control or WorldModelControlProtocol()
    control_trust = _clamp01(getattr(wm_control, 'control_trust', 0.5), 0.5)
    transition_confidence = _clamp01(getattr(wm_control, 'transition_confidence', 0.5), 0.5)
    hidden_drift = _clamp01(getattr(wm_control, 'hidden_drift_score', 0.0), 0.0)
    hidden_uncertainty = _clamp01(getattr(wm_control, 'hidden_uncertainty_score', 0.0), 0.0)
    state_shift_risk = _clamp01(getattr(wm_control, 'state_shift_risk', 0.0), 0.0)
    required_probes = list(dict.fromkeys(
        str(item) for item in list(getattr(wm_control, 'required_probes', []) or []) if str(item or '')
    ))
    probe_pressure = min(1.0, len(required_probes) / 3.0)
    latent_rows = _world_model_latent_branch_rows(wm_control)
    dominant_branch_id = str(getattr(wm_control, 'dominant_branch_id', '') or '').strip()
    dominant_branch: Dict[str, Any] = {}
    for row in latent_rows:
        if dominant_branch_id and str(row.get('branch_id', '') or '') == dominant_branch_id:
            dominant_branch = dict(row)
            break
    if not dominant_branch and latent_rows:
        dominant_branch = dict(latent_rows[0])
    dominant_branch_id = str(dominant_branch.get('branch_id', dominant_branch_id) or '').strip()
    dominant_branch_confidence = _clamp01(dominant_branch.get('confidence', 0.0), 0.0)
    dominant_anchor_functions = set(dominant_branch.get('anchor_functions', []) or [])
    dominant_risky_functions = set(dominant_branch.get('risky_functions', []) or [])
    latent_instability = _clamp01(
        (1.0 - dominant_branch_confidence) * 0.38
        + hidden_drift * 0.28
        + hidden_uncertainty * 0.22
        + state_shift_risk * 0.12,
        0.0,
    )
    world_model_probe_pressure = (
        probe_pressure >= 0.34
        and (
            control_trust <= 0.52
            or transition_confidence <= 0.48
            or hidden_drift >= 0.55
            or hidden_uncertainty >= 0.62
            or latent_instability >= 0.58
            or state_shift_risk >= 0.58
        )
    )
    soft_constraints = soft_constraints if isinstance(soft_constraints, list) else []

    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        candidate_profile = _failure_strategy_payload(meta.get('failure_strategy_profile', {}))
        strategy_mode = str(candidate_profile.get('strategy_mode_hint', global_mode) or global_mode)
        branch_budget = max(0, int(candidate_profile.get('branch_budget_hint', global_branch_budget) or global_branch_budget))
        verification_budget = max(0, int(candidate_profile.get('verification_budget_hint', global_verification_budget) or global_verification_budget))
        safe_fallback_class = str(candidate_profile.get('safe_fallback_class', global_safe_fallback_class) or global_safe_fallback_class)
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '')
        wait_like = _failure_preference_class(candidate) == 'wait'
        verification_like = _is_verification_like_candidate(candidate)
        execution_like = not wait_like and not verification_like
        candidate_blocked = set(candidate_profile.get('blocked_action_classes', [])) if isinstance(candidate_profile.get('blocked_action_classes', []), list) else set()
        blocked = action_matches_blocked_name(raw_action, global_blocked) or action_matches_blocked_name(raw_action, candidate_blocked)
        preferred_verification = fn_name in set(global_preferred_verification[:2]) or fn_name in set(candidate_profile.get('preferred_verification_functions', [])[:2])
        preferred_fallback = fn_name in set(global_preferred_fallback[:2]) or fn_name in set(candidate_profile.get('preferred_fallback_functions', [])[:2])
        failure_class = _failure_preference_class(candidate)
        branch_anchor_match = bool(fn_name and fn_name in dominant_anchor_functions)
        branch_risky_match = bool(fn_name and fn_name in dominant_risky_functions)
        delta = 0.0

        if blocked:
            delta -= 0.22
        if preferred_verification:
            delta += 0.11 + min(0.10, verification_budget * 0.03)
            if 'prefer_failure_verification' not in soft_constraints:
                soft_constraints.append('prefer_failure_verification')
        if preferred_fallback:
            delta += 0.07 + min(0.07, branch_budget * 0.02)
            if 'prefer_failure_fallback' not in soft_constraints:
                soft_constraints.append('prefer_failure_fallback')
        if safe_fallback_class and failure_class == safe_fallback_class:
            delta += 0.04
        if strategy_mode == 'recover':
            if wait_like or verification_like:
                delta += 0.06
            elif execution_like and not preferred_fallback and not preferred_verification:
                delta -= 0.07
        elif strategy_mode == 'verify':
            if verification_like:
                delta += 0.06
            elif execution_like and not preferred_fallback:
                delta -= 0.04
        elif strategy_mode == 'exploit':
            if execution_like and preferred_fallback:
                delta += 0.04
            elif verification_like and not preferred_verification:
                delta -= 0.03

        if world_model_probe_pressure:
            if verification_like:
                delta += 0.06 + probe_pressure * 0.08 + min(0.08, verification_budget * 0.025)
            elif wait_like and safe_fallback_class == 'wait':
                delta += 0.02 + min(0.05, branch_budget * 0.015)
            elif execution_like and not preferred_fallback and not preferred_verification:
                delta -= 0.06 + latent_instability * 0.10
            if preferred_verification:
                delta += 0.03 + latent_instability * 0.04
            if branch_anchor_match and (verification_like or preferred_verification):
                delta += 0.04 + dominant_branch_confidence * 0.03
            if branch_risky_match:
                delta -= 0.05 + max(hidden_drift, latent_instability) * 0.06
            if 'prefer_failure_probe_pressure' not in soft_constraints:
                soft_constraints.append('prefer_failure_probe_pressure')

        risk = float(candidate.get('risk', 0.05) or 0.05)
        opportunity = float(candidate.get('opportunity_estimate', 0.1) or 0.1)
        if delta >= 0.0:
            opportunity = max(0.0, min(1.0, opportunity + delta * 0.38))
            risk = max(0.0, min(1.0, risk - delta * 0.20))
        else:
            penalty = abs(delta)
            opportunity = max(0.0, min(1.0, opportunity - penalty * 0.26))
            risk = max(0.0, min(1.0, risk + penalty * 0.28))

        candidate['risk'] = risk
        candidate['opportunity_estimate'] = opportunity
        candidate['failure_preference_delta'] = float(delta)
        candidate['failure_preference_guidance'] = {
            'strategy_mode': strategy_mode,
            'branch_budget_hint': branch_budget,
            'verification_budget_hint': verification_budget,
            'safe_fallback_class': safe_fallback_class,
            'preferred_verification': bool(preferred_verification),
            'preferred_fallback': bool(preferred_fallback),
            'blocked': bool(blocked),
            'world_model_probe_pressure': bool(world_model_probe_pressure),
            'required_probe_count': len(required_probes),
            'probe_pressure': float(probe_pressure),
            'control_trust': float(control_trust),
            'transition_confidence': float(transition_confidence),
            'hidden_drift_score': float(hidden_drift),
            'hidden_uncertainty_score': float(hidden_uncertainty),
            'state_shift_risk': float(state_shift_risk),
            'latent_branch_id': dominant_branch_id,
            'latent_branch_confidence': float(dominant_branch_confidence),
            'latent_branch_instability': float(latent_instability),
            'branch_anchor_match': bool(branch_anchor_match),
            'branch_risky_match': bool(branch_risky_match),
        }
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + float(candidate.get('wm_score_delta', 0.0) or 0.0)
            + float(candidate.get('deliberation_bonus', 0.0) or 0.0)
        )

        if isinstance(raw_action, dict):
            meta['failure_preference_guidance'] = dict(candidate['failure_preference_guidance'])
            meta['failure_preference_delta'] = float(delta)
            raw_action['_candidate_meta'] = meta
            candidate['raw_action'] = raw_action


def _apply_hidden_state_scoring(
    governance_candidates: List[Dict[str, Any]],
    *,
    world_model_control: WorldModelControlProtocol,
    soft_constraints: Optional[List[str]] = None,
) -> None:
    wm_control = world_model_control or WorldModelControlProtocol()
    hidden_focus = set(wm_control.hidden_focus_functions or [])
    hidden_phase = str(wm_control.hidden_state_phase or '')
    hidden_depth = max(0, int(wm_control.hidden_state_depth or 0))
    hidden_phase_conf = max(0.0, min(1.0, float(wm_control.hidden_phase_confidence or 0.0)))
    hidden_drift = max(0.0, min(1.0, float(wm_control.hidden_drift_score or 0.0)))
    hidden_uncertainty = max(0.0, min(1.0, float(wm_control.hidden_uncertainty_score or 0.0)))
    expected_next_phase = str(getattr(wm_control, 'expected_next_phase', '') or '')
    expected_next_phase_conf = max(0.0, min(1.0, float(getattr(wm_control, 'expected_next_phase_confidence', 0.0) or 0.0)))
    transition_entropy = max(0.0, min(1.0, float(getattr(wm_control, 'phase_transition_entropy', 1.0) or 1.0)))
    stabilizing_focus = set(getattr(wm_control, 'stabilizing_focus_functions', []) or [])
    risky_focus = set(getattr(wm_control, 'risky_focus_functions', []) or [])
    influence_factor = _world_model_influence_factor(wm_control)
    soft_constraints = soft_constraints if isinstance(soft_constraints, list) else []

    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '')
        probe_like = _is_probe_like_candidate(candidate)
        wait_like = str(candidate.get('action_kind') or '') == 'wait' or fn_name == 'wait'
        delta = 0.0

        if fn_name and fn_name in hidden_focus and hidden_phase in {'stabilizing', 'committed'}:
            delta += min(0.26, hidden_depth * 0.05) * max(0.4, hidden_phase_conf)
        if fn_name and fn_name in stabilizing_focus:
            delta += (0.08 + expected_next_phase_conf * 0.10) * max(0.4, influence_factor)
            if expected_next_phase in {'stabilizing', 'committed'}:
                delta += 0.04 + expected_next_phase_conf * 0.06
        if fn_name and fn_name in risky_focus:
            delta -= 0.09 + max(hidden_drift, 1.0 - expected_next_phase_conf) * 0.07
        if probe_like and hidden_phase in {'exploring', 'disrupted'}:
            delta += 0.12 + hidden_drift * 0.10
        if probe_like and transition_entropy >= 0.65:
            delta += 0.06
        if wait_like and hidden_phase == 'disrupted':
            delta += 0.04 + hidden_drift * 0.06
        if hidden_phase == 'disrupted' and not probe_like and not wait_like and fn_name not in hidden_focus:
            delta -= 0.12 * max(0.4, hidden_phase_conf) + hidden_drift * 0.10
        if hidden_phase == 'committed' and hidden_depth >= 2 and not probe_like and fn_name not in hidden_focus and fn_name:
            delta -= 0.06 + min(0.08, hidden_depth * 0.015)
        if expected_next_phase == 'disrupted' and not probe_like and not wait_like and fn_name not in stabilizing_focus:
            delta -= 0.08 + expected_next_phase_conf * 0.06
        if hidden_uncertainty >= 0.72 and probe_like:
            delta += 0.08
        delta *= influence_factor
        if _should_protect_active_procedure_candidate(candidate, world_model_control=wm_control) and delta < 0.0:
            delta *= 0.15

        risk = float(candidate.get('risk', 0.05) or 0.05)
        opportunity = float(candidate.get('opportunity_estimate', 0.1) or 0.1)
        if delta >= 0.0:
            opportunity = max(0.0, min(1.0, opportunity + delta * 0.45))
            risk = max(0.0, min(1.0, risk - delta * 0.25))
        else:
            penalty = abs(delta)
            opportunity = max(0.0, min(1.0, opportunity - penalty * 0.30))
            risk = max(0.0, min(1.0, risk + penalty * 0.35))

        candidate['risk'] = risk
        candidate['opportunity_estimate'] = opportunity
        candidate['hidden_state_delta'] = float(delta)
        candidate['hidden_state_guidance'] = {
            'phase': hidden_phase,
            'depth': hidden_depth,
            'phase_confidence': hidden_phase_conf,
            'drift_score': hidden_drift,
            'focus_functions': sorted(hidden_focus),
            'latent_signature': str(wm_control.hidden_latent_signature or ''),
            'dominant_branch_id': str(getattr(wm_control, 'dominant_branch_id', '') or ''),
            'latent_branches': [dict(item) for item in list(getattr(wm_control, 'latent_branches', []) or []) if isinstance(item, dict)],
            'expected_next_phase': expected_next_phase,
            'expected_next_phase_confidence': expected_next_phase_conf,
            'transition_entropy': transition_entropy,
            'stabilizing_functions': sorted(stabilizing_focus),
            'risky_functions': sorted(risky_focus),
        }
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + float(candidate.get('wm_score_delta', 0.0) or 0.0)
            + float(candidate.get('deliberation_bonus', 0.0) or 0.0)
        )

        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        if isinstance(raw_action, dict):
            meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
            meta['hidden_state_guidance'] = dict(candidate['hidden_state_guidance'])
            meta['hidden_state_delta'] = float(delta)
            raw_action['_candidate_meta'] = meta
            candidate['raw_action'] = raw_action

    if hidden_phase in {'exploring', 'disrupted'} and 'probe_hidden_state_transition' in set(wm_control.required_probes or []):
        if 'prefer_probe_hidden_state' not in soft_constraints:
            soft_constraints.append('prefer_probe_hidden_state')
    if transition_entropy >= 0.65 and 'prefer_probe_hidden_state' not in soft_constraints:
        soft_constraints.append('prefer_probe_hidden_state')


def _is_commit_like_function(fn_name: str) -> bool:
    name = str(fn_name or '').strip().lower()
    if not name:
        return False
    return any(token in name for token in ('commit', 'apply', 'submit', 'advance', 'finalize', 'seal'))


def _step_constraints(step: Any) -> Dict[str, Any]:
    raw = getattr(step, 'constraints', {})
    return dict(raw) if isinstance(raw, dict) else {}


def _world_model_latent_branch_rows(world_model_control: WorldModelControlProtocol) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(getattr(world_model_control, 'latent_branches', []) or [])[:4]:
        if not isinstance(item, dict):
            continue
        rows.append({
            'branch_id': str(item.get('branch_id', '') or '').strip(),
            'target_phase': str(item.get('target_phase', '') or '').strip().lower(),
            'confidence': _clamp01(item.get('confidence', 0.0), 0.0),
            'anchor_functions': [
                str(fn or '').strip()
                for fn in (item.get('anchor_functions', []) if isinstance(item.get('anchor_functions', []), list) else [])
                if str(fn or '').strip()
            ],
            'risky_functions': [
                str(fn or '').strip()
                for fn in (item.get('risky_functions', []) if isinstance(item.get('risky_functions', []), list) else [])
                if str(fn or '').strip()
            ],
        })
    return rows


def _resolve_belief_branch_guidance(
    *,
    current_step: Any,
    world_model_control: WorldModelControlProtocol,
) -> Dict[str, Any]:
    constraints = _step_constraints(current_step) if current_step is not None else {}
    belief_branch_id = str(constraints.get('belief_branch_id', '') or '').strip()
    belief_target_phase = str(constraints.get('belief_target_phase', '') or '').strip().lower()
    belief_branch_confidence = _clamp01(constraints.get('belief_branch_confidence', 0.0), 0.0)
    belief_anchor_functions = [
        str(fn or '').strip()
        for fn in (constraints.get('belief_anchor_functions', []) if isinstance(constraints.get('belief_anchor_functions', []), list) else [])
        if str(fn or '').strip()
    ]
    frontier = constraints.get('branch_frontier', []) if isinstance(constraints.get('branch_frontier', []), list) else []
    frontier_targets: List[str] = []
    frontier_confidence: Dict[str, float] = {}
    frontier_branch_ids: Dict[str, str] = {}
    for row in frontier[:3]:
        if not isinstance(row, dict):
            continue
        fn_name = str(row.get('target_function', '') or '').strip()
        if not fn_name:
            residual_chain = row.get('residual_chain', []) if isinstance(row.get('residual_chain', []), list) else []
            fn_name = str(residual_chain[0] if residual_chain else '' or '').strip()
        if not fn_name or fn_name in frontier_targets:
            continue
        frontier_targets.append(fn_name)
        frontier_confidence[fn_name] = _clamp01(row.get('belief_branch_confidence', 0.0), belief_branch_confidence)
        frontier_branch_ids[fn_name] = str(row.get('belief_branch_id', '') or '').strip()

    latent_rows = _world_model_latent_branch_rows(world_model_control)
    dominant_branch_id = str(getattr(world_model_control, 'dominant_branch_id', '') or '').strip()
    selected_latent = {}
    for row in latent_rows:
        if dominant_branch_id and str(row.get('branch_id', '') or '') == dominant_branch_id:
            selected_latent = dict(row)
            break
    if not selected_latent and latent_rows:
        selected_latent = dict(latent_rows[0])

    branch_source = 'planner'
    belief_risky_functions: List[str] = []
    target_fn = str(getattr(current_step, 'target_function', '') or '').strip()

    if belief_branch_id:
        for row in latent_rows:
            if str(row.get('branch_id', '') or '') == belief_branch_id:
                if not belief_target_phase:
                    belief_target_phase = str(row.get('target_phase', '') or '').strip().lower()
                if belief_branch_confidence <= 0.0:
                    belief_branch_confidence = _clamp01(row.get('confidence', 0.0), 0.0)
                if not belief_anchor_functions:
                    belief_anchor_functions = list(row.get('anchor_functions', []) or [])
                belief_risky_functions = list(row.get('risky_functions', []) or [])
                break
    elif selected_latent:
        belief_branch_id = str(selected_latent.get('branch_id', '') or '').strip()
        belief_target_phase = str(selected_latent.get('target_phase', '') or '').strip().lower()
        belief_branch_confidence = _clamp01(selected_latent.get('confidence', 0.0), 0.0)
        belief_anchor_functions = list(selected_latent.get('anchor_functions', []) or [])
        belief_risky_functions = list(selected_latent.get('risky_functions', []) or [])
        branch_source = 'world_model'
        if not target_fn and belief_anchor_functions:
            target_fn = str(belief_anchor_functions[0] or '').strip()

    if branch_source == 'world_model' and latent_rows:
        for row in latent_rows[:3]:
            branch_id = str(row.get('branch_id', '') or '').strip()
            anchor_functions = list(row.get('anchor_functions', []) or [])
            if not anchor_functions or branch_id == belief_branch_id:
                continue
            fn_name = str(anchor_functions[0] or '').strip()
            if not fn_name or fn_name in frontier_targets:
                continue
            frontier_targets.append(fn_name)
            frontier_confidence[fn_name] = _clamp01(row.get('confidence', 0.0), belief_branch_confidence)
            frontier_branch_ids[fn_name] = branch_id

    return {
        'belief_branch_id': belief_branch_id,
        'belief_target_phase': belief_target_phase,
        'belief_branch_confidence': belief_branch_confidence,
        'belief_anchor_functions': belief_anchor_functions,
        'belief_risky_functions': belief_risky_functions,
        'frontier_targets': frontier_targets,
        'frontier_confidence': frontier_confidence,
        'frontier_branch_ids': frontier_branch_ids,
        'branch_source': branch_source,
        'target_function': target_fn,
    }


def _apply_belief_branch_scoring(
    governance_candidates: List[Dict[str, Any]],
    *,
    current_step: Any,
    world_model_control: WorldModelControlProtocol,
    soft_constraints: Optional[List[str]] = None,
) -> None:
    branch_guidance = _resolve_belief_branch_guidance(
        current_step=current_step,
        world_model_control=world_model_control,
    )
    belief_branch_id = str(branch_guidance.get('belief_branch_id', '') or '').strip()
    belief_target_phase = str(branch_guidance.get('belief_target_phase', '') or '').strip().lower()
    belief_branch_confidence = _clamp01(branch_guidance.get('belief_branch_confidence', 0.0), 0.0)
    belief_anchor_functions = [
        str(fn or '').strip()
        for fn in (branch_guidance.get('belief_anchor_functions', []) if isinstance(branch_guidance.get('belief_anchor_functions', []), list) else [])
        if str(fn or '').strip()
    ]
    belief_risky_functions = [
        str(fn or '').strip()
        for fn in (branch_guidance.get('belief_risky_functions', []) if isinstance(branch_guidance.get('belief_risky_functions', []), list) else [])
        if str(fn or '').strip()
    ]
    frontier_targets = [
        str(fn or '').strip()
        for fn in (branch_guidance.get('frontier_targets', []) if isinstance(branch_guidance.get('frontier_targets', []), list) else [])
        if str(fn or '').strip()
    ]
    frontier_confidence = dict(branch_guidance.get('frontier_confidence', {}) or {}) if isinstance(branch_guidance.get('frontier_confidence', {}), dict) else {}
    frontier_branch_ids = dict(branch_guidance.get('frontier_branch_ids', {}) or {}) if isinstance(branch_guidance.get('frontier_branch_ids', {}), dict) else {}
    branch_source = str(branch_guidance.get('branch_source', 'planner') or 'planner').strip()
    if not belief_branch_id and not belief_anchor_functions and not frontier_targets:
        return

    hidden_phase = str(world_model_control.hidden_state_phase or '').strip().lower()
    hidden_drift = _clamp01(world_model_control.hidden_drift_score, 0.0)
    hidden_uncertainty = _clamp01(world_model_control.hidden_uncertainty_score, 0.0)
    soft_constraints = soft_constraints if isinstance(soft_constraints, list) else []
    anchor_set = set(belief_anchor_functions)
    risky_set = set(belief_risky_functions)
    frontier_set = set(frontier_targets)
    target_fn = str(branch_guidance.get('target_function', '') or '').strip()

    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip()
        probe_like = _is_probe_like_candidate(candidate)
        wait_like = str(candidate.get('action_kind') or '').strip().lower() == 'wait' or fn_name == 'wait'
        commit_like = _is_commit_like_function(fn_name)
        anchor_match = bool(fn_name and fn_name in anchor_set)
        risky_match = bool(fn_name and fn_name in risky_set)
        frontier_match = bool(fn_name and fn_name in frontier_set)
        meta_anchor_match = bool(meta.get('planner_belief_anchor_match', False))
        candidate_branch_id = str(meta.get('planner_belief_branch_id', '') or '').strip()
        anchor_index = belief_anchor_functions.index(fn_name) if anchor_match and fn_name in belief_anchor_functions else -1

        delta = 0.0
        if risky_match:
            delta -= 0.10 + belief_branch_confidence * 0.10 + hidden_drift * 0.04
            if branch_source == 'world_model' and 'avoid_latent_branch_risk' not in soft_constraints:
                soft_constraints.append('avoid_latent_branch_risk')
        if anchor_match or meta_anchor_match:
            anchor_bonus = 0.07 + belief_branch_confidence * 0.11
            if anchor_index == 0:
                anchor_bonus += 0.05
            elif branch_source == 'world_model' and anchor_index > 0 and hidden_phase in {'exploring', 'disrupted'}:
                anchor_bonus -= min(0.06, anchor_index * 0.03)
            delta += anchor_bonus
        if frontier_match:
            delta += 0.08 + frontier_confidence.get(fn_name, belief_branch_confidence) * 0.10
            if 'prefer_belief_branch_alternative' not in soft_constraints:
                soft_constraints.append('prefer_belief_branch_alternative')
        if belief_target_phase in {'exploring', 'disrupted'}:
            if probe_like:
                delta += 0.10 + hidden_uncertainty * 0.08 + hidden_drift * 0.06
                if 'prefer_belief_branch_probe' not in soft_constraints:
                    soft_constraints.append('prefer_belief_branch_probe')
            elif commit_like:
                delta -= 0.12 + belief_branch_confidence * 0.10
                if anchor_match and not frontier_match:
                    delta -= 0.06
                if branch_source == 'world_model' and anchor_index > 0:
                    delta -= min(0.08, anchor_index * 0.03)
            elif not wait_like and not anchor_match and not frontier_match and not probe_like:
                delta -= 0.06 + hidden_drift * 0.06
        elif belief_target_phase == 'stabilizing':
            if probe_like or 'calibrate' in fn_name.lower() or 'tune' in fn_name.lower():
                delta += 0.06 + belief_branch_confidence * 0.08
            if commit_like and not anchor_match and not frontier_match:
                delta -= 0.08
        elif belief_target_phase == 'committed':
            setup_required = bool(belief_anchor_functions) and belief_anchor_functions[0] != target_fn
            if commit_like and fn_name == target_fn and not setup_required:
                delta += 0.08 + belief_branch_confidence * 0.08
            elif commit_like and branch_source == 'world_model' and anchor_index > 0 and target_fn and fn_name != target_fn:
                delta -= 0.12 + belief_branch_confidence * 0.08
            elif commit_like and setup_required and not anchor_match and not frontier_match:
                delta -= 0.14 + belief_branch_confidence * 0.10
            elif anchor_match or frontier_match:
                delta += 0.05 + belief_branch_confidence * 0.06
                if branch_source == 'world_model' and anchor_index > 0 and target_fn and fn_name != target_fn:
                    delta -= min(0.08, anchor_index * 0.03)

        if hidden_phase in {'disrupted', 'exploring'} and frontier_match:
            delta += 0.05
        if candidate_branch_id and belief_branch_id and candidate_branch_id == belief_branch_id and anchor_match:
            delta += 0.03
        if candidate_branch_id and frontier_match:
            frontier_branch_id = frontier_branch_ids.get(fn_name, '')
            if frontier_branch_id and frontier_branch_id == candidate_branch_id:
                delta += 0.03

        if delta == 0.0:
            continue

        risk = float(candidate.get('risk', 0.05) or 0.05)
        opportunity = float(candidate.get('opportunity_estimate', 0.1) or 0.1)
        if delta >= 0.0:
            opportunity = max(0.0, min(1.0, opportunity + delta * 0.40))
            risk = max(0.0, min(1.0, risk - delta * 0.22))
        else:
            penalty = abs(delta)
            opportunity = max(0.0, min(1.0, opportunity - penalty * 0.26))
            risk = max(0.0, min(1.0, risk + penalty * 0.28))

        candidate['risk'] = risk
        candidate['opportunity_estimate'] = opportunity
        candidate['belief_branch_delta'] = float(delta)
        candidate['belief_branch_guidance'] = {
            'belief_branch_id': belief_branch_id,
            'belief_target_phase': belief_target_phase,
            'belief_branch_confidence': belief_branch_confidence,
            'branch_source': branch_source,
            'belief_anchor_functions': list(belief_anchor_functions),
            'belief_risky_functions': list(belief_risky_functions),
            'frontier_targets': list(frontier_targets),
            'anchor_match': anchor_match or meta_anchor_match,
            'risky_match': risky_match,
            'frontier_match': frontier_match,
            'hidden_phase': hidden_phase,
        }
        candidate['final_score'] = (
            float(candidate.get('opportunity_estimate', 0.0) or 0.0)
            - float(candidate.get('risk', 0.0) or 0.0)
            + float(candidate.get('wm_score_delta', 0.0) or 0.0)
            + float(candidate.get('deliberation_bonus', 0.0) or 0.0)
        )

        meta['belief_branch_delta'] = float(delta)
        meta['belief_branch_guidance'] = dict(candidate['belief_branch_guidance'])
        raw_action['_candidate_meta'] = meta
        candidate['raw_action'] = raw_action


def _apply_active_procedure_guidance(raw_candidates: Sequence[Dict[str, Any]]) -> None:
    procedure_steps: List[Tuple[float, str, Dict[str, Any]]] = []
    actionable_functions: Set[str] = set()

    for action in raw_candidates:
        if not isinstance(action, dict):
            continue
        fn_name = extract_action_function_name(action, default='').strip()
        action_kind = extract_action_kind(action, default='call_tool')
        if action_kind == 'call_tool' and fn_name and fn_name != 'wait':
            actionable_functions.add(fn_name)

        meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        proc = meta.get('procedure', {}) if isinstance(meta.get('procedure', {}), dict) else {}
        if str(action.get('_source', '') or '') != 'procedure_reuse':
            continue
        if not bool(proc.get('is_next_step', False)) or not fn_name:
            continue
        procedure_steps.append((_procedure_alignment_strength(proc), fn_name, action))

    if not procedure_steps:
        return

    procedure_steps.sort(key=lambda row: row[0], reverse=True)
    preferred_strength, preferred_fn, preferred_action = procedure_steps[0]
    conflicting_functions = sorted(fn for fn in actionable_functions if fn != preferred_fn)

    for action in raw_candidates:
        if not isinstance(action, dict):
            continue
        meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        procedure_guidance = dict(meta.get('procedure_guidance', {})) if isinstance(meta.get('procedure_guidance', {}), dict) else {}
        fn_name = extract_action_function_name(action, default='').strip()
        action_kind = extract_action_kind(action, default='call_tool')

        if action is preferred_action:
            procedure_guidance.update({
                'active_next_step': True,
                'conflicts_active_procedure': False,
                'preferred_function': preferred_fn,
                'alignment_strength': preferred_strength,
                'conflicting_functions': list(conflicting_functions),
            })
        elif action_kind == 'call_tool' and fn_name and fn_name != 'wait' and fn_name != preferred_fn:
            procedure_guidance.update({
                'active_next_step': False,
                'conflicts_active_procedure': True,
                'preferred_function': preferred_fn,
                'alignment_strength': preferred_strength,
            })

        if procedure_guidance:
            meta['procedure_guidance'] = procedure_guidance
            action['_candidate_meta'] = meta


def _procedure_alignment_strength(proc: Dict[str, Any]) -> float:
    success_rate = max(0.0, min(1.0, float(proc.get('success_rate', 0.0) or 0.0)))
    failure_rate = max(0.0, min(1.0, float(proc.get('failure_rate', 0.0) or 0.0)))
    procedure_bonus = max(0.0, min(1.0, float(proc.get('procedure_bonus', 0.0) or 0.0)))
    strength = (procedure_bonus * 1.5) + (success_rate * 0.45) - (failure_rate * 0.25)
    if bool(proc.get('is_next_step', False)):
        strength += 0.10
    return max(0.0, min(1.0, strength))


def _build_signature_support_map(raw_candidates: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Set[str]]:
    support: Dict[Tuple[str, str], Set[str]] = {}
    for action in raw_candidates:
        if not isinstance(action, dict):
            continue
        signature = _action_signature(action)
        source = str(action.get('_source', 'unknown') or 'unknown')
        support.setdefault(signature, set()).add(source)
    return support


def _action_signature(action: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(action, dict):
        return ('wait', '{}')
    action_kind = extract_action_kind(action, default='call_tool')
    if action_kind == 'wait':
        return ('wait', '{}')
    function_name = extract_action_function_name(action, default='wait')
    payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    return (function_name, repr(kwargs))


def _is_probe_like_candidate(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    action_kind = str(candidate.get('action_kind') or '').strip().lower()
    fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip().lower()
    if action_kind == 'probe':
        return True
    if fn_name in {'probe', 'inspect', 'sample', 'test'}:
        return True
    return any(token in fn_name for token in ('probe', 'inspect', 'sample', 'test', 'check'))


def _compute_deliberation_profile(action: Dict[str, Any], support_sources: Set[str]) -> Dict[str, float]:
    if not isinstance(action, dict):
        return {
            'executable_clarity': 0.0,
            'consensus_support': 0.0,
            'prediction_alignment': 0.0,
            'counterfactual_alignment': 0.0,
            'reasoning_depth': 0.0,
            'ambiguity_penalty': 1.0,
            'score': 0.0,
        }
    action_kind = extract_action_kind(action, default='call_tool')
    is_wait = action_kind == 'wait'
    payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    function_name = extract_action_function_name(action, default='wait')
    kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args.get('kwargs', {}), dict) else {}
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    signature_allows_empty_kwargs = _signature_allows_empty_kwargs(meta)
    state_summary = summarize_action_state(action)
    structured_state = meta.get('structured_answer_state_abstraction', {}) if isinstance(meta.get('structured_answer_state_abstraction', {}), dict) else {}
    engine_rank = int(meta.get('deliberation_engine_rank', 0) or 0)
    engine_score = float(meta.get('deliberation_engine_score', 0.0) or 0.0)

    executable_clarity = 0.6 if is_wait else 0.2
    if function_name and function_name != 'wait':
        executable_clarity += 0.25
    if isinstance(kwargs, dict) and kwargs:
        executable_clarity += 0.20
    elif signature_allows_empty_kwargs:
        executable_clarity += 0.18
    if bool(meta.get('structured_answer_synthesized')):
        executable_clarity += 0.15
    if bool(meta.get('structured_answer_internal_simulation')):
        executable_clarity += 0.05
    executable_clarity = max(0.0, min(1.0, executable_clarity))

    state_model_clarity = 0.1
    if state_summary:
        state_model_clarity += 0.10
    if int(state_summary.get('grid_like_payloads', 0) or 0) > 0:
        state_model_clarity += 0.25
    if int(state_summary.get('max_value_depth', 0) or 0) >= 2:
        state_model_clarity += 0.10
    if structured_state:
        state_model_clarity += 0.20
    if signature_allows_empty_kwargs:
        state_model_clarity += 0.08
    first_grid_summary = structured_state.get('first_output_state', {}) if isinstance(structured_state.get('first_output_state', {}), dict) else state_summary.get('first_grid_summary', {})
    if isinstance(first_grid_summary, dict) and first_grid_summary:
        object_signal = float(first_grid_summary.get('color_component_count', 0) or 0.0)
        if object_signal > 0:
            state_model_clarity += min(0.25, object_signal * 0.04)
    state_model_clarity = max(0.0, min(1.0, state_model_clarity))

    support_count = len(support_sources)
    consensus_support = 0.0 if support_count <= 1 else min(1.0, (support_count - 1) / 2.0)
    if engine_rank > 0:
        consensus_support = max(consensus_support, max(0.0, 0.9 - ((engine_rank - 1) * 0.15)))

    prediction = meta.get('prediction', {}) if isinstance(meta.get('prediction', {}), dict) else {}
    pred_success = _nested_prediction_value(prediction, 'success', default=0.5)
    pred_confidence = max(0.0, min(1.0, float(prediction.get('overall_confidence', 0.5) or 0.5)))
    pred_info_gain = _nested_prediction_value(prediction, 'information_gain', default=0.5)
    prediction_alignment = max(
        0.0,
        min(1.0, (pred_success * 0.55) + (pred_confidence * 0.25) + (pred_info_gain * 0.20)),
    )

    cf_delta = float(meta.get('counterfactual_delta', 0.0) or 0.0)
    cf_advantage = 0.12 if bool(meta.get('counterfactual_advantage')) else 0.0
    cf_governance = meta.get('counterfactual_governance', {}) if isinstance(meta.get('counterfactual_governance', {}), dict) else {}
    cf_confidence = max(0.0, min(1.0, float(cf_governance.get('confidence', 0.0) or 0.0)))
    counterfactual_alignment = max(0.0, min(1.0, 0.5 + (cf_delta * 0.35) + cf_advantage + (cf_confidence * 0.15)))

    state_transition_alignment = 0.5
    if 'structured_answer_transition_alignment' in meta:
        state_transition_alignment = max(0.0, min(1.0, float(meta.get('structured_answer_transition_alignment', 0.5) or 0.5)))
    elif isinstance(structured_state.get('transition_alignment'), (int, float)):
        state_transition_alignment = max(0.0, min(1.0, float(structured_state.get('transition_alignment', 0.5) or 0.5)))
    elif int(state_summary.get('grid_like_payloads', 0) or 0) > 0:
        state_transition_alignment = 0.55

    reasoning_channels = 0
    if prediction:
        reasoning_channels += 1
    if 'counterfactual_delta' in meta or cf_governance:
        reasoning_channels += 1
    if bool(meta.get('structured_answer_synthesized')):
        reasoning_channels += 1
    if bool(meta.get('structured_answer_internal_simulation')):
        reasoning_channels += 1
    if state_model_clarity >= 0.45:
        reasoning_channels += 1
    sim_steps = int(meta.get('structured_answer_simulation_steps', 1) or 1)
    reasoning_depth = max(0.0, min(1.0, (reasoning_channels * 0.18) + (max(0, sim_steps - 1) * 0.12)))

    ambiguity_penalty = 0.0
    if action_kind not in {'wait', 'inspect', 'probe', 'call_tool'}:
        ambiguity_penalty += 0.20
    if not is_wait and (not function_name or function_name == 'wait'):
        ambiguity_penalty += 0.35
    if not is_wait and not kwargs and not signature_allows_empty_kwargs:
        ambiguity_penalty += 0.18
    if function_name.startswith(('submit', 'answer', 'solve')) and not bool(meta.get('structured_answer_synthesized')):
        ambiguity_penalty += 0.22
    if engine_score:
        ambiguity_penalty -= max(0.0, (engine_score - 0.5) * 0.25)
    ambiguity_penalty = max(0.0, min(1.0, ambiguity_penalty))

    score = max(
        0.0,
        min(
            1.0,
            0.20
            + (executable_clarity * 0.24)
            + (consensus_support * 0.14)
            + (prediction_alignment * 0.22)
            + (counterfactual_alignment * 0.20)
            + (reasoning_depth * 0.16)
            + (state_model_clarity * 0.16)
            + (state_transition_alignment * 0.12)
            + max(0.0, (engine_score - 0.5) * 0.12)
            - (ambiguity_penalty * 0.30),
        ),
    )
    return {
        'executable_clarity': executable_clarity,
        'consensus_support': consensus_support,
        'prediction_alignment': prediction_alignment,
        'counterfactual_alignment': counterfactual_alignment,
        'reasoning_depth': reasoning_depth,
        'state_model_clarity': state_model_clarity,
        'state_transition_alignment': state_transition_alignment,
        'ambiguity_penalty': ambiguity_penalty,
        'score': score,
    }


def _signature_allows_empty_kwargs(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    if bool(meta.get('signature_allows_empty_kwargs', False)):
        return True
    if not bool(meta.get('signature_known', False)):
        return False
    required_kwargs = meta.get('required_kwargs', [])
    missing_required_kwargs = meta.get('missing_required_kwargs', [])
    return isinstance(required_kwargs, list) and not required_kwargs and isinstance(missing_required_kwargs, list) and not missing_required_kwargs


def _is_verification_like_candidate(candidate: Dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    fn_name = str(candidate.get('function_name') or candidate.get('action') or '').strip().lower()
    action_kind = str(candidate.get('action_kind') or '').strip().lower()
    if action_kind == 'inspect':
        return True
    return any(token in fn_name for token in ('inspect', 'verify', 'test', 'probe', 'check'))


def _nested_prediction_value(payload: Dict[str, Any], key: str, *, default: float) -> float:
    node = payload.get(key, {}) if isinstance(payload.get(key, {}), dict) else {}
    raw = node.get('value', default) if isinstance(node, dict) else default
    try:
        return max(0.0, min(1.0, float(raw)))
    except Exception:
        return default


def _counterfactual_advice_payload(counterfactual_outcome: Any) -> Dict[str, Any]:
    if counterfactual_outcome is None:
        return {}
    confidence_obj = getattr(counterfactual_outcome, 'confidence', None)
    confidence_value = _counterfactual_confidence_score(confidence_obj)
    return {
        'preferred_action': str(getattr(counterfactual_outcome, 'preferred_action', '') or ''),
        'confidence': float(confidence_value),
        'reasoning': str(getattr(counterfactual_outcome, 'reasoning', '') or ''),
        'estimated_delta': float(getattr(counterfactual_outcome, 'estimated_delta', 0.0) or 0.0),
        'decision_path': str(getattr(counterfactual_outcome, 'decision_path', '') or ''),
        'rollout_trace': [
            dict(item) for item in list(getattr(counterfactual_outcome, 'rollout_trace', []) or [])[:8]
            if isinstance(item, dict)
        ],
        'rollout_summary': dict(getattr(counterfactual_outcome, 'rollout_summary', {}) or {}),
    }


def _resolve_counterfactual_policy(meta_control_state: Dict[str, Any]) -> Dict[str, Any]:
    raw = meta_control_state.get('counterfactual_policy', {}) if isinstance(meta_control_state.get('counterfactual_policy', {}), dict) else {}
    return {
        'top_k': max(1, int(raw.get('top_k', 2) or 2)),
        'high_confidence_threshold': max(0.0, min(1.0, float(raw.get('high_confidence_threshold', 0.75) or 0.75))),
        'opposition_policy': str(raw.get('opposition_policy', 'hard_override') or 'hard_override'),
        'downscore_penalty': max(0.0, float(raw.get('downscore_penalty', 0.35) or 0.35)),
    }


def compare_candidate_counterfactuals(
    *,
    loop,
    counterfactual_port: Optional[CounterfactualPort],
    state_slice: Any,
    governance_candidates: List[Dict[str, Any]],
    candidate_index: int,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if not counterfactual_port or candidate_index < 0 or candidate_index >= len(governance_candidates):
        return {}
    candidate = governance_candidates[candidate_index]
    candidate_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    candidate_name = str(candidate.get('function_name') or candidate.get('action') or 'wait')
    if not isinstance(candidate_action, dict):
        return {}
    baselines = _counterfactual_baselines(governance_candidates, candidate_index)
    comparisons: List[Dict[str, Any]] = []
    preferred_action = candidate_name
    max_confidence = 0.0
    best_reasoning = ''
    best_delta = 0.0
    best_outcome_obj = None
    for baseline in baselines:
        baseline_action = baseline.get('raw_action', {})
        outcome = counterfactual_port.simulate_action_difference(
            state_slice,
            candidate_action,
            baseline_action,
            context={**context, 'counterfactual_baseline': baseline.get('label', '')},
        )
        outcome_payload = _counterfactual_advice_payload(outcome)
        confidence = _counterfactual_confidence_score(getattr(outcome, 'confidence', None))
        preferred = str(outcome_payload.get('preferred_action') or '')
        estimated_delta = float(getattr(outcome, 'estimated_delta', 0.0) or 0.0)
        comparisons.append({
            'against': baseline.get('label', ''),
            'against_action': baseline.get('function_name', 'wait'),
            'preferred_action': preferred,
            'confidence': confidence,
            'reasoning': outcome_payload.get('reasoning', ''),
            'estimated_delta': estimated_delta,
            'decision_path': outcome_payload.get('decision_path', ''),
            'rollout_trace': list(outcome_payload.get('rollout_trace', [])),
        })
        if confidence >= max_confidence and preferred and preferred != candidate_name:
            max_confidence = confidence
            preferred_action = preferred
            best_reasoning = str(outcome_payload.get('reasoning') or '')
            best_delta = estimated_delta
            best_outcome_obj = outcome
    return {
        'preferred_action': preferred_action,
        'confidence': max_confidence,
        'reasoning': best_reasoning,
        'estimated_delta': best_delta,
        'comparisons': comparisons,
        'decision_path': str((_counterfactual_advice_payload(best_outcome_obj) if best_outcome_obj is not None else {}).get('decision_path', '') or ''),
        'rollout_trace': list((_counterfactual_advice_payload(best_outcome_obj) if best_outcome_obj is not None else {}).get('rollout_trace', [])),
        'rollout_summary': dict((_counterfactual_advice_payload(best_outcome_obj) if best_outcome_obj is not None else {}).get('rollout_summary', {})),
        'outcome_obj': best_outcome_obj,
    }


def _counterfactual_baselines(governance_candidates: List[Dict[str, Any]], candidate_index: int) -> List[Dict[str, Any]]:
    baselines: List[Dict[str, Any]] = [{'label': 'wait', 'function_name': 'wait', 'raw_action': {'kind': 'wait', 'payload': {}}}]
    probe_candidates = [c for c in governance_candidates if str(c.get('function_name') or '').find('probe') >= 0]
    if probe_candidates:
        best_probe = max(probe_candidates, key=lambda c: float(c.get('final_score', 0.0) or 0.0))
        baselines.append({'label': 'best_probe', 'function_name': best_probe.get('function_name', 'wait'), 'raw_action': best_probe.get('raw_action', {'kind': 'wait', 'payload': {}})})
    safer_candidates = [
        c for idx, c in enumerate(governance_candidates)
        if idx != candidate_index and str(c.get('function_name') or 'wait') != 'wait'
    ]
    if safer_candidates:
        best_safer = min(safer_candidates, key=lambda c: float(c.get('risk', 1.0) or 1.0))
        baselines.append({'label': 'best_safer_action', 'function_name': best_safer.get('function_name', 'wait'), 'raw_action': best_safer.get('raw_action', {'kind': 'wait', 'payload': {}})})
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for baseline in baselines:
        key = (baseline.get('label', ''), baseline.get('function_name', 'wait'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(baseline)
    return deduped


def _attach_counterfactual_to_candidate(candidate: Dict[str, Any], candidate_cf: Dict[str, Any]) -> None:
    payload = {
        'preferred_action': str(candidate_cf.get('preferred_action') or ''),
        'confidence': float(candidate_cf.get('confidence', 0.0) or 0.0),
        'reasoning': str(candidate_cf.get('reasoning') or ''),
        'estimated_delta': float(candidate_cf.get('estimated_delta', 0.0) or 0.0),
        'decision_path': str(candidate_cf.get('decision_path') or ''),
        'rollout_trace': list(candidate_cf.get('rollout_trace', [])),
        'rollout_summary': dict(candidate_cf.get('rollout_summary', {}) or {}),
        'comparisons': list(candidate_cf.get('comparisons', [])),
    }
    candidate['counterfactual'] = payload
    raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
    meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
    meta['counterfactual_governance'] = payload
    raw_action['_candidate_meta'] = meta
    candidate['raw_action'] = raw_action


def _apply_counterfactual_penalties(
    governance_candidates: List[Dict[str, Any]],
    candidate_counterfactual_map: Dict[int, Dict[str, Any]],
    policy: Dict[str, Any],
) -> None:
    threshold = float(policy.get('high_confidence_threshold', 0.75) or 0.75)
    penalty = float(policy.get('downscore_penalty', 0.35) or 0.35)
    for idx, candidate in enumerate(governance_candidates):
        advice = candidate_counterfactual_map.get(idx, {})
        preferred = str(advice.get('preferred_action') or '')
        confidence = float(advice.get('confidence', 0.0) or 0.0)
        fn_name = str(candidate.get('function_name') or candidate.get('action') or 'wait')
        if not preferred or preferred == fn_name or confidence < threshold:
            continue
        candidate['final_score'] = float(candidate.get('final_score', 0.0) or 0.0) - penalty
        candidate['counterfactual_downscored'] = True
        candidate['counterfactual_downscore_penalty'] = penalty


def _counterfactual_confidence_score(confidence_obj: Any) -> float:
    raw = getattr(confidence_obj, 'value', confidence_obj)
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    text = str(raw or '').strip().lower()
    if text == 'high':
        return 0.9
    if text == 'medium':
        return 0.6
    if text == 'low':
        return 0.3
    return 0.0


def _audit_organ_control_decisions(
    loop,
    organ_capability_port: Optional[OrganCapabilityPort],
    governance_state: Optional[GovernanceState],
    decision_outcome: Any,
    governance_candidates: List[Dict[str, Any]],
    selected_action_name: str,
    selection_reason: str,
    counterfactual_outcome: Any = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    selected_name = str(selected_action_name or 'wait')

    def _append_entry(organ: str, suggested_action: str, rationale: str) -> None:
        capability = _capability_for_organ(organ_capability_port, governance_state, organ)
        suggested = str(suggested_action or 'wait')
        adopted = suggested == selected_name
        low_risk = _is_low_risk_control_action(suggested)
        rejection_reason = ''
        if not adopted:
            rejection_reason = f"governance_selected={selected_name}; reason={selection_reason}"
        if capability == CAPABILITY_ADVISORY and adopted:
            rejection_reason = 'advisory_mode_only'
            adopted = False
        elif capability == CAPABILITY_CONSTRAINED_CONTROL and not low_risk:
            rejection_reason = 'constrained_control_non_low_risk'
            adopted = False
        entry = {
            'episode': loop._episode,
            'tick': loop._tick,
            'organ': organ,
            'capability': capability,
            'suggested_action': suggested,
            'adopted': adopted,
            'rejected': not adopted,
            'reason': rationale if adopted else rejection_reason or rationale,
            'expected_consequence': 'pending_execution',
            'formal_write_guard': 'proposal_validator_required',
        }
        entries.append(entry)
        loop._pending_organ_control_audit.append(entry)
        loop._organ_control_audit_log.append(dict(entry))

    if decision_outcome and getattr(decision_outcome, 'selected_candidate', None):
        selected_candidate = decision_outcome.selected_candidate
        source_name = str(getattr(getattr(selected_candidate, 'source', None), 'value', '') or '')
        suggested = str(getattr(selected_candidate, 'function_name', '') or 'wait')
        organ = _source_to_organ(source_name)
        if organ:
            _append_entry(organ, suggested, f"decision_arbiter_source={source_name}")

    if counterfactual_outcome:
        _append_entry(
            'world_model',
            str(getattr(counterfactual_outcome, 'preferred_action', '') or 'wait'),
            'counterfactual_preference',
        )

    best_prediction = None
    for candidate in governance_candidates:
        if not isinstance(candidate, dict):
            continue
        raw_action = candidate.get('raw_action', {}) if isinstance(candidate.get('raw_action', {}), dict) else {}
        meta = raw_action.get('_candidate_meta', {}) if isinstance(raw_action.get('_candidate_meta', {}), dict) else {}
        prediction = meta.get('prediction', {}) if isinstance(meta.get('prediction', {}), dict) else {}
        score = float(prediction.get('overall_confidence', 0.0) or 0.0)
        fn = str(candidate.get('function_name') or candidate.get('action') or 'wait')
        if best_prediction is None or score > best_prediction[1]:
            best_prediction = (fn, score)
    if best_prediction:
        _append_entry('prediction', best_prediction[0], f"prediction_confidence={best_prediction[1]:.3f}")
    return entries

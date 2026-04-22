from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from core.orchestration.commit_candidate_guard import (
    high_confidence_commit_guard_reason,
    is_probe_like,
    procedure_guard_reasons,
)
from modules.world_model.mechanism_runtime import mechanism_obs_state
from modules.hypothesis.mechanism_posterior_updater import (
    binding_token_frequency,
    candidate_binding_signal,
)

_KNOWN_MECHANISM_ROLES = {
    "commit",
    "discriminate",
    "recovery",
    "prerequisite",
    "prepare",
    "wait",
}


def _observed_function_role(function_name: str, obs_before: Optional[Dict[str, Any]]) -> str:
    if not function_name or not isinstance(obs_before, dict):
        return ""
    function_signatures = obs_before.get("function_signatures", {})
    if hasattr(function_signatures, "raw"):
        function_signatures = function_signatures.raw
    if not isinstance(function_signatures, dict):
        return ""
    signature = function_signatures.get(function_name, {})
    if hasattr(signature, "raw"):
        signature = signature.raw
    if not isinstance(signature, dict):
        return ""
    description = str(signature.get("description", "") or "").strip().lower()
    if not description:
        return ""
    if description in _KNOWN_MECHANISM_ROLES:
        return description
    for role_name in _KNOWN_MECHANISM_ROLES:
        if role_name in description:
            return role_name
    return ""


def _candidate_role(action: Any, *, obs_before: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(action, dict):
        return ""
    meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
    explicit_role = str(meta.get("role", "") or "").strip().lower()
    if explicit_role:
        return explicit_role
    inferred_role = _observed_function_role(_extract_action_function_name(action), obs_before)
    if inferred_role and isinstance(meta, dict):
        meta["role"] = inferred_role
        meta.setdefault("planner_observed_role", inferred_role)
    return inferred_role


@dataclass
class ConstraintInput:
    candidate_actions: List[Dict[str, Any]]
    obs_before: Optional[Dict[str, Any]]
    has_plan: bool
    current_step: Any
    tick: int
    episode: int
    mechanism_control_summary: Optional[Dict[str, Any]] = None


@dataclass
class ConstraintResult:
    filtered_candidates: List[Dict[str, Any]]
    viability_entry: Dict[str, Any]
    pending_replan_patch: Optional[Dict[str, Any]]


def check_step_condition(condition: Dict[str, Any], obs_before: Optional[Dict[str, Any]], action: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(condition, dict):
        return False
    key = str(condition.get('key', '') or '')
    expected = condition.get('equals')
    contains = condition.get('contains')

    value = None
    if key.startswith('obs.') and isinstance(obs_before, dict):
        cursor: Any = obs_before
        for part in key.split('.')[1:]:
            if isinstance(cursor, dict):
                cursor = cursor.get(part)
            else:
                cursor = None
            if cursor is None:
                break
        value = cursor
    elif key == 'action.function' and isinstance(action, dict):
        value = _extract_action_function_name(action)

    if expected is not None:
        return value == expected
    if contains is not None:
        try:
            return contains in value
        except TypeError:
            return False
    return bool(value)


def match_observation_pattern(pattern: Dict[str, Any], obs_before: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(pattern, dict):
        return False
    checks = pattern.get('checks', [])
    if not isinstance(checks, list) or not checks:
        return False
    for check in checks:
        if not check_step_condition(check, obs_before):
            return False
    expected_success = pattern.get('result_success')
    if expected_success is not None and isinstance(result, dict):
        return bool(result.get('success', False)) == bool(expected_success)
    return True


def _summarize_conditions(raw_conditions: List[Any]) -> List[str]:
    summary: List[str] = []
    for cond in raw_conditions if isinstance(raw_conditions, list) else []:
        if isinstance(cond, dict):
            key = str(cond.get('key') or cond.get('field') or cond.get('name') or cond.get('type') or 'dict_condition')
            op = str(cond.get('op') or cond.get('operator') or cond.get('relation') or '')
            val = cond.get('value', cond.get('target', cond.get('expected', '')))
            val_s = str(val)
            summary.append(f"{key}{op}{val_s}" if op else f"{key}:{val_s}")
        else:
            summary.append(str(cond))
    return summary[:8]


def _extract_action_function_name(action: Any) -> str:
    if not isinstance(action, dict):
        return ''
    if action.get('kind') == 'wait':
        return 'wait'
    payload = action.get('payload', {}) if isinstance(action.get('payload', {}), dict) else {}
    tool_args = payload.get('tool_args', {}) if isinstance(payload.get('tool_args', {}), dict) else {}
    fn_name = str(tool_args.get('function_name') or '').strip()
    if fn_name:
        return fn_name
    kind = str(action.get('kind', '') or '').strip().lower()
    if kind == 'inspect':
        return 'inspect'
    return ''


def _synthetic_wait_candidate() -> Dict[str, Any]:
    return {
        "kind": "wait",
        "payload": {},
        "_source": "planner_wait_synthesis",
        "_candidate_meta": {
            "role": "wait",
            "action_family": "wait",
            "planner_synthesized_wait": True,
        },
    }


def _binding_profiles_by_action(
    candidate_actions: List[Dict[str, Any]],
    *,
    obs_before: Optional[Dict[str, Any]],
    mechanism_control_summary: Optional[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    relevant = [
        action
        for action in list(candidate_actions or [])
        if isinstance(action, dict)
        and _candidate_role(action, obs_before=obs_before) in {'commit', 'recovery', 'prerequisite', 'prepare'}
    ]
    token_frequency = binding_token_frequency(relevant or candidate_actions)
    scored: Dict[int, Dict[str, Any]] = {}
    commit_scores_by_action: Dict[int, float] = {}
    for action in list(candidate_actions or []):
        if not isinstance(action, dict):
            continue
        profile = candidate_binding_signal(
            action,
            obs_before=obs_before,
            mechanism_control=mechanism_control_summary,
            token_frequency=token_frequency,
        )
        scored[id(action)] = profile
        role = _candidate_role(action, obs_before=obs_before)
        if role == 'commit':
            commit_scores_by_action[id(action)] = float(profile.get('score', 0.0) or 0.0)

    for action in list(candidate_actions or []):
        if not isinstance(action, dict):
            continue
        role = _candidate_role(action, obs_before=obs_before)
        profile = dict(scored.get(id(action), {}))
        own_score = float(profile.get('score', 0.0) or 0.0)
        runner_up = max(
            [score for action_id, score in commit_scores_by_action.items() if action_id != id(action)],
            default=0.0,
        ) if role == 'commit' else 0.0
        score_margin = own_score - runner_up
        profile['score_margin'] = round(float(score_margin), 6)
        control = dict(mechanism_control_summary or {})
        commitment_trust = float(control.get('commitment_trust', 0.0) or 0.0)
        commitment_revoked = bool(control.get('commitment_revoked', False))
        actionable = bool(
            role == 'commit'
            and not bool(profile.get('revoked_match', False))
            and float(profile.get('contradiction_penalty', 0.0) or 0.0) < 0.45
            and (
                (
                    bool(profile.get('active_match', False))
                    and not commitment_revoked
                    and commitment_trust >= 0.58
                )
                or (
                    float(profile.get('evidence_strength', 0.0) or 0.0) >= 0.85
                    and float(profile.get('specificity', 0.0) or 0.0) >= 0.45
                    and score_margin >= 0.18
                )
            )
        )
        profile['actionable'] = actionable
        scored[id(action)] = profile
        meta = action.setdefault('_candidate_meta', {})
        if isinstance(meta, dict):
            meta['mechanism_binding_score'] = round(float(profile.get('score', 0.0) or 0.0), 4)
            meta['mechanism_binding_margin'] = round(float(score_margin), 4)
            meta['mechanism_binding_specificity'] = round(float(profile.get('specificity', 0.0) or 0.0), 4)
            meta['mechanism_binding_evidence_strength'] = round(float(profile.get('evidence_strength', 0.0) or 0.0), 4)
            meta['mechanism_binding_actionable'] = bool(actionable)
            meta['mechanism_binding_support_matches'] = list(profile.get('support_matches', []) or [])[:4]
            meta['mechanism_binding_counter_matches'] = list(profile.get('counter_matches', []) or [])[:4]
    return scored


def _mechanism_release_reason(
    action: Any,
    obs_before: Optional[Dict[str, Any]],
    *,
    mechanism_control_summary: Optional[Dict[str, Any]] = None,
    binding_profile: Optional[Dict[str, Any]] = None,
) -> str:
    if not isinstance(action, dict):
        return ''

    obs_snapshot = dict(obs_before or {}) if isinstance(obs_before, dict) else {}
    mechanism_control = dict(mechanism_control_summary or {})
    obs_state = mechanism_obs_state(obs_snapshot, mechanism_control)
    profile = dict(binding_profile or {})
    meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
    role = _candidate_role(action, obs_before=obs_before)
    fn_name = _extract_action_function_name(action)
    kind = str(action.get('kind', '') or '').strip().lower()
    if not fn_name and kind != 'wait':
        return ''

    prerequisite_missing = bool(obs_state.get('prerequisite_ready', False))
    recovery_required = bool(obs_state.get('recovery_ready', False))
    delayed_pending = bool(obs_state.get('wait_ready', False))
    control_mode = str(mechanism_control.get('control_mode', '') or '')
    commitment_trust = float(mechanism_control.get('commitment_trust', 0.0) or 0.0)

    if kind == 'wait' or fn_name == 'wait':
        return 'mechanism_release_wait_pending' if delayed_pending or control_mode == 'wait' else ''
    if delayed_pending:
        return ''
    if not role:
        return ''
    if role == 'recovery' and (recovery_required or control_mode == 'recover'):
        return 'mechanism_release_recovery_required'
    if role in {'prerequisite', 'prepare'} and (prerequisite_missing or control_mode == 'prepare'):
        return 'mechanism_release_prerequisite_required'
    if role == 'commit':
        if prerequisite_missing or recovery_required or delayed_pending:
            return ''
        if bool(profile.get('revoked_match', False)):
            return ''
        if bool(profile.get('active_match', False)) and not bool(mechanism_control.get('commitment_revoked', False)) and commitment_trust >= 0.58 and float(profile.get('contradiction_penalty', 0.0) or 0.0) < 0.45:
            return 'mechanism_release_commit_followthrough'
        if bool(profile.get('actionable', False)):
            return 'mechanism_release_commit_signal_specific'
        return ''
    if role == 'discriminate' and control_mode == 'discriminate':
        return 'mechanism_release_discriminate_pending'
    if not role and control_mode == 'discriminate' and is_probe_like(fn_name, kind=kind):
        return 'mechanism_release_discriminate_pending'
    return ''


def _is_high_confidence_active_procedure_step(action: Any, visible_functions: Set[str]) -> bool:
    reason = high_confidence_commit_guard_reason(
        action,
        available_functions=visible_functions,
    )
    return reason in procedure_guard_reasons()


def apply_plan_constraints(input_obj: ConstraintInput) -> ConstraintResult:
    if not input_obj.has_plan or not input_obj.current_step:
        return ConstraintResult(filtered_candidates=input_obj.candidate_actions, viability_entry={}, pending_replan_patch=None)

    step = input_obj.current_step
    constraints = step.constraints if isinstance(step.constraints, dict) else {}
    target_fn = step.target_function
    fallback = constraints.get('fallback_functions', []) if isinstance(constraints.get('fallback_functions', []), list) else []
    belief_anchor_functions = [
        str(fn or '').strip()
        for fn in (constraints.get('belief_anchor_functions', []) if isinstance(constraints.get('belief_anchor_functions', []), list) else [])
        if str(fn or '').strip()
    ]
    belief_branch_confidence = float(constraints.get('belief_branch_confidence', 0.0) or 0.0)
    belief_anchor_enabled = bool(belief_anchor_functions) and belief_branch_confidence >= 0.45
    allowed = {fn for fn in [target_fn, *fallback] if fn and fn != 'combine'}
    if belief_anchor_enabled:
        allowed.update({fn for fn in belief_anchor_functions if fn and fn != 'combine'})
    allowed.add('wait')
    mechanism_control_summary = dict(input_obj.mechanism_control_summary or {})
    obs_state = mechanism_obs_state(
        dict(input_obj.obs_before or {}) if isinstance(input_obj.obs_before, dict) else {},
        mechanism_control_summary,
    )
    candidate_actions = list(input_obj.candidate_actions or []) if isinstance(input_obj.candidate_actions, list) else []
    delayed_pending = bool(obs_state.get('wait_ready', False))
    control_mode = str(mechanism_control_summary.get('control_mode', '') or '')
    has_wait_candidate = any(_extract_action_function_name(action) == 'wait' for action in candidate_actions)
    if not has_wait_candidate and (delayed_pending or control_mode == 'wait'):
        candidate_actions.append(_synthetic_wait_candidate())
        has_wait_candidate = True
    if delayed_pending and has_wait_candidate:
        allowed = {'wait'}
    binding_profiles = _binding_profiles_by_action(
        candidate_actions,
        obs_before=input_obj.obs_before,
        mechanism_control_summary=mechanism_control_summary,
    )
    mechanism_release_by_function: Dict[str, str] = {}
    for action in candidate_actions:
        meta = action.setdefault('_candidate_meta', {}) if isinstance(action, dict) else {}
        if isinstance(meta, dict):
            meta['mechanism_release_ready'] = bool(obs_state.get('release_ready', False))
            meta['mechanism_wait_ready'] = bool(obs_state.get('wait_ready', False))
            meta['mechanism_prerequisite_ready'] = bool(obs_state.get('prerequisite_ready', False))
            meta['mechanism_recovery_ready'] = bool(obs_state.get('recovery_ready', False))
        fn_name = _extract_action_function_name(action)
        if not fn_name:
            continue
        release_reason = _mechanism_release_reason(
            action,
            input_obj.obs_before,
            mechanism_control_summary=mechanism_control_summary,
            binding_profile=binding_profiles.get(id(action), {}),
        )
        if release_reason:
            mechanism_release_by_function.setdefault(fn_name, release_reason)
    allowed.update(mechanism_release_by_function.keys())
    allowed_without_wait = {fn for fn in allowed if fn != 'wait'}
    n_total_before = len(candidate_actions)
    n_non_wait_before = 0
    for action in candidate_actions:
        if not isinstance(action, dict) or action.get('kind') == 'wait':
            continue
        fn_name = _extract_action_function_name(action)
        if fn_name and fn_name != 'wait':
            n_non_wait_before += 1

    visible_functions: Set[str] = set()
    obs_novel_api = input_obj.obs_before.get('novel_api', {}) if isinstance(input_obj.obs_before, dict) else {}
    if hasattr(obs_novel_api, 'raw'):
        obs_novel_api = obs_novel_api.raw
    if isinstance(obs_novel_api, dict):
        raw_visible = obs_novel_api.get('visible_functions', [])
        if isinstance(raw_visible, list):
            visible_functions = {str(fn) for fn in raw_visible if fn}

    namespace_mismatch = bool(allowed_without_wait) and bool(visible_functions) and allowed_without_wait.isdisjoint(visible_functions)
    viability_entry = {
        'tick': input_obj.tick,
        'episode': input_obj.episode,
        'entry': 'plan_constraints_viability',
        'candidate_zeroed_at': None,
        'n_total_before': n_total_before,
        'n_non_wait_before': n_non_wait_before,
        'allowed': sorted(allowed),
        'visible_functions': sorted(visible_functions),
        'namespace_mismatch': namespace_mismatch,
        'target_function': str(target_fn or ''),
        'fallback_functions': [str(fn) for fn in fallback if fn],
        'belief_anchor_functions': list(belief_anchor_functions),
        'belief_anchor_enabled': belief_anchor_enabled,
        'belief_branch_id': str(constraints.get('belief_branch_id', '') or ''),
        'belief_target_phase': str(constraints.get('belief_target_phase', '') or ''),
        'mechanism_control_mode': str(mechanism_control_summary.get('control_mode', '') or ''),
        'mechanism_commitment_revoked': bool(mechanism_control_summary.get('commitment_revoked', False)),
        'mechanism_wait_ready': bool(obs_state.get('wait_ready', False)),
        'mechanism_prerequisite_ready': bool(obs_state.get('prerequisite_ready', False)),
        'mechanism_recovery_ready': bool(obs_state.get('recovery_ready', False)),
        'mechanism_release_ready': bool(obs_state.get('release_ready', False)),
        'mechanism_release_allowed_functions': sorted(mechanism_release_by_function),
        'preconditions_summary': _summarize_conditions(constraints.get('preconditions', [])),
        'forbidden_conditions_summary': _summarize_conditions(constraints.get('forbidden_conditions', [])),
    }

    pending_replan_patch = None
    if namespace_mismatch:
        preserved_namespace_mismatch_procedure_steps = 0
        preserved_namespace_mismatch_commit_candidates = 0
        salvaged_candidates: List[Dict[str, Any]] = []
        wait_candidates: List[Dict[str, Any]] = []
        for action in candidate_actions:
            if not isinstance(action, dict):
                continue
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['planner_namespace_mismatch'] = True
                meta['planner_constraint_bypassed'] = True
            if _extract_action_function_name(action) == 'wait':
                wait_candidates.append(action)
                continue
            guard_reason = high_confidence_commit_guard_reason(
                action,
                available_functions=visible_functions,
                plan_target_function=str(target_fn or ''),
            )
            if not guard_reason:
                continue
            if isinstance(meta, dict):
                meta['planner_constraint_kept'] = True
                meta['planner_constraint_bypassed'] = False
                meta['planner_constraint_override_reason'] = (
                    'namespace_mismatch_active_procedure_step'
                    if guard_reason in procedure_guard_reasons()
                    else f'namespace_mismatch_{guard_reason}'
                )
                meta['high_confidence_commit_guard'] = True
                meta['planner_allowed_set_size'] = len(allowed)
            salvaged_candidates.append(action)
            if guard_reason in procedure_guard_reasons():
                preserved_namespace_mismatch_procedure_steps += 1
            else:
                preserved_namespace_mismatch_commit_candidates += 1

        filtered_candidates = list(salvaged_candidates) + list(wait_candidates) if salvaged_candidates else candidate_actions
        n_non_wait_after = sum(
            1
            for action in filtered_candidates
            if isinstance(action, dict) and _extract_action_function_name(action) not in {'', 'wait'}
        )
        viability_entry.update({
            'n_total_after': len(filtered_candidates),
            'n_non_wait_after': n_non_wait_after,
            'fail_event': False,
            'preserved_active_procedure_steps': preserved_namespace_mismatch_procedure_steps,
            'preserved_high_confidence_commit_candidates': preserved_namespace_mismatch_commit_candidates,
            'namespace_mismatch_salvaged': bool(salvaged_candidates),
        })
        pending_replan_patch = None if salvaged_candidates else {
            'trigger': 'planner_namespace_mismatch',
            'tick': input_obj.tick,
            'target_function': target_fn,
            'fallback_functions': list(fallback),
            'visible_functions': sorted(visible_functions),
        }
        return ConstraintResult(filtered_candidates=filtered_candidates, viability_entry=viability_entry, pending_replan_patch=pending_replan_patch)

    preconditions = constraints.get('preconditions', []) if isinstance(constraints.get('preconditions', []), list) else []
    forbidden_conditions = constraints.get('forbidden_conditions', []) if isinstance(constraints.get('forbidden_conditions', []), list) else []

    filtered: List[Dict[str, Any]] = []
    preserved_active_procedure_candidates: List[Dict[str, Any]] = []
    wait_candidates: List[Dict[str, Any]] = []
    preserved_active_procedure_steps = 0
    preserved_high_confidence_commit_candidates = 0
    allowed_active_procedure_steps = 0
    for action in candidate_actions:
        fn_name = _extract_action_function_name(action) or 'wait'
        if fn_name == 'wait':
            wait_candidates.append(action)

        if allowed and fn_name not in allowed:
            if delayed_pending and has_wait_candidate:
                continue
            guard_reason = high_confidence_commit_guard_reason(
                action,
                available_functions=visible_functions,
                plan_target_function=str(target_fn or ''),
            )
            if guard_reason:
                meta = action.setdefault('_candidate_meta', {})
                if isinstance(meta, dict):
                    meta['planner_constraint_bypassed'] = True
                    meta['planner_constraint_override_reason'] = (
                        'high_confidence_active_procedure_step'
                        if guard_reason in procedure_guard_reasons()
                        else guard_reason
                    )
                    meta['high_confidence_commit_guard'] = True
                    meta['planner_allowed_set_size'] = len(allowed)
                if guard_reason in procedure_guard_reasons():
                    preserved_active_procedure_steps += 1
                    preserved_active_procedure_candidates.append(action)
                else:
                    preserved_high_confidence_commit_candidates += 1
                filtered.append(action)
                continue
            continue

        pre_ok = all(check_step_condition(cond, input_obj.obs_before, action) for cond in preconditions) if preconditions else True
        forbidden_hit = any(check_step_condition(cond, input_obj.obs_before, action) for cond in forbidden_conditions) if forbidden_conditions else False

        if pre_ok and not forbidden_hit:
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['planner_constraint_kept'] = True
                meta['planner_allowed_set_size'] = len(allowed)
                meta['planner_preconditions_passed'] = pre_ok
                meta['planner_forbidden_blocked'] = forbidden_hit
                release_reason = mechanism_release_by_function.get(fn_name, '')
                if release_reason:
                    meta['planner_constraint_mechanism_release'] = True
                    meta['planner_constraint_override_reason'] = release_reason
                if belief_anchor_enabled:
                    meta['planner_belief_branch_id'] = str(constraints.get('belief_branch_id', '') or '')
                    meta['planner_belief_target_phase'] = str(constraints.get('belief_target_phase', '') or '')
                    meta['planner_belief_branch_confidence'] = belief_branch_confidence
                    meta['planner_belief_anchor_match'] = fn_name in set(belief_anchor_functions)
                    meta['planner_belief_anchor_functions'] = list(belief_anchor_functions)
            filtered.append(action)
            if _is_high_confidence_active_procedure_step(action, visible_functions):
                allowed_active_procedure_steps += 1

    procedure_anchor_override = (
        bool(preserved_active_procedure_candidates)
        and allowed_active_procedure_steps == 0
        and not bool(mechanism_release_by_function)
    )
    if procedure_anchor_override:
        ordered_final_actions: List[Dict[str, Any]] = []
        seen_ids = set()
        for action in [*preserved_active_procedure_candidates, *wait_candidates]:
            action_id = id(action)
            if action_id in seen_ids:
                continue
            seen_ids.add(action_id)
            ordered_final_actions.append(action)
        final_actions = ordered_final_actions or preserved_active_procedure_candidates
    else:
        final_actions = filtered or candidate_actions
    n_non_wait_after = 0
    for action in final_actions:
        if not isinstance(action, dict) or action.get('kind') == 'wait':
            continue
        fn_name = _extract_action_function_name(action)
        if fn_name and fn_name != 'wait':
            n_non_wait_after += 1

    fail_event = n_non_wait_before > 0 and n_non_wait_after == 0
    viability_entry.update({
        'n_total_after': len(final_actions),
        'n_non_wait_after': n_non_wait_after,
        'fail_event': fail_event,
        'candidate_zeroed_at': 'plan_constraints' if fail_event else None,
        'preserved_active_procedure_steps': preserved_active_procedure_steps,
        'preserved_high_confidence_commit_candidates': preserved_high_confidence_commit_candidates,
        'procedure_anchor_override': procedure_anchor_override,
        'mechanism_release_preserved': bool(mechanism_release_by_function),
    })

    return ConstraintResult(filtered_candidates=final_actions, viability_entry=viability_entry, pending_replan_patch=None)

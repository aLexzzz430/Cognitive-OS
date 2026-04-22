from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.orchestration.action_utils import action_matches_blocked_name, extract_action_identity


@dataclass
class SuppressionInput:
    candidate_actions: List[Dict[str, Any]]
    recent_failure_summary: Dict[str, Any]
    resource_state: Dict[str, Any]
    continuity_snapshot: Optional[Dict[str, Any]]
    tick: int
    episode: int


@dataclass
class SuppressionResult:
    filtered_candidates: List[Dict[str, Any]]
    audit_records: List[Dict[str, Any]]


def _strategy_payload(strategy: Any) -> Dict[str, Any]:
    if isinstance(strategy, dict):
        return dict(strategy)
    if hasattr(strategy, "to_dict"):
        try:
            payload = strategy.to_dict()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _strong_active_procedure_function_names(
    candidate_actions: List[Dict[str, Any]],
    extract_action_function_name,
) -> List[str]:
    guided: List[str] = []
    for action in candidate_actions:
        if not isinstance(action, dict):
            continue
        meta = action.get('_candidate_meta', {})
        if not isinstance(meta, dict):
            continue
        procedure = meta.get('procedure', {})
        if not isinstance(procedure, dict):
            continue
        if not bool(procedure.get('is_next_step', False)):
            continue
        source = str(action.get('_source', '') or '').strip().lower()
        hit_source = str(procedure.get('hit_source', '') or '').strip().lower()
        support_domains = procedure.get('support_domains', [])
        if isinstance(support_domains, list):
            support_count = len([item for item in support_domains if str(item).strip()])
        else:
            support_count = 0
        support_count = max(support_count, int(procedure.get('support_count', 0) or 0))
        mapping_confidence = float(procedure.get('mapping_confidence', 0.0) or 0.0)
        active_guidance = (
            bool(meta.get('procedure_guidance', {}).get('active_next_step', False))
            if isinstance(meta.get('procedure_guidance', {}), dict)
            else False
        )
        if source != 'procedure_reuse' and not active_guidance:
            continue
        if hit_source == 'latent_mechanism_abstraction' and support_count < 2:
            continue
        if hit_source == 'latent_mechanism_abstraction' and mapping_confidence < 0.5:
            continue
        fn_name = extract_action_function_name(action, default='')
        if fn_name and fn_name != 'wait' and fn_name not in guided:
            guided.append(fn_name)
    return guided


def _clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return max(minimum, min(maximum, float(default)))


def _world_model_hidden_state(context: Dict[str, Any]) -> Dict[str, Any]:
    hidden = context.get('world_model_hidden_state', {})
    if isinstance(hidden, dict) and hidden:
        return dict(hidden)
    summary = context.get('world_model_summary', {})
    if not isinstance(summary, dict):
        return {}
    nested = summary.get('hidden_state', {})
    return dict(nested) if isinstance(nested, dict) else {}


def _world_model_control(context: Dict[str, Any]) -> Dict[str, Any]:
    raw = context.get('world_model_control', {})
    return dict(raw) if isinstance(raw, dict) else {}


def _dominant_branch_profile(context: Dict[str, Any]) -> Dict[str, Any]:
    control = _world_model_control(context)
    hidden = _world_model_hidden_state(context)
    transition_memory = hidden.get('transition_memory', {}) if isinstance(hidden.get('transition_memory', {}), dict) else {}
    dominant_branch_id = str(
        control.get(
            'dominant_branch_id',
            hidden.get('dominant_branch_id', transition_memory.get('dominant_branch_id', '')),
        ) or ''
    ).strip()
    latent_branches = control.get('latent_branches')
    if not isinstance(latent_branches, list) or not latent_branches:
        latent_branches = hidden.get('latent_branches', transition_memory.get('latent_branches', []))
    if not isinstance(latent_branches, list):
        return {}

    selected = {}
    for branch in latent_branches:
        if not isinstance(branch, dict):
            continue
        if dominant_branch_id and str(branch.get('branch_id', '') or '').strip() == dominant_branch_id:
            selected = branch
            break
    if not selected:
        for branch in latent_branches:
            if isinstance(branch, dict):
                selected = branch
                break
    if not selected:
        return {}

    return {
        'branch_id': str(selected.get('branch_id', dominant_branch_id) or dominant_branch_id or ''),
        'confidence': _clamp(
            selected.get('confidence', selected.get('support', selected.get('transition_score', 0.0))),
            0.0,
            1.0,
            0.0,
        ),
        'anchor_functions': [
            str(value or '').strip()
            for value in (
                selected.get('anchor_functions', selected.get('anchored_functions', []))
                if isinstance(selected.get('anchor_functions', selected.get('anchored_functions', [])), list)
                else []
            )
            if str(value or '').strip()
        ],
        'risky_functions': [
            str(value or '').strip()
            for value in (selected.get('risky_functions', []) if isinstance(selected.get('risky_functions', []), list) else [])
            if str(value or '').strip()
        ],
    }


def _world_model_competition_profile(
    context: Dict[str, Any],
    *,
    candidate_function_universe: List[str],
) -> Dict[str, Any]:
    control = _world_model_control(context)
    hidden = _world_model_hidden_state(context)
    summary = context.get('world_model_summary', {})
    if not isinstance(summary, dict):
        summary = {}
    dominant_branch = _dominant_branch_profile(context)
    required_probes = [
        str(value or '').strip()
        for value in (
            control.get('required_probes', summary.get('required_probes', []))
            if isinstance(control.get('required_probes', summary.get('required_probes', [])), list)
            else []
        )
        if str(value or '').strip()
    ]
    anchor_functions = [
        fn_name
        for fn_name in list(dominant_branch.get('anchor_functions', []) or [])
        if fn_name in candidate_function_universe
    ]
    risky_functions = [
        fn_name
        for fn_name in list(dominant_branch.get('risky_functions', []) or [])
        if fn_name in candidate_function_universe
    ]
    control_trust = _clamp(control.get('control_trust', summary.get('control_trust', 0.5)), 0.0, 1.0, 0.5)
    transition_confidence = _clamp(
        control.get('transition_confidence', summary.get('transition_confidence', 0.5)),
        0.0,
        1.0,
        0.5,
    )
    state_shift_risk = _clamp(
        control.get('state_shift_risk', summary.get('shift_risk', 0.0)),
        0.0,
        1.0,
        0.0,
    )
    hidden_drift = _clamp(
        control.get('hidden_drift_score', context.get('world_model_hidden_drift_score', hidden.get('drift_score', 0.0))),
        0.0,
        1.0,
        0.0,
    )
    hidden_uncertainty = _clamp(
        control.get('hidden_uncertainty_score', context.get('world_model_hidden_uncertainty_score', hidden.get('uncertainty_score', 0.0))),
        0.0,
        1.0,
        0.0,
    )
    dominant_branch_confidence = _clamp(dominant_branch.get('confidence', 0.0), 0.0, 1.0, 0.0)
    probe_pressure = min(1.0, len(required_probes) / 3.0)
    latent_instability = _clamp(
        (1.0 - dominant_branch_confidence) * 0.38
        + hidden_drift * 0.28
        + hidden_uncertainty * 0.22
        + state_shift_risk * 0.12,
        0.0,
        1.0,
        0.0,
    )
    probe_pressure_active = (
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
    return {
        'required_probes': required_probes,
        'probe_pressure': float(probe_pressure),
        'probe_pressure_active': bool(probe_pressure_active),
        'latent_instability': float(latent_instability),
        'dominant_branch_id': str(dominant_branch.get('branch_id', '') or ''),
        'dominant_anchor_functions': anchor_functions,
        'dominant_risky_functions': risky_functions,
    }


def _goal_text(context: Dict[str, Any]) -> str:
    for source in (
        context.get('perception', {}),
        context.get('world_model_summary', {}),
        context.get('world_model', {}),
    ):
        if not isinstance(source, dict):
            continue
        perception = source.get('perception', {})
        if isinstance(perception, dict):
            text = str(perception.get('goal', '') or '').strip()
            if text:
                return text.lower()
        text = str(source.get('goal', '') or '').strip()
        if text:
            return text.lower()
    return ''


def _force_variant(function_name: str) -> str:
    fn_name = str(function_name or '').strip().lower()
    if not fn_name or fn_name.startswith('force_') or '_' not in fn_name:
        return ''
    return f"force_{fn_name.split('_', 1)[1]}"


def _safe_goal_prefers_non_force_execution(context: Dict[str, Any]) -> bool:
    goal_text = _goal_text(context)
    if not goal_text:
        return False
    return (
        'safe' in goal_text
        or 'avoid force' in goal_text
        or 'unsafe force' in goal_text
        or 'safe unlock' in goal_text
    )


def _merge_world_model_competition_into_strategy(
    strategy_payload: Dict[str, Any],
    competition: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(strategy_payload)
    probe_pressure_active = bool(competition.get('probe_pressure_active', False))
    latent_instability = float(competition.get('latent_instability', 0.0) or 0.0)
    preferred_verification = list(merged.get('preferred_verification_functions', [])) if isinstance(merged.get('preferred_verification_functions', []), list) else []
    preferred_fallback = list(merged.get('preferred_fallback_functions', [])) if isinstance(merged.get('preferred_fallback_functions', []), list) else []

    if probe_pressure_active:
        if str(merged.get('strategy_mode_hint', 'balanced') or 'balanced') == 'balanced':
            merged['strategy_mode_hint'] = 'verify'
        merged['verification_budget_hint'] = max(1, int(merged.get('verification_budget_hint', 0) or 0))
        if latent_instability >= 0.58:
            merged['branch_budget_hint'] = max(2, int(merged.get('branch_budget_hint', 0) or 0))

    ordered_verification: List[str] = []
    for fn_name in [
        *list(competition.get('required_probes', []) or []),
        *list(competition.get('dominant_anchor_functions', []) or []),
        *preferred_verification,
    ]:
        text = str(fn_name or '').strip()
        if not text or text in ordered_verification:
            continue
        if any(token in text.lower() for token in ('probe', 'inspect', 'verify', 'check', 'test')):
            ordered_verification.append(text)
    ordered_fallback: List[str] = []
    for fn_name in [*list(competition.get('dominant_anchor_functions', []) or []), *preferred_fallback]:
        text = str(fn_name or '').strip()
        if not text or text in ordered_fallback or text in set(competition.get('dominant_risky_functions', []) or []):
            continue
        ordered_fallback.append(text)

    if ordered_verification:
        merged['preferred_verification_functions'] = ordered_verification
    if ordered_fallback:
        merged['preferred_fallback_functions'] = ordered_fallback
    return merged


def apply_self_model_suppression(
    input_obj: SuppressionInput,
    extract_action_function_name,
    infer_task_family,
    extract_phase_hint,
    reliability_tracker,
) -> SuppressionResult:
    candidate_actions = list(input_obj.candidate_actions or [])
    if not candidate_actions:
        return SuppressionResult(filtered_candidates=candidate_actions, audit_records=[])

    resource_low = bool(input_obj.resource_state.get('is_tight_budget', False))
    recent_failures = int(input_obj.recent_failure_summary.get('recent_failures', 0) or 0)
    suppress_threshold = 0.45 if (resource_low or recent_failures >= 2) else 0.7
    reliability_sample_threshold = 3

    legal_non_wait_fn_names: List[str] = []
    for candidate in candidate_actions:
        fn = extract_action_function_name(candidate, default='')
        if fn and fn != 'wait':
            legal_non_wait_fn_names.append(fn)

    legal_non_wait_count = len(legal_non_wait_fn_names)
    lone_viable_fn_name = legal_non_wait_fn_names[0] if legal_non_wait_count == 1 else ''
    lone_viable_reliability_samples = 0
    if lone_viable_fn_name and reliability_tracker is not None:
        rel = reliability_tracker.get_action_type_reliability(lone_viable_fn_name)
        lone_viable_reliability_samples = int(getattr(rel, 'total_attempts', 0) or 0)

    failure_history_insufficient = (recent_failures < 2) or (lone_viable_reliability_samples < reliability_sample_threshold)
    early_phase = int(input_obj.tick or 0) <= 2 and failure_history_insufficient

    snapshot_context = dict(input_obj.continuity_snapshot) if isinstance(input_obj.continuity_snapshot, dict) else {}
    suppression_context = {
        **snapshot_context,
        'task_family': infer_task_family(snapshot_context),
        'phase': extract_phase_hint(snapshot_context),
        'observation_mode': str(input_obj.resource_state.get('observation_mode', 'unknown')),
        'resource_band': str(input_obj.resource_state.get('budget_band', 'normal')),
    }
    candidate_function_universe = list(dict.fromkeys(legal_non_wait_fn_names))
    for key in ('available_functions', 'discovered_functions', 'visible_functions'):
        merged_functions: List[str] = []
        for fn_name in [*(snapshot_context.get(key, []) if isinstance(snapshot_context.get(key, []), list) else []), *candidate_function_universe]:
            text = str(fn_name or '').strip()
            if text and text not in merged_functions:
                merged_functions.append(text)
        suppression_context[key] = merged_functions
    raw_visible_function_set = {
        str(fn_name or '').strip()
        for fn_name in (snapshot_context.get('visible_functions', []) if isinstance(snapshot_context.get('visible_functions', []), list) else [])
        if str(fn_name or '').strip()
    }
    visible_function_set = raw_visible_function_set or {
        str(fn_name or '').strip()
        for fn_name in suppression_context.get('visible_functions', [])
        if str(fn_name or '').strip()
    }
    safe_goal_active = _safe_goal_prefers_non_force_execution(suppression_context)
    candidate_function_set = {
        str(fn_name or '').strip()
        for fn_name in candidate_function_universe
        if str(fn_name or '').strip()
    }
    safe_visible_execution_functions = {
        fn_name
        for fn_name in candidate_function_set
        if _force_variant(fn_name) in {value.lower() for value in candidate_function_set}
    } or {
        fn_name
        for fn_name in visible_function_set
        if _force_variant(fn_name) in {value.lower() for value in visible_function_set}
    }
    competition = _world_model_competition_profile(
        suppression_context,
        candidate_function_universe=list(candidate_function_universe),
    )
    short_term_pressure = min(1.0, recent_failures / 5.0)
    global_strategy = reliability_tracker.build_global_failure_strategy(
        short_term_pressure=short_term_pressure,
        context=suppression_context,
    ) if reliability_tracker is not None and hasattr(reliability_tracker, 'build_global_failure_strategy') else None
    global_strategy_payload = _strategy_payload(global_strategy)
    global_strategy_payload = _merge_world_model_competition_into_strategy(global_strategy_payload, competition)
    shared_planner_hint = {
        'strategy_mode': str(global_strategy_payload.get('strategy_mode_hint', 'balanced') or 'balanced'),
        'branch_budget': int(global_strategy_payload.get('branch_budget_hint', 0) or 0),
        'verification_budget': int(global_strategy_payload.get('verification_budget_hint', 0) or 0),
    }
    global_blocked = set(global_strategy_payload.get('blocked_action_classes', [])) if isinstance(global_strategy_payload.get('blocked_action_classes', []), list) else set()
    global_preferred_verification = list(global_strategy_payload.get('preferred_verification_functions', [])) if isinstance(global_strategy_payload.get('preferred_verification_functions', []), list) else []
    global_preferred_fallback = list(global_strategy_payload.get('preferred_fallback_functions', [])) if isinstance(global_strategy_payload.get('preferred_fallback_functions', []), list) else []
    strong_guided_functions = _strong_active_procedure_function_names(candidate_actions, extract_action_function_name)

    kept: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for action in candidate_actions:
        fn_name = extract_action_function_name(action, default='wait')
        action_identity = extract_action_identity(action, include_function_fallback=False)
        failure_key = action_identity or ('' if str(fn_name or '').strip().upper() == 'ACTION6' else fn_name)
        source_name = str(action.get('_source', '') or '').strip().lower() if isinstance(action, dict) else ''
        meta = action.get('_candidate_meta', {}) if isinstance(action.get('_candidate_meta', {}), dict) else {}
        action_kind = str(action.get('kind', '') or '').strip().lower() if isinstance(action, dict) else ''
        fn_name_lower = str(fn_name or '').strip().lower()
        exact_required_probe = fn_name in set(competition.get('required_probes', []) or [])
        safe_execution_preferred = bool(
            safe_goal_active and fn_name in safe_visible_execution_functions
        )
        force_variant_suppressed = bool(
            safe_goal_active
            and fn_name_lower.startswith('force_')
            and fn_name_lower[6:]
            and fn_name_lower[6:] in {
                safe_fn.split('_', 1)[1].lower()
                for safe_fn in safe_visible_execution_functions
                if '_' in safe_fn
            }
        )
        synthetic_hidden_probe = bool(
            safe_goal_active
            and safe_visible_execution_functions
            and not exact_required_probe
            and (
                action_kind == 'probe'
                or source_name == 'deliberation_probe'
                or bool(meta.get('deliberation_injected', False))
            )
        )
        if isinstance(action, dict):
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                if safe_execution_preferred:
                    meta['safe_execution_preferred'] = True
                    meta['safe_execution_visible_alternatives'] = sorted(safe_visible_execution_functions)
                if force_variant_suppressed:
                    meta['safe_execution_force_variant'] = True
                if synthetic_hidden_probe:
                    meta['safe_execution_hidden_probe'] = True
        if force_variant_suppressed:
            audit.append({
                'episode': input_obj.episode,
                'tick': input_obj.tick,
                'event': 'safe_execution_force_variant_suppressed',
                'function_name': fn_name,
                'safe_visible_functions': sorted(safe_visible_execution_functions),
            })
            continue
        if synthetic_hidden_probe:
            audit.append({
                'episode': input_obj.episode,
                'tick': input_obj.tick,
                'event': 'safe_execution_hidden_probe_suppressed',
                'function_name': fn_name,
                'visible_functions': sorted(visible_function_set),
            })
            continue
        if (
            source_name == 'self_model'
            and strong_guided_functions
            and fn_name not in strong_guided_functions
        ):
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['self_model_suppressed'] = True
                meta['self_model_suppression_reason'] = 'active_procedure_guidance'
                meta['active_procedure_functions'] = list(strong_guided_functions)
            audit.append({
                'episode': input_obj.episode,
                'tick': input_obj.tick,
                'event': 'self_model_suppressed_for_active_procedure',
                'function_name': fn_name,
                'guided_functions': list(strong_guided_functions),
            })
            continue
        risk = reliability_tracker.get_action_failure_risk(failure_key, context=suppression_context) if (reliability_tracker is not None and failure_key) else 0.0
        strategy = reliability_tracker.build_failure_strategy(
            failure_key,
            short_term_pressure=short_term_pressure,
            context=suppression_context,
            planner_control_profile=shared_planner_hint,
        ) if reliability_tracker is not None else None
        strategy_payload = _strategy_payload(strategy)
        strategy_payload = _merge_world_model_competition_into_strategy(strategy_payload, competition)
        strategy_blocked = set(strategy_payload.get('blocked_action_classes', [])) if isinstance(strategy_payload.get('blocked_action_classes', []), list) else set()
        blocked = action_matches_blocked_name(action, global_blocked) or action_matches_blocked_name(action, strategy_blocked)
        preferred_verification = fn_name in set(global_preferred_verification[:2]) or fn_name in set(strategy_payload.get('preferred_verification_functions', [])[:2])
        preferred_fallback = fn_name in set(global_preferred_fallback[:2]) or fn_name in set(strategy_payload.get('preferred_fallback_functions', [])[:2])
        shared_mode = str(strategy_payload.get('strategy_mode_hint', global_strategy_payload.get('strategy_mode_hint', 'balanced')) or 'balanced')
        preferred_recovery_candidate = fn_name != 'wait' and (preferred_verification or preferred_fallback)
        world_model_required_probe = fn_name in set(competition.get('required_probes', []) or [])
        world_model_anchor_match = fn_name in set(competition.get('dominant_anchor_functions', []) or [])
        world_model_risky = fn_name in set(competition.get('dominant_risky_functions', []) or [])
        world_model_competition_preserve = (
            fn_name != 'wait'
            and not world_model_risky
            and (
                world_model_required_probe
                or (
                    bool(competition.get('probe_pressure_active', False))
                    and any(token in fn_name.lower() for token in ('probe', 'inspect', 'verify', 'check', 'test'))
                )
                or (
                    float(competition.get('latent_instability', 0.0) or 0.0) >= 0.58
                    and world_model_anchor_match
                )
            )
        )

        if (
            early_phase
            and legal_non_wait_count == 1
            and fn_name == lone_viable_fn_name
            and fn_name != 'wait'
            and (risk >= suppress_threshold or blocked)
        ):
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['early_viability_override'] = True
                meta['early_viability_override_reason'] = 'single_legal_non_wait_candidate_in_early_phase'
                meta['self_model_failure_risk'] = risk
                meta['self_model_failure_risk_context'] = suppression_context
                meta['failure_strategy_profile'] = dict(strategy_payload)
                meta['global_failure_strategy'] = dict(global_strategy_payload)
            audit.append({
                'episode': input_obj.episode,
                'tick': input_obj.tick,
                'event': 'early_viability_override',
                'function_name': fn_name,
                'risk': float(risk),
                'reason': 'single_legal_non_wait_candidate_in_early_phase',
            })
            kept.append(action)
            continue

        if fn_name != 'wait' and (risk >= suppress_threshold or blocked):
            if preferred_recovery_candidate and not blocked and shared_mode in {'recover', 'verify'}:
                meta = action.setdefault('_candidate_meta', {})
                if isinstance(meta, dict):
                    meta['failure_preference_preserved'] = True
                    meta['failure_preference_reason'] = 'shared_failure_preference_keep'
                    meta['self_model_failure_risk'] = risk
                    meta['self_model_failure_risk_context'] = suppression_context
                    meta['failure_strategy_profile'] = dict(strategy_payload)
                    meta['global_failure_strategy'] = dict(global_strategy_payload)
                audit.append({
                    'episode': input_obj.episode,
                    'tick': input_obj.tick,
                    'event': 'failure_preference_preserved',
                    'function_name': fn_name,
                    'risk': float(risk),
                    'strategy_mode_hint': shared_mode,
                })
                kept.append(action)
                continue
            if world_model_competition_preserve:
                meta = action.setdefault('_candidate_meta', {})
                if isinstance(meta, dict):
                    meta['world_model_competition_preserved'] = True
                    meta['world_model_competition_reason'] = (
                        'required_probe'
                        if world_model_required_probe
                        else ('dominant_anchor_stabilization' if world_model_anchor_match else 'probe_pressure_preserve')
                    )
                    meta['world_model_probe_pressure'] = float(competition.get('probe_pressure', 0.0) or 0.0)
                    meta['world_model_latent_instability'] = float(competition.get('latent_instability', 0.0) or 0.0)
                    meta['world_model_dominant_branch_id'] = str(competition.get('dominant_branch_id', '') or '')
                    meta['self_model_failure_risk'] = risk
                    meta['self_model_failure_risk_context'] = suppression_context
                    meta['failure_strategy_profile'] = dict(strategy_payload)
                    meta['global_failure_strategy'] = dict(global_strategy_payload)
                audit.append({
                    'episode': input_obj.episode,
                    'tick': input_obj.tick,
                    'event': 'world_model_competition_preserved',
                    'function_name': fn_name,
                    'risk': float(risk),
                    'required_probe': bool(world_model_required_probe),
                    'anchor_match': bool(world_model_anchor_match),
                    'latent_branch_id': str(competition.get('dominant_branch_id', '') or ''),
                })
                kept.append(action)
                continue
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['self_model_suppressed'] = True
                meta['self_model_failure_risk'] = risk
                meta['self_model_failure_risk_context'] = suppression_context
                meta['failure_strategy_profile'] = dict(strategy_payload)
                meta['global_failure_strategy'] = dict(global_strategy_payload)
            audit.append({
                'episode': input_obj.episode,
                'tick': input_obj.tick,
                'event': 'self_model_suppressed',
                'function_name': fn_name,
                'risk': float(risk),
            })
            continue

        if strategy and isinstance(action, dict):
            meta = action.setdefault('_candidate_meta', {})
            if isinstance(meta, dict):
                meta['failure_strategy_profile'] = dict(strategy_payload)
                meta['global_failure_strategy'] = dict(global_strategy_payload)
        kept.append(action)

    return SuppressionResult(filtered_candidates=kept or candidate_actions, audit_records=audit)

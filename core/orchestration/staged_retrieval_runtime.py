from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from core.main_loop_components import RetrieveResult
from core.orchestration.retrieval_gate import (
    RetrievalAuxInput,
    RetrievalGateInput,
    RetrievalSignalInput,
    decide_retrieval_aux,
    is_gray_zone,
    resolve_retrieval_policy,
)
from core.orchestration.retrieval_runtime_helpers import RetrievalRuntimeHelpers
from modules.world_model.events import EventType, WorldModelEvent
from modules.world_model.protocol import WorldModelControlProtocol


@dataclass(frozen=True)
class RetrievalWorldModelState:
    retrieval_ctx: Dict[str, Any]
    world_model_control: WorldModelControlProtocol
    world_model_required_probes: List[str]
    latent_branch_instability: float


@dataclass(frozen=True)
class RetrievalGateState:
    query: Any
    llm_retrieval_ctx: Any
    current_signature: str
    phase_hint: str
    controls: Dict[str, Any]
    cheap_should_retrieve: bool
    should_retrieve: bool
    used_llm_gate: bool
    retrieval_decision_record: Dict[str, Any]


def _emit_observation_received(loop: Any, obs_before: Dict[str, Any], continuity_snapshot: Dict[str, Any]) -> None:
    top_goal_id = None
    if continuity_snapshot:
        top_goal = continuity_snapshot.get('top_goal')
        if top_goal:
            top_goal_id = getattr(top_goal, 'goal_id', None)
    loop._event_bus.emit(WorldModelEvent(
        event_type=EventType.OBSERVATION_RECEIVED,
        episode=loop._episode,
        tick=loop._tick,
        data={
            'has_novel_api': 'novel_api' in obs_before,
            'known_functions': obs_before.get('novel_api', {}).get('discovered_functions', []) if isinstance(obs_before.get('novel_api'), dict) else [],
            'continuity_top_goal': top_goal_id,
            'perception': obs_before.get('perception', {}),
        },
        source_stage='retrieval',
    ))


def _compute_latent_branch_instability(world_model_control: WorldModelControlProtocol) -> float:
    latent_branch_instability = 0.0
    for branch in list(world_model_control.latent_branches or [])[:4]:
        if not isinstance(branch, dict):
            continue
        branch_instability = max(
            0.0,
            min(
                1.0,
                float(branch.get('uncertainty_pressure', 0.0) or 0.0) * 0.55
                + (1.0 - float(branch.get('confidence', 0.0) or 0.0)) * 0.30
                + (1.0 - float(branch.get('success_rate', 0.0) or 0.0)) * 0.15,
            ),
        )
        latent_branch_instability = max(latent_branch_instability, branch_instability)
    return latent_branch_instability


def _prepare_world_model_state(loop: Any, obs_before: Dict[str, Any], ctx: Dict[str, Any]) -> RetrievalWorldModelState:
    retrieval_ctx = dict(ctx)
    world_model_summary = (
        obs_before.get('world_model', {})
        if isinstance(obs_before.get('world_model', {}), dict)
        else ctx.get('world_model_summary', {})
    )
    world_model_transition_priors = (
        ctx.get('world_model_transition_priors', {})
        if isinstance(ctx.get('world_model_transition_priors', {}), dict)
        else {}
    )
    if not world_model_transition_priors:
        world_model_transition_priors = loop._build_world_model_transition_priors(getattr(loop, '_last_perception_summary', {}))
    world_model_control = WorldModelControlProtocol.from_context({
        'world_model_summary': world_model_summary if isinstance(world_model_summary, dict) else {},
        'world_model_transition_priors': world_model_transition_priors if isinstance(world_model_transition_priors, dict) else {},
        'prediction_error_tail': loop._prediction_trace_log[-5:] if isinstance(loop._prediction_trace_log, list) else [],
        'predictor_trust': loop._prediction_registry.get_predictor_trust() if loop._prediction_enabled else {},
    })
    world_model_required_probes = list(dict.fromkeys(
        str(item) for item in list(world_model_control.required_probes or []) if str(item or '')
    ))
    latent_branch_instability = _compute_latent_branch_instability(world_model_control)
    retrieval_ctx.update({
        'retrieval_decision_record': dict(loop._last_retrieval_decision) if isinstance(loop._last_retrieval_decision, dict) else {},
        'plan_target_function': loop._plan_state.get_target_function_for_step(),
        'plan_intent': loop._plan_state.get_intent_for_step(),
        'failure_modes': [
            str(entry.get('outcome', {}).get('error', {}) or '').split(':')[0] if isinstance(entry, dict) else ''
            for entry in loop._episode_trace[-4:]
        ],
        'focus_object_ids': [
            c.get('object_id')
            for c in (obs_before.get('retrieved_objects', []) if isinstance(obs_before.get('retrieved_objects', []), list) else [])
            if isinstance(c, dict) and c.get('object_id')
        ],
        'world_focus_variables': [belief.variable_name for belief in loop._belief_ledger.get_active_beliefs()[:4]],
        'world_model_required_probes': list(world_model_required_probes),
        'world_model_hidden_phase': str(world_model_control.hidden_state_phase or ''),
        'world_model_dominant_branch_id': str(world_model_control.dominant_branch_id or ''),
    })
    return RetrievalWorldModelState(
        retrieval_ctx=retrieval_ctx,
        world_model_control=world_model_control,
        world_model_required_probes=world_model_required_probes,
        latent_branch_instability=latent_branch_instability,
    )


def _build_retrieval_signal_input(
    loop: Any,
    *,
    phase_hint: str,
    current_signature: str,
    effective_retrieval_aggressiveness: float,
    controls: Dict[str, Any],
    world_state: RetrievalWorldModelState,
) -> RetrievalSignalInput:
    return RetrievalSignalInput(
        tick=loop._tick,
        phase_hint=phase_hint,
        last_phase_hint=loop._last_phase_hint,
        current_signature_changed=bool(loop._last_observation_signature and current_signature != loop._last_observation_signature),
        recent_reward_stagnation=loop._recent_reward_stagnation(),
        recent_action_repetition=loop._recent_action_repetition(),
        pending_recovery_or_replan=bool(loop._pending_recovery_probe or loop._pending_replan),
        last_recovery_is_recent=bool(loop._recovery_log and isinstance(loop._recovery_log[-1], dict) and loop._recovery_log[-1].get('tick') == loop._tick - 1),
        retrieval_aggressiveness=effective_retrieval_aggressiveness,
        verification_bias=float(controls.get('verification_bias', 0.5) or 0.5),
        risk_tolerance=float(controls.get('risk_tolerance', 0.5) or 0.5),
        recovery_bias=float(controls.get('recovery_bias', 0.5) or 0.5),
        strategy_mode=str(controls.get('strategy_mode', 'balanced') or 'balanced'),
        cooldown_ready=loop._cooldown_ready(loop._last_retrieval_tick, loop._runtime_budget.retrieval_cooldown_ticks),
        force_retrieve=bool(loop._tick == 0 and loop._runtime_budget.force_retrieval_on_tick0),
        world_model_required_probe_count=len(world_state.world_model_required_probes),
        world_model_control_trust=float(world_state.world_model_control.control_trust),
        world_model_transition_confidence=float(world_state.world_model_control.transition_confidence),
        world_model_state_shift_risk=float(world_state.world_model_control.state_shift_risk),
        hidden_state_drift_score=float(world_state.world_model_control.hidden_drift_score),
        hidden_state_uncertainty_score=float(world_state.world_model_control.hidden_uncertainty_score),
        latent_branch_instability=float(world_state.latent_branch_instability),
    )


def _build_gate_state(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    continuity_snapshot: Dict[str, Any],
    world_state: RetrievalWorldModelState,
) -> RetrievalGateState:
    query = loop._retriever.build_query(obs_before, world_state.retrieval_ctx)
    llm_retrieval_ctx = RetrievalRuntimeHelpers.build_llm_retrieval_context(
        episode=loop._episode,
        tick=loop._tick,
        obs=obs_before,
        continuity_snapshot=continuity_snapshot,
        active_hypotheses=len(loop._hypotheses.get_active()),
        confirmed_hypotheses=len(loop._hypotheses.get_confirmed()),
        entropy=loop._hypotheses.entropy(),
        margin=loop._hypotheses.margin(),
    )
    current_signature = loop._compute_observation_signature(obs_before)
    phase_hint = loop._extract_phase_hint(continuity_snapshot)
    controls = loop._meta_control.for_retrieval_gate(
        episode=loop._episode,
        tick=loop._tick,
        context={'gate': 'retrieval', 'phase_hint': phase_hint},
    )
    effective_retrieval_aggressiveness = max(
        0.0,
        min(
            1.0,
            0.7 * float(controls['retrieval_aggressiveness']) + 0.3 * float(controls['retrieval_pressure']),
        ),
    )
    signal_input = _build_retrieval_signal_input(
        loop,
        phase_hint=phase_hint,
        current_signature=current_signature,
        effective_retrieval_aggressiveness=effective_retrieval_aggressiveness,
        controls=controls,
        world_state=world_state,
    )
    llm_gray_zone = is_gray_zone(
        llm_gate_enabled=loop._runtime_budget.enable_llm_retrieval_gate,
        llm_available=loop._resolve_llm_client("retrieval") is not None,
        active_hypotheses=llm_retrieval_ctx.active_hypotheses,
        entropy=llm_retrieval_ctx.entropy,
        margin=llm_retrieval_ctx.margin,
        cooldown_ready=loop._cooldown_ready(loop._last_retrieval_tick, loop._runtime_budget.retrieval_cooldown_ticks),
    )
    cheap_policy = resolve_retrieval_policy(
        signal_input=signal_input,
        gate_input=RetrievalGateInput(
            llm_gate_enabled=False,
            llm_available=loop._resolve_llm_client("retrieval") is not None,
            llm_gray_zone=False,
            llm_vote=None,
        ),
    )
    cheap_should_retrieve = cheap_policy.cheap_decision.should_retrieve
    loop._last_retrieval_decision_record = cheap_policy.cheap_decision
    loop._last_retrieval_decision = {
        **cheap_policy.cheap_decision.to_dict(),
        'meta_control_snapshot_id': controls['meta_control_snapshot_id'],
        'meta_control_inputs_hash': controls['meta_control_inputs_hash'],
        'signals': {
            'epistemic_uncertainty': cheap_policy.signals.epistemic_uncertainty,
            'plan_blockage_risk': cheap_policy.signals.plan_blockage_risk,
            'repeated_failure_pressure': cheap_policy.signals.repeated_failure_pressure,
            'novelty_deficit': cheap_policy.signals.novelty_deficit,
            'memory_usefulness_prediction': cheap_policy.signals.memory_usefulness_prediction,
            'cooldown_ready': cheap_policy.signals.cooldown_ready,
            'force_retrieve': cheap_policy.signals.force_retrieve,
        },
        'world_model_gate': {
            'required_probes': list(world_state.world_model_required_probes),
            'control_trust': float(world_state.world_model_control.control_trust),
            'transition_confidence': float(world_state.world_model_control.transition_confidence),
            'state_shift_risk': float(world_state.world_model_control.state_shift_risk),
            'hidden_drift_score': float(world_state.world_model_control.hidden_drift_score),
            'hidden_uncertainty_score': float(world_state.world_model_control.hidden_uncertainty_score),
            'latent_branch_instability': float(world_state.latent_branch_instability),
        },
        'strategy_mode': str(controls.get('strategy_mode', 'balanced') or 'balanced'),
        'verification_bias': float(controls.get('verification_bias', 0.5) or 0.5),
        'risk_tolerance': float(controls.get('risk_tolerance', 0.5) or 0.5),
        'recovery_bias': float(controls.get('recovery_bias', 0.5) or 0.5),
    }
    llm_vote = None
    if not cheap_should_retrieve and llm_gray_zone and loop._retrieval_llm.can_advise_retrieval_gate():
        loop._record_budgeted_llm_call('retrieval_gate', route_name='retrieval')
        llm_vote = loop._retrieval_llm.should_use_retrieval(llm_retrieval_ctx)
    gate_policy = resolve_retrieval_policy(
        signal_input=signal_input,
        gate_input=RetrievalGateInput(
            llm_gate_enabled=loop._runtime_budget.enable_llm_retrieval_gate,
            llm_available=loop._resolve_llm_client("retrieval") is not None,
            llm_gray_zone=llm_gray_zone,
            llm_vote=llm_vote,
        ),
    )
    retrieval_decision_record = {
        **(dict(loop._last_retrieval_decision) if isinstance(loop._last_retrieval_decision, dict) else {}),
        'gate_source': gate_policy.gate_source,
        'resolved': gate_policy.resolved_decision.to_dict(),
    }
    loop._llm_advice_log.append({
        'tick': loop._tick,
        'episode': loop._episode,
        'kind': 'retrieval_gate',
        'cheap_should_retrieve': cheap_should_retrieve,
        'used_llm_gate': gate_policy.used_llm_gate,
        'should_retrieve': gate_policy.should_retrieve,
        'retrieval_decision_record': retrieval_decision_record,
        'retrieval_aux_decisions': dict(loop._last_retrieval_aux_decisions),
        'meta_control_snapshot_id': retrieval_decision_record.get('meta_control_snapshot_id'),
        'meta_control_inputs_hash': retrieval_decision_record.get('meta_control_inputs_hash'),
    })
    return RetrievalGateState(
        query=query,
        llm_retrieval_ctx=llm_retrieval_ctx,
        current_signature=current_signature,
        phase_hint=phase_hint,
        controls=controls,
        cheap_should_retrieve=cheap_should_retrieve,
        should_retrieve=gate_policy.should_retrieve,
        used_llm_gate=gate_policy.used_llm_gate,
        retrieval_decision_record=retrieval_decision_record,
    )


def _build_aux_input(
    loop: Any,
    *,
    surfaced_count: int,
    candidate_margin: float,
    phase_hint: str,
    current_signature: str,
    aux_controls: Dict[str, Any],
    effective_retrieval_aggressiveness: float,
    world_state: RetrievalWorldModelState,
) -> RetrievalAuxInput:
    return RetrievalAuxInput(
        surfaced_count=surfaced_count,
        candidate_margin=candidate_margin,
        reward_stagnation=loop._recent_reward_stagnation(),
        pending_recovery_or_replan=bool(loop._pending_recovery_probe or loop._pending_replan),
        phase_changed=bool(phase_hint and phase_hint != loop._last_phase_hint),
        cooldown_ready=loop._cooldown_ready(loop._last_rerank_tick, loop._runtime_budget.llm_rerank_cooldown_ticks),
        retrieval_aggressiveness=effective_retrieval_aggressiveness,
        probe_bias=float(aux_controls['probe_bias']),
        verification_bias=float(aux_controls.get('verification_bias', 0.5) or 0.5),
        recovery_bias=float(aux_controls.get('recovery_bias', 0.5) or 0.5),
        strategy_mode=str(aux_controls.get('strategy_mode', 'balanced') or 'balanced'),
        signature_changed=bool(loop._last_observation_signature and current_signature != loop._last_observation_signature),
        world_model_required_probe_count=len(world_state.world_model_required_probes),
        world_model_control_trust=float(world_state.world_model_control.control_trust),
        world_model_state_shift_risk=float(world_state.world_model_control.state_shift_risk),
        hidden_state_drift_score=float(world_state.world_model_control.hidden_drift_score),
        hidden_state_uncertainty_score=float(world_state.world_model_control.hidden_uncertainty_score),
        latent_branch_instability=float(world_state.latent_branch_instability),
    )


def _run_retrieval_path(
    loop: Any,
    *,
    gate_state: RetrievalGateState,
    world_state: RetrievalWorldModelState,
) -> tuple[RetrieveResult, List[Any], bool, bool]:
    retrieve_result = RetrieveResult(
        candidates=[],
        selected_ids=[],
        action_influence='none',
        contract={'selected_ids': [], 'action_influence': 'none', 'candidate_count': 0},
    )
    surfaced: List[Any] = []
    query_rewritten = False
    reranked = False
    if not gate_state.should_retrieve:
        return retrieve_result, surfaced, query_rewritten, reranked

    aux_controls = loop._meta_control.for_rerank_query_gate(
        episode=loop._episode,
        tick=loop._tick,
        context={'gate': 'query_rewrite'},
    )
    effective_retrieval_aggressiveness = max(
        0.0,
        min(
            1.0,
            0.7 * float(aux_controls['retrieval_aggressiveness']) + 0.3 * float(aux_controls['retrieval_pressure']),
        ),
    )
    aux_input = _build_aux_input(
        loop,
        surfaced_count=len(retrieve_result.candidates),
        candidate_margin=loop._retrieval_candidate_margin(retrieve_result.candidates),
        phase_hint=gate_state.phase_hint,
        current_signature=gate_state.current_signature,
        aux_controls=aux_controls,
        effective_retrieval_aggressiveness=effective_retrieval_aggressiveness,
        world_state=world_state,
    )
    aux_decisions = decide_retrieval_aux(aux_input)
    loop._last_retrieval_aux_decisions['query_rewrite'] = {**aux_decisions['query_rewrite'], 'tick': loop._tick}
    if aux_decisions['query_rewrite']['enabled'] and loop._retrieval_llm.can_advise_retrieval_gate():
        loop._record_budgeted_llm_call('query_rewrite', route_name='retrieval')
        gate_state.query.query_text = loop._retrieval_llm.query_rewrite(
            gate_state.query.query_text,
            gate_state.llm_retrieval_ctx,
        )
        loop._last_query_rewrite_tick = loop._tick
        query_rewritten = True

    retrieve_result = loop._retriever.retrieve(gate_state.query)
    loop._last_retrieval_tick = loop._tick

    rerank_input = _build_aux_input(
        loop,
        surfaced_count=len(retrieve_result.candidates),
        candidate_margin=loop._retrieval_candidate_margin(retrieve_result.candidates),
        phase_hint=gate_state.phase_hint,
        current_signature=gate_state.current_signature,
        aux_controls=aux_controls,
        effective_retrieval_aggressiveness=effective_retrieval_aggressiveness,
        world_state=world_state,
    )
    rerank_decisions = decide_retrieval_aux(rerank_input)
    loop._last_retrieval_aux_decisions['rerank'] = {**rerank_decisions['rerank'], 'tick': loop._tick}
    if rerank_decisions['rerank']['enabled'] and loop._retrieval_llm.can_use_llm():
        loop._record_budgeted_llm_call('rerank_candidates', route_name='retrieval')
        retrieve_result.candidates = loop._retrieval_llm.rerank_candidates(
            retrieve_result.candidates,
            gate_state.query.query_text,
            gate_state.llm_retrieval_ctx,
        )
        loop._last_rerank_tick = loop._tick
        reranked = True

    surfaced = loop._retriever.surface(retrieve_result, top_k=5, consumed_fns=loop._consumed_fns)
    return retrieve_result, surfaced, query_rewritten, reranked


def _maybe_augment_hypotheses(
    loop: Any,
    *,
    obs_before: Dict[str, Any],
    continuity_snapshot: Dict[str, Any],
    gate_state: RetrievalGateState,
    world_state: RetrievalWorldModelState,
) -> bool:
    augment_decision = RetrievalRuntimeHelpers.decide_hypothesis_augment(
        entropy=loop._hypotheses.entropy(),
        reward_stagnation=loop._recent_reward_stagnation(),
        signature_changed=bool(
            loop._last_observation_signature and gate_state.current_signature != loop._last_observation_signature
        ),
        pending_recovery_probe=bool(loop._pending_recovery_probe),
        pending_replan=bool(loop._pending_replan),
        cooldown_ready=loop._cooldown_ready(
            loop._last_hypothesis_augment_tick,
            loop._runtime_budget.hypothesis_augment_cooldown_ticks,
        ),
        tick=loop._tick,
        world_model_required_probe_count=len(world_state.world_model_required_probes),
        world_model_control_trust=float(world_state.world_model_control.control_trust),
        world_model_transition_confidence=float(world_state.world_model_control.transition_confidence),
        world_model_state_shift_risk=float(world_state.world_model_control.state_shift_risk),
        hidden_state_drift_score=float(world_state.world_model_control.hidden_drift_score),
        hidden_state_uncertainty_score=float(world_state.world_model_control.hidden_uncertainty_score),
        latent_branch_instability=float(world_state.latent_branch_instability),
    )
    if loop._resolve_llm_client("hypothesis") is None or not augment_decision.should_augment:
        return False
    loop._record_budgeted_llm_call('augment_hypotheses', route_name='hypothesis')
    loop._augment_hypotheses_with_llm(obs_before, continuity_snapshot)
    loop._last_hypothesis_augment_tick = loop._tick
    return True


def run_stage1_retrieval(loop: Any, stage_input: Any) -> Dict[str, Any]:
    obs_before = stage_input.obs_before
    ctx = stage_input.context
    continuity_snapshot = stage_input.continuity_snapshot
    _emit_observation_received(loop, obs_before, continuity_snapshot)
    world_state = _prepare_world_model_state(loop, obs_before, ctx)
    gate_state = _build_gate_state(
        loop,
        obs_before=obs_before,
        continuity_snapshot=continuity_snapshot,
        world_state=world_state,
    )
    retrieve_result, surfaced, query_rewritten, reranked = _run_retrieval_path(
        loop,
        gate_state=gate_state,
        world_state=world_state,
    )
    augmented = _maybe_augment_hypotheses(
        loop,
        obs_before=obs_before,
        continuity_snapshot=continuity_snapshot,
        gate_state=gate_state,
        world_state=world_state,
    )

    loop._last_observation_signature = gate_state.current_signature
    loop._last_phase_hint = gate_state.phase_hint
    surfacing_protocol = RetrievalRuntimeHelpers.build_surfacing_protocol_payload(
        query=gate_state.query,
        surfaced=surfaced,
        retrieve_result=retrieve_result,
    )
    return {
        'query': gate_state.query,
        'retrieve_result': retrieve_result,
        'surfaced': surfaced,
        'surfacing_protocol': surfacing_protocol,
        'llm_retrieval_ctx': gate_state.llm_retrieval_ctx,
        'budget': {
            'cheap_should_retrieve': gate_state.cheap_should_retrieve,
            'used_llm_gate': gate_state.used_llm_gate,
            'should_retrieve': gate_state.should_retrieve,
            'query_rewritten': query_rewritten,
            'reranked': reranked,
            'augmented_hypotheses': augmented,
        },
    }

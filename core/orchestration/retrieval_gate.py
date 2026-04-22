from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.retrieval_control import RetrievalAuxPolicy, RetrievalControlState, RetrievalDecisionRecord, RetrievalSignals


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


@dataclass
class RetrievalSignalInput:
    tick: int
    phase_hint: str
    last_phase_hint: str
    current_signature_changed: bool
    recent_reward_stagnation: bool
    recent_action_repetition: bool
    pending_recovery_or_replan: bool
    last_recovery_is_recent: bool
    retrieval_aggressiveness: float
    verification_bias: float
    risk_tolerance: float
    recovery_bias: float
    strategy_mode: str
    cooldown_ready: bool
    force_retrieve: bool
    world_model_required_probe_count: int = 0
    world_model_control_trust: float = 0.5
    world_model_transition_confidence: float = 0.5
    world_model_state_shift_risk: float = 0.0
    hidden_state_drift_score: float = 0.0
    hidden_state_uncertainty_score: float = 0.0
    latent_branch_instability: float = 0.0


@dataclass
class RetrievalGateInput:
    llm_gate_enabled: bool
    llm_available: bool
    llm_gray_zone: bool
    llm_vote: Optional[bool]


@dataclass
class RetrievalAuxInput:
    surfaced_count: int
    candidate_margin: float
    reward_stagnation: bool
    pending_recovery_or_replan: bool
    phase_changed: bool
    cooldown_ready: bool
    retrieval_aggressiveness: float
    probe_bias: float
    verification_bias: float
    recovery_bias: float
    strategy_mode: str
    signature_changed: bool
    world_model_required_probe_count: int = 0
    world_model_control_trust: float = 0.5
    world_model_state_shift_risk: float = 0.0
    hidden_state_drift_score: float = 0.0
    hidden_state_uncertainty_score: float = 0.0
    latent_branch_instability: float = 0.0


@dataclass(frozen=True)
class HypothesisAugmentSignals:
    entropy: float
    reward_stagnation: bool
    signature_changed: bool
    pending_recovery_probe: bool
    pending_replan: bool
    world_model_required_probe_count: int = 0
    world_model_control_trust: float = 0.5
    world_model_transition_confidence: float = 0.5
    world_model_state_shift_risk: float = 0.0
    hidden_state_drift_score: float = 0.0
    hidden_state_uncertainty_score: float = 0.0
    latent_branch_instability: float = 0.0


@dataclass(frozen=True)
class HypothesisAugmentCooldownState:
    cooldown_ready: bool


@dataclass(frozen=True)
class HypothesisAugmentContext:
    tick: int


@dataclass(frozen=True)
class HypothesisAugmentDecision:
    should_augment: bool
    reason: str


@dataclass
class RetrievalPolicyResult:
    signals: RetrievalSignals
    cheap_decision: RetrievalDecisionRecord
    should_retrieve: bool
    used_llm_gate: bool
    gate_source: str
    resolved_decision: RetrievalDecisionRecord


def build_retrieval_signals(input_obj: RetrievalSignalInput) -> RetrievalSignals:
    epistemic_uncertainty = 0.2
    if input_obj.current_signature_changed:
        epistemic_uncertainty += 0.45
    if input_obj.phase_hint and input_obj.phase_hint != input_obj.last_phase_hint:
        epistemic_uncertainty += 0.25

    repeated_failure_pressure = 0.65 if input_obj.recent_reward_stagnation else 0.1
    if input_obj.recent_action_repetition:
        repeated_failure_pressure = max(repeated_failure_pressure, 0.7)

    plan_blockage_risk = 0.6 if input_obj.pending_recovery_or_replan else 0.2
    if input_obj.last_recovery_is_recent:
        plan_blockage_risk = max(plan_blockage_risk, 0.7)

    probe_pressure = min(1.0, max(0, int(input_obj.world_model_required_probe_count or 0)) / 3.0)
    control_trust = _clamp01(input_obj.world_model_control_trust, 0.5)
    transition_confidence = _clamp01(input_obj.world_model_transition_confidence, 0.5)
    state_shift_risk = _clamp01(input_obj.world_model_state_shift_risk, 0.0)
    hidden_drift = _clamp01(input_obj.hidden_state_drift_score, 0.0)
    hidden_uncertainty = _clamp01(input_obj.hidden_state_uncertainty_score, 0.0)
    latent_instability = _clamp01(input_obj.latent_branch_instability, 0.0)
    trust_gap = max(0.0, 0.58 - control_trust)
    transition_gap = max(0.0, 0.55 - transition_confidence)

    epistemic_uncertainty += (
        hidden_uncertainty * 0.26
        + probe_pressure * 0.18
        + trust_gap * 0.40
        + transition_gap * 0.22
        + latent_instability * 0.16
    )
    plan_blockage_risk += (
        state_shift_risk * 0.24
        + hidden_drift * 0.20
        + probe_pressure * 0.16
        + latent_instability * 0.12
    )
    repeated_failure_pressure = max(
        repeated_failure_pressure,
        0.10
        + state_shift_risk * 0.10
        + hidden_drift * 0.14
        + latent_instability * 0.12,
    )

    retrieval_bias = max(0.0, min(1.0, float(input_obj.retrieval_aggressiveness)))
    verification_bias = max(0.0, min(1.0, float(input_obj.verification_bias)))
    risk_tolerance = max(0.0, min(1.0, float(input_obj.risk_tolerance)))
    recovery_bias = max(0.0, min(1.0, float(input_obj.recovery_bias)))
    strategy_mode = str(input_obj.strategy_mode or 'balanced')
    novelty_deficit = 0.3 + max(0.0, 0.75 - retrieval_bias) * 0.4
    memory_usefulness_prediction = min(1.0, 0.35 + retrieval_bias * 0.7)
    novelty_deficit += probe_pressure * 0.10 + trust_gap * 0.12
    memory_usefulness_prediction += probe_pressure * 0.16 + hidden_uncertainty * 0.06 + state_shift_risk * 0.04

    if strategy_mode == 'recover':
        plan_blockage_risk += 0.12 + recovery_bias * 0.12
        repeated_failure_pressure += 0.08 + recovery_bias * 0.08
        epistemic_uncertainty += verification_bias * 0.10
    elif strategy_mode == 'verify':
        epistemic_uncertainty += 0.12 + verification_bias * 0.12
        plan_blockage_risk += recovery_bias * 0.05
    elif strategy_mode == 'explore':
        novelty_deficit += 0.14 + (1.0 - risk_tolerance) * 0.04
        memory_usefulness_prediction += 0.06
    elif strategy_mode == 'exploit':
        epistemic_uncertainty -= 0.05 * max(0.0, risk_tolerance - 0.4)
        novelty_deficit -= 0.06
        memory_usefulness_prediction += 0.08 + risk_tolerance * 0.06

    return RetrievalSignals(
        epistemic_uncertainty=min(1.0, epistemic_uncertainty),
        plan_blockage_risk=min(1.0, plan_blockage_risk),
        repeated_failure_pressure=min(1.0, repeated_failure_pressure),
        novelty_deficit=min(1.0, novelty_deficit),
        memory_usefulness_prediction=min(1.0, memory_usefulness_prediction),
        cooldown_ready=bool(input_obj.cooldown_ready),
        force_retrieve=bool(input_obj.force_retrieve),
    )


def is_gray_zone(*, llm_gate_enabled: bool, llm_available: bool, active_hypotheses: int, entropy: float, margin: float, cooldown_ready: bool) -> bool:
    return (
        llm_gate_enabled
        and llm_available
        and active_hypotheses > 0
        and entropy >= 0.25
        and margin <= 0.35
        and cooldown_ready
    )


def resolve_retrieval_policy(signal_input: RetrievalSignalInput, gate_input: RetrievalGateInput) -> RetrievalPolicyResult:
    signals = build_retrieval_signals(signal_input)
    cheap_decision = RetrievalControlState.decide(signals)
    gate_outcome = RetrievalControlState.resolve_gate(
        base_decision=cheap_decision,
        llm_gate_enabled=gate_input.llm_gate_enabled,
        llm_available=gate_input.llm_available,
        llm_gray_zone=gate_input.llm_gray_zone,
        llm_vote=gate_input.llm_vote,
    )
    return RetrievalPolicyResult(
        signals=signals,
        cheap_decision=cheap_decision,
        should_retrieve=bool(gate_outcome.should_retrieve),
        used_llm_gate=bool(gate_outcome.used_llm_gate),
        gate_source=str(gate_outcome.gate_source),
        resolved_decision=gate_outcome.decision_record,
    )


def decide_retrieval_aux(input_obj: RetrievalAuxInput) -> Dict[str, Dict[str, Any]]:
    rerank = RetrievalAuxPolicy.should_rerank(
        surfaced_count=input_obj.surfaced_count,
        candidate_margin=input_obj.candidate_margin,
        reward_stagnation=input_obj.reward_stagnation,
        pending_recovery_or_replan=input_obj.pending_recovery_or_replan,
        phase_changed=input_obj.phase_changed,
        cooldown_ready=input_obj.cooldown_ready,
        retrieval_aggressiveness=input_obj.retrieval_aggressiveness,
        recovery_bias=input_obj.recovery_bias,
        strategy_mode=input_obj.strategy_mode,
    )
    query_rewrite = RetrievalAuxPolicy.should_query_rewrite(
        reward_stagnation=input_obj.reward_stagnation,
        pending_recovery_or_replan=input_obj.pending_recovery_or_replan,
        signature_changed=input_obj.signature_changed,
        cooldown_ready=input_obj.cooldown_ready,
        probe_bias=input_obj.probe_bias,
        verification_bias=input_obj.verification_bias,
        recovery_bias=input_obj.recovery_bias,
        strategy_mode=input_obj.strategy_mode,
    )
    probe_pressure = min(1.0, max(0, int(input_obj.world_model_required_probe_count or 0)) / 3.0)
    control_trust = _clamp01(input_obj.world_model_control_trust, 0.5)
    hidden_drift = _clamp01(input_obj.hidden_state_drift_score, 0.0)
    hidden_uncertainty = _clamp01(input_obj.hidden_state_uncertainty_score, 0.0)
    latent_instability = _clamp01(input_obj.latent_branch_instability, 0.0)
    state_shift_risk = _clamp01(input_obj.world_model_state_shift_risk, 0.0)
    world_model_probe_pressure = (
        probe_pressure >= 0.34
        and (
            control_trust <= 0.52
            or hidden_drift >= 0.55
            or hidden_uncertainty >= 0.62
            or latent_instability >= 0.58
            or state_shift_risk >= 0.58
        )
    )
    if world_model_probe_pressure:
        if input_obj.surfaced_count >= 2 and input_obj.cooldown_ready:
            rerank = type(rerank)(enabled=True, reason='world_model_probe_pressure')
        if input_obj.cooldown_ready:
            query_rewrite = type(query_rewrite)(enabled=True, reason='world_model_probe_pressure')
    return {
        'rerank': {'enabled': bool(rerank.enabled), 'reason': str(rerank.reason)},
        'query_rewrite': {'enabled': bool(query_rewrite.enabled), 'reason': str(query_rewrite.reason)},
    }


def should_augment_hypotheses(
    signals: HypothesisAugmentSignals,
    cooldown_state: HypothesisAugmentCooldownState,
    context: HypothesisAugmentContext,
) -> HypothesisAugmentDecision:
    if not cooldown_state.cooldown_ready:
        return HypothesisAugmentDecision(should_augment=False, reason='cooldown_not_ready')
    if context.tick == 0:
        return HypothesisAugmentDecision(should_augment=True, reason='tick0_bootstrap')
    if signals.entropy > 0.45 and signals.reward_stagnation:
        return HypothesisAugmentDecision(should_augment=True, reason='high_entropy_stagnation')
    if signals.signature_changed:
        return HypothesisAugmentDecision(should_augment=True, reason='signature_changed')
    if signals.pending_recovery_probe or signals.pending_replan:
        return HypothesisAugmentDecision(should_augment=True, reason='pending_recovery_or_replan')
    probe_pressure = min(1.0, max(0, int(signals.world_model_required_probe_count or 0)) / 3.0)
    control_trust = _clamp01(signals.world_model_control_trust, 0.5)
    transition_confidence = _clamp01(signals.world_model_transition_confidence, 0.5)
    state_shift_risk = _clamp01(signals.world_model_state_shift_risk, 0.0)
    hidden_drift = _clamp01(signals.hidden_state_drift_score, 0.0)
    hidden_uncertainty = _clamp01(signals.hidden_state_uncertainty_score, 0.0)
    latent_instability = _clamp01(signals.latent_branch_instability, 0.0)
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
    if world_model_probe_pressure:
        return HypothesisAugmentDecision(should_augment=True, reason='world_model_probe_pressure')
    return HypothesisAugmentDecision(should_augment=False, reason='insufficient_pressure')

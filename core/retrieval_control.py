"""Structured retrieval control objects.

Reduces scattered heuristic gates by converting signals to a single auditable
decision record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RetrievalSignals:
    epistemic_uncertainty: float = 0.0
    plan_blockage_risk: float = 0.0
    repeated_failure_pressure: float = 0.0
    novelty_deficit: float = 0.0
    memory_usefulness_prediction: float = 0.5
    cooldown_ready: bool = True
    force_retrieve: bool = False


@dataclass
class RetrievalDecisionRecord:
    should_retrieve: bool
    dominant_pressure: str
    why_retrieve: List[str] = field(default_factory=list)
    why_skip: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            'should_retrieve': self.should_retrieve,
            'dominant_pressure': self.dominant_pressure,
            'why_retrieve': list(self.why_retrieve),
            'why_skip': list(self.why_skip),
        }


@dataclass
class RetrievalGateOutcome:
    should_retrieve: bool
    used_llm_gate: bool
    gate_source: str
    decision_record: RetrievalDecisionRecord

    def to_dict(self) -> Dict[str, object]:
        return {
            'should_retrieve': self.should_retrieve,
            'used_llm_gate': self.used_llm_gate,
            'gate_source': self.gate_source,
            'decision_record': self.decision_record.to_dict(),
        }


class RetrievalControlState:
    """Deterministic retrieval decision object."""

    @staticmethod
    def decide(signals: RetrievalSignals) -> RetrievalDecisionRecord:
        if signals.force_retrieve:
            return RetrievalDecisionRecord(
                should_retrieve=True,
                dominant_pressure='force_retrieve',
                why_retrieve=['force_retrieve'],
            )

        if not signals.cooldown_ready:
            return RetrievalDecisionRecord(
                should_retrieve=False,
                dominant_pressure='cooldown',
                why_skip=['cooldown_not_ready'],
            )

        pressure_map = {
            'epistemic_uncertainty': signals.epistemic_uncertainty,
            'plan_blockage_risk': signals.plan_blockage_risk,
            'repeated_failure_pressure': signals.repeated_failure_pressure,
            'novelty_deficit': signals.novelty_deficit,
            'memory_usefulness_prediction': signals.memory_usefulness_prediction,
        }

        dominant = max(pressure_map.items(), key=lambda kv: kv[1])
        should = (
            signals.epistemic_uncertainty >= 0.55
            or signals.plan_blockage_risk >= 0.55
            or signals.repeated_failure_pressure >= 0.60
            or signals.novelty_deficit >= 0.60
            or (signals.memory_usefulness_prediction >= 0.65 and dominant[1] >= 0.5)
        )

        if should:
            reasons = [k for k, v in pressure_map.items() if v >= 0.55]
            if not reasons:
                reasons = [dominant[0]]
            return RetrievalDecisionRecord(
                should_retrieve=True,
                dominant_pressure=dominant[0],
                why_retrieve=reasons,
            )

        return RetrievalDecisionRecord(
            should_retrieve=False,
            dominant_pressure=dominant[0],
            why_skip=['insufficient_pressure'],
        )

    @staticmethod
    def resolve_gate(
        base_decision: RetrievalDecisionRecord,
        llm_gate_enabled: bool,
        llm_available: bool,
        llm_gray_zone: bool,
        llm_vote: bool | None,
    ) -> RetrievalGateOutcome:
        """Unify cheap gate and gray-zone llm gate into one auditable outcome."""
        if base_decision.should_retrieve:
            return RetrievalGateOutcome(
                should_retrieve=True,
                used_llm_gate=False,
                gate_source='structured_signals',
                decision_record=base_decision,
            )

        if llm_gate_enabled and llm_available and llm_gray_zone and llm_vote is not None:
            merged = RetrievalDecisionRecord(
                should_retrieve=bool(llm_vote),
                dominant_pressure=base_decision.dominant_pressure,
                why_retrieve=list(base_decision.why_retrieve) + (['llm_gray_zone_vote'] if llm_vote else []),
                why_skip=list(base_decision.why_skip) + ([] if llm_vote else ['llm_gray_zone_reject']),
            )
            return RetrievalGateOutcome(
                should_retrieve=bool(llm_vote),
                used_llm_gate=True,
                gate_source='llm_gray_zone',
                decision_record=merged,
            )

        return RetrievalGateOutcome(
            should_retrieve=False,
            used_llm_gate=False,
            gate_source='structured_signals',
            decision_record=base_decision,
        )


@dataclass
class RetrievalAuxDecision:
    enabled: bool
    reason: str


class RetrievalAuxPolicy:
    """Policy-profile-aware auxiliary retrieval controls (rerank/query-rewrite)."""

    @staticmethod
    def should_rerank(
        surfaced_count: int,
        candidate_margin: float,
        reward_stagnation: bool,
        pending_recovery_or_replan: bool,
        phase_changed: bool,
        cooldown_ready: bool,
        retrieval_aggressiveness: float,
        recovery_bias: float,
        strategy_mode: str,
    ) -> RetrievalAuxDecision:
        if surfaced_count < 2 or not cooldown_ready:
            return RetrievalAuxDecision(False, 'insufficient_candidates_or_cooldown')
        uncertainty_like = reward_stagnation or pending_recovery_or_replan or phase_changed
        margin_pressure = candidate_margin <= (0.35 + (0.6 - retrieval_aggressiveness) * 0.1)
        recovery_pressure = str(strategy_mode or '') == 'recover' and float(recovery_bias or 0.0) >= 0.58
        if recovery_pressure:
            return RetrievalAuxDecision(True, 'meta_recovery_bias')
        if uncertainty_like or margin_pressure:
            return RetrievalAuxDecision(True, 'policy_rerank_pressure')
        return RetrievalAuxDecision(False, 'policy_rerank_skip')

    @staticmethod
    def should_query_rewrite(
        reward_stagnation: bool,
        pending_recovery_or_replan: bool,
        signature_changed: bool,
        cooldown_ready: bool,
        probe_bias: float,
        verification_bias: float,
        recovery_bias: float,
        strategy_mode: str,
    ) -> RetrievalAuxDecision:
        if not cooldown_ready:
            return RetrievalAuxDecision(False, 'cooldown')
        if pending_recovery_or_replan:
            return RetrievalAuxDecision(True, 'recovery_or_replan')
        if str(strategy_mode or '') in {'recover', 'verify'} and max(float(verification_bias or 0.0), float(recovery_bias or 0.0)) >= 0.62:
            return RetrievalAuxDecision(True, 'meta_strategy_bias')
        if signature_changed:
            return RetrievalAuxDecision(True, 'signature_changed')
        if reward_stagnation and probe_bias >= 0.45:
            return RetrievalAuxDecision(True, 'stagnation_with_probe_bias')
        return RetrievalAuxDecision(False, 'policy_rewrite_skip')

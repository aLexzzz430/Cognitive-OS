from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RetrievalGateResult:
    should_retrieve: bool
    decision_payload: Dict[str, Any]
    decision_record: Any


class RetrievalGateStrategy:
    """策略模块：检索门控决策。"""

    @staticmethod
    def decide(*, controls: Dict[str, Any], signals: Any, decide_fn) -> RetrievalGateResult:
        decision = decide_fn(signals)
        payload = {
            **decision.to_dict(),
            'meta_control_snapshot_id': controls['meta_control_snapshot_id'],
            'meta_control_inputs_hash': controls['meta_control_inputs_hash'],
            'signals': {
                'epistemic_uncertainty': signals.epistemic_uncertainty,
                'plan_blockage_risk': signals.plan_blockage_risk,
                'repeated_failure_pressure': signals.repeated_failure_pressure,
                'novelty_deficit': signals.novelty_deficit,
                'memory_usefulness_prediction': signals.memory_usefulness_prediction,
                'cooldown_ready': signals.cooldown_ready,
                'force_retrieve': signals.force_retrieve,
            },
        }
        return RetrievalGateResult(
            should_retrieve=decision.should_retrieve,
            decision_payload=payload,
            decision_record=decision,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class RerankStrategyResult:
    enabled: bool
    reason: str
    audit_payload: Dict[str, Any]


class RerankStrategy:
    """策略模块：是否触发 LLM rerank。"""

    @staticmethod
    def decide(
        *,
        surfaced_count: int,
        candidate_margin: float,
        reward_stagnation: bool,
        pending_recovery_or_replan: bool,
        phase_changed: bool,
        cooldown_ready: bool,
        retrieval_aggressiveness: float,
        controls: Dict[str, Any],
        tick: int,
        should_rerank_fn,
    ) -> RerankStrategyResult:
        decision = should_rerank_fn(
            surfaced_count=surfaced_count,
            candidate_margin=candidate_margin,
            reward_stagnation=reward_stagnation,
            pending_recovery_or_replan=pending_recovery_or_replan,
            phase_changed=phase_changed,
            cooldown_ready=cooldown_ready,
            retrieval_aggressiveness=retrieval_aggressiveness,
        )
        return RerankStrategyResult(
            enabled=decision.enabled,
            reason=decision.reason,
            audit_payload={
                'enabled': decision.enabled,
                'reason': decision.reason,
                'tick': tick,
                'meta_control_snapshot_id': controls['meta_control_snapshot_id'],
                'meta_control_inputs_hash': controls['meta_control_inputs_hash'],
            },
        )

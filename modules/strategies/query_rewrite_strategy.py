from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class QueryRewriteStrategyResult:
    enabled: bool
    reason: str
    audit_payload: Dict[str, Any]


class QueryRewriteStrategy:
    """策略模块：是否触发 query rewrite。"""

    @staticmethod
    def decide(
        *,
        reward_stagnation: bool,
        pending_recovery_or_replan: bool,
        signature_changed: bool,
        cooldown_ready: bool,
        probe_bias: float,
        controls: Dict[str, Any],
        tick: int,
        should_query_rewrite_fn,
    ) -> QueryRewriteStrategyResult:
        decision = should_query_rewrite_fn(
            reward_stagnation=reward_stagnation,
            pending_recovery_or_replan=pending_recovery_or_replan,
            signature_changed=signature_changed,
            cooldown_ready=cooldown_ready,
            probe_bias=probe_bias,
        )
        return QueryRewriteStrategyResult(
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

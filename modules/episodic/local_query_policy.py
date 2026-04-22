from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalQueryPolicyDecision:
    use_retrieval: bool
    reason: str


class DistilledLocalQueryPolicy:
    """Distilled local policy for retrieval gate + query rewrite.

    This policy is deterministic and intentionally lightweight so it can run
    without network/model dependencies.
    """

    @staticmethod
    def should_use_retrieval(*, active_hypotheses: int, entropy: float, margin: float, is_saturated: bool) -> LocalQueryPolicyDecision:
        if active_hypotheses <= 0:
            return LocalQueryPolicyDecision(use_retrieval=False, reason='no_active_hypotheses')
        if is_saturated and margin > 0.35:
            return LocalQueryPolicyDecision(use_retrieval=False, reason='saturated_and_margin_safe')
        if entropy >= 0.3 and margin <= 0.4:
            return LocalQueryPolicyDecision(use_retrieval=True, reason='entropy_margin_trigger')
        return LocalQueryPolicyDecision(use_retrieval=active_hypotheses >= 2, reason='hypothesis_count_trigger')

    @staticmethod
    def rewrite_query(base_query: str, *, phase: str, discovered_functions: list[str], available_functions: list[str], active_hypotheses: int) -> str:
        base = (base_query or '').strip()
        if not base:
            base = 'episodic retrieval'

        # Deterministic keyword enrichment from runtime state
        anchors: list[str] = []
        if phase:
            anchors.append(f'phase:{phase}')
        if active_hypotheses > 0:
            anchors.append(f'hyp:{active_hypotheses}')

        missing = [fn for fn in available_functions if fn not in set(discovered_functions)]
        if missing:
            anchors.append('target:' + ','.join(sorted(missing)[:2]))

        rewrite_suffix = ' | '.join(anchors)
        return base if not rewrite_suffix else f'{base} | {rewrite_suffix}'

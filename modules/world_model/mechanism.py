"""
modules/world_model/mechanism.py

Mechanism hypothesis schema + extraction helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MechanismHypothesis:
    mechanism_id: str
    variables: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    expected_transition: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)
    status: str = "candidate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "variables": list(self.variables),
            "trigger_conditions": list(self.trigger_conditions),
            "expected_transition": self.expected_transition,
            "confidence": float(self.confidence),
            "evidence_ids": list(self.evidence_ids),
            "invalidation_conditions": list(self.invalidation_conditions),
            "status": self.status,
        }


class MechanismExtractor:
    """Extract mechanism candidates from confirmed hypotheses/tests/repeated patterns."""

    def extract(
        self,
        confirmed_hypotheses: List[Any],
        latest_test: Optional[Dict[str, Any]],
        episode_trace: List[Dict[str, Any]],
        episode: int,
        tick: int,
        action_name_fn,
    ) -> List[MechanismHypothesis]:
        mechanisms: List[MechanismHypothesis] = []

        for hyp in confirmed_hypotheses[:10]:
            expected = getattr(hyp, "expected_transition", None)
            if not expected:
                continue
            mechanisms.append(MechanismHypothesis(
                mechanism_id=f"mech_hyp_{getattr(hyp, 'id', 'unknown')}",
                variables=[getattr(hyp, "type", "unknown"), getattr(hyp, "id", "unknown")],
                trigger_conditions=[getattr(hyp, "trigger_condition", "") or f"hypothesis:{getattr(hyp, 'id', 'unknown')}"],
                expected_transition=str(expected),
                confidence=float(getattr(hyp, "confidence", 0.0) or 0.0),
                evidence_ids=list(getattr(hyp, "evidence_ids", [])[-5:]),
                invalidation_conditions=list(getattr(hyp, "falsifiers", []) or ['test_refuted', 'conflicting_transition']),
                status='confirmed',
            ))

        test_result = latest_test if isinstance(latest_test, dict) else {}
        if test_result:
            test_fn = str(test_result.get('test_function', '') or '')
            passed = bool(test_result.get('test_result'))
            if test_fn:
                mechanisms.append(MechanismHypothesis(
                    mechanism_id=f"mech_test_{episode}_{tick}_{test_fn}",
                    variables=['test_function', test_fn],
                    trigger_conditions=[f"test:{test_fn}", "discriminating_test_result"],
                    expected_transition=f"{test_fn} => {'confirm_hypothesis_a' if passed else 'confirm_hypothesis_b'}",
                    confidence=0.7 if passed else 0.55,
                    evidence_ids=[str(test_result.get('evidence_id', f'test_{episode}_{tick}'))],
                    invalidation_conditions=['subsequent_test_opposite_outcome'],
                    status='candidate',
                ))

        pattern_counts: Dict[str, Dict[str, Any]] = {}
        for row in episode_trace[-12:]:
            if not isinstance(row, dict):
                continue
            fn = action_name_fn(row.get('action', {}), default='')
            if not fn:
                continue
            entry = pattern_counts.setdefault(fn, {'count': 0, 'reward_sum': 0.0, 'evidence_ids': []})
            entry['count'] += 1
            entry['reward_sum'] += float(row.get('reward', 0.0) or 0.0)
            row_tick = row.get('tick')
            if row_tick is not None:
                entry['evidence_ids'].append(f"trace_{episode}_{row_tick}")

        for fn, stat in pattern_counts.items():
            count = int(stat.get('count', 0) or 0)
            if count < 3:
                continue
            avg_reward = float(stat.get('reward_sum', 0.0) or 0.0) / max(1, count)
            mechanisms.append(MechanismHypothesis(
                mechanism_id=f"mech_pattern_{episode}_{fn}",
                variables=['action_function', fn],
                trigger_conditions=[f"action:{fn}", "repeated_transition_pattern"],
                expected_transition=f"{fn} => {'positive_reward' if avg_reward >= 0 else 'negative_reward'}",
                confidence=min(0.85, 0.45 + count * 0.08),
                evidence_ids=list(stat.get('evidence_ids', [])[-5:]),
                invalidation_conditions=['pattern_break', 'reward_sign_flip'],
                status='candidate',
            ))

        dedup: Dict[str, MechanismHypothesis] = {}
        for mech in mechanisms:
            dedup[mech.mechanism_id] = mech
        return list(dedup.values())


class MechanismFormalWriter:
    """Mechanism objects must go through validator -> object store -> commit."""

    def commit(self, mechanisms: List[MechanismHypothesis], validator, committer) -> List[str]:
        committed: List[str] = []
        for mechanism in mechanisms:
            proposal = {
                'type': 'memory_proposal',
                'memory_type': 'mechanism_hypothesis',
                'memory_layer': 'semantic',
                'content': mechanism.to_dict(),
                'confidence': float(mechanism.confidence),
                'retrieval_tags': ['mechanism', 'counterfactual', mechanism.status],
                'source_stage': 'hypothesis_test_aggregation',
            }
            decision = validator.validate(proposal)
            if getattr(decision, 'decision', '') != 'accept_new':
                continue
            committed.extend(committer.commit([(proposal, decision)]))
        return committed

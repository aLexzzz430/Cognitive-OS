"""
modules/hypothesis/hypothesis_tracker.py

P0-2: Hypothesis Tracking and Discriminating Test Engine

Separated from main_loop.py for clarity. CoreMainLoop imports this module.
"""

import random
from enum import Enum
from typing import Optional, Tuple, List, Dict, Iterator
import copy
from dataclasses import dataclass


class HypothesisStatus(Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    WEAKENED = "weakened"
    SUPERSEDED = "superseded"


class Hypothesis:
    """
    P0-2: Hypothesis object with id, claim, type, status, competing/conflict tracking.
    
    P3-C: Extended with mechanism-bearing fields:
    - trigger_condition: when does this mechanism activate
    - expected_transition: what state change does this mechanism predict
    - falsifiers: what evidence would contradict this hypothesis
    - competing_alternatives: what other mechanisms could explain the same evidence
    - supporting_beliefs: belief_ids that support this mechanism
    - confidence_interval: (lower, upper) bounds on confidence
    """
    def __init__(
        self,
        hyp_id: str,
        claim: str,
        hyp_type: str,
        created_from: str,
        confidence: float = 0.5,
        # P3-C: Mechanism fields
        trigger_condition: Optional[str] = None,
        expected_transition: Optional[str] = None,
        falsifiers: Optional[List[str]] = None,
        competing_alternatives: Optional[List[str]] = None,
        supporting_beliefs: Optional[List[str]] = None,
        confidence_interval: Optional[Tuple[float, float]] = None,
    ):
        self.id = hyp_id
        self.claim = claim
        self.type = hyp_type  # e.g., 'function_existence', 'parameter_constraint'
        self.status = HypothesisStatus.ACTIVE
        self.confidence = confidence
        self.created_from = created_from
        self.evidence_ids: List[str] = []
        self.competing_with: List[str] = []  # IDs of competing hypotheses
        self.conflict_with: List[str] = []
        self.skill_ids: List[str] = []  # C4: skills derived from this hypothesis
        self.created_at_tick = 0
        self.created_at_episode = 0
        self.tests_run = 0
        self.last_test_tick = -1
        # P3-C: Mechanism fields
        self.trigger_condition = trigger_condition  # e.g., "action=join_tables AND state=empty"
        self.expected_transition = expected_transition  # e.g., "state.prev=empty -> state.next=non_empty"
        self.falsifiers = falsifiers or []  # evidence that would contradict
        self.competing_alternatives = competing_alternatives or []  # other possible mechanisms
        self.supporting_beliefs = supporting_beliefs or []  # belief_ids
        self.confidence_interval = confidence_interval or (0.3, 0.7)  # (lower, upper)

    def mark_competing(self, other_id: str):
        """Record that this hypothesis competes with another."""
        if other_id not in self.competing_with:
            self.competing_with.append(other_id)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'claim': self.claim,
            'type': self.type,
            'status': self.status.value,
            'confidence': self.confidence,
            'created_from': self.created_from,
            'evidence_ids': list(self.evidence_ids),
            'competing_with': list(self.competing_with),
            'conflict_with': list(self.conflict_with),
            'skill_ids': list(self.skill_ids),
            'tests_run': self.tests_run,
            # P3-C: Mechanism fields
            'trigger_condition': self.trigger_condition,
            'expected_transition': self.expected_transition,
            'falsifiers': list(self.falsifiers),
            'competing_alternatives': list(self.competing_alternatives),
            'supporting_beliefs': list(self.supporting_beliefs),
            'confidence_interval': self.confidence_interval,
        }


class HypothesisTracker:
    """P0-2: Full hypothesis lifecycle with C4 skill invalidation."""
    def __init__(self, seed: int = 0, skill_rewriter=None):
        # Private internals: _hyps/_by_obj are module-owned state.
        # External callers should use iter_hypotheses/get_hypothesis/has_hypothesis/snapshot_for_audit.
        self._hyps: Dict[str, Hypothesis] = {}
        self._by_obj: Dict[str, str] = {}  # object_id → hypothesis_id
        self._rng = random.Random(seed)
        self._skill_rewriter = skill_rewriter  # C4: for skill invalidation on refutation
        # Hard metrics: entropy and margin tracking
        self._entropy_log: List[dict] = []
        self._test_log: List[dict] = []

    def entropy(self) -> float:
        """
        Shannon entropy of ACTIVE hypothesis confidence distribution (in bits).
        """
        from math import log2
        active = [h for h in self._hyps.values() if h.status == HypothesisStatus.ACTIVE]
        if not active:
            return 0.0
        total_conf = sum(h.confidence for h in active)
        if total_conf == 0:
            return 0.0
        ent = 0.0
        for h in active:
            p = h.confidence / total_conf
            if p > 0:
                ent -= p * log2(p)
        return ent

    def margin(self) -> float:
        """Confidence gap between top and second hypothesis."""
        active = self.get_active()
        if len(active) < 2:
            return 1.0
        confidences = sorted([h.confidence for h in active], reverse=True)
        return confidences[0] - confidences[1] if len(confidences) >= 2 else 1.0

    def posterior_gap(self) -> float:
        """Difference between confirmed and active hypothesis average confidence."""
        active = self.get_active()
        confirmed = self.get_confirmed()
        avg_active = sum(h.confidence for h in active) / max(len(active), 1)
        avg_confirmed = sum(h.confidence for h in confirmed) / max(len(confirmed), 1)
        return avg_confirmed - avg_active

    def log_state(self, tick: int = 0, episode: int = 0):
        self._entropy_log.append({
            'tick': tick, 'episode': episode,
            'entropy': self.entropy(),
            'margin': self.margin(),
            'posterior_gap': self.posterior_gap(),
            'confirmed_count': len(self.get_confirmed()),
            'active_count': len(self.get_active()),
            'refuted_count': len([h for h in self._hyps.values() if h.status == HypothesisStatus.REFUTED]),
        })

    def log_test(self, hyp_id: str, conf_before: float, conf_after: float,
                 eliminated: bool, tick: int = 0, episode: int = 0):
        entropy_after = self.entropy()
        entropy_before = self._entropy_log[-1]['entropy'] if self._entropy_log else entropy_after
        self._test_log.append({
            'tick': tick, 'episode': episode,
            'hyp_id': hyp_id,
            'confidence_before': conf_before,
            'confidence_after': conf_after,
            'confidence_delta': conf_after - conf_before,
            'eliminated': eliminated,
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'entropy_delta': entropy_after - entropy_before,
            'margin_before': self._entropy_log[-1]['margin'] if self._entropy_log else 1.0,
            'margin_after': self.margin(),
        })

    def get_entropy_log(self) -> List[dict]:
        return list(self._entropy_log)

    def get_test_log(self) -> List[dict]:
        return list(self._test_log)

    def iter_hypotheses(self) -> Iterator[Hypothesis]:
        """Read-only iteration over hypotheses."""
        return iter(self._hyps.values())

    def get_hypothesis(self, hyp_id: str) -> Optional[Hypothesis]:
        return self._hyps.get(hyp_id)

    def has_hypothesis(self, hyp_id: str) -> bool:
        return hyp_id in self._hyps


    def has_object_binding(self, object_id: str) -> bool:
        """Return True if an object_id is already bound to a hypothesis."""
        return object_id in self._by_obj

    def get_object_bound_hypothesis_ids(self) -> Dict[str, str]:
        """Read-only copy of object_id -> hypothesis_id bindings."""
        return dict(self._by_obj)

    def get_object_id_for_hypothesis(self, hyp_id: str) -> Optional[str]:
        """Return the object_id bound to a hypothesis_id, if present."""
        for object_id, bound_hyp_id in self._by_obj.items():
            if bound_hyp_id == hyp_id:
                return object_id
        return None

    def snapshot_for_audit(self) -> Tuple[dict, ...]:
        """Return immutable-by-convention snapshot for auditing."""
        return tuple(copy.deepcopy(h.to_dict()) for h in self._hyps.values())

    def create_from_object(self, obj: dict, obj_id: str, tick=0, episode=0) -> List[Hypothesis]:
        content = obj.get('content', {}) if isinstance(obj, dict) else {}
        tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
        fn = tool_args.get('function_name') or content.get('function_name', '')
        created = []
        if fn:
            h = Hypothesis(
                f"hyp_{fn}_{obj_id[:8]}",
                f"Function '{fn}' exists",
                'function_existence', obj_id,
                obj.get('confidence', 0.5)
            )
            h.created_at_tick = tick
            h.created_at_episode = episode
            h.skill_ids.append(f"s_{obj_id[:8]}")
            self._hyps[h.id] = h
            self._by_obj[obj_id] = h.id
            created.append(h)
        for k, v in tool_args.items():
            if k == 'function_name':
                continue
            if isinstance(v, (list, tuple, str, int, float)):
                h = Hypothesis(
                    f"hyp_{k}={v}_{obj_id[:8]}",
                    f"When calling {fn or 'unknown'}, '{k}={v}' is valid",
                    'parameter_constraint', obj_id,
                    obj.get('confidence', 0.4)
                )
                h.created_at_tick = tick
                h.created_at_episode = episode
                h.skill_ids.append(f"s_{obj_id[:8]}_{k}")
                self._hyps[h.id] = h
                created.append(h)
        return created

    def get_active(self) -> List[Hypothesis]:
        return [h for h in self._hyps.values() if h.status == HypothesisStatus.ACTIVE]

    def get_confirmed(self) -> List[Hypothesis]:
        return [h for h in self._hyps.values() if h.status == HypothesisStatus.CONFIRMED]

    def mark_competing(self, a: str, b: str):
        for hid, o in [(a, b), (b, a)]:
            if hid in self._hyps:
                self._hyps[hid].mark_competing(o)

    def update_test(self, hyp_id: str, passed: bool, tick: int = 0, episode: int = 0):
        h = self._hyps.get(hyp_id)
        if not h:
            return
        conf_before = h.confidence
        self.log_state(tick=tick, episode=episode)

        h.tests_run += 1
        h.last_test_tick = tick

        if passed:
            ns = HypothesisStatus.CONFIRMED if h.status == HypothesisStatus.ACTIVE else h.status
            h.status = ns
            h.confidence = min(1.0, h.confidence + 0.2)
        else:
            ns = HypothesisStatus.REFUTED if h.status == HypothesisStatus.ACTIVE else h.status
            h.status = ns
            h.confidence = max(0.0, h.confidence - 0.3)
            # C4: Invalidate skills derived from this refuted hypothesis
            if self._skill_rewriter and h.skill_ids:
                self._skill_rewriter.invalidate_skills_for_hyp(h.id, h.skill_ids)

        eliminated = (h.status == HypothesisStatus.REFUTED)
        self.log_test(hyp_id, conf_before, h.confidence, eliminated, tick=tick, episode=episode)
        self.log_state(tick=tick, episode=episode)

    def discriminating_pair(self) -> Optional[Tuple[Hypothesis, Hypothesis]]:
        """
        Return a pair of ACTIVE hypotheses that genuinely compete.

        No fallback to (active[0], confirmed[0]) — that was the root cause of
        "Full == Fresh" because testing CONFIRMED hypotheses produces no info gain.
        """
        active = self.get_active()
        for ha in active:
            for oid in ha.competing_with:
                hb = self._hyps.get(oid)
                if hb and hb.status == HypothesisStatus.ACTIVE:
                    return (ha, hb)
        return None

    def get_log(self) -> List[dict]:
        return [h.to_dict() for h in self._hyps.values()]


# =============================================================================
# P0-3: Discriminating Test Engine
# =============================================================================

@dataclass
class DiscriminatingTest:
    """
    P0-3: Discriminating test between competing hypotheses.
    
    P3-C: Extended with mechanism fields for targeted probing:
    - trigger_condition: condition under which hypothesis activates
    - expected_transition: predicted state change
    - falsifiers: what evidence would contradict
    """
    test_id: str
    hypothesis_a: str  # hypothesis ID
    hypothesis_b: str  # hypothesis ID
    test_function: str  # function to test
    test_params: dict   # parameters to test
    created_at_tick: int = 0
    created_at_episode: int = 0
    result: Optional[bool] = None  # True = A confirmed, False = B confirmed
    # P3-C: Mechanism fields
    trigger_condition: Optional[str] = None  # e.g., "join_tables under empty_state"
    expected_transition: Optional[str] = None  # e.g., "empty -> non_empty"
    falsifiers: Optional[List[str]] = None  # evidence that would contradict


class DiscriminatingTestEngine:
    """P0-3: Generates and executes discriminating tests between competing hypotheses."""
    def __init__(self, tracker: HypothesisTracker, seed: int = 0):
        self._tracker = tracker
        self._tests: List[DiscriminatingTest] = []
        self._results: List[dict] = []
        self._rng = random.Random(seed)
        self._saturated: bool = False

    def generate(self, known_functions: List[str]) -> Optional[DiscriminatingTest]:
        """Generate a discriminating test from a competing hypothesis pair."""
        pair = self._tracker.discriminating_pair()
        if not pair:
            return None

        ha, hb = pair

        # Try to generate a function-based test
        if ha.type == 'function_existence' and hb.type == 'function_existence':
            fn_a = ha.claim.split("'")[1] if "'" in ha.claim else ''
            fn_b = hb.claim.split("'")[1] if "'" in hb.claim else ''

            if fn_a and fn_b and fn_a != fn_b:
                test_fn = fn_a if fn_a in known_functions else (fn_b if fn_b in known_functions else None)
                if test_fn:
                    test = DiscriminatingTest(
                        test_id=f"test_{len(self._tests)}_{ha.id[:8]}_{hb.id[:8]}",
                        hypothesis_a=ha.id,
                        hypothesis_b=hb.id,
                        test_function=test_fn,
                        test_params={'data': [1, 2, 3], 'pred': 'x>0'},
                        # P3-C: Include mechanism fields from hypotheses
                        trigger_condition=ha.trigger_condition or hb.trigger_condition,
                        expected_transition=ha.expected_transition or hb.expected_transition,
                        falsifiers=(ha.falsifiers or []) + (hb.falsifiers or []),
                    )
                    self._tests.append(test)
                    return test

        return None

    def receive(self, test: DiscriminatingTest, result: dict):
        """Process test result and update hypothesis states."""
        has_disc = result.get('discovery_event') or result.get('correct_function')
        has_err = result.get('error') or result.get('unlocks_progress') is None

        passed = bool(has_disc) or (result.get('result') is not None and not has_err)

        self._tracker.update_test(test.hypothesis_a, passed, tick=test.created_at_tick, episode=test.created_at_episode)
        self._tracker.update_test(test.hypothesis_b, not passed, tick=test.created_at_tick, episode=test.created_at_episode)

        self._results.append({
            'test_id': test.test_id,
            'passed': passed,
            'hyp_a_verdict': 'confirmed' if passed else 'refuted',
            'hyp_b_verdict': 'refuted' if passed else 'confirmed',
        })

    def get_tests(self) -> List[DiscriminatingTest]:
        return list(self._tests)

    def get_results(self) -> List[dict]:
        return list(self._results)

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AdjudicationFlow:
    """Teacher-facing operations with graceful degradation when teacher is unavailable."""

    def __init__(self, loop: Any):
        self.loop = loop

    def handle_teacher_adjudication(self, competing_pair: Any, policy: Any) -> Optional[Dict[str, Any]]:
        if not competing_pair:
            return None
        hyp_a, hyp_b = competing_pair
        resolution = {
            'hyp_a_id': hyp_a.id,
            'hyp_b_id': hyp_b.id,
            'competing_pair_detected': True,
            'episode': self.loop._episode,
            'tick': self.loop._tick,
        }
        teacher = getattr(self.loop, '_teacher', None)
        if policy.should_adjudicate(competing_pair) and teacher is not None:
            teacher.teacher_adjudication(
                conflict_id=f'adj_{self.loop._episode}_{self.loop._tick}',
                resolution=resolution,
                rationale='Competing hypotheses detected, test will adjudicate',
                actor='system_hypothesis_tracker',
            )
            self.loop._teacher_log.append({
                'tick': self.loop._tick,
                'episode': self.loop._episode,
                'entry': 'teacher_adjudication',
                'hyp_a_id': hyp_a.id,
                'hyp_b_id': hyp_b.id,
            })
        return resolution

    def emit_teacher_critique(self, test: Any, probe: Any, policy: Any) -> None:
        teacher = getattr(self.loop, '_teacher', None)
        if teacher is None or test.result is None or not policy.should_critique(True, True):
            return
        confirmed_hyp = probe.hypothesis_a if test.result else probe.hypothesis_b
        falsified_hyp = probe.hypothesis_b if test.result else probe.hypothesis_a
        teacher.teacher_critique(
            target_id=confirmed_hyp,
            target_type='hypothesis',
            content={'verdict': 'confirmed', 'test_function': test.test_function, 'test_params': test.test_params},
            rationale=f'Test confirmed hypothesis via function {test.test_function}',
            actor='system_test_engine',
        )
        self.loop._teacher_log.append({
            'tick': self.loop._tick,
            'episode': self.loop._episode,
            'entry': 'teacher_critique',
            'target_id': confirmed_hyp,
            'verdict': 'confirmed',
        })
        teacher.teacher_critique(
            target_id=falsified_hyp,
            target_type='hypothesis',
            content={'verdict': 'falsified', 'test_function': test.test_function, 'test_params': test.test_params},
            rationale=f'Test falsified hypothesis via function {test.test_function}',
            actor='system_test_engine',
        )
        self.loop._teacher_log.append({
            'tick': self.loop._tick,
            'episode': self.loop._episode,
            'entry': 'teacher_critique',
            'target_id': falsified_hyp,
            'verdict': 'falsified',
        })

    def maybe_inject_recovery_task(self, recovery_task_id: str, description: str, path: Any, policy: Any) -> bool:
        teacher = getattr(self.loop, '_teacher', None)
        if teacher is None or not policy.should_inject_task(path):
            return False
        teacher.teacher_task_injection(
            task={'id': recovery_task_id, 'description': description},
            priority=0.95,
            rationale=f'Recovery path injected: {path.recovery_type.value}',
            actor='system_recovery',
        )
        return True

    def mark_probe_task_injected(self, recovery_task_id: str, recovery_type: str) -> None:
        self.loop._teacher_log.append({
            'tick': self.loop._tick,
            'episode': self.loop._episode,
            'entry': 'teacher_task_injection',
            'task': recovery_task_id,
            'recovery_type': recovery_type,
            'injected': True,
        })

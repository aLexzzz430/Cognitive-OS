from __future__ import annotations

from typing import Any, Dict


class ContinuityPersistenceAdapter:
    """Small adapter for continuity bootstrap/load/save against StateManager."""

    def __init__(self, state_mgr, state_sync, state_sync_input_cls):
        self._state_mgr = state_mgr
        self._state_sync = state_sync
        self._state_sync_input_cls = state_sync_input_cls

    def _bootstrap_continuity(self, loop) -> None:
        """Seed continuity and teacher layers so they can participate in the loop."""
        loop._continuity.identity.observe('capabilities', {'episodic_retrieval', 'hypothesis_tracking'}, source='system')
        loop._continuity.identity.observe('traits', {'exploratory'}, source='system')
        loop._continuity.identity.observe('values', {'truthfulness', 'formal_write_only'}, source='system')
        if not loop._continuity.goals.has_active_goal('goal_explore'):
            loop._continuity.goals.add_goal('goal_explore', 'Explore hidden functions and consolidate knowledge', source='self')
        if not loop._continuity.agenda.has_task('task_explore'):
            loop._continuity.agenda.add_task('task_explore', 'Continue exploration-focused interaction loop', priority=0.8)
        if not loop._continuity.experiments.get('exp_main_loop'):
            loop._continuity.experiments.start_experiment(
                'exp_main_loop',
                'Does integrated main loop improve object quality and continuity?',
                metadata={'run_id': loop.run_id, 'approved': True},
            )

    def _load_continuity(self, loop) -> None:
        """Restore persisted continuity ledgers if available."""
        snapshot = self._state_mgr.get('continuity', default={}) if self._state_mgr is not None else {}
        if isinstance(snapshot, dict) and snapshot:
            prior_verdict = (
                dict(getattr(loop, '_continuity_resume_verdict', {}))
                if isinstance(getattr(loop, '_continuity_resume_verdict', {}), dict)
                else {}
            )
            loaded = loop._continuity.load_from_dict(snapshot)
            verdict = loop._continuity.get_last_guard_verdict() or {
                'accepted': bool(loaded),
                'policy': 'degraded_restore',
                'reasons': [],
            }
            if prior_verdict:
                merged = dict(prior_verdict)
                merged.update(verdict)
                verdict = merged
            verdict['loaded'] = bool(loaded)
            loop._continuity_resume_verdict = verdict

    def _save_continuity(self, loop) -> None:
        """Persist continuity ledgers through StateManager formal state path."""
        if self._state_mgr is None:
            return
        self._state_sync.sync(self._state_sync_input_cls(
            updates={'continuity': loop._continuity.to_dict()},
            reason='continuity_snapshot_persist',
        ))

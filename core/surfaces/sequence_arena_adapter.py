"""
core/surfaces/sequence_arena_adapter.py

SurfaceAdapter for SequenceArena.

The committed objects from this arena represent "step N completed" facts.
With retrieval: agent knows current step index → generates only valid next step
Without memory: agent generates all 5 steps → random → often wrong order
"""

from __future__ import annotations
from typing import Any, Optional

from core.surfaces.base import SurfaceAdapter, SurfaceObservation


class SequenceArenaAdapter(SurfaceAdapter):
    """
    SurfaceAdapter for SequenceArena.

    update_continuity_state() syncs:
      - step_index: current position in sequence
      - completed_steps: which steps done
      - next_valid_step: what can be attempted next

    act() → ActionResult with raw containing step_index and completed list
    """

    def __init__(self, arena=None):
        from core.surfaces.sequence_arena import SequenceArena
        self._arena = arena or SequenceArena(seed=0)
        self._last_obs = None
        self._last_action = None
        self._continuity_state = {
            'phase': 'active',
            'step': 0,
        }

    def reset(self) -> SurfaceObservation:
        obs = self._arena.reset()
        self._last_obs = obs
        self._continuity_state = {'phase': 'active', 'step': 0}
        return SurfaceObservation(
            text=obs.get('last_result', ''),
            structured=obs,
            raw=obs,
        )

    def act(self, action: dict) -> 'ActionResult':
        """Execute action on arena. Returns ActionResult."""
        from core.surfaces.base import ActionResult
        result = self._arena.act(action)
        self._last_action = action.get('action', action.get('tool_name', ''))
        self._last_obs = result.observation.raw if hasattr(result, 'observation') else result
        return result

    def observe(self) -> SurfaceObservation:
        """Return current observation."""
        obs = self._arena.observe()
        return SurfaceObservation(
            text=obs.get('last_result', ''),
            structured=obs,
            raw=obs,
        )

    def get_state(self) -> dict:
        state = self._arena.get_state()
        self._continuity_state = {
            'phase': 'active',
            'step': state.get('step_index', 0),
        }
        return {
            'phase': self._continuity_state['phase'],
            'step': state.get('step_index', 0),
            'completed': state.get('completed', []),
            'total_reward': state.get('total_reward', 0.0),
        }

    def update_continuity_state(self, continuity: dict):
        """Sync continuity state from external source (e.g., record store retrieval)."""
        # If retrieved objects indicate "step 2 completed", update arena state
        completed = continuity.get('completed_steps', [])
        step_index = continuity.get('step_index', 0)
        if completed or step_index:
            # Sync the arena state
            self._arena._state.completed_steps = completed
            self._arena._state.current_step_index = step_index
        self._continuity_state = {
            'phase': continuity.get('phase', 'active'),
            'step': step_index,
        }

    def get_raw_evidence(self) -> dict:
        """Return raw evidence from last action for RawEvidenceExtractor."""
        return {
            'action': self._last_action,
            'observation': self._last_obs,
            'arena_state': self.get_state(),
        }

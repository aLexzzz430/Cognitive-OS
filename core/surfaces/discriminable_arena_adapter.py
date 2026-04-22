"""
core/surfaces/discriminable_arena_adapter.py

SurfaceAdapter for DiscriminableArena.

The key discriminator: with-memory agent knows which properties were investigated,
and can choose "solve" only when all 5 are done. Fresh agent doesn't know
and will waste actions re-investigating.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from core.surfaces.base import SurfaceAdapter, SurfaceObservation, SurfaceAction, ActionResult


@dataclass
class DiscriminableArenaAction:
    """An action in the discriminable arena."""
    kind: str = "execute"
    action: str = "wait"
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'action': self.action,
            'tool_name': self.tool_name,
            'tool_args': self.tool_args,
        }


class DiscriminableArenaAdapter(SurfaceAdapter):
    """
    SurfaceAdapter for DiscriminableArena.

    NovelAPI tools:
      - investigate_P0, investigate_P1, ..., investigate_P4
      - solve (only succeeds if all 5 investigated)
      - wait
    """

    def __init__(self, seed: int = 0):
        from core.surfaces.discriminable_arena import DiscriminableArena
        self._env = DiscriminableArena(seed=seed)
        self._last_obs: dict = {}

    def reset(self) -> SurfaceObservation:
        obs = self._env.reset()
        self._last_obs = obs
        return SurfaceObservation(
            text=obs.get('result', ''),
            structured=obs,
            raw=obs,
        )

    def observe(self) -> SurfaceObservation:
        obs = self._env.observe()
        self._last_obs = obs
        return SurfaceObservation(
            text=str(obs),
            structured=obs,
            raw=obs,
        )

    def act(self, action: SurfaceAction) -> ActionResult:
        """Execute an action. Returns ActionResult with reward."""
        if isinstance(action, dict):
            action_dict = action
        elif hasattr(action, 'to_dict'):
            action_dict = action.to_dict()
        else:
            action_dict = {'action': 'wait'}

        result = self._env.act(action_dict)

        obs = self._env.observe()
        self._last_obs = obs
        return ActionResult(
            ok=result.get('ok', False),
            observation=SurfaceObservation(
                text=result.get('result', ''),
                structured=obs,
                raw=obs,
            ),
            events=[],
            raw=result,
        )

    def get_state(self) -> dict:
        return self._env.get_state()

    def update_continuity_state(self, continuity: dict) -> None:
        # Store investigated properties in continuity for Step 3 retrieval
        state = self._env.get_state()
        continuity['investigated_properties'] = state.get('investigated', [])
        continuity['solved'] = state.get('solved', False)
        continuity['step'] = state.get('step', 0)

    def get_investigated_properties(self) -> set:
        return self._env.get_investigated_properties()

    def get_total_reward(self) -> float:
        return self._env.get_total_reward()

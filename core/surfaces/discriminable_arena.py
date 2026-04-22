"""
core/surfaces/discriminable_arena.py

A discriminable utility arena for G1 utility auditing.

Design principles:
  1. With-memory and fresh CAN take different actions
  2. Different actions lead to different rewards
  3. Retrieved committed objects can reasonably influence candidate ranking

Mechanics:
  - Hidden function has 5 discoverable properties (P0-P4)
  - Each episode, agent can "investigate" one property
  - When all 5 are investigated, function is "solved" (high reward)
  - Memory lets you know which properties were already investigated
  - Candidates: investigate_P0, investigate_P1, ..., solve, wait
  - Selecting an already-investigated property gives low/no reward
  - Selecting a never-investigated property gives high reward
  - Selecting "solve" without all 5 gives zero reward
  - Selecting "solve" with all 5 gives maximum reward

Thus:
  - With-memory: avoids already-investigated, solves correctly → HIGH reward
  - Fresh: randomly selects (or selects P0 each time), low chance of solving
  - Retrieve-on/reference-on: knows history, avoids duplicates
  - Retrieve-on/reference-masked: knows history but can't use it → same as fresh
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import random


@dataclass
class DiscriminableArenaState:
    """Track what's been discovered."""
    investigated_properties: set = field(default_factory=set)  # {P0, P1, P2, P3, P4}
    solved: bool = False
    episode: int = 0
    total_reward: float = 0.0
    steps_taken: int = 0

    def is_complete(self) -> bool:
        return len(self.investigated_properties) >= 5 or self.solved


class DiscriminableArena:
    """
    Discriminable environment where memory of past investigations changes behavior.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._state = DiscriminableArenaState()
        self._step: int = 0
        self._hidden_properties = {'P0', 'P1', 'P2', 'P3', 'P4'}
        self._hidden_preconditions = {
            'P0': ['call_api'],
            'P1': ['call_api', 'P0_done'],
            'P2': ['P0_done', 'P1_done'],
            'P3': ['P1_done'],
            'P4': ['P2_done'],
            'solve': ['P0_done', 'P1_done', 'P2_done', 'P3_done', 'P4_done'],
        }

    def reset(self):
        self._state = DiscriminableArenaState()
        self._step = 0
        return self.observe()

    def observe(self) -> dict:
        """Return current observation."""
        return {
            'type': 'discriminable_arena',
            'step': self._step,
            'episode': self._state.episode,
            'investigated_count': len(self._state.investigated_properties),
            'properties': sorted(self._state.investigated_properties),
            'solved': self._state.solved,
            'available_actions': self._available_actions(),
            'total_reward': self._state.total_reward,
        }

    def _available_actions(self) -> list[str]:
        """Return list of currently valid actions."""
        if self._state.solved:
            return ['wait', 'noop']
        actions = []
        for p in sorted(self._hidden_properties):
            if p not in self._state.investigated_properties:
                actions.append(f'investigate_{p}')
        actions.append('solve')
        if self._rng.random() < 0.3:
            actions.append('wait')
        return actions

    def act(self, action: dict) -> dict:
        """
        Execute action.

        action = {
            'kind': 'execute' | 'call_tool',
            'tool_name': str,
            'tool_args': dict,
            'action': str,  # 'investigate_P0', 'investigate_P1', 'solve', 'wait'
        }

        Returns:
            ok, observation, raw (with reward_signal)
        """
        from core.surfaces.base import ActionResult, SurfaceObservation

        self._step += 1
        self._state.steps_taken += 1

        # Parse action
        action_str = action.get('action', 'wait')
        if not action_str:
            action_str = action.get('tool_name', 'wait')

        raw = {'action_taken': action_str}

        reward = 0.0
        done = False

        if action_str == 'wait' or action_str == 'noop':
            reward = 0.0
            raw['result'] = 'No-op selected.'

        elif action_str.startswith('investigate_'):
            prop = action_str[len('investigate_'):]
            if prop in self._hidden_properties and prop not in self._state.investigated_properties:
                self._state.investigated_properties.add(prop)
                reward = 1.0  # Fresh discovery
                raw['result'] = f'Investigated {prop}. Properties now known: {sorted(self._state.investigated_properties)}'
                raw['discovery'] = {'property': prop}
                raw['reward_signal'] = reward
                raw['visible_docs'] = [{
                    'function_name': f'investigate_{prop}',
                    'signature': f'investigate_{prop}()',
                    'required_args': [],
                    'description': f'Investigate property {prop} of the hidden function.',
                }]
            elif prop in self._state.investigated_properties:
                reward = -0.2  # Wasted effort
                raw['result'] = f'Property {prop} already investigated. No new information.'
                raw['reward_signal'] = reward
            else:
                reward = -0.5
                raw['result'] = f'Unknown property: {prop}'
                raw['reward_signal'] = reward

        elif action_str == 'solve':
            # Check preconditions
            needed = self._hidden_preconditions.get('solve', [])
            missing = []
            for req in needed:
                prop = req.replace('_done', '')
                if prop not in self._state.investigated_properties:
                    missing.append(prop)

            if not missing:
                self._state.solved = True
                reward = 10.0  # Big reward for solving
                raw['result'] = 'Function solved! All properties discovered.'
                raw['reward_signal'] = reward
                raw['success'] = True
                done = True
            else:
                reward = -1.0  # Penalty for solving prematurely
                raw['result'] = f'Cannot solve. Missing properties: {missing}'
                raw['reward_signal'] = reward

        else:
            reward = -0.1
            raw['result'] = f'Unknown action: {action_str}'
            raw['reward_signal'] = reward

        self._state.total_reward += reward
        raw['reward_signal'] = reward
        raw['ok'] = reward >= 0

        obs = SurfaceObservation(
            text=raw.get('result', ''),
            structured=raw,
            raw=raw,
        )

        return ActionResult(
            ok=raw.get('ok', False),
            observation=obs,
            events=[],
            raw=raw,
        )

    def get_state(self) -> dict:
        return {
            'investigated': sorted(self._state.investigated_properties),
            'solved': self._state.solved,
            'step': self._step,
            'total_reward': self._state.total_reward,
        }

    def get_investigated_properties(self) -> set:
        """Return set of already investigated properties."""
        return set(self._state.investigated_properties)

    def get_total_reward(self) -> float:
        return self._state.total_reward

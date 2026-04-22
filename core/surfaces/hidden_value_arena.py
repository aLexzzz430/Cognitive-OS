"""
core/surfaces/hidden_value_arena.py

Second discriminable utility arena — DIFFERENT mechanism from "avoid duplicates".

Mechanism: Hidden Value Map

- 4 locations: L0, L1, L2, L3
- Each location has a HIDDEN value (unknown until explored)
- Hidden values: L0=3, L1=7, L2=2, L3=9
- Each step: choose exploit (known location) or explore (random)
- Exploit: get reward = known value (or 0 if unknown)
- Explore: reveal a location's value (add to known map)
- Total 10 steps, maximize total reward

Key discriminator:
  - With-memory: learns the map, exploits high-value locations → high total
  - Fresh: doesn't know values, may exploit low-value locations → lower total

This is DIFFERENT from "avoid duplicates" because:
  - No "duplicate" concept
  - Value is about SPATIAL/QUALITATIVE knowledge, not temporal avoidance
  - The decision is exploit-vs-explore, not "have I seen this before"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random


HIDDEN_VALUES = {
    'L0': 3,
    'L1': 7,
    'L2': 2,
    'L3': 9,
}

HIGHEST_VALUE_LOCATION = 'L3'  # Value = 9


@dataclass
class HiddenValueState:
    known_values: dict = field(default_factory=dict)  # location -> value
    steps_taken: int = 0
    total_reward: float = 0.0
    last_action: str = ""
    last_result: str = ""


class HiddenValueArena:
    """
    Hidden Value Map arena.

    Actions:
      - exploit_L0, exploit_L1, exploit_L2, exploit_L3
      - explore  (reveals one random unknown location)
      - wait

    Discriminator:
      - With-memory: knows which locations have high value, exploits L3 → high reward
      - Fresh: random exploitation → lower average
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._state = HiddenValueState()
        self._max_steps = 10

    def reset(self) -> dict:
        self._state = HiddenValueState()
        return self.observe()

    def observe(self) -> dict:
        """Return current observation."""
        return {
            'type': 'hidden_value_map',
            'step': self._state.steps_taken,
            'known_values': dict(self._state.known_values),
            'last_action': self._state.last_action,
            'last_result': self._state.last_result,
            'total_reward': self._state.total_reward,
            'available_actions': self._available_actions(),
        }

    def _available_actions(self) -> list:
        actions = []
        for loc in sorted(HIDDEN_VALUES.keys()):
            actions.append(f'exploit_{loc}')
        actions.append('explore')
        if self._state.steps_taken < self._max_steps:
            actions.append('wait')
        return actions

    def act(self, action: dict) -> dict:
        """Execute action."""
        from core.surfaces.base import ActionResult, SurfaceObservation

        action_str = action.get('action', '') or action.get('tool_name', 'wait')
        self._state.steps_taken += 1

        reward = 0.0
        result = ''
        discovered_location = None
        done = self._state.steps_taken >= self._max_steps

        if action_str == 'wait':
            reward = 0.0
            result = 'No-op.'

        elif action_str.startswith('exploit_'):
            loc = action_str[len('exploit_'):]
            if loc in self._state.known_values:
                reward = float(self._state.known_values[loc])
                result = f'Exploited {loc}, reward = {reward}'
            else:
                reward = 0.0
                result = f'Exploited {loc} but value unknown → 0. (Should have explored first!)'

        elif action_str == 'explore':
            # Reveal one random unknown location
            unknown = [loc for loc in HIDDEN_VALUES if loc not in self._state.known_values]
            if unknown:
                loc = self._rng.choice(unknown)
                self._state.known_values[loc] = HIDDEN_VALUES[loc]
                discovered_location = loc
                reward = 0.5  # Small reward for exploration
                result = f'Explored: discovered {loc} = {HIDDEN_VALUES[loc]}. Known map: {self._state.known_values}'
            else:
                # All known, small penalty
                reward = -0.1
                result = 'All locations already known.'

        self._state.total_reward += reward
        self._state.last_action = action_str
        self._state.last_result = result

        raw = {
            'action_taken': action_str,
            'result': result,
            'reward_signal': reward,
            'discovered_location': discovered_location,
            'known_values': dict(self._state.known_values),
            'total_reward': self._state.total_reward,
            'done': done,
            'ok': reward >= 0,
        }

        return ActionResult(
            ok=raw.get('ok', False),
            observation=SurfaceObservation(
                text=result,
                structured=raw,
                raw=raw,
            ),
            events=[],
            raw=raw,
        )

    def get_state(self) -> dict:
        return {
            'known_values': dict(self._state.known_values),
            'total_reward': self._state.total_reward,
            'steps': self._state.steps_taken,
        }

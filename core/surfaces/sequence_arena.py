"""
core/surfaces/sequence_arena.py

Third Discriminable Utility Family: Sequence / Prerequisite Dependency

Mechanism: Multi-step task with ordering constraints
- 5 steps: A → B → C → D → E (must be done in order)
- Wrong order: step fails + penalty
- Correct order: step succeeds + small reward
- All 5 correct: bonus reward

This is DIFFERENT from the other two families:
- Anti-redundancy: "have I already done this?" (temporal dedup)
- Hidden value map: "which location has highest value?" (spatial optimization)
- Sequence dependency: "is this the right step in the right order?" (prerequisite constraint)

Key discriminator:
  - With memory: knows current step index → only attempts correct next step → high reward
  - Without memory: random step selection → often wrong order → low reward

Evidence from the audit shows:
  - Memory retrieval of "step N completed" changes candidate generation
  - Candidate generation includes "step N+1" as only valid next action
  - Without retrieval, all 5 steps appear equally valid
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random


# The 5-step sequence (order matters)
SEQUENCE_STEPS = ['step_a', 'step_b', 'step_c', 'step_d', 'step_e']
SEQUENCE_DESCRIPTIONS = {
    'step_a': 'initialize the workspace',
    'step_b': 'load the configuration',
    'step_c': 'validate inputs',
    'step_d': 'execute the core logic',
    'step_e': 'produce the output',
}
COMPLETION_BONUS = 5.0  # Extra reward for completing full sequence


@dataclass
class SequenceState:
    completed_steps: list = field(default_factory=list)
    current_step_index: int = 0
    total_reward: float = 0.0
    last_action: str = ""
    last_result: str = ""
    steps_taken: int = 0


class SequenceArena:
    """
    Sequence/Prerequisite Dependency arena.

    Actions:
      - step_a, step_b, step_c, step_d, step_e (only one is valid at any time)
      - attempt_any (attempts the current step by name)

    Valid step at any point: the one at current_step_index
    Wrong step: fails with -0.5 penalty
    Correct step: +1.0 reward, advance index

    With-memory: knows current_step_index → only generates valid next step
    Without-memory: generates all 5 steps → random selection → many wrong
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._state = SequenceState()
        self._max_steps = 15

    def reset(self) -> dict:
        self._state = SequenceState()
        return self.observe()

    def observe(self) -> dict:
        """Return current observation."""
        return {
            'type': 'sequence_dependency',
            'step_index': self._state.current_step_index,
            'completed': list(self._state.completed_steps),
            'last_action': self._state.last_action,
            'last_result': self._state.last_result,
            'total_reward': self._state.total_reward,
            'available_actions': self._available_actions(),
        }

    def _available_actions(self) -> list:
        actions = []
        # Only the correct next step is truly "available"
        if self._state.current_step_index < len(SEQUENCE_STEPS):
            next_step = SEQUENCE_STEPS[self._state.current_step_index]
            actions.append(next_step)
        # But we also allow attempting other steps (they'll fail)
        for step in SEQUENCE_STEPS:
            if step not in actions:
                actions.append(step)
        return actions

    def act(self, action: dict) -> dict:
        """Execute action."""
        from core.surfaces.base import ActionResult, SurfaceObservation

        action_str = action.get('action', '') or action.get('tool_name', 'wait')
        self._state.steps_taken += 1

        reward = 0.0
        result = ''
        done = self._state.steps_taken >= self._max_steps

        expected_step = None
        if self._state.current_step_index < len(SEQUENCE_STEPS):
            expected_step = SEQUENCE_STEPS[self._state.current_step_index]

        # Check if action matches expected step
        if action_str == expected_step:
            # Correct step!
            self._state.completed_steps.append(expected_step)
            self._state.current_step_index += 1
            reward = 1.0
            result = f'{expected_step}: correct step in sequence'

            # Bonus for completing full sequence
            if self._state.current_step_index >= len(SEQUENCE_STEPS):
                reward += COMPLETION_BONUS
                result += f' — sequence complete! bonus +{COMPLETION_BONUS}'
                done = True

        elif action_str in SEQUENCE_STEPS:
            # Wrong step — prerequisite not met
            self._state.total_reward -= 0.5
            self._state.steps_taken += 0  # No advance
            reward = -0.5
            result = f'{action_str}: wrong step — must complete {expected_step} first'

        else:
            reward = -0.1
            result = 'invalid action'

        self._state.total_reward += reward
        self._state.last_action = action_str
        self._state.last_result = result

        raw = {
            'action_taken': action_str,
            'result': result,
            'reward_signal': reward,
            'step_index': self._state.current_step_index,
            'completed': list(self._state.completed_steps),
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
            'step_index': self._state.current_step_index,
            'completed': list(self._state.completed_steps),
            'total_reward': self._state.total_reward,
            'steps': getattr(self._state, 'steps_taken', 0),
        }

    @property
    def current_step(self) -> str:
        """Returns the current expected step name, or None if complete."""
        if self._state.current_step_index < len(SEQUENCE_STEPS):
            return SEQUENCE_STEPS[self._state.current_step_index]
        return None

    @property
    def remaining_steps(self) -> list:
        """Returns remaining steps."""
        return SEQUENCE_STEPS[self._state.current_step_index:]
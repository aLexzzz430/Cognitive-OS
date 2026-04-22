from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArcGameSessionState:
    requested_game_id: str
    resolved_game_id: str = ""
    seed: int = 0
    guid: str = ""
    episode_index: int = 1
    turn_index: int = 0
    total_reward: float = 0.0
    solved: bool = False
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    raw_last_observation: Dict[str, Any] = field(default_factory=dict)

    inspect_count: int = 0
    external_noop_cost: int = 0
    internal_action_history: List[Dict[str, Any]] = field(default_factory=list)

    def reset_for_episode(
        self,
        guid: str = "",
        resolved_game_id: str = "",
        episode_index: Optional[int] = None,
    ) -> None:
        self.guid = guid or self.guid
        self.resolved_game_id = resolved_game_id or self.resolved_game_id or self.requested_game_id
        if episode_index is not None:
            self.episode_index = int(episode_index)
        self.turn_index = 0
        self.total_reward = 0.0
        self.solved = False
        self.action_history = []
        self.observation_history = []
        self.raw_last_observation = {}
        self.inspect_count = 0
        self.external_noop_cost = 0
        self.internal_action_history = []

    def record_observation(self, obs: Dict[str, Any]) -> None:
        safe = dict(obs) if isinstance(obs, dict) else {}
        self.raw_last_observation = safe
        self.observation_history.append(safe)
        guid = str(safe.get("guid", "") or "").strip()
        resolved_game_id = str(safe.get("resolved_game_id", "") or safe.get("game_id", "") or "").strip()
        if guid:
            self.guid = guid
        if resolved_game_id:
            self.resolved_game_id = resolved_game_id

    def record_action(self, action: Dict[str, Any], response: Dict[str, Any], reward: float) -> None:
        self.turn_index += 1
        self.total_reward += float(reward or 0.0)
        self.solved = bool(response.get("solved", False) or response.get("success", False) and response.get("terminal", False))
        self.action_history.append(
            {
                "turn": self.turn_index,
                "action": dict(action) if isinstance(action, dict) else {"raw": str(action)},
                "reward": float(reward or 0.0),
                "terminal": bool(response.get("terminal", False) or response.get("done", False)),
                "success": bool(response.get("success", False)),
                "state": str(response.get("state", "")),
                "guid": str(response.get("guid", "") or self.guid),
                "resolved_game_id": str(response.get("resolved_game_id", "") or self.resolved_game_id),
            }
        )

    def record_internal_action(
        self,
        *,
        action_kind: str,
        reward: float,
        external_cost: int,
        learning_delta: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        self.turn_index += 1
        self.total_reward += float(reward or 0.0)
        self.external_noop_cost += int(external_cost or 0)
        if action_kind == "inspect":
            self.inspect_count += 1
        self.solved = bool(response.get("solved", False))

        self.internal_action_history.append(
            {
                "turn": self.turn_index,
                "action_kind": str(action_kind),
                "reward": float(reward or 0.0),
                "external_cost": int(external_cost or 0),
                "learning_delta": dict(learning_delta or {}),
                "state": str(response.get("state", "")),
                "terminal": bool(response.get("terminal", False)),
                "guid": str(response.get("guid", "") or self.guid),
                "resolved_game_id": str(response.get("resolved_game_id", "") or self.resolved_game_id),
            }
        )

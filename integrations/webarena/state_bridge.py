from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WebArenaSessionState:
    task_id: str = ""
    task_instruction: str = ""
    episode_index: int = 1
    turn_index: int = 0
    total_reward: float = 0.0
    terminal: bool = False
    current_url: str = ""
    current_title: str = ""
    active_tab_id: str = ""
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[Dict[str, Any]] = field(default_factory=list)
    raw_last_observation: Dict[str, Any] = field(default_factory=dict)

    def reset_for_episode(
        self,
        *,
        task_id: str = "",
        task_instruction: str = "",
        episode_index: Optional[int] = None,
    ) -> None:
        if task_id:
            self.task_id = str(task_id)
        if task_instruction:
            self.task_instruction = str(task_instruction)
        if episode_index is not None:
            self.episode_index = int(episode_index)
        self.turn_index = 0
        self.total_reward = 0.0
        self.terminal = False
        self.current_url = ""
        self.current_title = ""
        self.active_tab_id = ""
        self.action_history = []
        self.observation_history = []
        self.raw_last_observation = {}

    def record_observation(self, obs: Dict[str, Any]) -> None:
        safe = dict(obs) if isinstance(obs, dict) else {}
        self.raw_last_observation = safe
        self.observation_history.append(safe)
        metadata = safe.get("metadata", {}) if isinstance(safe.get("metadata", {}), dict) else {}
        page = safe.get("page", {}) if isinstance(safe.get("page", {}), dict) else {}
        if str(metadata.get("task_id", "") or "").strip():
            self.task_id = str(metadata.get("task_id", "") or "").strip()
        if str(metadata.get("instruction", "") or "").strip():
            self.task_instruction = str(metadata.get("instruction", "") or "").strip()
        if str(page.get("url", "") or "").strip():
            self.current_url = str(page.get("url", "") or "").strip()
        if str(page.get("title", "") or "").strip():
            self.current_title = str(page.get("title", "") or "").strip()
        if str(page.get("active_tab_id", "") or "").strip():
            self.active_tab_id = str(page.get("active_tab_id", "") or "").strip()

    def record_action(self, action: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        self.turn_index += 1
        self.total_reward += float(reward or 0.0)
        self.terminal = bool(result.get("terminal", False) or result.get("done", False))
        self.action_history.append(
            {
                "turn": self.turn_index,
                "action": dict(action) if isinstance(action, dict) else {"raw": str(action)},
                "reward": float(reward or 0.0),
                "terminal": self.terminal,
                "success": bool(result.get("success", False)),
                "state": str(result.get("state", "")),
                "url": str(result.get("url", "") or self.current_url),
            }
        )

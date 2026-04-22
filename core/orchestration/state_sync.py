from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StateSyncInput:
    updates: Dict[str, Any]
    reason: str
    module: str = 'core'


class StateSyncOrchestrator:
    """Single-point state sync wrapper for main loop orchestration."""

    def __init__(self, state_manager: Any):
        self._state_manager = state_manager

    def sync(self, input_obj: StateSyncInput) -> None:
        self._state_manager.update_state(dict(input_obj.updates), reason=input_obj.reason, module=input_obj.module)

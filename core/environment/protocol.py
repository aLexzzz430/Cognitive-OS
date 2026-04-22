from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from core.environment.types import (
    GenericActionEnvelope,
    GenericObservation,
    GenericTaskSpec,
    GenericTransition,
)


@runtime_checkable
class GenericEnvironmentAdapter(Protocol):
    """
    Higher-level environment protocol shared across domains.

    SurfaceAdapter only standardizes reset/observe/act.
    This protocol standardizes the semantic units that the world model consumes:
    task spec, observation graph, parameterized action envelope, and transition.
    """

    def get_generic_task_spec(self) -> GenericTaskSpec: ...

    def get_generic_observation(
        self,
        obs: Optional[Dict[str, Any]] = None,
    ) -> GenericObservation: ...

    def get_generic_action(
        self,
        action: Any,
        *,
        obs: Optional[Dict[str, Any]] = None,
    ) -> GenericActionEnvelope: ...

    def get_generic_transition(
        self,
        *,
        obs_before: Optional[Dict[str, Any]],
        action: Any,
        result: Any,
    ) -> GenericTransition: ...

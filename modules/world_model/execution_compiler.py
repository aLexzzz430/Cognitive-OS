from __future__ import annotations

"""Generic execution compiler interfaces.

Compilers translate intervention targets into environment actions. The core
interface is generic; environment-specific compilers should live next to their
respective integrations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from modules.world_model.intervention_targets import InterventionTarget


@dataclass(frozen=True)
class CompiledAction:
    action_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rationale: List[str] = field(default_factory=list)

    def to_action_dict(self) -> Dict[str, Any]:
        return {
            "kind": "call_tool",
            "payload": {
                "tool_name": "call_hidden_function",
                "tool_args": {
                    "function_name": self.action_name,
                    "kwargs": dict(self.kwargs),
                },
            },
            "_candidate_meta": {
                "compiled_from_intervention_target": True,
                "compiler_score": float(self.score),
                "compiler_rationale": list(self.rationale),
            },
        }


class ExecutionCompiler(Protocol):
    def compile(
        self,
        target: InterventionTarget,
        *,
        available_functions: Sequence[str],
        obs: Optional[Dict[str, Any]] = None,
    ) -> List[CompiledAction]: ...


class NullExecutionCompiler:
    """Fallback compiler that returns no actions."""

    def compile(
        self,
        target: InterventionTarget,
        *,
        available_functions: Sequence[str],
        obs: Optional[Dict[str, Any]] = None,
    ) -> List[CompiledAction]:
        _ = (target, available_functions, obs)
        return []

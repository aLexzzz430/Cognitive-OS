from __future__ import annotations

from typing import Any, Dict, List, Sequence


def search_candidate_outputs(
    *,
    workspace: Dict[str, Any],
    obs: Dict[str, Any],
    candidate_programs: Sequence[Dict[str, Any]],
    synthesizer: Any = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    if isinstance(obs, dict) and "arc_task" in obs and synthesizer is not None:
        outputs = synthesizer.enumerate_arc_candidate_outputs(
            obs.get("arc_task"),
            candidate_programs=candidate_programs,
            limit=limit,
        )
        return [dict(item) for item in outputs if isinstance(item, dict)]

    candidate_outputs = workspace.get("candidate_outputs", [])
    if not isinstance(candidate_outputs, list):
        candidate_outputs = []
    return [dict(item) for item in candidate_outputs[: max(0, int(limit))] if isinstance(item, dict)]

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
    candidate_outputs = workspace.get("candidate_outputs", [])
    if not isinstance(candidate_outputs, list):
        candidate_outputs = []
    return [dict(item) for item in candidate_outputs[: max(0, int(limit))] if isinstance(item, dict)]

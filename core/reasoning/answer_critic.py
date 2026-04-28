from __future__ import annotations

from typing import Any, Dict, List, Sequence

def rank_candidate_outputs(
    candidate_outputs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in candidate_outputs:
        if not isinstance(row, dict):
            continue
        score = float(row.get("score", 0.0) or 0.0)
        transition_alignment = float(row.get("transition_alignment", 0.0) or 0.0)
        first_output_state = row.get("first_output_state", {}) if isinstance(row.get("first_output_state", {}), dict) else {}
        structure_bonus = 0.0
        if first_output_state:
            structure_bonus += min(0.15, float(first_output_state.get("component_count", 0) or 0.0) * 0.03)
            structure_bonus += min(0.10, float(first_output_state.get("distinct_non_background_colors", 0) or 0.0) * 0.02)
        enriched = dict(row)
        enriched["critic_score"] = round(score + (transition_alignment * 0.35) + structure_bonus, 6)
        rows.append(enriched)
    rows.sort(key=lambda item: float(item.get("critic_score", 0.0) or 0.0), reverse=True)
    return rows

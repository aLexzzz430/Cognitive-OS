"""State sync helpers owned by world_model module."""

from __future__ import annotations

from typing import Any, Dict, List


def emit_state_patch(
    *,
    belief_count: int,
    active_beliefs: List[Any],
    established_beliefs: List[Any],
    high_uncertainty_beliefs: List[Any],
) -> Dict[str, Any]:
    """Build world_model namespace patch from belief-ledger summaries."""
    belief_state_summary = {
        'total_beliefs': int(belief_count),
        'active_count': len(active_beliefs),
        'established_count': len(established_beliefs),
        'uncertain_count': len(high_uncertainty_beliefs),
    }
    return {
        'world_model.belief_state': belief_state_summary,
        'world_model.belief_state.confidence': {
            b.belief_id: b.confidence for b in active_beliefs
        },
        'world_model.active_mechanisms': [
            {'id': b.belief_id, 'variable': b.variable_name, 'posterior': b.posterior}
            for b in established_beliefs
        ],
        'world_model.boundary_flags': [
            {'id': b.belief_id, 'variable': b.variable_name, 'uncertainty': b.uncertainty}
            for b in high_uncertainty_beliefs
        ],
    }

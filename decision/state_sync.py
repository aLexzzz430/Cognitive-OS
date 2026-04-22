"""Decision-owned state patch emitters."""

from __future__ import annotations

from typing import Any, Dict


def emit_state_patch(*, retrieval_aux_decisions: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'decision_context.retrieval_aux_decisions': dict(retrieval_aux_decisions or {}),
    }


def emit_surfacing_patch(*, surfacing_protocol: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'decision_context.retrieval_surfacing_protocol': dict(surfacing_protocol or {}),
    }

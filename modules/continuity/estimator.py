"""Continuity confidence estimator.

The estimator is intentionally lightweight and reusable so orchestration layers
can delegate continuity-confidence inference here.
"""

from typing import Any, Dict, Optional


def estimate_continuity_confidence(
    continuity_snapshot: Optional[Dict[str, Any]],
    *,
    fallback: float = 0.5,
) -> float:
    """Estimate continuity confidence from continuity snapshot signals."""
    if isinstance(continuity_snapshot, dict):
        score = float(fallback)
        for key in ('continuity_confidence', 'confidence'):
            value = continuity_snapshot.get(key)
            if value is not None:
                score = float(value)
                break

        durable_identity = continuity_snapshot.get('durable_identity', {})
        if isinstance(durable_identity, dict) and durable_identity:
            score += 0.10
        commitments = continuity_snapshot.get('active_commitments', [])
        if isinstance(commitments, list) and commitments:
            score += min(0.15, len(commitments) * 0.04)
        autobiographical = continuity_snapshot.get('autobiographical_summary', {})
        if isinstance(autobiographical, dict) and autobiographical:
            score += 0.12
        subject_continuity = continuity_snapshot.get('subject_continuity', {})
        if isinstance(subject_continuity, dict):
            score += max(0.0, min(0.25, float(subject_continuity.get('continuity_score', 0.0) or 0.0) * 0.25))
        return max(0.0, min(1.0, score))

    return max(0.0, min(1.0, float(fallback)))

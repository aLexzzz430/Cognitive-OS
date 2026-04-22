from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.reasoning.causal_inference import run_causal_inference


def update_hypothesis_posteriors(
    hypotheses: List[Dict[str, Any]],
    *,
    action: Dict[str, Any],
    result: Dict[str, Any],
    predicted_transition: Dict[str, Any] | None,
    actual_transition: Dict[str, Any] | None,
    reward: float,
    information_gain: float,
    obs_before: Optional[Dict[str, Any]] = None,
    obs_after: Optional[Dict[str, Any]] = None,
    world_model_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return run_causal_inference(
        list(hypotheses or []),
        action=dict(action or {}) if isinstance(action, dict) else {},
        result=dict(result or {}) if isinstance(result, dict) else {},
        predicted_transition=dict(predicted_transition or {}) if isinstance(predicted_transition, dict) else {},
        actual_transition=dict(actual_transition or {}) if isinstance(actual_transition, dict) else {},
        reward=float(reward or 0.0),
        information_gain=float(information_gain or 0.0),
        obs_before=dict(obs_before or {}) if isinstance(obs_before, dict) else None,
        obs_after=dict(obs_after or {}) if isinstance(obs_after, dict) else None,
        world_model_summary=dict(world_model_summary or {}) if isinstance(world_model_summary, dict) else None,
    )

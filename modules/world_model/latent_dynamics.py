from __future__ import annotations

from typing import Any, Dict, List


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _phase_alias(raw_phase: Any) -> str:
    phase = str(raw_phase or "").strip().lower()
    aliases = {
        "explore": "exploring",
        "stable": "stabilizing",
        "stabilize": "stabilizing",
        "commit": "committed",
        "completed": "committed",
        "failed": "disrupted",
        "fail": "disrupted",
        "error": "disrupted",
    }
    return aliases.get(phase, phase or "exploring")


def _flatten_prior_entry(entry: Dict[str, Any]) -> Dict[str, float]:
    metrics = _as_dict(entry.get("metrics", {}))
    flattened = {
        "long_horizon_reward": float(entry.get("long_horizon_reward", 0.0) or 0.0),
        "predicted_risk": _clamp01(entry.get("predicted_risk", 0.0), 0.0),
        "reversibility": _clamp01(entry.get("reversibility", 0.0), 0.0),
        "info_gain": _clamp01(entry.get("info_gain", 0.0), 0.0),
        "constraint_violation": _clamp01(entry.get("constraint_violation", 0.0), 0.0),
    }
    for metric_name in ("long_horizon_reward", "predicted_risk", "reversibility", "info_gain"):
        metric_payload = _as_dict(metrics.get(metric_name, {}))
        if metric_payload:
            flattened[metric_name] = float(metric_payload.get("value", flattened[metric_name]) or flattened[metric_name])
            flattened[f"{metric_name}_confidence"] = _clamp01(metric_payload.get("confidence", 0.0), 0.0)
    return flattened


def summarize_latent_dynamics(
    world_model_summary: Dict[str, Any],
    transition_priors: Dict[str, Any],
    *,
    limit: int = 6,
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    priors = dict(transition_priors or {})
    hidden_state = _as_dict(summary.get("hidden_state", {}))
    current_phase = _phase_alias(hidden_state.get("phase", summary.get("predicted_phase", "exploring")))
    expected_next_phase = _phase_alias(
        hidden_state.get("expected_next_phase", summary.get("expected_next_phase", current_phase))
    )
    transition_confidence = _clamp01(summary.get("transition_confidence", hidden_state.get("phase_confidence", 0.0)), 0.0)
    hidden_uncertainty = _clamp01(hidden_state.get("uncertainty_score", 0.0), 0.0)
    transition_entropy = _clamp01(summary.get("phase_transition_entropy", hidden_state.get("transition_entropy", 1.0)), 1.0)
    world_novelty = _clamp01(summary.get("world_novelty_score", hidden_state.get("novelty_score", 0.0)), 0.0)

    by_signature = _as_dict(priors.get("__by_signature", {}))
    predicted_rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(by_signature.values()):
        if not isinstance(entry, dict):
            continue
        key = _as_dict(entry.get("key", {}))
        function_name = str(key.get("function_name", "") or "")
        if not function_name:
            continue
        flat = _flatten_prior_entry(entry)
        risk = _clamp01(flat.get("predicted_risk", 0.0), 0.0)
        info_gain = _clamp01(flat.get("info_gain", 0.0), 0.0)
        affinity = float(entry.get("transition_affinity", 0.0) or 0.0)
        if risk >= 0.62:
            target_phase = "disrupted"
        elif affinity >= 0.18 or flat.get("long_horizon_reward", 0.0) > 0.18:
            target_phase = expected_next_phase or current_phase
        else:
            target_phase = current_phase
        predicted_rows.append(
            {
                "transition_id": f"predicted_transition_{index}",
                "function_name": function_name,
                "from_phase": current_phase,
                "to_phase": target_phase,
                "confidence": round(_clamp01((transition_confidence * 0.45) + (1.0 - transition_entropy) * 0.20 + (1.0 - risk) * 0.15 + info_gain * 0.20, 0.0), 4),
                "state_shift_risk": round(risk, 4),
                "expected_information_gain": round(info_gain, 4),
                "reversibility": round(_clamp01(flat.get("reversibility", 0.0), 0.0), 4),
                "expected_reward": round(float(flat.get("long_horizon_reward", 0.0) or 0.0), 4),
                "constraint_violation": round(_clamp01(flat.get("constraint_violation", 0.0), 0.0), 4),
            }
        )
    predicted_rows.sort(
        key=lambda item: (
            -float(item.get("confidence", 0.0) or 0.0),
            -float(item.get("expected_information_gain", 0.0) or 0.0),
            float(item.get("state_shift_risk", 0.0) or 0.0),
            str(item.get("function_name", "") or ""),
        )
    )
    predicted_rows = predicted_rows[: max(0, int(limit))]
    rollout_uncertainty = round(
        _clamp01(hidden_uncertainty * 0.52 + transition_entropy * 0.28 + world_novelty * 0.20, 0.0),
        4,
    )
    return {
        "current_phase": current_phase,
        "expected_next_phase": expected_next_phase or current_phase,
        "rollout_uncertainty": rollout_uncertainty,
        "predicted_transitions": predicted_rows,
        "transition_prior_signature": f"{current_phase}->{expected_next_phase or current_phase}:{len(predicted_rows)}",
    }

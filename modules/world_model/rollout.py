from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from modules.world_model.latent_dynamics import summarize_latent_dynamics


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
    if not phase:
        return "exploring"
    aliases = {
        "stable": "stabilizing",
        "commit": "committed",
        "completed": "committed",
        "failed": "disrupted",
    }
    return aliases.get(phase, phase)


def _match_targets(
    function_name: str,
    candidate_intervention_targets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fn = str(function_name or "").strip()
    rows: List[Dict[str, Any]] = []
    for row in candidate_intervention_targets:
        if not isinstance(row, dict):
            continue
        candidate_actions = [str(item or "") for item in _as_list(row.get("candidate_actions", [])) if str(item or "")]
        if fn in candidate_actions:
            rows.append(dict(row))
    return rows[:3]


def _match_mechanisms(
    function_name: str,
    mechanism_hypotheses: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fn = str(function_name or "").strip()
    rows: List[Dict[str, Any]] = []
    for row in mechanism_hypotheses:
        if not isinstance(row, dict):
            continue
        discriminating_actions = [str(item or "") for item in _as_list(row.get("best_discriminating_actions", [])) if str(item or "")]
        if fn in discriminating_actions:
            rows.append(dict(row))
    return rows[:3]


def simulate_function_rollout(
    function_name: str,
    *,
    world_model_summary: Dict[str, Any],
    transition_priors: Dict[str, Any],
    candidate_intervention_targets: Optional[Sequence[Dict[str, Any]]] = None,
    mechanism_hypotheses: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    summary = dict(world_model_summary or {})
    dynamics = summarize_latent_dynamics(summary, transition_priors)
    current_phase = str(dynamics.get("current_phase", "exploring") or "exploring")
    transitions = [dict(row) for row in _as_list(dynamics.get("predicted_transitions", [])) if isinstance(row, dict)]
    matched = next(
        (row for row in transitions if str(row.get("function_name", "") or "") == str(function_name or "")),
        {},
    )
    matched_targets = _match_targets(
        function_name,
        candidate_intervention_targets
        if isinstance(candidate_intervention_targets, list)
        else _as_list(summary.get("candidate_intervention_targets", [])),
    )
    matched_mechanisms = _match_mechanisms(
        function_name,
        mechanism_hypotheses
        if isinstance(mechanism_hypotheses, list)
        else _as_list(summary.get("mechanism_hypotheses", [])),
    )
    expected_info_gain = float(matched.get("expected_information_gain", 0.0) or 0.0)
    if matched_targets:
        expected_info_gain = max(
            expected_info_gain,
            max(
                float((_as_dict(row.get("priority_features", {})).get("expected_information_gain", 0.0)) or 0.0)
                for row in matched_targets
            ),
        )
    if matched_mechanisms:
        expected_info_gain = max(
            expected_info_gain,
            max(float(row.get("expected_information_gain", 0.0) or 0.0) for row in matched_mechanisms),
        )
    state_shift_risk = _clamp01(
        matched.get("state_shift_risk", summary.get("shift_risk", 0.0)),
        summary.get("shift_risk", 0.0),
    )
    if matched_targets:
        state_shift_risk = _clamp01(state_shift_risk * 0.92, state_shift_risk)
    if matched_mechanisms:
        state_shift_risk = _clamp01(state_shift_risk * 0.88, state_shift_risk)
    target_phase = _phase_alias(matched.get("to_phase", dynamics.get("expected_next_phase", current_phase)))
    confidence = _clamp01(matched.get("confidence", 0.0), 0.0)
    confidence = _clamp01(confidence + (0.08 if matched_targets else 0.0) + (0.10 if matched_mechanisms else 0.0), confidence)
    expected_reward = float(matched.get("expected_reward", 0.0) or 0.0)
    if matched_mechanisms:
        expected_reward += max(float(row.get("confidence", 0.0) or 0.0) for row in matched_mechanisms) * 0.12
    phase_path = [current_phase, target_phase]
    return {
        "function_name": str(function_name or ""),
        "from_phase": current_phase,
        "to_phase": target_phase,
        "phase_path": phase_path,
        "confidence": round(confidence, 4),
        "state_shift_risk": round(state_shift_risk, 4),
        "expected_information_gain": round(_clamp01(expected_info_gain, 0.0), 4),
        "expected_reward": round(expected_reward, 4),
        "reversibility": round(_clamp01(matched.get("reversibility", 0.35), 0.35), 4),
        "rollout_uncertainty": round(float(dynamics.get("rollout_uncertainty", 0.5) or 0.5), 4),
        "matched_targets": matched_targets,
        "matched_mechanisms": matched_mechanisms,
    }


def compare_function_rollouts(
    function_a: str,
    function_b: str,
    *,
    world_model_summary: Dict[str, Any],
    transition_priors: Dict[str, Any],
    candidate_intervention_targets: Optional[Sequence[Dict[str, Any]]] = None,
    mechanism_hypotheses: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rollout_a = simulate_function_rollout(
        function_a,
        world_model_summary=world_model_summary,
        transition_priors=transition_priors,
        candidate_intervention_targets=candidate_intervention_targets,
        mechanism_hypotheses=mechanism_hypotheses,
    )
    rollout_b = simulate_function_rollout(
        function_b,
        world_model_summary=world_model_summary,
        transition_priors=transition_priors,
        candidate_intervention_targets=candidate_intervention_targets,
        mechanism_hypotheses=mechanism_hypotheses,
    )
    score_a = (
        float(rollout_a.get("expected_reward", 0.0) or 0.0)
        + float(rollout_a.get("expected_information_gain", 0.0) or 0.0) * 0.28
        + (0.16 if str(rollout_a.get("to_phase", "")) == "committed" else 0.0)
        - float(rollout_a.get("state_shift_risk", 0.0) or 0.0) * 0.45
    )
    score_b = (
        float(rollout_b.get("expected_reward", 0.0) or 0.0)
        + float(rollout_b.get("expected_information_gain", 0.0) or 0.0) * 0.28
        + (0.16 if str(rollout_b.get("to_phase", "")) == "committed" else 0.0)
        - float(rollout_b.get("state_shift_risk", 0.0) or 0.0) * 0.45
    )
    preferred_action = function_a if score_a >= score_b else function_b
    return {
        "preferred_action": preferred_action,
        "estimated_delta": round(score_a - score_b, 4),
        "action_a": rollout_a,
        "action_b": rollout_b,
    }


def build_rollout_support(
    *,
    world_model_summary: Dict[str, Any],
    transition_priors: Dict[str, Any],
    candidate_intervention_targets: Optional[Sequence[Dict[str, Any]]] = None,
    mechanism_hypotheses: Optional[Sequence[Dict[str, Any]]] = None,
    limit: int = 6,
) -> Dict[str, Any]:
    dynamics = summarize_latent_dynamics(world_model_summary, transition_priors, limit=limit)
    predicted_transitions = [dict(row) for row in _as_list(dynamics.get("predicted_transitions", [])) if isinstance(row, dict)]
    contrasts: List[Dict[str, Any]] = []
    if len(predicted_transitions) >= 2:
        best = predicted_transitions[0]
        worst = sorted(
            predicted_transitions,
            key=lambda item: (
                float(item.get("state_shift_risk", 0.0) or 0.0),
                -float(item.get("expected_reward", 0.0) or 0.0),
            ),
            reverse=True,
        )[0]
        if str(best.get("function_name", "") or "") != str(worst.get("function_name", "") or ""):
            contrasts.append(
                compare_function_rollouts(
                    str(best.get("function_name", "") or ""),
                    str(worst.get("function_name", "") or ""),
                    world_model_summary=world_model_summary,
                    transition_priors=transition_priors,
                    candidate_intervention_targets=candidate_intervention_targets,
                    mechanism_hypotheses=mechanism_hypotheses,
                )
            )
    return {
        "predicted_transitions": predicted_transitions,
        "counterfactual_contrasts": contrasts,
        "rollout_uncertainty": float(dynamics.get("rollout_uncertainty", 0.5) or 0.5),
        "transition_prior_signature": str(dynamics.get("transition_prior_signature", "") or ""),
    }

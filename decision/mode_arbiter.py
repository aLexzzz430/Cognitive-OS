from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass
class ModeDecision:
    control_mode: str
    require_discriminating_action: bool
    unresolved_dimensions: List[str]
    diagnostics: Dict[str, Any]


def infer_mode(
    *,
    ranked_mechanisms: Sequence[Mapping[str, Any]],
    target_ranked: Sequence[Mapping[str, Any]],
    active_commitment: Optional[Mapping[str, Any]],
    obs_before: Optional[Mapping[str, Any]],
    entropy: float,
    margin: float,
    stagnant_ticks: int,
    low_confidence_threshold: float,
    uncertainty_margin_threshold: float,
) -> ModeDecision:
    top = dict(ranked_mechanisms[0]) if ranked_mechanisms else {}
    top_target = dict(target_ranked[0]) if target_ranked else {}
    runner_up_target = dict(target_ranked[1]) if len(target_ranked) > 1 else {}
    obs = _as_dict(obs_before)
    active = _as_dict(active_commitment)
    active_trust = _float(active.get("trust", 0.0), 0.0)
    target_trust = _float(top_target.get("trust", 0.0), 0.0)
    target_margin = target_trust - _float(runner_up_target.get("trust", 0.0), 0.0)
    counterevidence_tokens = set(str(item or "").strip().lower() for item in list(obs.get("counterevidence_token", "").split()) if str(item or "").strip())
    active_tokens = set(str(item or "").strip().lower() for item in list(active.get("binding_tokens", []) or []) if str(item or "").strip())
    revoked = bool(active.get("revoked", False) or (counterevidence_tokens and active_tokens and active_tokens.isdisjoint(counterevidence_tokens)))
    contradiction_pressure = _float(active.get("contradiction_pressure", 0.0), 0.0) + (0.45 if revoked else 0.0)

    prerequisite_missing = bool(obs.get("prerequisite_missing", False)) and not bool(obs.get("has_prerequisite", False))
    recovery_required = bool(obs.get("recovery_required", False))
    delayed_pending = int(obs.get("pending_countdown", 0) or 0) > 0 or bool(obs.get("delayed_resolution_pending", False))
    evidence_specificity = max(0.0, min(1.0, target_margin * 1.8 + target_trust * 0.55 - contradiction_pressure * 0.5))
    mechanism_ready = bool(
        ranked_mechanisms
        and _float(top.get("posterior", 0.0), 0.0) >= low_confidence_threshold
        and margin >= uncertainty_margin_threshold
        and entropy < 0.48
    )
    commitment_viable = bool(active and not revoked and active_trust >= 0.60 and evidence_specificity >= 0.26)

    if recovery_required:
        control_mode = "recover"
    elif prerequisite_missing:
        control_mode = "prepare"
    elif delayed_pending and commitment_viable:
        control_mode = "wait"
    elif commitment_viable and mechanism_ready:
        control_mode = "exploit"
    else:
        control_mode = "discriminate"

    unresolved: List[str] = []
    if not ranked_mechanisms:
        unresolved.append("no_mechanism_hypotheses")
    if _float(top.get("posterior", 0.0), 0.0) < low_confidence_threshold:
        unresolved.append("dominant_mechanism_confidence_low")
    if margin < uncertainty_margin_threshold:
        unresolved.append("top_mechanism_margin_low")
    if entropy >= 0.48:
        unresolved.append("posterior_entropy_high")
    if target_margin < 0.12:
        unresolved.append("target_binding_not_specific")
    if contradiction_pressure >= 0.45:
        unresolved.append("contradiction_pressure_high")
    if stagnant_ticks >= 1:
        unresolved.append("posterior_stagnant")
    if revoked:
        unresolved.append("commitment_revoked")

    require_discriminating = bool(
        control_mode == "discriminate"
        or not mechanism_ready
        or evidence_specificity < 0.24
        or stagnant_ticks >= 1
    )
    return ModeDecision(
        control_mode=control_mode,
        require_discriminating_action=require_discriminating,
        unresolved_dimensions=unresolved,
        diagnostics={
            "target_margin": round(target_margin, 4),
            "target_trust": round(target_trust, 4),
            "evidence_specificity": round(evidence_specificity, 4),
            "contradiction_pressure": round(contradiction_pressure, 4),
            "commitment_viable": bool(commitment_viable),
            "mechanism_ready": bool(mechanism_ready),
            "revoked": bool(revoked),
        },
    )

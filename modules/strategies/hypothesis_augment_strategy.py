from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


@dataclass(frozen=True)
class HypothesisAugmentInputs:
    cooldown_ready: bool
    tick: int
    entropy: float
    reward_stagnation: bool
    signature_changed: bool
    pending_recovery_probe: bool
    pending_replan: bool
    world_model_required_probe_count: int = 0
    world_model_control_trust: float = 0.5
    world_model_transition_confidence: float = 0.5
    world_model_state_shift_risk: float = 0.0
    hidden_state_drift_score: float = 0.0
    hidden_state_uncertainty_score: float = 0.0
    latent_branch_instability: float = 0.0


class HypothesisAugmentStrategy:
    """策略模块：是否执行 hypothesis augmentation。"""

    @staticmethod
    def should_augment(inputs: HypothesisAugmentInputs) -> bool:
        if not inputs.cooldown_ready:
            return False
        if inputs.tick == 0:
            return True
        if inputs.entropy > 0.45 and inputs.reward_stagnation:
            return True
        if inputs.signature_changed:
            return True
        if inputs.pending_recovery_probe or inputs.pending_replan:
            return True
        probe_pressure = min(1.0, max(0, int(inputs.world_model_required_probe_count or 0)) / 3.0)
        control_trust = _clamp01(inputs.world_model_control_trust, 0.5)
        transition_confidence = _clamp01(inputs.world_model_transition_confidence, 0.5)
        state_shift_risk = _clamp01(inputs.world_model_state_shift_risk, 0.0)
        hidden_drift = _clamp01(inputs.hidden_state_drift_score, 0.0)
        hidden_uncertainty = _clamp01(inputs.hidden_state_uncertainty_score, 0.0)
        latent_instability = _clamp01(inputs.latent_branch_instability, 0.0)
        if (
            probe_pressure >= 0.34
            and (
                control_trust <= 0.52
                or transition_confidence <= 0.48
                or hidden_drift >= 0.55
                or hidden_uncertainty >= 0.62
                or latent_instability >= 0.58
                or state_shift_risk >= 0.58
            )
        ):
            return True
        return False

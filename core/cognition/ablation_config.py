"""Configuration types for cognition/causal-layer ablation experiments."""

from dataclasses import dataclass


@dataclass
class CausalLayerAblationConfig:
    """
    Unified switchboard for causal-layer ablations.

    Migration note:
    - Causal-layer experiments should be configured via this dataclass.
    - `arm_mode` remains for legacy retrieval-arm behavior only.
    """

    enable_unified_context: bool = True
    # Ablation mode used when `enable_unified_context` is False:
    # - 'stripped': keep minimal compatibility contract (legacy behavior).
    # - 'hard_off': disable unified payload entirely; callers should use legacy path.
    unified_context_ablation_mode: str = 'stripped'
    enable_high_level_self_model: bool = True
    # Experiment-level switchboard gate for all mechanism-matching entry points
    # (extraction, formal commit attempts, and counterfactual mechanism-candidate
    # injection). Runtime budget knobs may only degrade/limit work *after* this is on.
    enable_mechanism_matching: bool = True
    enable_representation_adaptation: bool = True
    freeze_retrieval_pressure: bool = False

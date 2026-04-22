#!/usr/bin/env python3
"""
representations/extractor.py

Computes activation scores from a card's activation_logic and matcher scores.

NO consumer influence. NO planner ranking. Pure activation computation.

Public API:
    ActivationExtractor.compute_activation(card, obs, match_score, match_reason)
      -> (activation_score: float, activation_reason: str)

    ActivationExtractor.batch_compute(cards, obs, match_results)
      -> {rep_id: (score, reason)}
"""

from __future__ import annotations

from typing import Optional

from .schema import RepresentationCard


# ============================================================
# Activation computation
# ============================================================

class ActivationExtractor:
    """
    Computes how strongly a card should be activated given:
      1. The card's activation_logic (function + parameters)
      2. The matcher score (structural signature match quality)

    activation_score is NOT a planner influence score.
    It's a pure measure of "this card's situation/pattern is currently present."

    The B1 contract: consumer is ADVISORY only.
    This extractor does not make any decisions about planner influence.
    """

    def __init__(self, base_threshold: float = 0.5):
        """
        Args:
            base_threshold: minimum match_score to consider activating.
                           Cards with match_score < threshold get 0.0 activation.
        """
        self.base_threshold = base_threshold

    def compute_activation(
        self,
        card: RepresentationCard,
        obs: dict,
        match_score: float,
        match_reason: str,
    ) -> tuple[float, str]:
        """
        Returns (activation_score: float, activation_reason: str).

        activation_score: 0.0 (not activated) to 1.0 (strongly activated)
        activation_reason: human-readable description of why it activated
        """
        # Gate: card must have minimum structural match
        if match_score < self.base_threshold:
            return 0.0, f"match_score {match_score:.2f} < threshold {self.base_threshold}"

        func = card.activation_logic.function
        params = card.activation_logic.parameters or {}

        if func == "threshold":
            score, reason = self._threshold_activation(card, obs, params, match_score, match_reason)
        elif func == "trend":
            score, reason = self._trend_activation(card, obs, params, match_score, match_reason)
        elif func == "composite":
            score, reason = self._composite_activation(card, obs, params, match_score, match_reason)
        else:
            score, reason = 0.0, f"unknown activation function: {func}"

        return score, reason

    def _threshold_activation(
        self,
        card: RepresentationCard,
        obs: dict,
        params: dict,
        match_score: float,
        match_reason: str,
    ) -> tuple[float, str]:
        """
        Threshold activation: card activates when match_score exceeds threshold
        and a specific observation value crosses a boundary.
        """
        signal = obs.get("signal")
        threshold = params.get("signal_threshold", 0.5)

        if signal is not None:
            if params.get("signal_max") is not None:
                if signal <= params["signal_max"]:
                    activation = match_score * 1.0
                    return activation, f"threshold signal={signal:.2f} <= {params['signal_max']} (match={match_score:.2f})"
                else:
                    return 0.0, f"threshold signal={signal:.2f} > {params['signal_max']}"

            elif params.get("signal_min") is not None:
                if signal >= params["signal_min"]:
                    activation = match_score * 1.0
                    return activation, f"threshold signal={signal:.2f} >= {params['signal_min']} (match={match_score:.2f})"
                else:
                    return 0.0, f"threshold signal={signal:.2f} < {params['signal_min']}"

        # Generic threshold: just use match_score if observation conditions met
        return match_score, f"threshold activation (match={match_score:.2f})"

    def _trend_activation(
        self,
        card: RepresentationCard,
        obs: dict,
        params: dict,
        match_score: float,
        match_reason: str,
    ) -> tuple[float, str]:
        """
        Trend activation: card activates when a variable is trending in a
        specific direction with sufficient history.
        """
        history_window = params.get("history_window", 5)

        # Try to get the history key
        history_key = params.get("history_key", "signal_history")
        history = obs.get(history_key, [])

        if not history or len(history) < history_window:
            return 0.0, f"trend: insufficient history ({len(history) if history else 0}/{history_window})"

        recent = list(history[-history_window:])

        # Compute trend slope
        n = len(recent)
        indices = list(range(n))
        mean_x = sum(indices) / n
        mean_y = sum(recent) / n
        cov = sum((i - mean_x) * (v - mean_y) for i, v in zip(indices, recent))
        var = sum((i - mean_x) ** 2 for i in indices)

        if var == 0:
            slope = 0.0
        else:
            slope = cov / var

        trend_min = params.get("trend_min", 0.05)
        trend_max = params.get("trend_max", float("inf"))

        # Check if slope is in the expected direction
        if params.get("trend_direction", "up") == "up":
            if slope >= trend_min:
                boost = min(slope / trend_min, 2.0)  # up to 2x boost
                activation = match_score * boost
                cap = params.get("activation_cap", 1.0)
                activation = min(activation, cap)
                return activation, f"trend UP slope={slope:.4f} >= {trend_min} (match={match_score:.2f})"
            else:
                return 0.0, f"trend UP slope={slope:.4f} < {trend_min}"
        else:
            if slope <= -trend_min:
                boost = min(abs(slope) / trend_min, 2.0)
                activation = match_score * boost
                cap = params.get("activation_cap", 1.0)
                activation = min(activation, cap)
                return activation, f"trend DOWN slope={slope:.4f} <= -{trend_min} (match={match_score:.2f})"
            else:
                return 0.0, f"trend DOWN slope={slope:.4f} > -{trend_min}"

    def _composite_activation(
        self,
        card: RepresentationCard,
        obs: dict,
        params: dict,
        match_score: float,
        match_reason: str,
    ) -> tuple[float, str]:
        """
        Composite activation: card activates when multiple conditions are met.
        All sub-conditions must pass for activation.
        """
        subconditions = []

        # Check latent state condition
        if params.get("require_latent"):
            latent = obs.get("latent_state")
            if latent != params["require_latent"] and latent != f"latent-{params['require_latent']}":
                return 0.0, f"composite: latent_state={latent} != {params['require_latent']} [failed]"

        if params.get("latent_eq"):
            latent = obs.get("latent_state")
            if latent != params["latent_eq"] and latent != f"latent-{params['latent_eq']}":
                return 0.0, f"composite: latent_state={latent} != {params['latent_eq']} [failed]"
            subconditions.append(f"latent={latent}")

        # Check signal conditions
        if params.get("signal_max") is not None:
            signal = obs.get("signal")
            if signal is None or signal > params["signal_max"]:
                return 0.0, f"composite: signal={signal} > {params['signal_max']} [failed]"
            subconditions.append(f"signal={signal:.2f}<={params['signal_max']}")

        if params.get("signal_min") is not None:
            signal = obs.get("signal")
            if signal is None or signal < params["signal_min"]:
                return 0.0, f"composite: signal={signal} < {params['signal_min']} [failed]"
            subconditions.append(f"signal={signal:.2f}>={params['signal_min']}")

        # Check precursor_min
        if params.get("precursor_min") is not None:
            precursor = obs.get("precursor_remaining", 0)
            if precursor < params["precursor_min"]:
                return 0.0, f"composite: precursor_remaining={precursor} < {params['precursor_min']} [failed]"
            subconditions.append(f"precursor>={params['precursor_min']}")

        # Check energy_min
        if params.get("energy_min") is not None:
            energy = obs.get("energy", 0)
            if energy < params["energy_min"]:
                return 0.0, f"composite: energy={energy} < {params['energy_min']} [failed]"
            subconditions.append(f"energy>={params['energy_min']}")

        # Check drop_threshold (for recoverable-setback)
        if params.get("drop_threshold") is not None:
            progress_history = obs.get("progress_history", [])
            if not progress_history or len(progress_history) < 2:
                return 0.0, "composite: no progress_history for drop check [failed]"
            peak = max(progress_history)
            recent = progress_history[-1]
            drop = peak - recent
            if drop < params["drop_threshold"]:
                return 0.0, f"composite: progress_drop={drop:.1f} < {params['drop_threshold']} [failed]"
            subconditions.append(f"drop>={params['drop_threshold']}")

        # Check spike_threshold (for hazard-pulse)
        # For pulse: check if the PEAK in recent history exceeded threshold
        if params.get("spike_threshold") is not None:
            signal_history = obs.get("signal_history", [])
            if signal_history:
                recent_peaks = signal_history[-int(params.get("peak_within_steps", 3)):]
                peak_val = max(recent_peaks) if recent_peaks else 0
                if peak_val < params["spike_threshold"]:
                    return 0.0, f"composite: recent peak={peak_val:.2f} < spike_threshold {params['spike_threshold']} [failed]"
                subconditions.append(f"peak={peak_val:.2f}>={params['spike_threshold']}")

        # Check baseline_window for regime-shift
        if params.get("baseline_window"):
            history_key = params.get("history_key", "signal_history")
            history = obs.get(history_key, [])
            if not history or len(history) < params["baseline_window"]:
                return 0.0, f"composite: history len {len(history) if history else 0} < {params['baseline_window']} [insufficient]"
            subconditions.append(f"history>={params['baseline_window']}")

        # Check hazard_delta_min (for progress-stall-v1)
        if params.get("hazard_delta_min") is not None:
            history = obs.get("hazard_exposure_history", [])
            if len(history) < 2:
                return 0.0, "composite: no hazard_exposure_history for delta check [failed]"
            delta = history[-1] - history[0]
            if delta < params["hazard_delta_min"]:
                return 0.0, f"composite: hazard_delta={delta:.1f} < {params['hazard_delta_min']} [failed]"
            subconditions.append(f"hazard_delta={delta:.1f}>={params['hazard_delta_min']}")

        # Check progress_delta_max (for progress-stall-v1)
        if params.get("progress_delta_max") is not None:
            history = obs.get("progress_history", [])
            if len(history) < 2:
                return 0.0, "composite: no progress_history for delta check [failed]"
            delta = history[-1] - history[0]
            if delta > params["progress_delta_max"]:
                return 0.0, f"composite: progress_delta={delta:.1f} > {params['progress_delta_max']} [failed]"
            subconditions.append(f"progress_delta={delta:.1f}<={params['progress_delta_max']}")

        # Check rising_velocity_min (for rising-trend cards like hazard-buildup-v1)
        if params.get("rising_velocity_min") is not None:
            history = obs.get("signal_history", [])
            if len(history) < 2:
                return 0.0, "composite: no signal_history for rising check [failed]"
            velocity = history[-1] - history[-2]
            if velocity < params["rising_velocity_min"]:
                return 0.0, f"composite: rising_velocity={velocity:.4f} < {params['rising_velocity_min']} [failed]"
            subconditions.append(f"rising_vel={velocity:.4f}>={params['rising_velocity_min']}")

        # All conditions passed
        # Activation = match_score × number_of_subconditions_bonus
        conditions_bonus = len(subconditions) * 0.1
        activation = min(match_score + conditions_bonus, 1.0)

        return activation, f"composite [{len(subconditions)} conditions]: {', '.join(subconditions[:3])}"

    def batch_compute(
        self,
        cards: list[RepresentationCard],
        obs: dict,
        match_results: dict[str, tuple[float, str]],
    ) -> dict[str, tuple[float, str]]:
        """
        Compute activation for multiple cards.
        Returns {rep_id: (activation_score, reason)}.
        """
        results = {}
        for card in cards:
            match_score, match_reason = match_results.get(card.rep_id, (0.0, "no match"))
            act_score, act_reason = self.compute_activation(card, obs, match_score, match_reason)
            results[card.rep_id] = (act_score, act_reason)
        return results


# ============================================================
# Module-level singleton
# ============================================================

_extractor: Optional[ActivationExtractor] = None

def get_extractor() -> ActivationExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ActivationExtractor()
    return _extractor

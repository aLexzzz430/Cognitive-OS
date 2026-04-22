#!/usr/bin/env python3
"""
representations/matcher.py

Scores how well a card's structural_signature matches the current observation.

NO consumer influence. NO planner ranking. Pure signature-to-observation matching.

Public API:
    CardMatcher.match(card, obs, regime, planner_style)
      -> (match_score: float, reason: str)

    CardMatcher.batch_match(cards, obs, regime, planner_style)
      -> {rep_id: (score, reason)}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import RepresentationCard, ThresholdCondition


# ============================================================
# Operator registry
# ============================================================

def op_eq(a, b) -> bool:
    return a == b

def op_neq(a, b) -> bool:
    return a != b

def op_lt(a, b) -> bool:
    return a < b

def op_lte(a, b) -> bool:
    return a <= b

def op_gt(a, b) -> bool:
    return a > b

def op_gte(a, b) -> bool:
    return a >= b


OPERATORS = {
    "eq": op_eq,
    "neq": op_neq,
    "lt": op_lt,
    "lte": op_lte,
    "gt": op_gt,
    "gte": op_gte,
    # Structural operators (filled after function definitions below)
}


# ============================================================
# Structural operators
# ============================================================

def _get_nested(obs: dict, key: str):
    """Get value from nested observation dict, e.g. 'signal' or 'signal_history'."""
    if "." in key:
        parts = key.split(".")
        val = obs
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return None
        return val
    return obs.get(key)


def _trend_op(signal_history: list, min_slope: float) -> bool:
    """Check if signal_history shows upward trend >= min_slope."""
    if not signal_history or len(signal_history) < 3:
        return False
    recent = signal_history[-5:]
    if len(recent) < 2:
        return False
    # Simple linear slope
    n = len(recent)
    indices = list(range(n))
    mean_x = sum(indices) / n
    mean_y = sum(recent) / n
    cov = sum((i - mean_x) * (v - mean_y) for i, v in zip(indices, recent))
    var = sum((i - mean_x) ** 2 for i in indices)
    if var == 0:
        return False
    slope = cov / var
    return slope >= min_slope


def _rising_trend_op(signal_history: list, min_velocity: float) -> bool:
    """
    Check if signal is rising with meaningful velocity.
    A signal RISING means:
      1. Overall direction: recent half mean > older half mean
      2. Current velocity: most recent delta > min_velocity
    This avoids false positives from noise-driven second-derivative bouncing.
    """
    if not signal_history or len(signal_history) < 4:
        return False
    n = len(signal_history)
    half = n // 2
    older = signal_history[:half]
    recent = signal_history[half:]
    older_mean = sum(older) / len(older)
    recent_mean = sum(recent) / len(recent)

    if recent_mean <= older_mean:
        return False  # Not rising overall

    # Velocity: most recent step
    velocity = signal_history[-1] - signal_history[-2]
    return velocity >= min_velocity


def _drain_rate_op(energy_history: list, min_excess: float) -> bool:
    """Check if energy drain rate exceeds base rate by min_excess."""
    if not energy_history or len(energy_history) < 5:
        return False
    recent = list(energy_history[-5:])
    # Drain rate: negative delta per step
    deltas = [recent[i] - recent[i+1] for i in range(len(recent)-1)]
    if not deltas:
        return False
    avg_drain = sum(deltas) / len(deltas)
    # Base drain is ~0.2/tick (from ProductiveSafetyEnv), check if excess
    base_drain = 0.2
    return (avg_drain - base_drain) >= min_excess


def _stable_op(signal_history: list, max_std: float) -> bool:
    """Check if signal_history has std < max_std."""
    if not signal_history or len(signal_history) < 3:
        return False
    import statistics
    return statistics.stdev(signal_history) < max_std


def _volatility_op(signal_history: list, min_std: float) -> bool:
    """Check if signal_history has std > min_std."""
    if not signal_history or len(signal_history) < 3:
        return False
    import statistics
    return statistics.stdev(signal_history) > min_std


def _peaked_op(signal_history: list, steps_within: int) -> bool:
    """Check if signal peaked within steps_within and is now declining."""
    if not signal_history or len(signal_history) < 3:
        return False
    recent = signal_history[-steps_within:]
    if len(recent) < 3:
        return False
    # Peak at position (len-1 is most recent)
    peak_idx = recent.index(max(recent))
    # Peak should be at least 1 step ago (not current)
    if peak_idx == len(recent) - 1:
        return False  # peak is current step, not receding
    # Last value should be lower than peak
    return recent[-1] < recent[peak_idx]


def _regime_change_op(signal_history: list, threshold: float) -> bool:
    """Check if signal mean shifted significantly from baseline."""
    if not signal_history or len(signal_history) < 20:
        return False
    baseline = signal_history[:20]
    recent = signal_history[-10:]
    import statistics
    mean_baseline = statistics.mean(baseline)
    mean_recent = statistics.mean(recent)
    return abs(mean_recent - mean_baseline) >= threshold


def _delta_lte_op(raw, max_delta: float) -> bool:
    """
    Check if delta of a history list <= max_delta.
    raw: either a list (the history directly) or a dict with history key.
    """
    # raw is the observation value for the key this operator is registered under
    # For delta_lte on "progress", raw = obs["progress"] = int
    # But we need the history. The history key is always <key>_history.
    # We get raw as an int (current value), so we need to look up history ourselves.
    # Since we don't have access to the full obs here, return False
    # (delta operators need special handling in evaluate_condition)
    return False  # handled specially in evaluate_condition


def _delta_gte_op(raw, min_delta: float) -> bool:
    """Check if delta of key >= min_delta. See delta_lte_op notes."""
    return False  # handled specially in evaluate_condition


def _recent_drop_op(progress_history: list, min_drop: float) -> bool:
    """Check if progress dropped >= min_drop from recent peak."""
    if not progress_history or len(progress_history) < 3:
        return False
    peak = max(progress_history)
    recent = progress_history[-1]
    return (peak - recent) >= min_drop


def _repeating_sequence_op(action_history: list, min_repeat: int) -> bool:
    """Check if same action repeated min_repeat times."""
    if not action_history or len(action_history) < min_repeat:
        return False
    recent = action_history[-min_repeat:]
    if all(a == recent[0] for a in recent):
        return True
    # Check for 2-action oscillation
    if min_repeat >= 4 and len(action_history) >= 4:
        last4 = action_history[-4:]
        if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
            return True
    return False


# ============================================================
# Structural operators (filled after definitions)
# ============================================================

OPERATORS.update({
    "trend": _trend_op,
    "rising_trend": _rising_trend_op,
    "drain_rate": _drain_rate_op,
    "stable": _stable_op,
    "volatility": _volatility_op,
    "peaked": _peaked_op,
    "regime_change": _regime_change_op,
    # delta_lte / delta_gte are handled specially in _evaluate_condition
    # (they need access to full obs dict to look up <key>_history)
    "recent_drop": _recent_drop_op,
    "repeating_sequence": _repeating_sequence_op,
})


# ============================================================
# Match scoring
# ============================================================

@dataclass
class MatchResult:
    """Result of matching a card against an observation."""
    match_score: float       # 0.0–1.0
    reason: str              # human-readable explanation
    conditions_met: int
    conditions_total: int
    matched_conditions: list[str]


class CardMatcher:
    """
    Scores how well a card's structural_signature matches the current observation.

    Scope filtering is done OUTSIDE this class (in store.get_relevant_card_ids).
    This class only scores structural_signature match quality.
    """

    def __init__(self):
        pass

    def match(
        self,
        card: RepresentationCard,
        obs: dict,
        regime: str = "nominal",
        planner_style: str = "any",
    ) -> tuple[float, str]:
        """
        Returns (match_score: float, reason: str).

        match_score: 0.0 (no match) to 1.0 (perfect structural match)
        reason: human-readable description of what matched
        """
        sig = card.structural_signature

        if not sig.threshold_conditions:
            # No conditions = wildcard match
            return 0.5, "no structural conditions defined (wildcard)"

        matched = []
        unmet = []

        for tc in sig.threshold_conditions:
            ok, reason = self._evaluate_condition(tc, obs)
            if ok:
                matched.append(reason)
            else:
                unmet.append(reason)

        total = len(sig.threshold_conditions)
        score = len(matched) / total if total > 0 else 0.0

        # Build reason string
        if matched and unmet:
            reason_str = f"matched {len(matched)}/{total}: {', '.join(matched[:2])}"
        elif matched:
            reason_str = f"all {total} conditions met: {', '.join(matched[:3])}"
        else:
            reason_str = f"no conditions met: {', '.join(unmet[:2])}"

        return score, reason_str

    def _evaluate_condition(
        self,
        tc: ThresholdCondition,
        obs: dict,
    ) -> tuple[bool, str]:
        """
        Evaluate a single ThresholdCondition against the observation.
        Returns (passed: bool, description: str).
        """
        key = tc.observation_key
        value = tc.value
        op_name = tc.operator

        raw = _get_nested(obs, key)

        # Build description
        desc = f"{key} {op_name} {value} (got {raw})"

        # ---- Special operators needing full obs access ----
        if op_name == "delta_lte":
            # Look up <key>_history from obs (not just raw value)
            history = obs.get(f"{key}_history", [])
            if not isinstance(history, list) or len(history) < 2:
                if isinstance(obs.get(key), list):
                    history = obs.get(key)
                else:
                    return False, f"delta({key}): no history found"
            delta = history[-1] - history[0]
            ok = delta <= value
            return ok, f"delta({key})={delta:.1f} <= {value}: {ok}"

        elif op_name == "delta_gte":
            history = obs.get(f"{key}_history", [])
            if not isinstance(history, list) or len(history) < 2:
                if isinstance(obs.get(key), list):
                    history = obs.get(key)
                else:
                    return False, f"delta({key}): no history found"
            delta = history[-1] - history[0]
            ok = delta >= value
            return ok, f"delta({key})={delta:.1f} >= {value}: {ok}"

        # ---- Non-callable structural operators ----
        if op_name in OPERATORS and not callable(OPERATORS[op_name]):
            if op_name == "trend":
                if raw is None:
                    return False, desc
                ok = _trend_op(raw, value)
                return ok, f"trend({key}) >= {value}: {ok}"
            elif op_name == "rising_trend":
                if raw is None:
                    return False, desc
                ok = _rising_trend_op(raw, value)
                return ok, f"rising_trend({key}) >= {value}: {ok}"
            elif op_name == "drain_rate":
                if raw is None:
                    return False, desc
                ok = _drain_rate_op(raw, value)
                return ok, f"drain_rate({key}) > base+{value}: {ok}"
            elif op_name == "stable":
                if raw is None:
                    return False, desc
                ok = _stable_op(raw, value)
                return ok, f"stable({key}) std < {value}: {ok}"
            elif op_name == "volatility":
                if raw is None:
                    return False, desc
                ok = _volatility_op(raw, value)
                return ok, f"volatility({key}) std > {value}: {ok}"
            elif op_name == "peaked":
                if raw is None:
                    return False, desc
                ok = _peaked_op(raw, int(value))
                return ok, f"peaked({key}) within {value} steps: {ok}"
            elif op_name == "regime_change":
                if raw is None:
                    return False, desc
                ok = _regime_change_op(raw, value)
                return ok, f"regime_change({key}) shift >= {value}: {ok}"
            elif op_name == "recent_drop":
                ok = _recent_drop_op(raw or [], value)
                return ok, f"recent_drop({key}) >= {value}: {ok}"
            elif op_name == "repeating_sequence":
                ok = _repeating_sequence_op(raw or [], int(value))
                return ok, f"repeating_seq({key}) >= {value}: {ok}"
            else:
                return False, f"unknown non-callable operator: {op_name}"

        # ---- Simple callable operators ----
        if raw is None:
            return False, f"{desc} [key not in observation]"

        if op_name not in OPERATORS:
            return False, f"{desc} [unknown operator: {op_name}]"

        try:
            ok = OPERATORS[op_name](raw, value)
            return ok, desc
        except (TypeError, ValueError):
            return False, f"{desc} [type error comparing {type(raw).__name__} to {type(value).__name__}]"

    def batch_match(
        self,
        cards: list[RepresentationCard],
        obs: dict,
        regime: str = "nominal",
        planner_style: str = "any",
    ) -> dict[str, tuple[float, str]]:
        """Match multiple cards, return dict of rep_id -> (score, reason)."""
        return {
            card.rep_id: self.match(card, obs, regime, planner_style)
            for card in cards
        }


# ============================================================
# Module-level singleton
# ============================================================

_matcher: Optional[CardMatcher] = None

def get_matcher() -> CardMatcher:
    global _matcher
    if _matcher is None:
        _matcher = CardMatcher()
    return _matcher

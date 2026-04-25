from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.reasoning.posterior_update import update_hypothesis_posteriors


def _action() -> dict:
    return {
        "kind": "call_tool",
        "payload": {
            "tool_args": {
                "function_name": "probe",
                "kwargs": {
                    "target_family": "lever",
                    "anchor_ref": "A",
                },
            }
        },
    }


def _hypothesis(*, posterior: float, reward_sign: str = "positive") -> dict:
    return {
        "hypothesis_id": "h1",
        "posterior": posterior,
        "family": "lever_rule",
        "summary": "Probe reveals the green lever rule",
        "predicted_observation_tokens": ["green_signal"],
        "predicted_action_effects": {
            "probe": {
                "reward_sign": reward_sign,
                "valid_state_change": reward_sign == "positive",
                "phase_shift": "solved" if reward_sign == "positive" else "blocked",
                "predicted_observation_tokens": ["green_signal"],
                "target_family": "lever",
                "anchor_ref": "A",
                "predicted_information_gain": 1.0,
            }
        },
    }


def test_posterior_update_damps_single_strong_support_from_zero() -> None:
    out = update_hypothesis_posteriors(
        [_hypothesis(posterior=0.0)],
        action=_action(),
        result={"success": True, "state_changed": True, "solved": True, "belief_phase": "solved"},
        predicted_transition={},
        actual_transition={
            "valid_state_change": True,
            "next_phase": "solved",
            "observation_tokens": ["green_signal", "lever"],
        },
        reward=1.0,
        information_gain=1.0,
        verifier_teaching={"teaching_signal": "positive", "teaching_signal_score": 10.0},
    )
    row = out["updated_hypotheses"][0]
    event = out["posterior_events"][0]

    assert 0.0 < row["posterior"] <= 0.35
    assert event["delta"] <= 0.35
    assert event["metadata"]["raw_delta"] > event["delta"]
    assert row["metadata"]["last_posterior_revision_rate"] <= 0.35


def test_posterior_update_damps_single_strong_contradiction_from_one() -> None:
    out = update_hypothesis_posteriors(
        [_hypothesis(posterior=1.0)],
        action=_action(),
        result={"success": False, "state_changed": False, "failed": True, "belief_phase": "blocked"},
        predicted_transition={},
        actual_transition={
            "valid_state_change": False,
            "next_phase": "blocked",
            "observation_tokens": ["blocked", "counter_signal"],
        },
        reward=-1.0,
        information_gain=0.0,
        verifier_teaching={"teaching_signal": "negative", "teaching_signal_score": 10.0},
    )
    row = out["updated_hypotheses"][0]
    event = out["posterior_events"][0]

    assert 0.65 <= row["posterior"] < 1.0
    assert event["delta"] >= -0.35
    assert event["metadata"]["raw_delta"] < event["delta"]
    assert row["metadata"]["last_posterior_revision_rate"] <= 0.35

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.staged_execution_runtime import (
    ActionExecutionArtifacts,
    TraceArtifacts,
    _update_hypothesis_posteriors_after_action,
)
from core.reasoning.discriminating_experiment import build_discriminating_experiments


class CapturingStateManager:
    def __init__(self) -> None:
        self.updates: List[Dict[str, Any]] = []

    def update_state(self, patch: Dict[str, Any], *, reason: str = "", module: str = "") -> None:
        self.updates.append({"patch": dict(patch), "reason": reason, "module": module})


def _hypothesis(
    hypothesis_id: str,
    *,
    phase: str,
    reward_sign: str,
    state_changes: bool,
    tokens: List[str],
) -> Dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "summary": f"{hypothesis_id} expects {phase}",
        "posterior": 0.48,
        "confidence": 0.48,
        "predicted_observation_tokens": list(tokens),
        "predicted_action_effects": {
            "probe_color": {
                "reward_sign": reward_sign,
                "valid_state_change": state_changes,
                "predicted_phase_shift": phase,
                "predicted_observation_tokens": list(tokens),
                "predicted_information_gain": 0.72,
                "target_family": "color_panel",
                "anchor_ref": "cell_1",
            }
        },
        "metadata": {
            "target_binding_tokens": ["color_panel", "cell_1"],
        },
    }


def test_minimal_cognitive_loop_updates_posterior_and_unified_context() -> None:
    hypotheses = [
        _hypothesis(
            "h_red",
            phase="red_revealed",
            reward_sign="positive",
            state_changes=True,
            tokens=["red", "cell_1", "revealed"],
        ),
        _hypothesis(
            "h_blue",
            phase="blue_revealed",
            reward_sign="negative",
            state_changes=False,
            tokens=["blue", "cell_1", "blocked"],
        ),
    ]
    probe_action = {
        "kind": "probe",
        "payload": {
            "tool_args": {
                "function_name": "probe_color",
                "kwargs": {
                    "target_family": "color_panel",
                    "anchor_ref": "cell_1",
                },
            }
        },
        "_candidate_meta": {
            "role": "discriminate",
            "expected_information_gain": 0.72,
            "risk": 0.05,
            "grounded_binding_tokens": ["color_panel", "cell_1"],
            "intervention_target": {
                "target_kind": "color_panel",
                "anchor_ref": "cell_1",
            },
        },
        "estimated_cost": 0.2,
        "risk": 0.05,
    }
    experiments = build_discriminating_experiments(hypotheses, [probe_action], limit=1)

    assert experiments
    experiment = experiments[0]
    assert set(experiment["discriminates_between"]) == {"h_red", "h_blue"}
    assert experiment["candidate_action"]["payload"]["tool_args"]["function_name"] == "probe_color"

    unified_context = UnifiedCognitiveContext.from_parts(
        current_goal="identify the color rule",
        current_task="probe the color panel",
        active_beliefs_summary={
            "scene_state": {
                "phase": "unknown",
                "signal_tokens": [],
            }
        },
        competing_hypotheses=hypotheses,
        ranked_discriminating_experiments=experiments,
        posterior_summary={
            "summary_stage": "deliberation",
            "deliberation_snapshot": {"ranked_experiment_count": len(experiments)},
        },
    )
    state_mgr = CapturingStateManager()
    loop = SimpleNamespace(
        _active_tick_context_frame=SimpleNamespace(unified_context=unified_context),
        _episode_trace=[{}],
        _last_obs={"phase": "unknown"},
        _shared_store=None,
        _store=None,
        _state_mgr=state_mgr,
    )
    execution = ActionExecutionArtifacts(
        function_name="probe_color",
        hypotheses_before=hypotheses,
        result={
            "success": True,
            "state_changed": True,
            "belief_phase": "red_revealed",
            "observation_tokens": ["red", "cell_1", "revealed"],
        },
        reward=1.0,
    )
    trace = TraceArtifacts(
        information_gain=0.72,
        progress_markers=["probe_color", "red_revealed"],
        predicted_transition={
            "next_phase": "red_revealed",
            "predicted_observation_tokens": ["red", "cell_1", "revealed"],
        },
        actual_transition={
            "next_phase": "red_revealed",
            "valid_state_change": True,
            "observation_tokens": ["red", "cell_1", "revealed"],
        },
    )

    _update_hypothesis_posteriors_after_action(
        loop,
        action_to_use=experiment["candidate_action"],
        execution=execution,
        trace=trace,
    )

    updated_by_id = {
        str(item["hypothesis_id"]): item
        for item in unified_context.competing_hypotheses
    }
    assert updated_by_id["h_red"]["posterior"] > updated_by_id["h_blue"]["posterior"]
    assert updated_by_id["h_red"]["support_count"] >= 1
    assert updated_by_id["h_blue"]["contradiction_count"] >= 1

    assert unified_context.ranked_discriminating_experiments == experiments
    assert unified_context.posterior_summary["summary_stage"] == "post_execution"
    assert unified_context.posterior_summary["last_update_source"] == "execution"
    assert unified_context.posterior_summary["leading_hypothesis_id"] == "h_red"
    assert unified_context.posterior_summary["execution_snapshot"]["action_function_name"] == "probe_color"
    assert unified_context.active_hypotheses_summary

    assert loop._episode_trace[-1]["posterior_summary"]["leading_hypothesis_id"] == "h_red"
    assert loop._episode_trace[-1]["hypothesis_posterior_events"]

    assert state_mgr.updates
    state_update = state_mgr.updates[-1]
    patch = state_update["patch"]
    assert state_update["reason"] == "reasoning:posterior_update"
    assert state_update["module"] == "core.reasoning"
    assert patch["object_workspace.posterior_summary"]["leading_hypothesis_id"] == "h_red"
    assert len(patch["object_workspace.competing_hypotheses"]) == 2
    assert patch["object_workspace.active_hypotheses_summary"]

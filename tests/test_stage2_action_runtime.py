from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.stage2_action_runtime import run_stage2_action_generation
from core.orchestration.stage_types import GovernanceStageOutput, PlannerStageOutput


class _PlannerStage:
    def __init__(self, planner_out: PlannerStageOutput) -> None:
        self.planner_out = planner_out
        self.calls = []

    def run(self, loop, stage_input):
        self.calls.append((loop, stage_input))
        return self.planner_out


class _GovernanceStage:
    def __init__(self, governance_out: GovernanceStageOutput) -> None:
        self.governance_out = governance_out
        self.calls = []

    def run(self, loop, stage_input):
        self.calls.append((loop, stage_input))
        return self.governance_out


class _Loop:
    def __init__(self, planner_out: PlannerStageOutput, governance_out: GovernanceStageOutput) -> None:
        self._planner_stage = _PlannerStage(planner_out)
        self._governance_stage = _GovernanceStage(governance_out)
        self._meta_control = SimpleNamespace(policy_profile_object_id="policy-1")
        self.built_frames = []

    def _build_tick_context_frame(self, obs_before, continuity_snapshot):
        frame = SimpleNamespace(frame_id="built-frame")
        self.built_frames.append((obs_before, continuity_snapshot, frame))
        return frame


def _planner_output(arm: str = "candidate") -> PlannerStageOutput:
    return PlannerStageOutput(
        raw_base_action={"function": "raw"},
        base_action={"function": "base"},
        arm_action={"function": "arm"},
        arm_meta={"arm": arm},
        plan_tick_meta={"planned": True},
        candidate_actions=[{"function": "candidate"}],
        visible_functions=["visible"],
        discovered_functions=["discovered"],
        raw_candidates_snapshot=[{"fn": "candidate"}],
        decision_context={"ctx": True},
        stage_metrics={"n_raw_candidates": 1},
        deliberation_result={"mode": "test"},
    )


def _governance_output() -> GovernanceStageOutput:
    return GovernanceStageOutput(
        candidate_actions=[{"function": "candidate"}],
        decision_outcome=SimpleNamespace(primary_reason="picked"),
        decision_arbiter_selected={"function_name": "arm"},
        action_to_use={"function": "final"},
        governance_result={"selected_name": "final", "reason": "ok"},
    )


def test_stage2_action_generation_uses_built_frame_and_arm_action() -> None:
    planner_out = _planner_output(arm="candidate")
    governance_out = _governance_output()
    loop = _Loop(planner_out, governance_out)

    payload = run_stage2_action_generation(
        loop,
        {"obs": "before"},
        ["surface"],
        {"continuity": True},
    )

    assert loop.built_frames
    planner_call = loop._planner_stage.calls[0][1]
    assert planner_call.obs_before == {"obs": "before"}
    assert planner_call.surfaced == ["surface"]
    assert planner_call.continuity_snapshot == {"continuity": True}
    governance_call = loop._governance_stage.calls[0][1]
    assert governance_call.action_to_use == {"function": "arm"}
    assert governance_call.planner_output is planner_out
    assert governance_call.frame is planner_call.frame

    assert payload["raw_base_action"] == {"function": "raw"}
    assert payload["base_action"] == {"function": "base"}
    assert payload["arm_action"] == {"function": "arm"}
    assert payload["candidate_actions"] == [{"function": "candidate"}]
    assert payload["action_to_use"] == {"function": "final"}
    assert payload["governance_decision"] == "final"
    assert payload["governance_reason"] == "ok"
    assert payload["policy_profile_object_id"] == "policy-1"


def test_stage2_action_generation_uses_base_action_for_base_arm() -> None:
    planner_out = _planner_output(arm="base")
    governance_out = _governance_output()
    frame = SimpleNamespace(frame_id="provided-frame")
    loop = _Loop(planner_out, governance_out)

    run_stage2_action_generation(
        loop,
        {"obs": "before"},
        ["surface"],
        {"continuity": True},
        frame=frame,
    )

    assert not loop.built_frames
    governance_call = loop._governance_stage.calls[0][1]
    assert governance_call.action_to_use == {"function": "base"}
    assert governance_call.frame is frame

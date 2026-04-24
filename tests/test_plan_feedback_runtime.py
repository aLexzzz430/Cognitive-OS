from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.plan_feedback_runtime import (
    apply_step_transitions_with_feedback,
    plan_step_feedback_reference,
    recent_llm_route_usage_for_task,
    record_verification_feedback_for_transition,
    should_auto_consume_verifier_authority,
    verification_feedback_from_transition,
)


class _PlanState:
    has_plan = True

    def __init__(self) -> None:
        self.applied = []

    def get_plan_summary(self):
        return {
            "goal_contract": {"goal_id": "goal-1"},
            "task_graph": {
                "goal_id": "graph-goal",
                "nodes": [
                    {
                        "node_id": "task-1",
                        "title": "Verify door",
                        "provenance": {"step_id": "step-1"},
                        "verification_gate": {
                            "required": True,
                            "verifier_function": "check_door",
                        },
                    }
                ],
            },
            "task_contract": {"task_ref": "task-1"},
            "completion_gate": {"requires_verification": True},
            "execution_authority": {},
        }

    def apply_step_transition(self, transition):
        self.applied.append(dict(transition))
        return True


def _usage_log():
    return [
        {
            "event": "request",
            "episode": 2,
            "tick": 3,
            "route_name": "planner",
            "requested_route": "planner",
            "selected_route": "planner",
            "goal_id": "goal-1",
            "active_task_id": "task-1",
        },
        {
            "event": "request",
            "episode": 2,
            "tick": 4,
            "route_name": "planner",
            "requested_route": "planner",
            "selected_route": "planner",
            "goal_id": "goal-1",
            "active_task_id": "task-1",
        },
        {
            "event": "request",
            "episode": 2,
            "tick": 4,
            "route_name": "analyst",
            "requested_route": "analyst",
            "selected_route": "analyst",
            "goal_id": "goal-1",
            "active_task_id": "other-task",
        },
        {"event": "request", "episode": 1, "tick": 4, "selected_route": "old"},
        {"event": "blocked", "episode": 2, "tick": 4, "selected_route": "blocked"},
    ]


def test_plan_step_feedback_reference_uses_summary_task_graph() -> None:
    reference = plan_step_feedback_reference(_PlanState(), step_id="step-1")

    assert reference == {
        "goal_id": "goal-1",
        "step_id": "step-1",
        "task_node_id": "task-1",
        "step_title": "Verify door",
        "verifier_function": "check_door",
        "verification_required": True,
    }


def test_verification_feedback_parses_verification_and_completion_events() -> None:
    assert verification_feedback_from_transition(
        {
            "event": "verification_result",
            "verified": False,
            "verifier_function": "check_door",
            "verification_evidence": {"reason": "still locked"},
        }
    ) == {
        "verified": False,
        "feedback_kind": "verification_result",
        "verifier_function": "check_door",
        "evidence": {"reason": "still locked"},
    }
    assert verification_feedback_from_transition(
        {"event": "complete", "verification_evidence": {"ok": True}}
    )["feedback_kind"] == "completion_verification"
    assert verification_feedback_from_transition({"event": "advance"}) is None


def test_should_auto_consume_verifier_authority_on_failed_required_verification() -> None:
    parsed = verification_feedback_from_transition({"event": "verification_result", "verified": False})

    assert should_auto_consume_verifier_authority(
        {"event": "verification_result", "verified": False},
        parsed_feedback=parsed,
        step_ref={"verification_required": True},
        plan_state=_PlanState(),
    )
    assert not should_auto_consume_verifier_authority(
        {"event": "verification_result", "verified": True},
        parsed_feedback=verification_feedback_from_transition({"event": "verification_result", "verified": True}),
        step_ref={"verification_required": True},
        plan_state=_PlanState(),
    )


def test_recent_llm_route_usage_filters_by_episode_tick_and_task_or_goal() -> None:
    matched = recent_llm_route_usage_for_task(
        _usage_log(),
        task_node_id="task-1",
        goal_id="goal-1",
        current_episode=2,
        current_tick=4,
        tick_window=2,
    )

    assert [row["selected_route"] for row in matched] == ["analyst", "planner", "planner"]


def test_record_verification_feedback_writes_unique_route_feedback_and_result() -> None:
    feedback_rows = []
    result = record_verification_feedback_for_transition(
        {
            "event": "verification_result",
            "verified": False,
            "step_id": "step-1",
            "verifier_function": "check_door",
            "verification_evidence": {"reason": "still locked"},
        },
        plan_state=_PlanState(),
        route_usage_log=_usage_log(),
        current_episode=2,
        current_tick=4,
        record_route_feedback=lambda route_name, **kwargs: feedback_rows.append({"route_name": route_name, **kwargs}),
    )

    assert result["passed"] is False
    assert result["goal_ref"] == "goal-1"
    assert result["task_ref"] == "task-1"
    assert result["failure_mode"] == "block"
    assert [row["route_name"] for row in feedback_rows] == ["analyst", "planner"]
    assert feedback_rows[0]["score"] == -1.0
    assert feedback_rows[0]["metadata"]["verification_result"]["result_id"] == result["result_id"]


def test_apply_step_transitions_with_feedback_consumes_authority_and_returns_last_result() -> None:
    plan_state = _PlanState()
    feedback_rows = []

    result = apply_step_transitions_with_feedback(
        plan_state=plan_state,
        transitions=[
            {
                "event": "verification_result",
                "verified": False,
                "step_id": "step-1",
                "verifier_function": "check_door",
            }
        ],
        route_usage_log=_usage_log(),
        current_episode=2,
        current_tick=4,
        record_route_feedback=lambda route_name, **kwargs: feedback_rows.append({"route_name": route_name, **kwargs}),
    )

    assert result["applied"] == 1
    assert result["last_verification_result"]["passed"] is False
    assert plan_state.applied[0]["consume_verifier_authority"] is True
    assert {row["route_name"] for row in feedback_rows} == {"planner", "analyst"}

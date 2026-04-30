from pathlib import Path

from core.cognition.goal_pressure import build_goal_pressure_update
from core.cognition.outcome_model_update import build_outcome_model_update
from modules.state.manager import StateManager
from modules.state.schema import StateSchema


def test_goal_pressure_turns_failed_action_into_active_capability_goal(tmp_path: Path) -> None:
    state = StateSchema.get_default_state()
    outcome = build_outcome_model_update(
        action={"function_name": "file_read"},
        result={"success": False, "error_type": "invalid_kwargs"},
        existing_state=state,
        episode=1,
        tick=3,
    )

    pressure = build_goal_pressure_update(
        outcome_update=outcome,
        existing_state=state,
        episode=1,
        tick=3,
    )

    assert pressure.created_or_updated is True
    assert pressure.pressure_type == "capability_repair"
    assert pressure.goal_id == "goal:capability_repair:file_read"
    assert pressure.priority >= 0.72
    subgoal = pressure.goal_patch["goal_stack.subgoals"][-1]
    assert subgoal["status"] == "active"
    assert subgoal["metadata"]["capability"]["failures"] == 1
    assert "promote_failed_skill_without_verifier" in subgoal["forbidden_actions"]

    manager = StateManager(state_path=str(tmp_path / "state.json"))
    manager.initialize()
    manager.update_state(pressure.goal_patch, reason="test:goal_pressure", module="goal_runtime")
    saved = manager.get_state()
    assert saved["goal_stack"]["subgoals"][0]["goal_id"] == pressure.goal_id
    assert saved["goal_stack"]["goal_status"][pressure.goal_id]["pressure_type"] == "capability_repair"


def test_goal_pressure_turns_repeated_verified_success_into_skill_candidate_goal() -> None:
    state = StateSchema.get_default_state()
    state["self_summary"]["capability_estimate"] = {
        "run_test": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
            "verified_successes": 1,
            "reliability": 1.0,
        }
    }
    outcome = build_outcome_model_update(
        action={"function_name": "run_test"},
        result={"success": True, "verified": True, "state": "completed_verified"},
        existing_state=state,
        episode=2,
        tick=5,
    )

    pressure = build_goal_pressure_update(
        outcome_update=outcome,
        existing_state=state,
        episode=2,
        tick=5,
    )

    assert pressure.created_or_updated is True
    assert pressure.pressure_type == "skill_candidate"
    assert pressure.goal_id == "goal:skill_candidate:run_test"
    subgoal = pressure.goal_patch["goal_stack.subgoals"][-1]
    assert subgoal["metadata"]["capability"]["verified_successes"] == 2
    assert "install_skill_without_review" in subgoal["forbidden_actions"]


def test_goal_pressure_deduplicates_existing_subgoal() -> None:
    state = StateSchema.get_default_state()
    state["goal_stack"]["subgoals"] = [
        {
            "goal_id": "goal:capability_repair:repo_grep",
            "title": "old",
            "status": "active",
        }
    ]
    outcome = build_outcome_model_update(
        action={"function_name": "repo_grep"},
        result={"success": False, "message": "timeout"},
        existing_state=state,
        episode=3,
        tick=8,
    )

    pressure = build_goal_pressure_update(outcome_update=outcome, existing_state=state, episode=3, tick=8)

    subgoals = pressure.goal_patch["goal_stack.subgoals"]
    matching = [row for row in subgoals if row["goal_id"] == "goal:capability_repair:repo_grep"]
    assert len(matching) == 1
    assert matching[0]["title"] == "Improve unreliable action: repo_grep"

from core.cognition.outcome_model_update import build_outcome_model_update
from modules.state.schema import StateSchema


def test_outcome_model_update_records_failure_in_world_self_and_learning_state():
    state = StateSchema.get_default_state()

    update = build_outcome_model_update(
        action={"function_name": "inspect", "_candidate_meta": {"expected_success": True}},
        result={"success": False, "error_type": "invalid_observation", "message": "missing required field"},
        evidence_entries=[{"evidence_id": "ev-1", "status": "recorded", "claim": "inspect failed"}],
        existing_state=state,
        episode=2,
        tick=7,
    )

    assert update.outcome == "failure"
    assert update.verified is False
    assert update.world_patch["world_summary.risk_estimate"] > 0.5
    assert update.world_patch["world_summary.uncertainty_estimate"] > 0.5
    assert update.world_patch["world_summary.observed_facts"][-1]["action"] == "inspect"
    assert update.self_patch["self_summary.confidence"] < 0.5
    assert update.self_patch["self_summary.capability_estimate"]["inspect"]["failures"] == 1
    assert update.self_patch["self_summary.recent_failures"][-1]["failure_type"] == "invalid_observation"
    assert "last_action_failed" in update.self_patch["self_summary.error_flags"]
    assert update.learning_patch["learning_context.prediction_error"]["unexpected"] is True
    assert update.learning_patch["learning_context.belief_updates"][-1]["evidence_refs"][0]["evidence_id"] == "ev-1"


def test_outcome_model_update_raises_confidence_for_verified_success_and_rolls_capability():
    state = StateSchema.get_default_state()
    state["self_summary"]["capability_estimate"] = {
        "repair": {"attempts": 1, "successes": 0, "failures": 1, "unknowns": 0}
    }
    state["self_summary"]["error_flags"] = ["last_action_failed", "other_warning"]

    update = build_outcome_model_update(
        action={"payload": {"tool_args": {"function_name": "repair"}}},
        result={"success": True, "verified": True, "state": "completed_verified"},
        existing_state=state,
        episode=4,
        tick=11,
    )

    capability = update.self_patch["self_summary.capability_estimate"]["repair"]
    assert update.outcome == "success"
    assert update.verified is True
    assert capability["attempts"] == 2
    assert capability["successes"] == 1
    assert capability["failures"] == 1
    assert capability["verified_successes"] == 1
    assert capability["reliability"] == 0.5
    assert update.self_patch["self_summary.confidence"] > 0.5
    assert update.world_patch["world_summary.risk_estimate"] < 0.5
    assert "last_action_failed" not in update.self_patch["self_summary.error_flags"]
    assert "other_warning" in update.self_patch["self_summary.error_flags"]

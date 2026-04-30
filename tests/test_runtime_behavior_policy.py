from __future__ import annotations

from core.cognition.runtime_behavior_policy import derive_runtime_behavior_policy
from core.runtime.autonomous_tick import select_autonomous_goal_pressure


def test_world_model_uncertainty_drives_deep_think_and_strong_model() -> None:
    decision = derive_runtime_behavior_policy(
        {
            "world_summary": {
                "uncertainty_estimate": 0.81,
                "risk_estimate": 0.35,
                "latent_hypotheses": [{"id": "h1"}, {"id": "h2"}],
            },
            "goal_stack": {
                "subgoals": [
                    {
                        "goal_id": "goal-world",
                        "source": "world_model_homeostasis",
                        "objective": "Review uncertainty and propose discriminating observations.",
                        "priority": 0.6,
                        "permission_level": "L1",
                        "allowed_actions": ["read_reports", "write_report"],
                    }
                ]
            },
        }
    )

    assert decision.runtime_mode == "DEEP_THINK"
    assert decision.model_selection["model_tier"] == "strong"
    assert decision.model_selection["prefer_strongest_model"] is True
    assert decision.permission_policy["side_effects_allowed"] is False
    assert decision.selected_task["goal_id"] == "goal-world"


def test_failure_learning_rules_drive_budget_permissions_and_retrieval_objects() -> None:
    decision = derive_runtime_behavior_policy(
        {
            "self_summary": {
                "resource_tightness": "tight",
                "recent_failures": [{"failure_type": "sync_before_verifier"}],
            },
            "goal_stack": {
                "subgoals": [
                    {
                        "goal_id": "goal-repair",
                        "source": "self_model_failure",
                        "objective": "Investigate verifier bypass failure.",
                        "priority": 0.7,
                        "risk": 0.1,
                        "permission_level": "L2",
                        "allowed_actions": ["read_logs", "run_tests", "propose_patch"],
                    }
                ]
            },
            "learning_context": {
                "failure_objects": [
                    {
                        "failure_id": "failure-1",
                        "confidence": 0.88,
                        "failure_object": {
                            "failure_mode": "governance_block",
                            "future_retrieval_object": {
                                "preferred_next_actions": ["run_test"],
                                "avoid_actions": ["mirror_plan"],
                                "query_keys": ["verifier", "sync"],
                            },
                            "new_regression_test": {"description": "sync requires verifier evidence"},
                            "new_governance_rule": {
                                "description": "Block mirror planning before verified changes.",
                                "blocked_actions": ["mirror_plan"],
                            },
                        },
                    }
                ]
            },
        }
    )

    assert decision.runtime_mode == "ROUTINE_RUN"
    assert decision.llm_budget["max_llm_calls"] <= 1
    assert decision.model_selection["allow_cloud_escalation"] is False
    assert "execute" in decision.permission_policy["approval_required_capability_layers"]
    assert decision.learning_behavior_rules["blocked_actions"] == ["mirror_plan"]
    assert decision.regression_tests[0]["description"] == "sync requires verifier evidence"
    assert decision.retrieval_objects[0]["query_keys"] == ["verifier", "sync"]


def test_autonomous_goal_selection_uses_self_world_behavior_policy() -> None:
    selected = select_autonomous_goal_pressure(
        {
            "self_summary": {
                "recent_failures": [{"failure_type": "model_timeout"}],
                "error_flags": ["model_timeout"],
            },
            "goal_stack": {
                "subgoals": [
                    {
                        "goal_id": "low-value-docs",
                        "source": "opportunity",
                        "objective": "Write a cosmetic docs note.",
                        "priority": 0.75,
                        "permission_level": "L1",
                        "allowed_actions": ["read_reports", "write_report"],
                    },
                    {
                        "goal_id": "self-repair",
                        "source": "self_model_failure",
                        "objective": "Investigate repeated model timeout failure.",
                        "priority": 0.68,
                        "permission_level": "L1",
                        "allowed_actions": ["read_logs", "read_reports", "write_report"],
                    },
                ]
            },
        }
    )

    assert selected["goal_id"] == "self-repair"
    assert selected["metadata"]["runtime_behavior_policy"]["reason"] in {
        "selected_task_is_read_only_background_work",
        "self_model_adaptation_pressure_can_create_candidates",
    }

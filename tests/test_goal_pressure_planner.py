from core.orchestration.context_builder import UnifiedContextBuilder, UnifiedContextInput
from core.orchestration.goal_task_control import GoalTaskRuntime
from core.orchestration.state_abstraction import summarize_goal_agenda


def test_goal_agenda_preserves_structured_subgoals_and_priority_order() -> None:
    agenda = summarize_goal_agenda(
        goal_stack={
            "top_goal": "survive and improve",
            "subgoals": [
                {
                    "goal_id": "goal:low",
                    "title": "Low priority repair",
                    "status": "active",
                    "priority": 0.2,
                    "pressure_type": "capability_repair",
                },
                {
                    "goal_id": "goal:done",
                    "title": "Completed repair",
                    "status": "completed",
                    "priority": 1.0,
                },
                {
                    "goal_id": "goal:high",
                    "title": "High priority skill evaluation",
                    "status": "active",
                    "priority": 0.9,
                    "pressure_type": "skill_candidate",
                    "success_condition": "skill candidate bounded",
                },
            ],
        },
        continuity_snapshot={},
        plan_summary={},
    )

    subgoals = [item for item in agenda if item.get("horizon") == "subgoal"]
    assert [item["goal_id"] for item in subgoals] == ["goal:high", "goal:low"]
    assert subgoals[0]["pressure_type"] == "skill_candidate"
    assert subgoals[0]["success_condition"] == "skill candidate bounded"
    assert not any(item.get("goal_id") == "goal:done" for item in agenda)


def test_unified_context_uses_highest_priority_goal_pressure_when_current_task_empty() -> None:
    ctx = UnifiedContextBuilder.build(
        UnifiedContextInput(
            unified_enabled=True,
            unified_ablation_mode="stripped",
            obs={},
            continuity_snapshot={},
            world_model_summary={},
            self_model_summary={},
            plan_summary={},
            current_task="",
            active_hypotheses=[],
            episode_trace_tail=[],
            retrieval_should_query=False,
            probe_pressure=0.0,
            workspace_state={
                "goal_stack": {
                    "top_goal": "become a reliable cognitive runtime",
                    "subgoals": [
                        {
                            "goal_id": "goal:capability_repair:file_read",
                            "title": "Improve unreliable action: file_read",
                            "status": "active",
                            "priority": 0.91,
                            "pressure_type": "capability_repair",
                        },
                        {
                            "goal_id": "goal:skill_candidate:run_test",
                            "title": "Evaluate reusable skill: run_test",
                            "status": "active",
                            "priority": 0.74,
                            "pressure_type": "skill_candidate",
                        },
                    ],
                }
            },
        )
    )

    assert ctx.current_goal == "become a reliable cognitive runtime"
    assert ctx.current_task == "Improve unreliable action: file_read"
    assert ctx.goal_agenda[1]["goal_id"] == "goal:capability_repair:file_read"


def test_goal_task_runtime_promotes_goal_pressure_to_active_task_without_user_task() -> None:
    ctx = UnifiedContextBuilder.build(
        UnifiedContextInput(
            unified_enabled=True,
            unified_ablation_mode="stripped",
            obs={},
            continuity_snapshot={},
            world_model_summary={},
            self_model_summary={},
            plan_summary={},
            current_task="",
            active_hypotheses=[],
            episode_trace_tail=[],
            retrieval_should_query=False,
            probe_pressure=0.0,
            workspace_state={
                "goal_stack": {
                    "top_goal": "maintain agency",
                    "subgoals": [
                        {
                            "goal_id": "goal:skill_candidate:run_test",
                            "title": "Evaluate reusable skill: run_test",
                            "status": "active",
                            "priority": 0.88,
                            "pressure_type": "skill_candidate",
                            "success_criteria": ["skill candidate has verifier boundary"],
                        }
                    ],
                }
            },
        )
    )

    binding = GoalTaskRuntime().refresh(
        unified_context=ctx,
        state_mgr=None,
        run_id="run-goal-pressure",
        episode=4,
        tick=9,
    )

    assert binding.goal_contract is not None
    assert binding.active_task is not None
    assert binding.active_task.title == "Evaluate reusable skill: run_test"
    assert binding.active_task.metadata["pressure_type"] == "skill_candidate"
    assert binding.active_task.success_criteria == ["skill candidate has verifier boundary"]

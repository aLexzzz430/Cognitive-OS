from __future__ import annotations

from core.cognition.model_influence import ModelInfluenceInput, apply_cognitive_model_influence
from core.orchestration.stage2_candidate_filter_runtime import run_stage2_self_model_suppression


def test_world_model_boosts_required_observation_action() -> None:
    candidates = [
        {"function_name": "act", "final_score": 0.4},
        {"function_name": "observe", "final_score": 0.2},
    ]

    result = apply_cognitive_model_influence(
        ModelInfluenceInput(
            candidate_actions=candidates,
            world_model_state={
                "uncertainty": 0.9,
                "required_observations": ["observe"],
            },
            self_model_state={},
            tick=3,
            episode=1,
        )
    )

    assert result.candidate_actions[0]["function_name"] == "observe"
    influence = result.candidate_actions[0]["_candidate_meta"]["cognitive_model_influence"]
    assert influence["world_model_bonus"] > 0.0
    assert "world_model_required_observation" in influence["reasons"]


def test_self_model_blocks_known_bad_action() -> None:
    candidates = [
        {"function_name": "risky_action", "final_score": 0.9},
        {"function_name": "observe", "final_score": 0.1},
    ]

    result = apply_cognitive_model_influence(
        ModelInfluenceInput(
            candidate_actions=candidates,
            world_model_state={},
            self_model_state={
                "known_failure_modes": [
                    {
                        "action": "risky_action",
                        "risk": 0.95,
                        "block": True,
                        "reason": "repeated unsafe failure",
                    }
                ]
            },
        )
    )

    assert [row["function_name"] for row in result.candidate_actions] == ["observe"]
    assert result.blocked_count == 1
    assert any(row["event"] == "cognitive_model_action_influence" for row in result.audit_records)


def test_failure_learning_object_changes_future_action_ordering() -> None:
    candidates = [
        {"function_name": "mirror_plan", "final_score": 0.95},
        {"function_name": "run_test", "final_score": 0.35},
    ]

    result = apply_cognitive_model_influence(
        ModelInfluenceInput(
            candidate_actions=candidates,
            world_model_state={},
            self_model_state={
                "failure_learning_objects": [
                    {
                        "failure_id": "failure-sync-before-verifier",
                        "confidence": 0.9,
                        "failure_object": {
                            "failure_mode": "governance_block",
                            "future_retrieval_object": {
                                "preferred_next_actions": ["run_test"],
                                "avoid_actions": ["mirror_plan"],
                            },
                            "new_governance_rule": {
                                "description": "Do not plan source sync before verifier evidence.",
                                "blocked_actions": ["mirror_plan"],
                            },
                        },
                    }
                ]
            },
        )
    )

    assert [row["function_name"] for row in result.candidate_actions] == ["run_test"]
    assert result.blocked_count == 1
    influence = result.candidate_actions[0]["_candidate_meta"]["cognitive_model_influence"]
    assert "failure_learning_preferred_action" in influence["reasons"]


class _DummyPlanState:
    has_plan = False
    current_step = None


class _DummyLoop:
    _tick = 5
    _episode = 2
    _episode_trace = []
    _reliability_tracker = None
    _viability_audit_log: list[dict] = []

    def __init__(self) -> None:
        self._viability_audit_log = []

    def _extract_action_function_name(self, action: dict, default: str = "") -> str:
        return str(action.get("function_name") or default)

    def _infer_task_family(self, _snapshot: dict) -> str:
        return "generic"

    def _extract_phase_hint(self, _snapshot: dict) -> str:
        return "explore"

    def _build_self_model_prediction_summary(self) -> dict:
        return {
            "self_model_state": {
                "known_failure_modes": [
                    {
                        "action": "commit",
                        "risk": 0.9,
                        "block": True,
                        "reason": "self model says commit is currently unsafe",
                    }
                ],
                "continuity_confidence": 0.8,
            },
            "budget_tight": False,
            "resource_tightness": "normal",
        }


def test_stage2_applies_world_and_self_model_influence_to_candidates() -> None:
    loop = _DummyLoop()
    candidates = [
        {"function_name": "commit", "final_score": 0.95},
        {"function_name": "observe", "final_score": 0.1},
    ]
    stage_input = type(
        "StageInput",
        (),
        {
            "candidate_actions": candidates,
            "continuity_snapshot": {
                "world_model_summary": {
                    "uncertainty": 0.8,
                    "required_observations": ["observe"],
                }
            },
            "obs_before": {},
        },
    )()

    filtered = run_stage2_self_model_suppression(loop, stage_input)

    assert [row["function_name"] for row in filtered] == ["observe"]
    assert any(row.get("event") == "cognitive_model_action_influence" for row in loop._viability_audit_log)


def test_stage2_applies_failure_learning_from_local_machine_observation() -> None:
    loop = _DummyLoop()
    candidates = [
        {"function_name": "mirror_plan", "final_score": 0.95},
        {"function_name": "run_test", "final_score": 0.35},
    ]
    stage_input = type(
        "StageInput",
        (),
        {
            "candidate_actions": candidates,
            "continuity_snapshot": {},
            "obs_before": {
                "local_mirror": {
                    "end_to_end_learning": {
                        "failure_objects": [
                            {
                                "confidence": 0.9,
                                "failure_object": {
                                    "failure_mode": "governance_block",
                                    "future_retrieval_object": {
                                        "preferred_next_actions": ["run_test"],
                                        "avoid_actions": ["mirror_plan"],
                                    },
                                    "new_governance_rule": {"blocked_actions": ["mirror_plan"]},
                                },
                            }
                        ]
                    }
                }
            },
        },
    )()

    filtered = run_stage2_self_model_suppression(loop, stage_input)

    assert [row["function_name"] for row in filtered] == ["run_test"]
    assert any(
        "failure_learning_governance_block" in row.get("block_reasons", [])
        for row in loop._viability_audit_log
    )

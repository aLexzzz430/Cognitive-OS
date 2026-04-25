from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.action_utils import repair_action_function_name
from core.orchestration.planner_runtime import PlannerRuntime


def test_repair_action_function_name_does_not_invent_function_for_semantic_action() -> None:
    action = {
        "kind": "call_tool",
        "target": "tile-A",
        "payload": {
            "tool_args": {
                "kwargs": {
                    "x": 4,
                    "y": 7,
                    "target_family": "switch",
                }
            }
        },
    }

    repaired = repair_action_function_name(action, "ACTION6")

    assert repaired == action
    assert "function_name" not in repaired["payload"]["tool_args"]


def test_repair_action_function_name_only_backfills_empty_action() -> None:
    repaired = repair_action_function_name(
        {"kind": "call_tool", "payload": {"tool_args": {}}},
        "inspect",
    )

    assert repaired["payload"]["tool_args"]["function_name"] == "inspect"


def test_planner_rollout_synthetic_action_prefers_raw_action_over_planner_label() -> None:
    synthetic = PlannerRuntime._planner_rollout_action_to_synthetic_action(
        {
            "function": "planner_label",
            "kwargs": {"x": 99, "relation_type": "supports"},
            "raw_action": {
                "kind": "call_tool",
                "payload": {
                    "tool_args": {
                        "function_name": "real_probe",
                        "kwargs": {"x": 2, "y": 3, "target_family": "lever"},
                    }
                },
            },
            "selected_action": {
                "kind": "call_tool",
                "payload": {
                    "tool_args": {
                        "function_name": "selected_label",
                        "kwargs": {"x": 11, "y": 12},
                    }
                },
            },
        }
    )
    tool_args = synthetic["payload"]["tool_args"]

    assert tool_args["function_name"] == "real_probe"
    assert tool_args["kwargs"]["x"] == 2
    assert tool_args["kwargs"]["y"] == 3
    assert tool_args["kwargs"]["target_family"] == "lever"
    assert tool_args["kwargs"]["relation_type"] == "supports"

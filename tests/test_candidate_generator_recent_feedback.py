from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decision.candidate_generator import CandidateGenerator


def _repo_tree_action() -> dict:
    return _call_action("repo_tree")


def _call_action(function_name: str) -> dict:
    return {
        "kind": "call_tool",
        "payload": {
            "tool_name": "call_hidden_function",
            "tool_args": {"function_name": function_name, "kwargs": {}},
        },
        "_candidate_meta": {},
    }


def test_repeated_repo_tree_inventory_gets_cooldown_even_with_observation_reward() -> None:
    generator = object.__new__(CandidateGenerator)
    candidate = _repo_tree_action()
    trace = [{"action": _repo_tree_action(), "reward": 0.15} for _ in range(3)]

    generator._annotate_recent_action_feedback([candidate], trace)

    feedback = candidate["_candidate_meta"]["recent_action_feedback"]
    assert feedback["recent_same_action_count"] == 3
    assert feedback["positive_progress_count"] == 3
    assert feedback["repeated_inventory_probe"] is True
    assert feedback["action_cooldown_recommended"] is True


def test_repeated_empty_repo_grep_gets_cooldown_even_with_small_reward() -> None:
    generator = object.__new__(CandidateGenerator)
    candidate = _call_action("repo_grep")
    trace = [
        {"action": _call_action("repo_grep"), "reward": 0.02, "result": {"match_count": 0}}
        for _ in range(3)
    ]

    generator._annotate_recent_action_feedback([candidate], trace)

    feedback = candidate["_candidate_meta"]["recent_action_feedback"]
    assert feedback["recent_same_action_count"] == 3
    assert feedback["positive_progress_count"] == 3
    assert feedback["repeated_low_value_search_probe"] is True
    assert feedback["action_cooldown_recommended"] is True

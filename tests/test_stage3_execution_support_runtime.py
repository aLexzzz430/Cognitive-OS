from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.stage3_execution_support_runtime import (
    collect_executable_function_names,
    is_trackable_executable_function,
    record_memory_consumption_proof,
    resolve_action_for_execution,
)


class _RawApi:
    raw = {
        "visible_functions": ["visible", "", 3],
        "discovered_functions": ["discovered"],
    }


class _Resolver:
    def __init__(self, resolved):
        self.resolved = resolved
        self.calls = []

    def resolve(self, action_to_use, obs_before):
        self.calls.append((action_to_use, obs_before))
        return dict(self.resolved)


class _World:
    def __init__(self) -> None:
        self.decorated = []

    def decorate_candidate_action(self, action):
        self.decorated.append(dict(action))
        decorated = dict(action)
        decorated["decorated"] = True
        return decorated


class _PlanState:
    def get_target_function_for_step(self):
        return "target_fn"


class _BeliefLedger:
    def get_active_beliefs(self):
        return [
            SimpleNamespace(variable_name="belief-a"),
            SimpleNamespace(variable_name="belief-b"),
            SimpleNamespace(variable_name="belief-c"),
            SimpleNamespace(variable_name="belief-d"),
        ]


class _Loop:
    def __init__(self, resolved):
        self._selected_action_execution_resolver = _Resolver(resolved)
        self._world = _World()
        self._tick = 6
        self._episode = 2
        self._plan_state = _PlanState()
        self._belief_ledger = _BeliefLedger()
        self._governance_log = []

    def _extract_action_function_name(self, action, default=""):
        return action.get("function", default) if isinstance(action, dict) else default

    def _get_reward(self, result):
        return float(result.get("reward", 0.0))


def test_resolve_action_for_execution_decorates_wait_actions() -> None:
    loop = _Loop({"function": "wait"})

    resolved = resolve_action_for_execution(loop, {"function": "raw"}, {"obs": "before"})

    assert resolved == {"function": "wait", "decorated": True}
    assert loop._selected_action_execution_resolver.calls == [({"function": "raw"}, {"obs": "before"})]
    assert loop._world.decorated == [{"function": "wait"}]


def test_resolve_action_for_execution_returns_non_wait_without_decoration() -> None:
    loop = _Loop({"function": "move", "x": 1})

    resolved = resolve_action_for_execution(loop, {"function": "raw"}, {"obs": "before"})

    assert resolved == {"function": "move", "x": 1}
    assert loop._world.decorated == []


def test_collect_executable_function_names_reads_api_and_signatures() -> None:
    names = collect_executable_function_names(
        {
            "novel_api": _RawApi(),
            "function_signatures": {
                "signed": {"args": []},
                1: {"ignored": True},
            },
        }
    )

    assert names == {"visible", "discovered", "signed"}


def test_is_trackable_executable_function_excludes_synthetic_names() -> None:
    executable = {"move", "inspect"}

    assert is_trackable_executable_function("move", executable) is True
    assert is_trackable_executable_function("hyp_move", executable) is False
    assert is_trackable_executable_function("obj_move", executable) is False
    assert is_trackable_executable_function("missing", executable) is False
    assert is_trackable_executable_function("move", set()) is False


def test_record_memory_consumption_proof_appends_audit_entry() -> None:
    loop = _Loop({"function": "move"})
    query = SimpleNamespace(query_text="alpha beta gamma")
    action = {
        "kind": "tool",
        "payload": {
            "tool_args": {
                "function_name": "move",
                "kwargs": {"x": 1, "y": 2},
            }
        },
    }

    record_memory_consumption_proof(loop, action, query, {"reward": 1.25})

    assert len(loop._governance_log) == 1
    entry = loop._governance_log[0]
    assert entry["entry"] == "memory_consumption_proof"
    assert entry["tick"] == 6
    assert entry["episode"] == 2
    assert entry["selected_function"] == "move"
    assert entry["selected_kwargs_keys"] == ["x", "y"]
    assert entry["query_terms"] == ["alpha", "beta", "gamma"]
    assert entry["plan_target"] == "target_fn"
    assert entry["belief_context_used"] == ["belief-a", "belief-b", "belief-c"]
    assert entry["result_reward"] == 1.25

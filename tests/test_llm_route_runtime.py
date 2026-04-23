from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.llm_route_runtime import (
    RouteBudgetedLLMClient,
    estimate_llm_token_units,
)


class _DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, prompt: str, **kwargs):
        self.calls.append(("complete", prompt, dict(kwargs)))
        return "done"

    def complete_json(self, prompt: str, **kwargs):
        self.calls.append(("complete_json", prompt, dict(kwargs)))
        return {"ok": True}


def test_estimate_llm_token_units_is_monotonic_and_empty_safe() -> None:
    assert estimate_llm_token_units(None, "") == 0
    assert estimate_llm_token_units("abcd") == 1
    assert estimate_llm_token_units("abcd", "efgh") == 2
    assert estimate_llm_token_units({"long": "value"}) >= 1


def test_route_budgeted_llm_client_records_usage_with_model_call_ticket() -> None:
    budget_checks = []
    usage_rows = []
    blocked_rows = []
    llm = _DummyLLM()

    def preflight_budget_check(**kwargs):
        budget_checks.append(dict(kwargs))
        return {"allowed": True}

    client = RouteBudgetedLLMClient(
        route_name="planner",
        client=llm,
        route_metadata={
            "requested_route": "planner",
            "selected_route": "planner",
            "fallback_route": "general",
            "budget": {"request_budget": 2},
            "route_context": {
                "metadata": {
                    "goal_id": "goal-1",
                    "active_task_id": "task-1",
                    "graph_ref": "graph-1",
                    "capability_policy_source": "task_node",
                }
            },
        },
        preflight_budget_check=preflight_budget_check,
        record_usage=lambda **kwargs: usage_rows.append(dict(kwargs)),
        record_blocked=lambda **kwargs: blocked_rows.append(dict(kwargs)),
    )

    assert client.complete(
        "abcd",
        system_prompt="efgh",
        max_tokens=4,
        capability_request="planning.next_action",
        response_schema_name="planner_schema",
        capability_route_name="planner",
        temperature=0,
    ) == "done"

    assert not blocked_rows
    assert len(budget_checks) == 1
    assert len(usage_rows) == 1
    assert llm.calls == [("complete", "abcd", {"system_prompt": "efgh", "max_tokens": 4, "temperature": 0})]

    usage = usage_rows[0]
    assert usage["route_name"] == "planner"
    assert usage["method_name"] == "complete"
    assert usage["prompt_tokens"] == 2
    assert usage["response_tokens"] == 1
    ticket = usage["route_metadata"]["model_call_ticket"]
    assert ticket["route_name"] == "planner"
    assert ticket["capability_request"] == "planning.next_action"
    assert ticket["schema_name"] == "planner_schema"
    assert ticket["goal_ref"] == "goal-1"
    assert ticket["task_ref"] == "task-1"
    assert ticket["reserved_response_tokens"] == 4


def test_route_budgeted_llm_client_blocks_without_calling_underlying_client() -> None:
    blocked_rows = []
    llm = _DummyLLM()

    client = RouteBudgetedLLMClient(
        route_name="planner",
        client=llm,
        route_metadata={"budget": {"request_budget": 0}},
        preflight_budget_check=lambda **kwargs: {"allowed": False, "blocked_reason": "request_budget_exceeded"},
        record_usage=lambda **kwargs: None,
        record_blocked=lambda **kwargs: blocked_rows.append(dict(kwargs)),
    )

    assert client.complete_json("{}", capability_request="planning.json") == {}
    assert llm.calls == []
    assert len(blocked_rows) == 1
    assert blocked_rows[0]["entry_kind"] == "runtime_gate"
    assert blocked_rows[0]["budget_status"]["blocked_reason"] == "request_budget_exceeded"

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.llm_route_runtime import (
    RouteBudgetedLLMClient,
    ensure_llm_capability_registry,
    ensure_model_router,
    estimate_llm_token_units,
    initialize_model_router,
    llm_route_budget_status,
    llm_route_state,
    llm_route_usage_summary,
    record_llm_route_blocked,
    record_llm_route_failure,
    record_llm_route_usage,
    resolve_llm_capability_spec,
    resolve_llm_client,
    resolve_llm_gateway,
    select_model_profile_failover_candidate,
)
from modules.llm.failure_policy import decide_llm_failure_policy
from modules.llm.route_runtime_policy import route_runtime_policy_for_route


class _DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, prompt: str, **kwargs):
        self.calls.append(("complete", prompt, dict(kwargs)))
        return "done"

    def complete_json(self, prompt: str, **kwargs):
        self.calls.append(("complete_json", prompt, dict(kwargs)))
        return {"ok": True}


class _TimeoutLLM:
    def complete(self, prompt: str, **kwargs):
        raise TimeoutError("model timed out")


class _ProviderLimitLLM:
    def complete(self, prompt: str, **kwargs):
        raise RuntimeError("You've hit your usage limit for GPT-5.3-Codex-Spark. Switch to another model now.")


class _DummyLoop:
    def __init__(self) -> None:
        self._episode = 2
        self._tick = 5
        self._llm_route_usage_log = []
        self._llm_advice_log = []
        self._llm_calls_this_tick = 0
        self._feedback = {
            "planner": {
                "score": 0.5,
                "samples": 2,
            }
        }

    def _cooldown_ready(self, last_tick: int, cooldown_ticks: int) -> bool:
        return self._tick - int(last_tick or 0) >= int(cooldown_ticks or 0)

    def _json_safe(self, value):
        return value

    def _llm_route_feedback_summary(self):
        return dict(self._feedback)


class _RoutingLoop(_DummyLoop):
    def __init__(self) -> None:
        super().__init__()
        self._llm_client = _DummyLLM()
        self._llm_mode = "integrated"
        self._llm_shadow_client = None
        self._llm_analyst_client = None
        self._llm_route_specs = {}
        self._llm_capability_policies = {}
        self.route_context_calls = []

    def _default_llm_client_fallback(self):
        return self._llm_client

    def _resolved_llm_route_specs(self):
        return {
            "retrieval": {
                "served_routes": ["retrieval"],
                "client_alias": "default",
                "budget": {
                    "request_budget": 2,
                    "token_budget": 128,
                },
                "metadata": {"policy_source": "test"},
            }
        }

    def _resolved_llm_capability_specs(self):
        return {
            "retrieval.query_rewrite": {
                "route_name": "retrieval",
                "required_capabilities": ["retrieval", "grounding"],
                "metadata": {"policy_source": "test_capability"},
            }
        }

    def _build_llm_route_context(self, route_name: str, **kwargs):
        self.route_context_calls.append({"route_name": route_name, **dict(kwargs)})
        return {
            "required_capabilities": ["retrieval"],
            "prefer_low_cost": 0.1,
            "metadata": {"goal_id": "goal-1", "active_task_id": "task-1"},
        }


def _route_metadata():
    return {
        "requested_route": "planner",
        "selected_route": "planner-fast",
        "budget": {
            "request_budget": 1,
            "token_budget": 10,
            "cooldown_ticks": 2,
        },
        "decision_explanation": ["task policy preferred planner-fast"],
        "route_context": {
            "metadata": {
                "goal_id": "goal-1",
                "active_task_id": "task-1",
            }
        },
        "model_call_ticket": {
            "ticket_id": "ticket-1",
            "audit_event_id": "audit-1",
        },
    }


def test_estimate_llm_token_units_is_monotonic_and_empty_safe() -> None:
    assert estimate_llm_token_units(None, "") == 0
    assert estimate_llm_token_units("abcd") == 1
    assert estimate_llm_token_units("abcd", "efgh") == 2
    assert estimate_llm_token_units({"long": "value"}) >= 1


def test_route_accounting_records_usage_and_enforces_tick_budget() -> None:
    loop = _DummyLoop()
    metadata = _route_metadata()

    preflight = llm_route_budget_status(
        loop,
        route_name="planner",
        route_metadata=metadata,
        prompt_tokens=2,
        reserved_response_tokens=3,
    )

    assert preflight["allowed"] is True
    assert preflight["estimated_total_tokens"] == 5
    assert preflight["remaining_request_budget"] == 1

    record_llm_route_usage(
        loop,
        route_name="planner",
        method_name="complete",
        prompt_tokens=2,
        response_tokens=3,
        reserved_response_tokens=5,
        route_metadata=metadata,
    )

    state = llm_route_state(loop)
    assert loop._llm_calls_this_tick == 1
    assert state["per_tick_usage"]["planner"]["request_count"] == 1
    assert state["per_tick_usage"]["planner"]["token_count"] == 5
    assert state["lifetime_usage"]["planner"] == {"request_count": 1, "token_count": 5}
    assert state["last_call"]["planner"] == {"episode": 2, "tick": 5}

    usage = loop._llm_route_usage_log[0]
    assert usage["event"] == "request"
    assert usage["goal_id"] == "goal-1"
    assert usage["active_task_id"] == "task-1"
    assert usage["model_call_ticket_id"] == "ticket-1"
    assert loop._llm_advice_log[0]["entry"] == "llm_route_request"

    blocked_by_request_budget = llm_route_budget_status(
        loop,
        route_name="planner",
        route_metadata=metadata,
    )
    assert blocked_by_request_budget["allowed"] is False
    assert blocked_by_request_budget["blocked_reason"] == "request_budget_exceeded"

    loop._tick = 6
    blocked_by_cooldown = llm_route_budget_status(
        loop,
        route_name="planner",
        route_metadata=metadata,
    )
    assert blocked_by_cooldown["allowed"] is False
    assert blocked_by_cooldown["blocked_reason"] == "cooldown_active"

    loop._tick = 7
    allowed_after_cooldown = llm_route_budget_status(
        loop,
        route_name="planner",
        route_metadata=metadata,
    )
    assert allowed_after_cooldown["allowed"] is True

    summary = llm_route_usage_summary(loop)
    assert summary["lifetime_usage"]["planner"]["token_count"] == 5
    assert summary["feedback"]["planner"]["score"] == 0.5


def test_route_accounting_records_blocked_audit_rows() -> None:
    loop = _DummyLoop()
    metadata = _route_metadata()
    budget_status = {
        "allowed": False,
        "blocked_reason": "token_budget_exceeded",
    }

    record_llm_route_blocked(
        loop,
        route_name="planner",
        method_name="resolve",
        route_metadata=metadata,
        budget_status=budget_status,
        entry_kind="availability_gate",
    )

    blocked = llm_route_state(loop)["blocked"]["planner"][0]
    assert blocked["entry_kind"] == "availability_gate"
    assert blocked["blocked_reason"] == "token_budget_exceeded"
    assert blocked["goal_id"] == "goal-1"
    assert blocked["active_task_id"] == "task-1"
    assert blocked["model_call_ticket_id"] == "ticket-1"
    assert blocked["audit_event_id"] == "audit-1"

    log_row = loop._llm_route_usage_log[0]
    assert log_row["event"] == "blocked"
    assert log_row["route_name"] == "planner"
    advice_row = loop._llm_advice_log[0]
    assert advice_row["entry"] == "llm_route_budget_block"
    assert advice_row["blocked_reason"] == "token_budget_exceeded"


def test_route_accounting_records_failure_policy_and_consumes_budget() -> None:
    loop = _DummyLoop()
    metadata = {
        **_route_metadata(),
        "budget": {"request_budget": 1, "max_retry_count": 0, "escalation_allowed": True},
    }
    failure_policy = {
        "schema_version": "conos.llm.failure_policy/v1",
        "failure_type": "timeout",
        "recommended_action": "return_structured_timeout",
        "fallback_patch_allowed": False,
    }

    record_llm_route_failure(
        loop,
        route_name="patch_proposal",
        method_name="complete",
        prompt_tokens=7,
        reserved_response_tokens=64,
        route_metadata=metadata,
        error="TimeoutError: model timed out",
        failure_policy=failure_policy,
    )

    state = llm_route_state(loop)
    assert state["per_tick_usage"]["patch_proposal"]["request_count"] == 1
    assert state["lifetime_usage"]["patch_proposal"]["request_count"] == 1
    assert state["failures"]["patch_proposal"]["failure_count"] == 1
    assert loop._llm_route_usage_log[-1]["event"] == "failure"
    assert loop._llm_route_usage_log[-1]["failure_policy"]["fallback_patch_allowed"] is False
    assert loop._llm_advice_log[-1]["entry"] == "llm_route_failure"

    blocked_after_failure = llm_route_budget_status(
        loop,
        route_name="patch_proposal",
        route_metadata=metadata,
    )
    assert blocked_after_failure["allowed"] is False
    assert blocked_after_failure["blocked_reason"] == "request_budget_exceeded"


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
            "provider": "codex-cli",
            "model": "gpt-5.5",
            "fallback_route": "general",
            "budget": {"request_budget": 2},
            "metadata": {"primary_model_hint": "gpt-5.3-codex-spark"},
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
    model_selection = usage["route_metadata"]["model_selection"]
    assert model_selection["selected_model"] == "gpt-5.5"
    assert model_selection["selected_provider"] == "codex-cli"
    assert model_selection["primary_model_hint"] == "gpt-5.3-codex-spark"
    assert model_selection["was_provider_limit_failover"] is False
    assert model_selection["selection_reason"] == "profile_route_selected_by_stage_capability"
    assert usage["route_metadata"]["model_selection"]["selected_model"] == "gpt-5.5"


def test_route_budgeted_llm_client_records_timeout_failure_policy_before_reraising() -> None:
    failure_rows = []
    blocked_rows = []
    client = RouteBudgetedLLMClient(
        route_name="patch_proposal",
        client=_TimeoutLLM(),
        route_metadata={
            "requested_route": "patch_proposal",
            "selected_route": "patch_proposal",
            "budget": {"request_budget": 2, "max_retry_count": 0, "escalation_allowed": True},
            "route_context": {"metadata": {"runtime_mode": {"mode": "CREATING"}}},
        },
        preflight_budget_check=lambda **kwargs: {"allowed": True},
        record_usage=lambda **kwargs: None,
        record_failure=lambda **kwargs: failure_rows.append(dict(kwargs)),
        record_blocked=lambda **kwargs: blocked_rows.append(dict(kwargs)),
    )

    try:
        client.complete("draft patch", max_tokens=128, capability_request="patch_proposal.generate")
    except TimeoutError:
        pass
    else:
        raise AssertionError("timeout should be reraised")

    assert not blocked_rows
    assert len(failure_rows) == 1
    policy = failure_rows[0]["failure_policy"]
    assert policy["failure_type"] == "timeout"
    assert policy["recommended_action"] == "return_structured_timeout"
    assert policy["fallback_patch_allowed"] is False
    assert policy["should_escalate"] is True


def test_provider_limit_failure_policy_allows_profile_ranked_model_fallback() -> None:
    decision = decide_llm_failure_policy(
        route_name="patch_proposal",
        failure="RuntimeError: You've hit your usage limit for GPT-5.3-Codex-Spark",
        policy={
            "automatic_model_fallback_allowed": True,
            "fallback_patch_allowed": False,
            "timeout_is_terminal": True,
        },
    )

    assert decision.failure_type == "provider_limit"
    assert decision.recommended_action == "try_profile_ranked_model_after_provider_limit"
    assert decision.fallback_patch_allowed is False
    assert decision.should_downgrade is True


def test_model_profile_failover_selects_next_rank_then_upgrades_from_floor() -> None:
    candidates = [
        {"route_name": "gpt_5_4", "score": 1.2},
        {"route_name": "spark", "score": 1.0},
        {"route_name": "mini", "score": 0.6},
    ]

    next_rank = select_model_profile_failover_candidate(candidates, current_route="spark")
    floor = select_model_profile_failover_candidate(candidates, current_route="mini")

    assert next_rank["to_route"] == "mini"
    assert next_rank["selection_strategy"] == "next_profile_rank_after_provider_limit"
    assert floor["to_route"] == "spark"
    assert floor["selection_strategy"] == "upgrade_one_rank_from_lowest_profile"


def test_route_budgeted_llm_client_fails_over_on_provider_limit() -> None:
    failure_rows = []
    usage_rows = []
    blocked_rows = []
    fallback = _DummyLLM()

    def failover_resolver(**kwargs):
        assert kwargs["failure_type"] == "provider_limit"
        assert kwargs["current_route"] == "spark"
        return {
            "route_name": "gpt_5_4",
            "client": fallback,
            "route_metadata": {
                "requested_route": "patch_proposal",
                "selected_route": "gpt_5_4",
                "provider": "codex-cli",
                "model": "gpt-5.4",
                "budget": {"request_budget": 2},
                "metadata": {"primary_model_hint": "gpt-5.3-codex-spark"},
            },
            "selection": {
                "from_route": "spark",
                "to_route": "gpt_5_4",
                "selection_strategy": "upgrade_one_rank_from_lowest_profile",
            },
        }

    client = RouteBudgetedLLMClient(
        route_name="patch_proposal",
        client=_ProviderLimitLLM(),
        route_metadata={
            "requested_route": "patch_proposal",
            "selected_route": "spark",
            "budget": {"request_budget": 2, "max_retry_count": 0, "escalation_allowed": True},
        },
        preflight_budget_check=lambda **kwargs: {"allowed": True},
        record_usage=lambda **kwargs: usage_rows.append(dict(kwargs)),
        record_failure=lambda **kwargs: failure_rows.append(dict(kwargs)),
        record_blocked=lambda **kwargs: blocked_rows.append(dict(kwargs)),
        model_failover_resolver=failover_resolver,
    )

    assert client.complete("draft patch", max_tokens=128, capability_request="patch_proposal.generate") == "done"

    assert not blocked_rows
    assert len(failure_rows) == 1
    assert failure_rows[0]["failure_policy"]["failure_type"] == "provider_limit"
    assert failure_rows[0]["failure_policy"]["recommended_action"] == "try_profile_ranked_model_after_provider_limit"
    assert len(usage_rows) == 1
    assert usage_rows[0]["route_name"] == "gpt_5_4"
    assert usage_rows[0]["route_metadata"]["model_failover"]["selection_strategy"] == "upgrade_one_rank_from_lowest_profile"
    model_selection = usage_rows[0]["route_metadata"]["model_selection"]
    assert model_selection["selected_model"] == "gpt-5.4"
    assert model_selection["primary_model_hint"] == "gpt-5.3-codex-spark"
    assert model_selection["was_provider_limit_failover"] is True
    assert model_selection["selection_reason"] == "provider_limit_failover:upgrade_one_rank_from_lowest_profile"
    assert client.last_model_failover_trace()[-1]["status"] == "used"
    assert fallback.calls[0][0] == "complete"


def test_route_budgeted_llm_client_applies_runtime_policy_call_defaults() -> None:
    usage_rows = []
    llm = _DummyLLM()
    client = RouteBudgetedLLMClient(
        route_name="structured_answer",
        client=llm,
        route_metadata={
            "requested_route": "structured_answer",
            "selected_route": "ollama_json",
            "runtime_policy": {
                "route_name": "structured_answer",
                "call_defaults": {
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "think": False,
                    "thinking_budget": 0,
                    "timeout_sec": 8.0,
                },
                "budget": {"request_budget": 8},
            },
            "budget": {"request_budget": 8},
        },
        preflight_budget_check=lambda **kwargs: {"allowed": True},
        record_usage=lambda **kwargs: usage_rows.append(dict(kwargs)),
        record_blocked=lambda **kwargs: None,
    )

    assert client.complete("{}", capability_request="structured_answer.kwargs") == "done"

    kwargs = llm.calls[0][2]
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.0
    assert kwargs["think"] is False
    assert kwargs["thinking_budget"] == 0
    assert kwargs["timeout_sec"] == 8.0
    assert usage_rows[0]["reserved_response_tokens"] == 256


def test_route_runtime_policy_is_shaped_by_runtime_mode() -> None:
    sleep = route_runtime_policy_for_route("structured_answer", runtime_mode="SLEEP")
    creating = route_runtime_policy_for_route("patch_proposal", runtime_mode="CREATING")
    planning = route_runtime_policy_for_route("planning")

    assert sleep["budget"]["request_budget"] == 0
    assert sleep["call_defaults"]["max_tokens"] == 0
    assert sleep["call_defaults"]["think"] is False
    assert creating["runtime_mode"] == "CREATING"
    assert creating["budget"]["request_budget"] <= 2
    assert creating["model_selection"]["prefer_strongest_model"] is True
    assert creating["call_defaults"]["max_tokens"] <= 1500
    assert planning["runtime_mode"] == "DEEP_THINK"
    assert planning["call_defaults"]["prefer_strongest_model"] is True


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


def test_routing_facade_resolves_capabilities_and_caches_wrapped_clients() -> None:
    loop = _RoutingLoop()

    router = initialize_model_router(loop)
    assert ensure_model_router(loop) is router

    registry = ensure_llm_capability_registry(loop)
    assert ensure_llm_capability_registry(loop) is registry
    resolution = resolve_llm_capability_spec(
        loop,
        "retrieval.query_rewrite",
        fallback_route="general",
    )
    assert resolution["route_name"] == "retrieval"
    assert resolution["required_capabilities"] == ["retrieval", "grounding"]
    assert resolution["policy_source"] == "test_capability"

    client = resolve_llm_client(
        loop,
        "general",
        capability_request="retrieval.query_rewrite",
    )
    assert isinstance(client, RouteBudgetedLLMClient)
    assert loop.route_context_calls[-1]["route_name"] == "retrieval"
    assert loop.route_context_calls[-1]["capability_request"] == "retrieval.query_rewrite"

    cached = resolve_llm_client(
        loop,
        "general",
        capability_request="retrieval.query_rewrite",
    )
    assert cached is client

    assert client.complete("hello", max_tokens=8, capability_request="retrieval.query_rewrite") == "done"
    assert loop._llm_client.calls[-1][0] == "complete"
    assert llm_route_state(loop)["lifetime_usage"]["retrieval"]["request_count"] == 1


def test_routing_facade_gateway_uses_capability_prefix_and_budget_gate() -> None:
    loop = _RoutingLoop()

    gateway = resolve_llm_gateway(loop, "retrieval", capability_prefix="retrieval")
    assert resolve_llm_gateway(loop, "retrieval", capability_prefix="retrieval") is gateway
    assert gateway.route_name == "retrieval"
    assert gateway.capability_prefix == "retrieval"

    assert gateway.request_text("query_rewrite", "hello", max_tokens=4) == "done"
    assert loop._llm_client.calls[-1][0] == "complete"
    assert loop._llm_route_usage_log[-1]["event"] == "request"
    assert loop._llm_route_usage_log[-1]["route_name"] == "retrieval"

    # The route has a per-tick request budget of 2; the third call is blocked
    # before it reaches the underlying client.
    assert gateway.request_text("query_rewrite", "second", max_tokens=4) == "done"
    call_count = len(loop._llm_client.calls)
    assert gateway.request_text("query_rewrite", "blocked", max_tokens=4) == ""
    assert len(loop._llm_client.calls) == call_count
    assert loop._llm_route_usage_log[-1]["event"] == "blocked"
    assert loop._llm_route_usage_log[-1]["blocked_reason"] == "request_budget_exceeded"

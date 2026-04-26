from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.llm.budget import (
    LLMCostLedger,
    LLMRuntimeBudget,
    amplification_efficiency,
    classify_llm_layer,
    wrap_with_budget,
)
from integrations.local_machine.budget_policy import budget_policy_report


class _Client:
    model = "small-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_raw(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return "ok"


class _StrictClient:
    model = "strict-test-model"

    def __init__(self) -> None:
        self.timeout_seen: float | None = None

    def complete_raw(self, prompt: str, *, timeout_sec: float | None = None) -> str:
        self.timeout_seen = timeout_sec
        return prompt.upper()


def test_classify_llm_layer_keeps_formatting_routes_small_and_patch_routes_strong() -> None:
    assert classify_llm_layer("structured_answer") == "small_model"
    assert classify_llm_layer("retrieval") == "small_model"
    assert classify_llm_layer("patch_proposal") == "strong_model"
    assert classify_llm_layer("planning") == "strong_model"


def test_budget_wrapper_records_calls_and_blocks_over_budget() -> None:
    ledger = LLMCostLedger(LLMRuntimeBudget(max_llm_calls=1, max_prompt_tokens=100, max_completion_tokens=64))
    client = wrap_with_budget(_Client(), ledger)

    assert client.complete_raw("hello", max_tokens=32, capability_route_name="structured_answer") == "ok"
    with pytest.raises(RuntimeError, match="max_llm_calls_exceeded"):
        client.complete_raw("hello again", max_tokens=32, capability_route_name="structured_answer")

    summary = ledger.summary()
    assert summary["total_calls"] == 1
    assert summary["by_layer"]["small_model"]["calls"] == 1
    assert summary["requested_completion_tokens"] == 32


def test_budget_wrapper_sanitizes_kwargs_for_strict_clients() -> None:
    ledger = LLMCostLedger(LLMRuntimeBudget(max_llm_calls=2))
    strict = _StrictClient()
    client = wrap_with_budget(strict, ledger)

    assert client.complete_raw(
        "ok",
        timeout_sec=3.0,
        max_tokens=16,
        capability_route_name="structured_answer",
        unsupported_flag=True,
    ) == "OK"

    assert strict.timeout_seen == 3.0
    assert ledger.summary()["total_calls"] == 1


def test_budget_wrapper_blocks_strong_model_when_escalation_is_disabled() -> None:
    ledger = LLMCostLedger(LLMRuntimeBudget(max_llm_calls=2, escalation_allowed=False))
    client = wrap_with_budget(_Client(), ledger)

    assert client.complete_raw("cheap", max_tokens=16, capability_route_name="structured_answer") == "ok"
    with pytest.raises(RuntimeError, match="strong_model_escalation_not_allowed"):
        client.complete_raw("hard", max_tokens=16, capability_route_name="patch_proposal")

    assert ledger.summary()["total_calls"] == 1


def test_budget_wrapper_reads_route_from_capability_request_object() -> None:
    ledger = LLMCostLedger(LLMRuntimeBudget(max_llm_calls=2))
    client = wrap_with_budget(_Client(), ledger)

    assert client.complete_raw(
        "hard",
        max_tokens=16,
        capability_request=SimpleNamespace(route_name="patch_proposal"),
    ) == "ok"

    summary = ledger.summary()
    assert summary["by_route"]["patch_proposal"]["calls"] == 1
    assert summary["by_layer"]["strong_model"]["calls"] == 1


def test_amplification_efficiency_handles_normal_and_zero_baseline_cases() -> None:
    score = amplification_efficiency(
        verified_success_rate_os=0.9,
        verified_success_rate_baseline=0.6,
        cost_os=300.0,
        cost_baseline=100.0,
    )
    assert round(float(score["amplification_efficiency"]), 6) == 0.5

    undefined = amplification_efficiency(
        verified_success_rate_os=0.9,
        verified_success_rate_baseline=0.0,
        cost_os=300.0,
        cost_baseline=100.0,
    )
    assert undefined["amplification_efficiency"] is None
    assert undefined["undefined_reason"] == "zero_baseline_success_or_nonpositive_cost"


def test_budget_policy_does_not_escalate_after_verified_completion() -> None:
    report = budget_policy_report(
        {
            "terminal_state": "completed_verified",
            "verified_completion": True,
            "target_binding": {"top_target_file": "app.py", "target_confidence": 0.9},
            "validation_runs": [{"success": False}],
            "read_files": [{"path": "app.py"}],
            "action_history": [
                {"function_name": "no_op_complete"},
                {"function_name": "no_op_complete"},
                {"function_name": "no_op_complete"},
                {"function_name": "no_op_complete"},
            ],
        }
    )

    assert report["selected_path_hint"] == "terminal_complete"
    assert report["fast_path"]["eligible"] is False
    assert report["escalation_path"]["recommended"] is False

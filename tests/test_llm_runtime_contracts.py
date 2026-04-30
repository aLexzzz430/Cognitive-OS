from __future__ import annotations

import json

from modules.llm import (
    AuthProfile,
    CostPolicy,
    ExecutionRuntime,
    build_llm_runtime_plan,
)
from modules.llm.codex_cli_client import DEFAULT_CODEX_MODEL, CodexCliClient
from modules.llm.factory import build_llm_client
from modules.llm.budget import BudgetAwareLLMClient
from modules.llm.reliability_adapter import (
    LLMReliabilityPolicy,
    normalize_reliable_llm_output,
)
from modules.llm.failure_policy import decide_llm_failure_policy, failure_policy_catalog
from core.orchestration.structured_answer import StructuredAnswerSynthesizer


def test_codex_runtime_plan_separates_provider_auth_runtime_and_policies() -> None:
    plan = build_llm_runtime_plan("codex-cli", model="gpt-5.3-codex")
    payload = plan.to_dict()

    assert payload["provider"]["provider"] == "codex-cli"
    assert payload["provider"]["quota_scope"] == "chatgpt_codex_plan_or_api_org_via_codex_cli"
    assert payload["auth_profile"]["auth_type"] == "chatgpt_oauth_delegate"
    assert payload["auth_profile"]["direct_token_access"] is False
    assert payload["auth_profile"]["login_command"] == ["codex", "login"]
    assert payload["execution_runtime"]["runtime_type"] == "local_cli_agent"
    assert payload["tool_adapter"]["adapter_type"] == "codex_exec"
    assert payload["cost_policy"]["policy_id"] == "default"
    assert payload["context_policy"]["prompt_retention"] == "metadata_only"
    assert payload["verifier_policy"]["failure_mode"] == "return_structured_failure"


def test_codex_runtime_plan_defaults_to_spark_model() -> None:
    plan = build_llm_runtime_plan("codex-cli")

    assert plan.provider.model == "gpt-5.3-codex-spark"
    assert plan.provider.model == DEFAULT_CODEX_MODEL


def test_factory_applies_runtime_contract_to_codex_client(tmp_path) -> None:
    client = build_llm_client(
        "codex-cli",
        model="gpt-5.3-codex",
        execution_runtime=ExecutionRuntime(
            runtime_id="custom_codex",
            runtime_type="local_cli_agent",
            command="codex-test",
            cwd=str(tmp_path),
            sandbox="read-only",
            timeout_sec=333,
            local_credentials_allowed=True,
        ),
    )

    assert isinstance(client, CodexCliClient)
    assert client.model == "gpt-5.3-codex"
    runtime = client.execution_runtime()
    assert runtime["runtime_id"] == "custom_codex"
    assert runtime["command"] == "codex-test"
    assert runtime["cwd"] == str(tmp_path)
    assert client.auth_profile()["auth_type"] == "chatgpt_oauth_delegate"


def test_factory_wraps_client_when_cost_policy_has_limits() -> None:
    client = build_llm_client(
        "codex-cli",
        model="gpt-5.3-codex",
        auth_profile=AuthProfile(
            profile_id="codex_cli_chatgpt_oauth",
            provider="codex-cli",
            auth_type="chatgpt_oauth_delegate",
            credential_source="codex_cli_local_credentials",
            requires_user_login=True,
            direct_token_access=False,
        ),
        cost_policy=CostPolicy(max_llm_calls=1),
    )

    assert isinstance(client, BudgetAwareLLMClient)
    assert client.model == "gpt-5.3-codex"
    assert client._conos_llm_runtime_plan["cost_policy"]["max_llm_calls"] == 1


def test_llm_policy_cli_reports_unified_policy_surface(tmp_path, capsys) -> None:
    from modules.llm.cli import main

    assert main(
        [
            "--provider",
            "all",
            "policy",
            "--store",
            str(tmp_path / "profiles.json"),
            "--route-policy-file",
            str(tmp_path / "routes.json"),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "conos.llm.product_policy/v1"
    assert payload["policy_layers"] == [
        "Provider",
        "AuthProfile",
        "ExecutionRuntime",
        "ToolAdapter",
        "CostPolicy",
        "ContextPolicy",
        "VerifierPolicy",
    ]
    assert set(payload["runtime_plans"]) == {"ollama", "openai", "codex-cli"}
    assert payload["runtime_plans"]["codex-cli"]["auth_profile"]["auth_type"] == "chatgpt_oauth_delegate"
    assert payload["failure_policy"]["timeout"].startswith("return timeout failure by default")
    assert payload["failure_policy"]["fallback_patch"].startswith("disabled by default")


def test_llm_reliability_adapter_rejects_missing_kwargs_and_duplicate_actions() -> None:
    schema = {"parameters": {"required": ["url"]}}
    missing = normalize_reliable_llm_output(
        'KWARGS_JSON: {"filename": "x.json"}',
        policy=LLMReliabilityPolicy(output_kind="action_kwargs", expected_type="dict"),
        expected_prefixes=("KWARGS_JSON:",),
        function_name="internet_fetch",
        function_schema=schema,
    )
    duplicate = normalize_reliable_llm_output(
        'KWARGS_JSON: {"path": "README.md", "start_line": 1, "end_line": 20}',
        policy=LLMReliabilityPolicy(output_kind="action_kwargs", expected_type="dict"),
        expected_prefixes=("KWARGS_JSON:",),
        function_name="file_read",
        recent_actions=[
            {"function_name": "file_read", "kwargs": {"path": "README.md", "start_line": 1, "end_line": 20}}
        ],
    )

    assert missing.ok is False
    assert missing.status == "invalid_kwargs"
    assert missing.missing_fields == ["url"]
    assert duplicate.ok is False
    assert duplicate.status == "duplicate_action"
    assert duplicate.duplicate_action is True


def test_llm_reliability_adapter_marks_timeout_without_fallback() -> None:
    result = normalize_reliable_llm_output(
        "",
        policy=LLMReliabilityPolicy(
            output_kind="patch_proposal",
            expected_type="dict",
            fallback_on_timeout_allowed=False,
        ),
        timeout_error="TimeoutError: remote model timed out after 60s",
    )

    assert result.ok is False
    assert result.status == "timeout"
    assert result.timeout is True
    assert result.fallback_allowed is False
    assert result.should_escalate is True


def test_llm_failure_policy_timeout_is_structured_and_has_no_patch_fallback() -> None:
    decision = decide_llm_failure_policy(
        route_name="patch_proposal",
        failure="TimeoutError: model timed out after 60s",
        route_metadata={"route_context": {"metadata": {"runtime_mode": {"mode": "CREATING"}}}},
        budget={"escalation_allowed": True, "max_retry_count": 1},
    )

    assert decision.failure_type == "timeout"
    assert decision.recommended_action == "return_structured_timeout"
    assert decision.terminal is True
    assert decision.should_escalate is True
    assert decision.fallback_patch_allowed is False
    assert decision.audit_event["event_type"] == "llm_failure_policy_decision"


def test_llm_failure_policy_blocks_escalation_in_sleep_mode() -> None:
    decision = decide_llm_failure_policy(
        route_name="root_cause",
        failure="RuntimeError: invalid json",
        status="format_error",
        route_metadata={"route_context": {"metadata": {"runtime_mode": {"mode": "SLEEP"}}}},
        budget={"escalation_allowed": True, "max_retry_count": 0},
    )

    assert decision.should_escalate is False
    assert decision.recommended_action == "return_structured_failure"
    assert "runtime_mode_blocks_llm_recovery:SLEEP" in decision.reason


def test_failure_policy_catalog_is_exposed_in_product_policy() -> None:
    catalog = failure_policy_catalog()

    assert catalog["schema_version"] == "conos.llm.failure_policy/v1"
    assert catalog["decisions"]["timeout"]["fallback_patch_allowed"] is False


def test_structured_answer_llm_trace_includes_reliability_adapter_for_bad_kwargs() -> None:
    class FakeLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            return 'KWARGS_JSON: {"filename": "x.json"}'

    synthesizer = StructuredAnswerSynthesizer()
    obs = {
        "local_mirror": {
            "instruction": "fetch a public URL",
            "deterministic_fallback_enabled": False,
            "internet_enabled": True,
        },
        "function_signatures": {
            "internet_fetch": {"parameters": {"required": ["url"]}},
        },
    }

    kwargs, meta = synthesizer._draft_with_llm_with_trace("internet_fetch", obs, FakeLLM())

    trace = meta["llm_trace"][0]
    assert kwargs == {}
    assert trace["parsed_kwargs"] == {}
    assert trace["reliability_adapter"]["status"] == "invalid_kwargs"
    assert trace["reliability_adapter"]["missing_fields"] == ["url"]

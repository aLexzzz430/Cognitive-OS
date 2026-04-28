from __future__ import annotations

from modules.llm import (
    AuthProfile,
    CostPolicy,
    ExecutionRuntime,
    build_llm_runtime_plan,
)
from modules.llm.codex_cli_client import CodexCliClient
from modules.llm.factory import build_llm_client
from modules.llm.budget import BudgetAwareLLMClient


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

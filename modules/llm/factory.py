from __future__ import annotations

from typing import Any, Optional

from modules.llm.minimax_client import MinimaxClient
from modules.llm.codex_cli_client import CodexCliClient
from modules.llm.ollama_client import OllamaClient
from modules.llm.openai_client import OpenAIClient
from modules.llm.budget import LLMRuntimeBudget, LLMCostLedger, wrap_with_budget
from modules.llm.runtime_contracts import build_llm_runtime_plan, has_cost_limits


def build_llm_client(
    provider: str = "none",
    *,
    token_file: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 60.0,
    provider_spec: Any = None,
    auth_profile: Any = None,
    execution_runtime: Any = None,
    tool_adapter: Any = None,
    cost_policy: Any = None,
    context_policy: Any = None,
    verifier_policy: Any = None,
) -> Any:
    normalized = (provider or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return None
    runtime_plan = build_llm_runtime_plan(
        normalized,
        model=model or "",
        base_url=base_url or "",
        timeout_sec=timeout_sec,
        provider_spec=provider_spec,
        auth_profile=auth_profile,
        execution_runtime=execution_runtime,
        tool_adapter=tool_adapter,
        cost_policy=cost_policy,
        context_policy=context_policy,
        verifier_policy=verifier_policy,
    )
    runtime = runtime_plan.execution_runtime
    if normalized == "minimax":
        client = MinimaxClient(token_file=token_file)
        return _finalize_client(client, runtime_plan)
    if normalized in {"ollama", "local", "local-http"}:
        client = OllamaClient(
            base_url=runtime.base_url or base_url,
            model=runtime_plan.provider.model or model,
            timeout_sec=float(runtime.timeout_sec or timeout_sec),
        )
        return _finalize_client(client, runtime_plan)
    if normalized in {"openai", "responses"}:
        client = OpenAIClient(
            base_url=runtime.base_url or base_url,
            model=runtime_plan.provider.model or model,
            timeout_sec=float(runtime.timeout_sec or timeout_sec),
        )
        return _finalize_client(client, runtime_plan)
    if normalized in {"codex", "codex-cli", "openai-oauth-codex"}:
        client = CodexCliClient(
            model=runtime_plan.provider.model or model,
            command=runtime.command or None,
            cwd=runtime.cwd or None,
            sandbox=runtime.sandbox or None,
            timeout_sec=float(runtime.timeout_sec or timeout_sec),
            runtime_plan=runtime_plan.to_dict(),
        )
        return _finalize_client(client, runtime_plan)
    raise ValueError(f"Unsupported llm provider: {provider}")


def _finalize_client(client: Any, runtime_plan: Any) -> Any:
    try:
        setattr(client, "_conos_llm_runtime_plan", runtime_plan.to_dict())
    except Exception:
        pass
    if has_cost_limits(runtime_plan.cost_policy):
        budget = LLMRuntimeBudget.from_mapping(runtime_plan.cost_policy.to_dict())
        wrapped = wrap_with_budget(client, LLMCostLedger(budget))
        try:
            setattr(wrapped, "_conos_llm_runtime_plan", runtime_plan.to_dict())
        except Exception:
            pass
        return wrapped
    return client

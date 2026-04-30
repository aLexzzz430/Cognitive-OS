from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from modules.llm.model_profile import (
    build_model_route_summary,
    default_model_profile_store_path,
    default_model_route_policy_path,
    load_profile_backed_route_policies,
)
from modules.llm.runtime_contracts import build_llm_runtime_plan
from modules.llm.failure_policy import failure_policy_catalog


LLM_PRODUCT_POLICY_VERSION = "conos.llm.product_policy/v1"
SUPPORTED_PRODUCT_PROVIDERS = ("ollama", "openai", "codex-cli")


def _provider_name(provider: str) -> str:
    normalized = str(provider or "ollama").strip().lower() or "ollama"
    if normalized == "codex":
        return "codex-cli"
    return normalized


def _policy_path_report(path: str | Path | None, *, default_path: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser() if path else default_path.expanduser()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
    }


def provider_policy_report(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Return the product-level runtime contract for one provider."""

    normalized = _provider_name(provider)
    return build_llm_runtime_plan(
        normalized,
        model=model,
        base_url=base_url,
        timeout_sec=float(timeout_sec or 60.0),
    ).to_dict()


def build_llm_product_policy_report(
    *,
    provider: str = "all",
    model: str = "",
    base_url: str = "",
    openai_base_url: str = "",
    timeout_sec: float = 60.0,
    store_path: str | Path | None = None,
    route_policy_path: str | Path | None = None,
    explain_routes: bool = False,
    routes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Describe the unified model profile and policy surface without live calls."""

    selected_provider = _provider_name(provider)
    providers = (
        SUPPORTED_PRODUCT_PROVIDERS
        if selected_provider == "all"
        else tuple([selected_provider])
    )
    runtime_plans = {
        name: provider_policy_report(
            name,
            model=model,
            base_url=(openai_base_url if name == "openai" else base_url),
            timeout_sec=timeout_sec,
        )
        for name in providers
    }
    route_policies = load_profile_backed_route_policies(
        store_path=store_path,
        route_policy_path=route_policy_path,
        base_url=base_url or None,
    )
    route_summary = build_model_route_summary(
        route_policies,
        routes=routes,
        explain=bool(explain_routes),
    )
    profile_path = _policy_path_report(
        store_path,
        default_path=default_model_profile_store_path(),
    )
    route_path = _policy_path_report(
        route_policy_path,
        default_path=default_model_route_policy_path(),
    )
    return {
        "schema_version": LLM_PRODUCT_POLICY_VERSION,
        "provider": selected_provider,
        "supported_providers": list(SUPPORTED_PRODUCT_PROVIDERS),
        "policy_layers": [
            "Provider",
            "AuthProfile",
            "ExecutionRuntime",
            "ToolAdapter",
            "CostPolicy",
            "ContextPolicy",
            "VerifierPolicy",
        ],
        "runtime_plans": runtime_plans,
        "model_profile": {
            "store": profile_path,
            "route_policy": route_path,
            "route_policy_count": len(route_policies),
            "route_policy_names": sorted(route_policies.keys()),
        },
        "route_summary": route_summary,
        "failure_policy": {
            "model_unavailable": "try route fallback when configured; otherwise return structured failure",
            "timeout": "return timeout failure by default; fallback/escalation only when explicitly configured and audited",
            "format_error": "normalize through output and reliability adapters, then retry fallback/escalation only when configured",
            "invalid_kwargs": "reject before tool execution; action grounding may repair only through audited schema policy",
            "duplicate_action": "reject repeated action signatures before spending additional execution budget",
            "fallback_patch": "disabled by default; verifier-gated deterministic patch fallback must be explicitly enabled",
            "budget_exceeded": "prefer deterministic verifier/action policy and return structured blocked result",
        },
        "failure_policy_catalog": failure_policy_catalog(),
        "context_policy": {
            "default_retention": "metadata_only",
            "raw_prompt_persistence": "disabled_by_default",
            "raw_output_persistence": "disabled_by_default",
            "pruning_strategy": "budgeted_relevant_context",
        },
        "cost_policy": {
            "default_budget_shape": {
                "max_llm_calls": "task_or_route_budget",
                "max_prompt_tokens": "task_or_route_budget",
                "max_completion_tokens": "route_budget",
                "max_wall_clock_seconds": "task_budget",
                "max_retry_count": 1,
                "escalation_allowed": True,
            }
        },
    }


def policy_report_brief(payload: Mapping[str, Any]) -> Dict[str, Any]:
    policies = dict(dict(payload or {}).get("model_profile", {}) or {})
    return {
        "schema_version": str(dict(payload or {}).get("schema_version", "") or LLM_PRODUCT_POLICY_VERSION),
        "provider": str(dict(payload or {}).get("provider", "") or ""),
        "supported_providers": list(dict(payload or {}).get("supported_providers", []) or []),
        "route_policy_count": int(policies.get("route_policy_count", 0) or 0),
        "policy_layers": list(dict(payload or {}).get("policy_layers", []) or []),
    }

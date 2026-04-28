from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence


LLM_RUNTIME_CONTRACT_VERSION = "conos.llm.runtime_contract/v1"


def _clean(value: Any, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    return text if text else default


def _mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload or {}) if isinstance(payload, Mapping) else {}
    try:
        return asdict(value)
    except TypeError:
        return {}


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0.0 else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    family: str
    model: str = ""
    base_url: str = ""
    transport: str = ""
    quota_scope: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class AuthProfile:
    profile_id: str
    provider: str
    auth_type: str
    credential_source: str
    requires_user_login: bool = False
    login_command: list[str] = field(default_factory=list)
    status_command: list[str] = field(default_factory=list)
    logout_command: list[str] = field(default_factory=list)
    token_storage: str = ""
    direct_token_access: bool = False
    quota_scope: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class ExecutionRuntime:
    runtime_id: str
    runtime_type: str
    command: str = ""
    base_url: str = ""
    cwd: str = ""
    sandbox: str = ""
    timeout_sec: float = 60.0
    supports_stateful_sessions: bool = False
    local_credentials_allowed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class ToolAdapterSpec:
    adapter_id: str
    adapter_type: str
    allowed_tools: list[str] = field(default_factory=list)
    structured_output_mode: str = ""
    side_effect_policy: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class CostPolicy:
    policy_id: str = "default"
    max_llm_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_wall_clock_seconds: float | None = None
    max_retry_count: int | None = None
    escalation_allowed: bool = True
    prefer_low_cost: float = 0.0
    charge_scope: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class ContextPolicy:
    policy_id: str = "default"
    max_context_tokens: int | None = None
    pruning_strategy: str = "budgeted_relevant_context"
    prompt_retention: str = "metadata_only"
    output_retention: str = "metadata_only"
    allow_raw_prompt_persistence: bool = False
    allow_raw_output_persistence: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class VerifierPolicy:
    policy_id: str = "default"
    require_schema_validation: bool = True
    require_independent_verifier: bool = False
    verifier_route: str = "verifier"
    verifier_model: str = ""
    timeout_sec: float | None = None
    failure_mode: str = "return_structured_failure"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_RUNTIME_CONTRACT_VERSION
        return payload


@dataclass(frozen=True)
class LLMRuntimePlan:
    provider: ProviderSpec
    auth_profile: AuthProfile
    execution_runtime: ExecutionRuntime
    tool_adapter: ToolAdapterSpec
    cost_policy: CostPolicy
    context_policy: ContextPolicy
    verifier_policy: VerifierPolicy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": LLM_RUNTIME_CONTRACT_VERSION,
            "provider": self.provider.to_dict(),
            "auth_profile": self.auth_profile.to_dict(),
            "execution_runtime": self.execution_runtime.to_dict(),
            "tool_adapter": self.tool_adapter.to_dict(),
            "cost_policy": self.cost_policy.to_dict(),
            "context_policy": self.context_policy.to_dict(),
            "verifier_policy": self.verifier_policy.to_dict(),
            "metadata": dict(self.metadata),
        }
        return payload


def _provider_family(provider: str) -> str:
    normalized = _clean(provider, "none").lower()
    if normalized in {"codex", "codex-cli", "openai-oauth-codex"}:
        return "openai_codex"
    if normalized in {"openai", "responses"}:
        return "openai_api"
    if normalized in {"ollama", "local", "local-http"}:
        return "local_ollama"
    if normalized == "minimax":
        return "minimax"
    return normalized


def default_provider_spec(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
) -> ProviderSpec:
    normalized = _clean(provider, "none").lower()
    family = _provider_family(normalized)
    if family == "openai_codex":
        return ProviderSpec(
            provider="codex-cli",
            family=family,
            model=_clean(model, "gpt-5.3-codex"),
            transport="local_cli",
            quota_scope="chatgpt_codex_plan_or_api_org_via_codex_cli",
        )
    if family == "openai_api":
        return ProviderSpec(
            provider="openai",
            family=family,
            model=_clean(model),
            base_url=_clean(base_url, "https://api.openai.com/v1"),
            transport="https_responses_api",
            quota_scope="api_organization",
        )
    if family == "local_ollama":
        return ProviderSpec(
            provider="ollama",
            family=family,
            model=_clean(model),
            base_url=_clean(base_url, "http://127.0.0.1:11434"),
            transport="local_http",
            quota_scope="local_compute",
        )
    return ProviderSpec(provider=normalized, family=family, model=_clean(model), base_url=_clean(base_url))


def default_auth_profile(provider: str) -> AuthProfile:
    family = _provider_family(provider)
    if family == "openai_codex":
        return AuthProfile(
            profile_id="codex_cli_chatgpt_oauth",
            provider="codex-cli",
            auth_type="chatgpt_oauth_delegate",
            credential_source="codex_cli_local_credentials",
            requires_user_login=True,
            login_command=["codex", "login"],
            status_command=["codex", "login", "status"],
            logout_command=["codex", "logout"],
            token_storage="managed_by_codex_cli",
            direct_token_access=False,
            quota_scope="chatgpt_codex_plan_or_api_org_via_codex_cli",
            metadata={
                "note": "Con OS delegates ChatGPT OAuth to the official Codex CLI and does not read OAuth tokens.",
            },
        )
    if family == "openai_api":
        return AuthProfile(
            profile_id="openai_api_key",
            provider="openai",
            auth_type="api_key",
            credential_source="OPENAI_API_KEY",
            direct_token_access=False,
            quota_scope="api_organization",
        )
    if family == "local_ollama":
        return AuthProfile(
            profile_id="ollama_no_auth",
            provider="ollama",
            auth_type="none",
            credential_source="local_service",
            quota_scope="local_compute",
        )
    return AuthProfile(
        profile_id=f"{_clean(provider, 'unknown')}_default_auth",
        provider=_clean(provider, "unknown"),
        auth_type="unknown",
        credential_source="",
    )


def default_execution_runtime(
    provider: str,
    *,
    base_url: str = "",
    timeout_sec: float = 60.0,
) -> ExecutionRuntime:
    family = _provider_family(provider)
    if family == "openai_codex":
        codex_timeout = max(300.0, float(timeout_sec or 300.0))
        return ExecutionRuntime(
            runtime_id="codex_cli_exec",
            runtime_type="local_cli_agent",
            command="codex",
            sandbox="read-only",
            timeout_sec=codex_timeout,
            supports_stateful_sessions=False,
            local_credentials_allowed=True,
            metadata={"auth_boundary": "codex_cli"},
        )
    if family == "openai_api":
        return ExecutionRuntime(
            runtime_id="openai_responses_api",
            runtime_type="remote_api",
            base_url=_clean(base_url, "https://api.openai.com/v1"),
            timeout_sec=float(timeout_sec or 60.0),
        )
    if family == "local_ollama":
        return ExecutionRuntime(
            runtime_id="ollama_generate_api",
            runtime_type="local_http",
            base_url=_clean(base_url, "http://127.0.0.1:11434"),
            timeout_sec=float(timeout_sec or 60.0),
        )
    return ExecutionRuntime(runtime_id="unknown_runtime", runtime_type="unknown", timeout_sec=float(timeout_sec or 60.0))


def default_tool_adapter(provider: str) -> ToolAdapterSpec:
    family = _provider_family(provider)
    if family == "openai_codex":
        return ToolAdapterSpec(
            adapter_id="codex_cli_text_adapter",
            adapter_type="codex_exec",
            allowed_tools=[],
            structured_output_mode="last_message_file",
            side_effect_policy="read_only_sandbox_and_prompt_bound",
            metadata={"execution_mode": "bounded_llm_backend"},
        )
    if family == "openai_api":
        return ToolAdapterSpec(
            adapter_id="openai_responses_text_adapter",
            adapter_type="responses_api",
            structured_output_mode="output_text",
            side_effect_policy="no_tools_by_default",
        )
    if family == "local_ollama":
        return ToolAdapterSpec(
            adapter_id="ollama_generate_text_adapter",
            adapter_type="ollama_generate",
            structured_output_mode="text",
            side_effect_policy="no_tools",
        )
    return ToolAdapterSpec(adapter_id="unknown_adapter", adapter_type="unknown")


def coerce_cost_policy(value: Any) -> CostPolicy:
    payload = _mapping(value)
    if isinstance(value, CostPolicy):
        return value
    return CostPolicy(
        policy_id=_clean(payload.get("policy_id"), "default"),
        max_llm_calls=_int_or_none(payload.get("max_llm_calls")),
        max_prompt_tokens=_int_or_none(payload.get("max_prompt_tokens")),
        max_completion_tokens=_int_or_none(payload.get("max_completion_tokens")),
        max_wall_clock_seconds=_float_or_none(payload.get("max_wall_clock_seconds")),
        max_retry_count=_int_or_none(payload.get("max_retry_count")),
        escalation_allowed=bool(payload.get("escalation_allowed", True)),
        prefer_low_cost=float(payload.get("prefer_low_cost", 0.0) or 0.0),
        charge_scope=_clean(payload.get("charge_scope")),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def coerce_context_policy(value: Any) -> ContextPolicy:
    payload = _mapping(value)
    if isinstance(value, ContextPolicy):
        return value
    return ContextPolicy(
        policy_id=_clean(payload.get("policy_id"), "default"),
        max_context_tokens=_int_or_none(payload.get("max_context_tokens")),
        pruning_strategy=_clean(payload.get("pruning_strategy"), "budgeted_relevant_context"),
        prompt_retention=_clean(payload.get("prompt_retention"), "metadata_only"),
        output_retention=_clean(payload.get("output_retention"), "metadata_only"),
        allow_raw_prompt_persistence=bool(payload.get("allow_raw_prompt_persistence", False)),
        allow_raw_output_persistence=bool(payload.get("allow_raw_output_persistence", False)),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def coerce_verifier_policy(value: Any) -> VerifierPolicy:
    payload = _mapping(value)
    if isinstance(value, VerifierPolicy):
        return value
    return VerifierPolicy(
        policy_id=_clean(payload.get("policy_id"), "default"),
        require_schema_validation=bool(payload.get("require_schema_validation", True)),
        require_independent_verifier=bool(payload.get("require_independent_verifier", False)),
        verifier_route=_clean(payload.get("verifier_route"), "verifier"),
        verifier_model=_clean(payload.get("verifier_model")),
        timeout_sec=_float_or_none(payload.get("timeout_sec")),
        failure_mode=_clean(payload.get("failure_mode"), "return_structured_failure"),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def _coerce_provider_spec(value: Any, default: ProviderSpec) -> ProviderSpec:
    payload = _mapping(value)
    if isinstance(value, ProviderSpec):
        return value
    if not payload:
        return default
    return replace(
        default,
        provider=_clean(payload.get("provider"), default.provider),
        family=_clean(payload.get("family"), default.family),
        model=_clean(payload.get("model"), default.model),
        base_url=_clean(payload.get("base_url"), default.base_url),
        transport=_clean(payload.get("transport"), default.transport),
        quota_scope=_clean(payload.get("quota_scope"), default.quota_scope),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def _coerce_auth_profile(value: Any, default: AuthProfile) -> AuthProfile:
    payload = _mapping(value)
    if isinstance(value, AuthProfile):
        return value
    if not payload:
        return default
    return replace(
        default,
        profile_id=_clean(payload.get("profile_id"), default.profile_id),
        provider=_clean(payload.get("provider"), default.provider),
        auth_type=_clean(payload.get("auth_type"), default.auth_type),
        credential_source=_clean(payload.get("credential_source"), default.credential_source),
        requires_user_login=bool(payload.get("requires_user_login", default.requires_user_login)),
        login_command=_string_list(payload.get("login_command")) or list(default.login_command),
        status_command=_string_list(payload.get("status_command")) or list(default.status_command),
        logout_command=_string_list(payload.get("logout_command")) or list(default.logout_command),
        token_storage=_clean(payload.get("token_storage"), default.token_storage),
        direct_token_access=bool(payload.get("direct_token_access", default.direct_token_access)),
        quota_scope=_clean(payload.get("quota_scope"), default.quota_scope),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def _coerce_execution_runtime(value: Any, default: ExecutionRuntime) -> ExecutionRuntime:
    payload = _mapping(value)
    if isinstance(value, ExecutionRuntime):
        return value
    if not payload:
        return default
    return replace(
        default,
        runtime_id=_clean(payload.get("runtime_id"), default.runtime_id),
        runtime_type=_clean(payload.get("runtime_type"), default.runtime_type),
        command=_clean(payload.get("command"), default.command),
        base_url=_clean(payload.get("base_url"), default.base_url),
        cwd=_clean(payload.get("cwd"), default.cwd),
        sandbox=_clean(payload.get("sandbox"), default.sandbox),
        timeout_sec=float(payload.get("timeout_sec", default.timeout_sec) or default.timeout_sec),
        supports_stateful_sessions=bool(payload.get("supports_stateful_sessions", default.supports_stateful_sessions)),
        local_credentials_allowed=bool(payload.get("local_credentials_allowed", default.local_credentials_allowed)),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def _coerce_tool_adapter(value: Any, default: ToolAdapterSpec) -> ToolAdapterSpec:
    payload = _mapping(value)
    if isinstance(value, ToolAdapterSpec):
        return value
    if not payload:
        return default
    return replace(
        default,
        adapter_id=_clean(payload.get("adapter_id"), default.adapter_id),
        adapter_type=_clean(payload.get("adapter_type"), default.adapter_type),
        allowed_tools=_string_list(payload.get("allowed_tools")) or list(default.allowed_tools),
        structured_output_mode=_clean(payload.get("structured_output_mode"), default.structured_output_mode),
        side_effect_policy=_clean(payload.get("side_effect_policy"), default.side_effect_policy),
        metadata=dict(payload.get("metadata", {}) or {}) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def build_llm_runtime_plan(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
    timeout_sec: float = 60.0,
    provider_spec: Any = None,
    auth_profile: Any = None,
    execution_runtime: Any = None,
    tool_adapter: Any = None,
    cost_policy: Any = None,
    context_policy: Any = None,
    verifier_policy: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> LLMRuntimePlan:
    default_provider = default_provider_spec(provider, model=model, base_url=base_url)
    provider_value = _coerce_provider_spec(provider_spec, default_provider)
    auth_value = _coerce_auth_profile(auth_profile, default_auth_profile(provider_value.provider or provider))
    runtime_value = _coerce_execution_runtime(
        execution_runtime,
        default_execution_runtime(
            provider_value.provider or provider,
            base_url=provider_value.base_url or base_url,
            timeout_sec=timeout_sec,
        ),
    )
    adapter_value = _coerce_tool_adapter(tool_adapter, default_tool_adapter(provider_value.provider or provider))
    return LLMRuntimePlan(
        provider=provider_value,
        auth_profile=auth_value,
        execution_runtime=runtime_value,
        tool_adapter=adapter_value,
        cost_policy=coerce_cost_policy(cost_policy),
        context_policy=coerce_context_policy(context_policy),
        verifier_policy=coerce_verifier_policy(verifier_policy),
        metadata=dict(metadata or {}),
    )


def has_cost_limits(policy: CostPolicy) -> bool:
    return any(
        value is not None
        for value in (
            policy.max_llm_calls,
            policy.max_prompt_tokens,
            policy.max_completion_tokens,
            policy.max_wall_clock_seconds,
            policy.max_retry_count,
        )
    ) or not bool(policy.escalation_allowed)

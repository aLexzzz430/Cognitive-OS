from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


LLM_FAILURE_POLICY_VERSION = "conos.llm.failure_policy/v1"

_CHEAP_ROUTES = {
    "retrieval",
    "file_classification",
    "log_summary",
    "json_output",
    "structured_answer",
    "skill",
    "representation",
    "candidate_ranking",
}

_CRITICAL_ROUTES = {
    "planning",
    "planner",
    "plan_generation",
    "deliberation",
    "root_cause",
    "test_failure",
    "patch_proposal",
    "final_audit",
    "analyst",
}


def _route(value: Any) -> str:
    route = str(value or "general").strip().lower() or "general"
    if "." in route:
        route = route.split(".", 1)[0]
    return route


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "enabled"}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _is_timeout(value: Any) -> bool:
    lowered = str(value or "").lower()
    return "timeout" in lowered or "timed out" in lowered


def _is_provider_limit(value: Any) -> bool:
    lowered = str(value or "").lower()
    if not lowered:
        return False
    limit_markers = (
        "usage limit",
        "rate limit",
        "quota",
        "quota_exceeded",
        "insufficient_quota",
        "too many requests",
        "try again",
        "429",
    )
    return any(marker in lowered for marker in limit_markers)


def _failure_type(value: Any, *, status: Any = "") -> str:
    status_text = str(status or "").strip().lower()
    if status_text in {
        "timeout",
        "format_error",
        "invalid_kwargs",
        "duplicate_action",
        "budget_exceeded",
        "client_unavailable",
        "provider_limit",
        "quota_exceeded",
        "rate_limit",
    }:
        return "provider_limit" if status_text in {"quota_exceeded", "rate_limit"} else status_text
    lowered = str(value or "").lower()
    if _is_provider_limit(lowered):
        return "provider_limit"
    if _is_timeout(lowered):
        return "timeout"
    if "budget" in lowered and "exceeded" in lowered:
        return "budget_exceeded"
    if "format" in lowered or "json" in lowered or "parse" in lowered:
        return "format_error"
    if "missing_required" in lowered or "invalid_kwargs" in lowered:
        return "invalid_kwargs"
    if "duplicate" in lowered or "repeated_action" in lowered:
        return "duplicate_action"
    if "unavailable" in lowered or "no_client" in lowered:
        return "client_unavailable"
    return "model_error"


def classify_llm_failure_type(failure: Any = "", *, status: Any = "") -> str:
    return _failure_type(failure, status=status)


def _runtime_mode_from_metadata(route_metadata: Mapping[str, Any] | None, runtime_mode: Any = "") -> str:
    if runtime_mode:
        return str(runtime_mode or "").strip().upper()
    metadata = _as_dict(route_metadata)
    route_context = _as_dict(metadata.get("route_context"))
    context_metadata = _as_dict(route_context.get("metadata"))
    nested_runtime = context_metadata.get("runtime_mode") or route_context.get("runtime_mode")
    if isinstance(nested_runtime, Mapping):
        nested_runtime = nested_runtime.get("mode")
    if nested_runtime:
        return str(nested_runtime or "").strip().upper()
    runtime_policy = _as_dict(metadata.get("runtime_policy"))
    return str(runtime_policy.get("runtime_mode") or "").strip().upper()


@dataclass(frozen=True)
class LLMFailurePolicy:
    timeout_is_terminal: bool = True
    fallback_patch_allowed: bool = False
    automatic_model_fallback_allowed: bool = False
    escalation_allowed: bool = True
    max_retry_count: int = 0
    schema_version: str = LLM_FAILURE_POLICY_VERSION

    @classmethod
    def from_mapping(cls, value: Any) -> "LLMFailurePolicy":
        payload = _as_dict(value)
        return cls(
            timeout_is_terminal=_truthy(payload.get("timeout_is_terminal"), default=True),
            fallback_patch_allowed=_truthy(payload.get("fallback_patch_allowed"), default=False),
            automatic_model_fallback_allowed=_truthy(
                payload.get("automatic_model_fallback_allowed"),
                default=False,
            ),
            escalation_allowed=_truthy(payload.get("escalation_allowed"), default=True),
            max_retry_count=max(0, int(payload.get("max_retry_count", 0) or 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMFailureDecision:
    route_name: str
    failure_type: str
    recommended_action: str
    terminal: bool
    should_retry: bool = False
    should_escalate: bool = False
    should_downgrade: bool = False
    fallback_patch_allowed: bool = False
    automatic_model_fallback_allowed: bool = False
    retry_budget_remaining: int = 0
    reason: str = ""
    audit_event: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = LLM_FAILURE_POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decide_llm_failure_policy(
    *,
    route_name: Any,
    failure: Any = "",
    status: Any = "",
    route_metadata: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    policy: LLMFailurePolicy | Mapping[str, Any] | None = None,
    runtime_mode: Any = "",
    prior_failure_count: int = 0,
) -> LLMFailureDecision:
    """Turn model failure into an auditable next-step policy decision.

    This layer intentionally does not apply patches or call fallback models by
    itself. It only states what the runtime is allowed to do next.
    """

    route = _route(route_name)
    failure_type = _failure_type(failure, status=status)
    metadata = _as_dict(route_metadata)
    budget_payload = _as_dict(budget or metadata.get("budget"))
    effective_policy = policy if isinstance(policy, LLMFailurePolicy) else LLMFailurePolicy.from_mapping(policy)
    if "escalation_allowed" in budget_payload:
        effective_policy = LLMFailurePolicy(
            timeout_is_terminal=effective_policy.timeout_is_terminal,
            fallback_patch_allowed=effective_policy.fallback_patch_allowed,
            automatic_model_fallback_allowed=effective_policy.automatic_model_fallback_allowed,
            escalation_allowed=_truthy(budget_payload.get("escalation_allowed"), default=effective_policy.escalation_allowed),
            max_retry_count=effective_policy.max_retry_count,
        )
    if "max_retry_count" in budget_payload:
        effective_policy = LLMFailurePolicy(
            timeout_is_terminal=effective_policy.timeout_is_terminal,
            fallback_patch_allowed=effective_policy.fallback_patch_allowed,
            automatic_model_fallback_allowed=effective_policy.automatic_model_fallback_allowed,
            escalation_allowed=effective_policy.escalation_allowed,
            max_retry_count=max(0, int(budget_payload.get("max_retry_count", 0) or 0)),
        )
    retries_remaining = max(0, int(effective_policy.max_retry_count) - max(0, int(prior_failure_count or 0)))
    mode = _runtime_mode_from_metadata(metadata, runtime_mode=runtime_mode)
    status_monitor = _as_dict(_as_dict(_as_dict(metadata.get("route_context")).get("metadata")).get("status_monitor"))
    status_recommends_escalation = bool(status_monitor.get("should_escalate", False))
    critical = route in _CRITICAL_ROUTES
    cheap = route in _CHEAP_ROUTES
    should_retry = False
    should_escalate = False
    should_downgrade = False
    terminal = False
    recommended_action = "return_structured_failure"
    reason = "model_failure"

    if failure_type == "timeout":
        if bool(effective_policy.automatic_model_fallback_allowed):
            terminal = False
            recommended_action = "try_configured_model_fallback_after_timeout"
            should_escalate = bool(effective_policy.escalation_allowed and (critical or status_recommends_escalation))
            reason = "timeout_model_fallback_explicitly_configured"
        else:
            terminal = bool(effective_policy.timeout_is_terminal)
            recommended_action = "return_structured_timeout"
            should_escalate = bool(effective_policy.escalation_allowed and (critical or status_recommends_escalation))
            reason = "timeout_is_terminal_no_hidden_fallback"
    elif failure_type == "budget_exceeded":
        terminal = True
        recommended_action = "return_budget_blocked"
        reason = "budget_exceeded"
    elif failure_type == "provider_limit":
        if bool(effective_policy.automatic_model_fallback_allowed):
            terminal = False
            recommended_action = "try_profile_ranked_model_after_provider_limit"
            should_escalate = bool(effective_policy.escalation_allowed)
            should_downgrade = True
            reason = "provider_limit_profile_ranked_model_fallback_allowed"
        else:
            terminal = True
            recommended_action = "return_provider_limit"
            reason = "provider_limit_no_configured_model_fallback"
    elif failure_type == "duplicate_action":
        terminal = False
        recommended_action = "replan_action"
        should_downgrade = cheap
        reason = "duplicate_action_signature"
    elif failure_type in {"format_error", "invalid_kwargs"}:
        if retries_remaining > 0:
            should_retry = True
            recommended_action = "retry_with_output_adapter"
            reason = "schema_repair_retry_available"
        elif bool(effective_policy.escalation_allowed) and not cheap:
            should_escalate = True
            recommended_action = "escalate_to_structured_output_model"
            reason = "schema_failure_requires_more_reliable_model"
        else:
            terminal = True
            recommended_action = "return_schema_failure"
            reason = "schema_failure_no_retry_or_escalation"
    elif failure_type == "client_unavailable":
        terminal = not bool(effective_policy.automatic_model_fallback_allowed)
        recommended_action = (
            "try_configured_model_fallback"
            if effective_policy.automatic_model_fallback_allowed
            else "return_client_unavailable"
        )
        should_escalate = bool(effective_policy.automatic_model_fallback_allowed and effective_policy.escalation_allowed)
        reason = "client_unavailable"
    else:
        if retries_remaining > 0:
            should_retry = True
            recommended_action = "retry_same_route"
            reason = "retry_budget_available"
        elif bool(effective_policy.escalation_allowed) and (critical or status_recommends_escalation):
            should_escalate = True
            recommended_action = "escalate_with_audit"
            reason = "critical_route_or_status_pressure"
        else:
            terminal = True
            reason = "noncritical_model_error"

    if mode in {"SLEEP", "WAITING_HUMAN", "DEGRADED_RECOVERY", "STOPPED"}:
        should_retry = False
        should_escalate = False
        if recommended_action.startswith("escalate"):
            recommended_action = "return_structured_failure"
        reason = f"{reason};runtime_mode_blocks_llm_recovery:{mode}"

    audit_event = {
        "schema_version": LLM_FAILURE_POLICY_VERSION,
        "event_type": "llm_failure_policy_decision",
        "route_name": route,
        "failure_type": failure_type,
        "recommended_action": recommended_action,
        "terminal": bool(terminal),
        "should_retry": bool(should_retry),
        "should_escalate": bool(should_escalate),
        "should_downgrade": bool(should_downgrade),
        "fallback_patch_allowed": bool(effective_policy.fallback_patch_allowed),
        "automatic_model_fallback_allowed": bool(effective_policy.automatic_model_fallback_allowed),
        "runtime_mode": mode,
        "status_monitor_confidence": _bounded_float(status_monitor.get("confidence", 0.0)),
        "reason": reason,
    }
    return LLMFailureDecision(
        route_name=route,
        failure_type=failure_type,
        recommended_action=recommended_action,
        terminal=bool(terminal),
        should_retry=bool(should_retry),
        should_escalate=bool(should_escalate),
        should_downgrade=bool(should_downgrade),
        fallback_patch_allowed=bool(effective_policy.fallback_patch_allowed),
        automatic_model_fallback_allowed=bool(effective_policy.automatic_model_fallback_allowed),
        retry_budget_remaining=retries_remaining,
        reason=reason,
        audit_event=audit_event,
    )


def failure_policy_catalog() -> Dict[str, Any]:
    return {
        "schema_version": LLM_FAILURE_POLICY_VERSION,
        "default_policy": LLMFailurePolicy().to_dict(),
        "decisions": {
            "timeout": {
                "recommended_action": "return_structured_timeout",
                "fallback_patch_allowed": False,
                "note": "timeout does not trigger hidden deterministic patching",
            },
            "format_error": {
                "recommended_action": "retry_with_output_adapter_or_escalate",
                "note": "malformed output is normalized first, then retried or escalated only within budget",
            },
            "invalid_kwargs": {
                "recommended_action": "reject_or_replan_before_tool_execution",
                "note": "invalid kwargs cannot reach side-effect adapters",
            },
            "duplicate_action": {
                "recommended_action": "replan_action",
                "note": "repeated action signatures are blocked before spending more execution budget",
            },
            "budget_exceeded": {
                "recommended_action": "return_budget_blocked",
                "note": "budget exhaustion is terminal for the current route call",
            },
            "provider_limit": {
                "recommended_action": "try_profile_ranked_model_after_provider_limit",
                "fallback_patch_allowed": False,
                "note": "provider quota or usage limits trigger auditable model-route fallback, never deterministic patch fallback",
            },
        },
    }

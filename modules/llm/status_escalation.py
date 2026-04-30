from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core.runtime.runtime_modes import RuntimeMode, normalize_runtime_mode
from modules.llm.json_adaptor import normalize_llm_output


STATUS_ESCALATION_VERSION = "conos.llm.status_escalation/v1"

_SEVERE_STATUSES = {
    "degraded",
    "degraded_recovery",
    "failed",
    "failure",
    "error",
    "zombie_suspected",
    "zombie",
    "crashed",
    "timeout",
}
_CLOUD_PROVIDERS = {"openai", "responses", "codex", "codex-cli", "openai-oauth-codex"}


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "enabled"}


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _status_path_from_env(env: Mapping[str, str]) -> str:
    return (
        _string(env.get("CONOS_LLM_STATUS_MONITOR_FILE"))
        or _string(env.get("CONOS_STATUS_FILE"))
        or _string(env.get("CONOS_RUNTIME_STATUS_FILE"))
    )


def _status_file_enabled(env: Mapping[str, str], status_path: str) -> bool:
    if "CONOS_LLM_STATUS_MONITOR_ENABLED" in env:
        return _truthy(env.get("CONOS_LLM_STATUS_MONITOR_ENABLED"))
    return bool(status_path)


def _load_status_payload(path: str) -> tuple[Dict[str, Any], str]:
    if not path:
        return {}, "status_file_not_configured"
    status_path = Path(path).expanduser()
    if not status_path.exists():
        return {}, "status_file_missing"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"status_file_invalid_json:{exc}"
    if not isinstance(payload, Mapping):
        return {}, "status_file_not_object"
    return dict(payload), ""


def _walk_values(payload: Any, key_names: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key or "").strip().lower()
            if key_text in key_names:
                values.append(value)
            values.extend(_walk_values(value, key_names))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_walk_values(item, key_names))
    return values


def _first_status(payload: Mapping[str, Any]) -> str:
    values = _walk_values(payload, {"status", "run_status", "heartbeat_status"})
    for value in values:
        text = _string(value).lower()
        if text:
            return text
    return ""


def _first_runtime_mode(payload: Mapping[str, Any]) -> str:
    values = _walk_values(payload, {"runtime_mode", "mode"})
    for value in values:
        if isinstance(value, Mapping):
            text = normalize_runtime_mode(value.get("mode"))
        else:
            text = normalize_runtime_mode(value)
        if text:
            return text
    return ""


def _first_bool(payload: Mapping[str, Any], *keys: str) -> bool:
    values = _walk_values(payload, {str(key).lower() for key in keys})
    return any(_truthy(value) for value in values)


def _max_numeric(payload: Mapping[str, Any], *keys: str) -> float:
    values = _walk_values(payload, {str(key).lower() for key in keys})
    parsed = [_float_or_default(value, 0.0) for value in values]
    return max(parsed or [0.0])


def _text_contains(payload: Mapping[str, Any], tokens: set[str]) -> bool:
    haystack = json.dumps(payload, ensure_ascii=False, default=str).lower()
    return any(token in haystack for token in tokens)


def _collect_status_signals(
    *,
    status_payload: Mapping[str, Any],
    route_context: Mapping[str, Any],
    failure_threshold: int,
    latency_threshold_ms: float,
) -> Dict[str, Any]:
    context_metadata = _dict_or_empty(route_context.get("metadata", {}))
    status_text = _first_status(status_payload)
    context_runtime_mode = normalize_runtime_mode(context_metadata.get("runtime_mode") or route_context.get("runtime_mode"))
    runtime_mode = _first_runtime_mode(status_payload) or context_runtime_mode
    failure_count = int(
        max(
            _max_numeric(status_payload, "failure_count", "consecutive_failures", "error_count"),
            _max_numeric(context_metadata, "failure_count", "consecutive_failures"),
        )
    )
    ollama_latency_ms = max(
        _max_numeric(status_payload, "ollama_latency_ms"),
        _max_numeric(status_payload, "latency_ms"),
    )
    verification_pressure = _bounded_float(route_context.get("verification_pressure", 0.0))
    uncertainty_level = _bounded_float(route_context.get("uncertainty_level", 0.0))
    degraded = (
        status_text in _SEVERE_STATUSES
        or runtime_mode == RuntimeMode.DEGRADED_RECOVERY.value
        or _first_bool(status_payload, "degraded", "zombie_suspected", "model_timeout", "timeout")
        or _text_contains(status_payload, {"zombie_suspected", "runtime_degraded", "model_timeout"})
    )
    ollama_disconnected = (
        _first_bool(status_payload, "ollama_disconnected", "remote_ollama_disconnected")
        or any(value is False for value in _walk_values(status_payload, {"ollama_connected", "network_reachable"}))
    )
    verifier_failed = (
        str(context_metadata.get("verifier_verdict", "") or "").strip().lower() == "failed"
        or _text_contains(status_payload, {"verifier_failed", "verification_failed"})
    )
    high_uncertainty = uncertainty_level >= 0.75 and verification_pressure >= 0.5
    signals = {
        "status": status_text,
        "runtime_mode": runtime_mode,
        "degraded": bool(degraded),
        "failure_count": failure_count,
        "failure_threshold": int(failure_threshold),
        "ollama_disconnected": bool(ollama_disconnected),
        "ollama_latency_ms": float(ollama_latency_ms),
        "ollama_latency_threshold_ms": float(latency_threshold_ms),
        "verifier_failed": bool(verifier_failed),
        "verification_pressure": float(verification_pressure),
        "uncertainty_level": float(uncertainty_level),
        "high_uncertainty": bool(high_uncertainty),
        "waiting_approval": _first_bool(status_payload, "waiting_approval"),
    }
    return signals


def _deterministic_status_decision(
    *,
    signals: Mapping[str, Any],
    route_name: str,
    route_context: Mapping[str, Any],
) -> tuple[bool, float, str]:
    reasons: list[str] = []
    runtime_mode = str(signals.get("runtime_mode", "") or "")
    if runtime_mode in {
        RuntimeMode.STOPPED.value,
        RuntimeMode.SLEEP.value,
        RuntimeMode.IDLE.value,
        RuntimeMode.DREAM.value,
        RuntimeMode.WAITING_HUMAN.value,
    }:
        return False, 0.0, f"runtime_mode_{runtime_mode.lower()}_does_not_escalate"
    if bool(signals.get("degraded", False)):
        reasons.append("runtime_degraded")
    if int(signals.get("failure_count", 0) or 0) >= int(signals.get("failure_threshold", 2) or 2):
        reasons.append("repeated_failures")
    if bool(signals.get("ollama_disconnected", False)):
        reasons.append("local_model_unreachable")
    if float(signals.get("ollama_latency_ms", 0.0) or 0.0) >= float(signals.get("ollama_latency_threshold_ms", 12000.0) or 12000.0):
        reasons.append("local_model_latency_high")
    if bool(signals.get("verifier_failed", False)):
        reasons.append("verifier_failed")
    if bool(signals.get("high_uncertainty", False)):
        reasons.append("high_uncertainty_under_verification")
    if runtime_mode == RuntimeMode.CREATING.value and float(route_context.get("uncertainty_level", 0.0) or 0.0) >= 0.55:
        reasons.append("creative_search_under_uncertainty")
    if runtime_mode == RuntimeMode.DEEP_THINK.value and float(route_context.get("uncertainty_level", 0.0) or 0.0) >= 0.45:
        reasons.append("deep_think_under_uncertainty")
    route_key = str(route_name or "general").strip()
    if route_key in {"planning", "planner", "plan_generation"} and float(route_context.get("uncertainty_level", 0.0) or 0.0) >= 0.6:
        reasons.append("planning_uncertainty")
    if not reasons:
        return False, 0.0, "status_nominal"
    confidence = 0.58 + min(0.34, 0.06 * len(reasons))
    return True, _bounded_float(confidence), ",".join(reasons)


def _local_model_decision(
    *,
    env: Mapping[str, str],
    status_payload: Mapping[str, Any],
    signals: Mapping[str, Any],
    route_name: str,
    route_context: Mapping[str, Any],
) -> tuple[Optional[bool], float, str, str]:
    model = _string(env.get("CONOS_LLM_STATUS_MONITOR_MODEL"))
    if not model:
        return None, 0.0, "local_model_not_configured", ""
    use_llm = _truthy(env.get("CONOS_LLM_STATUS_MONITOR_USE_LLM"), default=True)
    if not use_llm:
        return None, 0.0, "local_model_disabled", ""
    provider = _string(env.get("CONOS_LLM_STATUS_MONITOR_PROVIDER")) or "ollama"
    base_url = _string(env.get("CONOS_LLM_STATUS_MONITOR_BASE_URL")) or _string(env.get("OLLAMA_BASE_URL"))
    timeout_sec = _float_or_default(env.get("CONOS_LLM_STATUS_MONITOR_TIMEOUT_SEC"), 4.0)
    max_tokens = _int_or_default(env.get("CONOS_LLM_STATUS_MONITOR_MAX_TOKENS"), 160)
    try:
        from modules.llm.factory import build_llm_client

        client = build_llm_client(
            provider=provider,
            base_url=base_url or None,
            model=model,
            timeout_sec=timeout_sec,
        )
        if client is None:
            return None, 0.0, "local_model_client_unavailable", ""
        prompt = {
            "task": "Decide whether this Cognitive OS route should escalate to a cloud high-trust model.",
            "route_name": str(route_name or "general"),
            "route_context": {
                "uncertainty_level": route_context.get("uncertainty_level", 0.0),
                "verification_pressure": route_context.get("verification_pressure", 0.0),
                "prefer_low_cost": route_context.get("prefer_low_cost", 0.0),
                "prefer_high_trust": route_context.get("prefer_high_trust", 0.0),
            },
            "signals": dict(signals),
            "status": dict(status_payload),
            "return_json_schema": {
                "should_escalate": "boolean",
                "confidence": "number 0..1",
                "reason": "short string",
            },
        }
        if hasattr(client, "complete_json"):
            response = client.complete_json(
                json.dumps(prompt, ensure_ascii=False, default=str),
                max_tokens=max_tokens,
                temperature=0.0,
                think=False,
                thinking_budget=0,
                timeout_sec=timeout_sec,
            )
        else:
            text = client.complete(
                json.dumps(prompt, ensure_ascii=False, default=str),
                max_tokens=max_tokens,
                temperature=0.0,
                think=False,
                thinking_budget=0,
                timeout_sec=timeout_sec,
            )
            response = normalize_llm_output(
                text,
                output_kind="status_escalation_decision",
                expected_type="dict",
            ).parsed_dict()
        if not isinstance(response, Mapping):
            return None, 0.0, "local_model_invalid_response", ""
        should_escalate = bool(response.get("should_escalate", False))
        confidence = _bounded_float(response.get("confidence", 0.5), default=0.5)
        reason = _string(response.get("reason")) or "local_model_decision"
        return should_escalate, confidence, reason, ""
    except Exception as exc:
        return None, 0.0, "local_model_error", str(exc)


@dataclass(frozen=True)
class StatusEscalationDecision:
    schema_version: str = STATUS_ESCALATION_VERSION
    enabled: bool = False
    status_path: str = ""
    source: str = "disabled"
    should_escalate: bool = False
    confidence: float = 0.0
    reason: str = ""
    cloud_route_bias: float = 0.0
    local_provider: str = ""
    local_model: str = ""
    signals: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decide_status_escalation(
    *,
    route_name: str,
    route_context: Mapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> StatusEscalationDecision:
    env = dict(environ or os.environ)
    status_path = _status_path_from_env(env)
    enabled = _status_file_enabled(env, status_path)
    local_provider = _string(env.get("CONOS_LLM_STATUS_MONITOR_PROVIDER")) or "ollama"
    local_model = _string(env.get("CONOS_LLM_STATUS_MONITOR_MODEL"))
    if not enabled:
        return StatusEscalationDecision(
            enabled=False,
            status_path=status_path,
            source="disabled",
            reason="status_monitor_disabled",
            local_provider=local_provider,
            local_model=local_model,
        )
    status_payload, load_error = _load_status_payload(status_path)
    if load_error:
        return StatusEscalationDecision(
            enabled=True,
            status_path=status_path,
            source="status_file",
            reason=load_error,
            local_provider=local_provider,
            local_model=local_model,
            error=load_error,
        )
    failure_threshold = _int_or_default(env.get("CONOS_LLM_STATUS_MONITOR_FAILURE_THRESHOLD"), 2)
    latency_threshold_ms = _float_or_default(env.get("CONOS_LLM_STATUS_MONITOR_LATENCY_MS"), 12000.0)
    signals = _collect_status_signals(
        status_payload=status_payload,
        route_context=route_context,
        failure_threshold=failure_threshold,
        latency_threshold_ms=latency_threshold_ms,
    )
    deterministic_should, deterministic_confidence, deterministic_reason = _deterministic_status_decision(
        signals=signals,
        route_name=route_name,
        route_context=route_context,
    )
    model_should, model_confidence, model_reason, model_error = _local_model_decision(
        env=env,
        status_payload=status_payload,
        signals=signals,
        route_name=route_name,
        route_context=route_context,
    )
    if model_should is not None and not deterministic_should:
        should_escalate = bool(model_should)
        confidence = model_confidence
        reason = model_reason
        source = "local_model"
    elif deterministic_should:
        should_escalate = True
        confidence = max(deterministic_confidence, model_confidence if model_should else 0.0)
        reason = deterministic_reason if model_should is None else f"{deterministic_reason};local_model={model_reason}"
        source = "deterministic_guardrail" if model_should is None else "deterministic_guardrail+local_model"
    else:
        should_escalate = False
        confidence = max(deterministic_confidence, model_confidence if model_should is not None else 0.0)
        reason = model_reason if model_should is not None else deterministic_reason
        source = "local_model" if model_should is not None else "deterministic_status"
    cloud_route_bias = _bounded_float(confidence if should_escalate else 0.0)
    return StatusEscalationDecision(
        enabled=True,
        status_path=status_path,
        source=source,
        should_escalate=bool(should_escalate),
        confidence=round(float(confidence), 4),
        reason=reason,
        cloud_route_bias=cloud_route_bias,
        local_provider=local_provider,
        local_model=local_model,
        signals=dict(signals),
        error=model_error,
    )


def apply_status_escalation_to_route_context(
    route_context: Mapping[str, Any],
    *,
    route_name: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    context = dict(route_context or {})
    metadata = _dict_or_empty(context.get("metadata", {}))
    decision = decide_status_escalation(
        route_name=route_name,
        route_context=context,
        environ=environ,
    )
    metadata["status_monitor"] = decision.to_dict()
    if decision.should_escalate:
        context["prefer_high_trust"] = round(max(_bounded_float(context.get("prefer_high_trust", 0.0)), 0.96), 4)
        context["prefer_low_cost"] = round(min(_bounded_float(context.get("prefer_low_cost", 0.0)), 0.12), 4)
        context["prefer_low_latency"] = round(min(_bounded_float(context.get("prefer_low_latency", 0.0)), 0.22), 4)
        metadata["cloud_escalation_recommended"] = True
        metadata["cloud_route_bias"] = decision.cloud_route_bias
        metadata["cloud_escalation_reason"] = decision.reason
        metadata["status_monitor_should_escalate"] = True
    else:
        metadata["cloud_escalation_recommended"] = False
        metadata["cloud_route_bias"] = 0.0
        metadata["status_monitor_should_escalate"] = False
    context["metadata"] = metadata
    return context


def is_cloud_provider(value: Any) -> bool:
    return str(value or "").strip().lower() in _CLOUD_PROVIDERS

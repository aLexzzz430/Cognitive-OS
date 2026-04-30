from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, Literal, Mapping, Sequence

from modules.llm.json_adaptor import (
    LLM_OUTPUT_ADAPTER_VERSION,
    normalize_llm_output,
)


LLM_RELIABILITY_ADAPTER_VERSION = "conos.llm.reliability_adapter/v1"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _is_timeout_error(value: Any) -> bool:
    lowered = str(value or "").lower()
    return "timeout" in lowered or "timed out" in lowered


def _schema_required_fields(function_schema: Mapping[str, Any] | None) -> list[str]:
    schema = _as_dict(function_schema)
    params = _as_dict(schema.get("parameters"))
    return _string_list(params.get("required"))


def _action_signature(function_name: str, kwargs: Mapping[str, Any]) -> str:
    return _json_hash({"function_name": str(function_name or ""), "kwargs": dict(kwargs or {})})


def _recent_action_signatures(recent_actions: Sequence[Mapping[str, Any]] | None) -> set[str]:
    signatures: set[str] = set()
    for row in list(recent_actions or []):
        if not isinstance(row, Mapping):
            continue
        function_name = str(row.get("function_name") or row.get("action") or "").strip()
        kwargs = row.get("kwargs", row.get("args", {}))
        if function_name and isinstance(kwargs, Mapping):
            signatures.add(_action_signature(function_name, kwargs))
    return signatures


@dataclass(frozen=True)
class LLMReliabilityPolicy:
    output_kind: str
    expected_type: Literal["dict", "list", "any"] = "dict"
    required_fields: tuple[str, ...] = ()
    timeout_is_terminal: bool = True
    fallback_on_timeout_allowed: bool = False
    escalation_allowed: bool = True
    duplicate_action_block: bool = True
    schema_version: str = LLM_RELIABILITY_ADAPTER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMReliabilityResult:
    ok: bool
    status: str
    parsed: Any = None
    error: str = ""
    output_kind: str = ""
    timeout: bool = False
    missing_fields: list[str] = field(default_factory=list)
    duplicate_action: bool = False
    action_signature: str = ""
    should_escalate: bool = False
    fallback_allowed: bool = False
    output_adapter: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = LLM_RELIABILITY_ADAPTER_VERSION

    def parsed_dict(self) -> Dict[str, Any]:
        return dict(self.parsed) if isinstance(self.parsed, Mapping) else {}

    def to_trace(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": bool(self.ok),
            "status": str(self.status),
            "error": str(self.error),
            "output_kind": str(self.output_kind),
            "timeout": bool(self.timeout),
            "missing_fields": list(self.missing_fields),
            "duplicate_action": bool(self.duplicate_action),
            "action_signature": str(self.action_signature),
            "should_escalate": bool(self.should_escalate),
            "fallback_allowed": bool(self.fallback_allowed),
            "output_adapter": dict(self.output_adapter),
            "policy": dict(self.policy),
        }


def normalize_reliable_llm_output(
    raw_output: Any,
    *,
    policy: LLMReliabilityPolicy,
    expected_prefixes: Sequence[str] | None = None,
    function_name: str = "",
    function_schema: Mapping[str, Any] | None = None,
    recent_actions: Sequence[Mapping[str, Any]] | None = None,
    timeout_error: Any = "",
) -> LLMReliabilityResult:
    if timeout_error:
        timeout = _is_timeout_error(timeout_error)
        if timeout:
            return LLMReliabilityResult(
                ok=False,
                status="timeout",
                error=str(timeout_error or raw_output or "timeout"),
                output_kind=policy.output_kind,
                timeout=True,
                should_escalate=bool(policy.escalation_allowed),
                fallback_allowed=bool(policy.fallback_on_timeout_allowed),
                policy=policy.to_dict(),
            )

    normalized = normalize_llm_output(
        raw_output,
        output_kind=policy.output_kind,
        expected_prefixes=expected_prefixes,
        expected_type=policy.expected_type,
    )
    adapter_trace = normalized.to_trace()
    if not normalized.ok:
        return LLMReliabilityResult(
            ok=False,
            status="format_error",
            error=normalized.error or "parse_failed",
            output_kind=policy.output_kind,
            should_escalate=bool(policy.escalation_allowed),
            fallback_allowed=True,
            output_adapter=adapter_trace,
            policy=policy.to_dict(),
        )

    parsed = normalized.parsed
    missing = []
    if isinstance(parsed, Mapping):
        required = list(policy.required_fields) + _schema_required_fields(function_schema)
        missing = [field for field in dict.fromkeys(required) if field not in parsed or parsed.get(field) in (None, "")]
    if missing:
        return LLMReliabilityResult(
            ok=False,
            status="invalid_kwargs",
            parsed=parsed,
            error="missing_required_fields",
            output_kind=policy.output_kind,
            missing_fields=missing,
            should_escalate=bool(policy.escalation_allowed),
            fallback_allowed=True,
            output_adapter=adapter_trace,
            policy=policy.to_dict(),
        )

    signature = ""
    duplicate = False
    if function_name and isinstance(parsed, Mapping):
        signature = _action_signature(function_name, parsed)
        duplicate = signature in _recent_action_signatures(recent_actions)
        if duplicate and bool(policy.duplicate_action_block):
            return LLMReliabilityResult(
                ok=False,
                status="duplicate_action",
                parsed=parsed,
                error="repeated_action_signature",
                output_kind=policy.output_kind,
                duplicate_action=True,
                action_signature=signature,
                should_escalate=bool(policy.escalation_allowed),
                fallback_allowed=True,
                output_adapter=adapter_trace,
                policy=policy.to_dict(),
            )

    return LLMReliabilityResult(
        ok=True,
        status="accepted",
        parsed=parsed,
        output_kind=policy.output_kind,
        duplicate_action=duplicate,
        action_signature=signature,
        output_adapter=adapter_trace,
        policy=policy.to_dict(),
    )


def list_llm_reliability_contracts() -> list[Dict[str, Any]]:
    return [
        {
            "schema_version": LLM_RELIABILITY_ADAPTER_VERSION,
            "output_kind": "action_kwargs",
            "covers": ["format_error", "timeout", "missing_required_kwargs", "duplicate_action"],
            "fallback_on_timeout_allowed": False,
        },
        {
            "schema_version": LLM_RELIABILITY_ADAPTER_VERSION,
            "output_kind": "patch_proposal",
            "covers": ["format_error", "timeout", "bounded_diff_validation"],
            "fallback_on_timeout_allowed": False,
        },
        {
            "schema_version": LLM_RELIABILITY_ADAPTER_VERSION,
            "output_kind": "reasoning_state",
            "covers": ["format_error", "timeout", "missing_reasoning_state"],
            "fallback_on_timeout_allowed": False,
        },
    ]

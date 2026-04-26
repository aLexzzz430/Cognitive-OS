from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .thinking_policy import thinking_policy_for_route


LLM_BUDGET_VERSION = "conos.llm.budget/v1"


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0.0 else None


def _estimate_tokens(text: Any) -> int:
    payload = str(text or "")
    if not payload:
        return 0
    return max(1, (len(payload) + 3) // 4)


def _route_from_kwargs(kwargs: Mapping[str, Any]) -> str:
    for key in ("capability_route_name", "route_name", "llm_route_name"):
        value = str(kwargs.get(key) or "").strip()
        if value:
            return value
    capability_request = kwargs.get("capability_request")
    capability_route = str(getattr(capability_request, "route_name", "") or "").strip()
    if capability_route:
        return capability_route
    if isinstance(capability_request, Mapping):
        capability_route = str(capability_request.get("route_name") or "").strip()
        if capability_route:
            return capability_route
    capability = str(capability_request or "").strip()
    if "." in capability:
        return capability.split(".", 1)[0]
    return capability or "general"


def classify_llm_layer(route_name: Any, kwargs: Optional[Mapping[str, Any]] = None) -> str:
    route = str(route_name or "general").strip().lower() or "general"
    if "." in route:
        route = route.split(".", 1)[0]
    if route in {
        "retrieval",
        "file_classification",
        "log_summary",
        "json_output",
        "structured_answer",
        "candidate_ranking",
        "skill",
        "representation",
    }:
        return "small_model"
    if route in {
        "hypothesis",
        "probe",
        "root_cause",
        "test_failure",
        "patch_proposal",
        "recovery",
        "final_audit",
        "analyst",
        "planning",
        "planner",
        "plan_generation",
        "deliberation",
    }:
        return "strong_model"
    decision = thinking_policy_for_route(route, mode=(kwargs or {}).get("thinking_mode", "auto"))
    return "strong_model" if int(decision.tier or 0) >= 2 or bool(decision.prefer_strongest_model) else "small_model"


@dataclass(frozen=True)
class LLMRuntimeBudget:
    max_llm_calls: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None
    max_retry_count: Optional[int] = None
    escalation_allowed: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "LLMRuntimeBudget":
        payload = dict(value or {}) if isinstance(value, Mapping) else {}
        return cls(
            max_llm_calls=_optional_int(payload.get("max_llm_calls")),
            max_prompt_tokens=_optional_int(payload.get("max_prompt_tokens")),
            max_completion_tokens=_optional_int(payload.get("max_completion_tokens")),
            max_wall_clock_seconds=_optional_float(payload.get("max_wall_clock_seconds")),
            max_retry_count=_optional_int(payload.get("max_retry_count")),
            escalation_allowed=bool(payload.get("escalation_allowed", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = LLM_BUDGET_VERSION
        return payload


@dataclass(frozen=True)
class LLMCallCost:
    call_id: str
    method: str
    route_name: str
    layer: str
    model: str
    prompt_tokens_estimated: int
    requested_completion_tokens: int
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    completion_tokens_estimated: int = 0
    wall_seconds: float = 0.0
    success: bool = True
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMCostLedger:
    def __init__(self, budget: Optional[LLMRuntimeBudget] = None) -> None:
        self.budget = budget or LLMRuntimeBudget()
        self._started_at = time.perf_counter()
        self._records: list[LLMCallCost] = []

    def can_start(
        self,
        *,
        prompt_tokens: int = 0,
        requested_completion_tokens: int = 0,
        layer: str = "",
        route_name: str = "",
    ) -> tuple[bool, str]:
        del route_name
        if str(layer or "").strip() == "strong_model" and not bool(self.budget.escalation_allowed):
            return False, "strong_model_escalation_not_allowed"
        summary = self.summary()
        if self.budget.max_llm_calls is not None and summary["total_calls"] >= self.budget.max_llm_calls:
            return False, "max_llm_calls_exceeded"
        if (
            self.budget.max_prompt_tokens is not None
            and summary["prompt_tokens_estimated"] + int(prompt_tokens or 0) > self.budget.max_prompt_tokens
        ):
            return False, "max_prompt_tokens_exceeded"
        if (
            self.budget.max_completion_tokens is not None
            and summary["requested_completion_tokens"] + int(requested_completion_tokens or 0) > self.budget.max_completion_tokens
        ):
            return False, "max_completion_tokens_exceeded"
        if (
            self.budget.max_wall_clock_seconds is not None
            and summary["wall_seconds"] >= self.budget.max_wall_clock_seconds
        ):
            return False, "max_wall_clock_seconds_exceeded"
        return True, ""

    def record(self, record: LLMCallCost) -> None:
        self._records.append(record)

    def records(self) -> list[Dict[str, Any]]:
        return [row.to_dict() for row in self._records]

    def summary(self) -> Dict[str, Any]:
        by_layer: dict[str, dict[str, Any]] = {}
        by_route: dict[str, dict[str, Any]] = {}
        total_wall = 0.0
        prompt_est = 0
        requested_completion = 0
        input_tokens = 0
        cached_input = 0
        output_tokens = 0
        completion_est = 0
        success_count = 0
        for row in self._records:
            total_wall += float(row.wall_seconds or 0.0)
            prompt_est += int(row.prompt_tokens_estimated or 0)
            requested_completion += int(row.requested_completion_tokens or 0)
            input_tokens += int(row.input_tokens or 0)
            cached_input += int(row.cached_input_tokens or 0)
            output_tokens += int(row.output_tokens or 0)
            completion_est += int(row.completion_tokens_estimated or 0)
            success_count += 1 if row.success else 0
            for bucket, key in ((by_layer, row.layer), (by_route, row.route_name)):
                current = bucket.setdefault(str(key or "unknown"), {"calls": 0, "wall_seconds": 0.0, "output_tokens": 0})
                current["calls"] += 1
                current["wall_seconds"] = round(float(current["wall_seconds"]) + float(row.wall_seconds or 0.0), 6)
                current["output_tokens"] += int(row.output_tokens or row.completion_tokens_estimated or 0)
        total_calls = len(self._records)
        strong_calls = by_layer.get("strong_model", {}).get("calls", 0)
        return {
            "schema_version": LLM_BUDGET_VERSION,
            "budget": self.budget.to_dict(),
            "total_calls": total_calls,
            "successful_calls": success_count,
            "failed_calls": total_calls - success_count,
            "prompt_tokens_estimated": prompt_est,
            "requested_completion_tokens": requested_completion,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input,
            "output_tokens": output_tokens,
            "completion_tokens_estimated": completion_est,
            "wall_seconds": round(total_wall, 6),
            "elapsed_wall_seconds": round(time.perf_counter() - self._started_at, 6),
            "strong_model_call_rate": (float(strong_calls) / float(total_calls)) if total_calls else 0.0,
            "by_layer": by_layer,
            "by_route": by_route,
        }

    def report(self, *, verified_success: Optional[bool] = None) -> Dict[str, Any]:
        summary = self.summary()
        summary["records"] = self.records()
        if verified_success is not None:
            success_value = 1 if verified_success else 0
            total_cost = int(summary["input_tokens"] or 0) + int(summary["output_tokens"] or 0)
            if total_cost <= 0:
                total_cost = int(summary["prompt_tokens_estimated"] or 0) + int(summary["completion_tokens_estimated"] or 0)
            summary["cost_per_verified_success"] = float(total_cost) / float(success_value) if success_value else None
            summary["tokens_per_verified_success"] = int(total_cost) if success_value else None
        return summary


class BudgetAwareLLMClient:
    def __init__(self, client: Any, ledger: LLMCostLedger) -> None:
        self._client = client
        self._ledger = ledger

    @property
    def ledger(self) -> LLMCostLedger:
        return self._ledger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def complete(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return str(self._invoke("complete", prompt, *args, **kwargs) or "")

    def complete_raw(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return str(self._invoke("complete_raw", prompt, *args, **kwargs) or "")

    def complete_json(self, prompt: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self._invoke("complete_json", prompt, *args, **kwargs)
        return dict(result or {}) if isinstance(result, Mapping) else {}

    def _invoke(self, method_name: str, prompt: str, *args: Any, **kwargs: Any) -> Any:
        route = _route_from_kwargs(kwargs)
        layer = classify_llm_layer(route, kwargs)
        prompt_tokens = _estimate_tokens(prompt)
        requested_completion = int(kwargs.get("max_tokens", 0) or 0)
        allowed, reason = self._ledger.can_start(
            prompt_tokens=prompt_tokens,
            requested_completion_tokens=requested_completion,
            layer=layer,
            route_name=route,
        )
        if not allowed:
            raise RuntimeError(f"llm_budget_exceeded:{reason}")
        method = getattr(self._client, method_name)
        started_at = time.perf_counter()
        success = False
        error = ""
        result: Any = None
        try:
            result = method(prompt, *args, **self._supported_kwargs(method, kwargs))
            success = True
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            wall_seconds = max(0.0, time.perf_counter() - started_at)
            usage = self._last_usage()
            output_text = "" if result is None or isinstance(result, Mapping) else str(result)
            output_est = _estimate_tokens(output_text)
            self._ledger.record(
                LLMCallCost(
                    call_id=f"llm_call_{len(self._ledger.records()) + 1:04d}",
                    method=method_name,
                    route_name=route,
                    layer=layer,
                    model=str(getattr(self._client, "model", "") or ""),
                    prompt_tokens_estimated=prompt_tokens,
                    requested_completion_tokens=requested_completion,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    cached_input_tokens=int(usage.get("cached_input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                    completion_tokens_estimated=output_est,
                    wall_seconds=round(wall_seconds, 6),
                    success=success,
                    error=error,
                    metadata={
                        "thinking_budget": kwargs.get("thinking_budget"),
                        "think": kwargs.get("think"),
                        "timeout_sec": kwargs.get("timeout_sec"),
                    },
                )
            )

    def _last_usage(self) -> Dict[str, Any]:
        candidate = getattr(self._client, "last_usage", None)
        if callable(candidate):
            try:
                payload = candidate()
            except Exception:
                payload = {}
        else:
            payload = candidate
        return dict(payload or {}) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _supported_kwargs(method: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return dict(kwargs)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return dict(kwargs)
        allowed = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in kwargs.items() if key in allowed}


def wrap_with_budget(client: Any, ledger: Optional[LLMCostLedger]) -> Any:
    if client is None or ledger is None or isinstance(client, BudgetAwareLLMClient):
        return client
    return BudgetAwareLLMClient(client, ledger)


def amplification_efficiency(
    *,
    verified_success_rate_os: float,
    verified_success_rate_baseline: float,
    cost_os: float,
    cost_baseline: float,
) -> Dict[str, Any]:
    if verified_success_rate_baseline <= 0 or cost_os <= 0 or cost_baseline <= 0:
        return {
            "schema_version": LLM_BUDGET_VERSION,
            "amplification_efficiency": None,
            "undefined_reason": "zero_baseline_success_or_nonpositive_cost",
            "absolute_success_delta": float(verified_success_rate_os) - float(verified_success_rate_baseline),
        }
    success_ratio = float(verified_success_rate_os) / float(verified_success_rate_baseline)
    cost_ratio = float(cost_os) / float(cost_baseline)
    return {
        "schema_version": LLM_BUDGET_VERSION,
        "verified_success_rate_os": float(verified_success_rate_os),
        "verified_success_rate_baseline": float(verified_success_rate_baseline),
        "cost_os": float(cost_os),
        "cost_baseline": float(cost_baseline),
        "success_ratio": success_ratio,
        "cost_ratio": cost_ratio,
        "amplification_efficiency": success_ratio / cost_ratio,
        "absolute_success_delta": float(verified_success_rate_os) - float(verified_success_rate_baseline),
    }

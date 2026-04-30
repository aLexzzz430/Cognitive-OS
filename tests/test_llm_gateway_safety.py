from __future__ import annotations

from modules.llm.gateway import LLMGateway


class _StrictClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        return "ok"


class _KwargsClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return "ok"


class _FailingClient:
    def complete(self, prompt: str, **kwargs) -> str:
        raise TimeoutError("remote model timeout")


class _LooseJsonClient:
    def complete(self, prompt: str, **kwargs) -> str:
        return "Sure.\n{'answer': 'ok', 'confidence': 0.9}\nDone."


class _BadJsonClient:
    def complete(self, prompt: str, **kwargs) -> str:
        return "I cannot format that right now."


def _gateway(client, route: str = "retrieval") -> LLMGateway:
    return LLMGateway(route_name=route, capability_prefix=route, client_resolver=lambda *_args: client)


def test_gateway_applies_lightweight_defaults_for_advisory_routes() -> None:
    client = _KwargsClient()
    gateway = _gateway(client, route="retrieval")

    assert gateway.request_text("query_rewrite", "hello") == "ok"

    kwargs = client.calls[0]["kwargs"]
    assert kwargs["think"] is False
    assert kwargs["max_tokens"] <= 64
    assert kwargs["timeout_sec"] <= 5.0


def test_gateway_filters_default_kwargs_for_strict_clients() -> None:
    client = _StrictClient()
    gateway = _gateway(client, route="retrieval")

    assert gateway.request_text("query_rewrite", "hello") == "ok"

    assert client.calls[0]["max_tokens"] <= 64
    assert client.calls[0]["temperature"] == 0.0


def test_gateway_returns_empty_string_on_model_timeout() -> None:
    gateway = _gateway(_FailingClient(), route="retrieval")

    assert gateway.request_text("query_rewrite", "hello") == ""
    assert "TimeoutError" in gateway.last_error
    assert gateway.last_failure_policy_trace[-1]["recommended_action"] == "return_structured_timeout"
    assert gateway.last_failure_policy_trace[-1]["fallback_patch_allowed"] is False


def test_gateway_request_json_uses_output_adapter_for_loose_model_text() -> None:
    gateway = _gateway(_LooseJsonClient(), route="structured_answer")

    payload = gateway.request_json("kwargs", "return json")

    assert payload == {"answer": "ok", "confidence": 0.9}


def test_gateway_fails_over_on_model_timeout_when_resolver_configured() -> None:
    fallback = _StrictClient()
    gateway = LLMGateway(
        route_name="retrieval",
        capability_prefix="retrieval",
        client_resolver=lambda *_args: _FailingClient(),
        fallback_client_resolver=lambda *_args: fallback,
    )

    assert gateway.request_text("query_rewrite", "hello") == "ok"
    assert gateway.last_error == ""
    assert gateway.last_failover_trace[-1]["status"] == "used"
    assert gateway.last_failover_trace[-1]["reason"] == "model_error:TimeoutError"
    assert gateway.last_failover_trace[-1]["failure_policy"]["recommended_action"] == "try_configured_model_fallback_after_timeout"
    assert fallback.calls[0]["prompt"] == "hello"


def test_gateway_fails_over_on_json_format_error_when_resolver_configured() -> None:
    fallback = _LooseJsonClient()
    gateway = LLMGateway(
        route_name="structured_answer",
        capability_prefix="structured_answer",
        client_resolver=lambda *_args: _BadJsonClient(),
        fallback_client_resolver=lambda *_args: fallback,
    )

    payload = gateway.request_json("kwargs", "return json")

    assert payload == {"answer": "ok", "confidence": 0.9}
    assert gateway.last_failover_trace[-1]["status"] == "used"
    assert gateway.last_failover_trace[-1]["reason"].startswith("format_error:")
    assert gateway.last_failover_trace[-1]["failure_policy"]["fallback_patch_allowed"] is False

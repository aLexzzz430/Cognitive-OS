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

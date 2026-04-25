from __future__ import annotations

from modules.episodic.llm_interface import LLMRetrievalContext, LLMRetrievalInterface


def _ctx() -> LLMRetrievalContext:
    return LLMRetrievalContext(
        episode=1,
        tick=0,
        phase="active",
        discovered_functions=[],
        available_functions=["repo_tree", "file_read"],
        active_hypotheses=0,
        confirmed_hypotheses=0,
        entropy=0.5,
        margin=0.0,
        is_saturated=False,
    )


class _RecordingClient:
    def __init__(self, response: str = "focused local repo tree") -> None:
        self.calls = []
        self.response = response

    def complete(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return self.response


class _FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, **kwargs):
        self.calls += 1
        raise TimeoutError("remote ollama timeout")


def test_retrieval_query_rewrite_uses_lightweight_no_think_request() -> None:
    client = _RecordingClient()
    iface = LLMRetrievalInterface(client)

    assert iface.query_rewrite("active plan_target:repo_tree", _ctx()) == "focused local repo tree"

    kwargs = client.calls[0]["kwargs"]
    assert kwargs["think"] is False
    assert kwargs["max_tokens"] <= 64
    assert kwargs["timeout_sec"] <= 5.0
    assert kwargs["temperature"] == 0.0


def test_retrieval_query_rewrite_falls_back_on_timeout() -> None:
    client = _FailingClient()
    iface = LLMRetrievalInterface(client)

    base_query = "active plan_target:repo_tree"

    assert iface.query_rewrite(base_query, _ctx()) == base_query
    assert "TimeoutError" in iface.last_llm_error


def test_retrieval_llm_circuit_breaker_opens_after_repeated_failures() -> None:
    client = _FailingClient()
    iface = LLMRetrievalInterface(client)
    base_query = "active plan_target:repo_tree"

    assert iface.query_rewrite(base_query, _ctx()) == base_query
    assert iface.query_rewrite(base_query, _ctx()) == base_query

    assert iface.can_advise_retrieval_gate() is False
    assert client.calls == 2

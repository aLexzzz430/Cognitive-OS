from __future__ import annotations

from modules.hypothesis.llm_interface import LLMHypothesisInterface


class _RecordingClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return self.response


class _FailingClient:
    def complete(self, prompt: str, **kwargs):
        raise TimeoutError("remote ollama timeout")


def test_hypothesis_generation_uses_lightweight_no_think_request() -> None:
    client = _RecordingClient(
        '[{"claim":"repo_tree should run first","hyp_type":"function_existence","confidence":0.7,"competing_with":[]}]'
    )
    iface = LLMHypothesisInterface(hypothesis_tracker=None, llm_client=client)

    candidates = iface.generate_hypothesis_candidates(
        obs={},
        context="local-machine inventory",
        known_functions=["repo_tree", "file_read"],
    )

    assert candidates[0]["claim"] == "repo_tree should run first"
    kwargs = client.calls[0]["kwargs"]
    assert kwargs["think"] is False
    assert kwargs["max_tokens"] <= 256
    assert kwargs["timeout_sec"] <= 6.0


def test_hypothesis_generation_returns_empty_on_timeout() -> None:
    iface = LLMHypothesisInterface(hypothesis_tracker=None, llm_client=_FailingClient())

    assert iface.generate_hypothesis_candidates({}, "ctx", ["repo_tree"]) == []
    assert "TimeoutError" in iface.last_llm_error

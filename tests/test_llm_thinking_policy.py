from __future__ import annotations

from modules.llm.capabilities import GENERAL_REASONING
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.model_router import ModelRouter
from modules.llm.ollama_client import OllamaClient
from modules.llm.thinking_policy import apply_thinking_policy, thinking_policy_for_route


def test_thinking_policy_uses_no_thinking_for_cheap_routes() -> None:
    decision = thinking_policy_for_route("structured_answer")

    assert decision.think is False
    assert decision.thinking_budget == 0
    assert decision.timeout_sec <= 8.0


def test_thinking_policy_budgets_hard_patch_decisions() -> None:
    kwargs = apply_thinking_policy("patch_proposal", {"max_tokens": 900})

    assert kwargs["think"] is True
    assert kwargs["thinking_budget"] == 1024
    assert kwargs["timeout_sec"] >= 90.0


def test_plan_generation_uses_unbounded_thinking_and_long_timeout() -> None:
    decision = thinking_policy_for_route("plan_generation")
    planner_decision = thinking_policy_for_route("planner")
    kwargs = apply_thinking_policy("plan_generation", {"timeout_sec": 30.0})

    assert decision.think is True
    assert decision.thinking_budget is None
    assert decision.prefer_strongest_model is True
    assert planner_decision.thinking_budget is None
    assert planner_decision.prefer_strongest_model is True
    assert kwargs["think"] is True
    assert "thinking_budget" not in kwargs
    assert kwargs["timeout_sec"] >= 300.0


def test_gateway_applies_route_thinking_policy_to_requests() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.kwargs = {}

        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            self.kwargs = dict(kwargs)
            return "ok"

    llm = FakeLLM()
    gateway = ensure_llm_gateway(llm, route_name="plan_generation", capability_prefix="general")

    assert gateway is not None
    assert gateway.request_raw(GENERAL_REASONING, "make a plan") == "ok"
    assert llm.kwargs["think"] is True
    assert "thinking_budget" not in llm.kwargs
    assert llm.kwargs["timeout_sec"] >= 300.0


def test_planning_route_prefers_high_trust_model_over_fast_model() -> None:
    fast = object()
    strong = object()
    router = ModelRouter(default_client=fast, route_specs={
        "fast_planner": {
            "served_routes": ["plan_generation"],
            "client_alias": "fast",
            "capability_profile": {
                "capabilities": ["reasoning", "planning"],
                "trust_score": 0.35,
                "cost_efficiency": 0.95,
                "latency_efficiency": 0.95,
                "uncertainty_tolerance": 0.4,
                "verification_strength": 0.3,
            },
        },
        "strong_planner": {
            "served_routes": ["plan_generation"],
            "client_alias": "strong",
            "capability_profile": {
                "capabilities": ["reasoning", "planning"],
                "trust_score": 0.95,
                "cost_efficiency": 0.2,
                "latency_efficiency": 0.2,
                "uncertainty_tolerance": 0.95,
                "verification_strength": 0.8,
            },
        },
    })
    router.register_client("fast", fast)
    router.register_client("strong", strong)

    decision = router.decide(
        "plan_generation",
        context={
            "required_capabilities": ["reasoning", "planning"],
            "prefer_high_trust": 0.98,
            "prefer_low_cost": 0.0,
            "prefer_low_latency": 0.0,
            "uncertainty_level": 0.9,
        },
    )

    assert decision.client is strong
    assert decision.route_name == "strong_planner"


def test_ollama_client_forwards_thinking_budget(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": "ok"}}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["json"] = dict(json)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(base_url="http://ollama.test", model="qwen3:8b", timeout_sec=10.0)
    assert client.complete_raw("hello", think=True, thinking_budget=512, timeout_sec=42.0) == "ok"

    payload = captured["json"]
    assert payload["think"] is True
    assert payload["options"]["thinking_budget"] == 512
    assert captured["timeout"] == 42.0


def test_ollama_client_disables_thinking_for_zero_budget(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": "ok"}}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        captured["json"] = dict(json)
        return FakeResponse()

    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(base_url="http://ollama.test", model="qwen3:8b")
    assert client.complete_raw("hello", think=True, thinking_budget=0) == "ok"

    payload = captured["json"]
    assert payload["think"] is False
    assert "thinking_budget" not in payload["options"]

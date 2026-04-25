from __future__ import annotations

import json
from pathlib import Path

from modules.llm.cli import main as llm_cli_main
from modules.llm.model_profile import (
    MODEL_PROFILE_REPORT_VERSION,
    MODEL_PROFILE_VERSION,
    ModelProfileStore,
    build_model_profile,
    profile_ollama_models,
    route_policies_from_profiles,
)
from modules.llm.model_router import ModelRouter


class _FakeProfileClient:
    def __init__(self, *, self_scores=None) -> None:
        self.calls = []
        self.self_scores = self_scores or {
            "reasoning": 0.8,
            "planning": 0.7,
            "structured_output": 0.6,
            "tool_use": 0.5,
            "verification": 0.7,
            "speed": 0.4,
            "instruction_following": 0.6,
            "retrieval": 0.5,
            "coding": 0.6,
            "long_context": 0.5,
        }

    def complete(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if "SELF_PROFILE_PROBE" in prompt:
            return json.dumps(
                {
                    "schema_version": "conos.model_self_profile/v1",
                    "declared_strengths": ["reasoning", "verification"],
                    "declared_weaknesses": ["latency"],
                    "capability_scores": self.self_scores,
                    "preferred_task_types": ["hypothesis", "recovery"],
                    "avoid_task_types": ["high_frequency_retrieval"],
                    "confidence_in_self_report": 0.55,
                }
            )
        if "STRUCTURED_JSON_PROBE" in prompt:
            return '{"answer":"ok","items":[1,2,3]}'
        if "INSTRUCTION_FOLLOWING_PROBE" in prompt:
            return '{"selected_id":"beta"}'
        if "REASONING_PROBE" in prompt:
            return '{"strongest":"A"}'
        if "VERIFICATION_PROBE" in prompt:
            return '{"false_claim":"claim_2"}'
        return "{}"


def test_build_model_profile_combines_self_report_and_probe_scores() -> None:
    profile = build_model_profile(
        client=_FakeProfileClient(),
        provider="ollama",
        base_url="http://fake-ollama",
        model="qwen3:8b",
    )

    assert profile["schema_version"] == MODEL_PROFILE_VERSION
    assert profile["model"] == "qwen3:8b"
    assert profile["self_profile"]["declared_strengths"] == ["reasoning", "verification"]
    assert profile["calibration"]["passed_count"] == 4
    assert profile["capability_scores"]["reasoning"] > 0.8
    assert profile["capability_scores"]["structured_output"] > 0.7


def test_model_profile_store_round_trips_profiles(tmp_path: Path) -> None:
    store = ModelProfileStore(tmp_path / "profiles.json")
    profile = build_model_profile(
        client=_FakeProfileClient(),
        provider="ollama",
        base_url="http://fake-ollama",
        model="qwen3:8b",
    )

    store.upsert_profile(profile)
    loaded = store.get_profile(provider="ollama", base_url="http://fake-ollama", model="qwen3:8b")

    assert loaded is not None
    assert loaded["model"] == "qwen3:8b"
    assert len(store.list_profiles()) == 1


def test_route_policies_from_profiles_allow_router_to_pick_task_specific_model() -> None:
    profiles = [
        {
            "schema_version": MODEL_PROFILE_VERSION,
            "provider": "ollama",
            "base_url": "http://fake-ollama",
            "model": "fast-small",
            "profiled_at": "now",
            "capability_scores": {
                "reasoning": 0.45,
                "planning": 0.35,
                "structured_output": 0.35,
                "verification": 0.3,
                "speed": 0.95,
                "instruction_following": 0.9,
                "retrieval": 0.9,
            },
        },
        {
            "schema_version": MODEL_PROFILE_VERSION,
            "provider": "ollama",
            "base_url": "http://fake-ollama",
            "model": "json-strong",
            "profiled_at": "now",
            "capability_scores": {
                "reasoning": 0.7,
                "planning": 0.6,
                "structured_output": 0.95,
                "verification": 0.9,
                "speed": 0.35,
                "instruction_following": 0.6,
                "retrieval": 0.4,
            },
        },
    ]
    policies = route_policies_from_profiles(profiles, base_url="http://fake-ollama")
    router = ModelRouter(route_specs=policies)

    structured = router.decide(
        "structured_answer",
        context={
            "required_capabilities": ["structured_output"],
            "prefer_structured_output": 1.0,
            "verification_pressure": 0.6,
        },
    )
    retrieval = router.decide(
        "retrieval",
        context={
            "required_capabilities": ["retrieval"],
            "prefer_low_latency": 1.0,
            "prefer_low_cost": 0.8,
        },
    )

    assert structured.metadata["model"] == "json-strong"
    assert retrieval.metadata["model"] == "fast-small"


class _Response:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self._payload


def test_profile_ollama_models_discovers_models_and_writes_route_policy(monkeypatch, tmp_path: Path) -> None:
    def fake_get(url, timeout):
        return _Response({"models": [{"name": "qwen3:8b"}]})

    def fake_post(url, json, timeout):
        prompt = json["messages"][-1]["content"]
        text = _FakeProfileClient().complete(prompt)
        return _Response({"message": {"content": text}})

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)
    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)

    report = profile_ollama_models(
        base_url="http://fake-ollama",
        store_path=tmp_path / "profiles.json",
        timeout_sec=2,
        force=True,
    )

    assert report["schema_version"] == MODEL_PROFILE_REPORT_VERSION
    assert report["model_count"] == 1
    assert report["generated_count"] == 1
    assert "ollama_qwen3_8b" in report["route_policies"]


def test_llm_cli_profile_writes_profile_store_and_route_policies(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_get(url, timeout):
        return _Response({"models": [{"name": "qwen3:8b"}]})

    def fake_post(url, json, timeout):
        prompt = json["messages"][-1]["content"]
        text = _FakeProfileClient().complete(prompt)
        return _Response({"message": {"content": text}})

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)
    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)
    store_path = tmp_path / "profiles.json"
    policy_path = tmp_path / "route_policies.json"

    assert llm_cli_main(
        [
            "--base-url",
            "http://fake-ollama",
            "--timeout",
            "2",
            "profile",
            "--store",
            str(store_path),
            "--route-policy-output",
            str(policy_path),
            "--force",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    policies = json.loads(policy_path.read_text(encoding="utf-8"))
    assert payload["generated_count"] == 1
    assert store_path.exists()
    assert "ollama_qwen3_8b" in policies

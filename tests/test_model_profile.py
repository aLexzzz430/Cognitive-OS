from __future__ import annotations

import json
from pathlib import Path

from modules.llm.cli import main as llm_cli_main
from modules.llm.model_profile import (
    MODEL_PROFILE_REPORT_VERSION,
    MODEL_PROFILE_VERSION,
    ModelProfileStore,
    build_model_profile,
    build_model_route_summary,
    profile_all_configured_models,
    load_profile_backed_route_policies,
    profile_ollama_models,
    profile_openai_models,
    render_model_route_summary,
    route_policies_from_profiles,
    write_model_route_policies,
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


def test_strict_route_guard_excludes_weak_models_from_strict_routes() -> None:
    policies = route_policies_from_profiles(
        [
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "ollama",
                "base_url": "http://fake-ollama",
                "model": "weak-json-talkative",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.0,
                    "planning": 0.0,
                    "structured_output": 0.56,
                    "verification": 0.0,
                    "speed": 0.42,
                    "instruction_following": 0.0,
                    "retrieval": 0.42,
                },
            }
        ],
        base_url="http://fake-ollama",
    )
    policy = policies["ollama_weak_json_talkative"]
    eligibility = policy["capability_profile"]["metadata"]["route_eligibility"]

    assert "probe" not in policy["served_routes"]
    assert "structured_answer" not in policy["served_routes"]
    assert "analyst" not in policy["served_routes"]
    assert eligibility["probe"]["blocked_reason"].startswith("strict_route_minimum_failed")


def test_route_policies_use_profile_provider_prefix_for_cloud_models() -> None:
    policies = route_policies_from_profiles(
        [
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "openai",
                "base_url": "https://fake-openai",
                "model": "gpt-test",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.8,
                    "planning": 0.7,
                    "structured_output": 0.9,
                    "verification": 0.85,
                    "speed": 0.5,
                    "instruction_following": 0.8,
                    "retrieval": 0.5,
                    "coding": 0.75,
                },
            }
        ],
        provider="openai",
        base_url="https://fake-openai",
    )

    assert "openai_gpt_test" in policies
    assert policies["openai_gpt_test"]["provider"] == "openai"
    assert policies["openai_gpt_test"]["base_url"] == "https://fake-openai"


def test_route_summary_explains_selected_models_and_deprioritized_models() -> None:
    policies = route_policies_from_profiles(
        [
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
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "ollama",
                "base_url": "http://fake-ollama",
                "model": "too-weak",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.0,
                    "planning": 0.0,
                    "structured_output": 0.2,
                    "verification": 0.0,
                    "speed": 0.2,
                    "instruction_following": 0.0,
                    "retrieval": 0.2,
                },
            },
        ],
        base_url="http://fake-ollama",
    )
    summary = build_model_route_summary(policies, routes=["structured_answer", "retrieval"], explain=True)
    rendered = render_model_route_summary(summary)
    selected = {row["route"]: row["selected_model"] for row in summary["routes"]}

    assert selected["structured_answer"] == "json-strong"
    assert selected["retrieval"] == "fast-small"
    assert summary["deprioritized_models"] == [
        {
            "route_policy": "ollama_too_weak",
            "model": "too-weak",
            "reason": "profile_route_scores_below_threshold_or_strict_minimums",
        }
    ]
    assert "Con OS model routes" in rendered


def test_load_profile_backed_route_policies_prefers_route_policy_file(tmp_path: Path) -> None:
    policies = {
        "ollama_json_strong": {
            "served_routes": ["structured_answer"],
            "provider": "ollama",
            "base_url": "http://fake-ollama",
            "model": "json-strong",
        }
    }
    policy_path = tmp_path / "route_policies.json"
    write_model_route_policies(policies, policy_path)

    loaded = load_profile_backed_route_policies(route_policy_path=policy_path)

    assert loaded == policies


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


def test_profile_openai_models_discovers_cloud_models_and_writes_route_policy(monkeypatch, tmp_path: Path) -> None:
    def fake_get(url, headers=None, timeout=None):
        return _Response(
            {
                "data": [
                    {"id": "gpt-test"},
                    {"id": "text-embedding-3-small"},
                    {"id": "gpt-json"},
                ]
            }
        )

    def fake_post(url, headers=None, json=None, timeout=None):
        prompt = json["input"]
        text = _FakeProfileClient().complete(prompt)
        return _Response({"output_text": text})

    monkeypatch.setattr("modules.llm.openai_client.requests.get", fake_get)
    monkeypatch.setattr("modules.llm.openai_client.requests.post", fake_post)

    report = profile_openai_models(
        api_key="sk-test",
        base_url="https://fake-openai",
        store_path=tmp_path / "profiles.json",
        timeout_sec=2,
        force=True,
        all_cloud_models=True,
        max_models=1,
    )

    assert report["schema_version"] == MODEL_PROFILE_REPORT_VERSION
    assert report["listed_model_count"] == 2
    assert report["model_count"] == 1
    assert report["generated_count"] == 1
    assert report["error_count"] == 0
    assert "openai_gpt_test" in report["route_policies"]


def test_profile_all_configured_models_merges_local_and_cloud_route_policies(monkeypatch, tmp_path: Path) -> None:
    def fake_ollama_get(url, timeout):
        return _Response({"models": [{"name": "qwen3:8b"}]})

    def fake_post(url, headers=None, json=None, timeout=None):
        if "input" in json:
            prompt = json["input"]
        else:
            prompt = json["messages"][-1]["content"]
        text = _FakeProfileClient().complete(prompt)
        if "input" in json:
            return _Response({"output_text": text})
        return _Response({"message": {"content": text}})

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_ollama_get)
    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)
    monkeypatch.setattr("modules.llm.openai_client.requests.post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    report = profile_all_configured_models(
        ollama_base_url="http://fake-ollama",
        openai_base_url="https://fake-openai",
        openai_models=["gpt-test"],
        store_path=tmp_path / "profiles.json",
        timeout_sec=2,
        force=True,
    )

    assert report["provider"] == "all"
    assert report["generated_count"] == 2
    assert "ollama_qwen3_8b" in report["route_policies"]
    assert "openai_gpt_test" in report["route_policies"]


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


def test_llm_cli_profile_openai_writes_profile_store_and_route_policies(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_post(url, headers=None, json=None, timeout=None):
        prompt = json["input"]
        text = _FakeProfileClient().complete(prompt)
        return _Response({"output_text": text})

    monkeypatch.setattr("modules.llm.openai_client.requests.post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    store_path = tmp_path / "profiles.json"
    policy_path = tmp_path / "route_policies.json"

    assert llm_cli_main(
        [
            "--provider",
            "openai",
            "--base-url",
            "https://fake-openai",
            "--model",
            "gpt-test",
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
    assert "openai_gpt_test" in policies


def test_llm_cli_routes_reads_precomputed_policy_file(tmp_path: Path, capsys) -> None:
    policies = route_policies_from_profiles(
        [
            {
                "schema_version": MODEL_PROFILE_VERSION,
                "provider": "ollama",
                "base_url": "http://fake-ollama",
                "model": "json-strong",
                "profiled_at": "now",
                "capability_scores": {
                    "reasoning": 0.8,
                    "planning": 0.7,
                    "structured_output": 0.95,
                    "verification": 0.9,
                    "speed": 0.5,
                    "instruction_following": 0.8,
                    "retrieval": 0.5,
                },
            }
        ],
        base_url="http://fake-ollama",
    )
    policy_path = tmp_path / "route_policies.json"
    write_model_route_policies(policies, policy_path)

    assert llm_cli_main(["routes", "--route-policy-file", str(policy_path), "--format", "json", "--explain"]) == 0

    payload = json.loads(capsys.readouterr().out)
    route_rows = {row["route"]: row for row in payload["routes"]}
    assert payload["route_policy_count"] == 1
    assert route_rows["structured_answer"]["selected_model"] == "json-strong"
    assert route_rows["structured_answer"]["candidates"]

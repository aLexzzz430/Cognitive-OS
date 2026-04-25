from __future__ import annotations

import json

from modules.llm.cli import main as llm_cli_main
from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient


class _Response:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self):
        return self._payload


def test_ollama_client_uses_lan_base_url_and_lists_models(monkeypatch) -> None:
    calls = []

    def fake_get(url, timeout):
        calls.append({"url": url, "timeout": timeout})
        return _Response({"models": [{"name": "qwen3:8b"}, {"name": "llama3.1:8b"}]})

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)

    client = OllamaClient(base_url="http://192.168.1.50:11434", auto_select_model=False, timeout_sec=3)

    assert client.base_url == "http://192.168.1.50:11434"
    assert client.list_models() == ["qwen3:8b", "llama3.1:8b"]
    assert calls == [{"url": "http://192.168.1.50:11434/api/tags", "timeout": 3.0}]


def test_ollama_client_health_reports_disconnected_without_raising(monkeypatch) -> None:
    def fake_get(url, timeout):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)

    health = OllamaClient(base_url="http://192.168.1.50:11434", auto_select_model=False).health()

    assert health["connected"] is False
    assert health["base_url"] == "http://192.168.1.50:11434"
    assert "connection refused" in health["error"]


def test_ollama_client_prompt_posts_to_remote_chat_endpoint(monkeypatch) -> None:
    posts = []

    def fake_post(url, json, timeout):
        posts.append({"url": url, "json": json, "timeout": timeout})
        return _Response({"message": {"content": "pong"}})

    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(base_url="http://10.0.0.8:11434", model="qwen3:4b", timeout_sec=4)

    assert client.complete("ping", max_tokens=16, temperature=0.1) == "pong"
    assert posts[0]["url"] == "http://10.0.0.8:11434/api/chat"
    assert posts[0]["json"]["model"] == "qwen3:4b"
    assert posts[0]["json"]["messages"][-1] == {"role": "user", "content": "ping"}
    assert posts[0]["timeout"] == 4.0


def test_ollama_client_supports_per_call_timeout_and_think_flag(monkeypatch) -> None:
    posts = []

    def fake_post(url, json, timeout):
        posts.append({"url": url, "json": json, "timeout": timeout})
        return _Response({"message": {"content": "rewritten"}})

    monkeypatch.setattr("modules.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(base_url="http://10.0.0.8:11434", model="batiai/gemma4-e4b:q4", timeout_sec=60)

    assert client.complete("rewrite", max_tokens=32, temperature=0.0, think=False, timeout_sec=5) == "rewritten"
    assert posts[0]["timeout"] == 5.0
    assert posts[0]["json"]["think"] is False
    assert posts[0]["json"]["options"]["num_predict"] == 32


def test_llm_cli_check_reports_remote_ollama_health(monkeypatch, capsys) -> None:
    def fake_get(url, timeout):
        return _Response({"models": [{"name": "qwen3:8b"}]})

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)

    assert llm_cli_main(["--base-url", "http://192.168.1.50:11434", "check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["connected"] is True
    assert payload["base_url"] == "http://192.168.1.50:11434"
    assert payload["models"] == ["qwen3:8b"]


def test_llm_cli_check_returns_nonzero_when_remote_ollama_is_unreachable(monkeypatch, capsys) -> None:
    def fake_get(url, timeout):
        raise RuntimeError("no route to host")

    monkeypatch.setattr("modules.llm.ollama_client.requests.get", fake_get)

    assert llm_cli_main(["--base-url", "http://192.168.1.50:11434", "check"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["connected"] is False
    assert "no route to host" in payload["error"]


def test_default_ollama_url_matches_standard_server_port(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    client = OllamaClient(auto_select_model=False)

    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert DEFAULT_OLLAMA_BASE_URL.endswith(":11434")

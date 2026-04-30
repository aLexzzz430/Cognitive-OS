from __future__ import annotations

from pathlib import Path

from modules.llm.codex_cli_client import DEFAULT_CODEX_MODEL, CodexCliClient
from modules.llm.factory import build_llm_client


class _Completed:
    def __init__(self, *, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_codex_cli_client_uses_output_last_message_and_oauth_login(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), "kwargs": dict(kwargs)})
        if command[:3] == ["codex", "login", "status"]:
            return _Completed(stdout="Logged in using ChatGPT\n")
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text("MODEL ANSWER\n", encoding="utf-8")
        return _Completed(stdout='{"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":7,"cached_input_tokens":3}}\n')

    monkeypatch.setattr("modules.llm.codex_cli_client.subprocess.run", fake_run)

    client = CodexCliClient(model="gpt-5.3-codex", cwd=str(tmp_path), timeout_sec=12)

    assert client.health()["connected"] is True
    assert client.complete("hello", think=False, timeout_sec=7) == "MODEL ANSWER"
    command = calls[1]["command"]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("-m") + 1] == "gpt-5.3-codex"
    assert "--json" in command
    assert command[-1] == "-"
    assert 'model_reasoning_effort="low"' in command
    assert calls[1]["kwargs"]["input"].endswith("hello\n")
    assert calls[1]["kwargs"]["timeout"] == 20.0
    assert client.last_usage()["input_tokens"] == 11
    assert client.last_usage()["output_tokens"] == 7


def test_codex_cli_client_defaults_to_spark(tmp_path: Path) -> None:
    client = CodexCliClient(cwd=str(tmp_path))

    assert client.model == "gpt-5.3-codex-spark"
    assert client.model == DEFAULT_CODEX_MODEL


def test_codex_cli_factory_aliases_to_oauth_cli_client() -> None:
    client = build_llm_client("codex-cli", model="gpt-5.3-codex")

    assert isinstance(client, CodexCliClient)
    assert client.model == "gpt-5.3-codex"


def test_codex_cli_client_surfaces_model_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        return _Completed(returncode=1, stdout="not logged in")

    monkeypatch.setattr("modules.llm.codex_cli_client.subprocess.run", fake_run)
    client = CodexCliClient(model="gpt-5.3-codex", cwd=str(tmp_path))

    try:
        client.complete("hello")
    except RuntimeError as exc:
        assert "not logged in" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

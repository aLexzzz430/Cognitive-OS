from __future__ import annotations

import conos_cli
from core.auth.codex_cli_oauth import codex_login_status, run_codex_login, run_codex_logout


class _Completed:
    def __init__(self, *, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_codex_login_status_delegates_to_codex_cli_without_token_access() -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), "kwargs": dict(kwargs)})
        return _Completed(stdout="Logged in using ChatGPT\n")

    status = codex_login_status(command="codex-test", runner=fake_run)

    assert calls[0]["command"] == ["codex-test", "login", "status"]
    assert status["connected"] is True
    assert status["auth_type"] == "chatgpt_oauth_delegate"
    assert status["token_access"] == "delegated_to_codex_cli"
    assert status["quota_scope"] == "chatgpt_codex_plan_or_api_org_via_codex_cli"


def test_codex_login_and_logout_commands_are_bounded_cli_delegates() -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["codex-test", "login"] and len(command) == 2:
            return _Completed(stdout="login complete\n")
        if command == ["codex-test", "login", "status"]:
            return _Completed(stdout="Logged in using ChatGPT\n")
        if command == ["codex-test", "logout"]:
            return _Completed(stdout="logged out\n")
        return _Completed(returncode=1, stdout="unexpected\n")

    login = run_codex_login(command="codex-test", runner=fake_run)
    logout = run_codex_logout(command="codex-test", runner=fake_run)

    assert calls[0] == ["codex-test", "login"]
    assert calls[1] == ["codex-test", "login", "status"]
    assert calls[2] == ["codex-test", "logout"]
    assert login["ok"] is True
    assert login["status"]["connected"] is True
    assert logout["ok"] is True


def test_product_cli_routes_codex_auth_provider(monkeypatch, capsys) -> None:
    def fake_run(command, **kwargs):
        return _Completed(stdout="Logged in using ChatGPT\n")

    monkeypatch.setattr("core.auth.codex_cli_oauth.subprocess.run", fake_run)

    assert conos_cli.main(["auth", "codex", "status"]) == 0
    payload = capsys.readouterr().out
    assert '"provider": "codex-cli"' in payload
    assert '"connected": true' in payload

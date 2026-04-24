from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import conos_cli


def test_product_cli_version_prints_product_metadata(capsys) -> None:
    assert conos_cli.main(["version"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["product"] == "Cognitive OS"
    assert payload["entrypoint"] == "conos"
    assert "run" in payload["commands"]
    assert "ui" in payload["commands"]
    assert "app" in payload["commands"]
    assert "auth" in payload["commands"]
    assert "mirror" in payload["commands"]
    assert "dashboard" in payload["commands"]
    assert payload["auth_providers"] == ["openai"]


def test_product_cli_delegates_arc_agi3_runner(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 7

    from integrations.arc_agi3 import runner

    monkeypatch.setattr(runner, "main", fake_main)

    assert conos_cli.main(["run", "arc-agi3", "--game", "vc33", "--max-ticks", "1"]) == 7
    assert captured["argv"] == ["--game", "vc33", "--max-ticks", "1"]


def test_product_cli_delegates_local_machine_runner(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 5

    from integrations.local_machine import runner

    monkeypatch.setattr(runner, "main", fake_main)

    assert conos_cli.main(["run", "local-machine", "--instruction", "inspect README"]) == 5
    assert captured["argv"] == ["--instruction", "inspect README"]


def test_product_cli_dashboard_renders_eval_panel(tmp_path: Path, capsys) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "run_id": "dash-run",
                "total_reward": 1.0,
                "planner_runtime_log": [
                    {
                        "verifier_runtime": {
                            "verifier_authority": {
                                "required": True,
                                "verdict": "passed",
                                "verifier_function": "check_goal",
                            }
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert conos_cli.main(["dashboard", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Cognitive OS evaluation metrics" in output
    assert "verified_success_rate" in output
    assert "100.0%" in output


def test_product_cli_openai_auth_status_reports_configuration(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-1")
    monkeypatch.setenv("OPENAI_OAUTH_AUTHORIZATION_URL", "https://auth.example.test/authorize")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://auth.example.test/token")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_STORE", str(tmp_path / "token.json"))

    assert conos_cli.main(["auth", "openai", "status"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["configured"] is True
    assert payload["token_present"] is False
    assert payload["redirect_uri"] == "http://127.0.0.1:8767/oauth/openai/callback"

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
    assert "llm" in payload["commands"]
    assert "supervisor" in payload["commands"]
    assert "dashboard" in payload["commands"]
    assert payload["auth_providers"] == ["openai"]
    assert payload["llm_providers"] == ["ollama"]


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


def test_product_cli_delegates_llm_cli(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 9

    from modules.llm import cli

    monkeypatch.setattr(cli, "main", fake_main)

    assert conos_cli.main(["llm", "check", "--base-url", "http://lan-host:11434"]) == 9
    assert captured["argv"] == ["check", "--base-url", "http://lan-host:11434"]


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


def test_product_cli_supervisor_create_tick_and_status(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "state.sqlite3"
    runs_root = tmp_path / "runs"
    base_args = ["supervisor", "--db", str(db_path), "--runs-root", str(runs_root)]

    assert conos_cli.main([*base_args, "create", "--goal", "stay alive", "--run-id", "cli-run"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["run_id"] == "cli-run"

    assert conos_cli.main([*base_args, "add-task", "cli-run", "--objective", "step one"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["status"] == "PENDING"

    assert conos_cli.main([*base_args, "tick", "cli-run"]) == 0
    ticked = json.loads(capsys.readouterr().out)
    assert ticked["status"] == "TASK_STARTED"

    assert conos_cli.main([*base_args, "status", "cli-run"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["run"]["status"] == "RUNNING"
    assert status["tasks"][0]["status"] == "RUNNING"


def test_supervisor_service_template_enables_auto_restart(tmp_path: Path) -> None:
    from core.runtime.supervisor_cli import generate_service_template

    launchd = generate_service_template(
        run_id="service-run",
        backend="launchd",
        repo_root=tmp_path,
        python=sys.executable,
        db_path="runtime/long_run/state.sqlite3",
        runs_root="runtime/runs",
        tick_interval=1.0,
        stdout_log=str(tmp_path / "out.log"),
        stderr_log=str(tmp_path / "err.log"),
    )
    systemd = generate_service_template(
        run_id="service-run",
        backend="systemd",
        repo_root=tmp_path,
        python=sys.executable,
        db_path="runtime/long_run/state.sqlite3",
        runs_root="runtime/runs",
        tick_interval=1.0,
        stdout_log=str(tmp_path / "out.log"),
        stderr_log=str(tmp_path / "err.log"),
    )

    assert "<key>KeepAlive</key>" in launchd["content"]
    assert "RunAtLoad" in launchd["content"]
    assert "StandardOutPath" in launchd["content"]
    assert "Restart=always" in systemd["content"]
    assert "StandardOutput=append:" in systemd["content"]
    assert "conos_cli.py supervisor" in systemd["content"]


def test_supervisor_service_install_and_uninstall_dry_paths(tmp_path: Path) -> None:
    from core.runtime.supervisor_cli import generate_service_template, install_service_file, uninstall_service_file

    rendered = generate_service_template(
        run_id="install-run",
        backend="launchd",
        repo_root=tmp_path,
        python=sys.executable,
        db_path="runtime/long_run/state.sqlite3",
        runs_root="runtime/runs",
        tick_interval=1.0,
    )
    output = tmp_path / "dev.conos.supervisor.install-run.plist"

    dry = install_service_file(
        run_id="install-run",
        backend=rendered["backend"],
        content=rendered["content"],
        output=output,
        dry_run=True,
    )
    installed = install_service_file(
        run_id="install-run",
        backend=rendered["backend"],
        content=rendered["content"],
        output=output,
    )
    removed = uninstall_service_file(run_id="install-run", backend=rendered["backend"], output=output)

    assert dry["installed"] is False
    assert "ProgramArguments" in dry["content"]
    assert installed["installed"] is True
    assert output.exists() is False
    assert removed["removed"] is True


def test_product_cli_supervisor_health_and_soak_test(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "state.sqlite3"
    runs_root = tmp_path / "runs"
    base_args = ["supervisor", "--db", str(db_path), "--runs-root", str(runs_root)]

    assert conos_cli.main([*base_args, "soak-test", "--run-id", "soak-run", "--tasks", "2", "--ticks", "5"]) == 0
    soak = json.loads(capsys.readouterr().out)
    assert soak["status"] == "PASSED"
    assert soak["health"]["metrics"]["run_count"] == 1

    assert conos_cli.main([*base_args, "health", "soak-run"]) == 0
    health = json.loads(capsys.readouterr().out)
    assert health["status"] == "OK"
    assert health["run"]["run"]["run_id"] == "soak-run"

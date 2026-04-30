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
    assert "auth" in payload["commands"]
    assert "mirror" in payload["commands"]
    assert "llm" in payload["commands"]
    assert "vm" in payload["commands"]
    assert "discover-tasks" in payload["commands"]
    assert "supervisor" in payload["commands"]
    assert "setup" in payload["commands"]
    assert "doctor" in payload["commands"]
    assert "ui" not in payload["commands"]
    assert "app" not in payload["commands"]
    assert "dashboard" not in payload["commands"]
    assert payload["run_targets"] == ["local-machine"]
    assert payload["auth_providers"] == ["openai", "codex"]
    assert payload["llm_providers"] == ["ollama", "openai", "codex-cli"]


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


def test_product_cli_delegates_managed_vm_cli(monkeypatch) -> None:
    captured = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 4

    import modules.local_mirror.managed_vm as managed_vm

    monkeypatch.setattr(managed_vm, "main", fake_main)

    assert conos_cli.main(["vm", "report", "--state-root", "/tmp/conos-vm"]) == 4
    assert captured["argv"] == ["report", "--state-root", "/tmp/conos-vm"]


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


def test_runtime_recovery_guidance_covers_common_product_failures() -> None:
    from core.runtime.recovery_guidance import (
        guidance_for_check,
        guidance_for_runtime_status,
        guidance_for_vm_report,
    )
    from core.runtime.recovery_playbook import build_recovery_diagnosis_tree

    missing_runner = guidance_for_vm_report(
        {
            "virtualization_runner_available": False,
            "image_manifest_present": False,
            "base_image_present": False,
            "status": "UNAVAILABLE",
        }
    )
    issue_codes = {item["issue"] for item in missing_runner}
    assert "missing_vm_runner" in issue_codes
    assert "missing_vm_image" in issue_codes
    assert any("conos vm build-runner" in item["next_actions"] for item in missing_runner)

    model = guidance_for_runtime_status(
        {
            "status": "DEGRADED",
            "watchdog": {"degraded_reasons": ["ollama_endpoint_unreachable"]},
            "waiting_approvals": [{"approval_id": "appr-1"}],
        }
    )
    assert {item["issue"] for item in model} == {"model_unavailable", "waiting_for_approval"}

    denied = guidance_for_check({"name": "write_path", "ok": False, "required": True, "detail": "permission denied"})
    assert denied["issue"] == "permission_denied"
    assert "approvals" in " ".join(denied["next_actions"])

    tree = build_recovery_diagnosis_tree(
        {
            "recovery_guidance": [
                *missing_runner,
                *model,
                denied,
                {
                    "issue": "run_failed_or_degraded",
                    "severity": "action_needed",
                    "message": "tests failed",
                    "next_actions": ["conos logs --tail 200"],
                },
            ]
        },
        surface="doctor",
    )
    active = {row["category"]: row for row in tree["categories"] if row["status"] != "OK"}
    assert tree["status"] == "ACTION_NEEDED"
    assert {"vm_boundary", "model_runtime", "approval_permission", "verifier_tests"} <= set(active)
    assert "missing_vm_runner" in active["vm_boundary"]["matched_issues"]
    assert "model_unavailable" in active["model_runtime"]["matched_issues"]
    assert "permission_denied" in active["approval_permission"]["matched_issues"]
    assert "run_failed_or_degraded" in active["verifier_tests"]["matched_issues"]
    assert any("conos vm" in action for action in active["vm_boundary"]["recovery_path"])


def test_product_cli_doctor_includes_operator_guidance(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_vm_report(**kwargs):
        return {
            "status": "UNAVAILABLE",
            "virtualization_runner_available": False,
            "image_manifest_present": False,
            "base_image_present": False,
            "guest_agent_gate": {},
        }

    import modules.local_mirror.managed_vm as managed_vm

    monkeypatch.setattr(managed_vm, "managed_vm_report", fake_vm_report)

    code = conos_cli.main(
        [
            "doctor",
            "--runtime-home",
            str(tmp_path / "runtime"),
            "--repo-root",
            str(REPO_ROOT),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code in {0, 1}
    assert payload["recovery_guidance"]
    issues = {item["issue"] for item in payload["recovery_guidance"]}
    assert "missing_vm_runner" in issues
    assert "missing_vm_image" in issues
    assert "operator_summary" in payload
    assert payload["operator_panel"]["surface"] == "doctor"
    assert payload["operator_panel"]["health"] in {"warning", "needs_action"}
    assert payload["operator_panel"]["next_actions"]
    assert payload["recovery_diagnosis_tree"]["status"] in {"WARN", "ACTION_NEEDED"}
    categories = {
        row["category"]
        for row in payload["recovery_diagnosis_tree"]["categories"]
        if row["status"] != "OK"
    }
    assert "vm_boundary" in categories


def test_product_cli_status_logs_and_approvals_include_operator_panel(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"

    assert conos_cli.main(["status", "--runtime-home", str(runtime_home), "--repo-root", str(REPO_ROOT)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["operator_panel"]["surface"] == "status"
    assert status["operator_panel"]["health"] in {"healthy", "warning", "needs_action"}
    assert "message" in status["operator_panel"]
    assert status["recovery_diagnosis_tree"]["surface"] == "status"

    logs_dir = runtime_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "conos.err.log").write_text("RuntimeError: simulated failure\n", encoding="utf-8")
    assert conos_cli.main(["logs", "--runtime-home", str(runtime_home), "--tail", "20"]) == 0
    logs = json.loads(capsys.readouterr().out)
    assert logs["operator_panel"]["surface"] == "logs"
    assert logs["operator_panel"]["health"] == "warning"
    assert "error" in logs["operator_panel"]["error_signals"]
    assert "conos doctor" in logs["operator_panel"]["next_actions"]
    assert "logs_runtime" in logs["recovery_diagnosis_tree"]["active_categories"]

    assert conos_cli.main(["approvals", "--runtime-home", str(runtime_home)]) == 0
    approvals = json.loads(capsys.readouterr().out)
    assert approvals["operator_panel"]["surface"] == "approvals"
    assert approvals["operator_panel"]["health"] == "healthy"
    assert approvals["operator_panel"]["waiting_count"] == 0
    assert approvals["recovery_diagnosis_tree"]["status"] == "OK"


def test_validate_install_operator_panel_surfaces_next_actions(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"

    assert conos_cli.main(["validate-install", "--runtime-home", str(runtime_home), "--no-vm"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["operator_panel"]["surface"] == "validate-install"
    assert payload["operator_panel"]["health"] in {"warning", "needs_action"}
    assert "setup_manifest" in payload["operator_panel"]["failed_checks"]
    assert any("setup" in action for action in payload["operator_panel"]["next_actions"])
    assert "install_runtime" in payload["recovery_diagnosis_tree"]["active_categories"]


def test_setup_one_click_includes_operator_recovery_panel(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"

    assert (
        conos_cli.main(
            [
                "setup",
                "--one-click",
                "--dry-run",
                "--runtime-home",
                str(runtime_home),
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["operator_panel"]["surface"] == "setup"
    assert payload["operator_panel"]["one_click"] is True
    assert payload["operator_panel"]["dry_run"] is True
    assert payload["operator_panel"]["next_actions"]
    assert "vm_default_boundary" in payload["operator_panel"]["top_issues"]
    assert "vm_boundary" in payload["recovery_diagnosis_tree"]["active_categories"]

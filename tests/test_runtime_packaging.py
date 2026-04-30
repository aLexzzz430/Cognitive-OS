from __future__ import annotations

import json
import plistlib
import sys
from pathlib import Path

import conos_cli
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds
from core.runtime.runtime_service import RuntimeService, RuntimeServiceConfig
from core.runtime.service_daemon import tick_runtime_once


def _service(tmp_path: Path, **kwargs) -> RuntimeService:
    home = tmp_path / "home"
    runtime_home = tmp_path / "runtime-home"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = RuntimeServiceConfig.from_args(
        runtime_home=str(runtime_home),
        repo_root=str(repo_root),
        python_executable=sys.executable,
        home=str(home),
        **kwargs,
    )
    return RuntimeService(config)


def test_runtime_launchd_plist_uses_user_launch_agent_and_runtime_paths(tmp_path: Path) -> None:
    service = _service(tmp_path, ollama_base_url="http://ollama.lan:11434")

    payload = service.install_service(dry_run=True)
    plist = plistlib.loads(payload["plist"].encode("utf-8"))

    assert payload["plist_path"].endswith("Library/LaunchAgents/com.conos.runtime.plist")
    assert plist["Label"] == "com.conos.runtime"
    assert plist["KeepAlive"] is True
    assert plist["RunAtLoad"] is True
    assert plist["WorkingDirectory"] == str(tmp_path / "repo")
    assert plist["StandardOutPath"].endswith("logs/conos.out.log")
    assert plist["StandardErrorPath"].endswith("logs/conos.err.log")
    assert plist["EnvironmentVariables"]["CONOS_STATE_DB"].endswith("runtime-home/conos.sqlite")
    assert plist["EnvironmentVariables"]["OLLAMA_BASE_URL"] == "http://ollama.lan:11434"
    assert plist["EnvironmentVariables"]["THE_AGI_STATE_PATH"].endswith("runtime-home/state/state.json")
    assert "-m" in plist["ProgramArguments"]
    assert "core.runtime.service_daemon" in plist["ProgramArguments"]
    assert "--autonomous-state-path" in plist["ProgramArguments"]
    assert "sudo" not in plist["ProgramArguments"]


def test_runtime_launchd_plist_can_enable_managed_vm_watchdog(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        vm_watchdog_enabled=True,
        vm_auto_recover=True,
        vm_guest_wait_seconds=7.0,
    )

    payload = service.install_service(dry_run=True)
    plist = plistlib.loads(payload["plist"].encode("utf-8"))
    args = plist["ProgramArguments"]

    assert "--vm-watchdog" in args
    assert "--vm-auto-recover" in args
    assert "--vm-state-root" in args
    assert args[args.index("--vm-state-root") + 1].endswith("runtime-home/vm")
    assert args[args.index("--vm-image-id") + 1] == "conos-base"
    assert args[args.index("--vm-instance-id") + 1] == "default"
    assert args[args.index("--vm-guest-wait-seconds") + 1] == "7.0"


def test_runtime_cli_install_status_and_logs_use_runtime_home(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (runtime_home / "logs").mkdir(parents=True)
    (runtime_home / "logs" / "conos.out.log").write_text("one\ntwo\n", encoding="utf-8")
    (runtime_home / "logs" / "conos.err.log").write_text("err-one\nerr-two\n", encoding="utf-8")

    assert conos_cli.main(
        [
            "install-service",
            "--dry-run",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    install_payload = json.loads(capsys.readouterr().out)
    assert install_payload["dry_run"] is True
    assert install_payload["runtime_paths"]["state_db"].endswith("conos.sqlite")

    assert conos_cli.main(["logs", "--runtime-home", str(runtime_home), "--tail", "1"]) == 0
    logs_payload = json.loads(capsys.readouterr().out)
    assert logs_payload["stdout"] == "two\n"
    assert logs_payload["stderr"] == "err-two\n"

    assert conos_cli.main(["status", "--runtime-home", str(runtime_home), "--repo-root", str(repo_root), "--home", str(home)]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["runtime_paths"]["state_db"].endswith("conos.sqlite")
    assert status_payload["runtime_mode"]["mode"] == "IDLE"
    assert "SLEEP" in {row["mode"] for row in status_payload["runtime_mode_catalog"]["modes"]}
    assert status_payload["metrics"]["run_count"] == 0
    assert status_payload["soak_sessions"] == []


def test_runtime_cli_setup_and_doctor_are_stable_product_commands(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = Path(__file__).resolve().parents[1]

    assert conos_cli.main(
        [
            "setup",
            "--dry-run",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    setup_payload = json.loads(capsys.readouterr().out)
    assert setup_payload["status"] == "READY"
    assert "conos doctor" in setup_payload["stable_commands"]
    assert setup_payload["llm_policy"]["route_policy_count"] == 0
    assert setup_payload["runtime_paths"]["runtime_home"] == str(runtime_home)
    assert setup_payload["operator_panel"]["surface"] == "setup"
    assert setup_payload["operator_panel"]["health"] == "healthy"
    assert setup_payload["recovery_diagnosis_tree"]["status"] == "OK"
    assert runtime_home.exists() is False

    assert conos_cli.main(
        [
            "setup",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    written_payload = json.loads(capsys.readouterr().out)
    assert written_payload["written"] is True
    assert written_payload["operator_panel"]["surface"] == "setup"
    assert written_payload["recovery_diagnosis_tree"]["status"] == "OK"
    assert (runtime_home / "setup.json").exists()

    assert conos_cli.main(
        [
            "doctor",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["status"] in {"OK", "WARN"}
    check_names = {row["name"] for row in doctor_payload["checks"]}
    assert "runtime_status_command" in check_names
    assert "llm_policy_contracts" in check_names
    assert "managed_vm_provider" in check_names


def test_runtime_cli_one_click_setup_dry_run_is_safe(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = Path(__file__).resolve().parents[1]

    assert conos_cli.main(
        [
            "setup",
            "--one-click",
            "--dry-run",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["one_click"] is True
    assert payload["written"] is False
    assert payload["status"] == "NEEDS_ACTION"
    assert payload["operator_panel"]["surface"] == "setup"
    assert payload["operator_panel"]["health"] == "warning"
    assert "vm_default_boundary" in payload["operator_panel"]["top_issues"]
    assert "vm_boundary" in payload["recovery_diagnosis_tree"]["active_categories"]
    report = payload["one_click_report"]
    assert report["dry_run"] is True
    assert report["service_install"]["dry_run"] is True
    assert report["vm_default_boundary"]["dry_run"] is True
    assert report["action_needed"]
    assert runtime_home.exists() is False
    assert (home / "Library" / "LaunchAgents" / "com.conos.runtime.plist").exists() is False


def test_runtime_cli_one_click_setup_can_install_core_without_vm(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = Path(__file__).resolve().parents[1]

    assert conos_cli.main(
        [
            "setup",
            "--one-click",
            "--no-vm",
            "--skip-doctor",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "READY"
    assert payload["written"] is True
    assert payload["operator_panel"]["surface"] == "setup"
    assert payload["operator_panel"]["health"] == "healthy"
    assert payload["recovery_diagnosis_tree"]["status"] == "OK"
    assert (runtime_home / "setup.json").exists()
    assert (home / "Library" / "LaunchAgents" / "com.conos.runtime.plist").exists()
    step_names = {step["name"] for step in payload["one_click_report"]["steps"]}
    assert {"runtime_home", "install_service"} <= step_names
    assert "vm_default_boundary" not in step_names
    assert payload["install_validation"]["status"] == "NEEDS_VALIDATION"
    assert payload["install_validation"]["setup_actions"] == []
    assert any(item["check"] == "launchd_loaded" for item in payload["install_validation"]["validation_remaining"])


def test_runtime_cli_validate_install_distinguishes_setup_from_validation(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = Path(__file__).resolve().parents[1]

    assert conos_cli.main(
        [
            "validate-install",
            "--no-vm",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "FAILED"
    assert {item["check"] for item in missing["setup_actions"]} >= {"setup_manifest", "runtime_directories", "launchd_plist"}

    assert conos_cli.main(
        [
            "setup",
            "--one-click",
            "--no-vm",
            "--skip-doctor",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    capsys.readouterr()

    assert conos_cli.main(
        [
            "validate-install",
            "--no-vm",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    installed = json.loads(capsys.readouterr().out)
    assert installed["status"] == "NEEDS_VALIDATION"
    assert installed["setup_actions"] == []
    assert [item["check"] for item in installed["validation_remaining"]] == ["launchd_loaded"]


def test_validate_install_product_mode_requires_vm_boundary(tmp_path: Path, capsys) -> None:
    runtime_home = tmp_path / "runtime"
    home = tmp_path / "home"
    repo_root = Path(__file__).resolve().parents[1]

    assert conos_cli.main(
        [
            "setup",
            "--one-click",
            "--no-vm",
            "--skip-doctor",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    ) == 0
    capsys.readouterr()

    exit_code = conos_cli.main(
        [
            "validate-install",
            "--product",
            "--runtime-home",
            str(runtime_home),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--home",
            str(home),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "FAILED"
    vm_check = next(row for row in payload["checks"] if row["name"] == "vm_default_boundary")
    assert vm_check["required"] is True
    assert vm_check["ok"] is False
    gate = payload["product_deployment_gate"]
    assert gate["status"] == "BLOCKED"
    assert gate["deployable"] is False
    assert gate["default_side_effect_boundary"]["host_side_effect_execution_allowed_by_default"] is False
    assert any(row["check"] == "vm_default_boundary" for row in gate["blockers"])


def test_validate_install_product_mode_passes_when_vm_boundary_is_ready(tmp_path: Path, monkeypatch) -> None:
    import modules.local_mirror.managed_vm as managed_vm

    service = _service(tmp_path)
    setup = service.setup(one_click=True, include_vm=False, run_doctor=False)
    assert setup["status"] == "READY"

    monkeypatch.setattr(service, "_launchd_print", lambda: {"loaded": True, "raw": "loaded"})
    monkeypatch.setattr(
        managed_vm,
        "managed_vm_setup_plan",
        lambda **_kwargs: {
            "schema_version": "conos.managed_vm_setup_plan/v1",
            "status": "READY",
            "safe_to_run_tasks": True,
            "real_vm_boundary": True,
            "operator_summary": "VM default execution boundary is ready",
        },
    )

    payload = service.validate_install(product=True)

    assert payload["status"] == "READY"
    assert payload["product_deployment_gate"]["status"] == "PASSED"
    assert payload["product_deployment_gate"]["deployable"] is True
    assert payload["product_deployment_gate"]["default_side_effect_boundary"]["real_vm_boundary"] is True


def test_approval_inbox_and_approve_resume_safely(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    try:
        run_id = supervisor.create_run("needs approval")
        task_id = supervisor.add_task(run_id, "write back to source", verifier={"requires_approval": True})
        assert supervisor.tick_once(run_id)["status"] == "TASK_STARTED"
        waiting = supervisor.tick_once(run_id)
        assert waiting["status"] == "WAITING_APPROVAL"
        approval_id = waiting["approval_id"]
    finally:
        supervisor.state_store.close()

    inbox = service.approvals()
    assert inbox["count"] == 1
    assert inbox["approvals"][0]["approval_id"] == approval_id

    approved = service.approve(approval_id, approved_by="test")
    assert approved["status"] == "APPROVED"
    assert approved["run"]["status"] == "RUNNING"

    supervisor = service.config.supervisor()
    try:
        task = supervisor.state_store.get_task(task_id)
        assert task["status"] == "PENDING"
        assert task["result"]["approval_granted"] == approval_id
        assert supervisor.tick_once(run_id)["status"] == "TASK_STARTED"
        assert supervisor.tick_once(run_id)["status"] == "COMPLETED"
        assert supervisor.state_store.get_run(run_id)["status"] == "COMPLETED"
    finally:
        supervisor.state_store.close()


def test_resource_watchdog_marks_active_run_degraded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    try:
        run_id = supervisor.create_run("needs ollama")
        supervisor.add_task(run_id, "call model")
        watchdog = ResourceWatchdog(
            runtime_home=tmp_path / "runtime-home",
            thresholds=WatchdogThresholds(
                ollama_base_url="http://127.0.0.1:1",
                ollama_timeout_seconds=0.1,
                ollama_required=True,
            ),
        )

        result = tick_runtime_once(supervisor, watchdog=watchdog, snapshot_path=None, max_event_rows=5000)

        assert result["watchdog"]["status"] == "DEGRADED"
        assert result["runtime_mode"]["mode"] == "DEGRADED_RECOVERY"
        assert supervisor.state_store.get_run(run_id)["status"] == "DEGRADED"
        event_types = [event["event_type"] for event in supervisor.state_store.list_events(run_id)]
        assert "run_degraded" in event_types
    finally:
        supervisor.state_store.close()


def test_vm_watchdog_degradation_marks_active_run_degraded(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()

    class FakeVMWatchdog:
        def evaluate(self):
            return {"status": "DEGRADED", "reason": "guest_agent_not_ready"}

    try:
        run_id = supervisor.create_run("needs vm")
        supervisor.add_task(run_id, "execute in vm")

        result = tick_runtime_once(
            supervisor,
            watchdog=None,
            vm_watchdog=FakeVMWatchdog(),
            snapshot_path=None,
            max_event_rows=5000,
        )

        assert result["watchdog"]["status"] == "DEGRADED"
        assert result["watchdog"]["vm_status"] == "DEGRADED"
        assert result["vm_watchdog"]["reason"] == "guest_agent_not_ready"
        assert result["runtime_mode"]["mode"] == "DEGRADED_RECOVERY"
        assert supervisor.state_store.get_run(run_id)["status"] == "DEGRADED"
        assert supervisor.state_store.get_run(run_id)["paused_reason"] == "vm_watchdog_degraded"
    finally:
        supervisor.state_store.close()


def test_daemon_tick_includes_runtime_maintenance_report(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    try:
        run_id = supervisor.create_run("maintenance in daemon")
        for index in range(4):
            supervisor.event_journal.append(run_id=run_id, event_type="noise", payload={"index": index})

        result = tick_runtime_once(supervisor, watchdog=None, snapshot_path=None, max_event_rows=2)

        assert result["watchdog"]["status"] == "SKIPPED"
        assert result["runtime_mode"]["mode"] in {"ACTING", "IDLE", "ROUTINE_RUN"}
        assert "maintenance" in result
        assert result["maintenance"]["prune"]["deleted"] >= 1
        assert result["maintenance"]["checkpoint"]["status"] in {"OK", "BUSY", "SKIPPED"}
        assert result["prune"] == result["maintenance"]["prune"]
    finally:
        supervisor.state_store.close()


def test_daemon_no_user_tick_schedules_safe_autonomous_goal_pressure(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    state_path = tmp_path / "runtime-home" / "state" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "goal_stack": {
                    "subgoals": [
                        {
                            "goal_id": "goal:capability_repair:file_read",
                            "objective": "Investigate file_read failures and write a bounded report.",
                            "status": "active",
                            "priority": 0.9,
                            "permission_level": "L1",
                            "allowed_actions": ["read_logs", "read_reports", "run_readonly_analysis", "write_report"],
                            "forbidden_actions": ["modify_core_runtime_without_approval"],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        result = tick_runtime_once(
            supervisor,
            watchdog=None,
            snapshot_path=None,
            max_event_rows=5000,
            autonomous_tick_enabled=True,
            autonomous_state_path=state_path,
        )

        assert result["autonomous_tick"]["status"] == "AUTONOMOUS_TASK_SCHEDULED"
        assert result["ticks"][0]["status"] == "TASK_STARTED"
        assert result["runtime_mode"]["mode"] == "ACTING"
        run = supervisor.state_store.get_run(result["autonomous_tick"]["run_id"])
        assert run["metadata"]["created_without_user_instruction"] is True
        event_types = [event["event_type"] for event in supervisor.state_store.list_events(run["run_id"])]
        assert "autonomous_no_user_tick_scheduled" in event_types

        completed = tick_runtime_once(
            supervisor,
            watchdog=None,
            snapshot_path=None,
            max_event_rows=5000,
            autonomous_tick_enabled=True,
            autonomous_state_path=state_path,
        )
        assert completed["ticks"][0]["status"] == "COMPLETED"
        task = supervisor.state_store.list_tasks(run["run_id"])[0]
        assert task["result"]["formal_evidence_id"].startswith("ev_")
        assert Path(task["result"]["homeostasis_report_path"]).exists()
    finally:
        supervisor.state_store.close()


def test_daemon_does_not_auto_clear_zombie_suspected_degraded_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    try:
        run_id = supervisor.create_run("zombie stays degraded")
        supervisor.maintenance_once(zombie_threshold_seconds=999)
        supervisor.state_store.update_heartbeat_status(run_id, "TICKING")
        supervisor.maintenance_once(zombie_threshold_seconds=0)

        result = tick_runtime_once(supervisor, watchdog=None, snapshot_path=None, max_event_rows=5000, zombie_threshold_seconds=0)

        run = supervisor.state_store.get_run(run_id)
        assert run["status"] == "DEGRADED"
        assert run["paused_reason"] == "zombie_suspected"
        assert result["ticks"] == [{"run_id": run_id, "status": "DEGRADED", "reason": "zombie_suspected"}]
    finally:
        supervisor.state_store.close()


def test_tick_exception_degrades_one_run_without_blocking_others(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    supervisor = service.config.supervisor()
    try:
        bad_run = supervisor.create_run("bad run")
        good_run = supervisor.create_run("good run")
        supervisor.add_task(bad_run, "bad task")
        good_task = supervisor.add_task(good_run, "good task")
        original_tick_once = supervisor.tick_once

        def flaky_tick_once(run_id: str):
            if run_id == bad_run:
                raise RuntimeError("simulated tick failure")
            return original_tick_once(run_id)

        monkeypatch.setattr(supervisor, "tick_once", flaky_tick_once)

        result = tick_runtime_once(supervisor, watchdog=None, snapshot_path=None, max_event_rows=5000)

        ticks_by_run = {item["run_id"]: item for item in result["ticks"]}
        assert ticks_by_run[bad_run]["status"] == "TICK_EXCEPTION"
        assert ticks_by_run[bad_run]["degraded"]["marked_degraded"] is True
        assert ticks_by_run[good_run]["status"] == "TASK_STARTED"
        assert supervisor.state_store.get_run(bad_run)["status"] == "DEGRADED"
        assert supervisor.state_store.get_run(bad_run)["paused_reason"] == "tick_exception"
        assert supervisor.state_store.get_task(good_task)["status"] == "RUNNING"
        event_types = [event["event_type"] for event in supervisor.state_store.list_events(bad_run)]
        assert "run_degraded" in event_types
    finally:
        supervisor.state_store.close()


def test_runtime_soak_short_duration_writes_snapshots(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.soak(duration_seconds=0.05, tick_interval=0.01, snapshot_interval=0.01)

    assert result["status"] == "PASSED"
    assert result["tick_count"] >= 1
    assert Path(result["snapshot_path"]).exists()
    assert (tmp_path / "runtime-home" / "conos.sqlite").exists()
    events_path = tmp_path / "runtime-home" / "runs" / result["run_id"] / "events.jsonl"
    assert events_path.exists()

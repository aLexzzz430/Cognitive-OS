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
    assert "-m" in plist["ProgramArguments"]
    assert "core.runtime.service_daemon" in plist["ProgramArguments"]
    assert "sudo" not in plist["ProgramArguments"]


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
    assert status_payload["metrics"]["run_count"] == 0
    assert status_payload["soak_sessions"] == []


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
        assert supervisor.state_store.get_run(run_id)["status"] == "DEGRADED"
        event_types = [event["event_type"] for event in supervisor.state_store.list_events(run_id)]
        assert "run_degraded" in event_types
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
        assert "maintenance" in result
        assert result["maintenance"]["prune"]["deleted"] >= 1
        assert result["maintenance"]["checkpoint"]["status"] in {"OK", "BUSY", "SKIPPED"}
        assert result["prune"] == result["maintenance"]["prune"]
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

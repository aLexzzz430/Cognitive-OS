from __future__ import annotations

from pathlib import Path
import sys

from core.runtime.runtime_service import RuntimeService, RuntimeServiceConfig
from core.runtime.soak_runner import ZombieDetector


def _service(tmp_path: Path, **kwargs) -> RuntimeService:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = RuntimeServiceConfig.from_args(
        runtime_home=str(tmp_path / "runtime"),
        repo_root=str(repo_root),
        python_executable=sys.executable,
        home=str(tmp_path / "home"),
        **kwargs,
    )
    return RuntimeService(config)


def test_soak_session_lifecycle_is_distinct_from_runs(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.soak(
        duration_seconds=0.03,
        mode="infrastructure",
        tick_interval=0.01,
        snapshot_interval=0.01,
        probe_types=("db_integrity", "event_roundtrip"),
    )

    supervisor = service.config.supervisor()
    try:
        session = supervisor.state_store.get_soak_session(result["soak_id"])
        run = supervisor.state_store.get_run(result["run_id"])
        assert result["status"] == "PASSED"
        assert session["soak_id"] == result["soak_id"]
        assert session["mode"] == "infrastructure"
        assert session["status"] == "PASSED"
        assert session["current_run_id"] == result["run_id"]
        assert session["summary"]["total_probe_tasks"] == result["total_probe_tasks"]
        assert run["run_id"] == result["run_id"]
        assert session["soak_id"] != run["run_id"]
    finally:
        supervisor.state_store.close()


def test_workload_mode_injects_probe_tasks_for_duration(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.soak(
        duration_seconds=0.08,
        mode="workload",
        tick_interval=0.01,
        snapshot_interval=0.01,
        task_interval=0.01,
        probe_types=("dummy_verifier_pass",),
    )

    supervisor = service.config.supervisor()
    try:
        tasks = supervisor.state_store.list_tasks(result["run_id"])
        assert result["status"] == "PASSED"
        assert result["mode"] == "workload"
        assert result["total_probe_tasks"] >= 2
        assert len(tasks) >= 2
        assert all(task["status"] == "COMPLETED" for task in tasks)
    finally:
        supervisor.state_store.close()


def test_zombie_detector_marks_suspected_then_failed() -> None:
    detector = ZombieDetector(threshold_seconds=5.0, fail_seconds=5.0)
    run = {"heartbeat_updated_at": 10.0}

    assert detector.observe(run=run, event_count=1, task_progress_at=10.0, now=10.0)["status"] == "OK"
    suspected = detector.observe(run={"heartbeat_updated_at": 20.0}, event_count=1, task_progress_at=10.0, now=16.0)
    failed = detector.observe(run={"heartbeat_updated_at": 30.0}, event_count=1, task_progress_at=10.0, now=22.0)

    assert suspected["status"] == "ZOMBIE_SUSPECTED"
    assert suspected["reason"] == "heartbeat_without_progress"
    assert failed["status"] == "FAILED"
    assert failed["reason"] == "zombie_persisted"


def test_approval_pause_resume_probe_records_result(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.soak(
        duration_seconds=0.03,
        mode="infrastructure",
        tick_interval=0.01,
        snapshot_interval=0.01,
        probe_types=("approval_pause_resume",),
    )

    supervisor = service.config.supervisor()
    try:
        approval_result = result["approval_pause_resume_result"]
        assert result["status"] == "PASSED"
        assert approval_result["passed"] is True
        assert approval_result["waiting"]["status"] == "WAITING_APPROVAL"
        assert approval_result["approved"]["status"] == "APPROVED"
        assert supervisor.state_store.list_approvals(result["run_id"], status="WAITING") == []
    finally:
        supervisor.state_store.close()

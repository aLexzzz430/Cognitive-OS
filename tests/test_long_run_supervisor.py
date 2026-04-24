from pathlib import Path
import time

from core.runtime.long_run_supervisor import LongRunSupervisor
from core.runtime.state_store import RuntimeStateStore


def _supervisor(tmp_path: Path, *, worker_id: str = "worker-1") -> LongRunSupervisor:
    return LongRunSupervisor(
        state_store=RuntimeStateStore(tmp_path / "state.sqlite3"),
        runs_root=tmp_path / "runs",
        worker_id=worker_id,
        lease_ttl_seconds=0.2,
    )


def test_supervisor_can_resume_after_simulated_crash(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, worker_id="worker-a")
    run_id = supervisor.create_run("keep working")
    task_id = supervisor.add_task(run_id, "step one", verifier={})

    assert supervisor.tick_once(run_id)["status"] == "TASK_STARTED"
    supervisor.state_store.update_task_status(task_id, "RUNNING")
    supervisor.state_store.acquire_lease(run_id, worker_id="crashed-worker", ttl_seconds=0.1)
    time.sleep(0.12)

    recovered = supervisor.recover_after_crash(run_id)

    assert recovered["status"] == "RUNNING"
    assert supervisor.state_store.get_task(task_id)["status"] == "PENDING"
    assert supervisor.tick_once(run_id)["status"] == "TASK_STARTED"
    event_types = [event["event_type"] for event in supervisor.state_store.list_events(run_id)]
    assert "crash_recovered" in event_types
    assert (tmp_path / "runs" / run_id / "events.jsonl").exists()


def test_lease_prevents_two_workers_from_ticking_same_run(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, worker_id="worker-a")
    run_id = supervisor.create_run("exclusive work")
    supervisor.add_task(run_id, "step one")
    supervisor.state_store.acquire_lease(run_id, worker_id="worker-b", ttl_seconds=60)

    result = supervisor.tick_once(run_id)

    assert result["status"] == "LEASE_HELD"
    assert result["lease"]["worker_id"] == "worker-b"
    assert supervisor.state_store.list_tasks(run_id)[0]["status"] == "PENDING"


def test_approval_request_pauses_run(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    run_id = supervisor.create_run("needs review")
    task_id = supervisor.add_task(run_id, "dangerous step", verifier={"requires_approval": True})

    assert supervisor.tick_once(run_id)["status"] == "TASK_STARTED"
    result = supervisor.tick_once(run_id)

    run = supervisor.state_store.get_run(run_id)
    approval = supervisor.state_store.get_latest_approval(run_id)
    assert result["status"] == "WAITING_APPROVAL"
    assert run["status"] == "WAITING_APPROVAL"
    assert run["heartbeat_status"] == "WAITING_APPROVAL"
    assert supervisor.state_store.get_task(task_id)["status"] == "WAITING_APPROVAL"
    assert approval["status"] == "WAITING"
    assert approval["request"]["task_id"] == task_id


def test_completed_task_advances_to_next_pending_task(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    run_id = supervisor.create_run("ordered work")
    first = supervisor.add_task(run_id, "first", priority=10)
    second = supervisor.add_task(run_id, "second", priority=1)

    assert supervisor.tick_once(run_id) == {
        "status": "TASK_STARTED",
        "run_id": run_id,
        "task_id": first,
    }
    result = supervisor.tick_once(run_id)

    assert result["status"] == "TASK_COMPLETED_NEXT_STARTED"
    assert result["task_id"] == first
    assert result["next_task_id"] == second
    assert supervisor.state_store.get_task(first)["status"] == "COMPLETED"
    assert supervisor.state_store.get_task(second)["status"] == "RUNNING"
    assert supervisor.state_store.get_run(run_id)["status"] == "RUNNING"


def test_heartbeat_stale_does_not_corrupt_run_state(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    run_id = supervisor.create_run("paused work")
    supervisor.pause_run(run_id, "operator pause")
    supervisor.state_store._conn.execute(
        "UPDATE runs SET heartbeat_status = 'TICKING', heartbeat_updated_at = 0 WHERE run_id = ?",
        (run_id,),
    )

    recovered = supervisor.recover_after_crash(run_id)

    assert recovered["status"] == "PAUSED"
    assert recovered["heartbeat_status"] == "PAUSED"
    assert recovered["paused_reason"] == "operator pause"

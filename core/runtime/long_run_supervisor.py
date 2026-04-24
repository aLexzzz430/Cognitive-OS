from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional
import uuid

from core.runtime.event_journal import DEFAULT_RUNS_ROOT, EventJournal
from core.runtime.state_store import DEFAULT_STATE_DB, RuntimeStateStore


RUNNING_STATUSES = {"RUNNING", "RECOVERING"}
TERMINAL_STATUSES = {"COMPLETED", "STOPPED", "FAILED"}


class LongRunSupervisor:
    """Local-first resumable supervisor for bounded Cognitive OS ticks."""

    def __init__(
        self,
        *,
        state_store: Optional[RuntimeStateStore] = None,
        db_path: str | Path = DEFAULT_STATE_DB,
        runs_root: str | Path = DEFAULT_RUNS_ROOT,
        worker_id: Optional[str] = None,
        lease_ttl_seconds: float = 30.0,
    ) -> None:
        self.state_store = state_store or RuntimeStateStore(db_path)
        self.event_journal = EventJournal(self.state_store, runs_root=runs_root)
        self.worker_id = str(worker_id or f"worker-{uuid.uuid4().hex[:12]}")
        self.lease_ttl_seconds = float(lease_ttl_seconds)

    def create_run(self, goal: str, *, run_id: Optional[str] = None) -> str:
        resolved_run_id = self.state_store.create_run(goal, run_id=run_id)
        self.event_journal.append(
            run_id=resolved_run_id,
            event_type="run_created",
            payload={"goal": str(goal or "")},
        )
        return resolved_run_id

    def add_task(
        self,
        run_id: str,
        objective: str,
        priority: int = 0,
        verifier: Optional[Mapping[str, Any]] = None,
    ) -> str:
        task_id = self.state_store.add_task(
            run_id,
            objective,
            priority=int(priority),
            verifier=dict(verifier or {}),
        )
        self.event_journal.append(
            run_id=run_id,
            task_id=task_id,
            event_type="task_added",
            payload={"objective": str(objective or ""), "priority": int(priority), "verifier": dict(verifier or {})},
        )
        return task_id

    def tick_once(self, run_id: str) -> Dict[str, Any]:
        lease = self.state_store.acquire_lease(
            run_id,
            worker_id=self.worker_id,
            ttl_seconds=self.lease_ttl_seconds,
        )
        if not bool(lease.get("acquired", False)):
            return {"status": "LEASE_HELD", "run_id": str(run_id), "lease": lease}
        try:
            run = self.state_store.get_run(run_id)
            if not run:
                return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
            if run["status"] in TERMINAL_STATUSES or run["status"] in {"PAUSED", "WAITING_APPROVAL"}:
                self.state_store.update_heartbeat_status(run_id, run["status"])
                return {"status": run["status"], "run_id": str(run_id)}

            self.state_store.update_heartbeat_status(run_id, "TICKING")
            active = self.state_store.first_active_task(run_id)
            if active:
                result = self._complete_active_task(run_id, active)
                self.state_store.update_heartbeat_status(run_id, self.state_store.get_run(run_id).get("status", "RUNNING"))
                return result

            pending = self.state_store.next_pending_task(run_id)
            if pending:
                self._start_task(run_id, pending)
                self.state_store.update_heartbeat_status(run_id, "RUNNING")
                return {"status": "TASK_STARTED", "run_id": str(run_id), "task_id": pending["task_id"]}

            tasks = self.state_store.list_tasks(run_id)
            if tasks and all(task["status"] == "COMPLETED" for task in tasks):
                self.state_store.update_run_status(run_id, "COMPLETED")
                self.event_journal.append(run_id=run_id, event_type="run_completed", payload={"task_count": len(tasks)})
                return {"status": "COMPLETED", "run_id": str(run_id)}

            self.state_store.update_heartbeat_status(run_id, "IDLE")
            return {"status": "IDLE", "run_id": str(run_id)}
        finally:
            self.state_store.release_lease(run_id, worker_id=self.worker_id)

    def run_until_stopped(self, run_id: str, tick_interval: float = 1.0) -> Dict[str, Any]:
        last: Dict[str, Any] = {"status": "NOT_STARTED", "run_id": str(run_id)}
        while True:
            run = self.state_store.get_run(run_id)
            if not run:
                return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
            if run["status"] in TERMINAL_STATUSES or run["status"] in {"PAUSED", "WAITING_APPROVAL"}:
                return {"status": run["status"], "run_id": str(run_id), "last_tick": last}
            last = self.tick_once(run_id)
            if str(last.get("status", "")) in TERMINAL_STATUSES or str(last.get("status", "")) in {"PAUSED", "WAITING_APPROVAL"}:
                return {"status": str(last.get("status")), "run_id": str(run_id), "last_tick": last}
            time.sleep(max(0.0, float(tick_interval)))

    def pause_run(self, run_id: str, reason: str) -> Dict[str, Any]:
        self.state_store.update_run_status(run_id, "PAUSED", paused_reason=str(reason or ""))
        self.event_journal.append(run_id=run_id, event_type="run_paused", payload={"reason": str(reason or "")})
        return self.state_store.get_run(run_id)

    def resume_run(self, run_id: str) -> Dict[str, Any]:
        self.state_store.update_run_status(run_id, "RUNNING")
        self.event_journal.append(run_id=run_id, event_type="run_resumed", payload={})
        return self.state_store.get_run(run_id)

    def mark_waiting_approval(self, run_id: str, approval_request: Mapping[str, Any]) -> Dict[str, Any]:
        active = self.state_store.first_active_task(run_id)
        task_id = str(active.get("task_id", "") or "")
        if task_id:
            self.state_store.update_task_status(task_id, "WAITING_APPROVAL", result={"approval_request": dict(approval_request)})
        approval_id = self.state_store.create_approval(run_id, dict(approval_request), task_id=task_id or None)
        self.state_store.update_run_status(run_id, "WAITING_APPROVAL", paused_reason="approval_required")
        self.event_journal.append(
            run_id=run_id,
            task_id=task_id or None,
            event_type="approval_requested",
            payload={"approval_id": approval_id, "request": dict(approval_request)},
        )
        return self.state_store.get_latest_approval(run_id)

    def recover_after_crash(self, run_id: str) -> Dict[str, Any]:
        run_before = self.state_store.get_run(run_id)
        if not run_before:
            return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
        cleared = self.state_store.clear_expired_lease(run_id)
        recovered_tasks = []
        if run_before["status"] in RUNNING_STATUSES:
            for task in self.state_store.list_tasks(run_id, statuses=["RUNNING"]):
                self.state_store.update_task_status(task["task_id"], "PENDING", result={"recovered_after_crash": True})
                recovered_tasks.append(task["task_id"])
            self.state_store.update_run_status(run_id, "RUNNING")
        else:
            self.state_store.update_heartbeat_status(run_id, run_before["status"])
        self.event_journal.append(
            run_id=run_id,
            event_type="crash_recovered",
            payload={
                "previous_status": run_before["status"],
                "cleared_expired_lease": bool(cleared),
                "recovered_task_ids": recovered_tasks,
            },
        )
        return self.state_store.get_run(run_id)

    def _start_task(self, run_id: str, task: Mapping[str, Any]) -> None:
        self.state_store.update_task_status(str(task["task_id"]), "RUNNING")
        self.event_journal.append(
            run_id=run_id,
            task_id=str(task["task_id"]),
            event_type="task_started",
            payload={"objective": str(task.get("objective", "") or "")},
        )

    def _complete_active_task(self, run_id: str, task: Mapping[str, Any]) -> Dict[str, Any]:
        verifier = dict(task.get("verifier", {}) or {})
        if bool(verifier.get("requires_approval", False)):
            approval = self.mark_waiting_approval(
                run_id,
                {
                    "task_id": str(task["task_id"]),
                    "objective": str(task.get("objective", "") or ""),
                    "verifier": verifier,
                },
            )
            return {
                "status": "WAITING_APPROVAL",
                "run_id": str(run_id),
                "task_id": str(task["task_id"]),
                "approval_id": str(approval.get("approval_id", "") or ""),
            }

        self.state_store.update_task_status(
            str(task["task_id"]),
            "COMPLETED",
            result={"verified": True, "verifier": verifier},
        )
        self.event_journal.append(
            run_id=run_id,
            task_id=str(task["task_id"]),
            event_type="task_completed",
            payload={"verified": True, "verifier": verifier},
        )
        pending = self.state_store.next_pending_task(run_id)
        if pending:
            self._start_task(run_id, pending)
            return {
                "status": "TASK_COMPLETED_NEXT_STARTED",
                "run_id": str(run_id),
                "task_id": str(task["task_id"]),
                "next_task_id": str(pending["task_id"]),
            }
        self.state_store.update_run_status(run_id, "COMPLETED")
        self.event_journal.append(run_id=run_id, event_type="run_completed", payload={})
        return {"status": "COMPLETED", "run_id": str(run_id), "task_id": str(task["task_id"])}

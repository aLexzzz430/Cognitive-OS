from __future__ import annotations

import os
import platform
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional
import uuid

from core.runtime.event_journal import DEFAULT_RUNS_ROOT, EventJournal
from core.runtime.homeostasis_executor import execute_homeostasis_task
from core.runtime.state_store import DEFAULT_STATE_DB, RuntimeStateStore


RUNNING_STATUSES = {"RUNNING", "RECOVERING", "DEGRADED"}
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
        task_watchdog_seconds: float = 300.0,
        max_task_retries: int = 3,
        retry_backoff_seconds: float = 30.0,
        event_jsonl_max_bytes: int = 5 * 1024 * 1024,
        event_jsonl_retained_files: int = 5,
    ) -> None:
        self.state_store = state_store or RuntimeStateStore(db_path)
        self.event_journal = EventJournal(
            self.state_store,
            runs_root=runs_root,
            max_jsonl_bytes=event_jsonl_max_bytes,
            retained_jsonl_files=event_jsonl_retained_files,
        )
        self.worker_id = str(worker_id or f"worker-{uuid.uuid4().hex[:12]}")
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.task_watchdog_seconds = float(task_watchdog_seconds)
        self.max_task_retries = int(max_task_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.started_at = time.time()

    def create_run(
        self,
        goal: str,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        resolved_run_id = self.state_store.create_run(goal, run_id=run_id, metadata=dict(metadata or {}))
        self.event_journal.append(
            run_id=resolved_run_id,
            event_type="run_created",
            payload={"goal": str(goal or ""), "metadata": dict(metadata or {})},
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
                if self._is_task_stale(active):
                    result = self._retry_or_fail_stale_task(run_id, active)
                else:
                    result = self._complete_active_task(run_id, active)
                heartbeat_status = "BACKOFF_WAIT" if result.get("status") == "TASK_RETRY_SCHEDULED" else self.state_store.get_run(run_id).get("status", "RUNNING")
                self.state_store.update_heartbeat_status(run_id, heartbeat_status)
                return result

            pending = self._next_runnable_task(run_id)
            if pending:
                self._start_task(run_id, pending)
                self.state_store.update_heartbeat_status(run_id, "RUNNING")
                return {"status": "TASK_STARTED", "run_id": str(run_id), "task_id": pending["task_id"]}
            if self._pending_tasks_waiting_for_retry(run_id):
                self.state_store.update_heartbeat_status(run_id, "BACKOFF_WAIT")
                return {"status": "WAITING_RETRY", "run_id": str(run_id)}

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

    def status(self, run_id: str) -> Dict[str, Any]:
        run = self.state_store.get_run(run_id)
        if not run:
            return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
        tasks = self.state_store.list_tasks(run_id)
        return {
            "run": run,
            "tasks": tasks,
            "latest_approval": self.state_store.get_latest_approval(run_id),
            "event_count": len(self.state_store.list_events(run_id)),
            "event_journal": self.event_journal.status(run_id),
        }

    def metrics(self) -> Dict[str, Any]:
        runs = self.state_store.list_runs()
        task_status_counts: Dict[str, int] = {}
        run_status_counts: Dict[str, int] = {}
        stale_running_task_count = 0
        waiting_retry_count = 0
        for run in runs:
            run_status = str(run.get("status", "UNKNOWN") or "UNKNOWN")
            run_status_counts[run_status] = run_status_counts.get(run_status, 0) + 1
            for task in self.state_store.list_tasks(str(run.get("run_id", ""))):
                task_status = str(task.get("status", "UNKNOWN") or "UNKNOWN")
                task_status_counts[task_status] = task_status_counts.get(task_status, 0) + 1
                if task_status == "RUNNING" and self._is_task_stale(task):
                    stale_running_task_count += 1
                if task_status == "PENDING" and float(dict(task.get("result", {}) or {}).get("next_retry_at", 0.0) or 0.0) > time.time():
                    waiting_retry_count += 1
        return {
            "run_count": len(runs),
            "run_status_counts": run_status_counts,
            "task_status_counts": task_status_counts,
            "stale_running_task_count": stale_running_task_count,
            "waiting_retry_count": waiting_retry_count,
            "waiting_approval_count": len(self.state_store.list_approvals(status="WAITING")),
            "lease_count": self.state_store.count_leases(),
        }

    def health(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        db_path = self.state_store.db_path
        db_size = db_path.stat().st_size if str(db_path) != ":memory:" and db_path.exists() else 0
        wal_path = Path(f"{db_path}-wal") if str(db_path) != ":memory:" else Path("")
        wal_size = wal_path.stat().st_size if str(db_path) != ":memory:" and wal_path.exists() else 0
        payload: Dict[str, Any] = {
            "status": "OK",
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "platform": platform.system(),
            "uptime_seconds": max(0.0, time.time() - self.started_at),
            "state_db": {
                "path": str(db_path),
                "size_bytes": db_size,
                "wal_size_bytes": wal_size,
            },
            "metrics": self.metrics(),
        }
        if run_id:
            payload["run"] = self.status(run_id)
        return payload

    def maintenance_once(
        self,
        *,
        max_events_per_run: int = 5000,
        zombie_threshold_seconds: float = 600.0,
        zombie_fail_seconds: float = 0.0,
        checkpoint_wal: bool = True,
    ) -> Dict[str, Any]:
        """Run bounded durability maintenance for long-lived local runtimes."""
        started_at = time.time()
        expired_leases = self.state_store.clear_expired_leases()
        prune = self.state_store.prune_events(max_events_per_run=int(max_events_per_run)) if max_events_per_run else {"deleted": 0}
        zombie = self._detect_zombie_runs(
            threshold_seconds=float(zombie_threshold_seconds),
            fail_seconds=float(zombie_fail_seconds),
        )
        integrity = self.state_store.quick_check()
        checkpoint = self.state_store.checkpoint_wal(mode="PASSIVE") if bool(checkpoint_wal) else {"status": "SKIPPED"}
        return {
            "status": "OK" if not zombie.get("failed_run_ids") and integrity.get("status") == "OK" else "FAILED",
            "created_at": time.time(),
            "duration_seconds": max(0.0, time.time() - started_at),
            "expired_leases": expired_leases,
            "prune": prune,
            "zombie": zombie,
            "integrity": integrity,
            "checkpoint": checkpoint,
        }

    def pause_run(self, run_id: str, reason: str) -> Dict[str, Any]:
        self.state_store.update_run_status(run_id, "PAUSED", paused_reason=str(reason or ""))
        self.event_journal.append(run_id=run_id, event_type="run_paused", payload={"reason": str(reason or "")})
        return self.state_store.get_run(run_id)

    def resume_run(self, run_id: str) -> Dict[str, Any]:
        self.state_store.update_run_status(run_id, "RUNNING")
        self.event_journal.append(run_id=run_id, event_type="run_resumed", payload={})
        return self.state_store.get_run(run_id)

    def mark_degraded(self, run_id: str, reason: str, *, details: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        run = self.state_store.get_run(run_id)
        if not run:
            return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
        if run["status"] in TERMINAL_STATUSES or run["status"] in {"PAUSED", "WAITING_APPROVAL"}:
            self.state_store.update_heartbeat_status(run_id, run["status"])
            return self.state_store.get_run(run_id)
        self.state_store.update_run_status(run_id, "DEGRADED", paused_reason=str(reason or "runtime_degraded"))
        self.event_journal.append(
            run_id=run_id,
            event_type="run_degraded",
            payload={"reason": str(reason or "runtime_degraded"), "details": dict(details or {})},
        )
        return self.state_store.get_run(run_id)

    def clear_degraded(self, run_id: str, reason: str = "watchdog_recovered") -> Dict[str, Any]:
        run = self.state_store.get_run(run_id)
        if not run:
            return {"status": "RUN_NOT_FOUND", "run_id": str(run_id)}
        if run["status"] != "DEGRADED":
            return run
        self.state_store.update_run_status(run_id, "RUNNING")
        self.event_journal.append(run_id=run_id, event_type="run_degraded_cleared", payload={"reason": str(reason or "")})
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

    def approve(self, approval_id: str, *, approved_by: str = "operator") -> Dict[str, Any]:
        approval = self.state_store.get_approval(approval_id)
        if not approval:
            return {"status": "APPROVAL_NOT_FOUND", "approval_id": str(approval_id)}
        if approval["status"] != "WAITING":
            return {"status": "APPROVAL_NOT_WAITING", "approval": approval}
        updated = self.state_store.approve_approval(approval_id, approved_by=approved_by)
        run_id = str(updated.get("run_id", "") or approval.get("run_id", ""))
        task_id = str(updated.get("task_id", "") or approval.get("task_id", ""))
        if task_id:
            task = self.state_store.get_task(task_id)
            result = dict(task.get("result", {}) or {})
            result.update(
                {
                    "approval_granted": str(approval_id),
                    "approved_by": str(approved_by or "operator"),
                    "approved_capability_layers": list(
                        dict(updated.get("response", {}) or {}).get("approved_capability_layers", []) or []
                    ),
                }
            )
            self.state_store.update_task_status(task_id, "PENDING", result=result)
        self.state_store.update_run_status(run_id, "RUNNING")
        self.event_journal.append(
            run_id=run_id,
            task_id=task_id or None,
            event_type="approval_approved",
            payload={"approval_id": str(approval_id), "approved_by": str(approved_by or "operator")},
        )
        return {"status": "APPROVED", "approval": updated, "run": self.state_store.get_run(run_id)}

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

    def _next_runnable_task(self, run_id: str) -> Dict[str, Any]:
        now = time.time()
        for task in self.state_store.list_tasks(run_id, statuses=["PENDING"]):
            next_retry_at = float(dict(task.get("result", {}) or {}).get("next_retry_at", 0.0) or 0.0)
            if next_retry_at <= now:
                return task
        return {}

    def _pending_tasks_waiting_for_retry(self, run_id: str) -> bool:
        now = time.time()
        for task in self.state_store.list_tasks(run_id, statuses=["PENDING"]):
            next_retry_at = float(dict(task.get("result", {}) or {}).get("next_retry_at", 0.0) or 0.0)
            if next_retry_at > now:
                return True
        return False

    def _is_task_stale(self, task: Mapping[str, Any]) -> bool:
        if self.task_watchdog_seconds <= 0:
            return False
        last_progress_at = float(task.get("updated_at", 0.0) or 0.0)
        return (time.time() - last_progress_at) > self.task_watchdog_seconds

    def _retry_or_fail_stale_task(self, run_id: str, task: Mapping[str, Any]) -> Dict[str, Any]:
        task_id = str(task["task_id"])
        result = dict(task.get("result", {}) or {})
        retry_count = int(result.get("retry_count", 0) or 0) + 1
        result.update(
            {
                "retry_count": retry_count,
                "last_error": "task_watchdog_timeout",
                "watchdog_seconds": self.task_watchdog_seconds,
            }
        )
        if retry_count > self.max_task_retries:
            result["failed_reason"] = "retry_budget_exhausted"
            self.state_store.update_task_status(task_id, "FAILED", result=result)
            self.state_store.update_run_status(run_id, "FAILED", paused_reason="retry_budget_exhausted")
            self.event_journal.append(
                run_id=run_id,
                task_id=task_id,
                event_type="task_failed",
                payload=result,
            )
            return {"status": "FAILED", "run_id": str(run_id), "task_id": task_id, "reason": "retry_budget_exhausted"}

        next_retry_at = time.time() + max(0.0, self.retry_backoff_seconds) * retry_count
        result["next_retry_at"] = next_retry_at
        self.state_store.update_task_status(task_id, "PENDING", result=result)
        self.state_store.update_run_status(run_id, "RUNNING")
        self.event_journal.append(
            run_id=run_id,
            task_id=task_id,
            event_type="task_retry_scheduled",
            payload=result,
        )
        return {
            "status": "TASK_RETRY_SCHEDULED",
            "run_id": str(run_id),
            "task_id": task_id,
            "retry_count": retry_count,
            "next_retry_at": next_retry_at,
        }

    def _complete_active_task(self, run_id: str, task: Mapping[str, Any]) -> Dict[str, Any]:
        verifier = dict(task.get("verifier", {}) or {})
        task_result = dict(task.get("result", {}) or {})
        if bool(verifier.get("requires_approval", False)) and not task_result.get("approval_granted"):
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

        if str(verifier.get("mode") or "") == "autonomous_no_user_tick":
            completion_result = execute_homeostasis_task(
                run=self.state_store.get_run(run_id),
                task=task,
                state_store=self.state_store,
                event_journal=self.event_journal,
            )
        else:
            completion_result = {
                "verified": True,
                "verifier": verifier,
                "approval_granted": task_result.get("approval_granted", ""),
            }
        self.state_store.update_task_status(
            str(task["task_id"]),
            "COMPLETED",
            result=completion_result,
        )
        self.event_journal.append(
            run_id=run_id,
            task_id=str(task["task_id"]),
            event_type="task_completed",
            payload={"verified": True, "verifier": verifier, "result": completion_result},
        )
        resolution_result: Dict[str, Any] = {}
        if str(verifier.get("mode") or "") == "autonomous_no_user_tick":
            resolution_result = self._apply_homeostasis_resolution_policy(
                run_id,
                task_id=str(task["task_id"]),
                completion_result=completion_result,
            )
            if resolution_result.get("status") == "WAITING_APPROVAL":
                return {
                    "status": "WAITING_APPROVAL",
                    "run_id": str(run_id),
                    "task_id": str(task["task_id"]),
                    "approval_id": str(resolution_result.get("approval_id", "") or ""),
                    "resolution_action": str(resolution_result.get("resolution_action", "") or ""),
                }
        pending = self._next_runnable_task(run_id)
        if pending:
            self._start_task(run_id, pending)
            return {
                "status": "TASK_COMPLETED_NEXT_STARTED",
                "run_id": str(run_id),
                "task_id": str(task["task_id"]),
                "next_task_id": str(pending["task_id"]),
                "resolution_action": str(resolution_result.get("resolution_action", "") or ""),
            }
        self.state_store.update_run_status(run_id, "COMPLETED")
        self.event_journal.append(run_id=run_id, event_type="run_completed", payload={})
        return {"status": "COMPLETED", "run_id": str(run_id), "task_id": str(task["task_id"])}

    def _apply_homeostasis_resolution_policy(
        self,
        run_id: str,
        *,
        task_id: str,
        completion_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        decision = dict(completion_result.get("resolution_decision", {}) or {})
        if not decision:
            return {"status": "NO_RESOLUTION_DECISION", "resolution_action": ""}
        action = str(decision.get("action") or "")
        suppress_seconds = float(decision.get("repeat_suppress_seconds", 0.0) or 0.0)
        now = time.time()
        metadata = {
            "homeostasis_resolution": {
                **decision,
                "applied_at": now,
                "repeat_suppress_until": now + max(0.0, suppress_seconds),
                "source_task_id": str(task_id),
                "formal_evidence_id": str(completion_result.get("formal_evidence_id") or ""),
            }
        }
        self.state_store.update_run_metadata(run_id, metadata)
        self.event_journal.append(
            run_id=run_id,
            task_id=task_id,
            event_type="homeostasis_resolution_policy_applied",
            payload=metadata["homeostasis_resolution"],
        )

        if action == "escalate_deep_think":
            follow_up = dict(decision.get("follow_up_task", {}) or {})
            next_task_id = self.add_task(
                run_id,
                str(follow_up.get("objective") or "Deep-think review of persistent homeostasis pressure."),
                priority=int(follow_up.get("priority", 65) or 65),
                verifier={
                    "mode": "deep_think_homeostasis_review",
                    "runtime_mode": "DEEP_THINK",
                    "read_only": True,
                    "source_task_id": str(task_id),
                    "allowed_actions": list(follow_up.get("allowed_actions", []) or []),
                    "forbidden_actions": list(follow_up.get("forbidden_actions", []) or []),
                },
            )
            return {"status": "FOLLOW_UP_TASK_ADDED", "resolution_action": action, "next_task_id": next_task_id}

        if action == "escalate_limited_l2_mirror_investigation":
            follow_up = dict(decision.get("follow_up_task", {}) or {})
            approval_request = dict(decision.get("approval_request", {}) or {})
            next_task_id = self.state_store.add_task(
                run_id,
                str(follow_up.get("objective") or "Limited L2 mirror investigation for persistent pressure."),
                priority=int(follow_up.get("priority", 70) or 70),
                verifier={
                    "mode": "limited_l2_mirror_investigation",
                    "permission_level": "limited_L2",
                    "requires_approval": True,
                    "source_task_id": str(task_id),
                    "allowed_actions": list(follow_up.get("allowed_actions", []) or []),
                    "forbidden_actions": list(follow_up.get("forbidden_actions", []) or []),
                },
                status="WAITING_APPROVAL",
            )
            self.event_journal.append(
                run_id=run_id,
                task_id=next_task_id,
                event_type="task_added",
                payload={
                    "objective": str(follow_up.get("objective") or ""),
                    "priority": int(follow_up.get("priority", 70) or 70),
                    "verifier": {"mode": "limited_l2_mirror_investigation", "requires_approval": True},
                },
            )
            request = {
                **approval_request,
                "task_id": next_task_id,
                "objective": str(follow_up.get("objective") or ""),
                "resolution_action": action,
            }
            approval_id = self.state_store.create_approval(run_id, request, task_id=next_task_id)
            self.state_store.update_run_status(run_id, "WAITING_APPROVAL", paused_reason="homeostasis_limited_l2_requires_approval")
            self.event_journal.append(
                run_id=run_id,
                task_id=next_task_id,
                event_type="approval_requested",
                payload={"approval_id": approval_id, "request": request},
            )
            return {"status": "WAITING_APPROVAL", "resolution_action": action, "approval_id": approval_id, "next_task_id": next_task_id}

        if action == "escalate_waiting_human":
            approval_request = dict(decision.get("approval_request", {}) or {})
            request = {
                **approval_request,
                "task_id": str(task_id),
                "objective": "Review unresolved homeostasis pressure.",
                "resolution_action": action,
            }
            approval_id = self.state_store.create_approval(run_id, request, task_id=task_id)
            self.state_store.update_run_status(run_id, "WAITING_APPROVAL", paused_reason="homeostasis_waiting_human")
            self.event_journal.append(
                run_id=run_id,
                task_id=task_id,
                event_type="approval_requested",
                payload={"approval_id": approval_id, "request": request},
            )
            return {"status": "WAITING_APPROVAL", "resolution_action": action, "approval_id": approval_id}

        return {"status": "RESOLUTION_APPLIED", "resolution_action": action}

    def _detect_zombie_runs(self, *, threshold_seconds: float, fail_seconds: float) -> Dict[str, Any]:
        now = time.time()
        suspected: list[Dict[str, Any]] = []
        recovered: list[Dict[str, Any]] = []
        failed: list[Dict[str, Any]] = []
        observed = 0
        threshold = max(0.0, float(threshold_seconds))
        fail_after = max(0.0, float(fail_seconds))
        for run in self.state_store.list_runs():
            status = str(run.get("status", "") or "")
            paused_reason = str(run.get("paused_reason", "") or "")
            if status in TERMINAL_STATUSES or status in {"PAUSED", "WAITING_APPROVAL"}:
                continue
            if status == "DEGRADED" and paused_reason != "zombie_suspected":
                continue
            observed += 1
            run_id = str(run.get("run_id", "") or "")
            previous = dict(dict(run.get("metadata", {}) or {}).get("runtime_maintenance", {}) or {})
            event_count = self.state_store.count_events(run_id)
            latest_event_at = self.state_store.latest_event_created_at(run_id)
            latest_task_progress_at = self.state_store.latest_task_progress_at(run_id)
            heartbeat_at = float(run.get("heartbeat_updated_at", 0.0) or 0.0)
            progress_changed = (
                event_count > int(previous.get("event_count", 0) or 0)
                or latest_event_at > float(previous.get("latest_event_at", 0.0) or 0.0)
                or latest_task_progress_at > float(previous.get("latest_task_progress_at", 0.0) or 0.0)
            )
            heartbeat_moved = heartbeat_at > float(previous.get("heartbeat_updated_at", 0.0) or 0.0)
            progress_seen_at = float(previous.get("progress_seen_at", 0.0) or 0.0)
            suspected_at = float(previous.get("zombie_suspected_at", 0.0) or 0.0)
            if progress_seen_at <= 0:
                progress_seen_at = now
            if progress_changed:
                progress_seen_at = now
                if suspected_at > 0:
                    recovered.append({"run_id": run_id, "suspected_at": suspected_at, "recovered_at": now})
                suspected_at = 0.0
                if status == "DEGRADED" and paused_reason == "zombie_suspected":
                    self.clear_degraded(run_id, reason="zombie_progress_resumed")
                    event_count = self.state_store.count_events(run_id)
                    latest_event_at = self.state_store.latest_event_created_at(run_id)

            if heartbeat_moved and not progress_changed and (now - progress_seen_at) >= threshold:
                if suspected_at <= 0:
                    suspected_at = now
                    self.mark_degraded(
                        run_id,
                        "zombie_suspected",
                        details={
                            "threshold_seconds": threshold,
                            "heartbeat_updated_at": heartbeat_at,
                            "progress_seen_at": progress_seen_at,
                            "event_count": event_count,
                            "latest_task_progress_at": latest_task_progress_at,
                        },
                    )
                    event_count = self.state_store.count_events(run_id)
                    latest_event_at = self.state_store.latest_event_created_at(run_id)
                    run = self.state_store.get_run(run_id) or run
                suspected.append(
                    {
                        "run_id": run_id,
                        "suspected_at": suspected_at,
                        "seconds_without_progress": max(0.0, now - progress_seen_at),
                    }
                )
                if fail_after > 0 and (now - suspected_at) >= fail_after:
                    self.state_store.update_run_status(run_id, "FAILED", paused_reason="zombie_persisted")
                    self.event_journal.append(
                        run_id=run_id,
                        event_type="run_zombie_failed",
                        payload={
                            "suspected_at": suspected_at,
                            "fail_seconds": fail_after,
                            "progress_seen_at": progress_seen_at,
                        },
                    )
                    event_count = self.state_store.count_events(run_id)
                    latest_event_at = self.state_store.latest_event_created_at(run_id)
                    failed.append({"run_id": run_id, "suspected_at": suspected_at, "failed_at": now})

            snapshot = {
                "event_count": event_count,
                "latest_event_at": latest_event_at,
                "latest_task_progress_at": latest_task_progress_at,
                "heartbeat_updated_at": float((self.state_store.get_run(run_id) or run).get("heartbeat_updated_at", heartbeat_at) or heartbeat_at),
                "progress_seen_at": progress_seen_at,
                "zombie_suspected_at": suspected_at,
                "checked_at": now,
            }
            self.state_store.update_run_metadata(run_id, {"runtime_maintenance": snapshot})
        return {
            "observed_run_count": observed,
            "suspected_run_ids": [item["run_id"] for item in suspected],
            "recovered_run_ids": [item["run_id"] for item in recovered],
            "failed_run_ids": [item["run_id"] for item in failed],
            "suspected": suspected,
            "recovered": recovered,
            "failed": failed,
            "threshold_seconds": threshold,
            "fail_seconds": fail_after,
        }

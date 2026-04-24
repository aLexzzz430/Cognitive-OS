from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.runtime.long_run_supervisor import LongRunSupervisor, TERMINAL_STATUSES
from core.runtime.paths import RuntimePaths
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds
from core.runtime.service_daemon import append_status_snapshot, tick_runtime_once
from core.runtime.state_store import new_id


SOAK_RUNNER_VERSION = "conos.soak_runner/v0.2"
SOAK_MODES = {"infrastructure", "workload"}
SUPPORTED_PROBE_TYPES = {
    "db_integrity",
    "event_roundtrip",
    "watchdog_snapshot",
    "dummy_verifier_pass",
    "dummy_verifier_fail",
    "approval_pause_resume",
    "ollama_ping",
}
DEFAULT_INFRASTRUCTURE_PROBES = (
    "db_integrity",
    "event_roundtrip",
    "watchdog_snapshot",
    "dummy_verifier_pass",
)
DEFAULT_WORKLOAD_PROBES = (
    "db_integrity",
    "event_roundtrip",
    "watchdog_snapshot",
    "dummy_verifier_pass",
    "approval_pause_resume",
    "ollama_ping",
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(percentile)))))
    return ordered[index]


def _json_dict(payload: Any) -> Dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


@dataclass
class SoakConfig:
    duration_seconds: float
    mode: str = "infrastructure"
    tick_interval: float = 5.0
    snapshot_interval: float = 60.0
    task_interval: float = 300.0
    zombie_threshold_seconds: float = 300.0
    zombie_fail_seconds: float = 600.0
    max_event_rows: int = 5000
    probe_types: Sequence[str] = field(default_factory=tuple)
    bad_ollama_base_url: str = "http://127.0.0.1:1"

    def resolved_probe_types(self) -> List[str]:
        probes = list(self.probe_types)
        if not probes:
            probes = list(DEFAULT_WORKLOAD_PROBES if self.mode == "workload" else DEFAULT_INFRASTRUCTURE_PROBES)
        unsupported = [probe for probe in probes if probe not in SUPPORTED_PROBE_TYPES]
        if unsupported:
            raise ValueError(f"Unsupported soak probe type(s): {', '.join(unsupported)}")
        return probes


@dataclass
class ZombieDetector:
    threshold_seconds: float
    fail_seconds: float
    last_heartbeat_at: float = 0.0
    last_event_count: int = 0
    last_task_progress_at: float = 0.0
    last_progress_seen_at: float = 0.0
    suspected_at: float = 0.0
    intervals: List[Dict[str, float]] = field(default_factory=list)

    def observe(self, *, run: Mapping[str, Any], event_count: int, task_progress_at: float, now: Optional[float] = None) -> Dict[str, Any]:
        observed_at = float(now if now is not None else time.time())
        heartbeat_at = float(run.get("heartbeat_updated_at", 0.0) or 0.0)
        progress_changed = event_count > self.last_event_count or task_progress_at > self.last_task_progress_at
        heartbeat_moved = heartbeat_at > self.last_heartbeat_at
        if self.last_progress_seen_at <= 0:
            self.last_progress_seen_at = observed_at
        if progress_changed:
            if self.suspected_at > 0:
                self.intervals.append({"started_at": self.suspected_at, "ended_at": observed_at})
            self.suspected_at = 0.0
            self.last_progress_seen_at = observed_at
        status = "OK"
        reason = ""
        if heartbeat_moved and not progress_changed and (observed_at - self.last_progress_seen_at) >= float(self.threshold_seconds):
            if self.suspected_at <= 0:
                self.suspected_at = observed_at
            status = "ZOMBIE_SUSPECTED"
            reason = "heartbeat_without_progress"
            if (observed_at - self.suspected_at) >= float(self.fail_seconds):
                status = "FAILED"
                reason = "zombie_persisted"
        self.last_heartbeat_at = max(self.last_heartbeat_at, heartbeat_at)
        self.last_event_count = max(self.last_event_count, int(event_count))
        self.last_task_progress_at = max(self.last_task_progress_at, float(task_progress_at))
        return {
            "status": status,
            "reason": reason,
            "suspected_at": self.suspected_at,
            "last_progress_seen_at": self.last_progress_seen_at,
            "intervals": list(self.intervals),
        }


class ResourceTrendSampler:
    def __init__(self, *, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.last_wall_at = 0.0
        self.last_cpu_seconds = 0.0
        self.snapshot_count = 0

    def snapshot(
        self,
        *,
        watchdog_payload: Mapping[str, Any],
        event_count: int,
        degraded_reason: str = "",
    ) -> Dict[str, Any]:
        now = time.time()
        cpu_seconds = self._process_cpu_seconds()
        cpu_percent = 0.0
        if self.last_wall_at > 0 and now > self.last_wall_at:
            cpu_count = max(1, os.cpu_count() or 1)
            cpu_percent = max(0.0, ((cpu_seconds - self.last_cpu_seconds) / (now - self.last_wall_at)) * 100.0 / cpu_count)
        self.last_wall_at = now
        self.last_cpu_seconds = cpu_seconds
        self.snapshot_count += 1
        checks = _json_dict(watchdog_payload.get("checks"))
        memory = _json_dict(checks.get("memory"))
        disk = _json_dict(checks.get("disk"))
        ollama = _json_dict(checks.get("ollama"))
        latency = float(ollama.get("latency_seconds", 0.0) or 0.0) * 1000.0
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "memory_mb": float(memory.get("rss_mb", 0.0) or 0.0),
            "cpu_percent": cpu_percent,
            "disk_free_gb": float(disk.get("free_mb", 0.0) or 0.0) / 1024.0,
            "db_size_mb": float(db_size) / (1024.0 * 1024.0),
            "event_count": int(event_count),
            "snapshot_count": self.snapshot_count,
            "ollama_latency_ms": latency,
            "status": str(watchdog_payload.get("status", "UNKNOWN")),
            "degraded_reason": degraded_reason,
        }

    def _process_cpu_seconds(self) -> float:
        try:
            times = os.times()
        except Exception:
            return 0.0
        return float(times.user + times.system)


class SoakRunner:
    def __init__(
        self,
        *,
        supervisor: LongRunSupervisor,
        watchdog: ResourceWatchdog,
        paths: RuntimePaths,
        config: SoakConfig,
    ) -> None:
        mode = str(config.mode or "infrastructure")
        if mode not in SOAK_MODES:
            raise ValueError(f"Unsupported soak mode: {mode}")
        self.supervisor = supervisor
        self.watchdog = watchdog
        self.paths = paths.ensure()
        self.config = config
        self.config.mode = mode
        self.probe_types = config.resolved_probe_types()
        self.probe_index = 0
        self.trend_sampler = ResourceTrendSampler(db_path=self.supervisor.state_store.db_path)
        self.zombie_detector = ZombieDetector(config.zombie_threshold_seconds, config.zombie_fail_seconds)
        self.probe_results: List[Dict[str, Any]] = []
        self.resource_trends: List[Dict[str, Any]] = []
        self.failures: List[str] = []
        self.degraded_count = 0
        self.tick_count = 0
        self.approval_pause_resume_result: Dict[str, Any] = {}
        self.crash_recovery_result: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        started_at = time.time()
        expected_end_at = started_at + max(0.0, float(self.config.duration_seconds))
        soak_id = new_id("soak")
        run_id = f"{soak_id}-run"
        snapshot_path = self.paths.soak_dir / f"{soak_id}.jsonl"
        self.supervisor.create_run(f"Con OS soak session ({self.config.mode})", run_id=run_id)
        self.supervisor.state_store.create_soak_session(
            soak_id=soak_id,
            mode=self.config.mode,
            expected_end_at=expected_end_at,
            current_run_id=run_id,
            summary={"schema_version": SOAK_RUNNER_VERSION, "probe_types": list(self.probe_types)},
        )
        next_snapshot_at = 0.0
        next_task_at = started_at
        if self.config.mode == "infrastructure":
            for probe_type in self.probe_types:
                self.inject_probe(run_id, probe_type)
            self.crash_recovery_result = self._simulate_worker_recovery(run_id)
        try:
            while time.time() <= expected_end_at:
                now = time.time()
                if self.config.mode == "workload" and now >= next_task_at:
                    self.inject_probe(run_id, self._next_probe_type())
                    next_task_at = now + max(0.01, float(self.config.task_interval))
                payload = tick_runtime_once(
                    self.supervisor,
                    watchdog=self.watchdog,
                    snapshot_path=None,
                    max_event_rows=int(self.config.max_event_rows),
                )
                self.tick_count += 1
                if str(payload.get("watchdog", {}).get("status", "")) == "DEGRADED":
                    self.degraded_count += 1
                zombie = self._observe_zombie(run_id)
                if zombie["status"] == "ZOMBIE_SUSPECTED":
                    self.supervisor.state_store.update_soak_session(soak_id, status="ZOMBIE_SUSPECTED")
                if zombie["status"] == "FAILED":
                    self.failures.append("zombie_persisted")
                    self.supervisor.state_store.update_run_status(run_id, "FAILED", paused_reason="soak_zombie_detected")
                    break
                if now >= next_snapshot_at:
                    snapshot = self._snapshot(soak_id, run_id, payload, zombie)
                    append_status_snapshot(snapshot_path, snapshot)
                    self.supervisor.state_store.update_soak_session(soak_id, last_snapshot_at=float(snapshot["created_at"]))
                    next_snapshot_at = now + max(0.01, float(self.config.snapshot_interval))
                sleep_for = max(0.01, float(self.config.tick_interval))
                if time.time() + sleep_for > expected_end_at:
                    remaining = expected_end_at - time.time()
                    if remaining > 0:
                        time.sleep(remaining)
                    break
                time.sleep(sleep_for)
            if not self.resource_trends:
                payload = tick_runtime_once(
                    self.supervisor,
                    watchdog=self.watchdog,
                    snapshot_path=None,
                    max_event_rows=int(self.config.max_event_rows),
                )
                self.tick_count += 1
                zombie = self._observe_zombie(run_id)
                snapshot = self._snapshot(soak_id, run_id, payload, zombie)
                append_status_snapshot(snapshot_path, snapshot)
                self.supervisor.state_store.update_soak_session(soak_id, last_snapshot_at=float(snapshot["created_at"]))
            status = "FAILED" if self.failures else "PASSED"
            summary = self._final_summary(
                soak_id=soak_id,
                run_id=run_id,
                started_at=started_at,
                expected_end_at=expected_end_at,
                snapshot_path=snapshot_path,
                status=status,
            )
            self.supervisor.state_store.update_soak_session(
                soak_id,
                status=status,
                failure_count=len(set(self.failures)),
                summary=summary,
            )
            append_status_snapshot(snapshot_path, summary)
            return summary
        except Exception as exc:
            self.failures.append("runtime_crashed")
            summary = self._final_summary(
                soak_id=soak_id,
                run_id=run_id,
                started_at=started_at,
                expected_end_at=expected_end_at,
                snapshot_path=snapshot_path,
                status="FAILED",
                error=str(exc),
            )
            self.supervisor.state_store.update_soak_session(
                soak_id,
                status="FAILED",
                failure_count=len(set(self.failures)),
                summary=summary,
            )
            append_status_snapshot(snapshot_path, summary)
            return summary

    def inject_probe(self, run_id: str, probe_type: str) -> Dict[str, Any]:
        if probe_type not in SUPPORTED_PROBE_TYPES:
            raise ValueError(f"Unsupported soak probe type: {probe_type}")
        run = self.supervisor.state_store.get_run(run_id)
        if run.get("status") in TERMINAL_STATUSES:
            self.supervisor.state_store.update_run_status(run_id, "RUNNING")
        if probe_type == "approval_pause_resume":
            result = self._probe_approval_pause_resume(run_id)
        elif probe_type == "dummy_verifier_fail":
            result = self._probe_manual(run_id, probe_type, status="FAILED", details={"expected_failure": True})
        elif probe_type == "db_integrity":
            result = self._probe_db_integrity(run_id)
        elif probe_type == "event_roundtrip":
            result = self._probe_event_roundtrip(run_id)
        elif probe_type == "watchdog_snapshot":
            result = self._probe_watchdog_snapshot(run_id)
        elif probe_type == "ollama_ping":
            result = self._probe_ollama_ping(run_id)
        else:
            result = self._probe_manual(run_id, probe_type, status="COMPLETED", details={"verified": True})
        self.probe_results.append(result)
        return result

    def _next_probe_type(self) -> str:
        probe_type = self.probe_types[self.probe_index % len(self.probe_types)]
        self.probe_index += 1
        return probe_type

    def _probe_manual(self, run_id: str, probe_type: str, *, status: str, details: Mapping[str, Any]) -> Dict[str, Any]:
        task_id = self.supervisor.add_task(run_id, f"soak probe: {probe_type}", priority=1, verifier={"probe_type": probe_type})
        self.supervisor.state_store.update_task_status(task_id, "RUNNING", result={"probe_type": probe_type})
        self.supervisor.event_journal.append(run_id=run_id, task_id=task_id, event_type="probe_started", payload={"probe_type": probe_type})
        result = {"probe_type": probe_type, **dict(details)}
        self.supervisor.state_store.update_task_status(task_id, status, result=result)
        self.supervisor.event_journal.append(
            run_id=run_id,
            task_id=task_id,
            event_type="probe_completed" if status == "COMPLETED" else "probe_failed",
            payload=result,
        )
        if status == "FAILED" and not result.get("expected_failure"):
            self.failures.append(f"probe_failed:{probe_type}")
        return {"probe_type": probe_type, "task_id": task_id, "status": status, "result": result}

    def _probe_db_integrity(self, run_id: str) -> Dict[str, Any]:
        try:
            db_path = self.supervisor.state_store.db_path
            if str(db_path) == ":memory:":
                integrity = "ok"
            else:
                with sqlite3.connect(str(db_path)) as conn:
                    row = conn.execute("PRAGMA integrity_check").fetchone()
                    integrity = str(row[0] if row else "")
            status = "COMPLETED" if integrity.lower() == "ok" else "FAILED"
            return self._probe_manual(run_id, "db_integrity", status=status, details={"integrity": integrity})
        except Exception as exc:
            return self._probe_manual(run_id, "db_integrity", status="FAILED", details={"error": str(exc)})

    def _probe_event_roundtrip(self, run_id: str) -> Dict[str, Any]:
        task_id = self.supervisor.add_task(run_id, "soak probe: event_roundtrip", priority=1, verifier={"probe_type": "event_roundtrip"})
        self.supervisor.state_store.update_task_status(task_id, "RUNNING", result={"probe_type": "event_roundtrip"})
        marker = self.supervisor.event_journal.append(
            run_id=run_id,
            task_id=task_id,
            event_type="probe_roundtrip_marker",
            payload={"probe_type": "event_roundtrip"},
        )
        found = any(event["event_id"] == marker["event_id"] for event in self.supervisor.state_store.list_events(run_id))
        status = "COMPLETED" if found else "FAILED"
        result = {"probe_type": "event_roundtrip", "event_id": marker["event_id"], "roundtrip_ok": found}
        self.supervisor.state_store.update_task_status(task_id, status, result=result)
        self.supervisor.event_journal.append(run_id=run_id, task_id=task_id, event_type="probe_completed", payload=result)
        if not found:
            self.failures.append("probe_failed:event_roundtrip")
        return {"probe_type": "event_roundtrip", "task_id": task_id, "status": status, "result": result}

    def _probe_watchdog_snapshot(self, run_id: str) -> Dict[str, Any]:
        payload = self.watchdog.evaluate()
        status = "COMPLETED" if payload.get("status") != "DEGRADED" else "FAILED"
        if status == "FAILED":
            self.degraded_count += 1
        return self._probe_manual(run_id, "watchdog_snapshot", status=status, details={"watchdog": payload})

    def _probe_ollama_ping(self, run_id: str) -> Dict[str, Any]:
        normal = self.watchdog.evaluate()
        bad_watchdog = ResourceWatchdog(
            runtime_home=self.paths.runtime_home,
            thresholds=WatchdogThresholds(
                ollama_base_url=self.config.bad_ollama_base_url,
                ollama_timeout_seconds=0.1,
                ollama_required=True,
            ),
        )
        simulated = bad_watchdog.evaluate()
        checks = _json_dict(normal.get("checks"))
        ollama = _json_dict(checks.get("ollama"))
        status = "COMPLETED" if simulated.get("status") == "DEGRADED" else "FAILED"
        details = {
            "normal_ollama": ollama,
            "simulated_unreachable": simulated,
            "simulation_used_temporary_endpoint": True,
        }
        return self._probe_manual(run_id, "ollama_ping", status=status, details=details)

    def _probe_approval_pause_resume(self, run_id: str) -> Dict[str, Any]:
        task_id = self.supervisor.add_task(
            run_id,
            "soak probe: approval_pause_resume",
            priority=1,
            verifier={"probe_type": "approval_pause_resume", "requires_approval": True},
        )
        started = self.supervisor.tick_once(run_id)
        waiting = self.supervisor.tick_once(run_id)
        approval_id = str(waiting.get("approval_id", "") or "")
        approved = self.supervisor.approve(approval_id, approved_by="soak_runner") if approval_id else {"status": "NO_APPROVAL"}
        restarted = self.supervisor.tick_once(run_id)
        completed = self.supervisor.tick_once(run_id)
        passed = (
            started.get("status") == "TASK_STARTED"
            and waiting.get("status") == "WAITING_APPROVAL"
            and approved.get("status") == "APPROVED"
            and completed.get("status") in {"COMPLETED", "TASK_COMPLETED_NEXT_STARTED"}
        )
        result = {
            "probe_type": "approval_pause_resume",
            "approval_id": approval_id,
            "started": started,
            "waiting": waiting,
            "approved": approved,
            "restarted": restarted,
            "completed": completed,
            "passed": passed,
        }
        self.approval_pause_resume_result = result
        if not passed:
            self.failures.append("probe_failed:approval_pause_resume")
            self.supervisor.state_store.update_task_status(task_id, "FAILED", result=result)
        return {"probe_type": "approval_pause_resume", "task_id": task_id, "status": "COMPLETED" if passed else "FAILED", "result": result}

    def _simulate_worker_recovery(self, run_id: str) -> Dict[str, Any]:
        run = self.supervisor.state_store.get_run(run_id)
        if run.get("status") in TERMINAL_STATUSES:
            self.supervisor.state_store.update_run_status(run_id, "RUNNING")
        task_id = self.supervisor.add_task(run_id, "soak probe: worker interruption recovery", priority=0, verifier={"probe_type": "worker_recovery"})
        started = self.supervisor.tick_once(run_id)
        recovered = self.supervisor.recover_after_crash(run_id)
        restarted = self.supervisor.tick_once(run_id)
        completed = self.supervisor.tick_once(run_id)
        passed = recovered.get("status") == "RUNNING" and completed.get("status") in {"COMPLETED", "TASK_COMPLETED_NEXT_STARTED"}
        result = {
            "task_id": task_id,
            "started": started,
            "recovered": recovered,
            "restarted": restarted,
            "completed": completed,
            "passed": passed,
        }
        if not passed:
            self.failures.append("probe_failed:worker_recovery")
        return result

    def _observe_zombie(self, run_id: str) -> Dict[str, Any]:
        run = self.supervisor.state_store.get_run(run_id)
        event_count = self.supervisor.state_store.count_events(run_id)
        task_progress = self.supervisor.state_store.latest_task_progress_at(run_id)
        return self.zombie_detector.observe(run=run, event_count=event_count, task_progress_at=task_progress)

    def _snapshot(
        self,
        soak_id: str,
        run_id: str,
        payload: Mapping[str, Any],
        zombie: Mapping[str, Any],
    ) -> Dict[str, Any]:
        event_count = self.supervisor.state_store.count_events(run_id)
        watchdog_payload = _json_dict(payload.get("watchdog"))
        degraded_reason = ",".join(str(item) for item in watchdog_payload.get("degraded_reasons", []) or [])
        trend = self.trend_sampler.snapshot(
            watchdog_payload=watchdog_payload,
            event_count=event_count,
            degraded_reason=degraded_reason,
        )
        self.resource_trends.append(trend)
        return {
            "schema_version": SOAK_RUNNER_VERSION,
            "created_at": time.time(),
            "soak_id": soak_id,
            "run_id": run_id,
            "mode": self.config.mode,
            "probe_count": len(self.probe_results),
            "runtime_tick": dict(payload),
            "resource_trend": trend,
            "zombie": dict(zombie),
        }

    def _final_summary(
        self,
        *,
        soak_id: str,
        run_id: str,
        started_at: float,
        expected_end_at: float,
        snapshot_path: Path,
        status: str,
        error: str = "",
    ) -> Dict[str, Any]:
        final = self.supervisor.status(run_id)
        completed = sum(1 for probe in self.probe_results if probe.get("status") == "COMPLETED")
        failed = sum(1 for probe in self.probe_results if probe.get("status") == "FAILED")
        ollama_latencies = [trend.get("ollama_latency_ms", 0.0) for trend in self.resource_trends if trend.get("ollama_latency_ms")]
        max_memory = max((trend.get("memory_mb", 0.0) for trend in self.resource_trends), default=0.0)
        zombie_intervals = list(self.zombie_detector.intervals)
        if self.zombie_detector.suspected_at > 0:
            zombie_intervals.append({"started_at": self.zombie_detector.suspected_at, "ended_at": time.time()})
        summary = {
            "schema_version": SOAK_RUNNER_VERSION,
            "status": status,
            "soak_id": soak_id,
            "run_id": run_id,
            "mode": self.config.mode,
            "started_at": started_at,
            "expected_end_at": expected_end_at,
            "completed_at": time.time(),
            "total_duration_seconds": max(0.0, time.time() - started_at),
            "configured_duration_seconds": float(self.config.duration_seconds),
            "snapshot_path": str(snapshot_path),
            "tick_count": self.tick_count,
            "total_probe_tasks": len(self.probe_results),
            "completed_probe_tasks": completed,
            "failed_probe_tasks": failed,
            "degraded_count": self.degraded_count,
            "max_memory_mb": max_memory,
            "p95_ollama_latency_ms": _percentile(ollama_latencies, 0.95),
            "zombie_intervals": zombie_intervals,
            "approval_pause_resume_result": self.approval_pause_resume_result,
            "crash_recovery_result": self.crash_recovery_result,
            "failures": sorted(set(self.failures)),
            "final": final,
        }
        if error:
            summary["error"] = error
        return summary

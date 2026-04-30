from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Dict, Iterable, Optional, Sequence

from core.runtime.autonomous_tick import DEFAULT_COOLDOWN_SECONDS, ensure_autonomous_run
from core.runtime.long_run_supervisor import LongRunSupervisor, TERMINAL_STATUSES
from core.runtime.paths import RuntimePaths
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds
from core.runtime.runtime_modes import infer_runtime_mode
from core.runtime.vm_watchdog import ManagedVMWatchdog, ManagedVMWatchdogConfig
from core.runtime_paths import default_state_path


SERVICE_DAEMON_VERSION = "conos.service_daemon/v1"
SKIP_TICK_STATUSES = TERMINAL_STATUSES | {"PAUSED", "WAITING_APPROVAL"}


class StopFlag:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self, *_args: object) -> None:
        self.stopped = True


def append_status_snapshot(path: str | Path, payload: Dict[str, Any]) -> None:
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _active_runs(supervisor: LongRunSupervisor) -> Iterable[Dict[str, Any]]:
    for run in supervisor.state_store.list_runs():
        if str(run.get("status", "")) not in SKIP_TICK_STATUSES:
            yield run


def _exception_payload(exc: BaseException) -> Dict[str, Any]:
    return {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc()[-4000:],
    }


def _mark_tick_exception(supervisor: LongRunSupervisor, run_id: str, error: Dict[str, Any]) -> Dict[str, Any]:
    try:
        run = supervisor.mark_degraded(run_id, "tick_exception", details=error)
        return {"run_status": run.get("status", "DEGRADED"), "marked_degraded": True}
    except Exception as mark_error:
        return {
            "run_status": "UNKNOWN",
            "marked_degraded": False,
            "mark_error": _exception_payload(mark_error),
        }


def _combined_watchdog_payload(resource_payload: Dict[str, Any], vm_payload: Dict[str, Any]) -> Dict[str, Any]:
    resource_status = str(resource_payload.get("status", "") or "")
    vm_status = str(vm_payload.get("status", "") or "")
    degraded_reasons = []
    if resource_status == "DEGRADED":
        degraded_reasons.extend(
            [f"resource:{reason}" for reason in resource_payload.get("degraded_reasons", [])]
            or [str(resource_payload.get("reason") or "resource_watchdog_degraded")]
        )
    if vm_status == "DEGRADED":
        degraded_reasons.append(f"vm:{vm_payload.get('reason') or 'managed_vm_unhealthy'}")
    if resource_status == "SKIPPED" and vm_status == "SKIPPED":
        status = "SKIPPED"
    else:
        status = "DEGRADED" if degraded_reasons else "OK"
    return {
        "status": status,
        "resource_status": resource_status or "UNKNOWN",
        "vm_status": vm_status or "UNKNOWN",
        "degraded_reasons": degraded_reasons,
        "resource": resource_payload,
        "vm": vm_payload,
    }


def _degraded_tick_reason(*, resource_degraded: bool, vm_degraded: bool) -> str:
    if resource_degraded and vm_degraded:
        return "runtime_watchdog_degraded"
    if vm_degraded:
        return "vm_watchdog_degraded"
    return "resource_watchdog_degraded"


def tick_runtime_once(
    supervisor: LongRunSupervisor,
    *,
    watchdog: Optional[ResourceWatchdog],
    vm_watchdog: Optional[ManagedVMWatchdog] = None,
    snapshot_path: str | Path | None = None,
    max_event_rows: int = 5000,
    zombie_threshold_seconds: float = 600.0,
    zombie_fail_seconds: float = 0.0,
    autonomous_tick_enabled: bool = False,
    autonomous_state_path: str | Path | None = None,
    autonomous_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> Dict[str, Any]:
    started_at = time.time()
    resource_watchdog_payload = watchdog.evaluate() if watchdog is not None else {"status": "SKIPPED"}
    vm_watchdog_payload = vm_watchdog.evaluate() if vm_watchdog is not None else {"status": "SKIPPED"}
    watchdog_payload = _combined_watchdog_payload(resource_watchdog_payload, vm_watchdog_payload)
    tick_results = []
    resource_degraded = str(resource_watchdog_payload.get("status", "")) == "DEGRADED"
    vm_degraded = str(vm_watchdog_payload.get("status", "")) == "DEGRADED"
    degraded = resource_degraded or vm_degraded
    degraded_reason = _degraded_tick_reason(resource_degraded=resource_degraded, vm_degraded=vm_degraded)
    autonomous_tick_payload: Dict[str, Any] = {"status": "DISABLED"}
    if not degraded and autonomous_tick_enabled:
        autonomous_tick_payload = ensure_autonomous_run(
            supervisor,
            state_path=autonomous_state_path,
            enabled=True,
            cooldown_seconds=float(autonomous_cooldown_seconds),
        )
    active_runs = list(_active_runs(supervisor))
    if degraded:
        for run in active_runs:
            run_id = str(run["run_id"])
            try:
                tick_results.append(
                    {
                        "run_id": run_id,
                        "status": supervisor.mark_degraded(
                            run_id,
                            degraded_reason,
                            details=watchdog_payload,
                        ).get("status", "DEGRADED"),
                    }
                )
            except Exception as exc:
                error = _exception_payload(exc)
                tick_results.append({"run_id": run_id, "status": "TICK_EXCEPTION", "error": error})
    else:
        for run in active_runs:
            run_id = str(run["run_id"])
            try:
                if run.get("status") == "DEGRADED":
                    if str(run.get("paused_reason", "")) == "zombie_suspected":
                        tick_results.append({"run_id": run_id, "status": "DEGRADED", "reason": "zombie_suspected"})
                        continue
                    supervisor.clear_degraded(run_id)
                tick_results.append(supervisor.tick_once(run_id))
            except Exception as exc:
                error = _exception_payload(exc)
                degraded_result = _mark_tick_exception(supervisor, run_id, error)
                tick_results.append(
                    {
                        "run_id": run_id,
                        "status": "TICK_EXCEPTION",
                        "error": error,
                        "degraded": degraded_result,
                    }
                )
    maintenance = supervisor.maintenance_once(
        max_events_per_run=int(max_event_rows),
        zombie_threshold_seconds=float(zombie_threshold_seconds),
        zombie_fail_seconds=float(zombie_fail_seconds),
        checkpoint_wal=True,
    )
    runs = supervisor.state_store.list_runs()
    mode_tasks = []
    for run in runs:
        mode_tasks.extend(supervisor.state_store.list_tasks(str(run.get("run_id", ""))))
    runtime_mode = infer_runtime_mode(
        runs=runs,
        tasks=mode_tasks,
        watchdog=watchdog_payload,
        soak_sessions=supervisor.state_store.list_soak_sessions(),
    ).to_dict()
    payload = {
        "schema_version": SERVICE_DAEMON_VERSION,
        "created_at": time.time(),
        "duration_seconds": max(0.0, time.time() - started_at),
        "runtime_mode": runtime_mode,
        "watchdog": watchdog_payload,
        "resource_watchdog": resource_watchdog_payload,
        "vm_watchdog": vm_watchdog_payload,
        "autonomous_tick": autonomous_tick_payload,
        "ticks": tick_results,
        "metrics": supervisor.metrics(),
        "maintenance": maintenance,
        "prune": maintenance.get("prune", {"deleted": 0}),
    }
    if snapshot_path is not None:
        append_status_snapshot(snapshot_path, payload)
    return payload


def run_daemon(
    *,
    supervisor: LongRunSupervisor,
    watchdog: Optional[ResourceWatchdog],
    vm_watchdog: Optional[ManagedVMWatchdog] = None,
    snapshot_path: str | Path,
    tick_interval: float,
    max_event_rows: int,
    zombie_threshold_seconds: float = 600.0,
    zombie_fail_seconds: float = 0.0,
    autonomous_tick_enabled: bool = True,
    autonomous_state_path: str | Path | None = None,
    autonomous_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    stop_flag: Optional[StopFlag] = None,
) -> Dict[str, Any]:
    flag = stop_flag or StopFlag()
    last: Dict[str, Any] = {"status": "NOT_STARTED"}
    while not flag.stopped:
        last = tick_runtime_once(
            supervisor,
            watchdog=watchdog,
            vm_watchdog=vm_watchdog,
            snapshot_path=snapshot_path,
            max_event_rows=max_event_rows,
            zombie_threshold_seconds=float(zombie_threshold_seconds),
            zombie_fail_seconds=float(zombie_fail_seconds),
            autonomous_tick_enabled=bool(autonomous_tick_enabled),
            autonomous_state_path=autonomous_state_path,
            autonomous_cooldown_seconds=float(autonomous_cooldown_seconds),
        )
        time.sleep(max(0.0, float(tick_interval)))
    return {"status": "STOPPED", "last": last}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m core.runtime.service_daemon")
    parser.add_argument("--runtime-home", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--tick-interval", type=float, default=5.0)
    parser.add_argument("--watchdog-interval", type=float, default=30.0)
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--max-event-rows", type=int, default=5000)
    parser.add_argument("--zombie-threshold", type=float, default=600.0)
    parser.add_argument("--zombie-fail-threshold", type=float, default=0.0)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-timeout", type=float, default=10.0)
    parser.add_argument("--ollama-required", action="store_true")
    parser.add_argument("--vm-watchdog", action="store_true")
    parser.add_argument("--vm-auto-recover", action="store_true")
    parser.add_argument("--vm-restart-unready", action="store_true")
    parser.add_argument("--vm-state-root", default=None)
    parser.add_argument("--vm-image-id", default="conos-base")
    parser.add_argument("--vm-instance-id", default="default")
    parser.add_argument("--vm-helper-path", default="")
    parser.add_argument("--vm-runner-path", default="")
    parser.add_argument("--vm-network-mode", default="provider_default")
    parser.add_argument("--vm-timeout-seconds", type=int, default=120)
    parser.add_argument("--vm-startup-wait-seconds", type=float, default=15.0)
    parser.add_argument("--vm-guest-wait-seconds", type=float, default=180.0)
    parser.add_argument("--vm-no-build-runner", action="store_true")
    parser.add_argument("--autonomous-state-path", default="")
    parser.add_argument("--autonomous-tick-cooldown", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--disable-autonomous-tick", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = RuntimePaths.from_env(args.runtime_home).resolved()
    if args.db:
        paths = RuntimePaths(
            runtime_home=paths.runtime_home,
            state_db=Path(args.db).expanduser(),
            runs_root=Path(args.runs_root).expanduser() if args.runs_root else paths.runs_root,
            logs_dir=paths.logs_dir,
            snapshots_dir=paths.snapshots_dir,
            soak_dir=paths.soak_dir,
        ).ensure()
    else:
        paths = paths.ensure()
    if args.runs_root and not args.db:
        paths = RuntimePaths(
            runtime_home=paths.runtime_home,
            state_db=paths.state_db,
            runs_root=Path(args.runs_root).expanduser(),
            logs_dir=paths.logs_dir,
            snapshots_dir=paths.snapshots_dir,
            soak_dir=paths.soak_dir,
        ).ensure()
    supervisor = LongRunSupervisor(
        db_path=paths.state_db,
        runs_root=paths.runs_root,
        worker_id=args.worker_id,
    )
    watchdog = ResourceWatchdog(
        runtime_home=paths.runtime_home,
        thresholds=WatchdogThresholds(
            ollama_base_url=args.ollama_base_url,
            ollama_timeout_seconds=float(args.ollama_timeout),
            ollama_required=bool(args.ollama_required),
        ),
    )
    vm_watchdog = None
    if bool(args.vm_watchdog):
        vm_state_root = Path(args.vm_state_root).expanduser() if args.vm_state_root else paths.runtime_home / "vm"
        vm_watchdog = ManagedVMWatchdog(
            ManagedVMWatchdogConfig(
                state_root=str(vm_state_root),
                image_id=str(args.vm_image_id),
                instance_id=str(args.vm_instance_id),
                helper_path=str(args.vm_helper_path or ""),
                runner_path=str(args.vm_runner_path or ""),
                network_mode=str(args.vm_network_mode or "provider_default"),
                timeout_seconds=int(args.vm_timeout_seconds),
                startup_wait_seconds=float(args.vm_startup_wait_seconds),
                guest_wait_seconds=float(args.vm_guest_wait_seconds),
                auto_build_runner=not bool(args.vm_no_build_runner),
                auto_recover=bool(args.vm_auto_recover),
                restart_unready=bool(args.vm_restart_unready),
            )
        )
    snapshot_path = Path(args.snapshot_path).expanduser() if args.snapshot_path else paths.service_status_log
    autonomous_state_path = Path(args.autonomous_state_path).expanduser() if args.autonomous_state_path else default_state_path(paths.runtime_home / "state" / "state.json")
    autonomous_tick_enabled = not bool(args.disable_autonomous_tick)
    try:
        if args.once:
            print(
                json.dumps(
                    tick_runtime_once(
                        supervisor,
                        watchdog=watchdog,
                        vm_watchdog=vm_watchdog,
                        snapshot_path=snapshot_path,
                        max_event_rows=int(args.max_event_rows),
                        zombie_threshold_seconds=float(args.zombie_threshold),
                        zombie_fail_seconds=float(args.zombie_fail_threshold),
                        autonomous_tick_enabled=autonomous_tick_enabled,
                        autonomous_state_path=autonomous_state_path,
                        autonomous_cooldown_seconds=float(args.autonomous_tick_cooldown),
                    ),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
            return 0
        stop_flag = StopFlag()
        signal.signal(signal.SIGTERM, stop_flag.stop)
        signal.signal(signal.SIGINT, stop_flag.stop)
        result = run_daemon(
            supervisor=supervisor,
            watchdog=watchdog,
            vm_watchdog=vm_watchdog,
            snapshot_path=snapshot_path,
            tick_interval=float(args.tick_interval),
            max_event_rows=int(args.max_event_rows),
            zombie_threshold_seconds=float(args.zombie_threshold),
            zombie_fail_seconds=float(args.zombie_fail_threshold),
            autonomous_tick_enabled=autonomous_tick_enabled,
            autonomous_state_path=autonomous_state_path,
            autonomous_cooldown_seconds=float(args.autonomous_tick_cooldown),
            stop_flag=stop_flag,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    finally:
        supervisor.state_store.close()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence

from core.runtime.long_run_supervisor import LongRunSupervisor, TERMINAL_STATUSES
from core.runtime.paths import RuntimePaths
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds


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


def tick_runtime_once(
    supervisor: LongRunSupervisor,
    *,
    watchdog: Optional[ResourceWatchdog],
    snapshot_path: str | Path | None = None,
    max_event_rows: int = 5000,
) -> Dict[str, Any]:
    started_at = time.time()
    watchdog_payload = watchdog.evaluate() if watchdog is not None else {"status": "SKIPPED"}
    tick_results = []
    degraded = str(watchdog_payload.get("status", "")) == "DEGRADED"
    active_runs = list(_active_runs(supervisor))
    if degraded:
        for run in active_runs:
            tick_results.append(
                {
                    "run_id": run["run_id"],
                    "status": supervisor.mark_degraded(
                        str(run["run_id"]),
                        "resource_watchdog_degraded",
                        details=watchdog_payload,
                    ).get("status", "DEGRADED"),
                }
            )
    else:
        for run in active_runs:
            if run.get("status") == "DEGRADED":
                supervisor.clear_degraded(str(run["run_id"]))
            tick_results.append(supervisor.tick_once(str(run["run_id"])))
    prune = supervisor.state_store.prune_events(max_events_per_run=int(max_event_rows)) if max_event_rows else {"deleted": 0}
    payload = {
        "schema_version": SERVICE_DAEMON_VERSION,
        "created_at": time.time(),
        "duration_seconds": max(0.0, time.time() - started_at),
        "watchdog": watchdog_payload,
        "ticks": tick_results,
        "metrics": supervisor.metrics(),
        "prune": prune,
    }
    if snapshot_path is not None:
        append_status_snapshot(snapshot_path, payload)
    return payload


def run_daemon(
    *,
    supervisor: LongRunSupervisor,
    watchdog: Optional[ResourceWatchdog],
    snapshot_path: str | Path,
    tick_interval: float,
    max_event_rows: int,
    stop_flag: Optional[StopFlag] = None,
) -> Dict[str, Any]:
    flag = stop_flag or StopFlag()
    last: Dict[str, Any] = {"status": "NOT_STARTED"}
    while not flag.stopped:
        last = tick_runtime_once(
            supervisor,
            watchdog=watchdog,
            snapshot_path=snapshot_path,
            max_event_rows=max_event_rows,
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
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-timeout", type=float, default=10.0)
    parser.add_argument("--ollama-required", action="store_true")
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
    snapshot_path = Path(args.snapshot_path).expanduser() if args.snapshot_path else paths.service_status_log
    try:
        if args.once:
            print(
                json.dumps(
                    tick_runtime_once(
                        supervisor,
                        watchdog=watchdog,
                        snapshot_path=snapshot_path,
                        max_event_rows=int(args.max_event_rows),
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
            snapshot_path=snapshot_path,
            tick_interval=float(args.tick_interval),
            max_event_rows=int(args.max_event_rows),
            stop_flag=stop_flag,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    finally:
        supervisor.state_store.close()


if __name__ == "__main__":
    raise SystemExit(main())

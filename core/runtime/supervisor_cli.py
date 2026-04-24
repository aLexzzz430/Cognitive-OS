from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Mapping, Sequence

from core.runtime.event_journal import DEFAULT_RUNS_ROOT
from core.runtime.long_run_supervisor import LongRunSupervisor, TERMINAL_STATUSES
from core.runtime.state_store import DEFAULT_STATE_DB


SUPERVISOR_CLI_VERSION = "conos.supervisor_cli/v1"


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True, default=str)


def _json_arg(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return dict(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos supervisor",
        description="Manage local-first long-running Cognitive OS runs.",
    )
    parser.add_argument("--db", default=str(DEFAULT_STATE_DB), help="SQLite supervisor state DB.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory for per-run event journals.")
    parser.add_argument("--worker-id", default=None, help="Stable worker ID for lease ownership.")
    parser.add_argument("--lease-ttl", type=float, default=30.0, help="Seconds before an abandoned lease can be reused.")
    parser.add_argument("--task-watchdog-seconds", type=float, default=300.0, help="Seconds before a RUNNING task is considered stale.")
    parser.add_argument("--max-task-retries", type=int, default=3, help="Retry budget for stale RUNNING tasks.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=30.0, help="Base retry backoff for stale tasks.")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create a resumable run.")
    create_parser.add_argument("--goal", required=True)
    create_parser.add_argument("--run-id", default=None)

    add_task_parser = subparsers.add_parser("add-task", help="Add a task to a run.")
    add_task_parser.add_argument("run_id")
    add_task_parser.add_argument("--objective", required=True)
    add_task_parser.add_argument("--priority", type=int, default=0)
    add_task_parser.add_argument("--verifier-json", type=_json_arg, default={})

    tick_parser = subparsers.add_parser("tick", help="Run one supervised tick.")
    tick_parser.add_argument("run_id")

    daemon_parser = subparsers.add_parser("daemon", help="Tick a run until it stops, waits, or reaches --max-ticks.")
    daemon_parser.add_argument("run_id")
    daemon_parser.add_argument("--tick-interval", type=float, default=1.0)
    daemon_parser.add_argument("--max-ticks", type=int, default=0)

    status_parser = subparsers.add_parser("status", help="Print run state, tasks, approval, and event count.")
    status_parser.add_argument("run_id")

    pause_parser = subparsers.add_parser("pause", help="Pause a run.")
    pause_parser.add_argument("run_id")
    pause_parser.add_argument("--reason", default="operator_pause")

    resume_parser = subparsers.add_parser("resume", help="Resume a paused run.")
    resume_parser.add_argument("run_id")

    recover_parser = subparsers.add_parser("recover", help="Recover a run after a crash.")
    recover_parser.add_argument("run_id")

    subparsers.add_parser("metrics", help="Print supervisor metrics.")

    service_parser = subparsers.add_parser("service-template", help="Write a launchd or systemd auto-restart template.")
    service_parser.add_argument("run_id")
    service_parser.add_argument("--backend", choices=("auto", "launchd", "systemd"), default="auto")
    service_parser.add_argument("--repo-root", default=str(Path.cwd()))
    service_parser.add_argument("--python", default=sys.executable)
    service_parser.add_argument("--tick-interval", type=float, default=1.0)
    service_parser.add_argument("--output", default=None)
    return parser


def _supervisor(args: argparse.Namespace) -> LongRunSupervisor:
    return LongRunSupervisor(
        db_path=str(args.db),
        runs_root=str(args.runs_root),
        worker_id=args.worker_id,
        lease_ttl_seconds=float(args.lease_ttl),
        task_watchdog_seconds=float(args.task_watchdog_seconds),
        max_task_retries=int(args.max_task_retries),
        retry_backoff_seconds=float(args.retry_backoff_seconds),
    )


def _print(payload: Mapping[str, Any]) -> None:
    print(_json_dumps(payload))


def generate_service_template(
    *,
    run_id: str,
    backend: str,
    repo_root: str | Path,
    python: str,
    db_path: str,
    runs_root: str,
    tick_interval: float,
) -> Dict[str, str]:
    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "launchd" if platform.system() == "Darwin" else "systemd"
    repo = Path(repo_root).resolve()
    command = [
        str(python),
        str(repo / "conos_cli.py"),
        "supervisor",
        "--db",
        str(db_path),
        "--runs-root",
        str(runs_root),
        "daemon",
        str(run_id),
        "--tick-interval",
        str(float(tick_interval)),
    ]
    if resolved_backend == "launchd":
        args = "\n".join(f"    <string>{item}</string>" for item in command)
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.conos.supervisor.{run_id}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>{repo}</string>
</dict>
</plist>
"""
    elif resolved_backend == "systemd":
        content = f"""[Unit]
Description=Cognitive OS LongRunSupervisor {run_id}
After=network.target

[Service]
Type=simple
WorkingDirectory={repo}
ExecStart={" ".join(command)}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
    else:
        raise ValueError(f"Unsupported service backend: {backend}")
    return {"backend": resolved_backend, "content": content}


def _daemon_loop(supervisor: LongRunSupervisor, run_id: str, *, tick_interval: float, max_ticks: int) -> Dict[str, Any]:
    last: Dict[str, Any] = {"status": "NOT_STARTED", "run_id": str(run_id)}
    tick_count = 0
    while True:
        run = supervisor.state_store.get_run(run_id)
        if not run:
            return {"status": "RUN_NOT_FOUND", "run_id": str(run_id), "last_tick": last, "tick_count": tick_count}
        if run["status"] in TERMINAL_STATUSES or run["status"] in {"PAUSED", "WAITING_APPROVAL"}:
            return {"status": run["status"], "run_id": str(run_id), "last_tick": last, "tick_count": tick_count}
        last = supervisor.tick_once(run_id)
        tick_count += 1
        status = str(last.get("status", ""))
        if status in TERMINAL_STATUSES or status in {"PAUSED", "WAITING_APPROVAL"}:
            return {"status": status, "run_id": str(run_id), "last_tick": last, "tick_count": tick_count}
        if max_ticks > 0 and tick_count >= max_ticks:
            return {"status": "MAX_TICKS_REACHED", "run_id": str(run_id), "last_tick": last, "tick_count": tick_count}
        time.sleep(max(0.0, float(tick_interval)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    supervisor = _supervisor(args)
    try:
        if args.command == "create":
            run_id = supervisor.create_run(str(args.goal), run_id=args.run_id)
            _print({"run_id": run_id, "status": supervisor.state_store.get_run(run_id).get("status", "")})
            return 0
        if args.command == "add-task":
            task_id = supervisor.add_task(args.run_id, args.objective, priority=args.priority, verifier=args.verifier_json)
            _print({"run_id": args.run_id, "task_id": task_id, "status": "PENDING"})
            return 0
        if args.command == "tick":
            _print(supervisor.tick_once(args.run_id))
            return 0
        if args.command == "daemon":
            _print(_daemon_loop(supervisor, args.run_id, tick_interval=args.tick_interval, max_ticks=args.max_ticks))
            return 0
        if args.command == "status":
            _print(supervisor.status(args.run_id))
            return 0
        if args.command == "pause":
            _print(supervisor.pause_run(args.run_id, args.reason))
            return 0
        if args.command == "resume":
            _print(supervisor.resume_run(args.run_id))
            return 0
        if args.command == "recover":
            _print(supervisor.recover_after_crash(args.run_id))
            return 0
        if args.command == "metrics":
            _print(supervisor.metrics())
            return 0
        if args.command == "service-template":
            rendered = generate_service_template(
                run_id=args.run_id,
                backend=args.backend,
                repo_root=args.repo_root,
                python=args.python,
                db_path=str(args.db),
                runs_root=str(args.runs_root),
                tick_interval=float(args.tick_interval),
            )
            if args.output:
                Path(args.output).write_text(rendered["content"], encoding="utf-8")
                _print({"backend": rendered["backend"], "output": str(args.output)})
            else:
                print(rendered["content"], end="")
            return 0
    finally:
        supervisor.state_store.close()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any, Dict, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape

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
    parser.add_argument("--event-jsonl-max-bytes", type=int, default=5 * 1024 * 1024, help="Rotate per-run events.jsonl after this many bytes.")
    parser.add_argument("--event-jsonl-retained-files", type=int, default=5, help="Number of rotated per-run JSONL files to retain.")
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

    health_parser = subparsers.add_parser("health", help="Print process, store, journal, and run health.")
    health_parser.add_argument("run_id", nargs="?")

    soak_parser = subparsers.add_parser("soak-test", help="Run a bounded local supervisor durability smoke test.")
    soak_parser.add_argument("--run-id", default="supervisor-soak")
    soak_parser.add_argument("--tasks", type=int, default=3)
    soak_parser.add_argument("--ticks", type=int, default=12)
    soak_parser.add_argument("--tick-interval", type=float, default=0.0)

    for command, help_text in (
        ("service-template", "Write a launchd or systemd auto-restart template."),
        ("service-install", "Install a launchd or systemd user service file."),
        ("service-uninstall", "Remove a launchd or systemd user service file."),
    ):
        service_parser = subparsers.add_parser(command, help=help_text)
        service_parser.add_argument("run_id")
        service_parser.add_argument("--backend", choices=("auto", "launchd", "systemd"), default="auto")
        service_parser.add_argument("--repo-root", default=str(Path.cwd()))
        service_parser.add_argument("--python", default=sys.executable)
        service_parser.add_argument("--tick-interval", type=float, default=1.0)
        service_parser.add_argument("--output", default=None)
        service_parser.add_argument("--stdout-log", default=None)
        service_parser.add_argument("--stderr-log", default=None)
        service_parser.add_argument("--dry-run", action="store_true")
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
        event_jsonl_max_bytes=int(args.event_jsonl_max_bytes),
        event_jsonl_retained_files=int(args.event_jsonl_retained_files),
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
    stdout_log: str | None = None,
    stderr_log: str | None = None,
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
        args = "\n".join(f"    <string>{xml_escape(item)}</string>" for item in command)
        stdout_block = f"  <key>StandardOutPath</key>\n  <string>{xml_escape(str(stdout_log))}</string>\n" if stdout_log else ""
        stderr_block = f"  <key>StandardErrorPath</key>\n  <string>{xml_escape(str(stderr_log))}</string>\n" if stderr_log else ""
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
  <string>{xml_escape(str(repo))}</string>
{stdout_block}{stderr_block}
</dict>
</plist>
"""
    elif resolved_backend == "systemd":
        stdout_block = f"StandardOutput=append:{stdout_log}\n" if stdout_log else ""
        stderr_block = f"StandardError=append:{stderr_log}\n" if stderr_log else ""
        content = f"""[Unit]
Description=Cognitive OS LongRunSupervisor {run_id}
After=network.target

[Service]
Type=simple
WorkingDirectory={repo}
ExecStart={shlex.join(command)}
Restart=always
RestartSec=5
{stdout_block}{stderr_block}

[Install]
WantedBy=default.target
"""
    else:
        raise ValueError(f"Unsupported service backend: {backend}")
    return {"backend": resolved_backend, "content": content}


def service_install_path(*, run_id: str, backend: str, home: str | Path | None = None) -> Path:
    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "launchd" if platform.system() == "Darwin" else "systemd"
    root = Path(home).expanduser() if home is not None else Path.home()
    if resolved_backend == "launchd":
        return root / "Library" / "LaunchAgents" / f"dev.conos.supervisor.{run_id}.plist"
    if resolved_backend == "systemd":
        return root / ".config" / "systemd" / "user" / f"dev.conos.supervisor.{run_id}.service"
    raise ValueError(f"Unsupported service backend: {backend}")


def install_service_file(
    *,
    run_id: str,
    backend: str,
    content: str,
    output: str | Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    path = Path(output) if output else service_install_path(run_id=run_id, backend=backend)
    payload = {"backend": backend, "path": str(path), "dry_run": bool(dry_run), "installed": False}
    if dry_run:
        payload["content"] = content
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    payload["installed"] = True
    return payload


def uninstall_service_file(
    *,
    run_id: str,
    backend: str,
    output: str | Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    path = Path(output) if output else service_install_path(run_id=run_id, backend=backend)
    existed = path.exists()
    if not dry_run:
        path.unlink(missing_ok=True)
    return {"backend": backend, "path": str(path), "dry_run": bool(dry_run), "removed": existed and not dry_run, "existed": existed}


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


def _run_soak_test(supervisor: LongRunSupervisor, *, run_id: str, task_count: int, ticks: int, tick_interval: float) -> Dict[str, Any]:
    existing = supervisor.state_store.get_run(run_id)
    if not existing:
        supervisor.create_run("supervisor soak test", run_id=run_id)
        for index in range(max(1, int(task_count))):
            supervisor.add_task(run_id, f"soak task {index + 1}", priority=max(1, int(task_count)) - index)
    tick_results = []
    for _ in range(max(1, int(ticks))):
        result = supervisor.tick_once(run_id)
        tick_results.append(result)
        status = str(result.get("status", ""))
        if status in TERMINAL_STATUSES or status in {"WAITING_APPROVAL", "PAUSED"}:
            break
        time.sleep(max(0.0, float(tick_interval)))
    recovered = supervisor.recover_after_crash(run_id)
    health = supervisor.health(run_id)
    return {
        "status": "PASSED" if recovered.get("status") in {"RUNNING", "COMPLETED"} else "CHECK",
        "run_id": run_id,
        "tick_count": len(tick_results),
        "last_tick": tick_results[-1] if tick_results else {},
        "recovered": recovered,
        "health": health,
    }


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
        if args.command == "health":
            _print(supervisor.health(args.run_id))
            return 0
        if args.command == "soak-test":
            _print(_run_soak_test(supervisor, run_id=args.run_id, task_count=args.tasks, ticks=args.ticks, tick_interval=args.tick_interval))
            return 0
        if args.command in {"service-template", "service-install"}:
            rendered = generate_service_template(
                run_id=args.run_id,
                backend=args.backend,
                repo_root=args.repo_root,
                python=args.python,
                db_path=str(args.db),
                runs_root=str(args.runs_root),
                tick_interval=float(args.tick_interval),
                stdout_log=args.stdout_log,
                stderr_log=args.stderr_log,
            )
            if args.command == "service-install":
                _print(
                    install_service_file(
                        run_id=args.run_id,
                        backend=rendered["backend"],
                        content=rendered["content"],
                        output=args.output,
                        dry_run=bool(args.dry_run),
                    )
                )
                return 0
            if args.output:
                Path(args.output).write_text(rendered["content"], encoding="utf-8")
                _print({"backend": rendered["backend"], "output": str(args.output)})
            else:
                print(rendered["content"], end="")
            return 0
        if args.command == "service-uninstall":
            backend = args.backend
            if backend == "auto":
                backend = "launchd" if platform.system() == "Darwin" else "systemd"
            _print(uninstall_service_file(run_id=args.run_id, backend=backend, output=args.output, dry_run=bool(args.dry_run)))
            return 0
    finally:
        supervisor.state_store.close()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

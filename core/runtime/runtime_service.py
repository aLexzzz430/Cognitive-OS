from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import plistlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from core.runtime.long_run_supervisor import LongRunSupervisor, TERMINAL_STATUSES
from core.runtime.paths import DEFAULT_SERVICE_LABEL, RuntimePaths
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds
from core.runtime.service_daemon import append_status_snapshot, tick_runtime_once


RUNTIME_SERVICE_VERSION = "conos.runtime_service/v0.1"


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)


def _print(payload: Dict[str, Any]) -> None:
    print(_json(payload))


def parse_duration_seconds(text: str) -> float:
    value = str(text or "").strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("duration is required")
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    suffix = value[-1]
    if suffix in units:
        return float(value[:-1]) * units[suffix]
    return float(value)


def _tail(path: Path, line_count: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = deque(handle, maxlen=max(0, int(line_count)))
    return "".join(lines)


@dataclass(frozen=True)
class RuntimeServiceConfig:
    paths: RuntimePaths
    repo_root: Path
    python_executable: str = sys.executable
    label: str = DEFAULT_SERVICE_LABEL
    home: Path = Path.home()
    tick_interval: float = 5.0
    watchdog_interval: float = 30.0
    snapshot_interval: float = 60.0
    max_event_rows: int = 5000
    ollama_base_url: Optional[str] = None
    ollama_timeout: float = 10.0
    ollama_required: bool = False

    @classmethod
    def from_args(
        cls,
        *,
        runtime_home: str | None = None,
        repo_root: str | None = None,
        python_executable: str | None = None,
        tick_interval: float = 5.0,
        watchdog_interval: float = 30.0,
        snapshot_interval: float = 60.0,
        max_event_rows: int = 5000,
        ollama_base_url: str | None = None,
        ollama_timeout: float = 10.0,
        ollama_required: bool = False,
        home: str | None = None,
    ) -> "RuntimeServiceConfig":
        paths = RuntimePaths.from_env(runtime_home).resolved()
        return cls(
            paths=paths,
            repo_root=Path(repo_root or Path.cwd()).expanduser().resolve(),
            python_executable=str(python_executable or sys.executable),
            home=Path(home).expanduser() if home else Path.home(),
            tick_interval=float(tick_interval),
            watchdog_interval=float(watchdog_interval),
            snapshot_interval=float(snapshot_interval),
            max_event_rows=int(max_event_rows),
            ollama_base_url=ollama_base_url or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("CONOS_OLLAMA_BASE_URL") or None,
            ollama_timeout=float(ollama_timeout),
            ollama_required=bool(ollama_required),
        )

    @property
    def launch_agent_path(self) -> Path:
        return self.home / "Library" / "LaunchAgents" / f"{self.label}.plist"

    def ensured_paths(self) -> RuntimePaths:
        return self.paths.ensure()

    def supervisor(self) -> LongRunSupervisor:
        paths = self.ensured_paths()
        return LongRunSupervisor(db_path=paths.state_db, runs_root=paths.runs_root)

    def watchdog(self) -> ResourceWatchdog:
        paths = self.ensured_paths()
        return ResourceWatchdog(
            runtime_home=paths.runtime_home,
            thresholds=WatchdogThresholds(
                ollama_base_url=self.ollama_base_url,
                ollama_timeout_seconds=float(self.ollama_timeout),
                ollama_required=bool(self.ollama_required),
            ),
        )

    def launchd_plist(self) -> Dict[str, Any]:
        paths = self.paths.resolved()
        env = paths.as_env()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(self.repo_root)
        if self.ollama_base_url:
            env["OLLAMA_BASE_URL"] = str(self.ollama_base_url)
            env["CONOS_OLLAMA_BASE_URL"] = str(self.ollama_base_url)
        program_arguments: List[str] = [
            str(self.python_executable),
            "-m",
            "core.runtime.service_daemon",
            "--runtime-home",
            str(paths.runtime_home),
            "--db",
            str(paths.state_db),
            "--runs-root",
            str(paths.runs_root),
            "--tick-interval",
            str(float(self.tick_interval)),
            "--watchdog-interval",
            str(float(self.watchdog_interval)),
            "--snapshot-path",
            str(paths.service_status_log),
            "--max-event-rows",
            str(int(self.max_event_rows)),
            "--ollama-timeout",
            str(float(self.ollama_timeout)),
        ]
        if self.ollama_base_url:
            program_arguments.extend(["--ollama-base-url", str(self.ollama_base_url)])
        if self.ollama_required:
            program_arguments.append("--ollama-required")
        return {
            "Label": self.label,
            "ProgramArguments": program_arguments,
            "WorkingDirectory": str(self.repo_root),
            "EnvironmentVariables": env,
            "StandardOutPath": str(paths.stdout_log),
            "StandardErrorPath": str(paths.stderr_log),
            "RunAtLoad": True,
            "KeepAlive": True,
        }

    def launchd_plist_text(self) -> str:
        return plistlib.dumps(self.launchd_plist(), sort_keys=False).decode("utf-8")


class RuntimeService:
    def __init__(self, config: RuntimeServiceConfig) -> None:
        self.config = config

    def install_service(self, *, dry_run: bool = False) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        payload = {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "install-service",
            "dry_run": bool(dry_run),
            "plist_path": str(self.config.launch_agent_path),
            "runtime_paths": paths.as_dict(),
            "launchd": self.config.launchd_plist(),
            "installed": False,
        }
        if dry_run:
            payload["plist"] = self.config.launchd_plist_text()
            return payload
        self.config.ensured_paths()
        self.config.launch_agent_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.launch_agent_path.write_text(self.config.launchd_plist_text(), encoding="utf-8")
        payload["installed"] = True
        return payload

    def uninstall_service(self, *, dry_run: bool = False) -> Dict[str, Any]:
        path = self.config.launch_agent_path
        existed = path.exists()
        if not dry_run:
            path.unlink(missing_ok=True)
        return {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "uninstall-service",
            "dry_run": bool(dry_run),
            "plist_path": str(path),
            "existed": existed,
            "removed": existed and not dry_run,
        }

    def start_service(self, *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            return {"schema_version": RUNTIME_SERVICE_VERSION, "action": "start", "dry_run": True, "commands": self._start_commands()}
        if not self.config.launch_agent_path.exists():
            self.install_service(dry_run=False)
        return self._run_launchctl_commands("start", self._start_commands())

    def stop_service(self, *, dry_run: bool = False) -> Dict[str, Any]:
        if dry_run:
            return {"schema_version": RUNTIME_SERVICE_VERSION, "action": "stop", "dry_run": True, "commands": self._stop_commands()}
        return self._run_launchctl_commands("stop", self._stop_commands())

    def status(self) -> Dict[str, Any]:
        paths = self.config.ensured_paths()
        supervisor = self.config.supervisor()
        try:
            watchdog = self.config.watchdog().evaluate()
            payload = {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "status": "OK" if watchdog.get("status") != "DEGRADED" else "DEGRADED",
                "label": self.config.label,
                "plist_path": str(self.config.launch_agent_path),
                "plist_installed": self.config.launch_agent_path.exists(),
                "runtime_paths": paths.as_dict(),
                "metrics": supervisor.metrics(),
                "runs": supervisor.state_store.list_runs(),
                "waiting_approvals": supervisor.state_store.list_approvals(status="WAITING"),
                "watchdog": watchdog,
                "launchd": self._launchd_print(),
            }
            return payload
        finally:
            supervisor.state_store.close()

    def logs(self, *, tail: int = 120) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        return {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "stdout_path": str(paths.stdout_log),
            "stderr_path": str(paths.stderr_log),
            "stdout": _tail(paths.stdout_log, int(tail)),
            "stderr": _tail(paths.stderr_log, int(tail)),
            "tail": int(tail),
        }

    def approvals(self, *, run_id: str | None = None, include_all: bool = False) -> Dict[str, Any]:
        supervisor = self.config.supervisor()
        try:
            approvals = supervisor.state_store.list_approvals(
                run_id=run_id,
                status=None if include_all else "WAITING",
            )
            return {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "approvals": approvals,
                "count": len(approvals),
                "include_all": bool(include_all),
            }
        finally:
            supervisor.state_store.close()

    def approve(self, approval_id: str, *, approved_by: str = "operator") -> Dict[str, Any]:
        supervisor = self.config.supervisor()
        try:
            return supervisor.approve(approval_id, approved_by=approved_by)
        finally:
            supervisor.state_store.close()

    def pause(self, run_id: str, *, reason: str = "operator_pause") -> Dict[str, Any]:
        supervisor = self.config.supervisor()
        try:
            return supervisor.pause_run(run_id, reason)
        finally:
            supervisor.state_store.close()

    def resume(self, run_id: str) -> Dict[str, Any]:
        supervisor = self.config.supervisor()
        try:
            return supervisor.resume_run(run_id)
        finally:
            supervisor.state_store.close()

    def soak(self, *, duration_seconds: float, tick_interval: float, snapshot_interval: float) -> Dict[str, Any]:
        paths = self.config.ensured_paths()
        supervisor = self.config.supervisor()
        watchdog = self.config.watchdog()
        run_id = f"soak-{int(time.time())}"
        snapshot_path = paths.soak_dir / f"{run_id}.jsonl"
        failures: List[str] = []
        tick_count = 0
        next_snapshot_at = 0.0
        deadline = time.time() + max(0.0, float(duration_seconds))
        try:
            supervisor.create_run("Con OS runtime soak probe", run_id=run_id)
            supervisor.add_task(run_id, "supervisor progress probe", priority=2)
            supervisor.add_task(run_id, "approval-safe idle probe", priority=1)
            while time.time() <= deadline:
                payload = tick_runtime_once(
                    supervisor,
                    watchdog=watchdog,
                    snapshot_path=None,
                    max_event_rows=int(self.config.max_event_rows),
                )
                tick_count += 1
                watchdog_status = str(payload.get("watchdog", {}).get("status", ""))
                if watchdog_status == "DEGRADED":
                    failures.append("resource_threshold_exceeded")
                if any(str(item.get("status", "")) == "FAILED" for item in payload.get("ticks", [])):
                    failures.append("runtime_task_failed")
                now = time.time()
                if now >= next_snapshot_at:
                    append_status_snapshot(snapshot_path, payload)
                    next_snapshot_at = now + max(0.1, float(snapshot_interval))
                if failures:
                    break
                sleep_for = max(0.01, float(tick_interval))
                if time.time() + sleep_for > deadline:
                    if deadline > time.time():
                        time.sleep(max(0.0, deadline - time.time()))
                    break
                time.sleep(sleep_for)
            final = supervisor.status(run_id)
            completed = str(final.get("run", {}).get("status", "")) in TERMINAL_STATUSES | {"RUNNING"}
            if not completed:
                failures.append("soak_probe_not_recoverable")
            result = {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "status": "FAILED" if failures else "PASSED",
                "run_id": run_id,
                "duration_seconds": float(duration_seconds),
                "tick_count": tick_count,
                "snapshot_path": str(snapshot_path),
                "failures": sorted(set(failures)),
                "final": final,
            }
            append_status_snapshot(snapshot_path, result)
            return result
        except Exception as exc:
            result = {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "status": "FAILED",
                "run_id": run_id,
                "duration_seconds": float(duration_seconds),
                "tick_count": tick_count,
                "snapshot_path": str(snapshot_path),
                "failures": ["runtime_crashed"],
                "error": str(exc),
            }
            append_status_snapshot(snapshot_path, result)
            return result
        finally:
            supervisor.state_store.close()

    def _start_commands(self) -> List[List[str]]:
        service_target = f"gui/{os.getuid()}/{self.config.label}"
        domain = f"gui/{os.getuid()}"
        return [
            ["launchctl", "bootstrap", domain, str(self.config.launch_agent_path)],
            ["launchctl", "kickstart", "-k", service_target],
        ]

    def _stop_commands(self) -> List[List[str]]:
        service_target = f"gui/{os.getuid()}/{self.config.label}"
        domain = f"gui/{os.getuid()}"
        return [
            ["launchctl", "bootout", service_target],
            ["launchctl", "bootout", domain, str(self.config.launch_agent_path)],
        ]

    def _run_launchctl_commands(self, action: str, commands: List[List[str]]) -> Dict[str, Any]:
        if platform.system() != "Darwin":
            return {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "action": action,
                "status": "UNSUPPORTED_PLATFORM",
                "platform": platform.system(),
                "commands": commands,
            }
        results = []
        ok = True
        for command in commands:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            accepted = completed.returncode == 0 or ("Bootstrap failed: 5" in completed.stderr and action == "start")
            ok = ok and accepted
            results.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        return {"schema_version": RUNTIME_SERVICE_VERSION, "action": action, "status": "OK" if ok else "FAILED", "results": results}

    def _launchd_print(self) -> Dict[str, Any]:
        if platform.system() != "Darwin":
            return {"status": "UNSUPPORTED_PLATFORM", "platform": platform.system()}
        command = ["launchctl", "print", f"gui/{os.getuid()}/{self.config.label}"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "loaded": completed.returncode == 0,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-home", default=None)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-timeout", type=float, default=10.0)
    parser.add_argument("--ollama-required", action="store_true")
    parser.add_argument("--home", default=None, help=argparse.SUPPRESS)


def _config(args: argparse.Namespace) -> RuntimeServiceConfig:
    return RuntimeServiceConfig.from_args(
        runtime_home=getattr(args, "runtime_home", None),
        repo_root=getattr(args, "repo_root", None),
        python_executable=getattr(args, "python", None),
        tick_interval=float(getattr(args, "tick_interval", 5.0)),
        watchdog_interval=float(getattr(args, "watchdog_interval", 30.0)),
        snapshot_interval=float(getattr(args, "snapshot_interval", 60.0)),
        max_event_rows=int(getattr(args, "max_event_rows", 5000)),
        ollama_base_url=getattr(args, "ollama_base_url", None),
        ollama_timeout=float(getattr(args, "ollama_timeout", 10.0)),
        ollama_required=bool(getattr(args, "ollama_required", False)),
        home=getattr(args, "home", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conos")
    subparsers = parser.add_subparsers(dest="command")
    for command in ("install-service", "uninstall-service", "start", "stop", "status"):
        command_parser = subparsers.add_parser(command)
        _add_common(command_parser)
        if command in {"install-service", "start", "stop", "uninstall-service"}:
            command_parser.add_argument("--dry-run", action="store_true")
        if command in {"install-service", "start"}:
            command_parser.add_argument("--tick-interval", type=float, default=5.0)
            command_parser.add_argument("--watchdog-interval", type=float, default=30.0)
            command_parser.add_argument("--snapshot-interval", type=float, default=60.0)
            command_parser.add_argument("--max-event-rows", type=int, default=5000)

    logs_parser = subparsers.add_parser("logs")
    _add_common(logs_parser)
    logs_parser.add_argument("--tail", type=int, default=120)

    approvals_parser = subparsers.add_parser("approvals")
    _add_common(approvals_parser)
    approvals_parser.add_argument("--run-id", default=None)
    approvals_parser.add_argument("--all", action="store_true", dest="include_all")

    approve_parser = subparsers.add_parser("approve")
    _add_common(approve_parser)
    approve_parser.add_argument("approval_id")
    approve_parser.add_argument("--approved-by", default="operator")

    pause_parser = subparsers.add_parser("pause")
    _add_common(pause_parser)
    pause_parser.add_argument("run_id")
    pause_parser.add_argument("--reason", default="operator_pause")

    resume_parser = subparsers.add_parser("resume")
    _add_common(resume_parser)
    resume_parser.add_argument("run_id")

    soak_parser = subparsers.add_parser("soak")
    _add_common(soak_parser)
    soak_parser.add_argument("--duration", default="24h")
    soak_parser.add_argument("--tick-interval", type=float, default=5.0)
    soak_parser.add_argument("--snapshot-interval", type=float, default=60.0)
    soak_parser.add_argument("--max-event-rows", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    service = RuntimeService(_config(args))
    if args.command == "install-service":
        _print(service.install_service(dry_run=bool(args.dry_run)))
        return 0
    if args.command == "uninstall-service":
        _print(service.uninstall_service(dry_run=bool(args.dry_run)))
        return 0
    if args.command == "start":
        _print(service.start_service(dry_run=bool(args.dry_run)))
        return 0
    if args.command == "stop":
        _print(service.stop_service(dry_run=bool(args.dry_run)))
        return 0
    if args.command == "status":
        _print(service.status())
        return 0
    if args.command == "logs":
        _print(service.logs(tail=int(args.tail)))
        return 0
    if args.command == "approvals":
        _print(service.approvals(run_id=args.run_id, include_all=bool(args.include_all)))
        return 0
    if args.command == "approve":
        _print(service.approve(args.approval_id, approved_by=args.approved_by))
        return 0
    if args.command == "pause":
        _print(service.pause(args.run_id, reason=args.reason))
        return 0
    if args.command == "resume":
        _print(service.resume(args.run_id))
        return 0
    if args.command == "soak":
        result = service.soak(
            duration_seconds=parse_duration_seconds(args.duration),
            tick_interval=float(args.tick_interval),
            snapshot_interval=float(args.snapshot_interval),
        )
        _print(result)
        return 0 if result.get("status") == "PASSED" else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

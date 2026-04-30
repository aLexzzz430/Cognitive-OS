from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import importlib.util
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
from core.runtime.product_deployment_gate import product_deployment_gate_report
from core.runtime.operator_console import attach_operator_panel
from core.runtime.recovery_playbook import attach_recovery_diagnosis
from core.runtime.recovery_guidance import (
    attach_guidance_to_checks,
    guidance_for_runtime_status,
    guidance_for_vm_report,
)
from core.runtime.resource_watchdog import ResourceWatchdog, WatchdogThresholds
from core.runtime.runtime_modes import infer_runtime_mode, runtime_mode_catalog
from core.runtime.soak_runner import SOAK_MODES, SoakConfig, SoakRunner, SUPPORTED_PROBE_TYPES
from core.runtime.vm_watchdog import ManagedVMWatchdog, ManagedVMWatchdogConfig
from modules.llm.product_policy import build_llm_product_policy_report, policy_report_brief


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


def _check_payload(name: str, ok: bool, detail: str, *, required: bool = True) -> Dict[str, Any]:
    return {
        "name": str(name),
        "ok": bool(ok),
        "required": bool(required),
        "status": "OK" if ok else ("FAIL" if required else "WARN"),
        "detail": str(detail or ""),
    }


def _runtime_preflight_checks(repo_root: Path, *, strict_dev: bool = False) -> List[Dict[str, Any]]:
    root = Path(repo_root).expanduser().resolve()
    checks: List[Dict[str, Any]] = []
    current = sys.version_info[:3]
    checks.append(
        _check_payload(
            "python_version",
            current >= (3, 10),
            f"current={current[0]}.{current[1]}.{current[2]}, required>=3.10",
        )
    )
    try:
        from core.main_loop import CoreMainLoop  # noqa: F401

        checks.append(_check_payload("core_import", True, "CoreMainLoop imports successfully"))
    except Exception as exc:
        checks.append(_check_payload("core_import", False, f"from core.main_loop import CoreMainLoop failed: {exc!r}"))

    try:
        from core.adapter_registry import find_adapter_registry_violations
        from core.conos_repository_layout import find_forbidden_public_core_imports

        forbidden = find_forbidden_public_core_imports(root)
        adapter = find_adapter_registry_violations()
        problems = []
        if forbidden:
            problems.append(f"forbidden_public_core_imports={len(forbidden)}")
        if adapter:
            problems.append(f"adapter_registry_violations={len(adapter)}")
        checks.append(
            _check_payload(
                "repo_layout",
                not problems,
                ", ".join(problems) if problems else "public/private boundary imports and adapter registry boundaries are clean",
            )
        )
    except Exception as exc:
        checks.append(_check_payload("repo_layout", False, f"layout checker imports failed: {exc!r}"))

    for script_name in ("run_local_machine.py", "conos.py", "local_mirror.py"):
        path = root / "scripts" / script_name
        try:
            source = path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError as exc:
            checks.append(_check_payload(f"entrypoint:{script_name}", False, f"cannot read script: {exc!r}"))
            continue
        checks.append(
            _check_payload(
                f"entrypoint:{script_name}",
                path.exists() and "main" in source,
                "script is present and bridges to a runner main" if path.exists() and "main" in source else f"missing or incomplete {path}",
            )
        )

    pytest_ok = importlib.util.find_spec("pytest") is not None
    checks.append(
        _check_payload(
            "dev_dependency:pytest",
            pytest_ok,
            "pytest is installed" if pytest_ok else "pytest is missing; install requirements-dev.txt before running tests",
            required=bool(strict_dev),
        )
    )
    smoke_path = root / "tests" / "test_public_repo_smoke.py"
    checks.append(
        _check_payload(
            "public_smoke_test",
            smoke_path.exists(),
            "tests/test_public_repo_smoke.py is present" if smoke_path.exists() else "tests/test_public_repo_smoke.py is missing",
            required=bool(strict_dev),
        )
    )
    return checks


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
    zombie_threshold_seconds: float = 600.0
    zombie_fail_seconds: float = 0.0
    ollama_base_url: Optional[str] = None
    ollama_timeout: float = 10.0
    ollama_required: bool = False
    vm_watchdog_enabled: bool = False
    vm_auto_recover: bool = False
    vm_restart_unready: bool = False
    vm_state_root: Optional[Path] = None
    vm_image_id: str = "conos-base"
    vm_instance_id: str = "default"
    vm_helper_path: str = ""
    vm_runner_path: str = ""
    vm_network_mode: str = "provider_default"
    vm_timeout_seconds: int = 120
    vm_startup_wait_seconds: float = 15.0
    vm_guest_wait_seconds: float = 180.0
    vm_auto_build_runner: bool = True

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
        zombie_threshold_seconds: float = 600.0,
        zombie_fail_seconds: float = 0.0,
        ollama_base_url: str | None = None,
        ollama_timeout: float = 10.0,
        ollama_required: bool = False,
        vm_watchdog_enabled: bool = False,
        vm_auto_recover: bool = False,
        vm_restart_unready: bool = False,
        vm_state_root: str | None = None,
        vm_image_id: str = "conos-base",
        vm_instance_id: str = "default",
        vm_helper_path: str = "",
        vm_runner_path: str = "",
        vm_network_mode: str = "provider_default",
        vm_timeout_seconds: int = 120,
        vm_startup_wait_seconds: float = 15.0,
        vm_guest_wait_seconds: float = 180.0,
        vm_auto_build_runner: bool = True,
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
            zombie_threshold_seconds=float(zombie_threshold_seconds),
            zombie_fail_seconds=float(zombie_fail_seconds),
            ollama_base_url=ollama_base_url or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("CONOS_OLLAMA_BASE_URL") or None,
            ollama_timeout=float(ollama_timeout),
            ollama_required=bool(ollama_required),
            vm_watchdog_enabled=bool(vm_watchdog_enabled),
            vm_auto_recover=bool(vm_auto_recover),
            vm_restart_unready=bool(vm_restart_unready),
            vm_state_root=Path(vm_state_root).expanduser() if vm_state_root else None,
            vm_image_id=str(vm_image_id or "conos-base"),
            vm_instance_id=str(vm_instance_id or "default"),
            vm_helper_path=str(vm_helper_path or ""),
            vm_runner_path=str(vm_runner_path or ""),
            vm_network_mode=str(vm_network_mode or "provider_default"),
            vm_timeout_seconds=int(vm_timeout_seconds),
            vm_startup_wait_seconds=float(vm_startup_wait_seconds),
            vm_guest_wait_seconds=float(vm_guest_wait_seconds),
            vm_auto_build_runner=bool(vm_auto_build_runner),
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

    def vm_watchdog(self, *, auto_recover: Optional[bool] = None) -> ManagedVMWatchdog:
        paths = self.ensured_paths()
        return ManagedVMWatchdog(
            ManagedVMWatchdogConfig(
                state_root=str((self.vm_state_root or (paths.runtime_home / "vm")).expanduser()),
                image_id=str(self.vm_image_id or "conos-base"),
                instance_id=str(self.vm_instance_id or "default"),
                helper_path=str(self.vm_helper_path or ""),
                runner_path=str(self.vm_runner_path or ""),
                network_mode=str(self.vm_network_mode or "provider_default"),
                timeout_seconds=int(self.vm_timeout_seconds),
                startup_wait_seconds=float(self.vm_startup_wait_seconds),
                guest_wait_seconds=float(self.vm_guest_wait_seconds),
                auto_build_runner=bool(self.vm_auto_build_runner),
                auto_recover=bool(self.vm_auto_recover if auto_recover is None else auto_recover),
                restart_unready=bool(self.vm_restart_unready),
            )
        )

    def launchd_plist(self) -> Dict[str, Any]:
        paths = self.paths.resolved()
        env = paths.as_env()
        autonomous_state_path = paths.runtime_home / "state" / "state.json"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(self.repo_root)
        env["THE_AGI_STATE_PATH"] = str(autonomous_state_path)
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
            "--zombie-threshold",
            str(float(self.zombie_threshold_seconds)),
            "--zombie-fail-threshold",
            str(float(self.zombie_fail_seconds)),
            "--autonomous-state-path",
            str(autonomous_state_path),
            "--ollama-timeout",
            str(float(self.ollama_timeout)),
        ]
        if self.ollama_base_url:
            program_arguments.extend(["--ollama-base-url", str(self.ollama_base_url)])
        if self.ollama_required:
            program_arguments.append("--ollama-required")
        if self.vm_watchdog_enabled:
            vm_state_root = self.vm_state_root or (paths.runtime_home / "vm")
            program_arguments.extend(
                [
                    "--vm-watchdog",
                    "--vm-state-root",
                    str(vm_state_root.expanduser()),
                    "--vm-image-id",
                    str(self.vm_image_id or "conos-base"),
                    "--vm-instance-id",
                    str(self.vm_instance_id or "default"),
                    "--vm-network-mode",
                    str(self.vm_network_mode or "provider_default"),
                    "--vm-timeout-seconds",
                    str(int(self.vm_timeout_seconds)),
                    "--vm-startup-wait-seconds",
                    str(float(self.vm_startup_wait_seconds)),
                    "--vm-guest-wait-seconds",
                    str(float(self.vm_guest_wait_seconds)),
                ]
            )
            if self.vm_helper_path:
                program_arguments.extend(["--vm-helper-path", str(self.vm_helper_path)])
            if self.vm_runner_path:
                program_arguments.extend(["--vm-runner-path", str(self.vm_runner_path)])
            if self.vm_auto_recover:
                program_arguments.append("--vm-auto-recover")
            if self.vm_restart_unready:
                program_arguments.append("--vm-restart-unready")
            if not self.vm_auto_build_runner:
                program_arguments.append("--vm-no-build-runner")
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

    def setup(
        self,
        *,
        dry_run: bool = False,
        one_click: bool = False,
        include_vm: bool = True,
        execute_vm_setup: bool = False,
        allow_vm_artifact_download: bool = False,
        install_service: Optional[bool] = None,
        start_service: bool = False,
        run_doctor: bool = True,
    ) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        profile_store = paths.runtime_home / "model_profiles.json"
        route_policy_file = paths.runtime_home / "llm_route_policies.json"
        setup_manifest = paths.runtime_home / "setup.json"
        planned_directories = [
            paths.runtime_home,
            paths.runs_root,
            paths.logs_dir,
            paths.snapshots_dir,
            paths.soak_dir,
            paths.state_db.parent,
        ]
        payload = {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "setup",
            "status": "READY",
            "dry_run": bool(dry_run),
            "one_click": bool(one_click),
            "runtime_paths": paths.as_dict(),
            "planned_directories": [str(path) for path in planned_directories],
            "setup_manifest": str(setup_manifest),
            "service": {
                "label": self.config.label,
                "plist_path": str(self.config.launch_agent_path),
                "install_command": ["conos", "install-service"],
                "start_command": ["conos", "start"],
                "keep_alive": True,
            },
            "stable_commands": [
                "conos setup",
                "conos setup --one-click",
                "conos validate-install",
                "conos status",
                "conos logs --tail 120",
                "conos approvals",
                "conos vm setup-plan",
                "conos vm setup-default",
                "conos vm status",
                "conos doctor",
                "conos llm policy --provider all",
            ],
            "llm_policy": policy_report_brief(
                build_llm_product_policy_report(
                    provider="all",
                    base_url=self.config.ollama_base_url or "",
                    timeout_sec=float(self.config.ollama_timeout),
                    store_path=profile_store,
                    route_policy_path=route_policy_file,
                )
            ),
            "llm_profile_store": str(profile_store),
            "llm_route_policy_file": str(route_policy_file),
        }
        if dry_run and not one_click:
            return attach_recovery_diagnosis(attach_operator_panel(payload, surface="setup"), surface="setup")
        if not dry_run:
            self.config.ensured_paths()
            manifest = {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "runtime_paths": paths.as_dict(),
                "llm_profile_store": str(profile_store),
                "llm_route_policy_file": str(route_policy_file),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            setup_manifest.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            payload["written"] = True
        else:
            payload["written"] = False
        if one_click:
            one_click_report = self._one_click_setup_report(
                dry_run=bool(dry_run),
                include_vm=bool(include_vm),
                execute_vm_setup=bool(execute_vm_setup),
                allow_vm_artifact_download=bool(allow_vm_artifact_download),
                install_service=(bool(one_click) if install_service is None else bool(install_service)),
                start_service=bool(start_service),
                run_doctor=bool(run_doctor),
            )
            payload["one_click_report"] = one_click_report
            payload["status"] = str(one_click_report.get("status") or payload["status"])
            payload["operator_summary"] = str(one_click_report.get("operator_summary") or "")
            if not dry_run:
                payload["install_validation"] = self.validate_install(include_vm=bool(include_vm))
        return attach_recovery_diagnosis(attach_operator_panel(payload, surface="setup"), surface="setup")

    def _validation_action(
        self,
        check: str,
        *,
        command: Sequence[str],
        kind: str = "validation",
        reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "check": str(check),
            "kind": str(kind),
            "command": [str(part) for part in command],
            "display": " ".join(str(part) for part in command),
            "reason": str(reason or ""),
        }

    def validate_install(self, *, include_vm: bool = True, product: bool = False) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        setup_manifest = paths.runtime_home / "setup.json"
        required_dirs = [paths.runtime_home, paths.runs_root, paths.logs_dir, paths.snapshots_dir, paths.soak_dir]
        checks: List[Dict[str, Any]] = []
        validation_remaining: List[Dict[str, Any]] = []
        setup_actions: List[Dict[str, Any]] = []

        setup_manifest_ok = setup_manifest.exists()
        checks.append(
            _check_payload(
                "setup_manifest",
                setup_manifest_ok,
                str(setup_manifest) if setup_manifest_ok else f"missing {setup_manifest}",
                required=True,
            )
        )
        if not setup_manifest_ok:
            setup_actions.append(
                self._validation_action(
                    "setup_manifest",
                    command=["conos", "setup", "--one-click"],
                    kind="setup",
                    reason="setup manifest is missing",
                )
            )

        missing_dirs = [str(path) for path in required_dirs if not path.exists()]
        checks.append(
            _check_payload(
                "runtime_directories",
                not missing_dirs,
                "all runtime directories exist" if not missing_dirs else "missing: " + ", ".join(missing_dirs),
                required=True,
            )
        )
        if missing_dirs:
            setup_actions.append(
                self._validation_action(
                    "runtime_directories",
                    command=["conos", "setup", "--one-click"],
                    kind="setup",
                    reason="runtime directories are missing",
                )
            )

        plist_exists = self.config.launch_agent_path.exists()
        checks.append(
            _check_payload(
                "launchd_plist",
                plist_exists,
                str(self.config.launch_agent_path) if plist_exists else f"missing {self.config.launch_agent_path}",
                required=True,
            )
        )
        if not plist_exists:
            setup_actions.append(
                self._validation_action(
                    "launchd_plist",
                    command=["conos", "setup", "--one-click"],
                    kind="setup",
                    reason="launchd service file is missing",
                )
            )

        launchd = self._launchd_print()
        launchd_loaded = bool(launchd.get("loaded"))
        checks.append(
            _check_payload(
                "launchd_loaded",
                launchd_loaded,
                "service is loaded" if launchd_loaded else "service is not loaded yet",
                required=False,
            )
        )
        if not launchd_loaded:
            validation_remaining.append(
                self._validation_action(
                    "launchd_loaded",
                    command=["conos", "start"],
                    reason="start the per-user runtime service and re-run validate-install",
                )
            )

        vm_setup_plan: Dict[str, Any] = {"status": "SKIPPED", "reason": "include_vm=false"}
        if include_vm:
            try:
                from modules.local_mirror.managed_vm import managed_vm_setup_plan

                vm_state_root = self.config.vm_state_root or (paths.runtime_home / "vm")
                vm_setup_plan = managed_vm_setup_plan(
                    state_root=str(vm_state_root),
                    helper_path=str(self.config.vm_helper_path or ""),
                    image_id=str(self.config.vm_image_id or "conos-base"),
                    instance_id=str(self.config.vm_instance_id or "default"),
                )
                vm_ready = bool(vm_setup_plan.get("safe_to_run_tasks"))
                checks.append(
                    _check_payload(
                        "vm_default_boundary",
                        vm_ready,
                        str(vm_setup_plan.get("operator_summary") or vm_setup_plan.get("status") or ""),
                        required=bool(product),
                    )
                )
                if not vm_ready:
                    validation_remaining.append(
                        self._validation_action(
                            "vm_default_boundary",
                            command=[
                                "conos",
                                "setup",
                                "--one-click",
                                "--execute-vm-setup",
                                "--allow-vm-artifact-download",
                            ],
                            reason="execute the built-in VM setup gate and re-run validate-install",
                        )
                    )
            except Exception as exc:
                vm_setup_plan = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                checks.append(
                    _check_payload(
                        "vm_default_boundary",
                        False,
                        str(vm_setup_plan["error"]),
                        required=bool(product),
                    )
                )
                validation_remaining.append(
                    self._validation_action(
                        "vm_default_boundary",
                        command=["conos", "vm", "setup-plan"],
                        reason="inspect VM setup-plan error",
                    )
                )

        llm_policy = build_llm_product_policy_report(
            provider="all",
            base_url=self.config.ollama_base_url or "",
            timeout_sec=float(self.config.ollama_timeout),
            store_path=paths.runtime_home / "model_profiles.json",
            route_policy_path=paths.runtime_home / "llm_route_policies.json",
        )
        llm_contracts_ok = bool(llm_policy.get("runtime_plans"))
        checks.append(
            _check_payload(
                "llm_policy_contracts",
                llm_contracts_ok,
                "provider/auth/runtime/tool/cost/context/verifier contracts available"
                if llm_contracts_ok
                else "LLM route contracts were not generated",
                required=True,
            )
        )
        if not llm_contracts_ok:
            setup_actions.append(
                self._validation_action(
                    "llm_policy_contracts",
                    command=["conos", "llm", "policy", "--provider", "all"],
                    kind="setup",
                    reason="LLM policy contracts are missing",
                )
            )

        product_gate = product_deployment_gate_report(
            checks=checks,
            vm_setup_plan=vm_setup_plan,
            include_vm=bool(include_vm and product),
        )
        required_failed = [row for row in checks if row["required"] and not row["ok"]]
        if required_failed:
            status = "FAILED"
        elif validation_remaining:
            status = "NEEDS_VALIDATION"
        else:
            status = "READY"
        payload = {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "validate-install",
            "status": status,
            "include_vm": bool(include_vm),
            "product": bool(product),
            "product_deployment_gate": product_gate,
            "checks": checks,
            "setup_actions": setup_actions,
            "validation_remaining": validation_remaining,
            "runtime_paths": paths.as_dict(),
            "launchd": launchd,
            "vm_setup_plan": vm_setup_plan,
            "llm_policy": policy_report_brief(llm_policy),
            "operator_summary": (
                str(product_gate.get("operator_summary") or "")
                if product and status == "FAILED"
                else "安装已完成，只剩真实运行验证"
                if status == "NEEDS_VALIDATION"
                else ("安装和验证均已就绪" if status == "READY" else "安装还缺少必需准备项")
            ),
        }
        return attach_recovery_diagnosis(attach_operator_panel(payload, surface="validate-install"), surface="validate-install")

    def _one_click_setup_report(
        self,
        *,
        dry_run: bool,
        include_vm: bool,
        execute_vm_setup: bool,
        allow_vm_artifact_download: bool,
        install_service: bool,
        start_service: bool,
        run_doctor: bool,
    ) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        steps: List[Dict[str, Any]] = []
        report: Dict[str, Any] = {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "one-click-setup",
            "dry_run": bool(dry_run),
            "include_vm": bool(include_vm),
            "install_service": bool(install_service),
            "start_service": bool(start_service),
            "execute_vm_setup": bool(execute_vm_setup),
            "allow_vm_artifact_download": bool(allow_vm_artifact_download),
            "steps": steps,
        }

        steps.append(
            {
                "name": "runtime_home",
                "status": "DRY_RUN" if dry_run else "READY",
                "ok": True,
                "path": str(paths.runtime_home),
            }
        )

        if install_service:
            service_payload = self.install_service(dry_run=bool(dry_run))
            service_ok = bool(dry_run) or bool(service_payload.get("installed"))
            steps.append(
                {
                    "name": "install_service",
                    "status": "DRY_RUN" if dry_run else ("READY" if service_ok else "FAILED"),
                    "ok": service_ok,
                    "plist_path": str(service_payload.get("plist_path") or ""),
                    "payload": service_payload,
                }
            )
            report["service_install"] = service_payload

        if include_vm:
            try:
                from modules.local_mirror.managed_vm import managed_vm_prepare_default_boundary

                vm_state_root = self.config.vm_state_root or (paths.runtime_home / "vm")
                vm_payload = managed_vm_prepare_default_boundary(
                    state_root=str(vm_state_root),
                    helper_path=str(self.config.vm_helper_path or ""),
                    runner_path=str(self.config.vm_runner_path or ""),
                    image_id=str(self.config.vm_image_id or "conos-base"),
                    instance_id=str(self.config.vm_instance_id or "default"),
                    network_mode=str(self.config.vm_network_mode or "provider_default"),
                    execute=bool(execute_vm_setup) and not bool(dry_run),
                    allow_artifact_download=bool(allow_vm_artifact_download),
                    timeout_seconds=int(self.config.vm_timeout_seconds),
                    startup_wait_seconds=float(self.config.vm_startup_wait_seconds),
                    guest_wait_seconds=float(self.config.vm_guest_wait_seconds),
                    verify_agent_exec=bool(execute_vm_setup) and not bool(dry_run),
                    write_audit=not bool(dry_run),
                )
                vm_ready = bool(vm_payload.get("safe_to_run_tasks"))
                steps.append(
                    {
                        "name": "vm_default_boundary",
                        "status": "READY" if vm_ready else str(vm_payload.get("status") or "NEEDS_ACTION"),
                        "ok": vm_ready or not bool(execute_vm_setup),
                        "safe_to_run_tasks": vm_ready,
                        "payload": vm_payload,
                    }
                )
                report["vm_default_boundary"] = vm_payload
            except Exception as exc:
                steps.append(
                    {
                        "name": "vm_default_boundary",
                        "status": "FAILED",
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if start_service:
            start_payload = self.start_service(dry_run=bool(dry_run))
            start_ok = bool(dry_run) or str(start_payload.get("status") or "") == "OK"
            steps.append(
                {
                    "name": "start_service",
                    "status": "DRY_RUN" if dry_run else ("READY" if start_ok else "FAILED"),
                    "ok": start_ok,
                    "payload": start_payload,
                }
            )
            report["service_start"] = start_payload

        if run_doctor and not dry_run:
            doctor_payload = self.doctor(strict_dev=False, live_llm=False)
            doctor_status = str(doctor_payload.get("status") or "")
            steps.append(
                {
                    "name": "doctor",
                    "status": doctor_status,
                    "ok": doctor_status != "FAILED",
                    "payload": doctor_payload,
                }
            )
            report["doctor"] = doctor_payload

        failed_steps = [step for step in steps if not bool(step.get("ok"))]
        action_needed = []
        vm_payload = report.get("vm_default_boundary")
        if isinstance(vm_payload, dict) and not bool(vm_payload.get("safe_to_run_tasks")):
            action_needed.extend(list(vm_payload.get("next_actions") or []))
        report["action_needed"] = action_needed
        if failed_steps:
            status = "FAILED"
        elif action_needed:
            status = "NEEDS_ACTION"
        else:
            status = "READY"
        report["status"] = status
        report["operator_summary"] = (
            "一键安装完成，仍需按 action_needed 完成 VM 默认执行边界"
            if status == "NEEDS_ACTION"
            else ("一键安装完成" if status == "READY" else "一键安装失败，请查看失败步骤")
        )
        return report

    def doctor(self, *, strict_dev: bool = False, live_llm: bool = False) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        checks: List[Dict[str, Any]] = []
        checks.extend(_runtime_preflight_checks(self.config.repo_root, strict_dev=bool(strict_dev)))

        checks.extend(
            [
                _check_payload("runtime_home_parent", paths.runtime_home.parent.exists(), str(paths.runtime_home.parent), required=True),
                _check_payload("runtime_home", paths.runtime_home.exists(), str(paths.runtime_home), required=False),
                _check_payload(
                    "state_db_parent",
                    paths.state_db.parent.exists() or paths.state_db.parent.parent.exists(),
                    str(paths.state_db.parent),
                    required=True,
                ),
            ]
        )
        try:
            status_payload = self.status()
            runtime_status = str(status_payload.get("status", "") or "")
            checks.append(_check_payload("runtime_status_command", runtime_status in {"OK", "DEGRADED"}, runtime_status or "status returned"))
        except Exception as exc:
            status_payload = {"error": f"{type(exc).__name__}: {exc}"}
            checks.append(_check_payload("runtime_status_command", False, status_payload["error"]))

        try:
            from modules.local_mirror.managed_vm import managed_vm_report, managed_vm_setup_plan

            vm_report = managed_vm_report(state_root=str(paths.runtime_home / "vm"))
            vm_setup_plan = managed_vm_setup_plan(state_root=str(paths.runtime_home / "vm"))
            vm_ok = str(vm_report.get("status", "") or "") == "AVAILABLE"
            checks.append(
                _check_payload(
                    "managed_vm_provider",
                    vm_ok,
                    str(vm_report.get("provider_reason", "") or vm_report.get("status", "")),
                    required=False,
                )
            )
            checks.append(
                _check_payload(
                    "managed_vm_execution_boundary",
                    bool(vm_setup_plan.get("safe_to_run_tasks")),
                    str(vm_setup_plan.get("operator_summary") or vm_setup_plan.get("status") or ""),
                    required=False,
                )
            )
        except Exception as exc:
            vm_report = {"error": f"{type(exc).__name__}: {exc}"}
            vm_setup_plan = {"error": f"{type(exc).__name__}: {exc}"}
            checks.append(_check_payload("managed_vm_provider", False, vm_report["error"], required=False))

        llm_policy = build_llm_product_policy_report(
            provider="all",
            base_url=self.config.ollama_base_url or "",
            timeout_sec=float(self.config.ollama_timeout),
            store_path=paths.runtime_home / "model_profiles.json",
            route_policy_path=paths.runtime_home / "llm_route_policies.json",
        )
        checks.append(
            _check_payload(
                "llm_policy_contracts",
                bool(llm_policy.get("runtime_plans")),
                "provider/auth/runtime/tool/cost/context/verifier contracts available",
            )
        )
        if live_llm:
            try:
                from modules.llm.cli import main as llm_main

                # The live probe is intentionally narrow and still isolated behind the provider CLI.
                live_result = llm_main(["--provider", "all", "--timeout", str(self.config.ollama_timeout), "check"])
                checks.append(_check_payload("llm_live_check", live_result == 0, f"exit_code={live_result}", required=False))
            except Exception as exc:
                checks.append(_check_payload("llm_live_check", False, f"{type(exc).__name__}: {exc}", required=False))

        checks = attach_guidance_to_checks(checks)
        required_failed = [row for row in checks if row["required"] and not row["ok"]]
        warnings = [row for row in checks if not row["required"] and not row["ok"]]
        status = "FAILED" if required_failed else ("WARN" if warnings else "OK")
        recovery_guidance = [
            dict(row["operator_guidance"])
            for row in checks
            if isinstance(row.get("operator_guidance"), dict)
        ]
        recovery_guidance.extend(guidance_for_runtime_status(status_payload))
        recovery_guidance.extend(guidance_for_vm_report(vm_report))
        payload = {
            "schema_version": RUNTIME_SERVICE_VERSION,
            "action": "doctor",
            "status": status,
            "strict_dev": bool(strict_dev),
            "live_llm": bool(live_llm),
            "runtime_paths": paths.as_dict(),
            "checks": checks,
            "summary": {
                "required_failed": len(required_failed),
                "warnings": len(warnings),
                "total": len(checks),
            },
            "operator_summary": "正常" if status == "OK" else ("需要处理必需问题" if status == "FAILED" else "可运行但有警告"),
            "recovery_guidance": recovery_guidance,
            "runtime_status": status_payload,
            "vm_status": vm_report,
            "vm_setup_plan": vm_setup_plan,
            "llm_policy": policy_report_brief(llm_policy),
        }
        return attach_recovery_diagnosis(attach_operator_panel(payload, surface="doctor"), surface="doctor")

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
            vm_watchdog = (
                self.config.vm_watchdog(auto_recover=False).evaluate()
                if self.config.vm_watchdog_enabled
                else {"status": "SKIPPED", "reason": "vm_watchdog_not_enabled"}
            )
            if str(vm_watchdog.get("status", "")) == "DEGRADED":
                watchdog = dict(watchdog)
                watchdog["status"] = "DEGRADED"
                reasons = list(watchdog.get("degraded_reasons") or [])
                reasons.append(f"vm:{vm_watchdog.get('reason') or 'managed_vm_unhealthy'}")
                watchdog["degraded_reasons"] = reasons
            runs = supervisor.state_store.list_runs()
            soak_sessions = supervisor.state_store.list_soak_sessions()
            runtime_mode = infer_runtime_mode(
                runs=runs,
                watchdog=watchdog,
                soak_sessions=soak_sessions,
            ).to_dict()
            payload = {
                "schema_version": RUNTIME_SERVICE_VERSION,
                "status": "OK" if watchdog.get("status") != "DEGRADED" else "DEGRADED",
                "runtime_mode": runtime_mode,
                "runtime_mode_catalog": runtime_mode_catalog(),
                "label": self.config.label,
                "plist_path": str(self.config.launch_agent_path),
                "plist_installed": self.config.launch_agent_path.exists(),
                "runtime_paths": paths.as_dict(),
                "metrics": supervisor.metrics(),
                "runs": runs,
                "soak_sessions": soak_sessions,
                "waiting_approvals": supervisor.state_store.list_approvals(status="WAITING"),
                "watchdog": watchdog,
                "vm_watchdog": vm_watchdog,
                "launchd": self._launchd_print(),
            }
            payload["recovery_guidance"] = guidance_for_runtime_status(payload)
            return attach_recovery_diagnosis(attach_operator_panel(payload, surface="status"), surface="status")
        finally:
            supervisor.state_store.close()

    def logs(self, *, tail: int = 120) -> Dict[str, Any]:
        paths = self.config.paths.resolved()
        return attach_recovery_diagnosis(attach_operator_panel({
            "schema_version": RUNTIME_SERVICE_VERSION,
            "stdout_path": str(paths.stdout_log),
            "stderr_path": str(paths.stderr_log),
            "stdout": _tail(paths.stdout_log, int(tail)),
            "stderr": _tail(paths.stderr_log, int(tail)),
            "tail": int(tail),
        }, surface="logs"), surface="logs")

    def approvals(self, *, run_id: str | None = None, include_all: bool = False) -> Dict[str, Any]:
        supervisor = self.config.supervisor()
        try:
            approvals = supervisor.state_store.list_approvals(
                run_id=run_id,
                status=None if include_all else "WAITING",
            )
            return attach_recovery_diagnosis(attach_operator_panel({
                "schema_version": RUNTIME_SERVICE_VERSION,
                "approvals": approvals,
                "count": len(approvals),
                "include_all": bool(include_all),
            }, surface="approvals"), surface="approvals")
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

    def soak(
        self,
        *,
        duration_seconds: float,
        tick_interval: float,
        snapshot_interval: float,
        mode: str = "infrastructure",
        task_interval: float = 300.0,
        zombie_threshold_seconds: float = 300.0,
        zombie_fail_seconds: float = 600.0,
        probe_types: Sequence[str] = (),
        bad_ollama_base_url: str = "http://127.0.0.1:1",
    ) -> Dict[str, Any]:
        paths = self.config.ensured_paths()
        supervisor = self.config.supervisor()
        watchdog = self.config.watchdog()
        try:
            runner = SoakRunner(
                supervisor=supervisor,
                watchdog=watchdog,
                paths=paths,
                config=SoakConfig(
                    duration_seconds=float(duration_seconds),
                    mode=str(mode or "infrastructure"),
                    tick_interval=float(tick_interval),
                    snapshot_interval=float(snapshot_interval),
                    task_interval=float(task_interval),
                    zombie_threshold_seconds=float(zombie_threshold_seconds),
                    zombie_fail_seconds=float(zombie_fail_seconds),
                    max_event_rows=int(self.config.max_event_rows),
                    probe_types=tuple(probe_types or ()),
                    bad_ollama_base_url=str(bad_ollama_base_url or "http://127.0.0.1:1"),
                ),
            )
            return runner.run()
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
        expected_launch_agents = Path.home().expanduser() / "Library" / "LaunchAgents"
        configured_launch_agents = self.config.launch_agent_path.expanduser().parent
        if configured_launch_agents != expected_launch_agents:
            return {
                "status": "PATH_SCOPE_MISMATCH",
                "loaded": False,
                "reason": "configured LaunchAgent path is not the current user's live launchd directory",
                "configured_launch_agent_path": str(self.config.launch_agent_path),
                "expected_launch_agents": str(expected_launch_agents),
            }
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
        zombie_threshold_seconds=float(getattr(args, "zombie_threshold", 600.0)),
        zombie_fail_seconds=float(getattr(args, "zombie_fail_threshold", 0.0)),
        ollama_base_url=getattr(args, "ollama_base_url", None),
        ollama_timeout=float(getattr(args, "ollama_timeout", 10.0)),
        ollama_required=bool(getattr(args, "ollama_required", False)),
        vm_watchdog_enabled=bool(getattr(args, "vm_watchdog", False)),
        vm_auto_recover=bool(getattr(args, "vm_auto_recover", False)),
        vm_restart_unready=bool(getattr(args, "vm_restart_unready", False)),
        vm_state_root=getattr(args, "vm_state_root", None),
        vm_image_id=str(getattr(args, "vm_image_id", "conos-base") or "conos-base"),
        vm_instance_id=str(getattr(args, "vm_instance_id", "default") or "default"),
        vm_helper_path=str(getattr(args, "vm_helper_path", "") or ""),
        vm_runner_path=str(getattr(args, "vm_runner_path", "") or ""),
        vm_network_mode=str(getattr(args, "vm_network_mode", "provider_default") or "provider_default"),
        vm_timeout_seconds=int(getattr(args, "vm_timeout_seconds", 120)),
        vm_startup_wait_seconds=float(getattr(args, "vm_startup_wait_seconds", 15.0)),
        vm_guest_wait_seconds=float(getattr(args, "vm_guest_wait_seconds", 180.0)),
        vm_auto_build_runner=not bool(getattr(args, "vm_no_build_runner", False)),
        home=getattr(args, "home", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conos")
    subparsers = parser.add_subparsers(dest="command")
    for command in ("setup", "install-service", "uninstall-service", "start", "stop", "status", "validate-install"):
        command_parser = subparsers.add_parser(command)
        _add_common(command_parser)
        if command in {"setup", "install-service", "start", "stop", "uninstall-service"}:
            command_parser.add_argument("--dry-run", action="store_true")
        if command == "setup":
            command_parser.add_argument("--one-click", action="store_true")
            command_parser.add_argument("--no-vm", action="store_true")
            command_parser.add_argument("--execute-vm-setup", action="store_true")
            command_parser.add_argument("--allow-vm-artifact-download", action="store_true")
            command_parser.add_argument("--no-install-service", action="store_true")
            command_parser.add_argument("--start-service", action="store_true")
            command_parser.add_argument("--skip-doctor", action="store_true")
        if command in {"install-service", "start"}:
            command_parser.add_argument("--tick-interval", type=float, default=5.0)
            command_parser.add_argument("--watchdog-interval", type=float, default=30.0)
            command_parser.add_argument("--snapshot-interval", type=float, default=60.0)
            command_parser.add_argument("--max-event-rows", type=int, default=5000)
            command_parser.add_argument("--zombie-threshold", type=float, default=600.0)
            command_parser.add_argument("--zombie-fail-threshold", type=float, default=0.0)
        if command == "validate-install":
            command_parser.add_argument("--no-vm", action="store_true")
            command_parser.add_argument("--product", action="store_true")

    doctor_parser = subparsers.add_parser("doctor")
    _add_common(doctor_parser)
    doctor_parser.add_argument("--strict-dev", action="store_true")
    doctor_parser.add_argument("--live-llm", action="store_true")

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
    soak_parser.add_argument("--mode", choices=sorted(SOAK_MODES), default="infrastructure")
    soak_parser.add_argument("--tick-interval", type=float, default=5.0)
    soak_parser.add_argument("--snapshot-interval", type=float, default=60.0)
    soak_parser.add_argument("--task-interval", type=float, default=300.0)
    soak_parser.add_argument("--zombie-threshold", type=float, default=300.0)
    soak_parser.add_argument("--zombie-fail-threshold", type=float, default=600.0)
    soak_parser.add_argument("--max-event-rows", type=int, default=5000)
    soak_parser.add_argument(
        "--probe-types",
        default="",
        help=f"Comma-separated soak probe types. Supported: {', '.join(sorted(SUPPORTED_PROBE_TYPES))}.",
    )
    soak_parser.add_argument("--bad-ollama-base-url", default="http://127.0.0.1:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    service = RuntimeService(_config(args))
    if args.command == "setup":
        _print(
            service.setup(
                dry_run=bool(args.dry_run),
                one_click=bool(getattr(args, "one_click", False)),
                include_vm=not bool(getattr(args, "no_vm", False)),
                execute_vm_setup=bool(getattr(args, "execute_vm_setup", False)),
                allow_vm_artifact_download=bool(getattr(args, "allow_vm_artifact_download", False)),
                install_service=not bool(getattr(args, "no_install_service", False))
                if bool(getattr(args, "one_click", False))
                else None,
                start_service=bool(getattr(args, "start_service", False)),
                run_doctor=not bool(getattr(args, "skip_doctor", False)),
            )
        )
        return 0
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
    if args.command == "validate-install":
        payload = service.validate_install(
            include_vm=not bool(getattr(args, "no_vm", False)),
            product=bool(getattr(args, "product", False)),
        )
        _print(payload)
        return 0 if not bool(getattr(args, "product", False)) or payload.get("status") == "READY" else 1
    if args.command == "doctor":
        payload = service.doctor(strict_dev=bool(args.strict_dev), live_llm=bool(args.live_llm))
        _print(payload)
        return 0 if payload.get("status") != "FAILED" else 1
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
            mode=str(args.mode),
            task_interval=float(args.task_interval),
            zombie_threshold_seconds=float(args.zombie_threshold),
            zombie_fail_seconds=float(args.zombie_fail_threshold),
            probe_types=tuple(item.strip() for item in str(args.probe_types or "").split(",") if item.strip()),
            bad_ollama_base_url=str(args.bad_ollama_base_url),
        )
        _print(result)
        return 0 if result.get("status") == "PASSED" else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

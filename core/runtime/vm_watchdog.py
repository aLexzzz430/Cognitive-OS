from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict


MANAGED_VM_WATCHDOG_VERSION = "conos.managed_vm_watchdog/v1"


@dataclass(frozen=True)
class ManagedVMWatchdogConfig:
    state_root: str
    image_id: str = "conos-base"
    instance_id: str = "default"
    helper_path: str = ""
    runner_path: str = ""
    network_mode: str = "provider_default"
    timeout_seconds: int = 120
    startup_wait_seconds: float = 15.0
    guest_wait_seconds: float = 180.0
    auto_build_runner: bool = True
    auto_recover: bool = False
    restart_unready: bool = False


class ManagedVMWatchdog:
    """Best-effort watchdog for the built-in managed VM execution boundary."""

    def __init__(self, config: ManagedVMWatchdogConfig) -> None:
        self.config = config

    def evaluate(self) -> Dict[str, Any]:
        started_at = time.time()
        try:
            from modules.local_mirror.managed_vm import (
                managed_vm_health_check,
                recover_managed_vm_instance,
            )

            health = managed_vm_health_check(
                state_root=self.config.state_root,
                helper_path=self.config.helper_path,
                image_id=self.config.image_id,
                instance_id=self.config.instance_id,
                network_mode=self.config.network_mode,
                timeout_seconds=min(max(1, int(self.config.timeout_seconds)), 30),
            )
            recovery: Dict[str, Any] = {}
            if not bool(health.get("healthy")) and self.config.auto_recover:
                recovery = recover_managed_vm_instance(
                    state_root=self.config.state_root,
                    helper_path=self.config.helper_path,
                    runner_path=self.config.runner_path,
                    image_id=self.config.image_id,
                    instance_id=self.config.instance_id,
                    network_mode=self.config.network_mode,
                    timeout_seconds=int(self.config.timeout_seconds),
                    startup_wait_seconds=float(self.config.startup_wait_seconds),
                    guest_wait_seconds=float(self.config.guest_wait_seconds),
                    auto_build_runner=bool(self.config.auto_build_runner),
                    restart_unready=bool(self.config.restart_unready),
                )
                final_health = (
                    recovery.get("final_health")
                    if isinstance(recovery.get("final_health"), dict)
                    else health
                )
            else:
                final_health = health

            healthy = bool(final_health.get("healthy"))
            return {
                "schema_version": MANAGED_VM_WATCHDOG_VERSION,
                "status": "OK" if healthy else "DEGRADED",
                "created_at": time.time(),
                "duration_seconds": max(0.0, time.time() - started_at),
                "state_root": self.config.state_root,
                "image_id": self.config.image_id,
                "instance_id": self.config.instance_id,
                "auto_recover": bool(self.config.auto_recover),
                "restart_unready": bool(self.config.restart_unready),
                "health": health,
                "recovery": recovery,
                "final_health": final_health,
                "recovered": bool(recovery.get("recovered", False)),
                "reason": "" if healthy else str(final_health.get("reason") or "managed VM is not healthy"),
            }
        except Exception as exc:
            return {
                "schema_version": MANAGED_VM_WATCHDOG_VERSION,
                "status": "DEGRADED",
                "created_at": time.time(),
                "duration_seconds": max(0.0, time.time() - started_at),
                "state_root": self.config.state_root,
                "image_id": self.config.image_id,
                "instance_id": self.config.instance_id,
                "auto_recover": bool(self.config.auto_recover),
                "restart_unready": bool(self.config.restart_unready),
                "health": {},
                "recovery": {},
                "final_health": {},
                "recovered": False,
                "reason": "vm_watchdog_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }

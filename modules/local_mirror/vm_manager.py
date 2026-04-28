"""Lifecycle manager for real VM-backed local mirror workspaces.

The manager deliberately refuses to emulate a VM with local subprocesses. It
builds lifecycle operations around a configured real provider from
``vm_backend`` and records enough metadata for audit and recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
import shlex
from pathlib import Path
from typing import Any, Dict

from modules.local_mirror.vm_backend import (
    VMBackendError,
    build_vm_invocation,
    run_vm_command,
)


VM_MANAGER_VERSION = "conos.local_mirror.vm_manager/v1"
SUPPORTED_VM_MANAGER_OPERATIONS = frozenset(
    {
        "preflight",
        "prepare",
        "checkpoint",
        "restore",
        "cleanup",
    }
)


@dataclass(frozen=True)
class VMWorkspaceOperationResult:
    schema_version: str
    operation: str
    status: str
    returncode: int
    stdout: str
    stderr: str
    timeout_seconds: int
    provider: str
    provider_command: list[str]
    provider_binary: str
    vm_name: str = ""
    vm_host: str = ""
    vm_workdir: str = "/workspace"
    vm_network_mode: str = "provider_default"
    real_vm_boundary: bool = True
    checkpoint_id: str = ""
    checkpoint_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_checkpoint_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        seed = f"{_now_compact()}:{datetime.now(timezone.utc).timestamp()}"
        value = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return safe[:80] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _quote_abs_path(raw: str) -> str:
    value = str(raw or "").strip() or "/workspace"
    if not value.startswith("/"):
        raise VMBackendError("VM manager paths must be absolute inside the VM")
    return shlex.quote(value.rstrip("/") or "/")


def _checkpoint_path(vm_workdir: str, checkpoint_id: str) -> str:
    workdir = str(vm_workdir or "/workspace").rstrip("/") or "/workspace"
    return f"{workdir}/.conos_vm_checkpoints/{_safe_checkpoint_id(checkpoint_id)}"


def vm_manager_report(
    *,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
) -> Dict[str, Any]:
    """Return a non-executing VM manager readiness report."""

    try:
        invocation = build_vm_invocation(
            ".",
            ["true"],
            provider=vm_provider,
            vm_name=vm_name,
            vm_host=vm_host,
            vm_workdir=vm_workdir,
            network_mode=vm_network_mode,
        )
    except VMBackendError as exc:
        return {
            "schema_version": VM_MANAGER_VERSION,
            "status": "UNAVAILABLE",
            "reason": str(exc),
            "real_vm_boundary": False,
            "vm_provider": str(vm_provider or "auto"),
            "vm_name": str(vm_name or ""),
            "vm_host": str(vm_host or ""),
            "vm_workdir": str(vm_workdir or "/workspace"),
            "vm_network_mode": str(vm_network_mode or "provider_default"),
            "supported_operations": sorted(SUPPORTED_VM_MANAGER_OPERATIONS),
            "requires_real_provider": True,
        }
    return {
        "schema_version": VM_MANAGER_VERSION,
        "status": "AVAILABLE",
        "provider": invocation.provider,
        "provider_binary": invocation.provider_binary,
        "provider_command_preview": list(invocation.redacted_command),
        "real_vm_boundary": True,
        "vm_name": invocation.vm_name,
        "vm_host": invocation.vm_host,
        "vm_workdir": invocation.vm_workdir,
        "vm_network_mode": invocation.network_mode,
        "supported_operations": sorted(SUPPORTED_VM_MANAGER_OPERATIONS),
        "workspace_lifecycle": {
            "prepare": "mkdir/chmod VM workdir",
            "checkpoint": "tar snapshot inside VM-managed checkpoint directory",
            "restore": "restore VM workdir from checkpoint tarball",
            "cleanup": "clear VM workdir while preserving checkpoints",
        },
        "limitations": [
            "depends_on_operator_configured_real_vm",
            "does_not_install_hypervisor",
            "does_not_copy_host_workspace_into_vm_by_itself",
        ],
    }


def build_vm_workspace_command(
    operation: str,
    *,
    vm_workdir: str = "/workspace",
    checkpoint_id: str = "",
) -> tuple[list[str], str, str]:
    """Build a shell command for a bounded VM workspace lifecycle operation."""

    selected = str(operation or "").strip().lower()
    if selected not in SUPPORTED_VM_MANAGER_OPERATIONS:
        supported = ", ".join(sorted(SUPPORTED_VM_MANAGER_OPERATIONS))
        raise VMBackendError(f"unsupported VM manager operation: {selected}; supported operations: {supported}")
    q_workdir = _quote_abs_path(vm_workdir)
    safe_checkpoint_id = _safe_checkpoint_id(checkpoint_id)
    checkpoint_path = _checkpoint_path(vm_workdir, safe_checkpoint_id)
    q_checkpoint_path = shlex.quote(checkpoint_path)
    q_checkpoint_tar = shlex.quote(f"{checkpoint_path}/workspace.tar")

    if selected == "prepare":
        shell = f"mkdir -p {q_workdir} && chmod 700 {q_workdir} && printf 'prepared\\n'"
    elif selected == "preflight":
        shell = f"test -d {q_workdir} && test -w {q_workdir} && printf 'ready\\n'"
    elif selected == "checkpoint":
        shell = (
            f"test -d {q_workdir} && mkdir -p {q_checkpoint_path} "
            f"&& tar -C {q_workdir} --exclude .conos_vm_checkpoints -cf {q_checkpoint_tar} . "
            "&& printf 'checkpointed\\n'"
        )
    elif selected == "restore":
        shell = (
            f"test -f {q_checkpoint_tar} "
            f"&& mkdir -p {q_workdir} "
            f"&& find {q_workdir} -mindepth 1 -maxdepth 1 ! -name .conos_vm_checkpoints -exec rm -rf {{}} + "
            f"&& tar -C {q_workdir} -xf {q_checkpoint_tar} "
            "&& printf 'restored\\n'"
        )
    else:
        shell = (
            f"test -d {q_workdir} "
            f"&& find {q_workdir} -mindepth 1 -maxdepth 1 ! -name .conos_vm_checkpoints -exec rm -rf {{}} + "
            "&& printf 'cleaned\\n'"
        )
    return ["sh", "-lc", shell], safe_checkpoint_id, checkpoint_path


def run_vm_workspace_operation(
    operation: str,
    *,
    timeout_seconds: int = 30,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    checkpoint_id: str = "",
    local_cwd: str | Path | None = None,
) -> VMWorkspaceOperationResult:
    """Run a lifecycle operation inside a configured real VM provider."""

    selected_operation = str(operation or "").strip().lower()
    if selected_operation == "restore" and not str(checkpoint_id or "").strip():
        raise VMBackendError("VM workspace restore requires checkpoint_id")
    command, safe_checkpoint_id, checkpoint_path = build_vm_workspace_command(
        selected_operation,
        vm_workdir=vm_workdir,
        checkpoint_id=checkpoint_id,
    )
    completed = run_vm_command(
        ".",
        command,
        timeout_seconds=max(1, int(timeout_seconds)),
        provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir="/",
        network_mode=vm_network_mode,
        local_cwd=local_cwd,
    )
    return VMWorkspaceOperationResult(
        schema_version=VM_MANAGER_VERSION,
        operation=selected_operation,
        status="COMPLETED" if int(completed.returncode) == 0 else "FAILED",
        returncode=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        timeout_seconds=max(1, int(timeout_seconds)),
        provider=str(completed.provider),
        provider_command=list(completed.provider_command),
        provider_binary=str(completed.provider_binary),
        vm_name=str(completed.vm_name),
        vm_host=str(completed.vm_host),
        vm_workdir=str(vm_workdir or "/workspace"),
        vm_network_mode=str(completed.network_mode),
        real_vm_boundary=bool(completed.real_vm_boundary),
        checkpoint_id=safe_checkpoint_id if selected_operation in {"checkpoint", "restore"} else "",
        checkpoint_path=checkpoint_path if selected_operation in {"checkpoint", "restore"} else "",
    )

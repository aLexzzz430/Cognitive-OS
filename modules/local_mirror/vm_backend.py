"""External VM execution support for local mirrors.

This module intentionally does not emulate a VM with a local subprocess. A VM
backend is considered available only when a real provider is callable. Providers
include the Con OS managed VM helper plus advanced Lima and SSH bridges.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Mapping, Sequence

from modules.local_mirror.managed_vm import (
    managed_vm_config,
    managed_vm_guest_agent_gate,
    managed_vm_helper_path,
    managed_vm_image_id,
    managed_vm_instance_id,
    managed_vm_report,
    managed_vm_state_root,
)

VM_BACKEND_VERSION = "conos.local_mirror.vm_backend/v1"
SUPPORTED_VM_PROVIDERS = frozenset({"auto", "managed", "managed-vm", "lima", "ssh"})
SUPPORTED_VM_NETWORK_MODES = frozenset({"provider_default", "configured_isolated"})


class VMBackendError(ValueError):
    """Raised when a real VM provider cannot be selected or executed."""


@dataclass(frozen=True)
class VMCommandInvocation:
    provider: str
    actual_command: list[str]
    redacted_command: list[str]
    vm_name: str = ""
    vm_host: str = ""
    vm_workdir: str = "/workspace"
    network_mode: str = "provider_default"
    provider_binary: str = ""
    real_vm_boundary: bool = True


@dataclass(frozen=True)
class VMCommandCompleted:
    provider: str
    command: list[str]
    provider_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timeout_seconds: int
    vm_name: str = ""
    vm_host: str = ""
    vm_workdir: str = "/workspace"
    network_mode: str = "provider_default"
    provider_binary: str = ""
    real_vm_boundary: bool = True


def _clean(value: object) -> str:
    return str(value or "").strip()


def _configured_value(explicit: str, *env_names: str) -> str:
    if explicit:
        return explicit
    for name in env_names:
        value = _clean(os.environ.get(name))
        if value:
            return value
    return ""


def _validate_provider(provider: str) -> str:
    selected = _clean(provider).lower() or "auto"
    if selected == "managed-vm":
        selected = "managed"
    if selected not in SUPPORTED_VM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_VM_PROVIDERS))
        raise VMBackendError(f"unsupported VM provider: {selected}; supported providers: {supported}")
    return selected


def _validate_workdir(vm_workdir: str) -> str:
    workdir = _clean(vm_workdir) or "/workspace"
    if not workdir.startswith("/"):
        raise VMBackendError("VM workdir must be an absolute path inside the VM")
    return workdir


def _validate_network_mode(network_mode: str) -> str:
    selected = _clean(network_mode).lower() or "provider_default"
    if selected not in SUPPORTED_VM_NETWORK_MODES:
        supported = ", ".join(sorted(SUPPORTED_VM_NETWORK_MODES))
        raise VMBackendError(f"unsupported VM network mode: {selected}; supported modes: {supported}")
    return selected


def _shell_command(
    command: Sequence[str],
    *,
    vm_workdir: str,
    extra_env: Mapping[str, str] | None = None,
    redact_env: bool = False,
) -> str:
    cmd = [str(part) for part in command]
    env_overlay = {str(key): str(value) for key, value in dict(extra_env or {}).items()}
    if env_overlay:
        env_args = [
            f"{key}={'<redacted>' if redact_env else value}"
            for key, value in sorted(env_overlay.items())
        ]
        executable = shlex.join(["env", *env_args, *cmd])
    else:
        executable = shlex.join(cmd)
    return f"cd {shlex.quote(vm_workdir)} && exec {executable}"


def build_vm_invocation(
    mirror_workspace: str | Path,
    command: Sequence[str],
    *,
    provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    network_mode: str = "provider_default",
    extra_env: Mapping[str, str] | None = None,
) -> VMCommandInvocation:
    """Build a command that executes inside a configured real VM provider."""

    _ = Path(mirror_workspace)
    selected_provider = _validate_provider(provider)
    workdir = _validate_workdir(vm_workdir)
    selected_network_mode = _validate_network_mode(network_mode)
    configured_name = _configured_value(vm_name, "CONOS_VM_NAME", "CONOS_LIMA_INSTANCE")
    configured_host = _configured_value(vm_host, "CONOS_VM_SSH_HOST")

    if selected_provider == "auto":
        if managed_vm_helper_path():
            selected_provider = "managed"
        elif configured_name and shutil.which("limactl"):
            selected_provider = "lima"
        elif configured_host:
            selected_provider = "ssh"
        else:
            raise VMBackendError(
                "vm backend requires a real VM provider; install the Con OS managed VM helper, "
                "or configure --vm-provider lima/ssh for advanced external VM use"
            )

    if selected_provider == "managed":
        report = managed_vm_report()
        if str(report.get("status", "") or "") != "AVAILABLE":
            raise VMBackendError(str(report.get("reason") or "managed VM provider is unavailable"))
        if not str(report.get("helper_path", "") or ""):
            raise VMBackendError("managed VM guest-agent exec requires the conos-managed-vm helper")
        config = managed_vm_config(
            state_root=managed_vm_state_root(),
            image_id=managed_vm_image_id(),
            instance_id=managed_vm_instance_id(_configured_value(vm_name, "CONOS_MANAGED_VM_INSTANCE_ID")),
        )
        gate = managed_vm_guest_agent_gate(
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
        )
        if not bool(gate.get("ready")):
            reason = str(gate.get("reason") or "guest agent not ready")
            raise VMBackendError(f"managed VM guest agent is not ready: {reason}")
        actual_shell = _shell_command(command, vm_workdir=workdir, extra_env=extra_env)
        redacted_shell = _shell_command(command, vm_workdir=workdir, extra_env=extra_env, redact_env=True)
        actual_command = [
            config.helper_path,
            "agent-exec",
            "--state-root",
            config.state_root,
            "--instance-id",
            config.instance_id,
            "--image-id",
            config.image_id,
            "--network-mode",
            selected_network_mode,
            "--",
            "bash",
            "-lc",
            actual_shell,
        ]
        redacted_command = [*actual_command[:-1], redacted_shell]
        return VMCommandInvocation(
            provider="managed",
            actual_command=actual_command,
            redacted_command=redacted_command,
            vm_name=config.instance_id,
            vm_workdir=workdir,
            network_mode=selected_network_mode,
            provider_binary=config.helper_path,
        )

    if selected_provider == "lima":
        limactl = shutil.which("limactl")
        if not limactl:
            raise VMBackendError("lima VM provider requested but limactl executable was not found")
        if not configured_name:
            raise VMBackendError("lima VM provider requires --vm-name or CONOS_VM_NAME")
        actual_shell = _shell_command(command, vm_workdir=workdir, extra_env=extra_env)
        redacted_shell = _shell_command(
            command,
            vm_workdir=workdir,
            extra_env=extra_env,
            redact_env=True,
        )
        return VMCommandInvocation(
            provider="lima",
            actual_command=[limactl, "shell", configured_name, "bash", "-lc", actual_shell],
            redacted_command=[limactl, "shell", configured_name, "bash", "-lc", redacted_shell],
            vm_name=configured_name,
            vm_workdir=workdir,
            network_mode=selected_network_mode,
            provider_binary=limactl,
        )

    if selected_provider == "ssh":
        ssh = shutil.which("ssh")
        if not ssh:
            raise VMBackendError("ssh VM provider requested but ssh executable was not found")
        if not configured_host:
            raise VMBackendError("ssh VM provider requires --vm-host or CONOS_VM_SSH_HOST")
        actual_shell = _shell_command(command, vm_workdir=workdir, extra_env=extra_env)
        redacted_shell = _shell_command(
            command,
            vm_workdir=workdir,
            extra_env=extra_env,
            redact_env=True,
        )
        return VMCommandInvocation(
            provider="ssh",
            actual_command=[ssh, configured_host, "bash", "-lc", actual_shell],
            redacted_command=[ssh, configured_host, "bash", "-lc", redacted_shell],
            vm_host=configured_host,
            vm_workdir=workdir,
            network_mode=selected_network_mode,
            provider_binary=ssh,
        )

    raise VMBackendError(f"unsupported VM provider: {selected_provider}")


def run_vm_command(
    mirror_workspace: str | Path,
    command: Sequence[str],
    *,
    timeout_seconds: int,
    provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    network_mode: str = "provider_default",
    extra_env: Mapping[str, str] | None = None,
    local_cwd: str | Path | None = None,
) -> VMCommandCompleted:
    invocation = build_vm_invocation(
        mirror_workspace,
        command,
        provider=provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        network_mode=network_mode,
        extra_env=extra_env,
    )
    timeout = max(1, int(timeout_seconds))
    try:
        completed = subprocess.run(
            invocation.actual_command,
            cwd=Path(local_cwd) if local_cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return VMCommandCompleted(
            provider=invocation.provider,
            command=[str(part) for part in command],
            provider_command=invocation.redacted_command,
            returncode=int(completed.returncode),
            stdout=str(completed.stdout or ""),
            stderr=str(completed.stderr or ""),
            timeout_seconds=timeout,
            vm_name=invocation.vm_name,
            vm_host=invocation.vm_host,
            vm_workdir=invocation.vm_workdir,
            network_mode=invocation.network_mode,
            provider_binary=invocation.provider_binary,
            real_vm_boundary=invocation.real_vm_boundary,
        )
    except subprocess.TimeoutExpired as exc:
        return VMCommandCompleted(
            provider=invocation.provider,
            command=[str(part) for part in command],
            provider_command=invocation.redacted_command,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "") + "\nvm mirror command timed out",
            timeout_seconds=timeout,
            vm_name=invocation.vm_name,
            vm_host=invocation.vm_host,
            vm_workdir=invocation.vm_workdir,
            network_mode=invocation.network_mode,
            provider_binary=invocation.provider_binary,
            real_vm_boundary=invocation.real_vm_boundary,
        )

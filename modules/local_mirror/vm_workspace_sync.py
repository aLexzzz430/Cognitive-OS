"""Workspace synchronization for real VM-backed local mirrors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import shlex
import subprocess
import tarfile
from typing import Any, Dict

from modules.local_mirror.vm_backend import VMBackendError, build_vm_invocation


VM_WORKSPACE_SYNC_VERSION = "conos.local_mirror.vm_workspace_sync/v1"
SUPPORTED_VM_SYNC_DIRECTIONS = frozenset({"push", "pull"})
SUPPORTED_VM_SYNC_MODES = frozenset({"none", "push", "pull", "push-pull"})


@dataclass(frozen=True)
class VMWorkspaceSyncResult:
    schema_version: str
    direction: str
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
    file_count: int = 0
    byte_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_vm_sync_mode(mode: str) -> str:
    selected = str(mode or "none").strip().lower() or "none"
    if selected not in SUPPORTED_VM_SYNC_MODES:
        supported = ", ".join(sorted(SUPPORTED_VM_SYNC_MODES))
        raise VMBackendError(f"unsupported VM workspace sync mode: {selected}; supported modes: {supported}")
    return selected


def _quote_abs_path(raw: str) -> str:
    value = str(raw or "").strip() or "/workspace"
    if not value.startswith("/"):
        raise VMBackendError("VM sync workdir must be an absolute path inside the VM")
    return shlex.quote(value.rstrip("/") or "/")


def _workspace_files(workspace_root: Path) -> list[Path]:
    if not workspace_root.exists():
        return []
    return sorted(path for path in workspace_root.rglob("*") if path.is_file())


def _build_tar_bytes(workspace_root: Path) -> tuple[bytes, int]:
    root = workspace_root.resolve()
    buffer = BytesIO()
    file_count = 0
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in _workspace_files(root):
            relative = path.resolve().relative_to(root).as_posix()
            if ".conos_vm_checkpoints" in Path(relative).parts:
                continue
            archive.add(path, arcname=relative, recursive=False)
            file_count += 1
    return buffer.getvalue(), file_count


def _validate_tar_member(member: tarfile.TarInfo) -> Path:
    name = str(member.name or "")
    path = Path(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VMBackendError(f"unsafe path in VM workspace tar: {name}")
    if member.issym() or member.islnk() or member.isdev():
        raise VMBackendError(f"unsafe tar member type in VM workspace tar: {name}")
    return path


def _clear_workspace(workspace_root: Path) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    for child in workspace_root.iterdir():
        if child.name == ".conos_vm_checkpoints":
            continue
        if child.is_dir() and not child.is_symlink():
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_dir() and not nested.is_symlink():
                    nested.rmdir()
                else:
                    nested.unlink()
            child.rmdir()
        else:
            child.unlink()


def _extract_tar_bytes(tar_bytes: bytes, workspace_root: Path) -> int:
    root = workspace_root.resolve()
    buffer = BytesIO(tar_bytes)
    file_count = 0
    _clear_workspace(root)
    with tarfile.open(fileobj=buffer, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = _validate_tar_member(member)
            destination = (root / relative).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise VMBackendError(f"unsafe extraction path in VM workspace tar: {member.name}") from exc
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise VMBackendError(f"unsupported tar member in VM workspace tar: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise VMBackendError(f"missing file body in VM workspace tar: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(extracted.read())
            file_count += 1
    return file_count


def _remote_push_command(vm_workdir: str) -> list[str]:
    q_workdir = _quote_abs_path(vm_workdir)
    return [
        "sh",
        "-lc",
        (
            f"mkdir -p {q_workdir} "
            f"&& find {q_workdir} -mindepth 1 -maxdepth 1 ! -name .conos_vm_checkpoints -exec rm -rf {{}} + "
            f"&& tar -C {q_workdir} -xf - "
            "&& printf 'workspace pushed\\n'"
        ),
    ]


def _remote_pull_command(vm_workdir: str) -> list[str]:
    q_workdir = _quote_abs_path(vm_workdir)
    return [
        "sh",
        "-lc",
        f"test -d {q_workdir} && tar -C {q_workdir} --exclude .conos_vm_checkpoints -cf - .",
    ]


def _run_provider_with_bytes(
    *,
    command: list[str],
    input_bytes: bytes | None,
    timeout_seconds: int,
    vm_provider: str,
    vm_name: str,
    vm_host: str,
    vm_workdir: str,
    vm_network_mode: str,
    local_cwd: str | Path | None,
) -> tuple[Any, Any]:
    invocation = build_vm_invocation(
        ".",
        command,
        provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir="/",
        network_mode=vm_network_mode,
    )
    try:
        completed = subprocess.run(
            invocation.actual_command,
            cwd=Path(local_cwd) if local_cwd is not None else None,
            input=input_bytes,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            args=invocation.actual_command,
            returncode=124,
            stdout=exc.stdout or b"",
            stderr=(exc.stderr or b"") + b"\nvm workspace sync timed out",
        )
    return invocation, completed


def _to_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", errors="replace")


def push_workspace_to_vm(
    workspace_root: str | Path,
    *,
    timeout_seconds: int = 30,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    local_cwd: str | Path | None = None,
) -> VMWorkspaceSyncResult:
    tar_bytes, file_count = _build_tar_bytes(Path(workspace_root))
    invocation, completed = _run_provider_with_bytes(
        command=_remote_push_command(vm_workdir),
        input_bytes=tar_bytes,
        timeout_seconds=timeout_seconds,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
        local_cwd=local_cwd,
    )
    return VMWorkspaceSyncResult(
        schema_version=VM_WORKSPACE_SYNC_VERSION,
        direction="push",
        status="COMPLETED" if int(completed.returncode) == 0 else "FAILED",
        returncode=int(completed.returncode),
        stdout=_to_bytes(completed.stdout).decode("utf-8", errors="replace"),
        stderr=_to_bytes(completed.stderr).decode("utf-8", errors="replace"),
        timeout_seconds=max(1, int(timeout_seconds)),
        provider=str(invocation.provider),
        provider_command=list(invocation.redacted_command),
        provider_binary=str(invocation.provider_binary),
        vm_name=str(invocation.vm_name),
        vm_host=str(invocation.vm_host),
        vm_workdir=str(vm_workdir or "/workspace"),
        vm_network_mode=str(invocation.network_mode),
        real_vm_boundary=bool(invocation.real_vm_boundary),
        file_count=file_count,
        byte_count=len(tar_bytes),
    )


def pull_workspace_from_vm(
    workspace_root: str | Path,
    *,
    timeout_seconds: int = 30,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    local_cwd: str | Path | None = None,
) -> VMWorkspaceSyncResult:
    invocation, completed = _run_provider_with_bytes(
        command=_remote_pull_command(vm_workdir),
        input_bytes=None,
        timeout_seconds=timeout_seconds,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
        local_cwd=local_cwd,
    )
    file_count = 0
    stdout_bytes = _to_bytes(completed.stdout)
    stderr = _to_bytes(completed.stderr).decode("utf-8", errors="replace")
    if int(completed.returncode) == 0:
        file_count = _extract_tar_bytes(stdout_bytes, Path(workspace_root))
        stdout_text = f"workspace pulled at {_now()}"
    else:
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    return VMWorkspaceSyncResult(
        schema_version=VM_WORKSPACE_SYNC_VERSION,
        direction="pull",
        status="COMPLETED" if int(completed.returncode) == 0 else "FAILED",
        returncode=int(completed.returncode),
        stdout=stdout_text,
        stderr=stderr,
        timeout_seconds=max(1, int(timeout_seconds)),
        provider=str(invocation.provider),
        provider_command=list(invocation.redacted_command),
        provider_binary=str(invocation.provider_binary),
        vm_name=str(invocation.vm_name),
        vm_host=str(invocation.vm_host),
        vm_workdir=str(vm_workdir or "/workspace"),
        vm_network_mode=str(invocation.network_mode),
        real_vm_boundary=bool(invocation.real_vm_boundary),
        file_count=file_count,
        byte_count=len(stdout_bytes),
    )

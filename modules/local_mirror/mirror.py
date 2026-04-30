from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

from modules.local_mirror.vm_backend import VMBackendError, build_vm_invocation, run_vm_command
from modules.local_mirror.managed_vm import managed_vm_report
from modules.local_mirror.vm_manager import (
    VM_MANAGER_VERSION,
    run_vm_workspace_operation,
    vm_manager_report,
)
from modules.local_mirror.vm_workspace_sync import (
    SUPPORTED_VM_SYNC_MODES,
    VM_WORKSPACE_SYNC_VERSION,
    normalize_vm_sync_mode,
    pull_workspace_from_vm,
    push_workspace_to_vm,
)


LOCAL_MIRROR_VERSION = "conos.local_mirror/v1"
LOCAL_MIRROR_SYNC_PLAN_VERSION = "conos.local_mirror_sync_plan/v1"
LOCAL_MIRROR_CHECKPOINT_VERSION = "conos.local_mirror_checkpoint/v1"
CONTROL_DIR_NAME = "control"
WORKSPACE_DIR_NAME = "workspace"
CHECKPOINT_DIR_NAME = "checkpoints"
DEFAULT_ALLOWED_COMMANDS = frozenset({"python", "python3"})
SUPPORTED_EXEC_BACKENDS = frozenset({"local", "docker", "vm", "managed-vm"})
DEFAULT_EXECUTION_BACKEND = "managed-vm"
EXECUTION_BOUNDARY_VERSION = "conos.local_mirror.execution_boundary/v1"
SANITIZED_PROCESS_ENV_VERSION = "conos.local_mirror.sanitized_process_env/v1"
SANITIZED_PROCESS_BASE_ENV = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
    "PYTHONUTF8": "1",
}
SENSITIVE_ENV_KEY_TOKENS = frozenset(
    {
        "api_key",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "oauth",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
MACHINE_APPROVABLE_SUFFIXES = frozenset(
    {
        ".cfg",
        ".css",
        ".html",
        ".ini",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
GENERATED_ARTIFACT_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
    }
)
GENERATED_ARTIFACT_SUFFIXES = frozenset({".pyc", ".pyo"})
MATERIALIZE_DIRECTORY_EXCLUDE_DIRS = GENERATED_ARTIFACT_DIRS | frozenset(
    {
        ".conos_vm_checkpoints",
        ".eggs",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
MAX_DIRECTORY_MATERIALIZATION_FILES = 1000
MAX_DIRECTORY_MATERIALIZATION_BYTES = 64 * 1024 * 1024


class MirrorScopeError(ValueError):
    """Raised when a requested source path is outside the declared mirror scope."""


def is_generated_mirror_artifact(relative_path: str | Path) -> bool:
    relative = Path(str(relative_path or ""))
    parts = set(relative.parts)
    if parts.intersection(GENERATED_ARTIFACT_DIRS):
        return True
    return relative.suffix.lower() in GENERATED_ARTIFACT_SUFFIXES


@dataclass(frozen=True)
class MaterializedFile:
    relative_path: str
    source_path: str
    mirror_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MirrorCommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timeout_seconds: int
    backend: str = DEFAULT_EXECUTION_BACKEND
    docker_image: str = ""
    provider_command: list[str] = field(default_factory=list)
    vm_provider: str = ""
    vm_name: str = ""
    vm_host: str = ""
    vm_workdir: str = ""
    vm_network_mode: str = ""
    vm_sync_mode: str = "none"
    vm_workspace_sync: list[Dict[str, Any]] = field(default_factory=list)
    real_vm_boundary: bool = False
    security_boundary: str = ""
    execution_boundary: Dict[str, Any] = field(default_factory=dict)
    env_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MirrorDiffEntry:
    relative_path: str
    status: str
    source_path: str
    mirror_path: str
    source_sha256: str
    mirror_sha256: str
    size_bytes: int
    text_patch: str = ""
    patch_sha256: str = ""
    source_endswith_newline: bool = False
    mirror_endswith_newline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalMirror:
    source_root: Path
    mirror_root: Path
    workspace_root: Path
    control_root: Path
    materialized_files: Dict[str, MaterializedFile] = field(default_factory=dict)
    external_baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    audit_events: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.control_root / "manifest.json"

    @property
    def sync_plan_path(self) -> Path:
        return self.control_root / "sync_plan.json"

    @property
    def checkpoint_root(self) -> Path:
        return self.control_root / CHECKPOINT_DIR_NAME

    def workspace_files(self) -> list[Path]:
        if not self.workspace_root.exists():
            return []
        files: list[Path] = []
        for path in self.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.workspace_root)
            if is_generated_mirror_artifact(relative):
                continue
            files.append(path)
        return sorted(files)

    def workspace_is_empty(self) -> bool:
        return not self.workspace_files()

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": LOCAL_MIRROR_VERSION,
            "source_root": str(self.source_root),
            "mirror_root": str(self.mirror_root),
            "workspace_root": str(self.workspace_root),
            "control_root": str(self.control_root),
            "workspace_initial_state": "empty",
            "workspace_file_count": len(self.workspace_files()),
            "materialized_files": [
                item.to_dict()
                for _, item in sorted(self.materialized_files.items())
            ],
            "external_baselines": [
                dict(item)
                for _, item in sorted(self.external_baselines.items())
            ],
            "audit_events": list(self.audit_events),
        }

    def save_manifest(self) -> None:
        self.control_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.to_manifest(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(event_type: str, **payload: Any) -> Dict[str, Any]:
    return {
        "event_type": str(event_type),
        "timestamp": _now(),
        "payload": dict(payload),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_env_key(key: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key or "")))


def sensitive_env_keys(env: Mapping[str, Any] | None) -> list[str]:
    keys: list[str] = []
    for key in dict(env or {}):
        key_text = str(key or "")
        lowered = key_text.lower()
        if any(token in lowered for token in SENSITIVE_ENV_KEY_TOKENS):
            keys.append(key_text)
    return sorted(keys)


def _sanitize_explicit_env(extra_env: Mapping[str, str] | None) -> tuple[Dict[str, str], Dict[str, Any]]:
    explicit: Dict[str, str] = {}
    invalid_keys: list[str] = []
    for key, value in dict(extra_env or {}).items():
        key_text = str(key or "")
        if not _valid_env_key(key_text):
            invalid_keys.append(key_text)
            continue
        explicit[key_text] = str(value)
    sensitive_keys = sensitive_env_keys(explicit)
    audit = {
        "schema_version": SANITIZED_PROCESS_ENV_VERSION,
        "host_env_passthrough": False,
        "host_env_forwarded": False,
        "sanitized_base_env_keys": sorted(SANITIZED_PROCESS_BASE_ENV),
        "explicit_env_keys": sorted(explicit),
        "sensitive_explicit_env_keys": sensitive_keys,
        "invalid_env_keys_ignored": sorted(invalid_keys),
        "value_hashes": {key: _hash_text(value) for key, value in sorted(explicit.items())},
        "values_redacted_in_audit": True,
    }
    return explicit, audit


def _sanitized_subprocess_env(explicit_env: Mapping[str, str]) -> Dict[str, str]:
    env = dict(SANITIZED_PROCESS_BASE_ENV)
    env.update({str(key): str(value) for key, value in dict(explicit_env or {}).items()})
    return env


def _redact_explicit_env_values(text: str, explicit_env: Mapping[str, str]) -> str:
    redacted = str(text or "")
    for key, value in sorted(dict(explicit_env or {}).items()):
        value_text = str(value)
        if value_text:
            redacted = redacted.replace(value_text, f"<redacted:{key}>")
    return redacted


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _load_manifest(mirror_root: Path) -> Dict[str, Any]:
    manifest_path = mirror_root / CONTROL_DIR_NAME / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _load_sync_plan(mirror_root: Path) -> Dict[str, Any]:
    plan_path = mirror_root / CONTROL_DIR_NAME / "sync_plan.json"
    if not plan_path.exists():
        return {}
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def execution_boundary_report(
    *,
    backend: str = DEFAULT_EXECUTION_BACKEND,
    docker_image: str = "python:3.10-slim",
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
) -> Dict[str, Any]:
    selected_backend = str(backend or "local").strip().lower()
    if selected_backend not in SUPPORTED_EXEC_BACKENDS:
        return {
            "schema_version": EXECUTION_BOUNDARY_VERSION,
            "status": "UNSUPPORTED",
            "backend": selected_backend,
            "reason": f"unsupported mirror execution backend: {selected_backend}",
            "supported_backends": sorted(SUPPORTED_EXEC_BACKENDS),
            "real_vm_boundary": False,
        }
    if selected_backend == "local":
        return {
            "schema_version": EXECUTION_BOUNDARY_VERSION,
            "status": "AVAILABLE",
            "backend": "local",
            "security_boundary": "best_effort_local_process",
            "real_vm_boundary": False,
            "filesystem_boundary": "mirror_workspace_only_best_effort",
            "network_boundary": "host_default",
            "network_policy": "host_default_best_effort_policy_required_for_internet_tools",
            "credential_boundary": "sanitized_process_env_explicit_env_only_redacted_in_audit",
            "host_env_forwarded": False,
            "host_env_passthrough": False,
            "sanitized_base_env_keys": sorted(SANITIZED_PROCESS_BASE_ENV),
            "sync_boundary": "diff_patch_gate_with_source_hash_and_rollback",
            "limitations": [
                "not_a_vm",
                "shares_host_kernel_and_process_privileges",
                "network_not_isolated",
                "development_only_host_execution",
            ],
        }
    if selected_backend == "docker":
        docker_binary = shutil.which("docker")
        return {
            "schema_version": EXECUTION_BOUNDARY_VERSION,
            "status": "AVAILABLE" if docker_binary else "UNAVAILABLE",
            "backend": "docker",
            "security_boundary": "container_best_effort",
            "real_vm_boundary": False,
            "docker_binary": docker_binary or "",
            "docker_image": str(docker_image or "python:3.10-slim"),
            "filesystem_boundary": "workspace_volume_mount",
            "network_boundary": "none",
            "network_policy": "disabled_by_default",
            "credential_boundary": "explicit_env_only",
            "host_env_forwarded_to_container": False,
            "sync_boundary": "diff_patch_gate_with_source_hash_and_rollback",
            "limitations": [
                "container_not_full_vm",
                "host_docker_daemon_trust_required",
            ],
        }
    if selected_backend == "managed-vm":
        report = managed_vm_report(
            instance_id=str(vm_name or ""),
        )
        available = str(report.get("status", "") or "") == "AVAILABLE"
        return {
            "schema_version": EXECUTION_BOUNDARY_VERSION,
            "status": "AVAILABLE" if available else "UNAVAILABLE",
            "backend": "managed-vm",
            "security_boundary": "conos_managed_vm_provider",
            "real_vm_boundary": bool(report.get("real_vm_boundary", False)),
            "provider": "managed",
            "provider_binary": str(report.get("helper_path", "") or ""),
            "provider_runner": str(report.get("virtualization_runner_path", "") or ""),
            "vm_name": str(report.get("instance_id", "") or ""),
            "vm_host": "",
            "vm_workdir": str(vm_workdir or "/workspace"),
            "vm_network_mode": str(vm_network_mode or "provider_default"),
            "filesystem_boundary": "managed_vm_workspace_via_explicit_sync",
            "network_boundary": str(vm_network_mode or "provider_default"),
            "network_policy": "provider_controlled_explicit_mode",
            "credential_boundary": "vm_guest_isolated_explicit_env_only_redacted_in_audit",
            "host_env_forwarded_to_guest": False,
            "sync_boundary": "diff_patch_gate_with_source_hash_and_rollback",
            "managed_vm": report,
            "reason": str(report.get("reason", "") or ""),
            "limitations": [
                "requires_apple_virtualization_runner",
                "requires_guest_agent_ready_for_exec",
                "does_not_fall_back_to_host_process",
            ],
        }
    try:
        invocation = build_vm_invocation(
            ".",
            ["true"],
            provider=str(vm_provider or "auto"),
            vm_name=str(vm_name or ""),
            vm_host=str(vm_host or ""),
            vm_workdir=str(vm_workdir or "/workspace"),
            network_mode=str(vm_network_mode or "provider_default"),
        )
    except VMBackendError as exc:
        return {
            "schema_version": EXECUTION_BOUNDARY_VERSION,
            "status": "UNAVAILABLE",
            "backend": "vm",
            "security_boundary": "external_vm_provider",
            "real_vm_boundary": False,
            "reason": str(exc),
            "vm_provider": str(vm_provider or "auto"),
            "vm_name": str(vm_name or ""),
            "vm_host": str(vm_host or ""),
            "vm_workdir": str(vm_workdir or "/workspace"),
            "vm_network_mode": str(vm_network_mode or "provider_default"),
            "sync_boundary": "diff_patch_gate_with_source_hash_and_rollback",
        }
    return {
        "schema_version": EXECUTION_BOUNDARY_VERSION,
        "status": "AVAILABLE",
        "backend": "vm",
        "security_boundary": "external_vm_provider",
        "real_vm_boundary": True,
        "provider": invocation.provider,
        "provider_binary": invocation.provider_binary,
        "vm_name": invocation.vm_name,
        "vm_host": invocation.vm_host,
        "vm_workdir": invocation.vm_workdir,
        "vm_network_mode": invocation.network_mode,
        "filesystem_boundary": "external_vm_workspace",
        "network_boundary": invocation.network_mode,
        "network_policy": "provider_controlled_explicit_mode",
        "credential_boundary": "vm_guest_isolated_explicit_env_only_redacted_in_audit",
        "host_env_forwarded_to_guest": False,
        "sync_boundary": "diff_patch_gate_with_source_hash_and_rollback",
        "limitations": [
            "requires_configured_real_vm_provider",
            "workspace_sync_is_explicit_via_vm_sync_mode_or_mirror_vm_sync",
            "network_isolation_depends_on_provider_configuration",
        ],
    }


def _checkpoint_path(mirror_root: Path, plan_id: str) -> Path:
    safe_plan_id = re.sub(r"[^a-fA-F0-9_.-]", "_", str(plan_id or ""))
    return mirror_root / CONTROL_DIR_NAME / CHECKPOINT_DIR_NAME / f"{safe_plan_id}.json"


def _load_checkpoint(mirror_root: Path, plan_id: str) -> Dict[str, Any]:
    path = _checkpoint_path(mirror_root, plan_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _restore_mirror(source_root: Path, mirror_root: Path) -> LocalMirror:
    workspace_root = mirror_root / WORKSPACE_DIR_NAME
    control_root = mirror_root / CONTROL_DIR_NAME
    manifest = _load_manifest(mirror_root)
    materialized: Dict[str, MaterializedFile] = {}
    for row in list(manifest.get("materialized_files", []) or []):
        if not isinstance(row, dict):
            continue
        rel = str(row.get("relative_path", "") or "")
        if not rel:
            continue
        materialized[rel] = MaterializedFile(
            relative_path=rel,
            source_path=str(row.get("source_path", "") or ""),
            mirror_path=str(row.get("mirror_path", "") or ""),
            size_bytes=int(row.get("size_bytes", 0) or 0),
            sha256=str(row.get("sha256", "") or ""),
        )
    audit_events = [dict(item) for item in list(manifest.get("audit_events", []) or []) if isinstance(item, dict)]
    external_baselines: Dict[str, Dict[str, Any]] = {}
    for row in list(manifest.get("external_baselines", []) or []):
        if not isinstance(row, dict):
            continue
        rel = _safe_relative_path(str(row.get("workspace_relative_path", "") or "")).as_posix()
        if rel:
            external_baselines[rel] = dict(row)
    return LocalMirror(
        source_root=source_root.resolve(),
        mirror_root=mirror_root.resolve(),
        workspace_root=workspace_root.resolve(),
        control_root=control_root.resolve(),
        materialized_files=materialized,
        external_baselines=external_baselines,
        audit_events=audit_events,
    )


def create_empty_mirror(source_root: str | Path, mirror_root: str | Path, *, reset: bool = False) -> LocalMirror:
    source = Path(source_root).resolve()
    root = Path(mirror_root).resolve()
    workspace = root / WORKSPACE_DIR_NAME
    control = root / CONTROL_DIR_NAME
    if reset and root.exists():
        shutil.rmtree(root)
    workspace.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    mirror = LocalMirror(
        source_root=source,
        mirror_root=root,
        workspace_root=workspace.resolve(),
        control_root=control.resolve(),
    )
    if mirror.workspace_files():
        raise MirrorScopeError(f"mirror workspace is not empty: {mirror.workspace_root}")
    mirror.audit_events.append(
        _event(
            "mirror_created_empty",
            source_root=str(source),
            workspace_root=str(mirror.workspace_root),
            control_root=str(mirror.control_root),
            user_files_materialized=0,
        )
    )
    mirror.save_manifest()
    return mirror


def _safe_relative_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise MirrorScopeError(f"absolute paths are not accepted inside mirror requests: {raw_path}")
    if not path.parts:
        raise MirrorScopeError("empty path is not accepted")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise MirrorScopeError(f"path escapes mirror scope: {raw_path}")
    return Path(*path.parts)


def _resolve_source_file(source_root: Path, relative_path: str | Path) -> tuple[Path, str]:
    safe_relative = _safe_relative_path(relative_path)
    raw_source_file = source_root / safe_relative
    if raw_source_file.is_symlink():
        raise MirrorScopeError(f"symlink materialization is not supported: {relative_path}")
    source_file = raw_source_file.resolve()
    try:
        source_file.relative_to(source_root.resolve())
    except ValueError as exc:
        raise MirrorScopeError(f"requested path is outside source root: {relative_path}") from exc
    if not source_file.exists():
        raise FileNotFoundError(str(source_file))
    if not source_file.is_file():
        raise MirrorScopeError(f"only regular files can be materialized: {relative_path}")
    return source_file, safe_relative.as_posix()


def _should_skip_directory_materialization(relative: Path) -> bool:
    if set(relative.parts).intersection(MATERIALIZE_DIRECTORY_EXCLUDE_DIRS):
        return True
    return relative.suffix.lower() in GENERATED_ARTIFACT_SUFFIXES


def _resolve_source_materialization_targets(
    source_root: Path,
    relative_path: str | Path,
) -> tuple[list[tuple[Path, str]], Dict[str, Any]]:
    safe_relative = _safe_relative_path(relative_path)
    raw_source_path = source_root / safe_relative
    if raw_source_path.is_symlink():
        raise MirrorScopeError(f"symlink materialization is not supported: {relative_path}")
    source_path = raw_source_path.resolve()
    try:
        source_path.relative_to(source_root.resolve())
    except ValueError as exc:
        raise MirrorScopeError(f"requested path is outside source root: {relative_path}") from exc
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    if source_path.is_file():
        return [(source_path, safe_relative.as_posix())], {
            "requested_path": safe_relative.as_posix(),
            "kind": "file",
            "expanded": False,
            "file_count": 1,
            "total_bytes": int(source_path.stat().st_size),
            "skipped_count": 0,
        }
    if not source_path.is_dir():
        raise MirrorScopeError(f"only regular files or directories can be materialized: {relative_path}")

    targets: list[tuple[Path, str]] = []
    skipped_count = 0
    total_bytes = 0
    for path in sorted(source_path.rglob("*")):
        try:
            relative = path.resolve().relative_to(source_root.resolve())
        except ValueError as exc:
            raise MirrorScopeError(f"requested path is outside source root: {path}") from exc
        if path.is_symlink() or _should_skip_directory_materialization(relative):
            skipped_count += 1
            continue
        if not path.is_file():
            continue
        size = int(path.stat().st_size)
        if len(targets) + 1 > MAX_DIRECTORY_MATERIALIZATION_FILES:
            raise MirrorScopeError(
                "directory materialization exceeds file limit: "
                f"{safe_relative.as_posix()} > {MAX_DIRECTORY_MATERIALIZATION_FILES}"
            )
        if total_bytes + size > MAX_DIRECTORY_MATERIALIZATION_BYTES:
            raise MirrorScopeError(
                "directory materialization exceeds byte limit: "
                f"{safe_relative.as_posix()} > {MAX_DIRECTORY_MATERIALIZATION_BYTES}"
            )
        targets.append((path, relative.as_posix()))
        total_bytes += size
    return targets, {
        "requested_path": safe_relative.as_posix(),
        "kind": "directory",
        "expanded": True,
        "file_count": len(targets),
        "total_bytes": total_bytes,
        "skipped_count": skipped_count,
        "max_files": MAX_DIRECTORY_MATERIALIZATION_FILES,
        "max_bytes": MAX_DIRECTORY_MATERIALIZATION_BYTES,
    }


def open_mirror(source_root: str | Path, mirror_root: str | Path) -> LocalMirror:
    source = Path(source_root).resolve()
    root = Path(mirror_root).resolve()
    if not (root / WORKSPACE_DIR_NAME).exists():
        return create_empty_mirror(source, root)
    return _restore_mirror(source, root)


def materialize_files(
    source_root: str | Path,
    mirror_root: str | Path,
    relative_paths: Iterable[str | Path],
) -> LocalMirror:
    mirror = open_mirror(source_root, mirror_root)
    for requested_path in relative_paths:
        targets, expansion = _resolve_source_materialization_targets(mirror.source_root, requested_path)
        if expansion.get("expanded"):
            mirror.audit_events.append(_event("directory_materialization_expanded", **expansion))
        for source_file, relative in targets:
            destination = mirror.workspace_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            materialized = MaterializedFile(
                relative_path=relative,
                source_path=str(source_file),
                mirror_path=str(destination),
                size_bytes=int(destination.stat().st_size),
                sha256=_sha256(destination),
            )
            mirror.materialized_files[relative] = materialized
            mirror.audit_events.append(
                _event(
                    "file_materialized_on_demand",
                    relative_path=relative,
                    source_path=str(source_file),
                    mirror_path=str(destination),
                    sha256=materialized.sha256,
                )
            )
    mirror.save_manifest()
    return mirror


def run_mirror_command(
    source_root: str | Path,
    mirror_root: str | Path,
    command: Sequence[str],
    *,
    allowed_commands: Iterable[str] | None = None,
    timeout_seconds: int = 30,
    backend: str = DEFAULT_EXECUTION_BACKEND,
    docker_image: str = "python:3.10-slim",
    extra_env: Mapping[str, str] | None = None,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    vm_sync_mode: str = "none",
) -> MirrorCommandResult:
    mirror = open_mirror(source_root, mirror_root)
    cmd = [str(part) for part in list(command) if str(part)]
    if not cmd:
        raise MirrorScopeError("mirror command is empty")
    executable_name = Path(cmd[0]).name
    allowed = {Path(item).name for item in (allowed_commands or DEFAULT_ALLOWED_COMMANDS)}
    if executable_name not in allowed:
        raise MirrorScopeError(f"command is not allowlisted for mirror execution: {executable_name}")
    selected_backend = str(backend or "local").strip().lower()
    if selected_backend not in SUPPORTED_EXEC_BACKENDS:
        raise MirrorScopeError(f"unsupported mirror execution backend: {selected_backend}")
    sync_mode = normalize_vm_sync_mode(vm_sync_mode)
    if selected_backend == "managed-vm" and sync_mode == "none":
        sync_mode = "push-pull"
    if selected_backend not in {"vm", "managed-vm"} and sync_mode != "none":
        raise MirrorScopeError("vm_sync_mode requires backend='vm' or backend='managed-vm'")
    boundary = execution_boundary_report(
        backend=selected_backend,
        docker_image=docker_image,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
    )
    run_cmd = list(cmd)
    run_cwd = mirror.workspace_root
    image = ""
    env_overlay, env_audit = _sanitize_explicit_env(extra_env)
    run_env = _sanitized_subprocess_env(env_overlay)
    if selected_backend in {"vm", "managed-vm"}:
        sync_results: list[Dict[str, Any]] = []
        provider_arg = "managed" if selected_backend == "managed-vm" else str(vm_provider or "auto")
        try:
            if sync_mode in {"push", "push-pull"}:
                pushed = push_workspace_to_vm(
                    mirror.workspace_root,
                    timeout_seconds=max(1, int(timeout_seconds)),
                    vm_provider=provider_arg,
                    vm_name=str(vm_name or ""),
                    vm_host=str(vm_host or ""),
                    vm_workdir=str(vm_workdir or "/workspace"),
                    vm_network_mode=str(vm_network_mode or "provider_default"),
                    local_cwd=mirror.control_root,
                )
                sync_results.append(pushed.to_dict())
                mirror.audit_events.append(
                    _event(
                        "mirror_vm_workspace_synced",
                        sync_schema=VM_WORKSPACE_SYNC_VERSION,
                        direction="push",
                        status=pushed.status,
                        returncode=pushed.returncode,
                        file_count=pushed.file_count,
                        byte_count=pushed.byte_count,
                        provider=pushed.provider,
                        provider_command=list(pushed.provider_command),
                        vm_workdir=pushed.vm_workdir,
                        real_vm_boundary=pushed.real_vm_boundary,
                    )
                )
                if int(pushed.returncode) != 0:
                    mirror.save_manifest()
                    raise VMBackendError(f"VM workspace push failed: {pushed.stderr or pushed.stdout}")
            vm_completed = run_vm_command(
                mirror.workspace_root,
                cmd,
                timeout_seconds=max(1, int(timeout_seconds)),
                provider=provider_arg,
                vm_name=str(vm_name or ""),
                vm_host=str(vm_host or ""),
                vm_workdir=str(vm_workdir or "/workspace"),
                network_mode=str(vm_network_mode or "provider_default"),
                extra_env=env_overlay,
                local_cwd=mirror.control_root,
            )
            if sync_mode in {"pull", "push-pull"}:
                pulled = pull_workspace_from_vm(
                    mirror.workspace_root,
                    timeout_seconds=max(1, int(timeout_seconds)),
                    vm_provider=provider_arg,
                    vm_name=str(vm_name or ""),
                    vm_host=str(vm_host or ""),
                    vm_workdir=str(vm_workdir or "/workspace"),
                    vm_network_mode=str(vm_network_mode or "provider_default"),
                    local_cwd=mirror.control_root,
                )
                sync_results.append(pulled.to_dict())
                mirror.audit_events.append(
                    _event(
                        "mirror_vm_workspace_synced",
                        sync_schema=VM_WORKSPACE_SYNC_VERSION,
                        direction="pull",
                        status=pulled.status,
                        returncode=pulled.returncode,
                        file_count=pulled.file_count,
                        byte_count=pulled.byte_count,
                        provider=pulled.provider,
                        provider_command=list(pulled.provider_command),
                        vm_workdir=pulled.vm_workdir,
                        real_vm_boundary=pulled.real_vm_boundary,
                    )
                )
                if int(pulled.returncode) != 0:
                    mirror.save_manifest()
                    raise VMBackendError(f"VM workspace pull failed: {pulled.stderr or pulled.stdout}")
        except VMBackendError as exc:
            mirror.audit_events.append(
                _event(
                    "mirror_vm_backend_unavailable",
                    command=cmd,
                    executable=executable_name,
                    backend=selected_backend,
                    execution_boundary=dict(boundary),
                    vm_provider=provider_arg,
                    vm_name=str(vm_name or ""),
                    vm_host=str(vm_host or ""),
                    vm_workdir=str(vm_workdir or "/workspace"),
                    vm_network_mode=str(vm_network_mode or "provider_default"),
                    vm_sync_mode=sync_mode,
                    vm_workspace_sync=sync_results,
                    reason=str(exc),
                    workspace_root=str(mirror.workspace_root),
                )
            )
            mirror.save_manifest()
            raise MirrorScopeError(str(exc)) from exc
        result = MirrorCommandResult(
            command=cmd,
            returncode=int(vm_completed.returncode),
            stdout=_redact_explicit_env_values(str(vm_completed.stdout or ""), env_overlay),
            stderr=_redact_explicit_env_values(str(vm_completed.stderr or ""), env_overlay),
            timeout_seconds=max(1, int(timeout_seconds)),
            backend=selected_backend,
            provider_command=list(vm_completed.provider_command),
            vm_provider=str(vm_completed.provider),
            vm_name=str(vm_completed.vm_name),
            vm_host=str(vm_completed.vm_host),
            vm_workdir=str(vm_completed.vm_workdir),
            vm_network_mode=str(vm_completed.network_mode),
            vm_sync_mode=sync_mode,
            vm_workspace_sync=sync_results,
            real_vm_boundary=bool(vm_completed.real_vm_boundary),
            security_boundary="external_vm_provider",
            execution_boundary=boundary,
            env_audit=dict(env_audit),
        )
        mirror.audit_events.append(
            _event(
                "mirror_command_executed",
                command=cmd,
                executable=executable_name,
                returncode=result.returncode,
                stdout_tail=_tail(_redact_explicit_env_values(result.stdout, env_overlay)),
                stderr_tail=_tail(_redact_explicit_env_values(result.stderr, env_overlay)),
                workspace_root=str(mirror.workspace_root),
                sandbox_label="external_vm_provider",
                not_os_security_sandbox=False,
                security_boundary="external_vm_provider",
                real_vm_boundary=True,
                backend=selected_backend,
                provider_command=list(result.provider_command),
                vm_provider=result.vm_provider,
                vm_name=result.vm_name,
                vm_host=result.vm_host,
                vm_workdir=result.vm_workdir,
                vm_network_mode=result.vm_network_mode,
                vm_sync_mode=sync_mode,
                vm_workspace_sync=sync_results,
                execution_boundary=dict(boundary),
                extra_env_keys=sorted(env_overlay),
                env_audit=dict(env_audit),
                credential_boundary="vm_guest_isolated_explicit_env_only_redacted_in_audit",
                host_env_forwarded_to_guest=False,
                host_env_passthrough=False,
                source_sync_allowed=False,
                source_sync_requires_patch_gate=True,
            )
        )
        mirror.save_manifest()
        return result
    if selected_backend == "docker":
        docker_binary = shutil.which("docker")
        if not docker_binary:
            mirror.audit_events.append(
                _event(
                    "mirror_docker_backend_unavailable",
                    command=cmd,
                    executable=executable_name,
                    backend=selected_backend,
                    execution_boundary=dict(boundary),
                    docker_image=str(docker_image or "python:3.10-slim"),
                    workspace_root=str(mirror.workspace_root),
                )
            )
            mirror.save_manifest()
            raise MirrorScopeError("docker backend requested but docker executable was not found")
        image = str(docker_image or "python:3.10-slim")
        container_cmd = list(cmd)
        if executable_name.startswith("python"):
            container_cmd[0] = "python"
        elif Path(container_cmd[0]).is_absolute():
            container_cmd[0] = executable_name
        env_flags: list[str] = []
        for key, value in sorted(env_overlay.items()):
            env_flags.extend(["-e", f"{key}={value}"])
        run_cmd = [
            docker_binary,
            "run",
            "--rm",
            "--network",
            "none",
            *env_flags,
            "-v",
            f"{mirror.workspace_root}:/workspace",
            "-w",
            "/workspace",
            image,
            *container_cmd,
        ]
        run_cwd = mirror.control_root
    try:
        completed = subprocess.run(
            run_cmd,
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
            env=run_env,
        )
        result = MirrorCommandResult(
            command=cmd,
            returncode=int(completed.returncode),
            stdout=_redact_explicit_env_values(str(completed.stdout or ""), env_overlay),
            stderr=_redact_explicit_env_values(str(completed.stderr or ""), env_overlay),
            timeout_seconds=max(1, int(timeout_seconds)),
            backend=selected_backend,
            docker_image=image,
            security_boundary="container_best_effort" if selected_backend == "docker" else "best_effort_local_process",
            execution_boundary=boundary,
            env_audit=dict(env_audit),
        )
    except subprocess.TimeoutExpired as exc:
        result = MirrorCommandResult(
            command=cmd,
            returncode=124,
            stdout=_redact_explicit_env_values(str(exc.stdout or ""), env_overlay),
            stderr=_redact_explicit_env_values(str(exc.stderr or "") + "\nmirror command timed out", env_overlay),
            timeout_seconds=max(1, int(timeout_seconds)),
            backend=selected_backend,
            docker_image=image,
            security_boundary="container_best_effort" if selected_backend == "docker" else "best_effort_local_process",
            execution_boundary=boundary,
            env_audit=dict(env_audit),
        )
    mirror.audit_events.append(
        _event(
            "mirror_command_executed",
        command=cmd,
        executable=executable_name,
        returncode=result.returncode,
        stdout_tail=_tail(_redact_explicit_env_values(result.stdout, env_overlay)),
        stderr_tail=_tail(_redact_explicit_env_values(result.stderr, env_overlay)),
        workspace_root=str(mirror.workspace_root),
        sandbox_label="best_effort_local_mirror",
        not_os_security_sandbox=True,
        security_boundary=result.security_boundary,
        backend=selected_backend,
        docker_image=image,
        execution_boundary=dict(boundary),
        extra_env_keys=sorted(env_overlay),
        env_audit=dict(env_audit),
        credential_boundary=(
            "container_env_explicit_only_redacted_in_audit"
            if selected_backend == "docker"
            else "sanitized_process_env_explicit_env_only_redacted_in_audit"
        ),
        host_env_forwarded=False,
        host_env_passthrough=False,
        source_sync_allowed=False,
        source_sync_requires_patch_gate=True,
    )
    )
    mirror.save_manifest()
    return result


def manage_vm_workspace(
    source_root: str | Path,
    mirror_root: str | Path,
    *,
    operation: str,
    timeout_seconds: int = 30,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    checkpoint_id: str = "",
) -> Dict[str, Any]:
    """Run an audited real-VM workspace lifecycle operation."""

    mirror = open_mirror(source_root, mirror_root)
    result = run_vm_workspace_operation(
        operation,
        timeout_seconds=timeout_seconds,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
        checkpoint_id=checkpoint_id,
        local_cwd=mirror.control_root,
    )
    payload = result.to_dict()
    mirror.audit_events.append(
        _event(
            "mirror_vm_workspace_operation",
            manager_schema=VM_MANAGER_VERSION,
            operation=str(operation or "").strip().lower(),
            status=str(payload.get("status", "") or ""),
            returncode=int(payload.get("returncode", 0) or 0),
            provider=str(payload.get("provider", "") or ""),
            provider_command=list(payload.get("provider_command", []) or []),
            vm_name=str(payload.get("vm_name", "") or ""),
            vm_host=str(payload.get("vm_host", "") or ""),
            vm_workdir=str(payload.get("vm_workdir", "") or ""),
            vm_network_mode=str(payload.get("vm_network_mode", "") or ""),
            real_vm_boundary=bool(payload.get("real_vm_boundary", False)),
            checkpoint_id=str(payload.get("checkpoint_id", "") or ""),
            checkpoint_path=str(payload.get("checkpoint_path", "") or ""),
            source_sync_allowed=False,
            source_sync_requires_patch_gate=True,
        )
    )
    mirror.save_manifest()
    return payload


def sync_vm_workspace(
    source_root: str | Path,
    mirror_root: str | Path,
    *,
    direction: str,
    timeout_seconds: int = 30,
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
) -> Dict[str, Any]:
    """Push or pull a mirror workspace to or from a configured real VM."""

    selected_direction = str(direction or "").strip().lower()
    if selected_direction == "push":
        operation = push_workspace_to_vm
    elif selected_direction == "pull":
        operation = pull_workspace_from_vm
    else:
        raise MirrorScopeError("vm workspace sync direction must be 'push' or 'pull'")
    mirror = open_mirror(source_root, mirror_root)
    result = operation(
        mirror.workspace_root,
        timeout_seconds=timeout_seconds,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
        local_cwd=mirror.control_root,
    )
    payload = result.to_dict()
    mirror.audit_events.append(
        _event(
            "mirror_vm_workspace_synced",
            sync_schema=VM_WORKSPACE_SYNC_VERSION,
            direction=selected_direction,
            status=str(payload.get("status", "") or ""),
            returncode=int(payload.get("returncode", 0) or 0),
            file_count=int(payload.get("file_count", 0) or 0),
            byte_count=int(payload.get("byte_count", 0) or 0),
            provider=str(payload.get("provider", "") or ""),
            provider_command=list(payload.get("provider_command", []) or []),
            vm_name=str(payload.get("vm_name", "") or ""),
            vm_host=str(payload.get("vm_host", "") or ""),
            vm_workdir=str(payload.get("vm_workdir", "") or ""),
            vm_network_mode=str(payload.get("vm_network_mode", "") or ""),
            real_vm_boundary=bool(payload.get("real_vm_boundary", False)),
            sync_boundary="vm_workspace_only_not_source_sync",
            source_sync_allowed=False,
            source_sync_requires_patch_gate=True,
        )
    )
    mirror.save_manifest()
    return payload


def _workspace_relative_path(mirror: LocalMirror, path: Path) -> str:
    relative = path.resolve().relative_to(mirror.workspace_root.resolve()).as_posix()
    return _safe_relative_path(relative).as_posix()


def _text_pair(source_path: Path, mirror_path: Path, *, max_bytes: int = 128 * 1024) -> tuple[str, str] | None:
    try:
        source_size = source_path.stat().st_size if source_path.exists() else 0
        mirror_size = mirror_path.stat().st_size if mirror_path.exists() else 0
    except OSError:
        return None
    if max(source_size, mirror_size) > max_bytes:
        return None
    try:
        source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
        mirror_text = mirror_path.read_text(encoding="utf-8") if mirror_path.exists() else ""
    except UnicodeDecodeError:
        return None
    return source_text, mirror_text


def _build_text_patch(source_text: str, mirror_text: str, relative: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            source_text.splitlines(),
            mirror_text.splitlines(),
            fromfile=relative,
            tofile=relative,
            lineterm="",
        )
    )


def _text_patch(source_path: Path, mirror_path: Path, relative: str, *, max_bytes: int = 128 * 1024) -> str:
    pair = _text_pair(source_path, mirror_path, max_bytes=max_bytes)
    if pair is None:
        return ""
    return _build_text_patch(pair[0], pair[1], relative)


def _endswith_newline(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            if handle.seek(0, 2) == 0:
                return False
            handle.seek(-1, 2)
            return handle.read(1) == b"\n"
    except OSError:
        return False


def _is_git_internal_relative(relative: str) -> bool:
    return ".git" in Path(str(relative or "")).parts


def _external_baseline_for_relative(mirror: LocalMirror, relative: str) -> tuple[Dict[str, Any], str] | tuple[None, str]:
    rel = _safe_relative_path(relative).as_posix()
    for root, baseline in sorted(mirror.external_baselines.items(), key=lambda item: len(item[0]), reverse=True):
        root_rel = _safe_relative_path(root).as_posix()
        if rel == root_rel:
            return baseline, ""
        prefix = f"{root_rel}/"
        if rel.startswith(prefix):
            return baseline, rel[len(prefix):]
    return None, rel


def _external_baseline_file(baseline: Mapping[str, Any], sub_relative: str) -> Path:
    base = Path(str(baseline.get("baseline_path", "") or ""))
    sub = _safe_relative_path(sub_relative)
    return (base / sub).resolve()


def compute_mirror_diff(source_root: str | Path, mirror_root: str | Path) -> list[MirrorDiffEntry]:
    mirror = open_mirror(source_root, mirror_root)
    entries: Dict[str, MirrorDiffEntry] = {}
    for workspace_file in mirror.workspace_files():
        relative = _workspace_relative_path(mirror, workspace_file)
        if is_generated_mirror_artifact(relative):
            continue
        baseline, baseline_subpath = _external_baseline_for_relative(mirror, relative)
        if baseline is not None and _is_git_internal_relative(baseline_subpath):
            continue
        source_file = (
            _external_baseline_file(baseline, baseline_subpath)
            if baseline is not None
            else (mirror.source_root / relative).resolve()
        )
        mirror_sha = _sha256(workspace_file)
        source_sha = _sha256(source_file) if source_file.exists() and source_file.is_file() else ""
        if not source_file.exists():
            status = "added"
        elif source_sha == mirror_sha:
            if baseline is not None:
                continue
            status = "unchanged"
        else:
            status = "modified"
        patch = _text_patch(source_file, workspace_file, relative)
        entries[relative] = MirrorDiffEntry(
            relative_path=relative,
            status=status,
            source_path=str(source_file),
            mirror_path=str(workspace_file),
            source_sha256=source_sha,
            mirror_sha256=mirror_sha,
            size_bytes=int(workspace_file.stat().st_size),
            text_patch=patch,
            patch_sha256=_hash_text(patch) if patch else "",
            source_endswith_newline=_endswith_newline(source_file),
            mirror_endswith_newline=_endswith_newline(workspace_file),
        )
    for root, baseline in sorted(mirror.external_baselines.items()):
        baseline_path = Path(str(baseline.get("baseline_path", "") or ""))
        if not baseline_path.exists() or not baseline_path.is_dir():
            continue
        for baseline_file in sorted(path for path in baseline_path.rglob("*") if path.is_file()):
            sub_relative = baseline_file.relative_to(baseline_path).as_posix()
            if is_generated_mirror_artifact(sub_relative):
                continue
            if _is_git_internal_relative(sub_relative):
                continue
            relative = f"{_safe_relative_path(root).as_posix()}/{sub_relative}" if sub_relative else _safe_relative_path(root).as_posix()
            if relative in entries:
                continue
            workspace_file = (mirror.workspace_root / relative).resolve()
            if workspace_file.exists():
                continue
            entries[relative] = MirrorDiffEntry(
                relative_path=relative,
                status="removed_in_mirror",
                source_path=str(baseline_file.resolve()),
                mirror_path=str(workspace_file),
                source_sha256=_sha256(baseline_file),
                mirror_sha256="",
                size_bytes=0,
                text_patch="",
                patch_sha256="",
                source_endswith_newline=_endswith_newline(baseline_file),
                mirror_endswith_newline=False,
            )
    for relative, materialized in mirror.materialized_files.items():
        if is_generated_mirror_artifact(relative):
            continue
        if relative in entries:
            continue
        entries[relative] = MirrorDiffEntry(
            relative_path=relative,
            status="removed_in_mirror",
            source_path=str(mirror.source_root / relative),
            mirror_path=str(mirror.workspace_root / relative),
            source_sha256=materialized.sha256,
            mirror_sha256="",
            size_bytes=0,
            text_patch="",
            patch_sha256="",
            source_endswith_newline=False,
            mirror_endswith_newline=False,
        )
    return [entry for _, entry in sorted(entries.items())]


def _command_failures(mirror: LocalMirror) -> list[Dict[str, Any]]:
    failures: list[Dict[str, Any]] = []
    for event in mirror.audit_events:
        if event.get("event_type") != "mirror_command_executed":
            continue
        payload = dict(event.get("payload", {}) or {})
        if int(payload.get("returncode", 0) or 0) != 0:
            failures.append(payload)
    return failures


def build_sync_plan(source_root: str | Path, mirror_root: str | Path) -> Dict[str, Any]:
    mirror = open_mirror(source_root, mirror_root)
    diff_entries = compute_mirror_diff(source_root, mirror_root)
    actionable = [entry for entry in diff_entries if entry.status in {"added", "modified"}]
    human_reasons: list[str] = []
    for failure in _command_failures(mirror):
        human_reasons.append(f"mirror_command_failed:{failure.get('command', [])}")
    for entry in diff_entries:
        suffix = Path(entry.relative_path).suffix.lower()
        if entry.status == "removed_in_mirror":
            human_reasons.append(f"delete_or_remove_requires_human_review:{entry.relative_path}")
        elif entry.status in {"added", "modified"} and suffix not in MACHINE_APPROVABLE_SUFFIXES:
            human_reasons.append(f"unsupported_suffix_requires_human_review:{entry.relative_path}")
        elif entry.status in {"added", "modified"} and not entry.text_patch:
            human_reasons.append(f"missing_text_patch_requires_human_review:{entry.relative_path}")
    machine_approved = bool(actionable) and not human_reasons
    plan_body = {
        "schema_version": LOCAL_MIRROR_SYNC_PLAN_VERSION,
        "source_root": str(mirror.source_root),
        "mirror_root": str(mirror.mirror_root),
        "workspace_root": str(mirror.workspace_root),
        "generated_at": _now(),
        "diff_entries": [entry.to_dict() for entry in diff_entries],
        "actionable_changes": [entry.to_dict() for entry in actionable],
        "approval": {
            "status": "machine_approved" if machine_approved else "human_review_required",
            "machine_approved": machine_approved,
            "human_required": not machine_approved,
            "reasons": human_reasons,
        },
        "apply_scope": {
            "mode": "patch_gate_added_or_modified_files_only",
            "apply_method": "unified_text_patch",
            "copy_back_allowed": False,
            "deletions_supported": False,
            "requires_plan_id": True,
            "requires_source_hash_match": True,
            "creates_rollback_checkpoint": True,
            "source_sync_policy": "no_copy_back_patch_gate_only",
        },
    }
    plan_id = _hash_text(json.dumps(plan_body, sort_keys=True, ensure_ascii=False, default=str))
    plan = {"plan_id": plan_id, **plan_body}
    mirror.control_root.mkdir(parents=True, exist_ok=True)
    mirror.sync_plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mirror.audit_events.append(
        _event(
            "sync_plan_built",
            plan_id=plan_id,
            actionable_change_count=len(actionable),
            approval_status=plan["approval"]["status"],
            sync_gate_mode="patch_gate_added_or_modified_files_only",
            apply_method="unified_text_patch",
            copy_back_allowed=False,
        )
    )
    mirror.save_manifest()
    return plan


def _parse_hunk_header(line: str) -> tuple[int, int, int, int]:
    match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
    if not match:
        raise MirrorScopeError(f"invalid patch hunk header: {line}")
    old_start = int(match.group(1) or 0)
    old_count = int(match.group(2) or 1)
    new_start = int(match.group(3) or 0)
    new_count = int(match.group(4) or 1)
    return old_start, old_count, new_start, new_count


def _apply_unified_text_patch(
    source_text: str,
    patch_text: str,
    *,
    result_endswith_newline: bool,
) -> str:
    if not str(patch_text or "").strip():
        raise MirrorScopeError("sync plan is missing a text patch for patch-gate apply")
    source_lines = source_text.splitlines()
    patch_lines = str(patch_text).splitlines()
    result_lines: list[str] = []
    source_index = 0
    index = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        if line.startswith("--- ") or line.startswith("+++ "):
            index += 1
            continue
        if not line.startswith("@@ "):
            raise MirrorScopeError(f"unexpected patch line before hunk: {line}")
        old_start, _old_count, _new_start, _new_count = _parse_hunk_header(line)
        hunk_start = max(0, old_start - 1)
        if hunk_start < source_index:
            raise MirrorScopeError("patch hunks overlap or move backwards")
        result_lines.extend(source_lines[source_index:hunk_start])
        source_index = hunk_start
        index += 1
        while index < len(patch_lines) and not patch_lines[index].startswith("@@ "):
            hunk_line = patch_lines[index]
            if hunk_line.startswith(" "):
                content = hunk_line[1:]
                if source_index >= len(source_lines) or source_lines[source_index] != content:
                    raise MirrorScopeError("patch context does not match current source")
                result_lines.append(content)
                source_index += 1
            elif hunk_line.startswith("-"):
                content = hunk_line[1:]
                if source_index >= len(source_lines) or source_lines[source_index] != content:
                    raise MirrorScopeError("patch deletion does not match current source")
                source_index += 1
            elif hunk_line.startswith("+"):
                result_lines.append(hunk_line[1:])
            elif hunk_line.startswith("\\"):
                pass
            else:
                raise MirrorScopeError(f"unsupported patch line: {hunk_line}")
            index += 1
    result_lines.extend(source_lines[source_index:])
    result_text = "\n".join(result_lines)
    if result_endswith_newline and (result_text or source_text):
        result_text += "\n"
    return result_text


def _safe_checkpoint_path(mirror: LocalMirror, relative: str) -> Path:
    safe_relative = _safe_relative_path(relative).as_posix()
    return (mirror.source_root / safe_relative).resolve()


def _build_checkpoint(
    mirror: LocalMirror,
    *,
    plan: Dict[str, Any],
    planned_rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    files: list[Dict[str, Any]] = []
    for row in planned_rows:
        relative = _safe_relative_path(str(row.get("relative_path", "") or "")).as_posix()
        source_file = _safe_checkpoint_path(mirror, relative)
        mirror_file = (mirror.workspace_root / relative).resolve()
        source_text = source_file.read_text(encoding="utf-8") if source_file.exists() and source_file.is_file() else ""
        mirror_text = mirror_file.read_text(encoding="utf-8")
        reverse_patch = _build_text_patch(mirror_text, source_text, relative)
        files.append(
            {
                "relative_path": relative,
                "status": str(row.get("status", "") or ""),
                "source_path": str(source_file),
                "original_exists": bool(source_file.exists()),
                "original_sha256": _sha256(source_file) if source_file.exists() and source_file.is_file() else "",
                "original_endswith_newline": _endswith_newline(source_file),
                "applied_sha256": str(row.get("mirror_sha256", "") or ""),
                "applied_endswith_newline": bool(row.get("mirror_endswith_newline", False)),
                "planned_source_sha256": str(row.get("source_sha256", "") or ""),
                "forward_patch": str(row.get("text_patch", "") or ""),
                "forward_patch_sha256": str(row.get("patch_sha256", "") or ""),
                "reverse_patch": reverse_patch,
                "reverse_patch_sha256": _hash_text(reverse_patch) if reverse_patch else "",
            }
        )
    return {
        "schema_version": LOCAL_MIRROR_CHECKPOINT_VERSION,
        "plan_id": str(plan.get("plan_id", "") or ""),
        "source_root": str(mirror.source_root),
        "mirror_root": str(mirror.mirror_root),
        "generated_at": _now(),
        "rollback_supported": True,
        "files": files,
    }


def _save_checkpoint(mirror: LocalMirror, checkpoint: Dict[str, Any]) -> Path:
    plan_id = str(checkpoint.get("plan_id", "") or "")
    path = _checkpoint_path(mirror.mirror_root, plan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def apply_sync_plan(
    source_root: str | Path,
    mirror_root: str | Path,
    *,
    plan_id: str,
    approved_by: str,
) -> Dict[str, Any]:
    mirror = open_mirror(source_root, mirror_root)
    plan = _load_sync_plan(mirror.mirror_root)
    if not plan:
        raise MirrorScopeError("sync plan is missing; run mirror plan first")
    if str(plan.get("plan_id", "") or "") != str(plan_id):
        raise MirrorScopeError("sync plan id does not match the approved plan")
    approver = str(approved_by or "").strip().lower()
    if approver not in {"human", "machine"}:
        raise MirrorScopeError("approved_by must be either 'human' or 'machine'")
    approval = dict(plan.get("approval", {}) or {})
    if approver == "machine" and not bool(approval.get("machine_approved", False)):
        raise MirrorScopeError("machine approval is not sufficient for this sync plan")

    planned_rows: list[Dict[str, Any]] = [
        dict(row)
        for row in list(plan.get("actionable_changes", []) or [])
        if isinstance(row, dict) and str(row.get("status", "") or "") in {"added", "modified"}
    ]
    patch_results: list[Dict[str, Any]] = []
    source_hash_checks: list[Dict[str, Any]] = []
    mirror_hash_checks: list[Dict[str, Any]] = []
    for row in planned_rows:
        status = str(row.get("status", "") or "")
        relative = _safe_relative_path(str(row.get("relative_path", "") or "")).as_posix()
        mirror_file = (mirror.workspace_root / relative).resolve()
        source_file = (mirror.source_root / relative).resolve()
        planned_source_sha = str(row.get("source_sha256", "") or "")
        planned_mirror_sha = str(row.get("mirror_sha256", "") or "")
        current_source_sha = _sha256(source_file) if source_file.exists() and source_file.is_file() else ""
        current_mirror_sha = _sha256(mirror_file) if mirror_file.exists() and mirror_file.is_file() else ""
        source_hash_checks.append(
            {
                "relative_path": relative,
                "planned_source_sha256": planned_source_sha,
                "current_source_sha256": current_source_sha,
                "matched": current_source_sha == planned_source_sha,
            }
        )
        if current_source_sha != planned_source_sha:
            mirror.audit_events.append(
                _event(
                    "sync_plan_rejected_source_hash_mismatch",
                    plan_id=str(plan_id),
                    relative_path=relative,
                    planned_source_sha256=planned_source_sha,
                    current_source_sha256=current_source_sha,
                )
            )
            mirror.save_manifest()
            raise MirrorScopeError(
                f"source hash mismatch for {relative}: "
                f"current_source_sha256={current_source_sha}, "
                f"planned_source_sha256={planned_source_sha}"
            )
        mirror_hash_checks.append(
            {
                "relative_path": relative,
                "planned_mirror_sha256": planned_mirror_sha,
                "current_mirror_sha256": current_mirror_sha,
                "matched": current_mirror_sha == planned_mirror_sha,
            }
        )
        if current_mirror_sha != planned_mirror_sha:
            mirror.audit_events.append(
                _event(
                    "sync_plan_rejected_mirror_hash_mismatch",
                    plan_id=str(plan_id),
                    relative_path=relative,
                    planned_mirror_sha256=planned_mirror_sha,
                    current_mirror_sha256=current_mirror_sha,
                )
            )
            mirror.save_manifest()
            raise MirrorScopeError(
                f"mirror hash mismatch for {relative}: "
                f"current_mirror_sha256={current_mirror_sha}, "
                f"planned_mirror_sha256={planned_mirror_sha}"
            )
        source_text = source_file.read_text(encoding="utf-8") if source_file.exists() and source_file.is_file() else ""
        patched_text = _apply_unified_text_patch(
            source_text,
            str(row.get("text_patch", "") or ""),
            result_endswith_newline=bool(row.get("mirror_endswith_newline", False)),
        )
        patched_sha = _hash_text(patched_text)
        if patched_sha != planned_mirror_sha:
            raise MirrorScopeError(
                f"patch result hash mismatch for {relative}: "
                f"patched_sha256={patched_sha}, planned_mirror_sha256={planned_mirror_sha}"
            )
        patch_results.append(
            {
                "relative_path": relative,
                "status": status,
                "source_path": str(source_file),
                "patched_text": patched_text,
                "patched_sha256": patched_sha,
            }
        )

    checkpoint_path = ""
    if planned_rows:
        checkpoint = _build_checkpoint(mirror, plan=plan, planned_rows=planned_rows)
        checkpoint_path = str(_save_checkpoint(mirror, checkpoint))
        mirror.audit_events.append(
            _event(
                "sync_plan_checkpoint_created",
                plan_id=str(plan_id),
                checkpoint_path=checkpoint_path,
                file_count=len(planned_rows),
            )
        )

    synced: list[Dict[str, Any]] = []
    for patch_result in patch_results:
        relative = str(patch_result.get("relative_path", "") or "")
        source_file = _safe_checkpoint_path(mirror, relative)
        source_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(source_file.parent),
            delete=False,
        ) as handle:
            handle.write(str(patch_result.get("patched_text", "") or ""))
            temp_path = Path(handle.name)
        temp_path.replace(source_file)
        synced.append(
            {
                "relative_path": relative,
                "status": str(patch_result.get("status", "") or ""),
                "source_path": str(source_file),
                "sha256": _sha256(source_file),
                "apply_method": "unified_text_patch",
            }
        )
    mirror.audit_events.append(
        _event(
            "sync_plan_applied",
            plan_id=str(plan_id),
            approved_by=approver,
            synced_files=synced,
            source_hash_checks=source_hash_checks,
            mirror_hash_checks=mirror_hash_checks,
            checkpoint_path=checkpoint_path,
            apply_method="unified_text_patch",
            sync_gate_mode="patch_gate_added_or_modified_files_only",
            copy_back_allowed=False,
            source_sync_policy="no_copy_back_patch_gate_only",
        )
    )
    mirror.save_manifest()
    return {
        "schema_version": "conos.local_mirror_sync_result/v1",
        "plan_id": str(plan_id),
        "approved_by": approver,
        "apply_method": "unified_text_patch",
        "sync_gate_mode": "patch_gate_added_or_modified_files_only",
        "copy_back_allowed": False,
        "source_sync_policy": "no_copy_back_patch_gate_only",
        "checkpoint_path": checkpoint_path,
        "source_hash_checks": source_hash_checks,
        "mirror_hash_checks": mirror_hash_checks,
        "synced_files": synced,
    }


def rollback_sync_plan(
    source_root: str | Path,
    mirror_root: str | Path,
    *,
    plan_id: str,
) -> Dict[str, Any]:
    mirror = open_mirror(source_root, mirror_root)
    checkpoint = _load_checkpoint(mirror.mirror_root, plan_id)
    if not checkpoint:
        raise MirrorScopeError("rollback checkpoint is missing for this plan id")
    if str(checkpoint.get("plan_id", "") or "") != str(plan_id):
        raise MirrorScopeError("rollback checkpoint plan id does not match")

    restored: list[Dict[str, Any]] = []
    hash_checks: list[Dict[str, Any]] = []
    for row in reversed(list(checkpoint.get("files", []) or [])):
        if not isinstance(row, dict):
            continue
        relative = _safe_relative_path(str(row.get("relative_path", "") or "")).as_posix()
        source_file = _safe_checkpoint_path(mirror, relative)
        expected_applied_sha = str(row.get("applied_sha256", "") or "")
        current_sha = _sha256(source_file) if source_file.exists() and source_file.is_file() else ""
        hash_checks.append(
            {
                "relative_path": relative,
                "expected_applied_sha256": expected_applied_sha,
                "current_source_sha256": current_sha,
                "matched": current_sha == expected_applied_sha,
            }
        )
        if current_sha != expected_applied_sha:
            mirror.audit_events.append(
                _event(
                    "rollback_rejected_source_hash_mismatch",
                    plan_id=str(plan_id),
                    relative_path=relative,
                    expected_applied_sha256=expected_applied_sha,
                    current_source_sha256=current_sha,
                )
            )
            mirror.save_manifest()
            raise MirrorScopeError(
                f"rollback source hash mismatch for {relative}: "
                f"current_source_sha256={current_sha}, "
                f"expected_applied_sha256={expected_applied_sha}"
            )
        if not bool(row.get("original_exists", False)):
            if source_file.exists():
                source_file.unlink()
            restored.append(
                {
                    "relative_path": relative,
                    "rollback_action": "removed_added_file",
                    "restored_sha256": "",
                }
            )
            continue
        current_text = source_file.read_text(encoding="utf-8")
        restored_text = _apply_unified_text_patch(
            current_text,
            str(row.get("reverse_patch", "") or ""),
            result_endswith_newline=bool(row.get("original_endswith_newline", False)),
        )
        restored_sha = _hash_text(restored_text)
        expected_original_sha = str(row.get("original_sha256", "") or "")
        if restored_sha != expected_original_sha:
            raise MirrorScopeError(
                f"rollback patch result hash mismatch for {relative}: "
                f"restored_sha256={restored_sha}, expected_original_sha256={expected_original_sha}"
            )
        source_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(source_file.parent),
            delete=False,
        ) as handle:
            handle.write(restored_text)
            temp_path = Path(handle.name)
        temp_path.replace(source_file)
        restored.append(
            {
                "relative_path": relative,
                "rollback_action": "reverse_patch_applied",
                "restored_sha256": _sha256(source_file),
            }
        )

    mirror.audit_events.append(
        _event(
            "sync_plan_rolled_back",
            plan_id=str(plan_id),
            checkpoint_schema=str(checkpoint.get("schema_version", "") or ""),
            restored_files=restored,
            source_hash_checks=hash_checks,
        )
    )
    mirror.save_manifest()
    return {
        "schema_version": "conos.local_mirror_rollback_result/v1",
        "plan_id": str(plan_id),
        "checkpoint_schema": str(checkpoint.get("schema_version", "") or ""),
        "source_hash_checks": hash_checks,
        "restored_files": restored,
    }


def _instruction_tokens(instruction: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-zA-Z0-9_./-]+", instruction.lower())
        if token
    }


def select_relevant_paths(
    instruction: str,
    candidate_paths: Iterable[str | Path],
    *,
    limit: int = 20,
) -> list[str]:
    tokens = _instruction_tokens(instruction)
    scored: list[tuple[int, str]] = []
    for raw_candidate in candidate_paths:
        candidate = _safe_relative_path(raw_candidate).as_posix()
        lowered = candidate.lower()
        parts = [part.lower() for part in Path(candidate).parts]
        stem = Path(candidate).stem.lower()
        score = 0
        if lowered in tokens:
            score += 8
        if Path(candidate).name.lower() in tokens:
            score += 6
        if stem in tokens:
            score += 4
        score += sum(1 for part in parts if part in tokens)
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, candidate in scored[: max(0, int(limit))]]


def acquire_relevant_files(
    source_root: str | Path,
    mirror_root: str | Path,
    *,
    instruction: str,
    candidate_paths: Iterable[str | Path],
    limit: int = 20,
) -> LocalMirror:
    candidates = list(candidate_paths)
    selected = select_relevant_paths(instruction, candidates, limit=limit)
    mirror = materialize_files(source_root, mirror_root, selected)
    mirror.audit_events.append(
        _event(
            "instruction_scoped_acquisition",
            instruction=instruction,
            selected_paths=selected,
            candidate_count=len(candidates),
        )
    )
    mirror.save_manifest()
    return mirror


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conos mirror", description="Manage an empty-first local mirror workspace.")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create an empty mirror workspace.")
    init_parser.add_argument("--source-root", default=".")
    init_parser.add_argument("--mirror-root", required=True)
    init_parser.add_argument("--reset", action="store_true")

    fetch_parser = subparsers.add_parser("fetch", help="Materialize explicit files into the mirror workspace.")
    fetch_parser.add_argument("--source-root", default=".")
    fetch_parser.add_argument("--mirror-root", required=True)
    fetch_parser.add_argument("--path", action="append", default=[])

    acquire_parser = subparsers.add_parser("acquire", help="Materialize files selected from an instruction and candidates.")
    acquire_parser.add_argument("--source-root", default=".")
    acquire_parser.add_argument("--mirror-root", required=True)
    acquire_parser.add_argument("--instruction", required=True)
    acquire_parser.add_argument("--candidate", action="append", default=[])
    acquire_parser.add_argument("--limit", type=int, default=20)

    manifest_parser = subparsers.add_parser("manifest", help="Print the mirror manifest.")
    manifest_parser.add_argument("--source-root", default=".")
    manifest_parser.add_argument("--mirror-root", required=True)

    boundary_parser = subparsers.add_parser("boundary", help="Inspect the requested mirror execution boundary.")
    boundary_parser.add_argument("--backend", choices=sorted(SUPPORTED_EXEC_BACKENDS), default=DEFAULT_EXECUTION_BACKEND)
    boundary_parser.add_argument("--docker-image", default="python:3.10-slim")
    boundary_parser.add_argument("--vm-provider", choices=["auto", "managed", "managed-vm", "lima", "ssh"], default="auto")
    boundary_parser.add_argument("--vm-name", default="")
    boundary_parser.add_argument("--vm-host", default="")
    boundary_parser.add_argument("--vm-workdir", default="/workspace")
    boundary_parser.add_argument(
        "--vm-network-mode",
        choices=["provider_default", "configured_isolated"],
        default="provider_default",
    )

    vm_parser = subparsers.add_parser("vm", help="Manage the real VM workspace lifecycle for this mirror.")
    vm_parser.add_argument("--source-root", default=".")
    vm_parser.add_argument("--mirror-root", required=False, default="")
    vm_parser.add_argument(
        "--operation",
        choices=["report", "preflight", "prepare", "checkpoint", "restore", "cleanup"],
        default="report",
    )
    vm_parser.add_argument("--timeout", type=int, default=30)
    vm_parser.add_argument("--vm-provider", choices=["auto", "managed", "managed-vm", "lima", "ssh"], default="auto")
    vm_parser.add_argument("--vm-name", default="")
    vm_parser.add_argument("--vm-host", default="")
    vm_parser.add_argument("--vm-workdir", default="/workspace")
    vm_parser.add_argument(
        "--vm-network-mode",
        choices=["provider_default", "configured_isolated"],
        default="provider_default",
    )
    vm_parser.add_argument("--checkpoint-id", default="")

    vm_sync_parser = subparsers.add_parser("vm-sync", help="Push or pull the mirror workspace to/from a real VM.")
    vm_sync_parser.add_argument("--source-root", default=".")
    vm_sync_parser.add_argument("--mirror-root", required=True)
    vm_sync_parser.add_argument("--direction", choices=["push", "pull"], required=True)
    vm_sync_parser.add_argument("--timeout", type=int, default=30)
    vm_sync_parser.add_argument("--vm-provider", choices=["auto", "managed", "managed-vm", "lima", "ssh"], default="auto")
    vm_sync_parser.add_argument("--vm-name", default="")
    vm_sync_parser.add_argument("--vm-host", default="")
    vm_sync_parser.add_argument("--vm-workdir", default="/workspace")
    vm_sync_parser.add_argument(
        "--vm-network-mode",
        choices=["provider_default", "configured_isolated"],
        default="provider_default",
    )

    exec_parser = subparsers.add_parser("exec", help="Run an allowlisted command inside the mirror workspace.")
    exec_parser.add_argument("--source-root", default=".")
    exec_parser.add_argument("--mirror-root", required=True)
    exec_parser.add_argument("--timeout", type=int, default=30)
    exec_parser.add_argument("--allow-command", action="append", default=[])
    exec_parser.add_argument("--backend", choices=sorted(SUPPORTED_EXEC_BACKENDS), default=DEFAULT_EXECUTION_BACKEND)
    exec_parser.add_argument("--docker-image", default="python:3.10-slim")
    exec_parser.add_argument("--vm-provider", choices=["auto", "managed", "managed-vm", "lima", "ssh"], default="auto")
    exec_parser.add_argument("--vm-name", default="")
    exec_parser.add_argument("--vm-host", default="")
    exec_parser.add_argument("--vm-workdir", default="/workspace")
    exec_parser.add_argument(
        "--vm-network-mode",
        choices=["provider_default", "configured_isolated"],
        default="provider_default",
    )
    exec_parser.add_argument(
        "--vm-sync-mode",
        choices=sorted(SUPPORTED_VM_SYNC_MODES),
        default="none",
        help="When backend=vm, explicitly push/pull the mirror workspace around command execution.",
    )
    exec_parser.add_argument("exec_args", nargs=argparse.REMAINDER)

    plan_parser = subparsers.add_parser("plan", help="Build a reviewed sync plan from mirror changes.")
    plan_parser.add_argument("--source-root", default=".")
    plan_parser.add_argument("--mirror-root", required=True)

    apply_parser = subparsers.add_parser("apply", help="Apply an approved sync plan to the source root.")
    apply_parser.add_argument("--source-root", default=".")
    apply_parser.add_argument("--mirror-root", required=True)
    apply_parser.add_argument("--plan-id", required=True)
    apply_parser.add_argument("--approved-by", required=True, choices=["human", "machine"])

    rollback_parser = subparsers.add_parser("rollback", help="Rollback an applied sync plan from its checkpoint.")
    rollback_parser.add_argument("--source-root", default=".")
    rollback_parser.add_argument("--mirror-root", required=True)
    rollback_parser.add_argument("--plan-id", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "init":
        mirror = create_empty_mirror(args.source_root, args.mirror_root, reset=bool(args.reset))
        payload = mirror.to_manifest()
    elif args.command == "fetch":
        mirror = materialize_files(args.source_root, args.mirror_root, list(args.path or []))
        payload = mirror.to_manifest()
    elif args.command == "acquire":
        mirror = acquire_relevant_files(
            args.source_root,
            args.mirror_root,
            instruction=str(args.instruction),
            candidate_paths=list(args.candidate or []),
            limit=int(args.limit),
        )
        payload = mirror.to_manifest()
    elif args.command == "manifest":
        mirror = open_mirror(args.source_root, args.mirror_root)
        payload = mirror.to_manifest()
    elif args.command == "boundary":
        payload = execution_boundary_report(
            backend=str(args.backend),
            docker_image=str(args.docker_image),
            vm_provider=str(args.vm_provider),
            vm_name=str(args.vm_name),
            vm_host=str(args.vm_host),
            vm_workdir=str(args.vm_workdir),
            vm_network_mode=str(args.vm_network_mode),
        )
    elif args.command == "vm":
        if str(args.operation) == "report":
            payload = vm_manager_report(
                vm_provider=str(args.vm_provider),
                vm_name=str(args.vm_name),
                vm_host=str(args.vm_host),
                vm_workdir=str(args.vm_workdir),
                vm_network_mode=str(args.vm_network_mode),
            )
        else:
            if not str(args.mirror_root or "").strip():
                raise MirrorScopeError("conos mirror vm requires --mirror-root for lifecycle operations")
            payload = manage_vm_workspace(
                args.source_root,
                args.mirror_root,
                operation=str(args.operation),
                timeout_seconds=int(args.timeout),
                vm_provider=str(args.vm_provider),
                vm_name=str(args.vm_name),
                vm_host=str(args.vm_host),
                vm_workdir=str(args.vm_workdir),
                vm_network_mode=str(args.vm_network_mode),
                checkpoint_id=str(args.checkpoint_id),
            )
    elif args.command == "vm-sync":
        payload = sync_vm_workspace(
            args.source_root,
            args.mirror_root,
            direction=str(args.direction),
            timeout_seconds=int(args.timeout),
            vm_provider=str(args.vm_provider),
            vm_name=str(args.vm_name),
            vm_host=str(args.vm_host),
            vm_workdir=str(args.vm_workdir),
            vm_network_mode=str(args.vm_network_mode),
        )
    elif args.command == "exec":
        raw_command = list(args.exec_args or [])
        if raw_command and raw_command[0] == "--":
            raw_command = raw_command[1:]
        result = run_mirror_command(
            args.source_root,
            args.mirror_root,
            raw_command,
            allowed_commands=list(args.allow_command or []) or None,
            timeout_seconds=int(args.timeout),
            backend=str(args.backend),
            docker_image=str(args.docker_image),
            vm_provider=str(args.vm_provider),
            vm_name=str(args.vm_name),
            vm_host=str(args.vm_host),
            vm_workdir=str(args.vm_workdir),
            vm_network_mode=str(args.vm_network_mode),
            vm_sync_mode=str(args.vm_sync_mode),
        )
        payload = result.to_dict()
    elif args.command == "plan":
        payload = build_sync_plan(args.source_root, args.mirror_root)
    elif args.command == "apply":
        payload = apply_sync_plan(
            args.source_root,
            args.mirror_root,
            plan_id=str(args.plan_id),
            approved_by=str(args.approved_by),
        )
    elif args.command == "rollback":
        payload = rollback_sync_plan(
            args.source_root,
            args.mirror_root,
            plan_id=str(args.plan_id),
        )
    else:
        parser.print_help()
        return 0
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

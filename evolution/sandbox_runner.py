from __future__ import annotations

import os
import platform
import resource
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from evolution.proposal_generator import PatchProposal


def _string_list(values: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _utf8_len(text: str) -> int:
    return len(str(text or "").encode("utf-8", errors="ignore"))


def _truncate_text(text: str, *, max_bytes: int) -> tuple[str, bool]:
    raw = str(text or "")
    if max_bytes <= 0:
        return "", bool(raw)
    encoded = raw.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return raw, False
    clipped = encoded[: max(0, max_bytes)]
    decoded = clipped.decode("utf-8", errors="ignore")
    return decoded, True


@dataclass(frozen=True)
class WorkspaceSecretLease:
    lease_id: str
    secret_id: str
    env_var: str
    required: bool = True
    command_indices: List[int] = field(default_factory=list)
    task_ref: str = ""
    max_uses: int = 0
    expires_at_command_index: int = -1

    def to_dict(self) -> Dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "secret_id": self.secret_id,
            "env_var": self.env_var,
            "required": bool(self.required),
            "command_indices": list(self.command_indices),
            "task_ref": self.task_ref,
            "max_uses": int(self.max_uses),
            "expires_at_command_index": int(self.expires_at_command_index),
        }


@dataclass(frozen=True)
class WorkspaceSecretBroker:
    secrets: Dict[str, str] = field(default_factory=dict)
    broker_source: str = "inline"

    def resolve(self, *, secret_id: str) -> Optional[str]:
        key = str(secret_id or "").strip()
        if not key:
            return None
        if key not in self.secrets:
            return None
        return str(self.secrets[key])


@dataclass(frozen=True)
class SandboxCommandResult:
    command: str
    passed: bool
    returncode: int
    command_index: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    blocked: bool = False
    block_reason: str = ""
    output_bytes: int = 0
    os_sandbox_applied: bool = False
    os_sandbox_mode: str = "none"
    secret_lease_ids: List[str] = field(default_factory=list)
    secret_env_keys: List[str] = field(default_factory=list)
    secret_missing_ids: List[str] = field(default_factory=list)
    write_policy_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "command": self.command,
            "command_index": int(self.command_index),
            "passed": bool(self.passed),
            "returncode": int(self.returncode),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": float(self.duration_sec),
            "timed_out": bool(self.timed_out),
            "stdout_truncated": bool(self.stdout_truncated),
            "stderr_truncated": bool(self.stderr_truncated),
            "blocked": bool(self.blocked),
            "block_reason": self.block_reason,
            "output_bytes": int(self.output_bytes),
            "os_sandbox_applied": bool(self.os_sandbox_applied),
            "os_sandbox_mode": self.os_sandbox_mode,
            "secret_lease_ids": list(self.secret_lease_ids),
            "secret_env_keys": list(self.secret_env_keys),
            "secret_missing_ids": list(self.secret_missing_ids),
            "write_policy_violations": list(self.write_policy_violations),
        }


@dataclass(frozen=True)
class SandboxRunResult:
    proposal_id: str
    workspace_dir: str
    prepared_files: List[str] = field(default_factory=list)
    command_results: List[SandboxCommandResult] = field(default_factory=list)
    runner_name: str = "workspace_runner"
    isolation_mode: str = "workspace_copy"
    secure_isolation: bool = False
    security_boundary: str = "best_effort_workspace_only"
    network_mode: str = "best_effort_disabled"
    os_sandbox_available: bool = False
    os_sandbox_active: bool = False
    os_sandbox_mode: str = "none"
    secret_broker_used: bool = False
    secret_broker_source: str = "none"
    secret_lease_ids: List[str] = field(default_factory=list)
    secret_env_keys: List[str] = field(default_factory=list)
    secret_missing_ids: List[str] = field(default_factory=list)
    write_policy_mode: str = "declared_targets_only"
    allowed_write_paths: List[str] = field(default_factory=list)
    write_policy_violations: List[str] = field(default_factory=list)
    resource_limits: Dict[str, int] = field(default_factory=dict)
    quarantine_dir: str = ""
    quarantined_artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=lambda: ["workspace_runner_is_not_a_security_sandbox"])
    workspace_file_count: int = 0
    workspace_bytes: int = 0

    @property
    def all_passed(self) -> bool:
        return bool(self.command_results) and all(item.passed for item in self.command_results)

    def to_dict(self) -> Dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "workspace_dir": self.workspace_dir,
            "prepared_files": list(self.prepared_files),
            "command_results": [item.to_dict() for item in self.command_results],
            "runner_name": self.runner_name,
            "isolation_mode": self.isolation_mode,
            "secure_isolation": bool(self.secure_isolation),
            "security_boundary": self.security_boundary,
            "network_mode": self.network_mode,
            "os_sandbox_available": bool(self.os_sandbox_available),
            "os_sandbox_active": bool(self.os_sandbox_active),
            "os_sandbox_mode": self.os_sandbox_mode,
            "secret_broker_used": bool(self.secret_broker_used),
            "secret_broker_source": self.secret_broker_source,
            "secret_lease_ids": list(self.secret_lease_ids),
            "secret_env_keys": list(self.secret_env_keys),
            "secret_missing_ids": list(self.secret_missing_ids),
            "write_policy_mode": self.write_policy_mode,
            "allowed_write_paths": list(self.allowed_write_paths),
            "write_policy_violations": list(self.write_policy_violations),
            "resource_limits": dict(self.resource_limits),
            "quarantine_dir": self.quarantine_dir,
            "quarantined_artifacts": list(self.quarantined_artifacts),
            "warnings": list(self.warnings),
            "workspace_file_count": int(self.workspace_file_count),
            "workspace_bytes": int(self.workspace_bytes),
            "all_passed": bool(self.all_passed),
        }


@dataclass(frozen=True)
class WorkspaceResourceLimits:
    cpu_seconds: int = 10
    file_size_bytes: int = 2 * 1024 * 1024
    open_files: int = 128
    # Disabled by default: hard per-user process caps are brittle on shared developer hosts.
    processes: int = 0
    # Disabled by default: hard address-space caps proved too fragile for Python bootstrap.
    address_space_bytes: int = 0
    data_segment_bytes: int = 256 * 1024 * 1024
    stack_bytes: int = 64 * 1024 * 1024
    core_bytes: int = 0
    workspace_files: int = 2048
    workspace_bytes: int = 16 * 1024 * 1024

    def to_dict(self) -> Dict[str, int]:
        processes = int(self.processes or 0)
        address_space_bytes = int(self.address_space_bytes or 0)
        return {
            "cpu_seconds": int(max(1, self.cpu_seconds)),
            "file_size_bytes": int(max(256, self.file_size_bytes)),
            "open_files": int(max(16, self.open_files)),
            "processes": 0 if processes <= 0 else int(max(1, processes)),
            "address_space_bytes": 0 if address_space_bytes <= 0 else int(max(64 * 1024 * 1024, address_space_bytes)),
            "data_segment_bytes": int(max(32 * 1024 * 1024, self.data_segment_bytes)),
            "stack_bytes": int(max(8 * 1024 * 1024, self.stack_bytes)),
            "core_bytes": int(max(0, self.core_bytes)),
            "workspace_files": int(max(1, self.workspace_files)),
            "workspace_bytes": int(max(1024, self.workspace_bytes)),
        }


class WorkspaceRunner:
    """Best-effort isolated workspace runner. This is not a security sandbox."""

    RUNNER_NAME = "workspace_runner"
    ISOLATION_MODE = "workspace_copy"
    SECURITY_BOUNDARY = "best_effort_workspace_only"
    OS_SANDBOX_SECURITY_BOUNDARY = "best_effort_workspace_plus_os_no_network"
    SECURITY_WARNING = "workspace_runner_is_not_a_security_sandbox"
    OS_SANDBOX_WARNING = "workspace_runner_os_no_network_unavailable"
    OS_SANDBOX_RUNTIME_WARNING = "workspace_runner_os_no_network_runtime_fallback"
    SECRET_WARNING = "workspace_runner_required_secret_unavailable"
    OS_SANDBOX_MODE = "darwin_sandbox_exec_no_network"
    WRITE_POLICY_MODE = "declared_targets_only"
    QUARANTINE_DIRNAME = ".workspace-runner-artifacts"
    NETWORK_DISABLED_COMMANDS = frozenset(
        {
            "curl",
            "wget",
            "ftp",
            "sftp",
            "scp",
            "ssh",
            "rsync",
            "telnet",
            "nc",
            "ncat",
            "netcat",
            "ping",
            "dig",
            "nslookup",
            "host",
            "traceroute",
            "tracepath",
        }
    )
    SAFE_ENV_OVERRIDE_KEYS = frozenset(
        {
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PATH",
            "PYTEST_ADDOPTS",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONHASHSEED",
            "PYTHONPATH",
        }
    )
    RESERVED_ENV_KEYS = frozenset(
        {
            "HOME",
            "TMPDIR",
            "TMP",
            "TEMP",
            "PATH",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "WORKSPACE_RUNNER_SECURITY_BOUNDARY",
            "WORKSPACE_RUNNER_NETWORK_MODE",
        }
    )
    QUARANTINE_PATTERNS = (
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        "runtime",
        ".sandbox-home",
        ".sandbox-tmp",
    )
    DEFAULT_ALLOWED_WRITE_PATHS = (
        "runtime",
        ".sandbox-home",
        ".sandbox-tmp",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        ".workspace-runner-artifacts",
    )

    def __init__(
        self,
        *,
        timeout_sec: float = 30.0,
        network_mode: str = "best_effort_disabled",
        max_output_bytes: int = 64 * 1024,
        command_allowlist: Optional[Sequence[str]] = None,
        command_denylist: Optional[Sequence[str]] = None,
        resource_limits: Optional[WorkspaceResourceLimits] = None,
    ) -> None:
        self._timeout_sec = max(1.0, float(timeout_sec or 30.0))
        self._network_mode = str(network_mode or "best_effort_disabled").strip() or "best_effort_disabled"
        self._max_output_bytes = max(256, int(max_output_bytes or 64 * 1024))
        self._command_allowlist = {item.casefold() for item in _string_list(command_allowlist or [])}
        denylist = set(item.casefold() for item in _string_list(command_denylist or []))
        if self._network_mode != "enabled":
            denylist.update(self.NETWORK_DISABLED_COMMANDS)
        self._command_denylist = denylist
        self._resource_limits = resource_limits or WorkspaceResourceLimits()

    @staticmethod
    def _normalize_relative_path(path: str) -> Path:
        raw = str(path or "").replace("\\", "/").strip()
        if not raw:
            raise ValueError("sandbox path must not be empty")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise ValueError(f"sandbox path must be relative: {raw}")
        parts: List[str] = []
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError(f"sandbox path escapes workspace: {raw}")
            parts.append(part)
        if not parts:
            raise ValueError("sandbox path must not resolve to the workspace root")
        return Path(*parts)

    @staticmethod
    def _resolved_within(root: Path, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _command_program(command: str) -> str:
        parts = shlex.split(str(command or "").strip())
        if not parts:
            return ""
        return Path(parts[0]).name.casefold()

    def _normalize_policy_paths(self, values: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            rel = self._normalize_relative_path(text)
            rel_s = rel.as_posix()
            if rel_s not in normalized:
                normalized.append(rel_s)
        return normalized

    def _allowed_write_paths(self, *, proposal: PatchProposal) -> list[str]:
        metadata = dict(proposal.metadata or {})
        declared = list(proposal.target_files) + list(proposal.file_overrides.keys())
        declared.extend(self.DEFAULT_ALLOWED_WRITE_PATHS)
        declared.extend(list(metadata.get("workspace_write_allowlist") or []))
        return sorted(set(self._normalize_policy_paths(declared)))

    def _normalize_secret_env_var(self, value: str) -> str:
        env_var = str(value or "").strip()
        if not env_var:
            raise ValueError("workspace runner secret lease env_var must not be empty")
        if not env_var.replace("_", "").isalnum():
            raise ValueError(f"workspace runner secret lease env_var must be alnum/underscore: {env_var}")
        if env_var.startswith("WORKSPACE_RUNNER_") or env_var in self.RESERVED_ENV_KEYS:
            raise ValueError(f"workspace runner secret lease env_var is reserved: {env_var}")
        return env_var

    def _normalize_command_indices(self, values: object) -> list[int]:
        if values is None or values == "":
            return []
        raw_values = values if isinstance(values, (list, tuple, set)) else [values]
        indices: list[int] = []
        for raw_value in raw_values:
            try:
                index = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"workspace runner secret lease command index must be int-like: {raw_value}") from exc
            if index < 0:
                raise ValueError(f"workspace runner secret lease command index must be >= 0: {index}")
            if index not in indices:
                indices.append(index)
        return sorted(indices)

    def _normalize_secret_max_uses(self, value: object) -> int:
        if value is None or value == "":
            return 0
        try:
            max_uses = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"workspace runner secret lease max_uses must be int-like: {value}") from exc
        if max_uses < 0:
            raise ValueError(f"workspace runner secret lease max_uses must be >= 0: {max_uses}")
        return max_uses

    def _normalize_secret_expiry_command_index(self, value: object) -> int:
        if value is None or value == "":
            return -1
        try:
            command_index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"workspace runner secret lease expires_at_command_index must be int-like: {value}"
            ) from exc
        if command_index < 0:
            raise ValueError(
                f"workspace runner secret lease expires_at_command_index must be >= 0: {command_index}"
            )
        return command_index

    def _coerce_secret_leases(self, *, proposal: PatchProposal) -> list[WorkspaceSecretLease]:
        metadata = dict(proposal.metadata or {})
        raw_leases = list(metadata.get("secret_leases") or [])
        leases: list[WorkspaceSecretLease] = []
        for index, raw_item in enumerate(raw_leases):
            if not isinstance(raw_item, dict):
                raise ValueError("workspace runner secret_leases must contain dict entries")
            secret_id = str(raw_item.get("secret_id") or "").strip()
            if not secret_id:
                raise ValueError("workspace runner secret lease secret_id must not be empty")
            env_var = self._normalize_secret_env_var(str(raw_item.get("env_var") or ""))
            lease_id = str(raw_item.get("lease_id") or f"lease_{index}_{secret_id}").strip()
            required = bool(raw_item.get("required", True))
            command_indices = self._normalize_command_indices(
                raw_item.get("command_indices", raw_item.get("command_index"))
            )
            task_ref = str(raw_item.get("task_ref") or "").strip()
            max_uses = self._normalize_secret_max_uses(raw_item.get("max_uses"))
            expires_at_command_index = self._normalize_secret_expiry_command_index(
                raw_item.get("expires_at_command_index")
            )
            leases.append(
                WorkspaceSecretLease(
                    lease_id=lease_id,
                    secret_id=secret_id,
                    env_var=env_var,
                    required=required,
                    command_indices=command_indices,
                    task_ref=task_ref,
                    max_uses=max_uses,
                    expires_at_command_index=expires_at_command_index,
                )
            )
        return leases

    def _resolve_secret_broker(
        self,
        *,
        secret_broker: Optional[WorkspaceSecretBroker | Dict[str, str]],
    ) -> Optional[WorkspaceSecretBroker]:
        if secret_broker is None:
            return None
        if isinstance(secret_broker, WorkspaceSecretBroker):
            return secret_broker
        if isinstance(secret_broker, dict):
            return WorkspaceSecretBroker(secrets={str(key): str(value) for key, value in secret_broker.items()})
        raise ValueError("workspace runner secret_broker must be a WorkspaceSecretBroker or dict")

    def _active_secret_leases(
        self,
        *,
        leases: Sequence[WorkspaceSecretLease],
        command_index: int,
        task_ref: str,
    ) -> list[WorkspaceSecretLease]:
        active: list[WorkspaceSecretLease] = []
        for lease in list(leases):
            if lease.command_indices and command_index not in lease.command_indices:
                continue
            if lease.task_ref and lease.task_ref != str(task_ref or "").strip():
                continue
            active.append(lease)
        return active

    def _lease_is_usable(
        self,
        *,
        lease: WorkspaceSecretLease,
        command_index: int,
        lease_use_counts: Dict[str, int],
    ) -> bool:
        if lease.expires_at_command_index >= 0 and command_index > lease.expires_at_command_index:
            return False
        if lease.max_uses > 0 and int(lease_use_counts.get(lease.lease_id, 0) or 0) >= lease.max_uses:
            return False
        return True

    def _resolve_secret_leases(
        self,
        *,
        leases: Sequence[WorkspaceSecretLease],
        secret_broker: Optional[WorkspaceSecretBroker],
        command_index: int,
        lease_use_counts: Dict[str, int],
    ) -> tuple[Dict[str, str], list[str], list[str]]:
        injected_env: Dict[str, str] = {}
        missing_ids: list[str] = []
        if not leases:
            return injected_env, missing_ids, []
        for lease in list(leases):
            if not self._lease_is_usable(
                lease=lease,
                command_index=command_index,
                lease_use_counts=lease_use_counts,
            ):
                if lease.required:
                    missing_ids.append(lease.secret_id)
                continue
            secret_value = secret_broker.resolve(secret_id=lease.secret_id) if secret_broker is not None else None
            if secret_value is None:
                if lease.required:
                    missing_ids.append(lease.secret_id)
                continue
            injected_env[lease.env_var] = secret_value
        return injected_env, sorted(set(missing_ids)), [lease.lease_id for lease in leases]

    def _copy_source_path(self, *, source_root: Path, workspace_root: Path, rel_path: str) -> Optional[str]:
        normalized = self._normalize_relative_path(rel_path)
        source_path = source_root / normalized
        if not source_path.exists():
            return None
        if not self._resolved_within(source_root, source_path):
            raise ValueError(f"requested source path escapes source root: {rel_path}")
        destination = workspace_root / normalized
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_symlink():
            raise ValueError(f"symlinked source paths are not allowed in workspace runner: {rel_path}")
        if source_path.is_dir():
            shutil.copytree(
                source_path,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "runtime"),
            )
        else:
            shutil.copy2(source_path, destination)
        return normalized.as_posix()

    def _validate_env_overrides(self, env: Optional[Dict[str, str]]) -> Dict[str, str]:
        if env is None:
            return {}
        if not isinstance(env, dict):
            raise ValueError("workspace runner env overrides must be a dict")
        overrides: Dict[str, str] = {}
        for raw_key, raw_value in env.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if key not in self.SAFE_ENV_OVERRIDE_KEYS and not key.startswith("WORKSPACE_RUNNER_"):
                raise ValueError(f"workspace runner env override not allowed: {key}")
            overrides[key] = str(raw_value)
        return overrides

    def _build_effective_env(
        self,
        *,
        workspace_dir: Path,
        env: Optional[Dict[str, str]],
        leased_secret_env: Optional[Dict[str, str]] = None,
        os_sandbox_active: Optional[bool] = None,
    ) -> Dict[str, str]:
        sandbox_home = workspace_dir / ".sandbox-home"
        sandbox_tmp = workspace_dir / ".sandbox-tmp"
        sandbox_home.mkdir(parents=True, exist_ok=True)
        sandbox_tmp.mkdir(parents=True, exist_ok=True)
        if os_sandbox_active is None:
            _, os_sandbox_active, _ = self._resolve_os_sandbox_prefix()
        effective_env: Dict[str, str] = {}
        for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
            value = os.environ.get(key)
            if value:
                effective_env[key] = value
        effective_env.update(
            {
                "HOME": str(sandbox_home),
                "TMPDIR": str(sandbox_tmp),
                "TMP": str(sandbox_tmp),
                "TEMP": str(sandbox_tmp),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "WORKSPACE_RUNNER_SECURITY_BOUNDARY": (
                    self.OS_SANDBOX_SECURITY_BOUNDARY if os_sandbox_active else self.SECURITY_BOUNDARY
                ),
                "WORKSPACE_RUNNER_NETWORK_MODE": self._network_mode,
            }
        )
        if self._network_mode != "enabled":
            for key in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "all_proxy",
                "FTP_PROXY",
                "ftp_proxy",
            ):
                effective_env.pop(key, None)
            effective_env["NO_PROXY"] = "*"
            effective_env["no_proxy"] = "*"
            effective_env["GIT_TERMINAL_PROMPT"] = "0"
            effective_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        effective_env.update(self._validate_env_overrides(env))
        effective_env.update({str(key): str(value) for key, value in dict(leased_secret_env or {}).items()})
        return effective_env

    def run_patch_proposal(
        self,
        *,
        source_dir: str,
        proposal: PatchProposal,
        commands: Optional[Sequence[str]] = None,
        include_paths: Sequence[str] = (),
        env: Optional[Dict[str, str]] = None,
        secret_broker: Optional[WorkspaceSecretBroker | Dict[str, str]] = None,
        task_ref: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> SandboxRunResult:
        source_root = Path(source_dir).resolve()
        workspace_root = Path(tempfile.mkdtemp(prefix="workspace-runner-"))
        command_timeout = max(1.0, float(timeout_sec or self._timeout_sec))
        validated_env = self._validate_env_overrides(env)
        allowed_write_paths = self._allowed_write_paths(proposal=proposal)
        secret_leases = self._coerce_secret_leases(proposal=proposal)
        resolved_secret_broker = self._resolve_secret_broker(secret_broker=secret_broker)
        effective_task_ref = str(task_ref or dict(proposal.metadata or {}).get("task_ref") or "").strip()
        os_sandbox_prefix, os_sandbox_available, os_sandbox_mode = self._resolve_os_sandbox_prefix()
        os_sandbox_active = os_sandbox_available
        if os_sandbox_available:
            probe_env = self._build_effective_env(
                workspace_dir=workspace_root,
                env=validated_env,
                os_sandbox_active=True,
            )
            os_sandbox_active = self._probe_os_sandbox_prefix(
                workspace_dir=workspace_root,
                effective_env=probe_env,
                timeout_sec=command_timeout,
                os_sandbox_prefix=os_sandbox_prefix,
            )
            if not os_sandbox_active:
                os_sandbox_prefix = []
                os_sandbox_mode = "none"
        prepared_files = self._prepare_workspace(
            source_root=source_root,
            workspace_root=workspace_root,
            proposal=proposal,
            include_paths=include_paths,
        )
        self._apply_file_overrides(workspace_root=workspace_root, proposal=proposal)
        command_results, os_sandbox_runtime_degraded = self._run_commands(
            workspace_dir=workspace_root,
            commands=list(commands if commands is not None else proposal.commands),
            env=validated_env,
            secret_leases=secret_leases,
            secret_broker=resolved_secret_broker,
            task_ref=effective_task_ref,
            timeout_sec=command_timeout,
            os_sandbox_prefix=os_sandbox_prefix,
            os_sandbox_mode=os_sandbox_mode,
            os_sandbox_active=os_sandbox_active,
            allowed_write_paths=allowed_write_paths,
        )
        workspace_file_count, workspace_bytes = self._workspace_metrics(workspace_root)
        quarantine_dir, quarantined_artifacts = self._quarantine_workspace_artifacts(workspace_root)
        write_policy_violations = sorted(
            {
                path
                for result in command_results
                for path in list(result.write_policy_violations or [])
            }
        )
        secret_env_keys = sorted(
            {
                env_key
                for result in command_results
                for env_key in list(result.secret_env_keys or [])
            }
        )
        secret_missing_ids = sorted(
            {
                secret_id
                for result in command_results
                for secret_id in list(result.secret_missing_ids or [])
            }
        )
        warnings = [self.SECURITY_WARNING]
        if self._network_mode != "enabled" and platform.system() == "Darwin" and not os_sandbox_active:
            warnings.append(self.OS_SANDBOX_WARNING)
        if os_sandbox_runtime_degraded:
            os_sandbox_active = False
            os_sandbox_mode = "none"
            warnings.append(self.OS_SANDBOX_RUNTIME_WARNING)
        if secret_missing_ids:
            warnings.append(self.SECRET_WARNING)
        return SandboxRunResult(
            proposal_id=proposal.proposal_id,
            workspace_dir=str(workspace_root),
            prepared_files=prepared_files,
            command_results=command_results,
            runner_name=self.RUNNER_NAME,
            isolation_mode=self.ISOLATION_MODE,
            secure_isolation=False,
            security_boundary=self.OS_SANDBOX_SECURITY_BOUNDARY if os_sandbox_active else self.SECURITY_BOUNDARY,
            network_mode=self._network_mode,
            os_sandbox_available=os_sandbox_available,
            os_sandbox_active=os_sandbox_active,
            os_sandbox_mode=os_sandbox_mode,
            secret_broker_used=bool(secret_leases),
            secret_broker_source=resolved_secret_broker.broker_source if resolved_secret_broker is not None else "none",
            secret_lease_ids=[lease.lease_id for lease in secret_leases],
            secret_env_keys=secret_env_keys,
            secret_missing_ids=secret_missing_ids,
            write_policy_mode=self.WRITE_POLICY_MODE,
            allowed_write_paths=allowed_write_paths,
            write_policy_violations=write_policy_violations,
            resource_limits=self._resource_limits.to_dict(),
            quarantine_dir=str(quarantine_dir) if quarantine_dir is not None else "",
            quarantined_artifacts=quarantined_artifacts,
            warnings=warnings,
            workspace_file_count=workspace_file_count,
            workspace_bytes=workspace_bytes,
        )

    def _prepare_workspace(
        self,
        *,
        source_root: Path,
        workspace_root: Path,
        proposal: PatchProposal,
        include_paths: Sequence[str],
    ) -> List[str]:
        requested_paths = list(proposal.target_files) + [str(path) for path in list(include_paths or [])]
        prepared: List[str] = []
        copied_any = False
        for rel_path in requested_paths:
            copied_rel = self._copy_source_path(
                source_root=source_root,
                workspace_root=workspace_root,
                rel_path=rel_path,
            )
            if copied_rel is None:
                continue
            copied_any = True
            prepared.append(copied_rel)

        if not copied_any:
            for child in source_root.iterdir():
                if child.name in {".git", "__pycache__", ".pytest_cache", "runtime"}:
                    continue
                if child.is_symlink():
                    continue
                copied_rel = self._copy_source_path(
                    source_root=source_root,
                    workspace_root=workspace_root,
                    rel_path=child.name,
                )
                if copied_rel is not None:
                    prepared.append(copied_rel)

        return sorted(set(prepared))

    def _apply_file_overrides(self, *, workspace_root: Path, proposal: PatchProposal) -> None:
        for rel_path, content in proposal.file_overrides.items():
            normalized = self._normalize_relative_path(rel_path)
            target = workspace_root / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")

    def _blocked_command_result(
        self,
        *,
        command: str,
        reason: str,
        started: float,
        command_index: int = -1,
        secret_lease_ids: Optional[Sequence[str]] = None,
        secret_env_keys: Optional[Sequence[str]] = None,
        secret_missing_ids: Optional[Sequence[str]] = None,
    ) -> SandboxCommandResult:
        elapsed = time.perf_counter() - started
        return SandboxCommandResult(
            command=command,
            command_index=command_index,
            passed=False,
            returncode=126,
            stdout="",
            stderr="",
            duration_sec=elapsed,
            timed_out=False,
            blocked=True,
            block_reason=reason,
            output_bytes=0,
            secret_lease_ids=list(secret_lease_ids or []),
            secret_env_keys=list(secret_env_keys or []),
            secret_missing_ids=list(secret_missing_ids or []),
        )

    def _resolve_os_sandbox_prefix(self) -> tuple[list[str], bool, str]:
        if self._network_mode == "enabled":
            return [], False, "none"
        if platform.system() != "Darwin":
            return [], False, "none"
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            return [], False, "none"
        return [sandbox_exec, "-n", "no-network"], True, self.OS_SANDBOX_MODE

    def _probe_os_sandbox_prefix(
        self,
        *,
        workspace_dir: Path,
        effective_env: Dict[str, str],
        timeout_sec: float,
        os_sandbox_prefix: Sequence[str],
    ) -> bool:
        if not os_sandbox_prefix:
            return False
        try:
            completed = subprocess.run(
                [*list(os_sandbox_prefix), "/usr/bin/true"],
                cwd=str(workspace_dir),
                env=effective_env,
                capture_output=True,
                text=True,
                timeout=max(1.0, min(float(timeout_sec or 1.0), 5.0)),
            )
        except Exception:
            return False
        return int(completed.returncode) == 0

    def _resource_preexec_fn(self):
        limits = self._resource_limits.to_dict()

        def _apply_limits() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_size_bytes"], limits["file_size_bytes"]))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
            except Exception:
                pass
            if int(limits["processes"] or 0) > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (limits["processes"], limits["processes"]))
                except Exception:
                    pass
            if int(limits["address_space_bytes"] or 0) > 0:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (limits["address_space_bytes"], limits["address_space_bytes"]))
                except Exception:
                    pass
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (limits["data_segment_bytes"], limits["data_segment_bytes"]))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_STACK, (limits["stack_bytes"], limits["stack_bytes"]))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (limits["core_bytes"], limits["core_bytes"]))
            except Exception:
                pass

        return _apply_limits

    def _truncate_command_output(self, stdout: str, stderr: str) -> tuple[str, str, bool, bool, int]:
        total_bytes = _utf8_len(stdout) + _utf8_len(stderr)
        if total_bytes <= self._max_output_bytes:
            return stdout, stderr, False, False, total_bytes
        stdout_budget = max(0, self._max_output_bytes // 2)
        stderr_budget = max(0, self._max_output_bytes - stdout_budget)
        trimmed_stdout, stdout_truncated = _truncate_text(stdout, max_bytes=stdout_budget)
        trimmed_stderr, stderr_truncated = _truncate_text(stderr, max_bytes=stderr_budget)
        if stdout_truncated:
            trimmed_stdout += "\n[workspace_runner stdout truncated]"
        if stderr_truncated:
            trimmed_stderr += "\n[workspace_runner stderr truncated]"
        return trimmed_stdout, trimmed_stderr, stdout_truncated, stderr_truncated, total_bytes

    def _workspace_metrics(self, workspace_dir: Path) -> tuple[int, int]:
        file_count = 0
        total_bytes = 0
        for candidate in workspace_dir.rglob("*"):
            if not candidate.exists():
                continue
            if candidate.name == self.QUARANTINE_DIRNAME or self.QUARANTINE_DIRNAME in candidate.parts:
                continue
            if candidate.is_file():
                file_count += 1
                try:
                    total_bytes += int(candidate.stat().st_size or 0)
                except OSError:
                    continue
        return file_count, total_bytes

    def _workspace_quota_violation(self, workspace_dir: Path) -> str:
        file_count, total_bytes = self._workspace_metrics(workspace_dir)
        limits = self._resource_limits.to_dict()
        reasons: list[str] = []
        if file_count > limits["workspace_files"]:
            reasons.append(f"workspace_files={file_count}>{limits['workspace_files']}")
        if total_bytes > limits["workspace_bytes"]:
            reasons.append(f"workspace_bytes={total_bytes}>{limits['workspace_bytes']}")
        return ",".join(reasons)

    def _workspace_state(self, workspace_dir: Path) -> Dict[str, tuple[str, int, int]]:
        state: Dict[str, tuple[str, int, int]] = {}
        for candidate in workspace_dir.rglob("*"):
            if not candidate.exists():
                continue
            rel_path = candidate.relative_to(workspace_dir)
            if candidate.name == self.QUARANTINE_DIRNAME or self.QUARANTINE_DIRNAME in rel_path.parts:
                continue
            try:
                stat = candidate.lstat()
            except OSError:
                continue
            kind = "dir" if candidate.is_dir() else "symlink" if candidate.is_symlink() else "file"
            size = 0 if kind == "dir" else int(stat.st_size or 0)
            state[rel_path.as_posix()] = (kind, size, int(stat.st_mtime_ns or 0))
        return state

    @staticmethod
    def _path_within_allowed_write_scope(path: str, allowed_write_paths: Sequence[str]) -> bool:
        candidate = Path(path)
        for allowed in list(allowed_write_paths or []):
            try:
                candidate.relative_to(Path(allowed))
                return True
            except ValueError:
                continue
        return False

    def _workspace_write_policy_violations(
        self,
        *,
        before: Dict[str, tuple[str, int, int]],
        after: Dict[str, tuple[str, int, int]],
        allowed_write_paths: Sequence[str],
    ) -> list[str]:
        changed_paths = sorted(
            path
            for path in set(before.keys()) | set(after.keys())
            if before.get(path) != after.get(path)
        )
        return [
            path
            for path in changed_paths
            if not self._path_within_allowed_write_scope(path, allowed_write_paths)
        ]

    @staticmethod
    def _looks_like_os_sandbox_runtime_fault(*, returncode: int, stderr: str) -> bool:
        if int(returncode) < 0:
            return True
        lowered = str(stderr or "").strip().lower()
        if not lowered:
            return False
        return any(
            marker in lowered
            for marker in (
                "sandbox-exec:",
                "sandbox_apply:",
                "bad file descriptor",
                "tokio",
                "operation not permitted",
            )
        )

    def _run_commands(
        self,
        *,
        workspace_dir: Path,
        commands: Sequence[str],
        env: Optional[Dict[str, str]],
        secret_leases: Sequence[WorkspaceSecretLease],
        secret_broker: Optional[WorkspaceSecretBroker],
        task_ref: str,
        timeout_sec: Optional[float],
        os_sandbox_prefix: Sequence[str],
        os_sandbox_mode: str,
        os_sandbox_active: bool,
        allowed_write_paths: Sequence[str],
    ) -> tuple[List[SandboxCommandResult], bool]:
        command_results: List[SandboxCommandResult] = []
        command_timeout = max(1.0, float(timeout_sec or self._timeout_sec))
        current_os_sandbox_prefix = list(os_sandbox_prefix)
        current_os_sandbox_mode = str(os_sandbox_mode or "none")
        current_os_sandbox_active = bool(os_sandbox_active and current_os_sandbox_prefix)
        os_sandbox_runtime_degraded = False
        workspace_state = self._workspace_state(workspace_dir)
        lease_use_counts: Dict[str, int] = {}

        for command_index, command in enumerate(list(commands or [])):
            command_s = str(command or "").strip()
            if not command_s:
                continue
            started = time.perf_counter()
            try:
                command_args = shlex.split(command_s)
                if not command_args:
                    continue
                program = Path(command_args[0]).name.casefold()
                active_secret_leases = self._active_secret_leases(
                    leases=secret_leases,
                    command_index=command_index,
                    task_ref=task_ref,
                )
                leased_secret_env: Dict[str, str] = {}
                secret_missing_ids: List[str] = []
                active_secret_lease_ids: List[str] = []
                if self._command_allowlist and program not in self._command_allowlist:
                    command_results.append(
                        self._blocked_command_result(
                            command=command_s,
                            reason=f"command_not_allowlisted:{program}",
                            started=started,
                            command_index=command_index,
                            secret_lease_ids=active_secret_lease_ids,
                            secret_env_keys=sorted(leased_secret_env.keys()),
                            secret_missing_ids=secret_missing_ids,
                        )
                    )
                    continue
                if program in self._command_denylist:
                    reason = (
                        f"network_disabled_command:{program}"
                        if self._network_mode != "enabled" and program in self.NETWORK_DISABLED_COMMANDS
                        else f"command_denied:{program}"
                    )
                    command_results.append(
                        self._blocked_command_result(
                            command=command_s,
                            reason=reason,
                            started=started,
                            command_index=command_index,
                            secret_lease_ids=active_secret_lease_ids,
                            secret_env_keys=sorted(leased_secret_env.keys()),
                            secret_missing_ids=secret_missing_ids,
                        )
                    )
                    continue
                leased_secret_env, secret_missing_ids, active_secret_lease_ids = self._resolve_secret_leases(
                    leases=active_secret_leases,
                    secret_broker=secret_broker,
                    command_index=command_index,
                    lease_use_counts=lease_use_counts,
                )
                if secret_missing_ids:
                    command_results.append(
                        self._blocked_command_result(
                            command=command_s,
                            reason=f"required_secret_unavailable:{','.join(secret_missing_ids)}",
                            started=started,
                            command_index=command_index,
                            secret_lease_ids=active_secret_lease_ids,
                            secret_env_keys=sorted(leased_secret_env.keys()),
                            secret_missing_ids=secret_missing_ids,
                        )
                    )
                    break
                effective_env = self._build_effective_env(
                    workspace_dir=workspace_dir,
                    env=env,
                    leased_secret_env=leased_secret_env,
                    os_sandbox_active=current_os_sandbox_active,
                )
                completed = subprocess.run(
                    [*current_os_sandbox_prefix, *command_args],
                    cwd=str(workspace_dir),
                    env=effective_env,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                    preexec_fn=self._resource_preexec_fn() if os.name == "posix" else None,
                )
                command_os_sandbox_applied = bool(current_os_sandbox_prefix)
                command_os_sandbox_mode = current_os_sandbox_mode if command_os_sandbox_applied else "none"
                if command_os_sandbox_applied and self._looks_like_os_sandbox_runtime_fault(
                    returncode=int(completed.returncode),
                    stderr=str(completed.stderr or ""),
                ):
                    fallback_env = self._build_effective_env(
                        workspace_dir=workspace_dir,
                        env=env,
                        leased_secret_env=leased_secret_env,
                        os_sandbox_active=False,
                    )
                    completed = subprocess.run(
                        command_args,
                        cwd=str(workspace_dir),
                        env=fallback_env,
                        capture_output=True,
                        text=True,
                        timeout=command_timeout,
                        preexec_fn=self._resource_preexec_fn() if os.name == "posix" else None,
                    )
                    command_os_sandbox_applied = False
                    command_os_sandbox_mode = "none"
                    current_os_sandbox_prefix = []
                    current_os_sandbox_mode = "none"
                    current_os_sandbox_active = False
                    os_sandbox_runtime_degraded = True
                for lease_id in active_secret_lease_ids:
                    lease_use_counts[lease_id] = int(lease_use_counts.get(lease_id, 0) or 0) + 1
                elapsed = time.perf_counter() - started
                stdout, stderr, stdout_truncated, stderr_truncated, output_bytes = self._truncate_command_output(
                    completed.stdout,
                    completed.stderr,
                )
                next_workspace_state = self._workspace_state(workspace_dir)
                write_policy_violations = self._workspace_write_policy_violations(
                    before=workspace_state,
                    after=next_workspace_state,
                    allowed_write_paths=allowed_write_paths,
                )
                workspace_state = next_workspace_state
                quota_violation = self._workspace_quota_violation(workspace_dir)
                block_reason = ""
                blocked = False
                if write_policy_violations:
                    blocked = True
                    block_reason = f"workspace_write_policy_violation:{','.join(write_policy_violations)}"
                elif quota_violation:
                    blocked = True
                    block_reason = f"workspace_quota_exceeded:{quota_violation}"
                command_results.append(
                    SandboxCommandResult(
                        command=command_s,
                        command_index=command_index,
                        passed=completed.returncode == 0 and not blocked,
                        returncode=int(completed.returncode),
                        stdout=stdout,
                        stderr=stderr,
                        duration_sec=elapsed,
                        timed_out=False,
                        stdout_truncated=stdout_truncated,
                        stderr_truncated=stderr_truncated,
                        blocked=blocked,
                        block_reason=block_reason,
                        output_bytes=output_bytes,
                        os_sandbox_applied=command_os_sandbox_applied,
                        os_sandbox_mode=command_os_sandbox_mode,
                        secret_lease_ids=active_secret_lease_ids,
                        secret_env_keys=sorted(leased_secret_env.keys()),
                        secret_missing_ids=secret_missing_ids,
                        write_policy_violations=write_policy_violations,
                    )
                )
                if blocked:
                    break
            except FileNotFoundError as exc:
                elapsed = time.perf_counter() - started
                command_results.append(
                    SandboxCommandResult(
                        command=command_s,
                        command_index=command_index,
                        passed=False,
                        returncode=127,
                        stdout="",
                        stderr=str(exc),
                        duration_sec=elapsed,
                        timed_out=False,
                        os_sandbox_applied=bool(current_os_sandbox_prefix),
                        os_sandbox_mode=current_os_sandbox_mode if current_os_sandbox_prefix else "none",
                        secret_lease_ids=active_secret_lease_ids,
                        secret_env_keys=sorted(leased_secret_env.keys()),
                        secret_missing_ids=secret_missing_ids,
                    )
                )
            except ValueError as exc:
                elapsed = time.perf_counter() - started
                command_results.append(
                    SandboxCommandResult(
                        command=command_s,
                        command_index=command_index,
                        passed=False,
                        returncode=2,
                        stdout="",
                        stderr=str(exc),
                        duration_sec=elapsed,
                        timed_out=False,
                        os_sandbox_applied=bool(current_os_sandbox_prefix),
                        os_sandbox_mode=current_os_sandbox_mode if current_os_sandbox_prefix else "none",
                        secret_lease_ids=active_secret_lease_ids,
                        secret_env_keys=sorted(leased_secret_env.keys()),
                        secret_missing_ids=secret_missing_ids,
                    )
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                stdout, stderr, stdout_truncated, stderr_truncated, output_bytes = self._truncate_command_output(
                    str(exc.stdout or ""),
                    str(exc.stderr or ""),
                )
                command_results.append(
                    SandboxCommandResult(
                        command=command_s,
                        command_index=command_index,
                        passed=False,
                        returncode=124,
                        stdout=stdout,
                        stderr=stderr,
                        duration_sec=elapsed,
                        timed_out=True,
                        stdout_truncated=stdout_truncated,
                        stderr_truncated=stderr_truncated,
                        output_bytes=output_bytes,
                        os_sandbox_applied=bool(current_os_sandbox_prefix),
                        os_sandbox_mode=current_os_sandbox_mode if current_os_sandbox_prefix else "none",
                        secret_lease_ids=active_secret_lease_ids,
                        secret_env_keys=sorted(leased_secret_env.keys()),
                        secret_missing_ids=secret_missing_ids,
                    )
                )

        return command_results, os_sandbox_runtime_degraded

    def _quarantine_workspace_artifacts(self, workspace_dir: Path) -> tuple[Path, list[str]]:
        quarantine_root = workspace_dir / self.QUARANTINE_DIRNAME
        quarantine_root.mkdir(parents=True, exist_ok=True)
        quarantined: list[str] = []
        for candidate in sorted(workspace_dir.rglob("*")):
            if not candidate.exists():
                continue
            if candidate == quarantine_root or quarantine_root in candidate.parents:
                continue
            if candidate.name not in self.QUARANTINE_PATTERNS:
                continue
            rel_path = candidate.relative_to(workspace_dir)
            destination = quarantine_root / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(destination))
            quarantined.append(rel_path.as_posix())
        return quarantine_root, quarantined


class SandboxRunner(WorkspaceRunner):
    """Compatibility alias for older call sites. Not a security sandbox."""

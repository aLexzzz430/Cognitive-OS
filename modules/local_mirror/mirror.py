from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, Iterable, Sequence


LOCAL_MIRROR_VERSION = "conos.local_mirror/v1"
LOCAL_MIRROR_SYNC_PLAN_VERSION = "conos.local_mirror_sync_plan/v1"
CONTROL_DIR_NAME = "control"
WORKSPACE_DIR_NAME = "workspace"
DEFAULT_ALLOWED_COMMANDS = frozenset({"python", "python3"})
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


class MirrorScopeError(ValueError):
    """Raised when a requested source path is outside the declared mirror scope."""


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalMirror:
    source_root: Path
    mirror_root: Path
    workspace_root: Path
    control_root: Path
    materialized_files: Dict[str, MaterializedFile] = field(default_factory=dict)
    audit_events: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.control_root / "manifest.json"

    @property
    def sync_plan_path(self) -> Path:
        return self.control_root / "sync_plan.json"

    def workspace_files(self) -> list[Path]:
        if not self.workspace_root.exists():
            return []
        return sorted(path for path in self.workspace_root.rglob("*") if path.is_file())

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
    return LocalMirror(
        source_root=source_root.resolve(),
        mirror_root=mirror_root.resolve(),
        workspace_root=workspace_root.resolve(),
        control_root=control_root.resolve(),
        materialized_files=materialized,
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
        source_file, relative = _resolve_source_file(mirror.source_root, requested_path)
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
) -> MirrorCommandResult:
    mirror = open_mirror(source_root, mirror_root)
    cmd = [str(part) for part in list(command) if str(part)]
    if not cmd:
        raise MirrorScopeError("mirror command is empty")
    executable_name = Path(cmd[0]).name
    allowed = {Path(item).name for item in (allowed_commands or DEFAULT_ALLOWED_COMMANDS)}
    if executable_name not in allowed:
        raise MirrorScopeError(f"command is not allowlisted for mirror execution: {executable_name}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=mirror.workspace_root,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        result = MirrorCommandResult(
            command=cmd,
            returncode=int(completed.returncode),
            stdout=str(completed.stdout or ""),
            stderr=str(completed.stderr or ""),
            timeout_seconds=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        result = MirrorCommandResult(
            command=cmd,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "") + "\nmirror command timed out",
            timeout_seconds=max(1, int(timeout_seconds)),
        )
    mirror.audit_events.append(
        _event(
            "mirror_command_executed",
            command=cmd,
            executable=executable_name,
            returncode=result.returncode,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
            workspace_root=str(mirror.workspace_root),
            sandbox_label="best_effort_local_mirror",
            not_os_security_sandbox=True,
        )
    )
    mirror.save_manifest()
    return result


def _workspace_relative_path(mirror: LocalMirror, path: Path) -> str:
    relative = path.resolve().relative_to(mirror.workspace_root.resolve()).as_posix()
    return _safe_relative_path(relative).as_posix()


def _text_patch(source_path: Path, mirror_path: Path, relative: str, *, max_bytes: int = 128 * 1024) -> str:
    try:
        source_size = source_path.stat().st_size if source_path.exists() else 0
        mirror_size = mirror_path.stat().st_size if mirror_path.exists() else 0
    except OSError:
        return ""
    if max(source_size, mirror_size) > max_bytes:
        return ""
    try:
        source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
        mirror_text = mirror_path.read_text(encoding="utf-8") if mirror_path.exists() else ""
    except UnicodeDecodeError:
        return ""
    return "\n".join(
        difflib.unified_diff(
            source_text.splitlines(),
            mirror_text.splitlines(),
            fromfile=f"source/{relative}",
            tofile=f"mirror/{relative}",
            lineterm="",
        )
    )


def compute_mirror_diff(source_root: str | Path, mirror_root: str | Path) -> list[MirrorDiffEntry]:
    mirror = open_mirror(source_root, mirror_root)
    entries: Dict[str, MirrorDiffEntry] = {}
    for workspace_file in mirror.workspace_files():
        relative = _workspace_relative_path(mirror, workspace_file)
        source_file = (mirror.source_root / relative).resolve()
        mirror_sha = _sha256(workspace_file)
        source_sha = _sha256(source_file) if source_file.exists() and source_file.is_file() else ""
        if not source_file.exists():
            status = "added"
        elif source_sha == mirror_sha:
            status = "unchanged"
        else:
            status = "modified"
        entries[relative] = MirrorDiffEntry(
            relative_path=relative,
            status=status,
            source_path=str(source_file),
            mirror_path=str(workspace_file),
            source_sha256=source_sha,
            mirror_sha256=mirror_sha,
            size_bytes=int(workspace_file.stat().st_size),
            text_patch=_text_patch(source_file, workspace_file, relative),
        )
    for relative, materialized in mirror.materialized_files.items():
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
            "mode": "added_or_modified_files_only",
            "deletions_supported": False,
            "requires_plan_id": True,
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
        )
    )
    mirror.save_manifest()
    return plan


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

    synced: list[Dict[str, Any]] = []
    source_hash_checks: list[Dict[str, Any]] = []
    for row in list(plan.get("actionable_changes", []) or []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "") or "")
        if status not in {"added", "modified"}:
            continue
        relative = _safe_relative_path(str(row.get("relative_path", "") or "")).as_posix()
        mirror_file = (mirror.workspace_root / relative).resolve()
        source_file = (mirror.source_root / relative).resolve()
        planned_source_sha = str(row.get("source_sha256", "") or "")
        current_source_sha = _sha256(source_file) if source_file.exists() and source_file.is_file() else ""
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
        source_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mirror_file, source_file)
        synced.append(
            {
                "relative_path": relative,
                "status": status,
                "source_path": str(source_file),
                "sha256": _sha256(source_file),
            }
        )
    mirror.audit_events.append(
        _event(
            "sync_plan_applied",
            plan_id=str(plan_id),
            approved_by=approver,
            synced_files=synced,
            source_hash_checks=source_hash_checks,
        )
    )
    mirror.save_manifest()
    return {
        "schema_version": "conos.local_mirror_sync_result/v1",
        "plan_id": str(plan_id),
        "approved_by": approver,
        "source_hash_checks": source_hash_checks,
        "synced_files": synced,
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

    exec_parser = subparsers.add_parser("exec", help="Run an allowlisted command inside the mirror workspace.")
    exec_parser.add_argument("--source-root", default=".")
    exec_parser.add_argument("--mirror-root", required=True)
    exec_parser.add_argument("--timeout", type=int, default=30)
    exec_parser.add_argument("--allow-command", action="append", default=[])
    exec_parser.add_argument("exec_args", nargs=argparse.REMAINDER)

    plan_parser = subparsers.add_parser("plan", help="Build a reviewed sync plan from mirror changes.")
    plan_parser.add_argument("--source-root", default=".")
    plan_parser.add_argument("--mirror-root", required=True)

    apply_parser = subparsers.add_parser("apply", help="Apply an approved sync plan to the source root.")
    apply_parser.add_argument("--source-root", default=".")
    apply_parser.add_argument("--mirror-root", required=True)
    apply_parser.add_argument("--plan-id", required=True)
    apply_parser.add_argument("--approved-by", required=True, choices=["human", "machine"])

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
    else:
        parser.print_help()
        return 0
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

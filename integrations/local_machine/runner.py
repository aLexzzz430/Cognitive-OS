from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from core.main_loop import CoreMainLoop
from core.runtime.long_run_supervisor import LongRunSupervisor
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter
from modules.llm import build_llm_client


def _default_mirror_root(run_id: str | None) -> str:
    safe = str(run_id or "local-machine-task").strip() or "local-machine-task"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in safe)
    return str(Path("runtime") / "mirrors" / safe)


def summarize_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    final_raw = dict(audit.get("final_surface_raw", {}) or {})
    mirror = dict(final_raw.get("local_mirror", {}) or {})
    sync_plan = dict(mirror.get("sync_plan", {}) or {})
    approval = dict(sync_plan.get("approval", {}) or {})
    artifact_check = dict(audit.get("local_machine_artifact_check", {}) or {})
    return {
        "run_id": str(audit.get("run_id", "") or ""),
        "target": "local-machine",
        "world_provider_source": str(audit.get("world_provider_source", "") or ""),
        "total_reward": float(audit.get("total_reward", 0.0) or 0.0),
        "instruction": str(audit.get("local_machine_instruction", "") or ""),
        "source_root": str(mirror.get("source_root", "") or ""),
        "mirror_root": str(mirror.get("mirror_root", "") or ""),
        "workspace_file_count": int(mirror.get("workspace_file_count", 0) or 0),
        "materialized_files": [
            row.get("relative_path", "")
            for row in list(mirror.get("materialized_files", []) or [])
            if isinstance(row, dict)
        ],
        "sync_plan_id": str(sync_plan.get("plan_id", "") or ""),
        "sync_plan_status": str(approval.get("status", "") or ""),
        "actionable_change_count": len(list(sync_plan.get("actionable_changes", []) or [])),
        "command_executed": bool(mirror.get("command_executed", False)),
        "final_terminal": bool(audit.get("final_surface_terminal", False)),
        "artifact_check_ok": bool(artifact_check.get("ok", True)),
        "artifact_check_failures": list(artifact_check.get("failures", []) or []),
        "llm_provider": str(audit.get("llm_provider", "") or "none"),
        "llm_base_url": str(audit.get("llm_base_url", "") or ""),
        "llm_model": str(audit.get("llm_model", "") or ""),
        "llm_mode": str(audit.get("llm_mode", "") or ""),
    }


def _latest_mirror_command_returncode(mirror: Dict[str, Any]) -> int | None:
    latest: int | None = None
    for event in list(mirror.get("audit_events", []) or []):
        if not isinstance(event, dict) or event.get("event_type") != "mirror_command_executed":
            continue
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        try:
            latest = int(payload.get("returncode"))
        except (TypeError, ValueError):
            latest = None
    return latest


def _workspace_glob_matches(mirror: Dict[str, Any], pattern: str) -> list[str]:
    raw_pattern = str(pattern or "").strip()
    if not raw_pattern:
        return []
    path = Path(raw_pattern)
    if path.is_absolute() or ".." in path.parts:
        return []
    workspace_root = str(mirror.get("workspace_root", "") or "")
    if not workspace_root:
        mirror_root = str(mirror.get("mirror_root", "") or "")
        workspace_root = str(Path(mirror_root) / "workspace") if mirror_root else ""
    if not workspace_root:
        return []
    root = Path(workspace_root)
    return sorted(str(match.relative_to(root)) for match in root.glob(raw_pattern) if match.is_file())


def _artifact_contract_check(
    audit: Dict[str, Any],
    *,
    daemon: bool = False,
    required_workspace_paths: Sequence[str] = (),
) -> Dict[str, Any]:
    final_raw = dict(audit.get("final_surface_raw", {}) or {})
    mirror = dict(final_raw.get("local_mirror", {}) or {})
    sync_plan = dict(mirror.get("sync_plan", {}) or {})
    actionable_changes = list(sync_plan.get("actionable_changes", []) or [])
    supervisor_state = dict(audit.get("long_run_supervisor", {}) or {})
    supervisor_run = dict(supervisor_state.get("run", {}) or {})
    checks = {
        "command_executed": bool(mirror.get("command_executed", False)),
        "workspace_has_files": int(mirror.get("workspace_file_count", 0) or 0) > 0,
        "sync_plan_present": bool(sync_plan.get("plan_id", "")),
        "actionable_changes_present": len(actionable_changes) > 0,
    }
    latest_returncode = _latest_mirror_command_returncode(mirror)
    if bool(mirror.get("command_executed", False)):
        checks["latest_command_succeeded"] = latest_returncode == 0
    required_path_matches: Dict[str, list[str]] = {}
    for pattern in list(required_workspace_paths or []):
        matches = _workspace_glob_matches(mirror, pattern)
        required_path_matches[str(pattern)] = matches
        checks[f"required_workspace_path:{pattern}"] = bool(matches)
    if daemon:
        checks["daemon_waiting_approval"] = str(supervisor_run.get("status", "") or "") == "WAITING_APPROVAL"
    failures = [name for name, ok in checks.items() if not bool(ok)]
    return {
        "schema_version": "conos.local_machine_artifact_contract/v1",
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "latest_command_returncode": latest_returncode,
        "required_workspace_path_matches": required_path_matches,
    }


def run_local_machine_task(
    *,
    instruction: str,
    source_root: str = ".",
    mirror_root: str | None = None,
    candidate_paths: Sequence[str] = (),
    fetch_paths: Sequence[str] = (),
    default_command: Sequence[str] | str | None = None,
    allowed_commands: Sequence[str] = (),
    agent_id: str = "cognitive_os",
    run_id: Optional[str] = None,
    max_episodes: int = 1,
    max_ticks_per_episode: int = 3,
    seed: int = 0,
    verbose: bool = False,
    reset_mirror: bool = True,
    terminal_after_plan: bool = True,
    expose_apply_tool: bool = False,
    llm_client: Any = None,
    llm_provider: str = "none",
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_timeout: float = 60.0,
    llm_mode: str = "integrated",
    daemon: bool = False,
    supervisor_db: str | None = None,
    allow_empty_exec: bool = False,
    require_artifacts: bool = False,
    required_artifact_paths: Sequence[str] = (),
    default_command_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    resolved_run_id = run_id or "local-machine-task"
    resolved_llm_client = llm_client
    if resolved_llm_client is None:
        resolved_llm_client = build_llm_client(
            llm_provider,
            base_url=llm_base_url,
            model=llm_model,
            timeout_sec=llm_timeout,
        )
    supervisor: LongRunSupervisor | None = None
    supervisor_task_id = ""
    if daemon:
        supervisor = LongRunSupervisor(db_path=supervisor_db or "runtime/long_run/state.sqlite3")
        existing = supervisor.state_store.get_run(resolved_run_id)
        if not existing:
            supervisor.create_run(instruction, run_id=resolved_run_id)
        supervisor_task_id = supervisor.add_task(
            resolved_run_id,
            instruction,
            priority=0,
            verifier={"kind": "local_machine_daemon", "requires_approval_on": "mirror_plan"},
        )
        terminal_after_plan = False
    world = LocalMachineSurfaceAdapter(
        instruction=instruction,
        source_root=source_root,
        mirror_root=mirror_root or _default_mirror_root(resolved_run_id),
        candidate_paths=candidate_paths,
        fetch_paths=fetch_paths,
        default_command=default_command,
        allowed_commands=allowed_commands or (),
        reset_mirror=reset_mirror,
        terminal_after_plan=terminal_after_plan,
        expose_apply_tool=expose_apply_tool,
        allow_empty_exec=allow_empty_exec,
        default_command_timeout_seconds=default_command_timeout_seconds,
        task_id=resolved_run_id,
    )
    loop = CoreMainLoop(
        agent_id=agent_id,
        run_id=resolved_run_id,
        seed=seed,
        max_episodes=max_episodes,
        max_ticks_per_episode=max_ticks_per_episode,
        verbose=verbose,
        world_adapter=world,
        llm_client=resolved_llm_client,
        llm_mode=llm_mode,
        world_provider_source="integrations.local_machine.runner",
    )
    audit = loop.run()
    final_observation = world.observe()
    task_spec = world.get_generic_task_spec()
    audit["run_id"] = resolved_run_id
    audit["local_machine_task_id"] = task_spec.task_id
    audit["local_machine_instruction"] = task_spec.instruction
    audit["local_machine_task_metadata"] = dict(task_spec.metadata)
    audit["llm_provider"] = str(llm_provider or "none")
    audit["llm_base_url"] = str(llm_base_url or "")
    audit["llm_model"] = str(llm_model or "")
    audit["llm_mode"] = str(llm_mode or "")
    audit["final_surface_structured"] = dict(final_observation.structured or {})
    audit["final_surface_terminal"] = bool(final_observation.terminal)
    audit["final_surface_raw"] = dict(final_observation.raw or {})
    artifact_check: Dict[str, Any] = {}
    if require_artifacts:
        artifact_check = _artifact_contract_check(
            audit,
            daemon=False,
            required_workspace_paths=list(required_artifact_paths or ()),
        )
        audit["local_machine_artifact_check"] = artifact_check
    if supervisor is not None:
        final_mirror = dict(audit["final_surface_raw"].get("local_mirror", {}) or {})
        sync_plan = dict(final_mirror.get("sync_plan", {}) or {})
        approval_request = {}
        latest_returncode = _latest_mirror_command_returncode(final_mirror)
        command_failed = bool(final_mirror.get("command_executed", False)) and latest_returncode != 0

        def _mark_supervisor_failed(reason: str, result: Dict[str, Any]) -> None:
            if supervisor_task_id:
                supervisor.state_store.update_task_status(supervisor_task_id, "FAILED", result=result)
            supervisor.state_store.update_run_status(resolved_run_id, "FAILED", paused_reason=reason)

        if require_artifacts and artifact_check and not bool(artifact_check.get("ok", False)):
            _mark_supervisor_failed(
                "artifact_contract_failed:" + ",".join(list(artifact_check.get("failures", []) or [])),
                {"artifact_check": artifact_check},
            )
        elif command_failed:
            _mark_supervisor_failed(
                f"mirror_command_failed:returncode={latest_returncode}",
                {"latest_command_returncode": latest_returncode, "sync_plan": sync_plan},
            )
        elif sync_plan and not bool(final_mirror.get("applied", False)):
            approval = dict(sync_plan.get("approval", {}) or {})
            actionable_count = len(list(sync_plan.get("actionable_changes", []) or []))
            approval_request = {
                "type": "local_mirror_sync_plan",
                "plan_id": str(sync_plan.get("plan_id", "") or ""),
                "approval_status": str(approval.get("status", "") or ""),
                "actionable_change_count": actionable_count,
                "source_root": str(final_mirror.get("source_root", "") or source_root),
                "mirror_root": str(final_mirror.get("mirror_root", "") or mirror_root or _default_mirror_root(resolved_run_id)),
            }
            if actionable_count > 0:
                supervisor.state_store.update_task_status(supervisor_task_id, "RUNNING")
                supervisor.mark_waiting_approval(resolved_run_id, approval_request)
            else:
                supervisor.state_store.update_task_status(supervisor_task_id, "COMPLETED", result={"sync_plan": sync_plan})
                supervisor.state_store.update_run_status(resolved_run_id, "COMPLETED")
        else:
            _mark_supervisor_failed(
                "local_machine_no_sync_plan",
                {"final_surface_terminal": bool(final_observation.terminal), "final_mirror": final_mirror},
            )
        audit["long_run_supervisor"] = {
            "run": supervisor.state_store.get_run(resolved_run_id),
            "latest_approval": supervisor.state_store.get_latest_approval(resolved_run_id),
            "approval_request": approval_request,
        }
        if require_artifacts and bool(daemon) and not artifact_check.get("ok", False):
            audit["long_run_supervisor"] = {
                "run": supervisor.state_store.get_run(resolved_run_id),
                "latest_approval": supervisor.state_store.get_latest_approval(resolved_run_id),
                "approval_request": approval_request,
            }
        elif require_artifacts and bool(daemon):
            daemon_artifact_check = _artifact_contract_check(
                audit,
                daemon=True,
                required_workspace_paths=list(required_artifact_paths or ()),
            )
            audit["local_machine_artifact_check"] = daemon_artifact_check
            if not bool(daemon_artifact_check.get("ok", False)):
                _mark_supervisor_failed(
                    "artifact_contract_failed:" + ",".join(list(daemon_artifact_check.get("failures", []) or [])),
                    {"artifact_check": daemon_artifact_check},
                )
                audit["long_run_supervisor"] = {
                    "run": supervisor.state_store.get_run(resolved_run_id),
                    "latest_approval": supervisor.state_store.get_latest_approval(resolved_run_id),
                    "approval_request": approval_request,
                }
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conos run local-machine",
        description="Run Cognitive OS against an empty-first local machine mirror.",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--mirror-root", default=None)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--fetch-path", action="append", default=[])
    parser.add_argument(
        "--default-command",
        default=None,
        help="Optional command string to execute inside the mirror after files are materialized.",
    )
    parser.add_argument("--allow-command", action="append", default=[])
    parser.add_argument("--agent-id", default="cognitive_os")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminal-after-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expose-apply-tool", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="none",
        choices=["none", "minimax", "ollama"],
        help="Optional LLM provider. Use ollama with --llm-base-url for a LAN-hosted local model.",
    )
    parser.add_argument("--llm-base-url", default=None, help="Ollama base URL, e.g. http://192.168.1.23:11434.")
    parser.add_argument("--llm-model", default=None, help="Ollama model name, e.g. qwen3:8b.")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="LLM HTTP timeout in seconds.")
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="integrated",
        choices=["integrated", "shadow", "analyst", "final_candidate"],
    )
    parser.add_argument("--daemon", action="store_true", help="Use LongRunSupervisor state and wait for approval after mirror_plan.")
    parser.add_argument("--supervisor-db", default=None, help="SQLite state DB for daemon mode. Defaults to runtime/long_run/state.sqlite3.")
    parser.add_argument("--allow-empty-exec", action="store_true", help="Allow the default command to run before source files are materialized.")
    parser.add_argument("--require-artifacts", action="store_true", help="Fail if no command execution, generated files, sync plan, or actionable changes are produced.")
    parser.add_argument(
        "--require-artifact-path",
        action="append",
        default=[],
        help="Require a file glob, relative to the mirror workspace, to exist before artifact checks pass. Repeatable.",
    )
    parser.add_argument("--default-command-timeout", type=int, default=30, help="Timeout in seconds for the configured default command.")
    parser.add_argument("--save-audit", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit = run_local_machine_task(
        instruction=args.instruction,
        source_root=args.source_root,
        mirror_root=args.mirror_root,
        candidate_paths=list(args.candidate or []),
        fetch_paths=list(args.fetch_path or []),
        default_command=args.default_command,
        allowed_commands=list(args.allow_command or []),
        agent_id=args.agent_id,
        run_id=args.run_id,
        max_episodes=int(args.max_episodes),
        max_ticks_per_episode=int(args.max_ticks),
        seed=int(args.seed),
        verbose=bool(args.verbose),
        reset_mirror=bool(args.reset_mirror),
        terminal_after_plan=bool(args.terminal_after_plan),
        expose_apply_tool=bool(args.expose_apply_tool),
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_timeout=float(args.llm_timeout),
        llm_mode=args.llm_mode,
        daemon=bool(args.daemon),
        supervisor_db=args.supervisor_db,
        allow_empty_exec=bool(args.allow_empty_exec),
        require_artifacts=bool(args.require_artifacts),
        required_artifact_paths=list(args.require_artifact_path or []),
        default_command_timeout_seconds=int(args.default_command_timeout),
    )

    print(json.dumps(summarize_audit(audit), indent=2, ensure_ascii=False, default=str))
    if args.save_audit:
        path = Path(args.save_audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"saved_audit={path}")
    artifact_check = dict(audit.get("local_machine_artifact_check", {}) or {})
    if bool(args.require_artifacts) and not bool(artifact_check.get("ok", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

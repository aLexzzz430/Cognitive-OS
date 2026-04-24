from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from core.main_loop import CoreMainLoop
from core.runtime.long_run_supervisor import LongRunSupervisor
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter


def _default_mirror_root(run_id: str | None) -> str:
    safe = str(run_id or "local-machine-task").strip() or "local-machine-task"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in safe)
    return str(Path("runtime") / "mirrors" / safe)


def summarize_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    final_raw = dict(audit.get("final_surface_raw", {}) or {})
    mirror = dict(final_raw.get("local_mirror", {}) or {})
    sync_plan = dict(mirror.get("sync_plan", {}) or {})
    approval = dict(sync_plan.get("approval", {}) or {})
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
        "final_terminal": bool(audit.get("final_surface_terminal", False)),
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
    llm_mode: str = "integrated",
    daemon: bool = False,
    supervisor_db: str | None = None,
) -> Dict[str, Any]:
    resolved_run_id = run_id or "local-machine-task"
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
        llm_client=llm_client,
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
    audit["final_surface_structured"] = dict(final_observation.structured or {})
    audit["final_surface_terminal"] = bool(final_observation.terminal)
    audit["final_surface_raw"] = dict(final_observation.raw or {})
    if supervisor is not None:
        final_mirror = dict(audit["final_surface_raw"].get("local_mirror", {}) or {})
        sync_plan = dict(final_mirror.get("sync_plan", {}) or {})
        approval_request = {}
        if sync_plan and not bool(final_mirror.get("applied", False)):
            approval = dict(sync_plan.get("approval", {}) or {})
            approval_request = {
                "type": "local_mirror_sync_plan",
                "plan_id": str(sync_plan.get("plan_id", "") or ""),
                "approval_status": str(approval.get("status", "") or ""),
                "actionable_change_count": len(list(sync_plan.get("actionable_changes", []) or [])),
                "source_root": str(final_mirror.get("source_root", "") or source_root),
                "mirror_root": str(final_mirror.get("mirror_root", "") or mirror_root or _default_mirror_root(resolved_run_id)),
            }
            supervisor.state_store.update_task_status(supervisor_task_id, "RUNNING")
            supervisor.mark_waiting_approval(resolved_run_id, approval_request)
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
    parser.add_argument("--daemon", action="store_true", help="Use LongRunSupervisor state and wait for approval after mirror_plan.")
    parser.add_argument("--supervisor-db", default=None, help="SQLite state DB for daemon mode. Defaults to runtime/long_run/state.sqlite3.")
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
        daemon=bool(args.daemon),
        supervisor_db=args.supervisor_db,
    )

    print(json.dumps(summarize_audit(audit), indent=2, ensure_ascii=False, default=str))
    if args.save_audit:
        path = Path(args.save_audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"saved_audit={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

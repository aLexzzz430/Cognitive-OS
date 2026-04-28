from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from core.main_loop import CoreMainLoop
from core.runtime_budget import RuntimeBudgetConfig
from core.runtime.end_to_end_learning import END_TO_END_LEARNING_VERSION, EndToEndLearningRuntime
from core.runtime.long_run_supervisor import LongRunSupervisor
from core.runtime.state_store import RuntimeStateStore
from integrations.local_machine.task_adapter import LocalMachineSurfaceAdapter
from modules.llm import (
    LLMCostLedger,
    LLMRuntimeBudget,
    build_llm_client,
    load_profile_backed_route_policies,
    profile_all_configured_models,
    profile_provider_models,
    wrap_with_budget,
    write_model_route_policies,
)


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
    llm_tool_trace = dict(audit.get("local_machine_llm_tool_trace", {}) or {})
    generation_contract = dict(audit.get("local_machine_generation_contract", {}) or {})
    llm_budget = dict(audit.get("llm_budget", {}) or {})
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
        "execution_backend": str(mirror.get("execution_backend", "") or "local"),
        "vm_sync_mode": str(mirror.get("vm_sync_mode", "") or "none"),
        "execution_boundary": dict(mirror.get("execution_boundary", {}) or {}),
        "final_terminal": bool(audit.get("final_surface_terminal", False)),
        "artifact_check_ok": bool(artifact_check.get("ok", True)),
        "artifact_check_failures": list(artifact_check.get("failures", []) or []),
        "llm_provider": str(audit.get("llm_provider", "") or "none"),
        "llm_base_url": str(audit.get("llm_base_url", "") or ""),
        "llm_model": str(audit.get("llm_model", "") or ""),
        "llm_mode": str(audit.get("llm_mode", "") or ""),
        "llm_auto_route_models": bool(audit.get("llm_auto_route_models", False)),
        "llm_profiled_model_count": int(dict(audit.get("llm_model_profile_report", {}) or {}).get("model_count", 0) or 0),
        "llm_call_trace_count": int(llm_tool_trace.get("llm_call_count", 0) or 0),
        "tool_call_trace_count": int(llm_tool_trace.get("tool_call_count", 0) or 0),
        "llm_budget_total_calls": int(llm_budget.get("total_calls", 0) or 0),
        "llm_budget_output_tokens": int(llm_budget.get("output_tokens", 0) or 0),
        "llm_budget_wall_seconds": float(llm_budget.get("wall_seconds", 0.0) or 0.0),
        "strong_model_call_rate": float(llm_budget.get("strong_model_call_rate", 0.0) or 0.0),
        "generation_contract": generation_contract,
        "internet_enabled": bool(mirror.get("internet_enabled", False)),
        "internet_artifact_count": int(dict(mirror.get("internet_ingress", {}) or {}).get("artifact_count", 0) or 0),
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


def _extract_tool_call(action: Dict[str, Any]) -> Dict[str, Any]:
    payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
    tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
    function_name = str(tool_args.get("function_name", action.get("function_name", "")) or "").strip()
    kwargs = tool_args.get("kwargs", action.get("kwargs", {}))
    if not isinstance(kwargs, dict):
        kwargs = {}
    return {
        "function_name": function_name,
        "kwargs": dict(kwargs),
        "strategy": str(dict(action.get("_candidate_meta", {}) or {}).get("structured_answer_strategy", "") or ""),
        "fallback_used": bool(dict(action.get("_candidate_meta", {}) or {}).get("structured_answer_fallback_used", False)),
    }


def _build_llm_tool_trace(audit: Dict[str, Any]) -> Dict[str, Any]:
    llm_calls: list[Dict[str, Any]] = []
    tool_calls: list[Dict[str, Any]] = []
    for entry in list(audit.get("episode_trace", []) or []):
        if not isinstance(entry, dict):
            continue
        action = entry.get("action", {}) if isinstance(entry.get("action", {}), dict) else {}
        tick = int(entry.get("tick", len(tool_calls)) or 0)
        call = _extract_tool_call(action)
        if call.get("function_name"):
            tool_calls.append({
                "tick": tick,
                **call,
            })
        meta = action.get("_candidate_meta", {}) if isinstance(action.get("_candidate_meta", {}), dict) else {}
        for row in list(meta.get("structured_answer_llm_trace", []) or []):
            if not isinstance(row, dict):
                continue
            llm_calls.append({
                "tick": tick,
                "selected_function": call.get("function_name", ""),
                **dict(row),
            })
        outcome = entry.get("outcome", {}) if isinstance(entry.get("outcome", {}), dict) else {}
        for row in list(outcome.get("patch_proposal_llm_trace", []) or []):
            if not isinstance(row, dict):
                continue
            llm_calls.append({
                "tick": tick,
                "selected_function": call.get("function_name", "propose_patch"),
                **dict(row),
            })
    return {
        "schema_version": "conos.local_machine.llm_tool_trace/v1",
        "llm_call_count": len(llm_calls),
        "tool_call_count": len(tool_calls),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
    }


def _read_workspace_texts(mirror: Dict[str, Any], *, max_files: int = 64) -> Dict[str, str]:
    workspace_root = str(mirror.get("workspace_root", "") or "")
    if not workspace_root:
        mirror_root = str(mirror.get("mirror_root", "") or "")
        workspace_root = str(Path(mirror_root) / "workspace") if mirror_root else ""
    if not workspace_root:
        return {}
    root = Path(workspace_root)
    texts: Dict[str, str] = {}
    if not root.exists():
        return texts
    for path in sorted(item for item in root.rglob("*") if item.is_file())[:max_files]:
        if path.name.endswith((".pyc", ".png", ".jpg", ".jpeg", ".gif", ".sqlite", ".db")):
            continue
        try:
            texts[str(path.relative_to(root))] = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return texts


def _market_evidence_reference_report(mirror: Dict[str, Any]) -> Dict[str, Any]:
    ingress = dict(mirror.get("internet_ingress", {}) or {})
    artifacts = [dict(row) for row in list(ingress.get("artifacts", []) or []) if isinstance(row, dict)]
    texts = _read_workspace_texts(mirror)
    corpus = "\n".join(texts.values()).lower()
    tokens: list[str] = []
    for artifact in artifacts:
        for key in ("artifact_id", "normalized_url"):
            value = str(artifact.get(key, "") or "").strip()
            if value:
                tokens.append(value.lower())
    matched = sorted({token for token in tokens if token and token in corpus})
    return {
        "artifact_count": len(artifacts),
        "checked_file_count": len(texts),
        "matched_references": matched,
        "ok": bool(matched),
    }


def _non_template_product_report(mirror: Dict[str, Any]) -> Dict[str, Any]:
    texts = _read_workspace_texts(mirror)
    corpus = "\n".join(texts.values()).lower()
    banned_markers = [
        "signalbrief ai",
        "signalbrief_ai",
        "local-first prompt quality and ai-workflow brief analyzer",
        "research seed captured by con os",
        "brief is clear enough for a first ai pass",
    ]
    matched = [marker for marker in banned_markers if marker in corpus]
    return {
        "checked_file_count": len(texts),
        "matched_template_markers": matched,
        "ok": not matched and bool(texts),
    }


def _normalize_auto_route_provider(provider: str) -> str:
    normalized = str(provider or "all").strip().lower() or "all"
    aliases = {
        "codex": "codex-cli",
        "openai-oauth-codex": "codex-cli",
        "none": "all",
        "off": "all",
        "disabled": "all",
    }
    return aliases.get(normalized, normalized)


def _route_policy_model_count(route_policies: Mapping[str, Any]) -> int:
    models = {
        (
            str(dict(policy or {}).get("provider", "") or ""),
            str(dict(policy or {}).get("base_url", "") or ""),
            str(dict(policy or {}).get("model", "") or ""),
        )
        for policy in dict(route_policies or {}).values()
        if isinstance(policy, Mapping) and str(dict(policy or {}).get("model", "") or "")
    }
    return len(models)


def _loaded_route_policy_report(
    *,
    provider: str,
    base_url: str,
    route_policies: Mapping[str, Any],
    route_policy_source: str,
    store_path: str | None,
) -> Dict[str, Any]:
    return {
        "schema_version": "conos.model_profile_runtime_load/v1",
        "provider": provider,
        "base_url": base_url,
        "model_count": _route_policy_model_count(route_policies),
        "generated_count": 0,
        "reused_count": len(dict(route_policies or {})),
        "store_path": str(store_path or ""),
        "route_policies": dict(route_policies or {}),
        "route_policy_source": route_policy_source,
    }


def _resolve_auto_route_policies(
    *,
    llm_provider: str,
    llm_base_url: str | None,
    llm_model: str | None,
    llm_timeout: float,
    llm_profile_store: str | None,
    llm_profile_force: bool,
    llm_route_policy_file: str | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    profile_provider = _normalize_auto_route_provider(llm_provider)
    if profile_provider not in {"all", "ollama", "openai", "codex-cli"}:
        raise ValueError(
            "llm_auto_route_models supports llm_provider='all', 'ollama', 'openai', 'codex', or 'codex-cli'"
        )
    route_policies: Dict[str, Any] = {}
    if not llm_profile_force:
        route_policies = load_profile_backed_route_policies(
            store_path=llm_profile_store,
            route_policy_path=llm_route_policy_file,
            base_url=llm_base_url,
        )
        if route_policies:
            return route_policies, _loaded_route_policy_report(
                provider=profile_provider,
                base_url=str(llm_base_url or ""),
                route_policies=route_policies,
                route_policy_source=str(llm_route_policy_file or "profile_store/default_route_policy"),
                store_path=llm_profile_store,
            )
    selected_models = [llm_model] if llm_model and profile_provider != "all" else None
    if profile_provider == "all":
        model_profile_report = profile_all_configured_models(
            ollama_base_url=llm_base_url,
            openai_base_url=None,
            timeout_sec=llm_timeout,
            store_path=llm_profile_store,
            force=llm_profile_force,
            include_codex=True,
            catalog_only=True,
        )
    else:
        model_profile_report = profile_provider_models(
            provider=profile_provider,
            base_url=llm_base_url,
            models=selected_models,
            timeout_sec=llm_timeout,
            store_path=llm_profile_store,
            force=llm_profile_force,
            catalog_only=True,
            discover_visible=True,
        )
    route_policies = dict(model_profile_report.get("route_policies", {}) or {})
    if llm_route_policy_file and route_policies:
        write_model_route_policies(route_policies, llm_route_policy_file)
    return route_policies, model_profile_report


def _artifact_contract_check(
    audit: Dict[str, Any],
    *,
    daemon: bool = False,
    required_workspace_paths: Sequence[str] = (),
    require_internet_artifact: bool = False,
    require_llm_generation: bool = False,
    require_market_evidence_reference: bool = False,
    require_non_template_product: bool = False,
) -> Dict[str, Any]:
    final_raw = dict(audit.get("final_surface_raw", {}) or {})
    mirror = dict(final_raw.get("local_mirror", {}) or {})
    sync_plan = dict(mirror.get("sync_plan", {}) or {})
    actionable_changes = list(sync_plan.get("actionable_changes", []) or [])
    supervisor_state = dict(audit.get("long_run_supervisor", {}) or {})
    supervisor_run = dict(supervisor_state.get("run", {}) or {})
    llm_tool_trace = dict(audit.get("local_machine_llm_tool_trace", {}) or {})
    checks = {
        "command_executed": bool(mirror.get("command_executed", False)),
        "workspace_has_files": int(mirror.get("workspace_file_count", 0) or 0) > 0,
        "sync_plan_present": bool(sync_plan.get("plan_id", "")),
        "actionable_changes_present": len(actionable_changes) > 0,
    }
    if require_internet_artifact:
        ingress = dict(mirror.get("internet_ingress", {}) or {})
        checks["internet_artifact_present"] = int(ingress.get("artifact_count", 0) or 0) > 0
    if require_llm_generation:
        tool_calls = [dict(row) for row in list(llm_tool_trace.get("tool_calls", []) or []) if isinstance(row, dict)]
        mirror_exec_llm_calls = [
            row for row in tool_calls
            if row.get("function_name") == "mirror_exec" and row.get("strategy") == "llm_draft"
        ]
        checks["llm_provider_enabled"] = str(audit.get("llm_provider", "") or "none").lower() not in {"", "none", "off", "disabled"}
        checks["llm_model_or_router_specified"] = bool(str(audit.get("llm_model", "") or "").strip()) or bool(audit.get("llm_auto_route_models", False))
        checks["llm_call_trace_present"] = int(llm_tool_trace.get("llm_call_count", 0) or 0) > 0
        checks["tool_call_trace_present"] = int(llm_tool_trace.get("tool_call_count", 0) or 0) > 0
        checks["mirror_exec_kwargs_from_llm"] = bool(mirror_exec_llm_calls)
        checks["deterministic_fallback_disabled"] = not bool(mirror.get("deterministic_fallback_enabled", True))
        checks["deterministic_fallback_not_used"] = not any(bool(row.get("fallback_used", False)) for row in tool_calls)
    market_reference_report: Dict[str, Any] = {}
    if require_market_evidence_reference:
        market_reference_report = _market_evidence_reference_report(mirror)
        checks["market_evidence_referenced"] = bool(market_reference_report.get("ok", False))
    non_template_report: Dict[str, Any] = {}
    if require_non_template_product:
        non_template_report = _non_template_product_report(mirror)
        checks["non_template_product_verifier_passed"] = bool(non_template_report.get("ok", False))
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
        "market_evidence_reference_report": market_reference_report,
        "non_template_product_report": non_template_report,
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
    llm_auto_route_models: bool = False,
    llm_profile_store: str | None = None,
    llm_profile_force: bool = False,
    llm_route_policy_file: str | None = None,
    llm_runtime_budget: Optional[Mapping[str, Any]] = None,
    daemon: bool = False,
    supervisor_db: str | None = None,
    allow_empty_exec: bool = False,
    require_artifacts: bool = False,
    required_artifact_paths: Sequence[str] = (),
    require_internet_artifact: bool = False,
    deterministic_fallback_enabled: bool = True,
    prefer_llm_kwargs: bool = False,
    prefer_llm_patch_proposals: bool = False,
    llm_thinking_mode: str = "auto",
    require_llm_generation: bool = False,
    require_market_evidence_reference: bool = False,
    require_non_template_product: bool = False,
    default_command_timeout_seconds: int = 30,
    execution_backend: str = "local",
    docker_image: str = "python:3.10-slim",
    vm_provider: str = "auto",
    vm_name: str = "",
    vm_host: str = "",
    vm_workdir: str = "/workspace",
    vm_network_mode: str = "provider_default",
    vm_sync_mode: str = "none",
    internet_enabled: bool = False,
    internet_output_root: str | None = None,
    internet_max_bytes: int = 2 * 1024 * 1024,
    internet_timeout_seconds: float = 20.0,
    internet_allow_private_networks: bool = False,
) -> Dict[str, Any]:
    resolved_run_id = run_id or "local-machine-task"
    runtime_budget: RuntimeBudgetConfig | None = None
    model_profile_report: Dict[str, Any] = {}
    if llm_auto_route_models:
        route_policies, model_profile_report = _resolve_auto_route_policies(
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            llm_profile_store=llm_profile_store,
            llm_profile_force=llm_profile_force,
            llm_route_policy_file=llm_route_policy_file,
        )
        runtime_budget = RuntimeBudgetConfig(
            llm_route_policies=route_policies
        )
    resolved_llm_client = llm_client
    if resolved_llm_client is None:
        client_provider = "none" if _normalize_auto_route_provider(llm_provider) == "all" else llm_provider
        resolved_llm_client = build_llm_client(
            client_provider,
            base_url=llm_base_url,
            model=llm_model,
            timeout_sec=llm_timeout,
        )
    llm_budget_config = LLMRuntimeBudget.from_mapping(llm_runtime_budget or {})
    llm_cost_ledger = LLMCostLedger(llm_budget_config)
    if resolved_llm_client is not None:
        resolved_llm_client = wrap_with_budget(resolved_llm_client, llm_cost_ledger)
    supervisor: LongRunSupervisor | None = None
    supervisor_task_id = ""
    effective_supervisor_db = supervisor_db or ("runtime/long_run/state.sqlite3" if daemon else None)
    if daemon:
        supervisor = LongRunSupervisor(db_path=effective_supervisor_db or "runtime/long_run/state.sqlite3")
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
    learning_runtime: EndToEndLearningRuntime | None = None
    if supervisor is not None:
        learning_runtime = EndToEndLearningRuntime(state_store=supervisor.state_store)
    elif supervisor_db:
        learning_runtime = EndToEndLearningRuntime(db_path=supervisor_db)
    learning_context: Dict[str, Any] = {
        "schema_version": END_TO_END_LEARNING_VERSION,
        "task_family": "local_machine",
        "objective_excerpt": str(instruction or "")[:240],
        "lesson_count": 0,
        "lessons": [],
        "hint_text": "",
    }
    if learning_runtime is not None:
        learning_context = learning_runtime.learning_context_for_task(
            task_family="local_machine",
            objective=instruction,
            limit=5,
            mark_used=True,
        )
    learning_hint_text = str(learning_context.get("hint_text", "") or "")
    effective_instruction = str(instruction or "")
    if learning_hint_text:
        effective_instruction = f"{effective_instruction}\n\n{learning_hint_text}"
    learning_env: Dict[str, str] = {}
    if learning_hint_text:
        learning_env["CONOS_LEARNING_HINTS"] = learning_hint_text
        learning_env["CONOS_LEARNING_HINTS_JSON"] = json.dumps(
            list(learning_context.get("lessons", []) or []),
            ensure_ascii=False,
            default=str,
        )
    if learning_context.get("failure_objects"):
        learning_env["CONOS_FAILURE_OBJECTS_JSON"] = json.dumps(
            list(learning_context.get("failure_objects", []) or []),
            ensure_ascii=False,
            default=str,
        )
    world = LocalMachineSurfaceAdapter(
        instruction=effective_instruction,
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
        internet_enabled=internet_enabled,
        internet_output_root=internet_output_root,
        internet_max_bytes=internet_max_bytes,
        internet_timeout_seconds=internet_timeout_seconds,
        internet_allow_private_networks=internet_allow_private_networks,
        deterministic_fallback_enabled=deterministic_fallback_enabled,
        prefer_llm_kwargs=prefer_llm_kwargs,
        prefer_llm_patch_proposals=prefer_llm_patch_proposals,
        llm_thinking_mode=llm_thinking_mode,
        llm_client=resolved_llm_client,
        require_llm_generation=require_llm_generation,
        require_market_evidence_reference=require_market_evidence_reference,
        require_non_template_product=require_non_template_product,
        extra_env=learning_env,
        execution_backend=execution_backend,
        docker_image=docker_image,
        vm_provider=vm_provider,
        vm_name=vm_name,
        vm_host=vm_host,
        vm_workdir=vm_workdir,
        vm_network_mode=vm_network_mode,
        vm_sync_mode=vm_sync_mode,
        learning_context=learning_context,
        evidence_db_path=effective_supervisor_db,
        task_id=resolved_run_id,
        llm_cost_ledger=llm_cost_ledger,
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
        runtime_budget=runtime_budget,
        world_provider_source="integrations.local_machine.runner",
    )
    if prefer_llm_patch_proposals and getattr(world, "llm_client", None) is None:
        try:
            world.llm_client = loop._resolve_llm_client("patch_proposal")
        except Exception:
            world.llm_client = None
    if supervisor is not None:
        loop._formal_evidence_state_store = supervisor.state_store
    elif supervisor_db:
        loop._formal_evidence_state_store = RuntimeStateStore(supervisor_db)
    audit = loop.run()
    final_observation = world.observe()
    task_spec = world.get_generic_task_spec()
    audit["run_id"] = resolved_run_id
    audit["local_machine_task_id"] = task_spec.task_id
    audit["local_machine_instruction"] = task_spec.instruction
    audit["local_machine_original_instruction"] = str(instruction or "")
    audit["local_machine_task_metadata"] = dict(task_spec.metadata)
    audit["end_to_end_learning"] = {"injected": dict(learning_context)}
    audit["llm_provider"] = str(llm_provider or "none")
    audit["llm_base_url"] = str(llm_base_url or "")
    audit["llm_model"] = str(llm_model or "")
    audit["llm_mode"] = str(llm_mode or "")
    audit["llm_auto_route_models"] = bool(llm_auto_route_models)
    audit["local_machine_generation_contract"] = {
        "deterministic_fallback_enabled": bool(deterministic_fallback_enabled),
        "prefer_llm_kwargs": bool(prefer_llm_kwargs),
        "prefer_llm_patch_proposals": bool(prefer_llm_patch_proposals),
        "llm_thinking_mode": str(llm_thinking_mode or "auto"),
        "require_llm_generation": bool(require_llm_generation),
        "require_market_evidence_reference": bool(require_market_evidence_reference),
        "require_non_template_product": bool(require_non_template_product),
    }
    final_state = {}
    try:
        final_state = world._load_investigation_state()
    except Exception:
        final_state = {}
    audit["llm_budget"] = llm_cost_ledger.report(
        verified_success=bool(dict(final_state or {}).get("verified_completion", False))
    )
    if model_profile_report:
        audit["llm_model_profile_report"] = {
            "schema_version": str(model_profile_report.get("schema_version", "") or ""),
            "provider": str(model_profile_report.get("provider", "") or ""),
            "base_url": str(model_profile_report.get("base_url", "") or ""),
            "model_count": int(model_profile_report.get("model_count", 0) or 0),
            "generated_count": int(model_profile_report.get("generated_count", 0) or 0),
            "reused_count": int(model_profile_report.get("reused_count", 0) or 0),
            "store_path": str(model_profile_report.get("store_path", "") or ""),
            "route_policy_source": str(model_profile_report.get("route_policy_source", "") or llm_route_policy_file or ""),
            "route_policy_names": sorted(dict(model_profile_report.get("route_policies", {}) or {}).keys()),
        }
    audit["final_surface_structured"] = dict(final_observation.structured or {})
    audit["final_surface_terminal"] = bool(final_observation.terminal)
    audit["final_surface_raw"] = dict(final_observation.raw or {})
    audit["local_machine_llm_tool_trace"] = _build_llm_tool_trace(audit)
    artifact_check: Dict[str, Any] = {}
    if require_artifacts:
        artifact_check = _artifact_contract_check(
            audit,
            daemon=False,
            required_workspace_paths=list(required_artifact_paths or ()),
            require_internet_artifact=bool(require_internet_artifact),
            require_llm_generation=bool(require_llm_generation),
            require_market_evidence_reference=bool(require_market_evidence_reference),
            require_non_template_product=bool(require_non_template_product),
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
                require_internet_artifact=bool(require_internet_artifact),
                require_llm_generation=bool(require_llm_generation),
                require_market_evidence_reference=bool(require_market_evidence_reference),
                require_non_template_product=bool(require_non_template_product),
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
    if learning_runtime is not None:
        learning_report = learning_runtime.learn_from_local_machine_audit(
            run_id=resolved_run_id,
            instruction=instruction,
            audit=audit,
        )
        e2e = dict(audit.get("end_to_end_learning", {}) or {})
        e2e["recorded"] = learning_report
        audit["end_to_end_learning"] = e2e
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
        choices=["none", "minimax", "ollama", "openai", "codex", "codex-cli", "all"],
        help="Optional LLM provider. Use codex/codex-cli to route through the locally OAuth-authenticated Codex CLI.",
    )
    parser.add_argument("--llm-base-url", default=None, help="Ollama base URL, e.g. http://192.168.1.23:11434.")
    parser.add_argument("--llm-model", default=None, help="Ollama model name, e.g. qwen3:8b.")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="LLM HTTP timeout in seconds.")
    parser.add_argument(
        "--llm-auto-route-models",
        action="store_true",
        help="Profile available provider models and let ModelRouter select models by task route.",
    )
    parser.add_argument("--llm-profile-store", default=None, help="Optional model profile store path.")
    parser.add_argument("--llm-profile-force", action="store_true", help="Regenerate model profiles before routing.")
    parser.add_argument("--llm-route-policy-file", default=None, help="Optional precomputed llm_route_policies JSON path.")
    parser.add_argument("--budget-max-llm-calls", type=int, default=None, help="Maximum LLM calls for this local-machine run.")
    parser.add_argument("--budget-max-prompt-tokens", type=int, default=None, help="Maximum estimated prompt tokens for this run.")
    parser.add_argument("--budget-max-completion-tokens", type=int, default=None, help="Maximum requested completion tokens for this run.")
    parser.add_argument("--budget-max-wall-clock-seconds", type=float, default=None, help="Maximum cumulative LLM wall-clock seconds for this run.")
    parser.add_argument("--budget-max-retry-count", type=int, default=None, help="Maximum retry count recorded in budget metadata.")
    parser.add_argument("--budget-escalation-allowed", action=argparse.BooleanOptionalAction, default=True, help="Whether strong-model escalation is allowed under the run budget.")
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
    parser.add_argument(
        "--require-internet-artifact",
        action="store_true",
        help="Require at least one audited internet_fetch artifact before artifact checks pass.",
    )
    parser.add_argument(
        "--disable-deterministic-fallback",
        action="store_true",
        help="Disable built-in local-machine deterministic kwargs/product-generation fallbacks.",
    )
    parser.add_argument(
        "--prefer-llm-kwargs",
        action="store_true",
        help="Try the configured LLM for structured action kwargs before using deterministic local fallbacks.",
    )
    parser.add_argument(
        "--prefer-llm-patch-proposals",
        action="store_true",
        help="Ask the configured LLM for bounded patch proposals, then keep existing verifier and rollback gates.",
    )
    parser.add_argument(
        "--llm-thinking-mode",
        default="auto",
        choices=["auto", "off", "on"],
        help="Route-level thinking policy: auto uses no-thinking for cheap steps and budgeted thinking for hard steps.",
    )
    parser.add_argument(
        "--require-llm-generation",
        action="store_true",
        help="Require the generated product command to come from an LLM draft and require LLM/tool-call traces.",
    )
    parser.add_argument(
        "--require-market-evidence-reference",
        action="store_true",
        help="Require product files to cite at least one fetched internet artifact id or normalized URL.",
    )
    parser.add_argument(
        "--require-non-template-product",
        action="store_true",
        help="Run the local verifier that rejects known built-in generated-product templates.",
    )
    parser.add_argument(
        "--require-genuine-llm-product",
        action="store_true",
        help="Convenience strict mode: disable deterministic fallback and require LLM generation, market evidence citation, and non-template verifier pass.",
    )
    parser.add_argument("--default-command-timeout", type=int, default=30, help="Timeout in seconds for the configured default command.")
    parser.add_argument(
        "--execution-backend",
        choices=["local", "docker", "vm", "managed-vm"],
        default="local",
        help="Execution backend for mirror_exec and atomic validation commands.",
    )
    parser.add_argument("--docker-image", default="python:3.10-slim", help="Docker image when --execution-backend=docker.")
    parser.add_argument(
        "--vm-provider",
        choices=["auto", "managed", "managed-vm", "lima", "ssh"],
        default="auto",
        help="Real VM provider when --execution-backend=vm.",
    )
    parser.add_argument("--vm-name", default="", help="Lima instance name when --vm-provider=lima.")
    parser.add_argument("--vm-host", default="", help="SSH host when --vm-provider=ssh.")
    parser.add_argument("--vm-workdir", default="/workspace", help="Workspace path inside the VM.")
    parser.add_argument(
        "--vm-network-mode",
        choices=["provider_default", "configured_isolated"],
        default="provider_default",
        help="Declared VM network boundary. Isolation depends on provider configuration.",
    )
    parser.add_argument(
        "--vm-sync-mode",
        choices=["none", "push", "pull", "push-pull"],
        default="none",
        help="When --execution-backend=vm, explicitly push/pull the mirror workspace around execution.",
    )
    parser.add_argument("--internet-enabled", action="store_true", help="Expose the audited generic HTTP/HTTPS internet_fetch tool.")
    parser.add_argument("--internet-output-root", default=None, help="Optional internet artifact store root. Defaults to mirror control/internet.")
    parser.add_argument("--internet-max-bytes", type=int, default=2 * 1024 * 1024, help="Maximum bytes per fetched internet artifact.")
    parser.add_argument("--internet-timeout", type=float, default=20.0, help="HTTP fetch timeout in seconds.")
    parser.add_argument(
        "--internet-allow-private-networks",
        action="store_true",
        help="Allow private/local network URL hosts for internet_fetch. Default blocks them.",
    )
    parser.add_argument("--save-audit", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    strict_generation_contract = (
        bool(args.require_genuine_llm_product)
        or bool(args.require_llm_generation)
        or bool(args.require_market_evidence_reference)
        or bool(args.require_non_template_product)
    )
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
        llm_auto_route_models=bool(args.llm_auto_route_models),
        llm_profile_store=args.llm_profile_store,
        llm_profile_force=bool(args.llm_profile_force),
        llm_route_policy_file=args.llm_route_policy_file,
        llm_runtime_budget={
            "max_llm_calls": args.budget_max_llm_calls,
            "max_prompt_tokens": args.budget_max_prompt_tokens,
            "max_completion_tokens": args.budget_max_completion_tokens,
            "max_wall_clock_seconds": args.budget_max_wall_clock_seconds,
            "max_retry_count": args.budget_max_retry_count,
            "escalation_allowed": bool(args.budget_escalation_allowed),
        },
        daemon=bool(args.daemon),
        supervisor_db=args.supervisor_db,
        allow_empty_exec=bool(args.allow_empty_exec),
        require_artifacts=bool(args.require_artifacts) or strict_generation_contract,
        required_artifact_paths=list(args.require_artifact_path or []),
        require_internet_artifact=bool(args.require_internet_artifact),
        deterministic_fallback_enabled=not (bool(args.disable_deterministic_fallback) or bool(args.require_genuine_llm_product)),
        prefer_llm_kwargs=bool(args.prefer_llm_kwargs),
        prefer_llm_patch_proposals=bool(args.prefer_llm_patch_proposals),
        llm_thinking_mode=args.llm_thinking_mode,
        require_llm_generation=bool(args.require_llm_generation) or bool(args.require_genuine_llm_product),
        require_market_evidence_reference=bool(args.require_market_evidence_reference) or bool(args.require_genuine_llm_product),
        require_non_template_product=bool(args.require_non_template_product) or bool(args.require_genuine_llm_product),
        default_command_timeout_seconds=int(args.default_command_timeout),
        execution_backend=str(args.execution_backend),
        docker_image=str(args.docker_image),
        vm_provider=str(args.vm_provider),
        vm_name=str(args.vm_name),
        vm_host=str(args.vm_host),
        vm_workdir=str(args.vm_workdir),
        vm_network_mode=str(args.vm_network_mode),
        vm_sync_mode=str(args.vm_sync_mode),
        internet_enabled=bool(args.internet_enabled),
        internet_output_root=args.internet_output_root,
        internet_max_bytes=int(args.internet_max_bytes),
        internet_timeout_seconds=float(args.internet_timeout),
        internet_allow_private_networks=bool(args.internet_allow_private_networks),
    )

    print(json.dumps(summarize_audit(audit), indent=2, ensure_ascii=False, default=str))
    if args.save_audit:
        path = Path(args.save_audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"saved_audit={path}")
    artifact_check = dict(audit.get("local_machine_artifact_check", {}) or {})
    if (bool(args.require_artifacts) or strict_generation_contract) and not bool(artifact_check.get("ok", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional


RUNTIME_MODE_VERSION = "conos.runtime_mode/v1"
RUNTIME_MODE_POLICY_VERSION = "conos.runtime_mode_policy/v1"


class RuntimeMode(str, Enum):
    STOPPED = "STOPPED"
    SLEEP = "SLEEP"
    IDLE = "IDLE"
    ROUTINE_RUN = "ROUTINE_RUN"
    DEEP_THINK = "DEEP_THINK"
    CREATING = "CREATING"
    ACTING = "ACTING"
    DREAM = "DREAM"
    WAITING_HUMAN = "WAITING_HUMAN"
    DEGRADED_RECOVERY = "DEGRADED_RECOVERY"


MODE_DESCRIPTIONS: Dict[RuntimeMode, str] = {
    RuntimeMode.STOPPED: "process is not running; only durable state remains",
    RuntimeMode.SLEEP: "minimal live standby; heartbeat and housekeeping only, no autonomous thinking",
    RuntimeMode.IDLE: "awake and available; no-user ticks may promote safe L0/L1 goal pressure",
    RuntimeMode.ROUTINE_RUN: "normal low-risk task progress through fast deterministic or small-model paths",
    RuntimeMode.DEEP_THINK: "high-uncertainty deliberation, root-cause analysis, or final judgment",
    RuntimeMode.CREATING: "bounded generation of candidate plans, patches, experiments, or designs",
    RuntimeMode.ACTING: "executing tools, tests, file reads, patches, VM actions, or mirror sync",
    RuntimeMode.DREAM: "offline consolidation, soak, profiling, learning, or maintenance without world-changing actions",
    RuntimeMode.WAITING_HUMAN: "blocked for approval, ambiguity resolution, or human review",
    RuntimeMode.DEGRADED_RECOVERY: "resource degradation, crash recovery, zombie suspicion, or retry recovery",
}


THINKING_ROUTE_NAMES = {
    "analyst",
    "deliberation",
    "final_audit",
    "hypothesis",
    "planning",
    "planner",
    "plan_generation",
    "recovery",
    "root_cause",
    "test_failure",
}

CREATING_ROUTE_NAMES = {
    "candidate_generation",
    "creation",
    "creative",
    "patch_proposal",
    "proposal",
    "world_model_proposal",
}

ACTING_ACTION_NAMES = {
    "apply_patch",
    "edit_replace_range",
    "file_read",
    "mirror_apply",
    "mirror_exec",
    "mirror_plan",
    "repo_find",
    "repo_grep",
    "repo_tree",
    "run_build",
    "run_lint",
    "run_test",
    "run_typecheck",
    "internet_fetch",
    "internet_fetch_project",
}

CREATING_ACTION_NAMES = {
    "candidate_files_set",
    "candidate_files_update",
    "create_file",
    "note_write",
    "propose_patch",
}

SLEEP_HEARTBEATS = {"BACKOFF_WAIT", "PAUSED", "SLEEP"}
ACTING_HEARTBEATS = {"RUNNING", "TASK_STARTED", "TICKING"}
DEGRADED_STATUSES = {"DEGRADED", "RECOVERING"}
TERMINAL_RUN_STATUSES = {"COMPLETED", "FAILED", "STOPPED"}


MODE_POLICIES: Dict[RuntimeMode, Dict[str, Any]] = {
    RuntimeMode.STOPPED: {
        "scheduler": {"planner_tick": False, "tick_interval_seconds": None, "background_only": False},
        "llm_budget": {
            "max_llm_calls": 0,
            "max_prompt_tokens": 0,
            "max_completion_tokens": 0,
            "max_wall_clock_seconds": 0,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "none",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": [],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["service_start"],
    },
    RuntimeMode.SLEEP: {
        "scheduler": {"planner_tick": False, "tick_interval_seconds": 60, "background_only": True},
        "llm_budget": {
            "max_llm_calls": 0,
            "max_prompt_tokens": 0,
            "max_completion_tokens": 0,
            "max_wall_clock_seconds": 0,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "none",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": [],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["new_task", "manual_resume", "critical_watchdog_event"],
    },
    RuntimeMode.IDLE: {
        "scheduler": {"planner_tick": True, "tick_interval_seconds": 15, "background_only": False},
        "llm_budget": {
            "max_llm_calls": 1,
            "max_prompt_tokens": 1200,
            "max_completion_tokens": 256,
            "max_wall_clock_seconds": 10,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "local_small",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read"],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["task_pending", "explicit_sleep", "service_stop"],
    },
    RuntimeMode.ROUTINE_RUN: {
        "scheduler": {"planner_tick": True, "tick_interval_seconds": 5, "background_only": False},
        "llm_budget": {
            "max_llm_calls": 3,
            "max_prompt_tokens": 6000,
            "max_completion_tokens": 900,
            "max_wall_clock_seconds": 45,
            "max_retry_count": 1,
            "escalation_allowed": True,
        },
        "model_selection": {
            "model_tier": "local_small_or_fast",
            "thinking_mode": "mostly_off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": True,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read", "propose_patch", "execute"],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["uncertainty_high", "side_effect_selected", "waiting_approval", "verified_complete"],
    },
    RuntimeMode.DEEP_THINK: {
        "scheduler": {"planner_tick": True, "tick_interval_seconds": 10, "background_only": False},
        "llm_budget": {
            "max_llm_calls": 2,
            "max_prompt_tokens": 12000,
            "max_completion_tokens": 2048,
            "max_wall_clock_seconds": 300,
            "max_retry_count": 0,
            "escalation_allowed": True,
        },
        "model_selection": {
            "model_tier": "strong",
            "thinking_mode": "on_or_budgeted",
            "prefer_strongest_model": True,
            "allow_cloud_escalation": True,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read"],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["decision_state_distilled", "hypothesis_ranked", "plan_ready", "needs_human_review"],
    },
    RuntimeMode.CREATING: {
        "scheduler": {"planner_tick": True, "tick_interval_seconds": 8, "background_only": False, "max_mode_ticks": 4},
        "llm_budget": {
            "max_llm_calls": 2,
            "max_prompt_tokens": 9000,
            "max_completion_tokens": 1500,
            "max_wall_clock_seconds": 120,
            "max_retry_count": 0,
            "escalation_allowed": True,
        },
        "model_selection": {
            "model_tier": "strong_or_best_creative",
            "thinking_mode": "budgeted",
            "prefer_strongest_model": True,
            "allow_cloud_escalation": True,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read", "propose_patch"],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": [
            "candidate_plan_created",
            "patch_proposal_generated",
            "hypothesis_created",
            "evidence_refs_written",
            "needs_human_review",
            "max_mode_ticks_without_artifact",
        ],
    },
    RuntimeMode.ACTING: {
        "scheduler": {"planner_tick": True, "tick_interval_seconds": 2, "background_only": False},
        "llm_budget": {
            "max_llm_calls": 1,
            "max_prompt_tokens": 4000,
            "max_completion_tokens": 512,
            "max_wall_clock_seconds": 30,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "none_or_local_small_for_formatting",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read", "propose_patch", "execute", "network", "credential", "sync_back"],
            "approval_required_capability_layers": ["sync_back"],
            "side_effects_allowed": True,
        },
        "exit_conditions": ["action_result_recorded", "verifier_required", "waiting_approval", "degraded"],
    },
    RuntimeMode.DREAM: {
        "scheduler": {"planner_tick": False, "tick_interval_seconds": 60, "background_only": True},
        "llm_budget": {
            "max_llm_calls": 1,
            "max_prompt_tokens": 4000,
            "max_completion_tokens": 512,
            "max_wall_clock_seconds": 30,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "local_small",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read"],
            "approval_required_capability_layers": [],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["maintenance_complete", "new_task", "degraded"],
    },
    RuntimeMode.WAITING_HUMAN: {
        "scheduler": {"planner_tick": False, "tick_interval_seconds": 30, "background_only": True},
        "llm_budget": {
            "max_llm_calls": 0,
            "max_prompt_tokens": 0,
            "max_completion_tokens": 0,
            "max_wall_clock_seconds": 0,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "none",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read"],
            "approval_required_capability_layers": ["propose_patch", "execute", "network", "credential", "sync_back"],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["approval_granted", "user_rejects", "user_updates_goal"],
    },
    RuntimeMode.DEGRADED_RECOVERY: {
        "scheduler": {"planner_tick": False, "tick_interval_seconds": 20, "background_only": True},
        "llm_budget": {
            "max_llm_calls": 0,
            "max_prompt_tokens": 0,
            "max_completion_tokens": 0,
            "max_wall_clock_seconds": 0,
            "max_retry_count": 0,
            "escalation_allowed": False,
        },
        "model_selection": {
            "model_tier": "none_until_recovered",
            "thinking_mode": "off",
            "prefer_strongest_model": False,
            "allow_cloud_escalation": False,
        },
        "permission_policy": {
            "allowed_capability_layers": ["read"],
            "approval_required_capability_layers": ["execute", "network", "credential", "sync_back"],
            "side_effects_allowed": False,
        },
        "exit_conditions": ["watchdog_ok", "vm_recovered", "human_intervention_required"],
    },
}


@dataclass(frozen=True)
class RuntimeModeSnapshot:
    schema_version: str
    mode: str
    description: str
    reason: str
    source_status: Dict[str, Any] = field(default_factory=dict)
    allowed_autonomy: Dict[str, bool] = field(default_factory=dict)
    mode_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _iter_mappings(items: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping):
                yield item


def normalize_runtime_mode(value: Any) -> str:
    if isinstance(value, RuntimeMode):
        return value.value
    text = _upper(value)
    if text in RuntimeMode.__members__:
        return RuntimeMode[text].value
    for mode in RuntimeMode:
        if text == mode.value:
            return mode.value
    return ""


def mode_policy_for_mode(mode: Any) -> Dict[str, Any]:
    normalized = normalize_runtime_mode(mode)
    selected = RuntimeMode(normalized) if normalized else RuntimeMode.IDLE
    policy = deepcopy(MODE_POLICIES[selected])
    policy["schema_version"] = RUNTIME_MODE_POLICY_VERSION
    policy["mode"] = selected.value
    return policy


def autonomy_for_mode(mode: RuntimeMode) -> Dict[str, bool]:
    if mode == RuntimeMode.STOPPED:
        return {
            "heartbeat": False,
            "housekeeping": False,
            "planner_tick": False,
            "llm_calls": False,
            "tool_execution": False,
            "side_effects": False,
        }
    if mode == RuntimeMode.SLEEP:
        return {
            "heartbeat": True,
            "housekeeping": True,
            "planner_tick": False,
            "llm_calls": False,
            "tool_execution": False,
            "side_effects": False,
        }
    if mode == RuntimeMode.DREAM:
        return {
            "heartbeat": True,
            "housekeeping": True,
            "planner_tick": False,
            "llm_calls": True,
            "tool_execution": False,
            "side_effects": False,
        }
    if mode == RuntimeMode.WAITING_HUMAN:
        return {
            "heartbeat": True,
            "housekeeping": True,
            "planner_tick": False,
            "llm_calls": False,
            "tool_execution": False,
            "side_effects": False,
        }
    if mode == RuntimeMode.DEGRADED_RECOVERY:
        return {
            "heartbeat": True,
            "housekeeping": True,
            "planner_tick": False,
            "llm_calls": False,
            "tool_execution": False,
            "side_effects": False,
        }
    if mode in {RuntimeMode.ACTING, RuntimeMode.ROUTINE_RUN}:
        return {
            "heartbeat": True,
            "housekeeping": True,
            "planner_tick": True,
            "llm_calls": True,
            "tool_execution": True,
            "side_effects": mode == RuntimeMode.ACTING,
        }
    return {
        "heartbeat": True,
        "housekeeping": True,
        "planner_tick": True,
        "llm_calls": True,
        "tool_execution": False,
        "side_effects": False,
    }


def snapshot_for_mode(
    mode: RuntimeMode,
    *,
    reason: str,
    source_status: Optional[Mapping[str, Any]] = None,
) -> RuntimeModeSnapshot:
    return RuntimeModeSnapshot(
        schema_version=RUNTIME_MODE_VERSION,
        mode=mode.value,
        description=MODE_DESCRIPTIONS[mode],
        reason=str(reason or ""),
        source_status=dict(source_status or {}),
        allowed_autonomy=autonomy_for_mode(mode),
        mode_policy=mode_policy_for_mode(mode),
    )


def creating_exit_status(state: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = _mapping(state)
    policy = mode_policy_for_mode(RuntimeMode.CREATING)
    max_ticks = int(_mapping(policy.get("scheduler")).get("max_mode_ticks", 4) or 4)
    notes = list(payload.get("notes") or []) if isinstance(payload.get("notes"), list) else []
    evidence_refs = list(payload.get("evidence_refs") or []) if isinstance(payload.get("evidence_refs"), list) else []
    for row in notes:
        if isinstance(row, Mapping):
            evidence_refs.extend(list(row.get("evidence_refs") or []) if isinstance(row.get("evidence_refs"), list) else [])
    checks = {
        "candidate_plan_created": bool(payload.get("candidate_plan") or payload.get("candidate_plans") or payload.get("plan_candidates")),
        "patch_proposal_generated": bool(payload.get("patch_proposal") or payload.get("patch_proposals")),
        "hypothesis_created": bool(payload.get("hypothesis") or payload.get("hypotheses") or payload.get("competing_hypotheses")),
        "evidence_refs_written": bool(evidence_refs),
        "needs_human_review": bool(payload.get("needs_human_review") or _lower(payload.get("terminal_state")) == "needs_human_review"),
        "max_mode_ticks_without_artifact": int(payload.get("creating_tick_count", 0) or 0) >= max_ticks,
    }
    satisfied = [key for key, value in checks.items() if value]
    return {
        "schema_version": RUNTIME_MODE_POLICY_VERSION,
        "mode": RuntimeMode.CREATING.value,
        "status": "exit_ready" if satisfied else "continue_creating",
        "satisfied_exit_conditions": satisfied,
        "missing_exit_conditions": [key for key in checks if key not in satisfied],
        "max_mode_ticks": max_ticks,
    }


def infer_route_runtime_mode(route_name: str, route_context: Optional[Mapping[str, Any]] = None) -> RuntimeModeSnapshot:
    route_key = _lower(route_name or "general") or "general"
    context = _mapping(route_context)
    metadata = _mapping(context.get("metadata", {}))
    explicit = normalize_runtime_mode(metadata.get("runtime_mode") or context.get("runtime_mode"))
    if explicit:
        return snapshot_for_mode(RuntimeMode(explicit), reason="explicit_route_runtime_mode", source_status={"route_name": route_key})
    if route_key in CREATING_ROUTE_NAMES or any(token in route_key for token in ("proposal", "candidate", "creative", "creation")):
        return snapshot_for_mode(RuntimeMode.CREATING, reason=f"route:{route_key}", source_status={"route_name": route_key})
    if route_key in THINKING_ROUTE_NAMES:
        return snapshot_for_mode(RuntimeMode.DEEP_THINK, reason=f"route:{route_key}", source_status={"route_name": route_key})
    return snapshot_for_mode(RuntimeMode.ROUTINE_RUN, reason=f"route:{route_key}", source_status={"route_name": route_key})


def infer_runtime_mode(
    *,
    service_running: bool = True,
    run: Optional[Mapping[str, Any]] = None,
    runs: Optional[Iterable[Mapping[str, Any]]] = None,
    tasks: Optional[Iterable[Mapping[str, Any]]] = None,
    watchdog: Optional[Mapping[str, Any]] = None,
    soak_sessions: Optional[Iterable[Mapping[str, Any]]] = None,
    active_action: Optional[Mapping[str, Any]] = None,
    route_name: str = "",
    route_context: Optional[Mapping[str, Any]] = None,
    explicit_mode: Any = "",
) -> RuntimeModeSnapshot:
    if not service_running:
        return snapshot_for_mode(RuntimeMode.STOPPED, reason="service_not_running")

    explicit = normalize_runtime_mode(explicit_mode)
    watchdog_payload = _mapping(watchdog)
    run_payload = _mapping(run)
    run_rows = list(runs or [])
    task_rows = list(tasks or [])
    action_payload = _mapping(active_action)
    source = {
        "run_status": _upper(run_payload.get("status")),
        "heartbeat_status": _upper(run_payload.get("heartbeat_status")),
        "watchdog_status": _upper(watchdog_payload.get("status")),
        "active_action": _text(action_payload.get("function_name") or action_payload.get("name")),
        "route_name": _text(route_name),
    }

    if _upper(watchdog_payload.get("status")) == "DEGRADED":
        return snapshot_for_mode(RuntimeMode.DEGRADED_RECOVERY, reason="watchdog_degraded", source_status=source)

    all_runs = [run_payload] if run_payload else []
    all_runs.extend([dict(row) for row in run_rows if isinstance(row, Mapping)])
    if any(_upper(item.get("status")) in DEGRADED_STATUSES for item in all_runs):
        return snapshot_for_mode(RuntimeMode.DEGRADED_RECOVERY, reason="run_degraded_or_recovering", source_status=source)
    if any("zombie" in _lower(item.get("paused_reason")) for item in all_runs):
        return snapshot_for_mode(RuntimeMode.DEGRADED_RECOVERY, reason="zombie_recovery", source_status=source)

    if explicit == RuntimeMode.SLEEP.value:
        return snapshot_for_mode(RuntimeMode.SLEEP, reason="explicit_sleep", source_status=source)
    if explicit:
        return snapshot_for_mode(RuntimeMode(explicit), reason="explicit_runtime_mode", source_status=source)

    if any(_upper(item.get("status")) == "WAITING_APPROVAL" for item in all_runs) or any(
        _upper(item.get("status")) == "WAITING_APPROVAL" for item in task_rows
    ):
        return snapshot_for_mode(RuntimeMode.WAITING_HUMAN, reason="waiting_approval", source_status=source)
    if any(_lower(item.get("terminal_state")) == "needs_human_review" for item in all_runs):
        return snapshot_for_mode(RuntimeMode.WAITING_HUMAN, reason="needs_human_review", source_status=source)

    if any(_upper(item.get("status")) == "PAUSED" for item in all_runs):
        return snapshot_for_mode(RuntimeMode.SLEEP, reason="run_paused_low_power_standby", source_status=source)
    if _upper(run_payload.get("heartbeat_status")) in SLEEP_HEARTBEATS:
        return snapshot_for_mode(RuntimeMode.SLEEP, reason="heartbeat_low_power_wait", source_status=source)

    for session in _iter_mappings(soak_sessions):
        if _upper(session.get("status")) in {"RUNNING", "ZOMBIE_SUSPECTED"}:
            return snapshot_for_mode(RuntimeMode.DREAM, reason="soak_or_background_session_running", source_status=source)

    action_name = _lower(action_payload.get("function_name") or action_payload.get("name"))
    if action_name in CREATING_ACTION_NAMES:
        return snapshot_for_mode(RuntimeMode.CREATING, reason=f"action:{action_name}", source_status=source)
    if action_name in ACTING_ACTION_NAMES:
        return snapshot_for_mode(RuntimeMode.ACTING, reason=f"action:{action_name}", source_status=source)

    if route_name:
        return infer_route_runtime_mode(route_name, route_context)

    if any(_upper(item.get("status")) == "RUNNING" for item in task_rows):
        return snapshot_for_mode(RuntimeMode.ACTING, reason="task_running", source_status=source)
    if any(_upper(item.get("status")) == "PENDING" for item in task_rows):
        return snapshot_for_mode(RuntimeMode.ROUTINE_RUN, reason="task_pending", source_status=source)
    if _upper(run_payload.get("heartbeat_status")) in ACTING_HEARTBEATS:
        return snapshot_for_mode(RuntimeMode.ACTING, reason="heartbeat_acting", source_status=source)
    if any(_upper(item.get("status")) == "RUNNING" for item in all_runs):
        return snapshot_for_mode(RuntimeMode.IDLE, reason="run_online_no_runnable_work", source_status=source)
    if all_runs and all(_upper(item.get("status")) in TERMINAL_RUN_STATUSES for item in all_runs):
        return snapshot_for_mode(RuntimeMode.IDLE, reason="only_terminal_runs", source_status=source)
    return snapshot_for_mode(RuntimeMode.IDLE, reason="no_active_work", source_status=source)


def runtime_mode_catalog() -> Dict[str, Any]:
    return {
        "schema_version": RUNTIME_MODE_VERSION,
        "modes": [
            {
                "mode": mode.value,
                "description": MODE_DESCRIPTIONS[mode],
                "allowed_autonomy": autonomy_for_mode(mode),
                "mode_policy": mode_policy_for_mode(mode),
            }
            for mode in RuntimeMode
        ],
    }

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional

from core.runtime_paths import default_state_path
from core.cognition.runtime_behavior_policy import derive_runtime_behavior_policy
from core.task_discovery.models import READ_ONLY_ACTIONS, string_list

if TYPE_CHECKING:
    from core.runtime.long_run_supervisor import LongRunSupervisor


AUTONOMOUS_TICK_VERSION = "conos.autonomous_tick/v0.1"
TERMINAL_STATUSES = {"COMPLETED", "STOPPED", "FAILED"}

SAFE_AUTONOMOUS_ACTIONS = {
    *READ_ONLY_ACTIONS,
    "run_eval",
    "run_readonly_analysis",
    "propose_skill_candidate",
}
SAFE_PERMISSION_LEVELS = {"L0", "L1"}
DEFAULT_COOLDOWN_SECONDS = 300.0


def load_autonomous_state(path: str | Path | None = None) -> Dict[str, Any]:
    state_path = Path(path).expanduser() if path is not None else default_state_path()
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def ensure_autonomous_run(
    supervisor: LongRunSupervisor,
    *,
    state_path: str | Path | None = None,
    enabled: bool = True,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
) -> Dict[str, Any]:
    """Turn an empty no-user tick into a safe, auditable background task.

    This is intentionally conservative: it only promotes active L0/L1 goal
    pressure whose allowed actions are read/report/analysis oriented. Side
    effects, network, credentials, sync-back, and core modification still need
    the normal task governance path.
    """
    if not enabled:
        return {"status": "DISABLED", "schema_version": AUTONOMOUS_TICK_VERSION}
    active = _active_nonterminal_runs(supervisor)
    if active:
        return {
            "status": "ACTIVE_RUN_PRESENT",
            "schema_version": AUTONOMOUS_TICK_VERSION,
            "active_run_ids": [str(run.get("run_id", "")) for run in active],
        }

    state = load_autonomous_state(state_path)
    if not state:
        return {"status": "NO_STATE", "schema_version": AUTONOMOUS_TICK_VERSION, "state_path": str(state_path or default_state_path())}
    selected = select_autonomous_goal_pressure(state)
    if not selected:
        selected = synthesize_homeostasis_goal_pressure(state)
    if not selected:
        return {"status": "NO_AUTONOMOUS_GOAL_PRESSURE", "schema_version": AUTONOMOUS_TICK_VERSION}

    goal_id = str(selected.get("goal_id") or selected.get("title") or selected.get("objective") or "autonomous_goal")
    recent = _recent_autonomous_run_for_goal(
        supervisor,
        goal_id=goal_id,
        cooldown_seconds=float(cooldown_seconds),
    )
    if recent:
        return {
            "status": "THROTTLED_RECENT_AUTONOMOUS_RUN",
            "schema_version": AUTONOMOUS_TICK_VERSION,
            "source_goal_id": goal_id,
            "recent_run_id": str(recent.get("run_id", "")),
            "cooldown_seconds": float(cooldown_seconds),
        }

    objective = str(selected.get("objective") or selected.get("title") or selected.get("success_condition") or "").strip()
    if not objective:
        objective = f"Review autonomous goal pressure {goal_id} and write a bounded status note."
    goal = f"[autonomous-no-user-tick] {objective}"
    priority = _priority_to_int(selected.get("priority", 0.0))
    metadata = {
        "schema_version": AUTONOMOUS_TICK_VERSION,
        "autonomous_no_user_tick": True,
        "created_without_user_instruction": True,
        "source_goal_id": goal_id,
        "pressure_type": str(selected.get("pressure_type") or selected.get("source") or ""),
        "permission_level": str(selected.get("permission_level") or "L1"),
        "allowed_actions": string_list(selected.get("allowed_actions")),
        "forbidden_actions": string_list(selected.get("forbidden_actions")),
        "success_condition": str(selected.get("success_condition") or ""),
    }
    run_id = supervisor.create_run(goal, metadata=metadata)
    task_id = supervisor.add_task(
        run_id,
        objective,
        priority=priority,
        verifier={
            "mode": "autonomous_no_user_tick",
            "schema_version": AUTONOMOUS_TICK_VERSION,
            "read_only": True,
            "source_goal_id": goal_id,
            "source_goal": selected,
            "state_path": str(state_path or default_state_path()),
            "allowed_actions": string_list(selected.get("allowed_actions")),
            "success_condition": str(selected.get("success_condition") or ""),
        },
    )
    supervisor.event_journal.append(
        run_id=run_id,
        task_id=task_id,
        event_type="autonomous_no_user_tick_scheduled",
        payload={
            "schema_version": AUTONOMOUS_TICK_VERSION,
            "source_goal": selected,
            "state_path": str(state_path or default_state_path()),
            "cooldown_seconds": float(cooldown_seconds),
        },
    )
    return {
        "status": "AUTONOMOUS_TASK_SCHEDULED",
        "schema_version": AUTONOMOUS_TICK_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "source_goal_id": goal_id,
        "permission_level": metadata["permission_level"],
        "priority": priority,
    }


def select_autonomous_goal_pressure(state: Mapping[str, Any]) -> Dict[str, Any]:
    goal_stack = state.get("goal_stack") if isinstance(state.get("goal_stack"), Mapping) else {}
    subgoals = goal_stack.get("subgoals", []) if isinstance(goal_stack, Mapping) else []
    if not isinstance(subgoals, Iterable) or isinstance(subgoals, (str, bytes, Mapping)):
        return {}
    candidates = [dict(item) for item in subgoals if isinstance(item, Mapping) and _is_safe_active_goal(item)]
    if not candidates:
        return {}
    decision = derive_runtime_behavior_policy(state, task_candidates=candidates)
    selected_id = str(
        _mapping(decision.selected_task).get("goal_id")
        or _mapping(decision.selected_task).get("task_id")
        or _mapping(decision.selected_task).get("title")
        or ""
    )
    if selected_id:
        for candidate in candidates:
            candidate_id = str(candidate.get("goal_id") or candidate.get("task_id") or candidate.get("title") or "")
            if candidate_id == selected_id:
                selected = dict(candidate)
                metadata = _mapping(selected.get("metadata"))
                metadata["runtime_behavior_policy"] = {
                    "schema_version": decision.schema_version,
                    "runtime_mode": decision.runtime_mode,
                    "reason": decision.reason,
                    "task_priority_updates": decision.task_priority_updates,
                }
                selected["metadata"] = metadata
                return selected
    candidates.sort(
        key=lambda item: (
            _safe_float(item.get("priority")),
            _safe_float(_mapping(item.get("metadata")).get("confidence", 0.0)),
            str(item.get("goal_id") or item.get("title") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def synthesize_homeostasis_goal_pressure(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive a safe autonomous goal from self/world/runtime pressure.

    This is the minimal bridge from passive model state to endogenous action:
    the system can notice its own failures, anomalies, or uncertainty and create
    a read-only diagnostic task even when no explicit user instruction exists.
    """
    self_summary = _mapping(state.get("self_summary"))
    world_summary = _mapping(state.get("world_summary"))
    telemetry = _mapping(state.get("telemetry_summary"))

    error_flags = string_list(self_summary.get("error_flags"))
    recent_failures = _dict_list(self_summary.get("recent_failures"))
    anomaly_flags = string_list(telemetry.get("anomaly_flags"))
    uncertainty = _safe_float(world_summary.get("uncertainty_estimate"))
    risk = _safe_float(world_summary.get("risk_estimate"))
    latent_hypotheses = _dict_list(world_summary.get("latent_hypotheses"))

    candidates: list[Dict[str, Any]] = []
    if error_flags or recent_failures:
        candidates.append(
            _homeostasis_goal(
                goal_id="goal:homeostasis:self_model_failure_review",
                pressure_type="self_model_homeostasis",
                title="Review self-model failure pressure",
                objective=(
                    "Inspect recent self-model failures, summarize the repeated failure boundary, "
                    "and write a bounded diagnostic report without modifying source."
                ),
                priority=max(0.72, min(0.95, 0.72 + 0.04 * len(error_flags) + 0.03 * len(recent_failures))),
                success_condition="Recent failures are summarized with evidence refs and a safe next-step recommendation.",
                evidence_refs=[*error_flags[:4], *[str(row.get("action_name") or row.get("failure_type") or "recent_failure") for row in recent_failures[:4]]],
            )
        )
    if anomaly_flags:
        candidates.append(
            _homeostasis_goal(
                goal_id="goal:homeostasis:runtime_anomaly_review",
                pressure_type="runtime_homeostasis",
                title="Review runtime anomaly pressure",
                objective=(
                    "Inspect runtime anomaly flags, identify whether the system is degraded, "
                    "and write a bounded recovery recommendation."
                ),
                priority=max(0.74, min(0.96, 0.74 + 0.05 * len(anomaly_flags))),
                success_condition="Anomaly flags are explained or marked for human review with no side effects.",
                evidence_refs=anomaly_flags[:8],
            )
        )
    if uncertainty >= 0.65 or risk >= 0.72 or latent_hypotheses:
        candidates.append(
            _homeostasis_goal(
                goal_id="goal:homeostasis:world_model_uncertainty_review",
                pressure_type="world_model_homeostasis",
                title="Review world-model uncertainty pressure",
                objective=(
                    "Inspect world-model uncertainty, risk, and latent hypotheses; produce a short "
                    "evidence report with candidate discriminating observations."
                ),
                priority=max(0.7, min(0.94, 0.45 + 0.32 * uncertainty + 0.22 * risk + 0.03 * len(latent_hypotheses))),
                success_condition="Uncertainty source is summarized and next observations are proposed, without acting on the world.",
                evidence_refs=[f"uncertainty={uncertainty:.2f}", f"risk={risk:.2f}"],
            )
        )
    if not candidates:
        return {}
    candidates.sort(key=lambda row: (_safe_float(row.get("priority")), str(row.get("goal_id") or "")), reverse=True)
    return candidates[0]


def _is_safe_active_goal(goal: Mapping[str, Any]) -> bool:
    status = str(goal.get("status") or "active").strip().lower()
    if status in {"completed", "done", "cancelled", "canceled", "rejected", "inactive", "blocked"}:
        return False
    permission = str(goal.get("permission_level") or "L1").strip()
    if permission not in SAFE_PERMISSION_LEVELS:
        return False
    if bool(goal.get("requires_human_approval", False)):
        return False
    allowed = set(string_list(goal.get("allowed_actions")))
    if not allowed:
        return False
    if not allowed <= SAFE_AUTONOMOUS_ACTIONS:
        return False
    return True


def _homeostasis_goal(
    *,
    goal_id: str,
    pressure_type: str,
    title: str,
    objective: str,
    priority: float,
    success_condition: str,
    evidence_refs: list[str],
) -> Dict[str, Any]:
    return {
        "schema_version": AUTONOMOUS_TICK_VERSION,
        "goal_id": goal_id,
        "source": "autonomous_homeostasis",
        "pressure_type": pressure_type,
        "title": title,
        "objective": objective,
        "status": "active",
        "priority": max(0.0, min(1.0, float(priority))),
        "permission_level": "L1",
        "success_condition": success_condition,
        "allowed_actions": ["read_logs", "read_reports", "read_files", "run_readonly_analysis", "write_report"],
        "forbidden_actions": ["modify_core_runtime_without_approval", "network", "credential", "sync_back"],
        "evidence_refs": list(evidence_refs or []),
        "metadata": {
            "homeostasis_generated": True,
            "side_effects_allowed": False,
        },
    }


def _active_nonterminal_runs(supervisor: LongRunSupervisor) -> list[Dict[str, Any]]:
    active: list[Dict[str, Any]] = []
    for run in supervisor.state_store.list_runs():
        status = str(run.get("status", "") or "")
        if status not in TERMINAL_STATUSES and status not in {"PAUSED", "WAITING_APPROVAL"}:
            active.append(run)
    return active


def _recent_autonomous_run_for_goal(
    supervisor: LongRunSupervisor,
    *,
    goal_id: str,
    cooldown_seconds: float,
) -> Dict[str, Any]:
    now = time.time()
    cooldown = max(0.0, float(cooldown_seconds))
    for run in supervisor.state_store.list_runs():
        metadata = dict(run.get("metadata", {}) or {})
        if not metadata.get("autonomous_no_user_tick"):
            continue
        if str(metadata.get("source_goal_id") or "") != str(goal_id):
            continue
        resolution = _mapping(metadata.get("homeostasis_resolution"))
        suppress_until = _safe_float(resolution.get("repeat_suppress_until"))
        if suppress_until > now:
            return run
        status = str(run.get("status", "") or "")
        if status not in TERMINAL_STATUSES and status not in {"PAUSED", "WAITING_APPROVAL"}:
            return run
        if cooldown and (now - float(run.get("created_at", 0.0) or 0.0)) <= cooldown:
            return run
    return {}


def _priority_to_int(value: Any) -> int:
    return int(round(_safe_float(value) * 100))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]

from __future__ import annotations

from typing import Any, Dict, Mapping


LOCAL_MACHINE_BUDGET_POLICY_VERSION = "conos.local_machine.budget_policy/v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _latest_failed_validation(state: Mapping[str, Any]) -> dict[str, Any]:
    for row in reversed(_as_list(state.get("validation_runs"))):
        payload = _as_dict(row)
        if payload and not bool(payload.get("success", False)):
            return payload
    return {}


def _read_file_paths(state: Mapping[str, Any]) -> set[str]:
    return {
        str(_as_dict(row).get("path") or "")
        for row in _as_list(state.get("read_files"))
        if str(_as_dict(row).get("path") or "")
    }


def evaluate_fast_path_eligibility(state: Mapping[str, Any]) -> Dict[str, Any]:
    binding = _as_dict(state.get("target_binding"))
    top_target = str(binding.get("top_target_file") or "")
    confidence = float(binding.get("target_confidence", 0.0) or 0.0)
    terminal_state = str(state.get("terminal_state") or "")
    if terminal_state == "completed_verified":
        return {
            "schema_version": LOCAL_MACHINE_BUDGET_POLICY_VERSION,
            "path": "fast_path",
            "eligible": False,
            "target_file": top_target,
            "target_confidence": confidence,
            "reasons": ["terminal_completed_verified"],
            "blockers": ["terminal_completed_verified"],
        }
    latest_failure = _latest_failed_validation(state)
    read_paths = _read_file_paths(state)
    reasons: list[str] = []
    blockers: list[str] = []
    if latest_failure:
        reasons.append("failing_test_observed")
    else:
        blockers.append("no_failing_test_observed")
    if confidence >= 0.75 and top_target:
        reasons.append("high_confidence_target_binding")
    else:
        blockers.append("target_confidence_below_threshold")
    if top_target and top_target in read_paths:
        reasons.append("target_file_read")
    elif top_target:
        blockers.append("target_file_not_read")
    else:
        blockers.append("target_file_unknown")
    patch_events = [_as_dict(row) for row in _as_list(state.get("patch_proposals"))]
    rollback_count = sum(int(row.get("rollback_count", 0) or 0) for row in patch_events)
    if rollback_count:
        blockers.append("prior_patch_rollback")
    terminal_state = str(state.get("terminal_state") or "")
    if terminal_state == "needs_human_review":
        blockers.append("needs_human_review")
    eligible = not blockers
    return {
        "schema_version": LOCAL_MACHINE_BUDGET_POLICY_VERSION,
        "path": "fast_path",
        "eligible": bool(eligible),
        "target_file": top_target,
        "target_confidence": confidence,
        "reasons": reasons,
        "blockers": blockers,
    }


def evaluate_escalation_path(state: Mapping[str, Any], *, budget_summary: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    binding = _as_dict(state.get("target_binding"))
    confidence = float(binding.get("target_confidence", 0.0) or 0.0)
    terminal_state = str(state.get("terminal_state") or "")
    patch_events = [_as_dict(row) for row in _as_list(state.get("patch_proposals"))]
    stalled_events = [_as_dict(row) for row in _as_list(state.get("stalled_events"))]
    action_history = [_as_dict(row) for row in _as_list(state.get("action_history"))]
    rollback_count = sum(int(row.get("rollback_count", 0) or 0) for row in patch_events)
    triggers: list[str] = []
    if terminal_state == "completed_verified":
        return {
            "schema_version": LOCAL_MACHINE_BUDGET_POLICY_VERSION,
            "path": "escalation_path",
            "recommended": False,
            "triggers": [],
            "rollback_count": rollback_count,
            "target_confidence": confidence,
            "recent_actions": [str(row.get("function_name") or "") for row in action_history[-4:]],
            "budget_summary": _as_dict(budget_summary),
            "terminal_state": terminal_state,
        }
    if confidence and confidence < 0.75:
        triggers.append("low_target_confidence")
    if rollback_count:
        triggers.append("patch_rollback")
    if stalled_events:
        triggers.append("investigation_stalled")
    if str(state.get("terminal_state") or "") == "needs_human_review":
        triggers.append("needs_human_review")
    recent = [str(row.get("function_name") or "") for row in action_history[-4:]]
    if len(recent) >= 4 and len(set(recent)) <= 2 and not _as_list(state.get("patch_proposals")):
        triggers.append("repeated_low_progress_actions")
    summary = _as_dict(budget_summary)
    strong_call_rate = float(summary.get("strong_model_call_rate", 0.0) or 0.0)
    if strong_call_rate > 0.5:
        triggers.append("strong_model_budget_pressure")
    return {
        "schema_version": LOCAL_MACHINE_BUDGET_POLICY_VERSION,
        "path": "escalation_path",
        "recommended": bool(triggers),
        "triggers": list(dict.fromkeys(triggers)),
        "rollback_count": rollback_count,
        "target_confidence": confidence,
        "recent_actions": recent,
        "budget_summary": summary,
    }


def budget_policy_report(state: Mapping[str, Any], *, budget_summary: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    fast_path = evaluate_fast_path_eligibility(state)
    escalation = evaluate_escalation_path(state, budget_summary=budget_summary)
    terminal_state = str(state.get("terminal_state") or "")
    if terminal_state == "completed_verified":
        selected_hint = "terminal_complete"
    elif fast_path.get("eligible") and not escalation.get("recommended"):
        selected_hint = "fast_path"
    else:
        selected_hint = "escalation_path"
    return {
        "schema_version": LOCAL_MACHINE_BUDGET_POLICY_VERSION,
        "fast_path": fast_path,
        "escalation_path": escalation,
        "selected_path_hint": selected_hint,
    }

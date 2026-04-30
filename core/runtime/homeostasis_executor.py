from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from core.runtime.autonomous_tick import AUTONOMOUS_TICK_VERSION, load_autonomous_state
from core.runtime.event_journal import EventJournal
from core.runtime.evidence_ledger import FORMAL_EVIDENCE_LEDGER_VERSION, FormalEvidenceLedger
from core.runtime.state_store import RuntimeStateStore


HOMEOSTASIS_EXECUTOR_VERSION = "conos.homeostasis_executor/v0.1"


def execute_homeostasis_task(
    *,
    run: Mapping[str, Any],
    task: Mapping[str, Any],
    state_store: RuntimeStateStore,
    event_journal: EventJournal,
) -> Dict[str, Any]:
    """Execute a safe autonomous-no-user diagnostic task.

    This executor deliberately stays inside the runtime/object layer: it reads
    durable state and runtime events, writes a bounded report artifact, records
    formal evidence, and returns a verified task result. It does not call LLMs,
    touch source trees, access credentials, fetch the network, or sync back.
    """
    run_id = str(run.get("run_id") or "")
    task_id = str(task.get("task_id") or "")
    verifier = _mapping(task.get("verifier"))
    run_metadata = _mapping(run.get("metadata"))
    source_goal = _mapping(verifier.get("source_goal")) or _goal_from_metadata(run_metadata)
    state_path = str(verifier.get("state_path") or "")
    state = load_autonomous_state(state_path) if state_path else {}
    report = build_homeostasis_report(
        run=run,
        task=task,
        source_goal=source_goal,
        state=state,
        recent_events=state_store.list_events(run_id)[-20:],
    )
    report_path = _report_path(event_journal, run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    event_journal.append(
        run_id=run_id,
        task_id=task_id,
        event_type="homeostasis_report_written",
        payload={"report_path": str(report_path), "report": _compact_report(report)},
    )

    evidence = _record_homeostasis_evidence(
        report=report,
        report_path=report_path,
        state_store=state_store,
        event_journal=event_journal,
        run_id=run_id,
        task_id=task_id,
    )
    return {
        "verified": True,
        "verifier": verifier,
        "approval_granted": _mapping(task.get("result")).get("approval_granted", ""),
        "homeostasis_report": _compact_report(report),
        "homeostasis_report_path": str(report_path),
        "formal_evidence_id": evidence.get("evidence_id", ""),
        "formal_evidence_ref": {
            "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
            "evidence_id": str(evidence.get("evidence_id") or ""),
            "ledger_hash": str(evidence.get("ledger_hash") or ""),
            "claim": str(evidence.get("claim") or ""),
            "status": str(evidence.get("status") or ""),
        },
        "side_effects_executed": False,
        "pressure_resolved": bool(report.get("pressure_resolved", False)),
        "needs_human_review": bool(report.get("needs_human_review", False)),
        "resolution_decision": _compact_resolution(_mapping(report.get("resolution_decision"))),
    }


def build_homeostasis_report(
    *,
    run: Mapping[str, Any],
    task: Mapping[str, Any],
    source_goal: Mapping[str, Any],
    state: Mapping[str, Any],
    recent_events: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    task_id = str(task.get("task_id") or "")
    pressure = _observed_pressure(source_goal=source_goal, state=state)
    pressure_type = str(source_goal.get("pressure_type") or _mapping(run.get("metadata")).get("pressure_type") or "unknown")
    diagnosis = _diagnosis(pressure_type=pressure_type, pressure=pressure)
    review = _review_policy(pressure_type=pressure_type, pressure=pressure)
    resolution = _resolution_decision(pressure_type=pressure_type, pressure=pressure, review=review)
    return {
        "schema_version": HOMEOSTASIS_EXECUTOR_VERSION,
        "autonomous_tick_schema_version": AUTONOMOUS_TICK_VERSION,
        "created_at": time.time(),
        "run_id": run_id,
        "task_id": task_id,
        "source_goal_id": str(source_goal.get("goal_id") or _mapping(run.get("metadata")).get("source_goal_id") or ""),
        "pressure_type": pressure_type,
        "trigger_source": str(source_goal.get("source") or "runtime_state"),
        "objective": str(task.get("objective") or source_goal.get("objective") or ""),
        "observed_pressure": pressure,
        "diagnosis": diagnosis,
        "recommended_next_action": review["recommended_next_action"],
        "pressure_resolved": bool(review["pressure_resolved"]),
        "needs_human_review": bool(review["needs_human_review"]),
        "escalation_boundary": review["escalation_boundary"],
        "resolution_decision": resolution,
        "allowed_actions": list(source_goal.get("allowed_actions") or _mapping(task.get("verifier")).get("allowed_actions") or []),
        "forbidden_actions": list(source_goal.get("forbidden_actions") or []),
        "recent_event_types": [str(event.get("event_type") or "") for event in recent_events[-12:]],
        "side_effects_executed": False,
        "llm_calls": 0,
    }


def _observed_pressure(*, source_goal: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    self_summary = _mapping(state.get("self_summary"))
    world_summary = _mapping(state.get("world_summary"))
    telemetry = _mapping(state.get("telemetry_summary"))
    error_flags = _string_list(self_summary.get("error_flags"))
    recent_failures = _dict_list(self_summary.get("recent_failures"))
    anomaly_flags = _string_list(telemetry.get("anomaly_flags"))
    uncertainty = _safe_float(world_summary.get("uncertainty_estimate"))
    risk = _safe_float(world_summary.get("risk_estimate"))
    latent_hypotheses = _dict_list(world_summary.get("latent_hypotheses"))
    source_refs = _string_list(source_goal.get("evidence_refs"))
    pressure_score = max(
        min(1.0, 0.25 + 0.1 * len(error_flags) + 0.08 * len(recent_failures)),
        min(1.0, 0.3 + 0.12 * len(anomaly_flags)),
        min(1.0, 0.2 + 0.45 * uncertainty + 0.25 * risk + 0.04 * len(latent_hypotheses)),
        _safe_float(source_goal.get("priority")),
    )
    return {
        "pressure_score": round(pressure_score, 4),
        "error_flags": error_flags[:12],
        "recent_failure_count": len(recent_failures),
        "recent_failures": recent_failures[-5:],
        "anomaly_flags": anomaly_flags[:12],
        "world_uncertainty": round(uncertainty, 4),
        "world_risk": round(risk, 4),
        "latent_hypothesis_count": len(latent_hypotheses),
        "source_evidence_refs": source_refs[:12],
    }


def _diagnosis(*, pressure_type: str, pressure: Mapping[str, Any]) -> str:
    if pressure.get("error_flags") or int(pressure.get("recent_failure_count", 0) or 0) > 0:
        return "Self-model pressure remains active; recent failures should be summarized before any capability change."
    if pressure.get("anomaly_flags"):
        return "Runtime anomaly pressure remains active; inspect daemon/watchdog evidence before escalation."
    if float(pressure.get("world_uncertainty", 0.0) or 0.0) >= 0.65:
        return "World-model uncertainty is elevated; gather discriminating observations before acting."
    if float(pressure.get("world_risk", 0.0) or 0.0) >= 0.72:
        return "World-model risk is elevated; stay in read-only review until risk source is identified."
    if "capability_repair" in pressure_type or "skill_candidate" in pressure_type:
        return "Explicit goal pressure is active; keep this as bounded L1 investigation until verifier evidence exists."
    return "No active homeostasis pressure is visible in the current state snapshot."


def _review_policy(*, pressure_type: str, pressure: Mapping[str, Any]) -> Dict[str, Any]:
    score = float(pressure.get("pressure_score", 0.0) or 0.0)
    hard_pressure = (
        bool(pressure.get("error_flags"))
        or bool(pressure.get("anomaly_flags"))
        or int(pressure.get("recent_failure_count", 0) or 0) > 0
        or float(pressure.get("world_uncertainty", 0.0) or 0.0) >= 0.65
        or float(pressure.get("world_risk", 0.0) or 0.0) >= 0.72
    )
    if not hard_pressure and score < 0.65:
        return {
            "pressure_resolved": True,
            "needs_human_review": False,
            "recommended_next_action": "Remain idle; no follow-up autonomous task is needed.",
            "escalation_boundary": "none",
        }
    if score >= 0.9 or bool(pressure.get("anomaly_flags")):
        return {
            "pressure_resolved": False,
            "needs_human_review": False,
            "recommended_next_action": "Continue L1 diagnostic review; escalate only if the same pressure repeats after cooldown.",
            "escalation_boundary": "repeat_pressure_after_cooldown_or_degraded_watchdog",
        }
    return {
        "pressure_resolved": False,
        "needs_human_review": False,
        "recommended_next_action": "Keep investigation read-only and collect more evidence before any L2 action.",
        "escalation_boundary": "requires_evidence_refs_and_operator_approval_for_side_effects",
    }


def _resolution_decision(
    *,
    pressure_type: str,
    pressure: Mapping[str, Any],
    review: Mapping[str, Any],
) -> Dict[str, Any]:
    score = float(pressure.get("pressure_score", 0.0) or 0.0)
    if bool(review.get("pressure_resolved", False)):
        return {
            "action": "deprioritize_repeat",
            "reason": "pressure_resolved_or_known_handled",
            "repeat_priority_multiplier": 0.25,
            "repeat_suppress_seconds": 3600,
            "run_status_after_task": "COMPLETED",
            "follow_up_task": {},
            "approval_request": {},
        }
    if pressure.get("anomaly_flags") or score >= 0.97:
        return {
            "action": "escalate_waiting_human",
            "reason": "runtime_anomaly_or_critical_pressure",
            "run_status_after_task": "WAITING_APPROVAL",
            "repeat_priority_multiplier": 1.0,
            "repeat_suppress_seconds": 0,
            "follow_up_task": {},
            "approval_request": {
                "reason": "homeostasis_pressure_requires_operator_review",
                "required_capability_layers": ["read"],
                "approval_effect": {"approved_capability_layers": ["read"]},
            },
        }
    if int(pressure.get("recent_failure_count", 0) or 0) >= 2 or pressure.get("error_flags"):
        return {
            "action": "escalate_limited_l2_mirror_investigation",
            "reason": "self_model_failure_pressure_persists",
            "run_status_after_task": "WAITING_APPROVAL",
            "repeat_priority_multiplier": 0.8,
            "repeat_suppress_seconds": 0,
            "follow_up_task": {
                "objective": "Prepare a limited L2 mirror investigation for persistent self-model failure pressure.",
                "priority": 70,
                "permission_level": "limited_L2",
                "allowed_actions": [
                    "read_logs",
                    "read_reports",
                    "read_files",
                    "run_readonly_analysis",
                    "propose_patch",
                    "edit_in_mirror",
                ],
                "forbidden_actions": ["sync_back", "credential", "network"],
            },
            "approval_request": {
                "reason": "limited_l2_mirror_investigation_requires_approval",
                "required_capability_layers": ["propose_patch"],
                "approval_effect": {"approved_capability_layers": ["propose_patch"]},
            },
        }
    if (
        float(pressure.get("world_uncertainty", 0.0) or 0.0) >= 0.65
        or float(pressure.get("world_risk", 0.0) or 0.0) >= 0.72
        or int(pressure.get("latent_hypothesis_count", 0) or 0) > 0
    ):
        return {
            "action": "escalate_deep_think",
            "reason": "world_model_uncertainty_or_risk_persists",
            "run_status_after_task": "RUNNING",
            "repeat_priority_multiplier": 0.9,
            "repeat_suppress_seconds": 0,
            "follow_up_task": {
                "objective": "Deep-think review of persistent world-model uncertainty using read-only evidence.",
                "priority": 65,
                "runtime_mode": "DEEP_THINK",
                "permission_level": "L1",
                "allowed_actions": ["read_logs", "read_reports", "read_files", "run_readonly_analysis", "write_report"],
                "forbidden_actions": ["sync_back", "credential", "network"],
            },
            "approval_request": {},
        }
    return {
        "action": "continue_l1_diagnostic",
        "reason": "pressure_persists_but_below_escalation_threshold",
        "run_status_after_task": "COMPLETED",
        "repeat_priority_multiplier": 0.65,
        "repeat_suppress_seconds": 900,
        "follow_up_task": {},
        "approval_request": {},
    }


def _record_homeostasis_evidence(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    state_store: RuntimeStateStore,
    event_journal: EventJournal,
    run_id: str,
    task_id: str,
) -> Dict[str, Any]:
    entry = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "run_id": str(run_id),
        "task_family": "autonomous_homeostasis",
        "evidence_type": "homeostasis_diagnostic",
        "claim": f"Homeostasis diagnostic for {report.get('source_goal_id') or 'runtime pressure'}: {report.get('diagnosis')}",
        "evidence": dict(report),
        "hypotheses": [],
        "action": {
            "function_name": "homeostasis_diagnostic",
            "mode": "autonomous_no_user_tick",
            "side_effects_allowed": False,
        },
        "result": {
            "success": True,
            "verified": True,
            "pressure_resolved": bool(report.get("pressure_resolved", False)),
            "needs_human_review": bool(report.get("needs_human_review", False)),
            "resolution_decision": _mapping(report.get("resolution_decision")),
        },
        "update": {
            "kind": "homeostasis_report",
            "direction": "updates_runtime_self_world_pressure",
            "pressure_type": str(report.get("pressure_type") or ""),
            "resolution_action": str(_mapping(report.get("resolution_decision")).get("action") or ""),
        },
        "source_refs": [
            f"run:{run_id}",
            f"task:{task_id}",
            f"homeostasis_report:{report_path.name}",
            *list(_mapping(report.get("observed_pressure")).get("source_evidence_refs", []) or []),
        ],
        "confidence": 0.74,
        "status": "recorded",
    }
    ledger = FormalEvidenceLedger(event_journal.runs_root / str(run_id) / "formal_evidence_ledger.jsonl", state_store=state_store)
    recorded = ledger.record(entry)
    event_journal.append(
        run_id=run_id,
        task_id=task_id,
        event_type="homeostasis_evidence_committed",
        payload={
            "evidence_id": str(recorded.get("evidence_id") or ""),
            "ledger_hash": str(recorded.get("ledger_hash") or ""),
            "claim": str(recorded.get("claim") or ""),
            "diagnostic": _compact_report(report),
        },
    )
    return recorded


def _report_path(event_journal: EventJournal, run_id: str) -> Path:
    return event_journal.runs_root / str(run_id) / "homeostasis_report.json"


def _compact_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": str(report.get("schema_version") or HOMEOSTASIS_EXECUTOR_VERSION),
        "source_goal_id": str(report.get("source_goal_id") or ""),
        "trigger_source": str(report.get("trigger_source") or ""),
        "pressure_type": str(report.get("pressure_type") or ""),
        "observed_pressure": _compact_pressure(_mapping(report.get("observed_pressure"))),
        "diagnosis": str(report.get("diagnosis") or ""),
        "recommended_next_action": str(report.get("recommended_next_action") or ""),
        "pressure_resolved": bool(report.get("pressure_resolved", False)),
        "needs_human_review": bool(report.get("needs_human_review", False)),
        "resolution_decision": _compact_resolution(_mapping(report.get("resolution_decision"))),
        "side_effects_executed": bool(report.get("side_effects_executed", False)),
        "llm_calls": int(report.get("llm_calls", 0) or 0),
    }


def _compact_resolution(resolution: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "action": str(resolution.get("action") or ""),
        "reason": str(resolution.get("reason") or ""),
        "run_status_after_task": str(resolution.get("run_status_after_task") or ""),
        "repeat_priority_multiplier": _safe_float(resolution.get("repeat_priority_multiplier")),
        "repeat_suppress_seconds": int(resolution.get("repeat_suppress_seconds", 0) or 0),
        "follow_up_task": _mapping(resolution.get("follow_up_task")),
        "approval_request": _mapping(resolution.get("approval_request")),
    }


def _compact_pressure(pressure: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pressure_score": _safe_float(pressure.get("pressure_score")),
        "error_flags": _string_list(pressure.get("error_flags"))[:8],
        "recent_failure_count": int(pressure.get("recent_failure_count", 0) or 0),
        "anomaly_flags": _string_list(pressure.get("anomaly_flags"))[:8],
        "world_uncertainty": _safe_float(pressure.get("world_uncertainty")),
        "world_risk": _safe_float(pressure.get("world_risk")),
        "latent_hypothesis_count": int(pressure.get("latent_hypothesis_count", 0) or 0),
        "source_evidence_refs": _string_list(pressure.get("source_evidence_refs"))[:8],
    }


def _goal_from_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "goal_id": str(metadata.get("source_goal_id") or ""),
        "pressure_type": str(metadata.get("pressure_type") or ""),
        "source": "run_metadata",
        "allowed_actions": list(metadata.get("allowed_actions") or []),
        "forbidden_actions": list(metadata.get("forbidden_actions") or []),
        "success_condition": str(metadata.get("success_condition") or ""),
    }


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

from __future__ import annotations

import json
from pathlib import Path

from core.runtime.autonomous_tick import (
    ensure_autonomous_run,
    select_autonomous_goal_pressure,
    synthesize_homeostasis_goal_pressure,
)
from core.runtime.long_run_supervisor import LongRunSupervisor
from core.runtime.state_store import RuntimeStateStore


def _supervisor(tmp_path: Path) -> LongRunSupervisor:
    return LongRunSupervisor(
        state_store=RuntimeStateStore(tmp_path / "state.sqlite3"),
        runs_root=tmp_path / "runs",
        worker_id="autonomous-test",
    )


def _write_state(path: Path, subgoals: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"goal_stack": {"subgoals": subgoals}}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_select_autonomous_goal_pressure_picks_highest_safe_active_goal() -> None:
    selected = select_autonomous_goal_pressure(
        {
            "goal_stack": {
                "subgoals": [
                    {
                        "goal_id": "unsafe",
                        "status": "active",
                        "priority": 1.0,
                        "permission_level": "L3",
                        "allowed_actions": ["sync_back"],
                    },
                    {
                        "goal_id": "safe-low",
                        "status": "active",
                        "priority": 0.2,
                        "permission_level": "L1",
                        "allowed_actions": ["read_logs", "write_report"],
                    },
                    {
                        "goal_id": "safe-high",
                        "status": "active",
                        "priority": 0.9,
                        "permission_level": "L1",
                        "allowed_actions": ["read_logs", "run_readonly_analysis", "write_report"],
                    },
                ]
            }
        }
    )

    assert selected["goal_id"] == "safe-high"


def test_autonomous_tick_creates_bounded_run_from_goal_pressure(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_state(
        state_path,
        [
            {
                "goal_id": "goal:capability_repair:file_read",
                "title": "Improve unreliable action: file_read",
                "objective": "Investigate repeated file_read failures and write an evidence report.",
                "status": "active",
                "priority": 0.82,
                "permission_level": "L1",
                "success_condition": "Evidence report exists.",
                "allowed_actions": ["read_logs", "read_reports", "run_readonly_analysis", "write_report"],
                "forbidden_actions": ["modify_core_runtime_without_approval"],
            }
        ],
    )
    supervisor = _supervisor(tmp_path)
    try:
        result = ensure_autonomous_run(supervisor, state_path=state_path)

        assert result["status"] == "AUTONOMOUS_TASK_SCHEDULED"
        run = supervisor.state_store.get_run(result["run_id"])
        task = supervisor.state_store.get_task(result["task_id"])
        assert run["metadata"]["autonomous_no_user_tick"] is True
        assert run["metadata"]["created_without_user_instruction"] is True
        assert run["metadata"]["source_goal_id"] == "goal:capability_repair:file_read"
        assert task["verifier"]["mode"] == "autonomous_no_user_tick"
        assert task["verifier"]["read_only"] is True
        assert "autonomous_no_user_tick_scheduled" in [
            event["event_type"] for event in supervisor.state_store.list_events(result["run_id"])
        ]
    finally:
        supervisor.state_store.close()


def test_autonomous_tick_dedups_existing_and_throttles_recent_completed_runs(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_state(
        state_path,
        [
            {
                "goal_id": "goal:skill_candidate:repo_grep",
                "objective": "Evaluate repeated repo_grep successes as a candidate skill.",
                "status": "active",
                "priority": 0.75,
                "permission_level": "L1",
                "allowed_actions": ["read_logs", "write_report"],
            }
        ],
    )
    supervisor = _supervisor(tmp_path)
    try:
        first = ensure_autonomous_run(supervisor, state_path=state_path)
        second = ensure_autonomous_run(supervisor, state_path=state_path)
        assert first["status"] == "AUTONOMOUS_TASK_SCHEDULED"
        assert second["status"] == "ACTIVE_RUN_PRESENT"

        supervisor.tick_once(first["run_id"])
        supervisor.tick_once(first["run_id"])
        throttled = ensure_autonomous_run(supervisor, state_path=state_path, cooldown_seconds=3600)
        assert throttled["status"] == "THROTTLED_RECENT_AUTONOMOUS_RUN"
        assert throttled["recent_run_id"] == first["run_id"]
    finally:
        supervisor.state_store.close()


def test_autonomous_tick_refuses_unsafe_permission_or_actions(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_state(
        state_path,
        [
            {
                "goal_id": "unsafe",
                "objective": "Sync changes to the host.",
                "status": "active",
                "priority": 1.0,
                "permission_level": "L3",
                "allowed_actions": ["sync_back"],
            }
        ],
    )
    supervisor = _supervisor(tmp_path)
    try:
        result = ensure_autonomous_run(supervisor, state_path=state_path)
        assert result["status"] == "NO_AUTONOMOUS_GOAL_PRESSURE"
        assert supervisor.state_store.list_runs() == []
    finally:
        supervisor.state_store.close()


def test_homeostasis_goal_synthesizes_from_self_model_failure_pressure() -> None:
    goal = synthesize_homeostasis_goal_pressure(
        {
            "self_summary": {
                "error_flags": ["last_action_failed"],
                "recent_failures": [{"action_name": "repo_grep", "failure_type": "timeout"}],
            },
            "world_summary": {"uncertainty_estimate": 0.2, "risk_estimate": 0.1},
        }
    )

    assert goal["goal_id"] == "goal:homeostasis:self_model_failure_review"
    assert goal["permission_level"] == "L1"
    assert "write_report" in goal["allowed_actions"]
    assert "sync_back" in goal["forbidden_actions"]


def test_autonomous_tick_creates_homeostasis_run_when_goal_stack_is_empty(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {"subgoals": []},
            "world_summary": {
                "uncertainty_estimate": 0.82,
                "risk_estimate": 0.35,
                "latent_hypotheses": [{"claim": "runtime state may be stale"}],
            },
            "self_summary": {"error_flags": [], "recent_failures": []},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        result = ensure_autonomous_run(supervisor, state_path=state_path)

        assert result["status"] == "AUTONOMOUS_TASK_SCHEDULED"
        run = supervisor.state_store.get_run(result["run_id"])
        assert run["metadata"]["source_goal_id"] == "goal:homeostasis:world_model_uncertainty_review"
        assert run["metadata"]["pressure_type"] == "world_model_homeostasis"
        task = supervisor.state_store.get_task(result["task_id"])
        assert task["verifier"]["read_only"] is True
    finally:
        supervisor.state_store.close()


def test_homeostasis_task_executes_report_and_formal_evidence(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {"subgoals": []},
            "self_summary": {
                "error_flags": ["last_action_failed"],
                "recent_failures": [{"action_name": "repo_grep", "failure_type": "timeout"}],
            },
            "world_summary": {"uncertainty_estimate": 0.4, "risk_estimate": 0.2},
            "telemetry_summary": {"anomaly_flags": []},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        scheduled = ensure_autonomous_run(supervisor, state_path=state_path)
        assert supervisor.tick_once(scheduled["run_id"])["status"] == "TASK_STARTED"

        completed = supervisor.tick_once(scheduled["run_id"])

        assert completed["status"] == "WAITING_APPROVAL"
        assert completed["resolution_action"] == "escalate_limited_l2_mirror_investigation"
        task = supervisor.state_store.get_task(scheduled["task_id"])
        result = task["result"]
        assert result["verified"] is True
        assert result["side_effects_executed"] is False
        assert result["formal_evidence_id"].startswith("ev_")
        assert result["resolution_decision"]["action"] == "escalate_limited_l2_mirror_investigation"
        report_path = Path(result["homeostasis_report_path"])
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["source_goal_id"] == "goal:homeostasis:self_model_failure_review"
        assert report["llm_calls"] == 0
        assert report["trigger_source"] == "autonomous_homeostasis"
        assert report["observed_pressure"]["error_flags"] == ["last_action_failed"]
        assert report["diagnosis"]
        assert report["recommended_next_action"]
        assert report["pressure_resolved"] is False
        assert report["resolution_decision"]["action"] == "escalate_limited_l2_mirror_investigation"
        evidence_rows = supervisor.state_store.list_evidence_entries(
            run_id=scheduled["run_id"],
            evidence_type="homeostasis_diagnostic",
        )
        assert evidence_rows[0]["evidence_id"] == result["formal_evidence_id"]
        assert evidence_rows[0]["evidence"]["trigger_source"] == "autonomous_homeostasis"
        assert evidence_rows[0]["evidence"]["observed_pressure"]["recent_failure_count"] == 1
        assert evidence_rows[0]["evidence"]["diagnosis"] == report["diagnosis"]
        assert evidence_rows[0]["evidence"]["recommended_next_action"] == report["recommended_next_action"]
        assert evidence_rows[0]["evidence"]["pressure_resolved"] is False
        event_types = [event["event_type"] for event in supervisor.state_store.list_events(scheduled["run_id"])]
        assert "homeostasis_report_written" in event_types
        assert "homeostasis_evidence_committed" in event_types
        events = supervisor.state_store.list_events(scheduled["run_id"])
        report_event = next(event for event in events if event["event_type"] == "homeostasis_report_written")
        committed_event = next(event for event in events if event["event_type"] == "homeostasis_evidence_committed")
        for payload in (report_event["payload"]["report"], committed_event["payload"]["diagnostic"]):
            assert payload["trigger_source"] == "autonomous_homeostasis"
            assert payload["observed_pressure"]["recent_failure_count"] == 1
            assert payload["diagnosis"] == report["diagnosis"]
            assert payload["recommended_next_action"] == report["recommended_next_action"]
            assert payload["pressure_resolved"] is False
            assert payload["resolution_decision"]["action"] == "escalate_limited_l2_mirror_investigation"
        follow_up = [
            row
            for row in supervisor.state_store.list_tasks(scheduled["run_id"])
            if row["task_id"] != scheduled["task_id"]
        ][0]
        assert follow_up["status"] == "WAITING_APPROVAL"
        assert follow_up["verifier"]["mode"] == "limited_l2_mirror_investigation"
        assert supervisor.state_store.get_latest_approval(scheduled["run_id"])["status"] == "WAITING"
    finally:
        supervisor.state_store.close()


def test_pressure_resolution_deprioritizes_resolved_repeat_goal(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {
                "subgoals": [
                    {
                        "goal_id": "goal:low_pressure_review",
                        "objective": "Review low pressure status.",
                        "status": "active",
                        "priority": 0.4,
                        "permission_level": "L1",
                        "allowed_actions": ["read_logs", "write_report"],
                    }
                ]
            },
            "world_summary": {"uncertainty_estimate": 0.1, "risk_estimate": 0.1},
            "self_summary": {"error_flags": [], "recent_failures": []},
            "telemetry_summary": {"anomaly_flags": []},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        scheduled = ensure_autonomous_run(supervisor, state_path=state_path)
        assert supervisor.tick_once(scheduled["run_id"])["status"] == "TASK_STARTED"
        completed = supervisor.tick_once(scheduled["run_id"])
        assert completed["status"] == "COMPLETED"
        run = supervisor.state_store.get_run(scheduled["run_id"])
        resolution = run["metadata"]["homeostasis_resolution"]
        assert resolution["action"] == "deprioritize_repeat"
        assert resolution["repeat_priority_multiplier"] == 0.25
        assert resolution["repeat_suppress_until"] > resolution["applied_at"]

        throttled = ensure_autonomous_run(supervisor, state_path=state_path, cooldown_seconds=0)
        assert throttled["status"] == "THROTTLED_RECENT_AUTONOMOUS_RUN"
        assert throttled["recent_run_id"] == scheduled["run_id"]
    finally:
        supervisor.state_store.close()


def test_pressure_resolution_escalates_world_uncertainty_to_deep_think(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {"subgoals": []},
            "world_summary": {
                "uncertainty_estimate": 0.82,
                "risk_estimate": 0.2,
                "latent_hypotheses": [{"claim": "unknown transition"}],
            },
            "self_summary": {"error_flags": [], "recent_failures": []},
            "telemetry_summary": {"anomaly_flags": []},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        scheduled = ensure_autonomous_run(supervisor, state_path=state_path)
        assert supervisor.tick_once(scheduled["run_id"])["status"] == "TASK_STARTED"
        completed = supervisor.tick_once(scheduled["run_id"])
        assert completed["status"] == "TASK_COMPLETED_NEXT_STARTED"
        assert completed["resolution_action"] == "escalate_deep_think"
        next_task = supervisor.state_store.get_task(completed["next_task_id"])
        assert next_task["status"] == "RUNNING"
        assert next_task["verifier"]["runtime_mode"] == "DEEP_THINK"
        assert next_task["verifier"]["read_only"] is True
    finally:
        supervisor.state_store.close()


def test_pressure_resolution_escalates_anomaly_to_waiting_human(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {"subgoals": []},
            "world_summary": {"uncertainty_estimate": 0.2, "risk_estimate": 0.1},
            "self_summary": {"error_flags": [], "recent_failures": []},
            "telemetry_summary": {"anomaly_flags": ["watchdog_repeated_degraded"]},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        scheduled = ensure_autonomous_run(supervisor, state_path=state_path)
        assert supervisor.tick_once(scheduled["run_id"])["status"] == "TASK_STARTED"
        completed = supervisor.tick_once(scheduled["run_id"])
        assert completed["status"] == "WAITING_APPROVAL"
        assert completed["resolution_action"] == "escalate_waiting_human"
        run = supervisor.state_store.get_run(scheduled["run_id"])
        assert run["status"] == "WAITING_APPROVAL"
        assert run["paused_reason"] == "homeostasis_waiting_human"
        approval = supervisor.state_store.get_latest_approval(scheduled["run_id"])
        assert approval["request"]["resolution_action"] == "escalate_waiting_human"
    finally:
        supervisor.state_store.close()


def test_autonomous_tick_stays_quiet_when_no_goal_or_homeostasis_pressure(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "state.json"
    _write_payload(
        state_path,
        {
            "goal_stack": {"subgoals": []},
            "world_summary": {"uncertainty_estimate": 0.2, "risk_estimate": 0.1},
            "self_summary": {"error_flags": [], "recent_failures": []},
            "telemetry_summary": {"anomaly_flags": []},
        },
    )
    supervisor = _supervisor(tmp_path)
    try:
        result = ensure_autonomous_run(supervisor, state_path=state_path)
        assert result["status"] == "NO_AUTONOMOUS_GOAL_PRESSURE"
        assert supervisor.state_store.list_runs() == []
    finally:
        supervisor.state_store.close()

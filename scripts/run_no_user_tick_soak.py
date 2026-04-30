from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.long_run_supervisor import LongRunSupervisor
from core.runtime.service_daemon import tick_runtime_once
from core.runtime.state_store import RuntimeStateStore


SOAK_SCHEMA_VERSION = "conos.no_user_tick_soak/v0.1"


def run_soak(*, ticks_per_scenario: int = 50, runtime_root: str | Path | None = None) -> Dict[str, Any]:
    root = Path(runtime_root) if runtime_root is not None else Path(tempfile.mkdtemp(prefix="conos-no-user-tick-soak-"))
    root.mkdir(parents=True, exist_ok=True)
    quiet = _run_scenario(root=root / "quiet", name="quiet_no_pressure", ticks=ticks_per_scenario, state=_quiet_state())
    pressure = _run_scenario(root=root / "pressure", name="pressure_self_model_failure", ticks=ticks_per_scenario, state=_pressure_state())
    checks = _checks(quiet=quiet, pressure=pressure)
    return {
        "schema_version": SOAK_SCHEMA_VERSION,
        "created_at": time.time(),
        "runtime_root": str(root),
        "ticks_per_scenario": int(ticks_per_scenario),
        "total_ticks": int(ticks_per_scenario) * 2,
        "passed": all(bool(row.get("passed")) for row in checks),
        "checks": checks,
        "scenarios": {
            "quiet_no_pressure": quiet,
            "pressure_self_model_failure": pressure,
        },
    }


def _run_scenario(*, root: Path, name: str, ticks: int, state: Mapping[str, Any]) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    supervisor = LongRunSupervisor(
        state_store=RuntimeStateStore(root / "state.sqlite3"),
        runs_root=root / "runs",
        worker_id=f"no-user-soak:{name}",
    )
    snapshots: list[Dict[str, Any]] = []
    try:
        for index in range(int(ticks)):
            tick = tick_runtime_once(
                supervisor,
                watchdog=None,
                snapshot_path=None,
                max_event_rows=5000,
                autonomous_tick_enabled=True,
                autonomous_state_path=state_path,
                autonomous_cooldown_seconds=3600,
            )
            snapshots.append(
                {
                    "tick": index,
                    "autonomous_status": str(tick.get("autonomous_tick", {}).get("status") or ""),
                    "tick_statuses": [str(row.get("status") or "") for row in tick.get("ticks", []) if isinstance(row, Mapping)],
                    "run_count": int(tick.get("metrics", {}).get("run_count", 0) or 0),
                    "task_status_counts": dict(tick.get("metrics", {}).get("task_status_counts", {}) or {}),
                    "event_count": _total_event_count(supervisor),
                }
            )
        runs = supervisor.state_store.list_runs()
        tasks = [task for run in runs for task in supervisor.state_store.list_tasks(str(run.get("run_id", "")))]
        events = [event for run in runs for event in supervisor.state_store.list_events(str(run.get("run_id", "")))]
        reports = _homeostasis_reports(tasks)
        return {
            "name": name,
            "ticks": int(ticks),
            "state_path": str(state_path),
            "run_count": len(runs),
            "task_count": len(tasks),
            "event_count": len(events),
            "run_status_counts": _counts(str(run.get("status") or "") for run in runs),
            "task_status_counts": _counts(str(task.get("status") or "") for task in tasks),
            "event_type_counts": _counts(str(event.get("event_type") or "") for event in events),
            "autonomous_status_counts": _counts(str(row.get("autonomous_status") or "") for row in snapshots),
            "tick_status_counts": _counts(status for row in snapshots for status in row.get("tick_statuses", [])),
            "first_stable_tick": _first_stable_tick(snapshots),
            "event_growth_after_stable": _event_growth_after_stable(snapshots),
            "llm_calls_total": sum(int(report.get("llm_calls", 0) or 0) for report in reports),
            "side_effects_executed_count": sum(1 for report in reports if bool(report.get("side_effects_executed", False))),
            "homeostasis_report_count": len(reports),
            "homeostasis_reports": reports,
            "created_report_paths": [str(task.get("result", {}).get("homeostasis_report_path") or "") for task in tasks if _is_homeostasis_task(task)],
            "sample_ticks": snapshots[:5] + snapshots[-5:],
        }
    finally:
        supervisor.state_store.close()


def _checks(*, quiet: Mapping[str, Any], pressure: Mapping[str, Any]) -> list[Dict[str, Any]]:
    quiet_statuses = dict(quiet.get("autonomous_status_counts", {}) or {})
    pressure_statuses = dict(pressure.get("autonomous_status_counts", {}) or {})
    pressure_reports = list(pressure.get("homeostasis_reports", []) or [])
    checks = [
        _check(
            "quiet_stays_quiet",
            quiet.get("run_count") == 0 and quiet_statuses.get("NO_AUTONOMOUS_GOAL_PRESSURE") == quiet.get("ticks"),
            f"quiet_runs={quiet.get('run_count')} quiet_statuses={quiet_statuses}",
        ),
        _check(
            "pressure_creates_single_autonomous_run",
            pressure.get("run_count") == 1 and pressure_statuses.get("AUTONOMOUS_TASK_SCHEDULED", 0) == 1,
            f"pressure_runs={pressure.get('run_count')} pressure_statuses={pressure_statuses}",
        ),
        _check(
            "does_not_repeat_tasks",
            int(pressure.get("task_count", 0) or 0) <= 2 and pressure_statuses.get("THROTTLED_RECENT_AUTONOMOUS_RUN", 0) >= 1,
            f"pressure_task_count={pressure.get('task_count')} pressure_statuses={pressure_statuses}",
        ),
        _check(
            "does_not_execute_side_effects",
            quiet.get("side_effects_executed_count") == 0 and pressure.get("side_effects_executed_count") == 0,
            f"quiet_side_effects={quiet.get('side_effects_executed_count')} pressure_side_effects={pressure.get('side_effects_executed_count')}",
        ),
        _check(
            "does_not_call_cloud_or_llm",
            quiet.get("llm_calls_total") == 0 and pressure.get("llm_calls_total") == 0,
            f"quiet_llm={quiet.get('llm_calls_total')} pressure_llm={pressure.get('llm_calls_total')}",
        ),
        _check(
            "pressure_writes_report",
            len(pressure_reports) == 1 and bool(pressure_reports[0].get("diagnosis")) and pressure_reports[0].get("pressure_resolved") is False,
            f"pressure_report_count={len(pressure_reports)}",
        ),
        _check(
            "logs_do_not_grow_after_stable",
            quiet.get("event_growth_after_stable") == 0 and pressure.get("event_growth_after_stable") == 0,
            f"quiet_growth={quiet.get('event_growth_after_stable')} pressure_growth={pressure.get('event_growth_after_stable')}",
        ),
        _check(
            "bounded_log_volume",
            int(quiet.get("event_count", 0) or 0) == 0 and int(pressure.get("event_count", 0) or 0) <= 20,
            f"quiet_events={quiet.get('event_count')} pressure_events={pressure.get('event_count')}",
        ),
    ]
    return checks


def _quiet_state() -> Dict[str, Any]:
    return {
        "goal_stack": {"subgoals": []},
        "self_summary": {"error_flags": [], "recent_failures": []},
        "world_summary": {"uncertainty_estimate": 0.2, "risk_estimate": 0.1, "latent_hypotheses": []},
        "telemetry_summary": {"anomaly_flags": []},
    }


def _pressure_state() -> Dict[str, Any]:
    return {
        "goal_stack": {"subgoals": []},
        "self_summary": {
            "error_flags": ["last_action_failed"],
            "recent_failures": [{"action_name": "repo_grep", "failure_type": "timeout"}],
        },
        "world_summary": {"uncertainty_estimate": 0.4, "risk_estimate": 0.2, "latent_hypotheses": []},
        "telemetry_summary": {"anomaly_flags": []},
    }


def _homeostasis_reports(tasks: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    reports: list[Dict[str, Any]] = []
    for task in tasks:
        result = task.get("result", {}) if isinstance(task.get("result"), Mapping) else {}
        report = result.get("homeostasis_report", {}) if isinstance(result.get("homeostasis_report"), Mapping) else {}
        if report:
            reports.append(dict(report))
    return reports


def _is_homeostasis_task(task: Mapping[str, Any]) -> bool:
    result = task.get("result", {}) if isinstance(task.get("result"), Mapping) else {}
    return bool(result.get("homeostasis_report_path"))


def _first_stable_tick(snapshots: list[Mapping[str, Any]]) -> int:
    for row in snapshots:
        status = str(row.get("autonomous_status") or "")
        if status in {"NO_AUTONOMOUS_GOAL_PRESSURE", "THROTTLED_RECENT_AUTONOMOUS_RUN"} and not row.get("tick_statuses"):
            return int(row.get("tick", 0) or 0)
    return -1


def _event_growth_after_stable(snapshots: list[Mapping[str, Any]]) -> int:
    stable_tick = _first_stable_tick(snapshots)
    if stable_tick < 0:
        return 0
    tail = [row for row in snapshots if int(row.get("tick", 0) or 0) >= stable_tick]
    if not tail:
        return 0
    counts = [int(row.get("event_count", 0) or 0) for row in tail]
    return max(counts) - min(counts)


def _total_event_count(supervisor: LongRunSupervisor) -> int:
    return sum(supervisor.state_store.count_events(str(run.get("run_id", ""))) for run in supervisor.state_store.list_runs())


def _counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": str(name), "passed": bool(passed), "detail": str(detail or "")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a no-user autonomous tick soak.")
    parser.add_argument("--ticks-per-scenario", type=int, default=50)
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = run_soak(
        ticks_per_scenario=max(1, int(args.ticks_per_scenario)),
        runtime_root=Path(args.runtime_root).expanduser() if args.runtime_root else None,
    )
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(report.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

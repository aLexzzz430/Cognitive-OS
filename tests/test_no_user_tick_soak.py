from __future__ import annotations

from pathlib import Path

from scripts.run_no_user_tick_soak import run_soak


def test_no_user_tick_soak_contract(tmp_path: Path) -> None:
    report = run_soak(ticks_per_scenario=50, runtime_root=tmp_path / "soak")

    assert report["passed"] is True
    assert report["total_ticks"] == 100
    checks = {row["name"]: row for row in report["checks"]}
    for name in (
        "quiet_stays_quiet",
        "pressure_creates_single_autonomous_run",
        "does_not_repeat_tasks",
        "does_not_execute_side_effects",
        "does_not_call_cloud_or_llm",
        "pressure_writes_report",
        "logs_do_not_grow_after_stable",
        "bounded_log_volume",
    ):
        assert checks[name]["passed"] is True

    quiet = report["scenarios"]["quiet_no_pressure"]
    assert quiet["run_count"] == 0
    assert quiet["event_count"] == 0
    assert quiet["autonomous_status_counts"] == {"NO_AUTONOMOUS_GOAL_PRESSURE": 50}

    pressure = report["scenarios"]["pressure_self_model_failure"]
    assert pressure["run_count"] == 1
    assert pressure["task_count"] <= 2
    assert pressure["homeostasis_report_count"] == 1
    assert pressure["llm_calls_total"] == 0
    assert pressure["side_effects_executed_count"] == 0
    assert pressure["event_growth_after_stable"] == 0

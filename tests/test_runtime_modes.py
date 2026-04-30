from __future__ import annotations

from core.runtime.runtime_modes import (
    RuntimeMode,
    creating_exit_status,
    infer_route_runtime_mode,
    infer_runtime_mode,
    mode_policy_for_mode,
    runtime_mode_catalog,
)


def test_runtime_mode_catalog_contains_product_states() -> None:
    catalog = runtime_mode_catalog()
    modes = {row["mode"] for row in catalog["modes"]}

    assert {
        "STOPPED",
        "SLEEP",
        "IDLE",
        "ROUTINE_RUN",
        "DEEP_THINK",
        "CREATING",
        "ACTING",
        "DREAM",
        "WAITING_HUMAN",
        "DEGRADED_RECOVERY",
    } <= modes
    creating = next(row for row in catalog["modes"] if row["mode"] == "CREATING")
    assert creating["mode_policy"]["permission_policy"]["allowed_capability_layers"] == ["read", "propose_patch"]
    assert "patch_proposal_generated" in creating["mode_policy"]["exit_conditions"]


def test_sleep_is_live_low_power_not_stopped() -> None:
    snapshot = infer_runtime_mode(run={"status": "PAUSED", "heartbeat_status": "PAUSED"})

    assert snapshot.mode == RuntimeMode.SLEEP.value
    assert snapshot.allowed_autonomy["heartbeat"] is True
    assert snapshot.allowed_autonomy["housekeeping"] is True
    assert snapshot.allowed_autonomy["planner_tick"] is False
    assert snapshot.allowed_autonomy["llm_calls"] is False
    assert snapshot.allowed_autonomy["side_effects"] is False
    assert snapshot.mode_policy["llm_budget"]["max_llm_calls"] == 0


def test_stopped_means_service_not_running() -> None:
    snapshot = infer_runtime_mode(service_running=False)

    assert snapshot.mode == RuntimeMode.STOPPED.value
    assert snapshot.allowed_autonomy["heartbeat"] is False


def test_waiting_approval_maps_to_waiting_human() -> None:
    snapshot = infer_runtime_mode(run={"status": "WAITING_APPROVAL"}, tasks=[])

    assert snapshot.mode == RuntimeMode.WAITING_HUMAN.value


def test_degraded_or_zombie_maps_to_recovery() -> None:
    by_watchdog = infer_runtime_mode(watchdog={"status": "DEGRADED", "reason": "ollama_timeout"})
    by_zombie = infer_runtime_mode(run={"status": "RUNNING", "paused_reason": "zombie_suspected"})

    assert by_watchdog.mode == RuntimeMode.DEGRADED_RECOVERY.value
    assert by_zombie.mode == RuntimeMode.DEGRADED_RECOVERY.value


def test_actions_and_routes_map_to_acting_creating_and_deep_think() -> None:
    acting = infer_runtime_mode(active_action={"function_name": "run_test"})
    creating = infer_runtime_mode(active_action={"function_name": "propose_patch"})
    route_creating = infer_route_runtime_mode("patch_proposal")
    route_thinking = infer_route_runtime_mode("root_cause")

    assert acting.mode == RuntimeMode.ACTING.value
    assert creating.mode == RuntimeMode.CREATING.value
    assert route_creating.mode == RuntimeMode.CREATING.value
    assert route_thinking.mode == RuntimeMode.DEEP_THINK.value
    assert route_creating.mode_policy["model_selection"]["prefer_strongest_model"] is True


def test_background_soak_maps_to_dream() -> None:
    snapshot = infer_runtime_mode(soak_sessions=[{"status": "RUNNING", "mode": "infrastructure"}])

    assert snapshot.mode == RuntimeMode.DREAM.value
    assert snapshot.allowed_autonomy["side_effects"] is False


def test_mode_policy_drives_budget_model_selection_and_permissions() -> None:
    sleep = mode_policy_for_mode(RuntimeMode.SLEEP)
    routine = mode_policy_for_mode(RuntimeMode.ROUTINE_RUN)
    deep = mode_policy_for_mode(RuntimeMode.DEEP_THINK)
    acting = mode_policy_for_mode(RuntimeMode.ACTING)

    assert sleep["scheduler"]["planner_tick"] is False
    assert sleep["llm_budget"]["max_llm_calls"] == 0
    assert routine["model_selection"]["model_tier"] == "local_small_or_fast"
    assert deep["model_selection"]["prefer_strongest_model"] is True
    assert acting["permission_policy"]["side_effects_allowed"] is True
    assert "sync_back" in acting["permission_policy"]["approval_required_capability_layers"]


def test_creating_has_explicit_exit_conditions() -> None:
    blocked = creating_exit_status({"creating_tick_count": 1})
    proposal = creating_exit_status({"patch_proposals": [{"proposal_id": "p1"}]})
    timeout = creating_exit_status({"creating_tick_count": 4})

    assert blocked["status"] == "continue_creating"
    assert proposal["status"] == "exit_ready"
    assert proposal["satisfied_exit_conditions"] == ["patch_proposal_generated"]
    assert timeout["status"] == "exit_ready"
    assert "max_mode_ticks_without_artifact" in timeout["satisfied_exit_conditions"]

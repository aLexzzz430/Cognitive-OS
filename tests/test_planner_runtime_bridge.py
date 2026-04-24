from types import SimpleNamespace

from core.orchestration.planner_runtime_bridge import (
    apply_planner_state_patch,
    build_planner_ports,
    consume_planner_runtime_result,
)


class _PlanState:
    def __init__(self):
        self.current_step = object()
        self.calls = []

    def update_context(self, *, tick, reward, discovered_functions):
        self.calls.append(("update_context", tick, reward, discovered_functions))

    def advance_step(self):
        self.calls.append(("advance_step",))

    def fail_current_step(self, *, reason):
        self.calls.append(("fail_current_step", reason))

    def clear_plan(self):
        self.calls.append(("clear_plan",))

    def set_plan(self, plan):
        self.calls.append(("set_plan", plan))


def _loop():
    transition_calls = []
    completed = []
    cancelled = []
    loop = SimpleNamespace(
        _plan_state=_PlanState(),
        _objective_decomposer=SimpleNamespace(name="objective"),
        _plan_reviser=SimpleNamespace(name="reviser"),
        _meta_control=SimpleNamespace(name="meta"),
        _extract_available_functions=lambda obs: ["inspect", obs.get("function", "")],
        _infer_task_family=lambda: "arc",
        _ablation_flags_snapshot=lambda: {"ablate": False},
        _mark_continuity_task_completed=lambda task_id, reason: completed.append((task_id, reason)),
        _mark_continuity_task_cancelled=lambda task_id, reason: cancelled.append((task_id, reason)),
        _build_world_model_context=lambda obs: {"world": obs},
        _build_world_model_transition_priors=lambda obs: {"priors": obs},
        _hypotheses=SimpleNamespace(get_active=lambda: ["hyp-1"]),
        _reliability_tracker=SimpleNamespace(name="reliability"),
        _episode=4,
        _tick=8,
        max_ticks=30,
        _episode_reward=1.25,
        _episode_trace=[{"tick": 7}],
        _pending_replan={"task_id": "task-1"},
        _world_provider_meta={"runtime_env": "arc"},
        _causal_ablation=SimpleNamespace(name="ablation"),
        _learned_dynamics_shadow_predictor=SimpleNamespace(name="predictor"),
        _learned_dynamics_deployment_mode="active",
        _persistent_object_identity_tracker=SimpleNamespace(name="identity"),
        _planner_runtime_log=[],
        _last_planner_runtime_payload={},
        _apply_step_transitions_with_feedback=lambda transitions: transition_calls.append(list(transitions)),
    )
    loop._build_tick_context_frame = lambda obs, continuity: {
        "obs": obs,
        "continuity": continuity,
    }
    loop.transition_calls = transition_calls
    loop.completed = completed
    loop.cancelled = cancelled
    return loop


def test_build_planner_ports_preserves_loop_dependency_bindings():
    loop = _loop()

    ports = build_planner_ports(loop)

    assert ports.plan_state is loop._plan_state
    assert ports.objective_decomposer is loop._objective_decomposer
    assert ports.plan_reviser is loop._plan_reviser
    assert ports.meta_control is loop._meta_control
    assert ports.build_tick_context_frame({"tile": 1}, {"goal": "g"}) == {
        "obs": {"tile": 1},
        "continuity": {"goal": "g"},
    }
    assert ports.extract_available_functions({"function": "move"}) == ["inspect", "move"]
    assert ports.infer_task_family() == "arc"
    assert ports.ablation_flags_snapshot() == {"ablate": False}
    ports.mark_continuity_task_completed("task-2", "done")
    ports.mark_continuity_task_cancelled("task-3", "cancelled")
    assert loop.completed == [("task-2", "done")]
    assert loop.cancelled == [("task-3", "cancelled")]
    assert ports.build_world_model_context({"obs": True}) == {"world": {"obs": True}}
    assert ports.build_world_model_transition_priors({"obs": True}) == {"priors": {"obs": True}}
    assert ports.get_active_hypotheses() == ["hyp-1"]
    assert ports.get_reliability_tracker() is loop._reliability_tracker
    assert ports.get_episode() == 4
    assert ports.get_tick() == 8
    assert ports.get_max_ticks() == 30
    assert ports.get_episode_reward() == 1.25
    assert ports.get_episode_trace() == [{"tick": 7}]
    assert ports.get_pending_replan() == {"task_id": "task-1"}
    assert ports.get_world_provider_meta() == {"runtime_env": "arc"}
    assert ports.get_causal_ablation() is loop._causal_ablation
    assert ports.get_learned_dynamics_predictor() is loop._learned_dynamics_shadow_predictor
    assert ports.get_learned_dynamics_deployment_mode() == "active"
    assert ports.get_persistent_object_identity_tracker() is loop._persistent_object_identity_tracker


def test_apply_planner_state_patch_preserves_transition_and_plan_update_order():
    loop = _loop()

    apply_planner_state_patch(
        loop,
        {
            "update_context": {"tick": 9, "reward": 2.0, "discovered_functions": ["inspect"]},
            "step_transitions": [{"from": "a", "to": "b"}],
            "advance_step": True,
            "mark_failed_reason": "ignored-when-transitions-present",
            "clear_plan": True,
            "set_plan": ["new-plan"],
            "pending_replan": {"task_id": "next"},
        },
    )

    assert loop._plan_state.calls == [
        ("update_context", 9, 2.0, ["inspect"]),
        ("clear_plan",),
        ("set_plan", ["new-plan"]),
    ]
    assert loop.transition_calls == [[{"from": "a", "to": "b"}]]
    assert loop._pending_replan == {"task_id": "next"}

    apply_planner_state_patch(loop, {"advance_step": True, "mark_failed_reason": "bad-step"})

    assert loop._plan_state.calls[-2:] == [
        ("advance_step",),
        ("fail_current_step", "bad-step"),
    ]


def test_consume_planner_runtime_result_applies_patch_and_records_bounded_payload_log():
    loop = _loop()
    loop._planner_runtime_log = [{"old": idx} for idx in range(61)]
    runtime_out = SimpleNamespace(
        selected_action=None,
        state_patch={"advance_step": True},
        decision_flags={"events": ["advance"]},
        telemetry={"phase": "control"},
    )
    fallback_action = {"function": "wait"}

    payload = consume_planner_runtime_result(loop, runtime_out, fallback_action=fallback_action)

    assert payload == {
        "selected_action": fallback_action,
        "state_patch": {"advance_step": True},
        "decision_flags": {"events": ["advance"]},
        "telemetry": {"phase": "control"},
    }
    assert loop._plan_state.calls == [("advance_step",)]
    assert loop._last_planner_runtime_payload == {
        "episode": 4,
        "tick": 8,
        "state_patch": {"advance_step": True},
        "decision_flags": {"events": ["advance"]},
        "telemetry": {"phase": "control"},
    }
    assert len(loop._planner_runtime_log) == 60
    assert loop._planner_runtime_log[-1] == loop._last_planner_runtime_payload

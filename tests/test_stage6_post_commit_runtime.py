from types import SimpleNamespace

from core.orchestration.runtime_stage_contracts import Stage6PostCommitInput
from core.orchestration.stage6_post_commit_runtime import run_stage6_post_commit


class _PlannerRuntime:
    def __init__(self, calls):
        self.calls = calls
        self.tick_kwargs = []

    def tick(self, **kwargs):
        self.calls.append("planner_tick")
        self.tick_kwargs.append(kwargs)
        return {"planner": "out"}


class _GradTracker:
    def __init__(self, calls):
        self.calls = calls
        self.epochs = []

    def on_commit_epoch_end(self, *, epoch):
        self.calls.append("grad_epoch")
        self.epochs.append(epoch)


class _StateManager:
    def __init__(self, calls):
        self.calls = calls
        self.payloads = []

    def commit_tick(self, payload):
        self.calls.append("state_commit")
        self.payloads.append(payload)


class _Loop:
    def __init__(self):
        self.calls = []
        self._episode = 4
        self._tick = 11
        self._planner_runtime = _PlannerRuntime(self.calls)
        self._grad_tracker = _GradTracker(self.calls)
        self._state_mgr = _StateManager(self.calls)
        self.consumed_planner = []
        self.integration_calls = []
        self.learning_calls = []

    def _consume_planner_runtime_result(self, runtime_out, *, fallback_action):
        self.calls.append("consume_planner")
        self.consumed_planner.append((runtime_out, fallback_action))

    def _post_commit_integration(self, committed_ids, obs_before, result):
        self.calls.append("post_commit_integration")
        self.integration_calls.append((committed_ids, obs_before, result))
        return {
            "committed_count": len(committed_ids),
            "surfaced_object_ids": list(committed_ids),
        }

    def _process_graduation_candidates(self):
        self.calls.append("process_graduation")

    def _collect_outcome_learning_signal(self, **kwargs):
        self.calls.append("collect_learning")
        self.learning_calls.append(kwargs)

    def _write_world_model_state(self):
        self.calls.append("write_world_model_state")


def test_stage6_post_commit_runs_post_commit_side_effects_in_order():
    loop = _Loop()
    action = {
        "kind": "tool",
        "payload": {
            "tool_args": {
                "function_name": "inspect",
                "kwargs": {"target": "door"},
            }
        },
    }
    obs_before = {"obs": True}
    result = {"success": True}
    stage_input = Stage6PostCommitInput(
        committed_ids=["obj-1", "obj-2"],
        obs_before=obs_before,
        result=result,
        action_to_use=action,
        reward=1.25,
    )

    output = run_stage6_post_commit(loop, stage_input)

    assert loop.calls == [
        "planner_tick",
        "consume_planner",
        "post_commit_integration",
        "grad_epoch",
        "process_graduation",
        "collect_learning",
        "state_commit",
        "write_world_model_state",
    ]
    assert loop._planner_runtime.tick_kwargs == [
        {
            "phase": "progress",
            "obs": obs_before,
            "selected_action": action,
            "result": result,
            "reward": 1.25,
        }
    ]
    assert loop.consumed_planner == [({"planner": "out"}, action)]
    assert loop.integration_calls == [(["obj-1", "obj-2"], obs_before, result)]
    assert loop._grad_tracker.epochs == [11]
    assert loop.learning_calls == [
        {
            "action_to_use": action,
            "obs_before": obs_before,
            "result": result,
            "reward": 1.25,
        }
    ]
    assert loop._state_mgr.payloads == [
        {
            "episode": 4,
            "tick": 11,
            "action": str(action)[:80],
            "reward": 1.25,
            "committed": 2,
            "post_commit_integration": {
                "committed_count": 2,
                "surfaced_object_ids": ["obj-1", "obj-2"],
            },
        }
    ]
    assert output == {
        "integration_summary": {
            "committed_count": 2,
            "surfaced_object_ids": ["obj-1", "obj-2"],
        }
    }


def test_stage6_post_commit_preserves_empty_commit_flow():
    loop = _Loop()
    stage_input = Stage6PostCommitInput(
        committed_ids=[],
        obs_before={},
        result={},
        action_to_use=SimpleNamespace(name="noop"),
        reward=0.0,
    )

    output = run_stage6_post_commit(loop, stage_input)

    assert output == {
        "integration_summary": {
            "committed_count": 0,
            "surfaced_object_ids": [],
        }
    }
    assert loop._state_mgr.payloads[-1]["committed"] == 0
    assert loop.calls[-1] == "write_world_model_state"

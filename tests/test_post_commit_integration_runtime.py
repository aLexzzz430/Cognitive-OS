from types import SimpleNamespace

from core.orchestration import post_commit_integration_runtime as runtime
from core.orchestration.runtime_stage_contracts import PostCommitIntegrationInput


class _Continuity:
    def __init__(self):
        self.autobiographical = []
        self.memory = []

    def record_autobiographical_summary(self, summary):
        self.autobiographical.append(summary)

    def record_memory_summary(self, **kwargs):
        self.memory.append(kwargs)


class _StateManager:
    def __init__(self):
        self.updates = []

    def update_state(self, patch, *, reason, module):
        self.updates.append((patch, reason, module))


class _Loop:
    def __init__(self):
        self._processed_committed_ids = {"already-seen"}
        self._shared_store = SimpleNamespace(name="shared")
        self._runtime_store = SimpleNamespace(name="runtime")
        self._family_registry = SimpleNamespace(name="families")
        self._confirmed_functions = set()
        self._commit_log = []
        self._teacher = SimpleNamespace(name="teacher")
        self._teacher_log = []
        self._teacher_allows_intervention = lambda: False
        self._tick = 7
        self._episode = 3
        self._continuity = _Continuity()
        self._state_mgr = _StateManager()
        self.procedure_calls = []

    def _get_reward(self, result):
        return result.get("reward", 0.0)

    def _maybe_commit_procedure_chain(self, **kwargs):
        self.procedure_calls.append(kwargs)


def test_post_commit_integration_preserves_empty_commit_contract(monkeypatch):
    called = []

    def fake_integrate(**_kwargs):
        called.append(True)
        return {}

    monkeypatch.setattr(runtime, "integrate_committed_objects", fake_integrate)

    output = runtime.run_post_commit_integration(
        SimpleNamespace(),
        PostCommitIntegrationInput(committed_ids=[], obs_before={}, result={}),
    )

    assert output == {"integration_summary": {"committed_count": 0}}
    assert called == []


def test_post_commit_integration_fans_out_commit_and_records_memory(monkeypatch):
    loop = _Loop()
    captured_kwargs = []
    obs_before = {"tile": "door"}
    result = {"success": True, "reward": 1.5}

    summary = {
        "surfaced_object_ids": ["repr-1"],
        "mechanism_object_ids": ["mech-1"],
        "object_competitions": ["hyp-1"],
        "active_tests": ["test-1"],
        "candidate_tests": [{"object_id": "test-1"}],
        "candidate_programs": [{"program": "inspect"}],
        "candidate_outputs": [{"output": "open"}],
        "planner_prior_object_ids": ["skill-1"],
        "cross_domain_prior_object_ids": ["transfer-1"],
        "current_identity_snapshot": {"object_id": "identity-1"},
        "autobiographical_summary": {"object_id": "auto-1"},
    }

    def fake_integrate(**kwargs):
        captured_kwargs.append(kwargs)
        return dict(summary)

    monkeypatch.setattr(runtime, "integrate_committed_objects", fake_integrate)

    output = runtime.run_post_commit_integration(
        loop,
        PostCommitIntegrationInput(
            committed_ids=["obj-1", "obj-2"],
            obs_before=obs_before,
            result=result,
        ),
    )

    assert output["integration_summary"]["committed_count"] == 2
    assert captured_kwargs == [
        {
            "committed_ids": ["obj-1", "obj-2"],
            "processed_committed_ids": loop._processed_committed_ids,
            "shared_store": loop._shared_store,
            "runtime_store": loop._runtime_store,
            "family_registry": loop._family_registry,
            "confirmed_functions": loop._confirmed_functions,
            "commit_log": loop._commit_log,
            "teacher": loop._teacher,
            "teacher_log": loop._teacher_log,
            "teacher_allows_intervention": loop._teacher_allows_intervention,
            "tick": 7,
            "episode": 3,
            "obs_before": obs_before,
            "result": result,
            "reward": 1.5,
        }
    ]
    assert loop.procedure_calls == [
        {
            "committed_ids": ["obj-1", "obj-2"],
            "obs_before": obs_before,
            "result": result,
        }
    ]
    assert loop._continuity.autobiographical == [{"object_id": "auto-1"}]
    assert loop._continuity.memory == [
        {
            "semantic_memory": {"surfaced_object_ids": ["repr-1"]},
            "procedural_memory": {"planner_prior_object_ids": ["skill-1"]},
            "transfer_memory": {"cross_domain_prior_object_ids": ["transfer-1"]},
        }
    ]
    assert loop._state_mgr.updates == [
        (
            {
                "object_workspace.surfaced_object_ids": ["repr-1"],
                "object_workspace.mechanism_object_ids": ["mech-1"],
                "object_workspace.object_competitions": ["hyp-1"],
                "object_workspace.active_tests": ["test-1"],
                "object_workspace.current_identity_snapshot": {"object_id": "identity-1"},
                "object_workspace.autobiographical_summary": {"object_id": "auto-1"},
                "object_workspace.candidate_tests": [{"object_id": "test-1"}],
                "object_workspace.candidate_programs": [{"program": "inspect"}],
                "object_workspace.candidate_outputs": [{"output": "open"}],
            },
            "workflow:post_commit_object_workspace",
            "core",
        )
    ]

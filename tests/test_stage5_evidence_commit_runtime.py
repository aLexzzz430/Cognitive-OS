from types import SimpleNamespace
from pathlib import Path

from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING, REJECT
from modules.world_model.events import EventType

from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.runtime_stage_contracts import Stage5EvidenceCommitInput
from core.orchestration.stage5_evidence_commit_runtime import run_stage5_evidence_commit
from core.runtime.state_store import RuntimeStateStore
from modules.state.schema import StateSchema


class _Extractor:
    def __init__(self, packets):
        self.packets = list(packets)
        self.calls = []

    def extract(self, payload, result):
        self.calls.append((payload, result))
        return list(self.packets)


class _Validator:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def validate(self, packet):
        self.calls.append(packet)
        decision = self.decisions[len(self.calls) - 1]
        return SimpleNamespace(decision=decision)


class _Committer:
    def __init__(self, committed_ids):
        self.committed_ids = list(committed_ids)
        self.calls = []

    def commit(self, validated):
        self.calls.append(list(validated))
        return list(self.committed_ids)


class _EventBus:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class _StateManager:
    def __init__(self):
        self.updates = []
        self.state = StateSchema.get_default_state()

    def get_state(self):
        return self.state

    def update_state(self, patch, *, reason, module):
        self.updates.append((patch, reason, module))
        for path, value in patch.items():
            target = self.state
            parts = path.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value


def _loop(*, packets, decisions, committed_ids):
    return SimpleNamespace(
        _episode=3,
        _tick=9,
        _extractor=_Extractor(packets),
        _validator=_Validator(decisions),
        _committer=_Committer(committed_ids),
        _event_bus=_EventBus(),
        _event_log=[],
    )


def test_stage5_evidence_commit_validates_commits_and_emits_events():
    packets = [{"packet": "new"}, {"packet": "merge"}, {"packet": "reject"}]
    loop = _loop(
        packets=packets,
        decisions=[ACCEPT_NEW, MERGE_UPDATE_EXISTING, REJECT],
        committed_ids=["obj-1", "obj-2"],
    )
    stage_input = Stage5EvidenceCommitInput(
        action_to_use={"action": "inspect"},
        result={"success": True},
    )

    output = run_stage5_evidence_commit(loop, stage_input)

    assert loop._extractor.calls == [
        (
            {
                "action": {"action": "inspect"},
                "result": {"success": True},
            },
            {"success": True},
        )
    ]
    assert loop._validator.calls == packets
    assert [packet for packet, _ in loop._committer.calls[0]] == packets[:2]
    assert [decision.decision for _, decision in loop._committer.calls[0]] == [
        ACCEPT_NEW,
        MERGE_UPDATE_EXISTING,
    ]
    assert output["committed_ids"] == ["obj-1", "obj-2"]
    assert [packet for packet, _ in output["validated"]] == packets[:2]

    event_types = [event.event_type for event in loop._event_bus.events]
    assert event_types == [
        EventType.COMMIT_WRITTEN,
        EventType.OBJECT_CREATED,
        EventType.OBJECT_CREATED,
    ]
    commit_event = loop._event_bus.events[0]
    assert commit_event.episode == 3
    assert commit_event.tick == 9
    assert commit_event.source_stage == "evidence_commit"
    assert commit_event.data == {
        "committed_count": 2,
        "validated_count": 2,
        "extracted_count": 3,
        "formal_evidence_count": 0,
    }
    assert [event.data for event in loop._event_bus.events[1:]] == [
        {"object_id": "obj-1"},
        {"object_id": "obj-2"},
    ]
    assert loop._event_log == [
        {
            "event_type": "commit_written",
            "episode": 3,
            "tick": 9,
            "data": {
                "committed_count": 2,
                "object_ids": ["obj-1", "obj-2"],
                "formal_evidence_ids": [],
            },
            "source_module": "core",
            "source_stage": "evidence_commit",
        }
    ]


def test_stage5_evidence_commit_rejected_packets_still_emit_commit_summary_without_raw_log():
    loop = _loop(
        packets=[{"packet": "reject"}],
        decisions=[REJECT],
        committed_ids=[],
    )

    output = run_stage5_evidence_commit(
        loop,
        Stage5EvidenceCommitInput(action_to_use={"action": "wait"}, result={}),
    )

    assert output["validated"] == []
    assert output["committed_ids"] == []
    assert output["formal_evidence_ids"] == []
    assert loop._committer.calls == [[]]
    assert len(loop._event_bus.events) == 1
    assert loop._event_bus.events[0].event_type == EventType.COMMIT_WRITTEN
    assert loop._event_bus.events[0].data == {
        "committed_count": 0,
        "validated_count": 0,
        "extracted_count": 1,
        "formal_evidence_count": 0,
    }
    assert loop._event_log == []


def test_stage5_failed_action_creates_goal_pressure_update():
    loop = _loop(
        packets=[],
        decisions=[],
        committed_ids=[],
    )
    loop._state_mgr = _StateManager()

    output = run_stage5_evidence_commit(
        loop,
        Stage5EvidenceCommitInput(
            action_to_use={"function_name": "file_read"},
            result={"success": False, "error_type": "invalid_kwargs", "message": "missing path"},
        ),
    )

    pressure = output["outcome_model_update"]["goal_pressure_update"]
    assert pressure["created_or_updated"] is True
    assert pressure["pressure_type"] == "capability_repair"
    assert pressure["goal_id"] == "goal:capability_repair:file_read"
    assert any(
        patch.get("goal_stack.subgoals", [])[-1]["goal_id"] == "goal:capability_repair:file_read"
        for patch, reason, module in loop._state_mgr.updates
        if reason == "evidence:goal_pressure_update" and module == "goal_runtime"
    )
    assert loop._event_log[-1]["event_type"] == "goal_pressure_update"


def test_stage5_records_formal_evidence_for_accepted_and_rejected_packets(tmp_path: Path):
    packets = [
        {
            "content": {"summary": "accepted packet", "function_name": "inspect"},
            "confidence": 0.8,
            "evidence_id": "packet-accepted",
            "evidence_kind": "successful_tool_invocation",
        },
        {
            "content": {"summary": "rejected packet", "function_name": "inspect"},
            "confidence": 0.7,
            "evidence_id": "packet-rejected",
            "evidence_kind": "failed_tool_invocation",
        },
    ]
    state_store = RuntimeStateStore(tmp_path / "state.sqlite3")
    loop = _loop(
        packets=packets,
        decisions=[ACCEPT_NEW, REJECT],
        committed_ids=["obj-1"],
    )
    loop.run_id = "stage5-ledger-run"
    loop._formal_evidence_ledger_path = tmp_path / "formal_evidence_ledger.jsonl"
    loop._formal_evidence_state_store = state_store
    loop._formal_evidence_recent = []
    loop._formal_evidence_summary = {}
    loop._state_mgr = _StateManager()
    loop._active_tick_context_frame = SimpleNamespace(
        unified_context=UnifiedCognitiveContext.from_parts(current_goal="test", current_task="stage5")
    )

    output = run_stage5_evidence_commit(
        loop,
        Stage5EvidenceCommitInput(
            action_to_use={"function_name": "inspect"},
            result={"success": True, "state": "INSPECTED"},
        ),
    )

    state_store.close()
    assert len(output["formal_evidence_ids"]) == 2
    assert output["formal_evidence_summary"]["object_layer_evidence"] is True
    assert len(loop._formal_evidence_recent) == 2
    assert loop._formal_evidence_recent[0]["formal_commit"]["committed_object_id"] == "obj-1"
    assert loop._formal_evidence_recent[1]["status"] == "rejected"
    assert loop._active_tick_context_frame.unified_context.evidence_queue[-1]["evidence_id"] == output["formal_evidence_ids"][-1]
    formal_update = next(
        patch
        for patch, _, _ in loop._state_mgr.updates
        if "object_workspace.formal_evidence_ledger" in patch
    )
    assert formal_update["object_workspace.formal_evidence_ledger"]["last_evidence_id"] == output["formal_evidence_ids"][-1]
    assert output["outcome_model_update"]["state_applied"] is True
    assert output["outcome_model_update"]["outcome"] == "success"
    assert loop._last_outcome_model_update["action_name"] == "inspect"
    assert any(
        patch.get("world_summary.observed_facts", [])[-1]["action"] == "inspect"
        for patch, reason, module in loop._state_mgr.updates
        if reason == "evidence:outcome_model_update:world" and module == "world_model"
    )
    assert any(
        patch.get("self_summary.capability_estimate", {}).get("inspect", {}).get("successes") == 1
        for patch, reason, module in loop._state_mgr.updates
        if reason == "evidence:outcome_model_update:self" and module == "learning"
    )
    assert loop._event_log[-1]["event_type"] == "outcome_model_update"

    reopened = RuntimeStateStore(tmp_path / "state.sqlite3")
    rows = reopened.list_evidence_entries(run_id="stage5-ledger-run", limit=10)
    reopened.close()
    assert {row["evidence_id"] for row in rows} == set(output["formal_evidence_ids"])

from types import SimpleNamespace

from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING, REJECT
from modules.world_model.events import EventType

from core.orchestration.runtime_stage_contracts import Stage5EvidenceCommitInput
from core.orchestration.stage5_evidence_commit_runtime import run_stage5_evidence_commit


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

    assert output == {"validated": [], "committed_ids": []}
    assert loop._committer.calls == [[]]
    assert len(loop._event_bus.events) == 1
    assert loop._event_bus.events[0].event_type == EventType.COMMIT_WRITTEN
    assert loop._event_bus.events[0].data == {
        "committed_count": 0,
        "validated_count": 0,
        "extracted_count": 1,
    }
    assert loop._event_log == []

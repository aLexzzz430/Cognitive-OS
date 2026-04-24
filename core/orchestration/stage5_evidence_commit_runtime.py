from __future__ import annotations

from typing import Any, Dict, List, Tuple

from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING
from modules.world_model.events import EventType, WorldModelEvent

from core.orchestration.runtime_stage_contracts import Stage5EvidenceCommitInput


def run_stage5_evidence_commit(loop: Any, stage_input: Stage5EvidenceCommitInput) -> Dict[str, Any]:
    action_to_use = stage_input.action_to_use
    result = stage_input.result
    evidence_packets = loop._extractor.extract({'action': action_to_use, 'result': result}, result)
    validated: List[Tuple[Any, Any]] = []
    for packet in evidence_packets:
        decision = loop._validator.validate(packet)
        if getattr(decision, 'decision', None) in (ACCEPT_NEW, MERGE_UPDATE_EXISTING):
            validated.append((packet, decision))

    committed_ids = loop._committer.commit(validated)
    _emit_stage5_commit_events(
        loop,
        committed_ids=committed_ids,
        validated_count=len(validated),
        extracted_count=len(evidence_packets),
    )
    _append_stage5_commit_log(loop, committed_ids=committed_ids)
    return {'validated': validated, 'committed_ids': committed_ids}


def _emit_stage5_commit_events(
    loop: Any,
    *,
    committed_ids: List[str],
    validated_count: int,
    extracted_count: int,
) -> None:
    loop._event_bus.emit(WorldModelEvent(
        event_type=EventType.COMMIT_WRITTEN,
        episode=loop._episode,
        tick=loop._tick,
        data={
            'committed_count': len(committed_ids),
            'validated_count': validated_count,
            'extracted_count': extracted_count,
        },
        source_stage='evidence_commit',
    ))
    for obj_id in committed_ids:
        loop._event_bus.emit(WorldModelEvent(
            event_type=EventType.OBJECT_CREATED,
            episode=loop._episode,
            tick=loop._tick,
            data={'object_id': obj_id},
            source_stage='evidence_commit',
        ))


def _append_stage5_commit_log(loop: Any, *, committed_ids: List[str]) -> None:
    if not committed_ids:
        return
    loop._event_log.append({
        'event_type': 'commit_written',
        'episode': loop._episode,
        'tick': loop._tick,
        'data': {
            'committed_count': len(committed_ids),
            'object_ids': committed_ids,
        },
        'source_module': 'core',
        'source_stage': 'evidence_commit',
    })

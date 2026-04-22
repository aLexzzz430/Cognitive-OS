from __future__ import annotations

from typing import Any, Dict, List, Tuple

from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING
from modules.world_model.events import EventType, WorldModelEvent


def commit_consolidation_candidates(loop: Any, report: Any, *, episode: int) -> Any:
    """Route consolidation-produced proposals through the formal write path."""
    candidates = [
        dict(candidate)
        for candidate in list(getattr(report, "new_memory_candidates", []) or [])
        if isinstance(candidate, dict)
    ]
    if not candidates:
        report.promoted_count = 0
        report.validated_promotion_count = 0
        report.promoted_object_ids = []
        return report

    validated: List[Tuple[Dict[str, Any], Any]] = []
    for proposal in candidates:
        decision = loop._validator.validate(proposal)
        if getattr(decision, "decision", None) in (ACCEPT_NEW, MERGE_UPDATE_EXISTING):
            validated.append((proposal, decision))

    committed_ids = loop._committer.commit(validated, top_k=len(validated)) if validated else []
    report.validated_promotion_count = len(validated)
    report.promoted_count = len(committed_ids)
    report.promoted_object_ids = list(committed_ids)

    if not committed_ids:
        return report

    tick = int(getattr(loop, "_tick", 0) or 0)
    if hasattr(loop, "_event_bus") and loop._event_bus is not None:
        loop._event_bus.emit(
            WorldModelEvent(
                event_type=EventType.COMMIT_WRITTEN,
                episode=episode,
                tick=tick,
                data={
                    "committed_count": len(committed_ids),
                    "validated_count": len(validated),
                    "candidate_count": len(candidates),
                    "object_ids": list(committed_ids),
                },
                source_stage="episode_consolidation",
            )
        )
        for object_id in committed_ids:
            loop._event_bus.emit(
                WorldModelEvent(
                    event_type=EventType.OBJECT_CREATED,
                    episode=episode,
                    tick=tick,
                    data={"object_id": object_id},
                    source_stage="episode_consolidation",
                )
            )

    if hasattr(loop, "_event_log") and loop._event_log is not None:
        loop._event_log.append(
            {
                "event_type": "consolidation_commit_written",
                "episode": episode,
                "tick": tick,
                "data": {
                    "candidate_count": len(candidates),
                    "validated_count": len(validated),
                    "committed_count": len(committed_ids),
                    "object_ids": list(committed_ids),
                },
                "source_module": "core",
                "source_stage": "episode_consolidation",
            }
        )

    return report

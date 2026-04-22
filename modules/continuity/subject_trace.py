from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, (tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class SubjectTraceEvent:
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind or ""),
            "payload": dict(self.payload or {}),
        }


class SubjectTrace:
    """Track continuity-relevant subject events across episodes without forming a second control chain."""

    def __init__(self, subject_id: str):
        self._subject_id = str(subject_id or "unknown")
        self._events: List[SubjectTraceEvent] = []

    def record_tick(self, snapshot: Dict[str, Any]) -> None:
        payload = snapshot if isinstance(snapshot, dict) else {}
        self._events.append(
            SubjectTraceEvent(
                kind="tick",
                payload={
                    "goal_count": int(payload.get("active_goal_count", 0) or 0),
                    "running_experiments": int(payload.get("running_experiments", 0) or 0),
                },
            )
        )
        del self._events[:-200]

    def record_commitments(self, commitments: List[Dict[str, Any]]) -> None:
        self._events.append(
            SubjectTraceEvent(
                kind="commitments",
                payload={"items": _normalize_list([item.get("commitment", "") for item in commitments if isinstance(item, dict)])},
            )
        )
        del self._events[:-200]

    def record_autobiographical_summary(self, summary: Dict[str, Any]) -> None:
        if not isinstance(summary, dict) or not summary:
            return
        self._events.append(
            SubjectTraceEvent(
                kind="autobiographical",
                payload={
                    "summary": str(summary.get("summary", "") or ""),
                    "episode_refs": _normalize_list(summary.get("episode_refs", [])),
                    "identity_implications": _normalize_list(summary.get("identity_implications", [])),
                },
            )
        )
        del self._events[:-200]

    def summarize(self) -> Dict[str, Any]:
        commitment_count = 0
        autobiographical_count = 0
        tick_count = 0
        last_identity_thread = ""
        for event in self._events:
            if event.kind == "commitments":
                commitment_count += len(_normalize_list(event.payload.get("items", [])))
            elif event.kind == "autobiographical":
                autobiographical_count += 1
                if not last_identity_thread:
                    items = _normalize_list(event.payload.get("identity_implications", []))
                    if items:
                        last_identity_thread = items[0]
            elif event.kind == "tick":
                tick_count += 1
        continuity_score = max(0.0, min(1.0, 0.3 + min(0.3, commitment_count * 0.05) + min(0.2, autobiographical_count * 0.08) + min(0.2, tick_count * 0.01)))
        return {
            "subject_id": self._subject_id,
            "event_count": len(self._events),
            "commitment_count": commitment_count,
            "autobiographical_count": autobiographical_count,
            "continuity_score": continuity_score,
            "last_identity_thread": last_identity_thread,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self._subject_id,
            "events": [event.to_dict() for event in self._events[-50:]],
            "summary": self.summarize(),
        }

from __future__ import annotations

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


def build_autobiographical_summary(
    *,
    continuity_snapshot: Dict[str, Any],
    explicit_summary: Dict[str, Any] | None = None,
    recent_failure_modes: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    continuity = continuity_snapshot if isinstance(continuity_snapshot, dict) else {}
    base = explicit_summary if isinstance(explicit_summary, dict) else {}
    continuity_summary = continuity.get("autobiographical_summary", {})
    continuity_summary = continuity_summary if isinstance(continuity_summary, dict) else {}

    summary = dict(continuity_summary)
    summary.update(base)

    continuity_markers = summary.get("continuity_markers", {})
    continuity_markers = dict(continuity_markers) if isinstance(continuity_markers, dict) else {}
    identity_implications = _normalize_list(summary.get("identity_implications", []))

    if not identity_implications and continuity_markers:
        identity_implications = [f"{key}:{value}" for key, value in continuity_markers.items() if str(value or "").strip()]

    if recent_failure_modes:
        dominant = str(recent_failure_modes[0].get("failure_mode", "") or "") if recent_failure_modes and isinstance(recent_failure_modes[0], dict) else ""
        if dominant and dominant not in identity_implications:
            identity_implications.append(f"recent_failure:{dominant}")

    episode_refs = _normalize_list(summary.get("episode_refs", []))
    lesson = str(
        continuity_markers.get("lesson")
        or continuity_markers.get("discovery")
        or summary.get("summary", "")
        or ""
    ).strip()
    narrative_parts = []
    if lesson:
        narrative_parts.append(lesson)
    if episode_refs:
        narrative_parts.append(f"episodes={','.join(episode_refs[:2])}")
    if identity_implications:
        narrative_parts.append(f"identity={identity_implications[0]}")

    return {
        "summary": str(summary.get("summary") or lesson or ""),
        "episode_refs": episode_refs,
        "continuity_markers": continuity_markers,
        "identity_implications": identity_implications,
        "narrative": "; ".join(part for part in narrative_parts if part),
    }

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Iterable, Mapping, Sequence


HYPOTHESIS_LIFECYCLE_VERSION = "conos.hypothesis_lifecycle/v1"
ACTIVE_STATUSES = {"active", "supported", "weakened"}
TERMINAL_STATUSES = {"confirmed", "rejected"}
MAX_REVISION_RATE = 0.35


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _string_list(value: Any) -> list[str]:
    if value is None:
        raw_items: Iterable[Any] = []
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]
    result: list[str] = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def hypothesis_status(*, posterior: float, support_count: int, contradiction_count: int) -> str:
    if posterior >= 0.85 and support_count >= max(2, contradiction_count + 1):
        return "confirmed"
    if posterior <= 0.15 and contradiction_count >= max(2, support_count + 1):
        return "rejected"
    if posterior >= 0.62 and support_count > contradiction_count:
        return "supported"
    if posterior <= 0.38 and contradiction_count > support_count:
        return "weakened"
    return "active"


def normalize_hypothesis(
    *,
    claim: str,
    hypothesis_id: str = "",
    run_id: str = "",
    task_family: str = "local_machine",
    family: str = "generic",
    confidence: float = 0.5,
    evidence_refs: Sequence[str] = (),
    competing_with: Sequence[str] = (),
    predictions: Mapping[str, Any] | None = None,
    falsifiers: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: float | None = None,
) -> Dict[str, Any]:
    text = str(claim or "").strip()
    if not text:
        raise ValueError("hypothesis claim is required")
    posterior = _clamp01(confidence)
    created = float(created_at or time.time())
    payload = {
        "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
        "hypothesis_id": str(hypothesis_id or ""),
        "run_id": str(run_id or ""),
        "task_family": str(task_family or "local_machine"),
        "family": str(family or "generic"),
        "claim": text,
        "status": hypothesis_status(posterior=posterior, support_count=0, contradiction_count=0),
        "confidence": posterior,
        "prior": posterior,
        "posterior": posterior,
        "support_count": 0,
        "contradiction_count": 0,
        "evidence_refs": _string_list(evidence_refs),
        "competing_with": _string_list(competing_with),
        "predictions": dict(predictions or {}),
        "falsifiers": dict(falsifiers or {}),
        "metadata": dict(metadata or {}),
        "created_at": created,
        "updated_at": created,
    }
    if not payload["hypothesis_id"]:
        id_payload = dict(payload)
        id_payload.pop("hypothesis_id", None)
        payload["hypothesis_id"] = f"hyp_{_json_hash(id_payload)[:16]}"
    return payload


def apply_hypothesis_evidence(
    hypothesis: Mapping[str, Any],
    *,
    signal: str,
    evidence_refs: Sequence[str] = (),
    strength: float = 0.2,
    rationale: str = "",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    row = dict(hypothesis or {})
    old_posterior = _clamp01(row.get("posterior", row.get("confidence", 0.5)))
    revision_rate = min(MAX_REVISION_RATE, max(0.02, abs(float(strength or 0.0))))
    normalized_signal = str(signal or "neutral").strip().lower()
    support_count = int(row.get("support_count", 0) or 0)
    contradiction_count = int(row.get("contradiction_count", 0) or 0)
    if normalized_signal in {"support", "supports", "positive", "confirm"}:
        new_posterior = old_posterior + (1.0 - old_posterior) * revision_rate
        support_count += 1
        direction = "support"
    elif normalized_signal in {"contradict", "contradicts", "negative", "refute"}:
        new_posterior = old_posterior - old_posterior * revision_rate
        contradiction_count += 1
        direction = "contradiction"
    else:
        new_posterior = old_posterior
        direction = "neutral"
    new_posterior = _clamp01(new_posterior)
    refs = _string_list(row.get("evidence_refs", [])) + [
        ref for ref in _string_list(evidence_refs) if ref not in _string_list(row.get("evidence_refs", []))
    ]
    row.update(
        {
            "posterior": new_posterior,
            "confidence": new_posterior,
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "status": hypothesis_status(
                posterior=new_posterior,
                support_count=support_count,
                contradiction_count=contradiction_count,
            ),
            "evidence_refs": refs,
            "updated_at": time.time(),
        }
    )
    event = {
        "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
        "hypothesis_id": str(row.get("hypothesis_id") or ""),
        "event_type": f"evidence_{direction}",
        "signal": direction,
        "delta": round(new_posterior - old_posterior, 6),
        "prior_posterior": round(old_posterior, 6),
        "posterior": round(new_posterior, 6),
        "revision_rate": round(revision_rate, 6),
        "evidence_refs": _string_list(evidence_refs),
        "rationale": str(rationale or ""),
        "created_at": row["updated_at"],
    }
    return row, event


def mark_competing(hypotheses: Sequence[Mapping[str, Any]], left_id: str, right_id: str) -> list[Dict[str, Any]]:
    left = str(left_id or "").strip()
    right = str(right_id or "").strip()
    rows = [dict(row) for row in list(hypotheses or []) if isinstance(row, Mapping)]
    if not left or not right or left == right:
        return rows
    for row in rows:
        row_id = str(row.get("hypothesis_id") or "")
        if row_id == left and right not in _string_list(row.get("competing_with", [])):
            row["competing_with"] = _string_list(row.get("competing_with", [])) + [right]
            row["updated_at"] = time.time()
        if row_id == right and left not in _string_list(row.get("competing_with", [])):
            row["competing_with"] = _string_list(row.get("competing_with", [])) + [left]
            row["updated_at"] = time.time()
    return rows


def hypothesis_lifecycle_summary(hypotheses: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(hypotheses or []) if isinstance(row, Mapping)]
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "active")
        counts[status] = counts.get(status, 0) + 1
    active_rows = [row for row in rows if str(row.get("status") or "active") in ACTIVE_STATUSES]
    ranked = sorted(active_rows, key=lambda row: float(row.get("posterior", row.get("confidence", 0.0)) or 0.0), reverse=True)
    competition_edges = []
    seen = set()
    for row in rows:
        row_id = str(row.get("hypothesis_id") or "")
        for other in _string_list(row.get("competing_with", [])):
            key = tuple(sorted([row_id, other]))
            if row_id and other and key not in seen:
                seen.add(key)
                competition_edges.append({"left": key[0], "right": key[1]})
    return {
        "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
        "hypothesis_count": len(rows),
        "active_count": len(active_rows),
        "status_counts": counts,
        "leading_hypothesis_id": str(ranked[0].get("hypothesis_id") or "") if ranked else "",
        "leading_posterior": round(float(ranked[0].get("posterior", ranked[0].get("confidence", 0.0)) or 0.0), 6) if ranked else 0.0,
        "competition_count": len(competition_edges),
        "competition_edges": competition_edges[:20],
        "needs_discriminating_test": bool(len(active_rows) >= 2 and competition_edges),
    }


def build_discriminating_test(
    *,
    hypothesis_a: Mapping[str, Any],
    hypothesis_b: Mapping[str, Any],
    action: Mapping[str, Any],
    expected_if_a: str,
    expected_if_b: str,
    why: str,
    test_id: str = "",
) -> Dict[str, Any]:
    payload = {
        "schema_version": HYPOTHESIS_LIFECYCLE_VERSION,
        "test_id": str(test_id or ""),
        "hypothesis_a": str(hypothesis_a.get("hypothesis_id") or ""),
        "hypothesis_b": str(hypothesis_b.get("hypothesis_id") or ""),
        "action": dict(action or {}),
        "expected_if_a": str(expected_if_a or ""),
        "expected_if_b": str(expected_if_b or ""),
        "why_discriminating": str(why or ""),
        "status": "proposed",
        "created_at": time.time(),
    }
    if not payload["test_id"]:
        payload["test_id"] = f"dtest_{_json_hash(payload)[:16]}"
    return payload

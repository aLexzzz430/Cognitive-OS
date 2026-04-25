from __future__ import annotations

from pathlib import Path

from core.runtime.hypothesis_lifecycle import (
    apply_hypothesis_evidence,
    hypothesis_lifecycle_summary,
    mark_competing,
    normalize_hypothesis,
)
from core.runtime.state_store import RuntimeStateStore


def test_hypothesis_evidence_revision_is_damped() -> None:
    low = normalize_hypothesis(claim="A weak explanation", confidence=0.0)
    revised, event = apply_hypothesis_evidence(
        low,
        signal="support",
        evidence_refs=["file:a.py:1"],
        strength=1.0,
    )

    assert 0.0 < revised["posterior"] <= 0.35
    assert event["delta"] <= 0.35
    assert revised["support_count"] == 1

    high = normalize_hypothesis(claim="A strong explanation", confidence=1.0)
    revised_high, event_high = apply_hypothesis_evidence(
        high,
        signal="contradict",
        evidence_refs=["test:failed"],
        strength=1.0,
    )

    assert 0.65 <= revised_high["posterior"] < 1.0
    assert event_high["delta"] >= -0.35
    assert revised_high["contradiction_count"] == 1


def test_hypothesis_competition_summary_requires_discriminating_test() -> None:
    h1 = normalize_hypothesis(claim="Red object is target", hypothesis_id="hyp_red")
    h2 = normalize_hypothesis(claim="Blue object is target", hypothesis_id="hyp_blue")

    rows = mark_competing([h1, h2], "hyp_red", "hyp_blue")
    summary = hypothesis_lifecycle_summary(rows)

    assert summary["hypothesis_count"] == 2
    assert summary["competition_count"] == 1
    assert summary["needs_discriminating_test"] is True


def test_state_store_persists_hypothesis_lifecycle_events(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = RuntimeStateStore(db_path)
    row = normalize_hypothesis(
        claim="Timeout is caused by too much prompt context",
        hypothesis_id="hyp_context",
        run_id="run-hyp",
        task_family="local_machine",
        family="failure_mode",
        confidence=0.55,
    )
    store.upsert_hypothesis_lifecycle(row)
    store.record_hypothesis_lifecycle_event(
        hypothesis_id="hyp_context",
        run_id="run-hyp",
        event_type="hypothesis_created",
        evidence_ref="note:001",
        delta=0.0,
        payload={"note": "seed"},
    )

    rows = store.list_hypothesis_lifecycle(run_id="run-hyp")
    events = store.list_hypothesis_lifecycle_events(hypothesis_id="hyp_context")
    store.close()

    assert rows[0]["hypothesis_id"] == "hyp_context"
    assert rows[0]["posterior"] == 0.55
    assert events[0]["event_type"] == "hypothesis_created"
    assert events[0]["payload"]["note"] == "seed"

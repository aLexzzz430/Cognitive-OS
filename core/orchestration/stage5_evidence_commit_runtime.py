from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.cognition.goal_pressure import build_goal_pressure_update
from core.cognition.outcome_model_update import build_outcome_model_update
from core.cognition.unified_context import UnifiedCognitiveContext
from core.runtime.evidence_ledger import (
    FORMAL_EVIDENCE_LEDGER_VERSION,
    FormalEvidenceLedger,
    apply_evidence_entries_to_unified_context,
    build_action_result_evidence_entry,
    build_stage5_evidence_entry,
    compact_evidence_entries_for_context,
)
from modules.governance.object_store import ACCEPT_NEW, MERGE_UPDATE_EXISTING
from modules.world_model.events import EventType, WorldModelEvent

from core.orchestration.runtime_stage_contracts import Stage5EvidenceCommitInput


def run_stage5_evidence_commit(loop: Any, stage_input: Stage5EvidenceCommitInput) -> Dict[str, Any]:
    action_to_use = stage_input.action_to_use
    result = stage_input.result
    evidence_packets = loop._extractor.extract({'action': action_to_use, 'result': result}, result)
    validated: List[Tuple[Any, Any]] = []
    decisions: List[Any] = []
    for packet in evidence_packets:
        decision = loop._validator.validate(packet)
        decisions.append(decision)
        if getattr(decision, 'decision', None) in (ACCEPT_NEW, MERGE_UPDATE_EXISTING):
            validated.append((packet, decision))

    committed_ids = loop._committer.commit(validated)
    formal_entries = _record_formal_evidence_entries(
        loop,
        action_to_use=action_to_use,
        result=result,
        evidence_packets=list(evidence_packets),
        decisions=decisions,
        committed_ids=committed_ids,
    )
    formal_evidence_ids = [str(entry.get("evidence_id") or "") for entry in formal_entries if entry.get("evidence_id")]
    formal_evidence_refs = [
        {
            "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
            "evidence_id": str(entry.get("evidence_id") or ""),
            "ledger_hash": str(entry.get("ledger_hash") or ""),
            "claim": str(entry.get("claim") or ""),
            "status": str(entry.get("status") or ""),
        }
        for entry in formal_entries
    ]
    formal_evidence_summary = _formal_evidence_summary(loop, formal_entries)
    _emit_stage5_commit_events(
        loop,
        committed_ids=committed_ids,
        validated_count=len(validated),
        extracted_count=len(evidence_packets),
        formal_evidence_count=len(formal_entries),
    )
    _append_stage5_commit_log(
        loop,
        committed_ids=committed_ids,
        formal_evidence_ids=formal_evidence_ids,
    )
    outcome_model_update = _apply_outcome_model_update(
        loop,
        action_to_use=action_to_use,
        result=result,
        formal_entries=formal_entries,
    )
    return {
        'validated': validated,
        'committed_ids': committed_ids,
        'formal_evidence_ids': formal_evidence_ids,
        'formal_evidence_refs': formal_evidence_refs,
        'formal_evidence_summary': formal_evidence_summary,
        'outcome_model_update': outcome_model_update,
    }


def _record_formal_evidence_entries(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    result: Dict[str, Any],
    evidence_packets: List[Any],
    decisions: List[Any],
    committed_ids: List[str],
) -> List[Dict[str, Any]]:
    ledger_path = getattr(loop, "_formal_evidence_ledger_path", None)
    state_store = getattr(loop, "_formal_evidence_state_store", None)
    if ledger_path is None and state_store is None:
        _surface_existing_formal_evidence_ref(loop, result)
        return []

    committed_iter = iter(list(committed_ids or []))
    entries: List[Dict[str, Any]] = []
    if evidence_packets:
        for packet, decision in zip(evidence_packets, decisions):
            decision_name = str(getattr(decision, "decision", "") or "")
            committed_object_id = next(committed_iter, "") if decision_name in (ACCEPT_NEW, MERGE_UPDATE_EXISTING) else ""
            entries.append(
                build_stage5_evidence_entry(
                    run_id=str(getattr(loop, "run_id", "") or ""),
                    action=action_to_use if isinstance(action_to_use, dict) else {},
                    result=result if isinstance(result, dict) else {},
                    packet=packet,
                    validation_decision=decision,
                    committed_object_id=committed_object_id,
                    episode=int(getattr(loop, "_episode", 0) or 0),
                    tick=int(getattr(loop, "_tick", 0) or 0),
                )
            )
    elif not _surface_existing_formal_evidence_ref(loop, result):
        entries.append(
            build_action_result_evidence_entry(
                run_id=str(getattr(loop, "run_id", "") or ""),
                action=action_to_use if isinstance(action_to_use, dict) else {},
                result=result if isinstance(result, dict) else {},
                episode=int(getattr(loop, "_episode", 0) or 0),
                tick=int(getattr(loop, "_tick", 0) or 0),
            )
        )
    if not entries:
        return []

    ledger = FormalEvidenceLedger(ledger_path or "runtime/runs/unknown/formal_evidence_ledger.jsonl", state_store=state_store)
    recorded = [ledger.record(entry) for entry in entries]
    _update_loop_formal_evidence_state(loop, recorded)
    return recorded


def _surface_existing_formal_evidence_ref(loop: Any, result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    formal_ref = result.get("formal_evidence_ref", {})
    if not isinstance(formal_ref, dict) or not formal_ref.get("evidence_id"):
        return False
    compact = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "evidence_id": str(formal_ref.get("evidence_id") or ""),
        "ledger_hash": str(formal_ref.get("ledger_hash") or ""),
        "claim": "External surface already recorded formal evidence for this action.",
        "evidence_type": "surface_recorded",
        "source_refs": [f"formal_evidence:{formal_ref.get('evidence_id')}"],
        "confidence": 0.7,
        "status": "recorded",
    }
    _update_loop_formal_evidence_state(loop, [compact])
    return True


def _update_loop_formal_evidence_state(loop: Any, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    recent = [
        dict(row)
        for row in list(getattr(loop, "_formal_evidence_recent", []) or [])
        if isinstance(row, dict)
    ]
    recent.extend(entries)
    recent = recent[-50:]
    loop._formal_evidence_recent = recent
    compact = compact_evidence_entries_for_context(recent, limit=12)
    summary = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "recent_entry_count": len(compact),
        "last_evidence_id": compact[-1]["evidence_id"] if compact else "",
        "ledger_path": str(getattr(loop, "_formal_evidence_ledger_path", "") or ""),
        "object_layer_evidence": True,
    }
    loop._formal_evidence_summary = summary

    active_frame = getattr(loop, "_active_tick_context_frame", None)
    unified_context = getattr(active_frame, "unified_context", None)
    if isinstance(unified_context, UnifiedCognitiveContext):
        active_frame.unified_context = apply_evidence_entries_to_unified_context(unified_context, entries)

    state_mgr = getattr(loop, "_state_mgr", None)
    if state_mgr is not None and hasattr(state_mgr, "update_state"):
        state_mgr.update_state(
            {
                "object_workspace.formal_evidence_ledger": summary,
                "object_workspace.formal_evidence_recent": compact,
            },
            reason="evidence:formal_ledger_update",
            module="core.runtime.evidence_ledger",
        )


def _formal_evidence_summary(loop: Any, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if entries:
        return {
            "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
            "entry_count": len(entries),
            "last_evidence_id": str(entries[-1].get("evidence_id") or ""),
            "ledger_path": str(getattr(loop, "_formal_evidence_ledger_path", "") or ""),
            "object_layer_evidence": True,
        }
    return dict(getattr(loop, "_formal_evidence_summary", {}) or {})


def _apply_outcome_model_update(
    loop: Any,
    *,
    action_to_use: Dict[str, Any],
    result: Dict[str, Any],
    formal_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    state_mgr = getattr(loop, "_state_mgr", None)
    existing_state: Dict[str, Any] = {}
    if state_mgr is not None and hasattr(state_mgr, "get_state"):
        try:
            loaded = state_mgr.get_state()
            if isinstance(loaded, dict):
                existing_state = loaded
        except Exception:
            existing_state = {}

    update = build_outcome_model_update(
        action=action_to_use if isinstance(action_to_use, dict) else {},
        result=result if isinstance(result, dict) else {},
        evidence_entries=formal_entries,
        existing_state=existing_state,
        episode=int(getattr(loop, "_episode", 0) or 0),
        tick=int(getattr(loop, "_tick", 0) or 0),
    )
    summary = update.to_summary()
    loop._last_outcome_model_update = summary
    recent = [
        dict(row)
        for row in list(getattr(loop, "_outcome_model_update_recent", []) or [])
        if isinstance(row, dict)
    ]
    recent.append(summary)
    loop._outcome_model_update_recent = recent[-50:]

    if state_mgr is None or not hasattr(state_mgr, "update_state"):
        summary["state_applied"] = False
        summary["goal_pressure_update"] = {
            "created_or_updated": False,
            "reason": "state_manager_unavailable",
        }
        return summary

    state_mgr.update_state(
        update.world_patch,
        reason="evidence:outcome_model_update:world",
        module="world_model",
    )
    state_mgr.update_state(
        update.self_patch,
        reason="evidence:outcome_model_update:self",
        module="learning",
    )
    state_mgr.update_state(
        update.learning_patch,
        reason="evidence:outcome_model_update:learning",
        module="learning",
    )
    updated_state: Dict[str, Any] = {}
    if hasattr(state_mgr, "get_state"):
        try:
            loaded = state_mgr.get_state()
            if isinstance(loaded, dict):
                updated_state = loaded
        except Exception:
            updated_state = {}
    goal_pressure = build_goal_pressure_update(
        outcome_update=update,
        existing_state=updated_state,
        episode=int(getattr(loop, "_episode", 0) or 0),
        tick=int(getattr(loop, "_tick", 0) or 0),
    )
    goal_pressure_summary = goal_pressure.to_summary()
    if goal_pressure.created_or_updated:
        state_mgr.update_state(
            goal_pressure.goal_patch,
            reason="evidence:goal_pressure_update",
            module="goal_runtime",
        )
        loop._last_goal_pressure_update = goal_pressure_summary
        recent_goal_pressure = [
            dict(row)
            for row in list(getattr(loop, "_goal_pressure_update_recent", []) or [])
            if isinstance(row, dict)
        ]
        recent_goal_pressure.append(goal_pressure_summary)
        loop._goal_pressure_update_recent = recent_goal_pressure[-50:]
    summary["goal_pressure_update"] = goal_pressure_summary
    if hasattr(loop, "_event_log"):
        loop._event_log.append(dict(update.audit_event))
        if goal_pressure.created_or_updated:
            loop._event_log.append(dict(goal_pressure.audit_event))
    summary["state_applied"] = True
    return summary


def _emit_stage5_commit_events(
    loop: Any,
    *,
    committed_ids: List[str],
    validated_count: int,
    extracted_count: int,
    formal_evidence_count: int,
) -> None:
    loop._event_bus.emit(WorldModelEvent(
        event_type=EventType.COMMIT_WRITTEN,
        episode=loop._episode,
        tick=loop._tick,
        data={
            'committed_count': len(committed_ids),
            'validated_count': validated_count,
            'extracted_count': extracted_count,
            'formal_evidence_count': formal_evidence_count,
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


def _append_stage5_commit_log(loop: Any, *, committed_ids: List[str], formal_evidence_ids: List[str]) -> None:
    if not committed_ids and not formal_evidence_ids:
        return
    loop._event_log.append({
        'event_type': 'commit_written',
        'episode': loop._episode,
        'tick': loop._tick,
        'data': {
            'committed_count': len(committed_ids),
            'object_ids': committed_ids,
            'formal_evidence_ids': formal_evidence_ids,
        },
        'source_module': 'core',
        'source_stage': 'evidence_commit',
    })

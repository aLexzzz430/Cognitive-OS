from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from core.cognition.unified_context import UnifiedCognitiveContext


FORMAL_EVIDENCE_LEDGER_VERSION = "conos.formal_evidence_ledger/v1"
MAX_LEDGER_STRING_CHARS = 4000
MAX_LEDGER_LIST_ITEMS = 80
MAX_LEDGER_DICT_ITEMS = 80


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _clip(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "<truncated:depth>"
    if isinstance(value, str):
        if len(value) <= MAX_LEDGER_STRING_CHARS:
            return value
        return value[:MAX_LEDGER_STRING_CHARS] + f"...<truncated:{len(value) - MAX_LEDGER_STRING_CHARS}>"
    if isinstance(value, Mapping):
        items = list(value.items())[:MAX_LEDGER_DICT_ITEMS]
        payload = {str(key): _clip(item, depth=depth + 1) for key, item in items}
        if len(value) > len(items):
            payload["_truncated_items"] = len(value) - len(items)
        return payload
    if isinstance(value, (list, tuple, set)):
        rows = list(value)[:MAX_LEDGER_LIST_ITEMS]
        payload = [_clip(item, depth=depth + 1) for item in rows]
        if len(value) > len(rows):
            payload.append({"_truncated_items": len(value) - len(rows)})
        return payload
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _string_refs(values: Iterable[Any]) -> list[str]:
    refs: list[str] = []
    seen = set()
    for value in values:
        ref = str(value or "").strip()
        if not ref or ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
    return refs


def _source_refs_for(action_name: str, action_args: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    if action_name in {"file_read", "file_outline", "file_summary"}:
        path = str(result.get("path") or action_args.get("path") or "")
        start_line = result.get("start_line")
        end_line = result.get("end_line")
        if path and start_line and end_line:
            refs.append(f"file:{path}:{start_line}-{end_line}")
        elif path:
            refs.append(f"file:{path}")
    if action_name == "repo_tree":
        refs.append(f"tree:{result.get('root') or action_args.get('path') or action_args.get('root') or '.'}:depth={result.get('depth') or action_args.get('depth') or 2}")
    if action_name == "repo_find":
        for row in list(result.get("results", []) or [])[:50]:
            if isinstance(row, Mapping) and row.get("path"):
                refs.append(f"{row.get('kind') or 'path'}:{row.get('path')}")
    if action_name == "repo_grep":
        for row in list(result.get("matches", []) or [])[:80]:
            if isinstance(row, Mapping) and row.get("path"):
                line = row.get("line")
                refs.append(f"file:{row.get('path')}:{line}" if line else f"file:{row.get('path')}")
    if action_name == "note_write":
        note = result.get("note", {}) if isinstance(result.get("note"), Mapping) else {}
        refs.extend(note.get("evidence_refs", []) or [])
        if note.get("note_id"):
            refs.append(f"note:{note.get('note_id')}")
    if action_name == "hypothesis_add":
        hypothesis = result.get("hypothesis", {}) if isinstance(result.get("hypothesis"), Mapping) else {}
        refs.extend(hypothesis.get("evidence_refs", []) or [])
        if hypothesis.get("hypothesis_id"):
            refs.append(f"hypothesis:{hypothesis.get('hypothesis_id')}")
    if action_name == "hypothesis_update":
        hypothesis = result.get("hypothesis", {}) if isinstance(result.get("hypothesis"), Mapping) else {}
        event = result.get("hypothesis_event", {}) if isinstance(result.get("hypothesis_event"), Mapping) else {}
        refs.extend(hypothesis.get("evidence_refs", []) or [])
        refs.extend(event.get("evidence_refs", []) or [])
        if hypothesis.get("hypothesis_id"):
            refs.append(f"hypothesis:{hypothesis.get('hypothesis_id')}")
    if action_name == "hypothesis_compete":
        competition = result.get("competition", {}) if isinstance(result.get("competition"), Mapping) else {}
        for key in ("hypothesis_a", "hypothesis_b"):
            if competition.get(key):
                refs.append(f"hypothesis:{competition.get(key)}")
    if action_name == "discriminating_test_add":
        test = result.get("discriminating_test", {}) if isinstance(result.get("discriminating_test"), Mapping) else {}
        if test.get("test_id"):
            refs.append(f"discriminating_test:{test.get('test_id')}")
        for key in ("hypothesis_a", "hypothesis_b"):
            if test.get(key):
                refs.append(f"hypothesis:{test.get(key)}")
    if action_name in {"candidate_files_set", "candidate_files_update"}:
        refs.extend(f"file:{path}" for path in list(result.get("candidate_files", []) or []))
    if action_name in {"apply_patch", "edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        patch_sha = str(result.get("patch_sha256") or "").strip()
        if patch_sha:
            refs.append(f"patch:{patch_sha}")
        refs.extend(f"file:{path}" for path in list(result.get("touched_files", []) or []))
        if result.get("path"):
            refs.append(f"file:{result.get('path')}")
    if action_name in {"run_test", "run_lint", "run_typecheck", "run_build", "read_run_output", "read_test_failure"}:
        if result.get("run_ref"):
            refs.append(f"run_output:{result.get('run_ref')}")
    if action_name == "mirror_plan":
        plan = result.get("sync_plan", {}) if isinstance(result.get("sync_plan"), Mapping) else {}
        if plan.get("plan_id"):
            refs.append(f"sync_plan:{plan.get('plan_id')}")
    if action_name == "mirror_exec":
        command = result.get("mirror_command", {}) if isinstance(result.get("mirror_command"), Mapping) else {}
        if command:
            refs.append(f"mirror_command:returncode={command.get('returncode')}")
    if action_name.startswith("internet_"):
        artifact = result.get("internet_artifact", {}) if isinstance(result.get("internet_artifact"), Mapping) else {}
        if artifact.get("artifact_id"):
            refs.append(f"internet_artifact:{artifact.get('artifact_id')}")
    return _string_refs(refs)


def _evidence_type_for(action_name: str, result: Mapping[str, Any]) -> str:
    if not bool(result.get("success", False)):
        return "failure_observation"
    if action_name in {"repo_tree", "repo_find", "repo_grep", "file_read", "file_outline", "file_summary"}:
        return "codebase_observation"
    if action_name in {
        "note_write",
        "hypothesis_add",
        "hypothesis_update",
        "hypothesis_compete",
        "discriminating_test_add",
        "candidate_files_set",
        "candidate_files_update",
        "investigation_status",
    }:
        return "investigation_state"
    if action_name in {"apply_patch", "edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        return "edit_observation"
    if action_name in {"run_test", "run_lint", "run_typecheck", "run_build", "read_run_output", "read_test_failure"}:
        return "validation_observation"
    if action_name in {"mirror_plan", "mirror_apply"}:
        return "sync_governance"
    if action_name.startswith("internet_"):
        return "internet_observation"
    return "local_machine_observation"


def _claim_for(action_name: str, action_args: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    if not bool(result.get("success", False)):
        return f"{action_name or 'unknown_action'} failed: {str(result.get('failure_reason') or result.get('state') or 'unknown')[:240]}"
    if action_name == "repo_tree":
        return f"Repository tree under {result.get('root') or action_args.get('path') or '.'} has {result.get('entry_count', 0)} visible entries."
    if action_name == "repo_grep":
        return f"Search for {action_args.get('query') or result.get('query')!r} found {result.get('match_count', 0)} match(es)."
    if action_name == "repo_find":
        return f"Find pattern {action_args.get('name_pattern') or action_args.get('pattern') or '*'} returned {result.get('result_count', 0)} result(s)."
    if action_name == "file_read":
        return f"File {result.get('path')} lines {result.get('start_line')}-{result.get('end_line')} were read from {result.get('location')}."
    if action_name in {"file_outline", "file_summary"}:
        return f"File {result.get('path')} was inspected with {action_name}."
    if action_name == "note_write":
        note = result.get("note", {}) if isinstance(result.get("note"), Mapping) else {}
        return f"Investigation note recorded: {str(note.get('content') or '')[:240]}"
    if action_name == "hypothesis_add":
        hypothesis = result.get("hypothesis", {}) if isinstance(result.get("hypothesis"), Mapping) else {}
        return f"Hypothesis recorded: {str(hypothesis.get('claim') or '')[:240]}"
    if action_name == "hypothesis_update":
        hypothesis = result.get("hypothesis", {}) if isinstance(result.get("hypothesis"), Mapping) else {}
        event = result.get("hypothesis_event", {}) if isinstance(result.get("hypothesis_event"), Mapping) else {}
        return (
            f"Hypothesis {hypothesis.get('hypothesis_id')} revised to "
            f"{hypothesis.get('posterior')} by {event.get('signal') or 'neutral'} evidence."
        )
    if action_name == "hypothesis_compete":
        competition = result.get("competition", {}) if isinstance(result.get("competition"), Mapping) else {}
        return (
            f"Hypotheses {competition.get('hypothesis_a')} and "
            f"{competition.get('hypothesis_b')} were marked as competing explanations."
        )
    if action_name == "discriminating_test_add":
        test = result.get("discriminating_test", {}) if isinstance(result.get("discriminating_test"), Mapping) else {}
        return f"Discriminating test {test.get('test_id')} was proposed for competing hypotheses."
    if action_name in {"candidate_files_set", "candidate_files_update"}:
        return f"Candidate file set now contains {len(list(result.get('candidate_files', []) or []))} file(s)."
    if action_name == "apply_patch":
        return f"Bounded patch was applied to {', '.join(list(result.get('touched_files', []) or []))}."
    if action_name in {"edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        return f"{action_name} changed mirror file {result.get('path')}."
    if action_name in {"run_test", "run_lint", "run_typecheck", "run_build"}:
        return f"{action_name} finished with return code {result.get('returncode')}."
    if action_name in {"read_run_output", "read_test_failure"}:
        return f"Validation output {result.get('run_ref')} was read."
    if action_name == "mirror_plan":
        return f"Mirror sync plan was built with state {result.get('state')}."
    if action_name == "mirror_apply":
        return "Approved mirror sync plan was applied to source."
    if action_name.startswith("internet_"):
        return f"Internet action {action_name} produced artifact evidence."
    return f"Local-machine action {action_name or 'unknown_action'} produced state {result.get('state')}."


def _update_for(action_name: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    success = bool(result.get("success", False))
    if not success:
        kind = "failure"
        direction = "contradicts_expected_action"
    elif action_name in {"run_test", "run_lint", "run_typecheck", "run_build"}:
        kind = "validation"
        direction = "supports_candidate" if int(result.get("returncode", 1) or 1) == 0 else "weakens_candidate"
    elif action_name in {"apply_patch", "edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        kind = "edit"
        direction = "produces_candidate_change"
    elif action_name == "hypothesis_add":
        kind = "hypothesis_update"
        direction = "adds_competing_hypothesis"
    elif action_name == "hypothesis_update":
        kind = "hypothesis_update"
        event = result.get("hypothesis_event", {}) if isinstance(result.get("hypothesis_event"), Mapping) else {}
        signal = str(event.get("signal") or "neutral")
        direction = {
            "support": "supports_hypothesis",
            "contradiction": "weakens_hypothesis",
            "neutral": "keeps_hypothesis_stable",
        }.get(signal, "updates_hypothesis")
    elif action_name == "hypothesis_compete":
        kind = "hypothesis_competition"
        direction = "requires_discriminating_test"
    elif action_name == "discriminating_test_add":
        kind = "discriminating_test_update"
        direction = "proposes_discriminating_experiment"
    elif action_name in {"note_write", "candidate_files_set", "candidate_files_update", "investigation_status"}:
        kind = "investigation_update"
        direction = "updates_investigation_state"
    else:
        kind = "observation"
        direction = "adds_evidence"
    return {
        "kind": kind,
        "direction": direction,
        "from_action": action_name,
        "result_state": str(result.get("state") or ""),
    }


def _confidence_for(action_name: str, result: Mapping[str, Any]) -> float:
    if not bool(result.get("success", False)):
        return 0.6
    if action_name in {"run_test", "run_lint", "run_typecheck", "run_build"}:
        return 0.85
    if action_name in {"file_read", "repo_grep", "repo_tree", "repo_find"}:
        return 0.78
    if action_name in {"apply_patch", "edit_replace_range", "edit_insert_after", "create_file", "delete_file"}:
        return 0.72
    return 0.65


def compact_evidence_entries_for_context(
    entries: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> list[Dict[str, Any]]:
    compact: list[Dict[str, Any]] = []
    for entry in list(entries or [])[-max(0, int(limit)) :]:
        if not isinstance(entry, Mapping):
            continue
        compact.append(
            {
                "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
                "evidence_id": str(entry.get("evidence_id") or ""),
                "claim": str(entry.get("claim") or ""),
                "evidence_type": str(entry.get("evidence_type") or ""),
                "source_refs": list(entry.get("source_refs", []) or [])[:20],
                "confidence": float(entry.get("confidence") or 0.0),
                "status": str(entry.get("status") or ""),
                "ledger_hash": str(entry.get("ledger_hash") or ""),
                "formal_commit": dict(entry.get("formal_commit", {}) or {})
                if isinstance(entry.get("formal_commit", {}), Mapping)
                else {},
            }
        )
    return compact


def _mapping_from_packet(packet: Any) -> Dict[str, Any]:
    if is_dataclass(packet):
        try:
            return asdict(packet)
        except TypeError:
            pass
    if isinstance(packet, Mapping):
        return dict(packet)
    content = getattr(packet, "content", None)
    if isinstance(content, Mapping):
        return {
            "content": dict(content),
            "confidence": float(getattr(packet, "confidence", 0.5) or 0.5),
            "evidence_id": str(getattr(packet, "evidence_id", "") or ""),
            "evidence_kind": str(getattr(packet, "evidence_kind", "") or getattr(packet, "type", "") or ""),
            "source_action": getattr(packet, "source_action", None),
            "surface_result": getattr(packet, "surface_result", None),
            "source_refs": list(getattr(packet, "source_refs", []) or []),
        }
    return {"content": {"repr": str(packet)}}


def _decision_name(decision: Any) -> str:
    if decision is None:
        return "not_validated"
    if isinstance(decision, Mapping):
        return str(decision.get("decision") or "")
    return str(getattr(decision, "decision", "") or "")


def _decision_reason(decision: Any) -> str:
    if decision is None:
        return ""
    if isinstance(decision, Mapping):
        return str(decision.get("reason") or "")
    return str(getattr(decision, "reason", "") or "")


def _packet_kind(packet_payload: Mapping[str, Any]) -> str:
    for key in ("evidence_kind", "type", "kind"):
        text = str(packet_payload.get(key) or "").strip()
        if text:
            return text
    content = packet_payload.get("content", {})
    if isinstance(content, Mapping):
        for key in ("evidence_kind", "type", "kind"):
            text = str(content.get(key) or "").strip()
            if text:
                return text
    return "raw_evidence_packet"


def _packet_source_refs(packet_payload: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    refs: list[Any] = []
    refs.extend(packet_payload.get("source_refs", []) or [])
    content = packet_payload.get("content", {})
    if isinstance(content, Mapping):
        refs.extend(content.get("source_refs", []) or [])
        if content.get("function_name"):
            refs.append(f"function:{content.get('function_name')}")
    if packet_payload.get("evidence_id"):
        refs.append(f"packet:{packet_payload.get('evidence_id')}")
    formal_ref = result.get("formal_evidence_ref", {}) if isinstance(result.get("formal_evidence_ref", {}), Mapping) else {}
    if formal_ref.get("evidence_id"):
        refs.append(f"formal_evidence:{formal_ref.get('evidence_id')}")
    return _string_refs(refs)


def build_stage5_evidence_entry(
    *,
    run_id: str,
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    packet: Any,
    validation_decision: Any = None,
    committed_object_id: str = "",
    episode: int = 0,
    tick: int = 0,
) -> Dict[str, Any]:
    packet_payload = _mapping_from_packet(packet)
    decision_name = _decision_name(validation_decision)
    content = packet_payload.get("content", {})
    content = dict(content) if isinstance(content, Mapping) else {"value": str(content)}
    packet_kind = _packet_kind(packet_payload)
    accepted = decision_name in {"accept_new", "merge_update_existing"}
    claim = (
        str(packet_payload.get("claim") or content.get("claim") or content.get("summary") or "")
        or f"Stage5 extracted {packet_kind} from action result."
    )
    status = "committed" if committed_object_id else ("validated" if accepted else "rejected")
    entry = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "evidence_id": "",
        "run_id": str(run_id or ""),
        "task_family": str(result.get("environment_family") or result.get("task_family") or "core_stage5"),
        "evidence_type": f"stage5:{packet_kind}",
        "claim": claim,
        "evidence": {
            "packet": _clip(packet_payload),
            "packet_kind": packet_kind,
            "stage": "evidence_commit",
            "episode": int(episode or 0),
            "tick": int(tick or 0),
        },
        "hypotheses": _clip(content.get("hypotheses", []) if isinstance(content.get("hypotheses", []), list) else []),
        "action": {
            "name": str(action.get("function_name") or action.get("action") or action.get("kind") or ""),
            "payload": _clip(dict(action or {})),
        },
        "result": _clip(dict(result or {})),
        "update": {
            "kind": "formal_validation",
            "direction": "accepted" if accepted else "rejected",
            "validation_decision": decision_name,
            "validation_reason": _decision_reason(validation_decision),
            "committed_object_id": str(committed_object_id or ""),
        },
        "source_refs": _packet_source_refs(packet_payload, result),
        "confidence": float(packet_payload.get("confidence") or content.get("confidence") or 0.5),
        "status": status,
        "formal_commit": {
            "commit_type": "stage5_evidence_commit",
            "chat_memory": False,
            "object_layer": True,
            "committed_object_id": str(committed_object_id or ""),
            "validation_decision": decision_name,
        },
        "created_at": time.time(),
        "ledger_hash": "",
    }
    id_payload = dict(entry)
    id_payload.pop("evidence_id", None)
    id_payload.pop("ledger_hash", None)
    entry["evidence_id"] = f"ev_{_hash_payload(id_payload)[:24]}"
    hash_payload = dict(entry)
    hash_payload.pop("ledger_hash", None)
    entry["ledger_hash"] = _hash_payload(hash_payload)
    return entry


def build_action_result_evidence_entry(
    *,
    run_id: str,
    action: Mapping[str, Any],
    result: Mapping[str, Any],
    episode: int = 0,
    tick: int = 0,
) -> Dict[str, Any]:
    action_name = str(action.get("function_name") or action.get("action") or action.get("kind") or "unknown_action")
    source_refs = []
    formal_ref = result.get("formal_evidence_ref", {}) if isinstance(result.get("formal_evidence_ref", {}), Mapping) else {}
    if formal_ref.get("evidence_id"):
        source_refs.append(f"formal_evidence:{formal_ref.get('evidence_id')}")
    entry = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "evidence_id": "",
        "run_id": str(run_id or ""),
        "task_family": str(result.get("environment_family") or result.get("task_family") or "core_execution"),
        "evidence_type": "stage5:action_result",
        "claim": f"Action {action_name} completed with state {result.get('state') or 'unknown'}.",
        "evidence": {
            "stage": "evidence_commit",
            "episode": int(episode or 0),
            "tick": int(tick or 0),
            "success": bool(result.get("success", False)),
        },
        "hypotheses": [],
        "action": {"name": action_name, "payload": _clip(dict(action or {}))},
        "result": _clip(dict(result or {})),
        "update": {
            "kind": "execution_observation",
            "direction": "adds_evidence" if bool(result.get("success", False)) else "records_failure",
        },
        "source_refs": _string_refs(source_refs),
        "confidence": 0.55 if bool(result.get("success", False)) else 0.65,
        "status": "recorded" if bool(result.get("success", False)) else "failed",
        "formal_commit": {
            "commit_type": "stage5_action_result",
            "chat_memory": False,
            "object_layer": True,
            "validation_decision": "not_applicable",
        },
        "created_at": time.time(),
        "ledger_hash": "",
    }
    id_payload = dict(entry)
    id_payload.pop("evidence_id", None)
    id_payload.pop("ledger_hash", None)
    entry["evidence_id"] = f"ev_{_hash_payload(id_payload)[:24]}"
    hash_payload = dict(entry)
    hash_payload.pop("ledger_hash", None)
    entry["ledger_hash"] = _hash_payload(hash_payload)
    return entry


def build_local_machine_evidence_entry(
    *,
    run_id: str,
    action_name: str,
    action_args: Mapping[str, Any],
    result: Mapping[str, Any],
    instruction: str = "",
    task_family: str = "local_machine",
    hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
    action_input: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    action = {
        "name": str(action_name or ""),
        "args": _clip(dict(action_args or {})),
    }
    if action_input:
        action["input_shape"] = {
            key: type(value).__name__
            for key, value in dict(action_input or {}).items()
        }
    entry = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "evidence_id": "",
        "run_id": str(run_id or ""),
        "task_family": str(task_family or "local_machine"),
        "evidence_type": _evidence_type_for(action_name, result),
        "claim": _claim_for(action_name, action_args, result),
        "evidence": {
            "instruction_excerpt": str(instruction or "")[:500],
            "result_state": str(result.get("state") or ""),
            "success": bool(result.get("success", False)),
            "source": "local_machine_adapter",
        },
        "hypotheses": _clip(list(hypotheses or [])),
        "action": action,
        "result": _clip(dict(result or {})),
        "update": _update_for(action_name, result),
        "source_refs": _source_refs_for(action_name, action_args, result),
        "confidence": _confidence_for(action_name, result),
        "status": "recorded" if bool(result.get("success", False)) else "failed",
        "formal_commit": {
            "commit_type": "object_layer_evidence",
            "chat_memory": False,
            "object_layer": True,
            "requires_source_refs": True,
        },
        "created_at": time.time(),
        "ledger_hash": "",
    }
    id_payload = dict(entry)
    id_payload.pop("evidence_id", None)
    id_payload.pop("ledger_hash", None)
    entry["evidence_id"] = f"ev_{_hash_payload(id_payload)[:24]}"
    hash_payload = dict(entry)
    hash_payload.pop("ledger_hash", None)
    entry["ledger_hash"] = _hash_payload(hash_payload)
    return entry


class FormalEvidenceLedger:
    """Append-only object-layer evidence ledger backed by JSONL and optional SQLite."""

    def __init__(self, jsonl_path: str | Path, *, state_store: Any = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.state_store = state_store

    def record(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(entry or {})
        payload.setdefault("schema_version", FORMAL_EVIDENCE_LEDGER_VERSION)
        payload.setdefault("created_at", time.time())
        if not payload.get("evidence_id"):
            id_payload = dict(payload)
            id_payload.pop("evidence_id", None)
            id_payload.pop("ledger_hash", None)
            payload["evidence_id"] = f"ev_{_hash_payload(id_payload)[:24]}"
        hash_payload = dict(payload)
        hash_payload.pop("ledger_hash", None)
        payload["ledger_hash"] = _hash_payload(hash_payload)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        if self.state_store is not None:
            self.state_store.record_evidence_entry(payload)
        return payload

    def recent(self, *, limit: int = 20) -> list[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        rows: deque[Dict[str, Any]] = deque(maxlen=max(1, int(limit or 20)))
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return list(rows)


def apply_evidence_entries_to_unified_context(
    context: UnifiedCognitiveContext,
    entries: Sequence[Mapping[str, Any]],
    *,
    max_queue: int = 50,
) -> UnifiedCognitiveContext:
    compact_entries = compact_evidence_entries_for_context(entries, limit=len(list(entries or [])))
    evidence_queue = list(context.evidence_queue or [])
    evidence_queue.extend(compact_entries)
    evidence_queue = evidence_queue[-max(1, int(max_queue or 50)) :]
    posterior_summary = dict(context.posterior_summary or {})
    posterior_summary["formal_evidence_ledger"] = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "recent_entry_count": len(compact_entries),
        "last_evidence_id": compact_entries[-1]["evidence_id"] if compact_entries else "",
        "object_layer_evidence": True,
    }
    return UnifiedCognitiveContext.from_parts(
        **{
            key: getattr(context, key)
            for key in context.__dataclass_fields__
            if key not in {"evidence_queue", "posterior_summary"}
        },
        evidence_queue=evidence_queue,
        posterior_summary=posterior_summary,
    )

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.context_builder import UnifiedContextBuilder, UnifiedContextInput
from core.runtime.evidence_ledger import (
    FORMAL_EVIDENCE_LEDGER_VERSION,
    FormalEvidenceLedger,
    apply_evidence_entries_to_unified_context,
    build_local_machine_evidence_entry,
)
from core.runtime.state_store import RuntimeStateStore


def test_formal_evidence_ledger_writes_jsonl_and_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = RuntimeStateStore(db_path)
    entry = build_local_machine_evidence_entry(
        run_id="run-1",
        instruction="inspect app",
        action_name="file_read",
        action_args={"path": "core/app.py", "start_line": 1, "end_line": 2},
        result={
            "success": True,
            "state": "FILE_READ",
            "function_name": "file_read",
            "path": "core/app.py",
            "location": "source",
            "start_line": 1,
            "end_line": 2,
            "content": "print('ok')\n",
        },
        hypotheses=[{"hypothesis_id": "hyp_0001", "claim": "app.py contains the entrypoint"}],
    )

    recorded = FormalEvidenceLedger(tmp_path / "events" / "formal_evidence.jsonl", state_store=store).record(entry)
    store.close()

    lines = (tmp_path / "events" / "formal_evidence.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == FORMAL_EVIDENCE_LEDGER_VERSION
    assert payload["evidence_id"] == recorded["evidence_id"]
    assert payload["source_refs"] == ["file:core/app.py:1-2"]

    reopened = RuntimeStateStore(db_path)
    rows = reopened.list_evidence_entries(run_id="run-1")
    reopened.close()
    assert rows[0]["evidence_id"] == recorded["evidence_id"]
    assert rows[0]["claim"].startswith("File core/app.py lines 1-2")
    assert rows[0]["hypotheses"][0]["hypothesis_id"] == "hyp_0001"


def test_formal_evidence_entries_write_back_to_unified_context() -> None:
    ctx = UnifiedCognitiveContext.from_parts(current_goal="inspect", current_task="read app")
    entry = build_local_machine_evidence_entry(
        run_id="run-2",
        action_name="repo_grep",
        action_args={"root": ".", "query": "mirror_exec"},
        result={
            "success": True,
            "state": "REPO_GREP_READ",
            "function_name": "repo_grep",
            "query": "mirror_exec",
            "match_count": 1,
            "matches": [{"path": "core/local_machine.py", "line": 7, "text": "mirror_exec"}],
        },
    )

    updated = apply_evidence_entries_to_unified_context(ctx, [entry])

    assert updated.evidence_queue[-1]["evidence_id"] == entry["evidence_id"]
    assert updated.evidence_queue[-1]["source_refs"] == ["file:core/local_machine.py:7"]
    assert updated.posterior_summary["formal_evidence_ledger"]["object_layer_evidence"] is True


def test_context_builder_prefers_formal_evidence_ledger_over_trace_tail() -> None:
    formal_entry = {
        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
        "evidence_id": "ev_formal_1",
        "claim": "Formal evidence should be used as object-layer context.",
        "evidence_type": "stage5:successful_tool_invocation",
        "source_refs": ["packet:p1"],
        "confidence": 0.8,
        "status": "committed",
        "ledger_hash": "hash1",
    }

    unified = UnifiedContextBuilder.build(
        UnifiedContextInput(
            unified_enabled=True,
            unified_ablation_mode="full",
            obs={},
            continuity_snapshot={},
            world_model_summary={},
            self_model_summary={},
            plan_summary={},
            current_task="inspect",
            active_hypotheses=[],
            episode_trace_tail=[
                {
                    "tick": 1,
                    "action_snapshot": {"function_name": "trace_only"},
                    "reward": 0.0,
                    "outcome": {"success": True},
                }
            ],
            retrieval_should_query=False,
            probe_pressure=0.0,
            workspace_state={
                "object_workspace": {
                    "formal_evidence_ledger": {
                        "schema_version": FORMAL_EVIDENCE_LEDGER_VERSION,
                        "last_evidence_id": "ev_formal_1",
                        "object_layer_evidence": True,
                    },
                    "formal_evidence_recent": [formal_entry],
                }
            },
            cognitive_object_records={},
        )
    )

    assert unified.evidence_queue[0]["evidence_id"] == "ev_formal_1"
    assert unified.evidence_queue[0]["claim"] == formal_entry["claim"]
    assert unified.workspace_provenance["evidence_queue_source"] == "formal_evidence_ledger"
    assert unified.posterior_summary["formal_evidence_ledger"]["last_evidence_id"] == "ev_formal_1"

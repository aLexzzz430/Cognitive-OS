from __future__ import annotations

import json
from pathlib import Path

from core.orchestration.structured_answer import StructuredAnswerSynthesizer
from integrations.local_machine.patch_proposal import generate_patch_proposals
from modules.llm.cli import main as llm_cli_main
from modules.llm.json_adaptor import (
    list_llm_output_contracts,
    normalize_llm_output,
    summarize_llm_output_adapter_traces,
)


def test_output_adapter_normalizes_prefixed_json_with_surrounding_text() -> None:
    result = normalize_llm_output(
        "Sure.\nKWARGS_JSON: {'path': 'README.md', 'start_line': 1}\nDone.",
        output_kind="action_kwargs",
        expected_prefixes=("KWARGS_JSON:",),
        expected_type="dict",
    )

    assert result.ok is True
    assert result.status == "repaired_single_quotes"
    assert result.parsed_dict() == {"path": "README.md", "start_line": 1}
    assert result.to_trace()["schema_version"] == "conos.llm.output_adapter/v1"


def test_output_adapter_normalizes_fenced_list() -> None:
    result = normalize_llm_output(
        '```json\n[{"claim": "A"}, {"claim": "B"}]\n```',
        output_kind="hypothesis_generation",
        expected_type="list",
    )

    assert result.ok is True
    assert [row["claim"] for row in result.parsed_list()] == ["A", "B"]


def test_output_contract_registry_supplies_known_patch_contract() -> None:
    result = normalize_llm_output(
        'PATCH_JSON: {"unified_diff": "--- a/app.py\\n+++ b/app.py", "risk": 0.1}',
        output_kind="patch_proposal",
    )

    trace = result.to_trace()
    assert result.ok is True
    assert result.parsed_dict()["risk"] == 0.1
    assert trace["contract_id"] == "conos.llm.output_adapter/v1:patch_proposal"
    assert trace["prefix"] == "PATCH_JSON:"
    assert trace["normalization_applied"] is True


def test_output_contract_registry_lists_core_output_kinds() -> None:
    kinds = {row["output_kind"] for row in list_llm_output_contracts()}

    assert {
        "action_kwargs",
        "reasoning_state",
        "patch_proposal",
        "status_escalation_decision",
        "skill_candidate_generation",
        "recovery_plan_synthesis",
        "representation_card_proposal",
    } <= kinds


def test_output_adapter_trace_summary_counts_repairs_and_rejections() -> None:
    repaired = normalize_llm_output("{'answer': 'ok'}", output_kind="gateway_json").to_trace()
    rejected = normalize_llm_output("not json", output_kind="gateway_json").to_trace()

    summary = summarize_llm_output_adapter_traces([repaired, rejected])

    assert summary["total"] == 2
    assert summary["ok_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["repair_applied_count"] == 1
    assert summary["by_output_kind"]["gateway_json"]["repair_applied"] == 1
    assert summary["errors"]


def test_llm_cli_lists_output_contracts(capsys) -> None:
    assert llm_cli_main(["output-contracts"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["contract_count"] >= 4
    assert any(row["output_kind"] == "patch_proposal" for row in payload["contracts"])


def test_structured_answer_records_output_adapter_trace() -> None:
    class FakeLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            return "KWARGS_JSON: {'path': 'README.md', 'start_line': 1, 'end_line': 20}"

    obs = {
        "available_functions": ["file_read"],
        "function_signatures": {
            "file_read": {
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        },
        "local_mirror": {
            "instruction": "read README",
            "source_root": "/tmp/source",
            "prefer_llm_kwargs": True,
        },
    }
    action = {
        "kind": "call_tool",
        "function_name": "file_read",
        "kwargs": {},
        "payload": {"tool_args": {"function_name": "file_read", "kwargs": {}}},
    }

    updated = StructuredAnswerSynthesizer().maybe_populate_action_kwargs(
        action,
        obs,
        llm_client=FakeLLM(),
    )
    trace = updated["_candidate_meta"]["structured_answer_llm_trace"][0]

    assert updated["payload"]["tool_args"]["kwargs"]["path"] == "README.md"
    assert trace["output_adapter"]["ok"] is True
    assert trace["output_adapter"]["output_kind"] == "action_kwargs"
    assert trace["output_adapter"]["contract_id"] == "conos.llm.output_adapter/v1:action_kwargs"


def test_patch_proposal_records_output_adapter_for_distill_and_act(tmp_path: Path) -> None:
    class FragmentLLM:
        def complete_raw(self, prompt: str, **kwargs: object) -> str:
            if "Distill the raw thinking" in prompt:
                return (
                    "Some preface.\n"
                    'REASONING_STATE_JSON: {"reasoning_state": {'
                    '"evidence": ["boundary test fails"], '
                    '"hypothesis": {"summary": "threshold is exclusive", "target_file": "app/score.py"}, '
                    '"decision": "patch", '
                    '"next_action": {"action": "propose_bounded_diff", "target_file": "app/score.py"}, '
                    '"confidence": 0.9, '
                    '"failure_boundary": ["do not modify tests"], '
                    '"patch_intent": "make threshold inclusive"}}'
                )
            if "Generate one minimal unified diff" in prompt:
                return (
                    'PATCH_JSON: {"unified_diff": "-    if value > 10:\\n+    if value >= 10:", '
                    '"rationale": "make threshold inclusive", "expected_tests": ["."], "risk": 0.2}'
                )
            return "brief notes"

    target = tmp_path / "app" / "score.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'def score_label(value):\n'
        '    if value > 10:\n'
        '        return "high"\n'
        '    return "low"\n',
        encoding="utf-8",
    )

    payload = generate_patch_proposals(
        {
            "source_root": str(tmp_path),
            "workspace_root": str(tmp_path),
            "instruction": "Fix inclusive threshold behavior",
            "investigation_state": {
                "target_binding": {"top_target_file": "app/score.py", "target_confidence": 0.8},
                "hypotheses": [
                    {
                        "hypothesis_id": "h_threshold",
                        "status": "leading",
                        "summary": "Threshold comparison is exclusive.",
                        "target_file": "app/score.py",
                    }
                ],
            },
        },
        top_target_file="app/score.py",
        llm_client=FragmentLLM(),
    )

    assert payload["patch_proposals"]
    traces = {row["stage"]: row for row in payload["llm_trace"]}
    assert traces["distill_pass"]["output_adapter"]["ok"] is True
    assert traces["distill_pass"]["output_adapter"]["output_kind"] == "reasoning_state"
    assert traces["act_pass"]["output_adapter"]["ok"] is True
    assert traces["act_pass"]["output_adapter"]["output_kind"] == "patch_proposal"
    assert payload["patch_proposals"][0]["proposal_source"] == "bounded_llm_intent_diff"

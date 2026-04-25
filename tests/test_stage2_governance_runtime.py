from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.runtime_stage_contracts import Stage2GovernanceInput
from core.orchestration.stage2_governance_runtime import run_stage2_governance
from core.orchestration import stage2_governance_runtime as runtime


class _StateSync:
    def __init__(self) -> None:
        self.calls = []

    def sync(self, input_obj) -> None:
        self.calls.append(input_obj)


class _StructuredAnswerSynthesizer:
    def __init__(self) -> None:
        self.calls = []

    def maybe_populate_action_kwargs(self, action, obs, *, llm_client=None):
        self.calls.append({"action": action, "obs": obs, "llm_client": llm_client})
        patched = dict(action)
        payload = dict(patched.get("payload", {}) or {})
        tool_args = dict(payload.get("tool_args", {}) or {})
        tool_args["kwargs"] = {"path": "README.md"}
        payload["tool_args"] = tool_args
        patched["payload"] = payload
        patched["_candidate_meta"] = {"structured_answer_synthesized": True}
        return patched


class _Loop:
    def __init__(self) -> None:
        self._governance_ports = SimpleNamespace(name="ports")
        self._organ_failure_streaks = {"planner": 1}
        self._organ_capability_flags = {"planner": "limited"}
        self._organ_failure_threshold = 3
        self._state_sync = _StateSync()
        self._structured_answer_synthesizer = None
        self.repair_calls = []

    def _json_safe(self, value):
        return {"safe": value}

    def _extract_action_function_name(self, action, default=""):
        if not isinstance(action, dict):
            return default
        if action.get("function"):
            return action.get("function")
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        return tool_args.get("function_name", default)

    def _resolve_structured_answer_llm_client(self):
        return "structured-client"

    def _repair_action_function_name(self, action, selected_name):
        self.repair_calls.append((dict(action), selected_name))
        patched = dict(action)
        patched["function"] = selected_name or patched.get("function", "")
        patched["repaired"] = True
        return patched


def _decision_outcome(*, function_name: str, action: dict, reason: str = "arbiter selected"):
    return SimpleNamespace(
        selected_candidate=SimpleNamespace(
            function_name=function_name,
            action=action,
            score=0.73,
        ),
        primary_reason=reason,
    )


def test_stage2_governance_uses_arbiter_selection_and_syncs_metadata(monkeypatch) -> None:
    loop = _Loop()
    captured = {}

    def fake_govern_action(**kwargs):
        captured.update(kwargs)
        return {
            "selected_action": {"function": "move", "target": "A"},
            "selected_name": "move",
            "meta_control_snapshot_id": "snap-1",
            "meta_control_inputs_hash": "hash-1",
        }

    monkeypatch.setattr(runtime, "govern_action", fake_govern_action)
    decision_outcome = _decision_outcome(
        function_name="move",
        action={"function": "move", "target": "A"},
    )

    out = run_stage2_governance(
        loop,
        Stage2GovernanceInput(
            action_to_use={"function": "wait"},
            candidate_actions=[{"function": "move"}],
            arm_meta={"arm": "candidate"},
            continuity_snapshot={"identity": "agent"},
            obs_before={"novel_api": {"visible_functions": ["move"]}},
            decision_outcome=decision_outcome,
            frame=SimpleNamespace(frame_id="frame-1"),
        ),
    )

    assert captured["action"] == {"function": "move", "target": "A"}
    assert captured["candidate_actions"] == [{"function": "move"}]
    assert captured["continuity_snapshot"] == {"identity": "agent"}
    assert captured["reliability_port"] is loop._governance_ports
    assert captured["counterfactual_port"] is loop._governance_ports
    assert captured["governance_log_port"] is loop._governance_ports
    assert captured["organ_capability_port"] is loop._governance_ports
    assert captured["governance_state"].organ_failure_streaks == {"planner": 1}
    assert captured["governance_state"].organ_capability_flags == {"planner": "limited"}
    assert captured["governance_state"].organ_failure_threshold == 3
    assert captured["meta_control_state"]["arm_meta"] == {"arm": "candidate"}
    assert captured["meta_control_state"]["decision_outcome"] is decision_outcome

    assert out.decision_arbiter_selected == {
        "function_name": "move",
        "action": {"safe": {"function": "move", "target": "A"}},
        "score": 0.73,
        "reason": "arbiter selected",
    }
    assert out.action_to_use == {"function": "move", "target": "A", "repaired": True}
    assert out.governance_result["selected_action"] == out.action_to_use
    assert loop._state_sync.calls[0].updates == {
        "decision_context.governance_meta_control_snapshot_id": "snap-1",
        "decision_context.governance_meta_control_inputs_hash": "hash-1",
    }
    assert loop._state_sync.calls[0].reason == "governance_meta_control_snapshot_sync"


def test_stage2_governance_preserves_inspect_probe_baseline_without_visible_functions(monkeypatch) -> None:
    loop = _Loop()
    captured = {}

    def fake_govern_action(**kwargs):
        captured.update(kwargs)
        return {
            "selected_action": dict(kwargs["action"]),
            "selected_name": "inspect",
            "meta_control_snapshot_id": "",
            "meta_control_inputs_hash": "",
        }

    monkeypatch.setattr(runtime, "govern_action", fake_govern_action)

    out = run_stage2_governance(
        loop,
        Stage2GovernanceInput(
            action_to_use={"function": "inspect", "kind": "inspect"},
            candidate_actions=[{"function": "inspect"}],
            arm_meta={},
            continuity_snapshot={},
            obs_before={"novel_api": {"visible_functions": []}},
            decision_outcome=_decision_outcome(
                function_name="probe_candidate",
                action={
                    "function": "probe_candidate",
                    "_source": "deliberation_probe",
                    "kind": "probe",
                },
                reason="probe before commit",
            ),
            frame=SimpleNamespace(),
        ),
    )

    assert captured["action"] == {"function": "inspect", "kind": "inspect"}
    assert out.action_to_use == {"function": "inspect", "kind": "inspect", "repaired": True}
    assert out.decision_arbiter_selected["function_name"] == "probe_candidate"


def test_stage2_governance_materializes_selected_action_kwargs_after_selection(monkeypatch) -> None:
    loop = _Loop()
    loop._structured_answer_synthesizer = _StructuredAnswerSynthesizer()
    selected = {
        "kind": "call_tool",
        "payload": {"tool_args": {"function_name": "file_read", "kwargs": {}}},
    }
    captured = {}

    def fake_govern_action(**kwargs):
        captured.update(kwargs)
        return {
            "selected_action": selected,
            "selected_name": "file_read",
            "meta_control_snapshot_id": "",
            "meta_control_inputs_hash": "",
        }

    monkeypatch.setattr(runtime, "govern_action", fake_govern_action)

    out = run_stage2_governance(
        loop,
        Stage2GovernanceInput(
            action_to_use={"function": "wait"},
            candidate_actions=[selected],
            arm_meta={},
            continuity_snapshot={},
            obs_before={"local_mirror": {"instruction": "read README"}},
            decision_outcome=_decision_outcome(function_name="file_read", action=selected),
            frame=SimpleNamespace(),
        ),
    )

    assert captured["candidate_actions"] == [selected]
    assert len(loop._structured_answer_synthesizer.calls) == 1
    assert loop._structured_answer_synthesizer.calls[0]["llm_client"] == "structured-client"
    assert out.action_to_use["payload"]["tool_args"]["kwargs"] == {"path": "README.md"}
    assert out.action_to_use["_candidate_meta"]["structured_answer_synthesized"] is True

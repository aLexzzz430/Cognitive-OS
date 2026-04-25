from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.planner_stage import PlannerStage, _materialize_candidate_kwargs
from core.orchestration.stage_types import PlannerStageInput, PlannerStageOutput


def _call_action(function_name: str, *, missing: list[str] | None = None) -> dict:
    return {
        "kind": "non_executable_call",
        "function_name": function_name,
        "payload": {
            "tool_name": "call_hidden_function",
            "tool_args": {"function_name": function_name, "kwargs": {}},
        },
        "_candidate_meta": {
            "kwargs_status": "insufficient_required_kwargs",
            "missing_required_kwargs": list(missing or ["patch"]),
            "non_executable_reason": "missing_required_kwargs",
            "executable": False,
        },
    }


class _Synthesizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def maybe_populate_action_kwargs(self, action, obs, *, llm_client=None):
        self.calls.append({"action": action, "obs": obs, "llm_client": llm_client})
        function_name = action["payload"]["tool_args"]["function_name"]
        kwargs = (
            {"patch": "*** Begin Patch\n*** End Patch\n", "max_files": 1}
            if function_name == "apply_patch"
            else {"kind": "finding", "content": "recorded", "evidence_refs": []}
        )
        patched = dict(action)
        payload = dict(patched.get("payload", {}) or {})
        tool_args = dict(payload.get("tool_args", {}) or {})
        tool_args["kwargs"] = kwargs
        payload["tool_args"] = tool_args
        patched["payload"] = payload
        patched["_candidate_meta"] = dict(patched.get("_candidate_meta", {}) or {})
        patched["_candidate_meta"]["structured_answer_synthesized"] = True
        return patched


class _Loop:
    def __init__(self) -> None:
        self._structured_answer_synthesizer = _Synthesizer()
        self._plan_state = SimpleNamespace(current_step=SimpleNamespace(target_function="apply_patch"))
        self._procedure_enabled = False
        self._episode_trace = []
        self.candidates_seen_by_plan_constraints: list[dict] = []

    def _resolve_structured_answer_llm_client(self):
        return "structured-client"

    def _extract_action_function_name(self, action, default=""):
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args"), dict) else {}
        return tool_args.get("function_name") or action.get("function_name") or default

    def _stage2_candidate_generation_substage(self, *_args):
        return PlannerStageOutput(
            raw_base_action={"function": "base"},
            base_action={"function": "base"},
            arm_action={"function": "arm"},
            arm_meta={},
            plan_tick_meta={},
            candidate_actions=[_call_action("note_write", missing=["content"]), _call_action("apply_patch")],
            visible_functions=["note_write", "apply_patch"],
            discovered_functions=[],
            raw_candidates_snapshot=[],
            decision_context={},
            stage_metrics={},
            deliberation_result={},
        )

    def _candidate_counts(self, candidate_actions):
        return len(candidate_actions), len(candidate_actions)

    def _snapshot_candidate_list(self, candidate_actions):
        rows = []
        for action in candidate_actions:
            tool_args = action.get("payload", {}).get("tool_args", {})
            rows.append({"fn": tool_args.get("function_name"), "kwargs": dict(tool_args.get("kwargs", {}) or {})})
        return rows

    def _stage2_plan_constraints_substage(self, _obs_before, candidate_actions):
        self.candidates_seen_by_plan_constraints = list(candidate_actions)
        return candidate_actions

    def _stage2_self_model_suppression_substage(self, candidate_actions, *_args):
        return candidate_actions

    def _annotate_candidates_with_learning_updates(self, candidate_actions, _continuity_snapshot):
        return candidate_actions

    def _annotate_candidates_with_counterfactual(self, _candidate_actions, _continuity_snapshot):
        return None

    def _counterfactual_rank_candidates(self, candidate_actions):
        return candidate_actions

    def _stage2_prediction_runtime_substage(self, _candidate_actions):
        return None

    def _decision_bridge_input_cls(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def _stage2_prediction_context_bridge_substage(self, _bridge_input):
        return {"decision_context": {"bridged": True}}


def test_materialize_candidate_kwargs_prioritizes_current_plan_target() -> None:
    loop = _Loop()

    actions = _materialize_candidate_kwargs(
        loop,
        [_call_action("note_write", missing=["content"]), _call_action("apply_patch")],
        {"local_mirror": {"instruction": "patch one file"}},
        limit=1,
    )

    assert [call["action"]["payload"]["tool_args"]["function_name"] for call in loop._structured_answer_synthesizer.calls] == ["apply_patch"]
    assert actions[0]["kind"] == "non_executable_call"
    assert actions[1]["kind"] == "call_tool"
    assert actions[1]["kwargs"]["patch"]
    assert actions[1]["_candidate_meta"]["executable"] is True
    assert actions[1]["_candidate_meta"]["missing_required_kwargs"] == []


def test_planner_stage_materializes_required_kwargs_before_plan_constraints() -> None:
    loop = _Loop()

    out = PlannerStage().run(
        loop,
        PlannerStageInput(
            obs_before={"local_mirror": {"instruction": "patch one file"}},
            surfaced=[],
            continuity_snapshot={},
            frame=SimpleNamespace(perception_summary={}, world_model_summary={}),
        ),
    )

    patch_candidates = [
        action
        for action in loop.candidates_seen_by_plan_constraints
        if action["payload"]["tool_args"]["function_name"] == "apply_patch"
    ]
    assert patch_candidates
    assert patch_candidates[0]["kind"] == "call_tool"
    assert patch_candidates[0]["payload"]["tool_args"]["kwargs"]["patch"]
    assert out.stage_metrics["after_kwargs_materialization"][1]["kwargs"]["patch"]

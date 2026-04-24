from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.runtime_stage_contracts import Stage2CandidateGenerationInput
from core.orchestration.stage2_candidate_generation_runtime import run_stage2_candidate_generation


class _PlannerRuntime:
    def __init__(self) -> None:
        self.calls = []

    def tick(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(selected_action={"function": "planner"})


class _PlanState:
    has_plan = True

    def get_plan_summary(self):
        return {"current_step": "step-1"}


class _Hypotheses:
    def __init__(self) -> None:
        self.active = [SimpleNamespace(id="hyp-1")]

    def get_active(self):
        return list(self.active)


class _SkillRewriter:
    def retrieve_skills(self, active_hypothesis, top_k: int):
        assert active_hypothesis.id == "hyp-1"
        assert top_k == 3
        return ["skill-1"]

    def rewrite(self, base_action, skills, active_hypothesis):
        assert skills == ["skill-1"]
        action = dict(base_action)
        action["skill_rewritten"] = active_hypothesis.id
        return action


class _SkillFrontend:
    def rewrite_with_llm(self, *, base_action, hypotheses, obs, episode, tick):
        assert hypotheses and hypotheses[0].id == "hyp-1"
        action = dict(base_action)
        action["llm_rewritten"] = f"{episode}:{tick}"
        return action


class _StructuredAnswerSynthesizer:
    def maybe_populate_action_kwargs(self, action, obs, *, llm_client=None):
        patched = dict(action)
        kwargs = dict(patched.get("kwargs", {}) or {})
        kwargs.setdefault("filled", True)
        patched["kwargs"] = kwargs
        patched["_llm_client"] = llm_client
        return patched


class _Retriever:
    def arm_evaluate(self, surfaced, base_action, obs):
        return (
            {
                "function": "arm",
                "kwargs": dict(base_action.get("kwargs", {}) or {}),
                "base_seen": base_action.get("function"),
            },
            {"arm": "meta"},
        )


class _CandidateGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        assert kwargs["procedure_objects"] == [{"object_id": "proc-1"}]
        assert kwargs["perception_summary"] == {"visible": 1}
        assert kwargs["world_model_summary"] == {"world": "ok"}
        return [
            {"function": "first", "kwargs": {}},
            {"function": "second", "kwargs": {}},
        ]


class _Loop:
    def __init__(self) -> None:
        self._planner_runtime = _PlannerRuntime()
        self._plan_state = _PlanState()
        self._hypotheses = _Hypotheses()
        self._skill_rewriter = _SkillRewriter()
        self._skill_frontend = _SkillFrontend()
        self._structured_answer_synthesizer = _StructuredAnswerSynthesizer()
        self._retriever = _Retriever()
        self._candidate_generator = _CandidateGenerator()
        self._capability_profile = {"can": True}
        self._reliability_tracker = SimpleNamespace(name="reliability")
        self._episode_trace = [{"reward": 1.0}]
        self._episode = 4
        self._tick = 9

    def _consume_planner_runtime_result(self, runtime_out):
        assert runtime_out.selected_action == {"function": "planner"}
        return {
            "decision_flags": {
                "events": ["planned"],
                "policy_profile": {"policy": "p"},
                "representation_profile": {"repr": "r"},
                "meta_control_snapshot_id": "snap-1",
                "meta_control_inputs_hash": "hash-1",
            }
        }

    def _generate_action(self, obs_before, continuity_snapshot):
        return {"function": "base", "kwargs": {"seed": continuity_snapshot["seed"]}}

    def _resolve_structured_answer_llm_client(self):
        return "structured-client"

    def _load_procedure_objects(self, obs_before):
        return [{"object_id": "proc-1"}]

    def _run_deliberation_engine(self, **kwargs):
        assert [row["function"] for row in kwargs["candidate_actions"]] == ["first", "second"]
        return {
            "mode": "symbolic",
            "backend": "test",
            "deliberation_trace": [{"step": 1}, {"step": 2}],
            "probe_before_commit": True,
            "_ranked_candidate_actions": [
                {"function": "second", "kwargs": {"rank": 1}},
                {"function": "first", "kwargs": {"rank": 2}},
            ],
        }

    def _snapshot_candidate_list(self, candidate_actions):
        return [{"fn": row.get("function")} for row in candidate_actions]


def test_run_stage2_candidate_generation_builds_ranked_planner_output() -> None:
    loop = _Loop()
    stage_input = Stage2CandidateGenerationInput(
        obs_before={
            "novel_api": {
                "visible_functions": ["visible_fn"],
                "discovered_functions": ["discovered_fn"],
            }
        },
        surfaced=["surface-1"],
        continuity_snapshot={"seed": "continuity"},
        frame=SimpleNamespace(
            perception_summary={"visible": 1},
            world_model_summary={"world": "ok"},
        ),
    )

    out = run_stage2_candidate_generation(loop, stage_input)

    assert out.raw_base_action == {"function": "base", "kwargs": {"seed": "continuity"}}
    assert out.base_action["skill_rewritten"] == "hyp-1"
    assert out.base_action["llm_rewritten"] == "4:9"
    assert out.base_action["kwargs"]["filled"] is True
    assert out.arm_action["function"] == "arm"
    assert out.arm_meta == {"arm": "meta"}
    assert [row["function"] for row in out.candidate_actions] == ["second", "first"]
    assert out.raw_candidates_snapshot == [{"fn": "second"}, {"fn": "first"}]
    assert out.visible_functions == ["visible_fn"]
    assert out.discovered_functions == ["discovered_fn"]
    assert out.decision_context == {}
    assert out.stage_metrics == {}
    assert out.plan_tick_meta["events"] == ["planned"]
    assert out.plan_tick_meta["has_plan"] is True
    assert out.plan_tick_meta["deliberation_mode"] == "symbolic"
    assert out.plan_tick_meta["deliberation_backend"] == "test"
    assert out.plan_tick_meta["deliberation_trace_length"] == 2
    assert out.plan_tick_meta["probe_before_commit"] is True
    assert loop._planner_runtime.calls[0]["phase"] == "control"

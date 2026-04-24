from types import SimpleNamespace

from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.deliberation_context_runtime import apply_deliberation_to_unified_context


class _StateManager:
    def __init__(self):
        self.updates = []

    def update_state(self, patch, *, reason, module):
        self.updates.append({"patch": patch, "reason": reason, "module": module})


def test_apply_deliberation_to_unified_context_updates_workspace_and_state_patch():
    unified = UnifiedCognitiveContext.from_parts(
        workspace_provenance={"source": "initial"},
    )
    frame = SimpleNamespace(unified_context=unified)
    state_mgr = _StateManager()
    deliberation_result = {
        "ranked_candidate_hypothesis_objects": [
            {
                "object_id": "hyp-obj-1",
                "object_type": "hypothesis",
                "family": "color",
                "summary": "red wins",
                "confidence": 0.9,
            },
            "ignore",
        ],
        "ranked_candidate_hypotheses": [{"hypothesis_id": "h-red"}, "ignore"],
        "ranked_candidate_tests": [{"test_id": "t1"}, None],
        "active_test_ids": [" t1 ", "", None, "t2"],
        "ranked_candidate_programs": [{"program": "inspect"}, object()],
        "ranked_candidate_outputs": [{"output": "open"}],
        "ranked_discriminating_experiments": [{"experiment": "probe-color"}],
        "posterior_summary": {"leading_hypothesis_id": "h-red"},
        "budget": {"remaining": 2},
        "mode": "contrastive",
        "deliberation_trace": [{"step": 1}, {"step": 2}],
        "backend": "symbolic",
        "control_policy": {"strategy": "verify"},
    }

    apply_deliberation_to_unified_context(
        frame=frame,
        deliberation_result=deliberation_result,
        state_mgr=state_mgr,
    )

    assert unified.active_hypotheses_summary[0]["object_id"] == "hyp-obj-1"
    assert unified.competing_hypotheses == [{"hypothesis_id": "h-red"}]
    assert unified.candidate_tests == [{"test_id": "t1"}]
    assert unified.candidate_programs == [{"program": "inspect"}]
    assert unified.candidate_outputs == [{"output": "open"}]
    assert unified.ranked_discriminating_experiments == [{"experiment": "probe-color"}]
    assert unified.posterior_summary == {"leading_hypothesis_id": "h-red"}
    assert unified.deliberation_budget == {"remaining": 2}
    assert unified.deliberation_mode == "contrastive"
    assert unified.workspace_provenance == {
        "source": "initial",
        "deliberation_trace_length": 2,
        "deliberation_backend": "symbolic",
        "deliberation_control_strategy": "verify",
    }

    assert state_mgr.updates == [
        {
            "patch": {
                "object_workspace.competing_hypothesis_objects": [
                    {
                        "object_id": "hyp-obj-1",
                        "object_type": "hypothesis",
                        "family": "color",
                        "summary": "red wins",
                        "confidence": 0.9,
                    }
                ],
                "object_workspace.active_hypotheses_summary": unified.active_hypotheses_summary,
                "object_workspace.competing_hypotheses": [{"hypothesis_id": "h-red"}],
                "object_workspace.candidate_tests": [{"test_id": "t1"}],
                "object_workspace.active_tests": ["t1", "t2"],
                "object_workspace.candidate_programs": [{"program": "inspect"}],
                "object_workspace.candidate_outputs": [{"output": "open"}],
                "object_workspace.ranked_discriminating_experiments": [{"experiment": "probe-color"}],
                "object_workspace.posterior_summary": {"leading_hypothesis_id": "h-red"},
            },
            "reason": "reasoning:deliberation_context_update",
            "module": "core.reasoning",
        }
    ]


def test_apply_deliberation_to_unified_context_ignores_invalid_inputs_and_metadata_only_patch():
    state_mgr = _StateManager()
    apply_deliberation_to_unified_context(
        frame=SimpleNamespace(unified_context={}),
        deliberation_result={"ranked_candidate_tests": [{"test_id": "ignored"}]},
        state_mgr=state_mgr,
    )
    assert state_mgr.updates == []

    unified = UnifiedCognitiveContext()
    apply_deliberation_to_unified_context(
        frame=SimpleNamespace(unified_context=unified),
        deliberation_result={
            "budget": {"remaining": 1},
            "mode": "reactive",
            "deliberation_trace": [],
            "backend": "none",
        },
        state_mgr=state_mgr,
    )

    assert unified.deliberation_budget == {"remaining": 1}
    assert unified.deliberation_mode == "reactive"
    assert unified.workspace_provenance == {
        "deliberation_trace_length": 0,
        "deliberation_backend": "none",
    }
    assert state_mgr.updates == []

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestration.audit_report import build_main_loop_audit


class _Hypotheses:
    def get_entropy_log(self):
        return [{"entropy": 0.4}]

    def get_test_log(self):
        return [{"test": "probe"}]


class _TransferTrace:
    def get_cycles(self):
        return [{"cycle": 1}]


class _RuntimeBudget:
    def to_dict(self):
        return {"budget": "ok"}


class _FamilyRegistry:
    def report(self):
        return {"families": 1}


class _MetaControl:
    policy_profile_object_id = "policy-profile"
    representation_profile_object_id = "representation-profile"
    policy_read_fallback_events = [{"fallback": False}]

    def describe_state(self):
        return {"state": "ready"}


class _UpdateEngine:
    def get_update_stats(self):
        return {"updates": 2}


class _ReliabilityTracker:
    def build_failure_preference_audit_report(self):
        return {"failures": 0}


class _PredictionError:
    def to_dict(self):
        return {"error": "none"}


class _PredictionRegistry:
    def summarize(self):
        return {"predictors": 1}

    def get_recent_errors(self):
        return [_PredictionError()]

    def get_predictor_trust(self):
        return {"predictor": 0.8}


class _HiddenStateTracker:
    def summary(self):
        return {"hidden": False}


class _ProcedureRegistry:
    def summarize(self):
        return {"procedures": 1}


class _LearnedDynamicsPredictor:
    def summary(self):
        return {"samples": 3}


class _FakeLoop:
    arm_mode = "full"
    _world_provider_meta = {"world_provider_source": "unit-test"}
    _total_reward = 1.5
    _hypotheses = _Hypotheses()
    _pre_sat_test_count = 2
    _action_divergence_from_test = 1
    _commit_quality_log = [{"quality": "high"}]
    _confirmed_functions = {"ACTION1"}
    _transfer_trace = _TransferTrace()
    _commit_log = [{"commit": "object"}]
    _recovery_log = [{"recovered": True}]
    _continuity_log = [{"continuity": True}]
    _teacher_log = [{"teacher": "posterior"}]
    _representation_log = [{"representation": "card"}]
    _governance_log = [{"verdict": "passed"}]
    _episode_trace = [{"observation": {"type": "unit"}}]
    _organ_capability_flags = {"planner": True}
    _organ_failure_streaks = {"planner": 0}
    _organ_control_audit_log = [{"organ": "planner"}]
    _candidate_viability_log = [{"candidate": "ACTION1"}]
    _planner_runtime_log = [{"stage": "planner"}]
    _llm_advice_log = [{"advice": "none"}]
    _llm_calls_per_tick = [{"tick": 0, "calls": 1}]
    _llm_route_usage_log = [{"route": "shadow"}]
    _llm_mode = "shadow"
    _llm_shadow_log = [{"verdict": "ok"}]
    _llm_analyst_log = [{"verdict": "ok"}]
    _llm_world_model_snapshot = {"world": "snapshot"}
    _llm_world_model_proposal_candidates = [{"proposal": "candidate"}]
    _llm_world_model_validation_feedback = [{"feedback": "accepted"}]
    _learned_dynamics_shadow_predictor = _LearnedDynamicsPredictor()
    _learned_dynamics_deployment_mode = "shadow"
    _learned_dynamics_shadow_log = [{"shadow": "event"}]
    _runtime_budget = _RuntimeBudget()
    _family_registry = _FamilyRegistry()
    _learning_update_log = [{"learning": "update"}]
    _learning_policy_snapshot = {"policy": "snapshot"}
    _meta_control = _MetaControl()
    _update_engine = _UpdateEngine()
    _mechanism_runtime_state = {"mechanism": "state"}
    _last_mechanism_runtime_view = {"mechanism": "view"}
    _last_task_frame_summary = {"task": "frame"}
    _last_mechanism_prior_usage = {"prior": "used"}
    _mechanism_control_audit_log = [{"mechanism": "audit"}]
    _reliability_tracker = _ReliabilityTracker()
    _prediction_enabled = True
    _prediction_registry = _PredictionRegistry()
    _prediction_trace_log = [{"prediction": "trace"}]
    _hidden_state_tracker = _HiddenStateTracker()
    _procedure_enabled = True
    _procedure_registry = _ProcedureRegistry()
    _procedure_promotion_log = [{"id": index} for index in range(12)]
    _procedure_proposal_log = [{"id": index} for index in range(11)]
    _procedure_execution_log = [{"id": index} for index in range(10)]

    def _json_safe(self, value):
        return value

    def _llm_route_usage_summary(self):
        return {"shadow": 1}

    def _ablation_flags_snapshot(self):
        return {"ablation": False}


def test_build_main_loop_audit_preserves_public_snapshot_shape() -> None:
    audit = build_main_loop_audit(_FakeLoop())

    assert audit["world_provider_source"] == "unit-test"
    assert audit["total_reward"] == 1.5
    assert audit["entropy_log"] == [{"entropy": 0.4}]
    assert audit["confirmed_functions"] == ["ACTION1"]
    assert audit["llm_route_usage_summary"] == {"shadow": 1}
    assert audit["learned_dynamics_shadow_enabled"] is True
    assert audit["runtime_budget"] == {"budget": "ok"}
    assert audit["prediction_recent_errors"] == [{"error": "none"}]
    assert audit["procedure_recent_promotions"] == [{"id": index} for index in range(2, 12)]
    assert audit["procedure_recent_proposals"] == [{"id": index} for index in range(1, 11)]
    assert audit["procedure_recent_executions"] == [{"id": index} for index in range(10)]

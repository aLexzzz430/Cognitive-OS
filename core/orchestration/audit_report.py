from __future__ import annotations

from typing import Any, Dict

from core.orchestration.arc3_action_coverage import summarize_arc3_action_coverage
from core.orchestration.llm_shadow_runtime import (
    build_llm_analyst_summary,
    build_llm_shadow_summary,
)


def build_main_loop_audit(loop: Any) -> Dict[str, Any]:
    """Build the public audit snapshot for CoreMainLoop without owning execution."""
    return {
        'arm_mode': loop.arm_mode,
        'world_provider_source': loop._world_provider_meta.get('world_provider_source', 'unknown'),
        'total_reward': loop._total_reward,
        'entropy_log': loop._hypotheses.get_entropy_log(),
        'test_log': loop._hypotheses.get_test_log(),
        'pre_sat_test_count': loop._pre_sat_test_count,
        'action_divergence_count': getattr(loop, '_action_divergence_from_test', 0),
        'step10_quality_log': loop._commit_quality_log,
        'confirmed_functions': list(loop._confirmed_functions),
        'transfer_trace': loop._transfer_trace.get_cycles(),
        'commit_log': loop._commit_log,
        'recovery_log': list(loop._recovery_log),
        'continuity_log': list(loop._continuity_log),
        'teacher_log': list(loop._teacher_log),
        'representation_log': list(loop._representation_log),
        'governance_log': list(loop._governance_log),
        'episode_trace': loop._json_safe(list(loop._episode_trace)),
        'organ_capability_flags': dict(loop._organ_capability_flags),
        'organ_failure_streaks': dict(loop._organ_failure_streaks),
        'organ_control_audit_log': list(loop._organ_control_audit_log),
        'candidate_viability_log': list(loop._candidate_viability_log),
        'planner_runtime_log': list(loop._planner_runtime_log),
        'llm_advice_log': list(loop._llm_advice_log),
        'llm_calls_per_tick': list(loop._llm_calls_per_tick),
        'llm_route_usage_log': list(getattr(loop, '_llm_route_usage_log', [])),
        'llm_route_usage_summary': loop._llm_route_usage_summary(),
        'llm_mode': loop._llm_mode,
        'llm_shadow_log': list(getattr(loop, '_llm_shadow_log', [])),
        'llm_shadow_summary': build_llm_shadow_summary(list(getattr(loop, '_llm_shadow_log', []))),
        'llm_analyst_log': list(getattr(loop, '_llm_analyst_log', [])),
        'llm_analyst_summary': build_llm_analyst_summary(list(getattr(loop, '_llm_analyst_log', []))),
        'llm_world_model_snapshot': loop._json_safe(getattr(loop, '_llm_world_model_snapshot', {})),
        'llm_world_model_proposal_candidates': loop._json_safe(list(getattr(loop, '_llm_world_model_proposal_candidates', []))),
        'llm_world_model_validation_feedback': loop._json_safe(list(getattr(loop, '_llm_world_model_validation_feedback', []))),
        'learned_dynamics_shadow_enabled': bool(getattr(loop, '_learned_dynamics_shadow_predictor', None) is not None),
        'learned_dynamics_deployment_mode': str(getattr(loop, '_learned_dynamics_deployment_mode', 'shadow') or 'shadow'),
        'learned_dynamics_shadow_model_summary': (
            loop._json_safe(getattr(loop._learned_dynamics_shadow_predictor, 'summary', lambda: {})())
            if getattr(loop, '_learned_dynamics_shadow_predictor', None) is not None
            else {}
        ),
        'learned_dynamics_shadow_log': loop._json_safe(list(getattr(loop, '_learned_dynamics_shadow_log', []))),
        'runtime_budget': (
            loop._runtime_budget.to_dict()
            if hasattr(loop._runtime_budget, 'to_dict')
            else vars(loop._runtime_budget)
        ),
        'ablation_flags': loop._ablation_flags_snapshot(),
        'family_registry': loop._family_registry.report(),
        'learning_update_log': list(loop._learning_update_log),
        'learning_policy_snapshot': dict(loop._learning_policy_snapshot) if isinstance(loop._learning_policy_snapshot, dict) else {},
        'policy_profile_object_id': loop._meta_control.policy_profile_object_id,
        'representation_profile_object_id': getattr(loop._meta_control, 'representation_profile_object_id', ''),
        'meta_control_state': loop._meta_control.describe_state() if hasattr(loop._meta_control, 'describe_state') else {},
        'policy_read_fallback_events': list(loop._meta_control.policy_read_fallback_events),
        'learning_update_stats': loop._update_engine.get_update_stats(),
        'mechanism_runtime_state': loop._json_safe(loop._mechanism_runtime_state),
        'mechanism_runtime_view': loop._json_safe(loop._last_mechanism_runtime_view),
        'last_task_frame_summary': loop._json_safe(getattr(loop, '_last_task_frame_summary', {})),
        'mechanism_prior_usage': loop._json_safe(getattr(loop, '_last_mechanism_prior_usage', {})),
        'mechanism_control_audit_log': list(loop._mechanism_control_audit_log),
        'failure_preference_audit': loop._reliability_tracker.build_failure_preference_audit_report() if hasattr(loop._reliability_tracker, 'build_failure_preference_audit_report') else {},
        'prediction_enabled': bool(loop._prediction_enabled),
        'prediction_registry_summary': loop._prediction_registry.summarize() if loop._prediction_enabled else {},
        'prediction_recent_errors': [e.to_dict() for e in loop._prediction_registry.get_recent_errors()] if loop._prediction_enabled else [],
        'predictor_trust': loop._prediction_registry.get_predictor_trust() if loop._prediction_enabled else {},
        'prediction_trace_log': list(loop._prediction_trace_log),
        'arc3_action_coverage': summarize_arc3_action_coverage(
            list(loop._episode_trace),
            list(loop._candidate_viability_log),
            list(loop._governance_log),
        ),
        'hidden_state_summary': loop._hidden_state_tracker.summary() if loop._hidden_state_tracker is not None else {},
        'procedure_enabled': bool(loop._procedure_enabled),
        'procedure_registry_summary': loop._procedure_registry.summarize() if loop._procedure_enabled else {},
        'procedure_recent_promotions': list(loop._procedure_promotion_log[-10:]),
        'procedure_recent_proposals': list(loop._procedure_proposal_log[-10:]),
        'procedure_recent_executions': list(loop._procedure_execution_log[-10:]),
    }

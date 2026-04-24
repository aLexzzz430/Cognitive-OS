from __future__ import annotations

from typing import Any, Dict

from core.cognition.unified_context import UnifiedCognitiveContext
from core.orchestration.state_abstraction import summarize_cognitive_object_records


def apply_deliberation_to_unified_context(
    *,
    frame: Any,
    deliberation_result: Dict[str, Any],
    state_mgr: Any = None,
) -> None:
    unified = getattr(frame, 'unified_context', None)
    if not isinstance(unified, UnifiedCognitiveContext) or not isinstance(deliberation_result, dict):
        return

    workspace_patch: Dict[str, Any] = {}
    if isinstance(deliberation_result.get('ranked_candidate_hypothesis_objects'), list):
        competing_hypothesis_objects = [
            dict(item)
            for item in deliberation_result.get('ranked_candidate_hypothesis_objects', [])
            if isinstance(item, dict)
        ]
        if competing_hypothesis_objects:
            active_hypotheses_summary = summarize_cognitive_object_records(
                competing_hypothesis_objects,
                limit=8,
            )
            unified.active_hypotheses_summary = active_hypotheses_summary
            workspace_patch['object_workspace.competing_hypothesis_objects'] = competing_hypothesis_objects
            workspace_patch['object_workspace.active_hypotheses_summary'] = active_hypotheses_summary
    if isinstance(deliberation_result.get('ranked_candidate_hypotheses'), list):
        competing_hypotheses = [
            dict(item)
            for item in deliberation_result.get('ranked_candidate_hypotheses', [])
            if isinstance(item, dict)
        ]
        unified.competing_hypotheses = competing_hypotheses
        workspace_patch['object_workspace.competing_hypotheses'] = competing_hypotheses
    if isinstance(deliberation_result.get('ranked_candidate_tests'), list):
        candidate_tests = [dict(item) for item in deliberation_result.get('ranked_candidate_tests', []) if isinstance(item, dict)]
        unified.candidate_tests = candidate_tests
        workspace_patch['object_workspace.candidate_tests'] = candidate_tests
    if isinstance(deliberation_result.get('active_test_ids'), list):
        workspace_patch['object_workspace.active_tests'] = [
            str(item or '').strip()
            for item in deliberation_result.get('active_test_ids', [])
            if str(item or '').strip()
        ]
    if isinstance(deliberation_result.get('ranked_candidate_programs'), list):
        candidate_programs = [dict(item) for item in deliberation_result.get('ranked_candidate_programs', []) if isinstance(item, dict)]
        unified.candidate_programs = candidate_programs
        workspace_patch['object_workspace.candidate_programs'] = candidate_programs
    if isinstance(deliberation_result.get('ranked_candidate_outputs'), list):
        candidate_outputs = [dict(item) for item in deliberation_result.get('ranked_candidate_outputs', []) if isinstance(item, dict)]
        unified.candidate_outputs = candidate_outputs
        workspace_patch['object_workspace.candidate_outputs'] = candidate_outputs
    if isinstance(deliberation_result.get('ranked_discriminating_experiments'), list):
        experiments = [dict(item) for item in deliberation_result.get('ranked_discriminating_experiments', []) if isinstance(item, dict)]
        unified.ranked_discriminating_experiments = experiments
        workspace_patch['object_workspace.ranked_discriminating_experiments'] = experiments
    if isinstance(deliberation_result.get('posterior_summary'), dict):
        posterior_summary = dict(deliberation_result.get('posterior_summary', {}) or {})
        unified.posterior_summary = posterior_summary
        workspace_patch['object_workspace.posterior_summary'] = posterior_summary
    if isinstance(deliberation_result.get('budget'), dict):
        unified.deliberation_budget = dict(deliberation_result.get('budget', {}))
    if deliberation_result.get('mode'):
        unified.deliberation_mode = str(deliberation_result.get('mode') or unified.deliberation_mode or 'reactive')
    if isinstance(deliberation_result.get('deliberation_trace'), list):
        workspace_provenance = dict(unified.workspace_provenance or {})
        workspace_provenance['deliberation_trace_length'] = len(deliberation_result.get('deliberation_trace', []))
        workspace_provenance['deliberation_backend'] = str(deliberation_result.get('backend', '') or '')
        if isinstance(deliberation_result.get('control_policy'), dict):
            workspace_provenance['deliberation_control_strategy'] = str(
                deliberation_result.get('control_policy', {}).get('strategy', '') or ''
            )
        unified.workspace_provenance = workspace_provenance
    if workspace_patch and state_mgr is not None:
        state_mgr.update_state(
            workspace_patch,
            reason='reasoning:deliberation_context_update',
            module='core.reasoning',
        )

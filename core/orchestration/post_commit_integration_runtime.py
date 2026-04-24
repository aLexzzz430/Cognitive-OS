from __future__ import annotations

from typing import Any, Dict

from core.orchestration.post_commit_integration import integrate_committed_objects
from core.orchestration.runtime_stage_contracts import PostCommitIntegrationInput


def run_post_commit_integration(loop: Any, stage_input: PostCommitIntegrationInput) -> Dict[str, Any]:
    committed_ids = stage_input.committed_ids
    obs_before = stage_input.obs_before
    result = stage_input.result
    if not committed_ids:
        return {'integration_summary': {'committed_count': 0}}

    integration_summary = integrate_committed_objects(
        committed_ids=committed_ids,
        processed_committed_ids=loop._processed_committed_ids,
        shared_store=loop._shared_store,
        runtime_store=loop._runtime_store,
        family_registry=loop._family_registry,
        confirmed_functions=loop._confirmed_functions,
        commit_log=loop._commit_log,
        teacher=loop._teacher,
        teacher_log=loop._teacher_log,
        teacher_allows_intervention=loop._teacher_allows_intervention,
        tick=loop._tick,
        episode=loop._episode,
        obs_before=obs_before,
        result=result,
        reward=loop._get_reward(result),
    )
    record_post_commit_continuity(loop, integration_summary)
    loop._maybe_commit_procedure_chain(committed_ids=committed_ids, obs_before=obs_before, result=result)
    write_object_workspace_state(loop, integration_summary)
    integration_summary.setdefault('committed_count', len(committed_ids))
    return {'integration_summary': integration_summary}


def record_post_commit_continuity(loop: Any, integration_summary: Dict[str, Any]) -> None:
    if not hasattr(loop, '_continuity'):
        return

    autobiographical_summary = integration_summary.get('autobiographical_summary', {})
    if isinstance(autobiographical_summary, dict) and autobiographical_summary:
        loop._continuity.record_autobiographical_summary(autobiographical_summary)
    loop._continuity.record_memory_summary(
        semantic_memory={
            'surfaced_object_ids': list(integration_summary.get('surfaced_object_ids', []) or []),
        },
        procedural_memory={
            'planner_prior_object_ids': list(integration_summary.get('planner_prior_object_ids', []) or []),
        },
        transfer_memory={
            'cross_domain_prior_object_ids': list(integration_summary.get('cross_domain_prior_object_ids', []) or []),
        },
    )


def write_object_workspace_state(loop: Any, integration_summary: Dict[str, Any]) -> None:
    if not isinstance(integration_summary, dict) or not hasattr(loop, '_state_mgr'):
        return

    patch = {
        'object_workspace.surfaced_object_ids': list(integration_summary.get('surfaced_object_ids', []) or []),
        'object_workspace.mechanism_object_ids': list(integration_summary.get('mechanism_object_ids', []) or []),
        'object_workspace.object_competitions': list(integration_summary.get('object_competitions', []) or []),
        'object_workspace.active_tests': list(integration_summary.get('active_tests', []) or []),
        'object_workspace.current_identity_snapshot': dict(integration_summary.get('current_identity_snapshot', {}) or {}),
        'object_workspace.autobiographical_summary': dict(integration_summary.get('autobiographical_summary', {}) or {}),
    }
    if 'candidate_tests' in integration_summary:
        patch['object_workspace.candidate_tests'] = list(integration_summary.get('candidate_tests', []) or [])
    if 'candidate_programs' in integration_summary:
        patch['object_workspace.candidate_programs'] = list(integration_summary.get('candidate_programs', []) or [])
    if 'candidate_outputs' in integration_summary:
        patch['object_workspace.candidate_outputs'] = list(integration_summary.get('candidate_outputs', []) or [])
    loop._state_mgr.update_state(
        patch,
        reason='workflow:post_commit_object_workspace',
        module='core',
    )

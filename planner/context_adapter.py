from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.cognition.unified_context import UnifiedCognitiveContext


@dataclass
class PlannerContextAdapter:
    """Planner adapter to read/write plan payload via canonical unified context contract."""

    warnings: List[str] = field(default_factory=list)

    def read(self, context: Dict[str, Any]) -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        payload = dict(unified.plan_state_summary)
        payload['goal_agenda'] = [dict(item) for item in unified.goal_agenda]
        payload['long_horizon_commitments'] = [dict(item) for item in unified.long_horizon_commitments]
        payload['active_skills'] = [dict(item) for item in unified.active_skills]
        payload['transfer_candidates'] = [dict(item) for item in unified.transfer_candidates]
        payload['ranked_discriminating_experiments'] = [dict(item) for item in unified.ranked_discriminating_experiments]
        payload['candidate_tests'] = [dict(item) for item in unified.candidate_tests]
        payload['candidate_programs'] = [dict(item) for item in unified.candidate_programs]
        payload['candidate_outputs'] = [dict(item) for item in unified.candidate_outputs]
        payload['deliberation_budget'] = dict(unified.deliberation_budget)
        payload['deliberation_mode'] = str(unified.deliberation_mode or 'reactive')
        payload['compute_budget'] = dict(unified.compute_budget)
        payload['safety_budget'] = dict(unified.safety_budget)
        return payload

    def write(self, context: Dict[str, Any], plan_state_summary: Dict[str, Any], *, current_task: str = '') -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        summary = dict(plan_state_summary or {})
        if isinstance(summary.get('goal_agenda'), list):
            unified.goal_agenda = [dict(item) for item in summary.get('goal_agenda', []) if isinstance(item, dict)]
        if isinstance(summary.get('long_horizon_commitments'), list):
            unified.long_horizon_commitments = [
                dict(item) for item in summary.get('long_horizon_commitments', []) if isinstance(item, dict)
            ]
        if isinstance(summary.get('active_skills'), list):
            unified.active_skills = [dict(item) for item in summary.get('active_skills', []) if isinstance(item, dict)]
        if isinstance(summary.get('transfer_candidates'), list):
            unified.transfer_candidates = [
                dict(item) for item in summary.get('transfer_candidates', []) if isinstance(item, dict)
            ]
        if isinstance(summary.get('ranked_discriminating_experiments'), list):
            unified.ranked_discriminating_experiments = [
                dict(item) for item in summary.get('ranked_discriminating_experiments', []) if isinstance(item, dict)
            ]
        if isinstance(summary.get('candidate_tests'), list):
            unified.candidate_tests = [dict(item) for item in summary.get('candidate_tests', []) if isinstance(item, dict)]
        if isinstance(summary.get('candidate_programs'), list):
            unified.candidate_programs = [dict(item) for item in summary.get('candidate_programs', []) if isinstance(item, dict)]
        if isinstance(summary.get('candidate_outputs'), list):
            unified.candidate_outputs = [dict(item) for item in summary.get('candidate_outputs', []) if isinstance(item, dict)]
        if isinstance(summary.get('deliberation_budget'), dict):
            unified.deliberation_budget = dict(summary.get('deliberation_budget', {}))
        if isinstance(summary.get('compute_budget'), dict):
            unified.compute_budget = dict(summary.get('compute_budget', {}))
        if isinstance(summary.get('safety_budget'), dict):
            unified.safety_budget = dict(summary.get('safety_budget', {}))
        if summary.get('deliberation_mode'):
            unified.deliberation_mode = str(summary.get('deliberation_mode') or unified.deliberation_mode or 'reactive')
        for key in (
            'goal_agenda',
            'long_horizon_commitments',
            'active_skills',
            'transfer_candidates',
            'ranked_discriminating_experiments',
            'candidate_tests',
            'candidate_programs',
            'candidate_outputs',
            'deliberation_budget',
            'deliberation_mode',
            'compute_budget',
            'safety_budget',
        ):
            summary.pop(key, None)
        unified.plan_state_summary = summary
        if current_task:
            unified.current_task = str(current_task)
        out = dict(context)
        out['unified_cognitive_context'] = unified.to_dict()
        if self.warnings:
            out.setdefault('contract_warnings', [])
            out['contract_warnings'] = list(out['contract_warnings']) + list(self.warnings)
        return out

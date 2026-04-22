from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.cognition.unified_context import UnifiedCognitiveContext


@dataclass
class WorldModelContextAdapter:
    """World model IO adapter that only reads/writes canonical unified context."""

    warnings: List[str] = field(default_factory=list)

    def read(self, context: Dict[str, Any]) -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        payload = dict(unified.active_beliefs_summary)
        payload['surfaced_representations'] = [dict(item) for item in unified.surfaced_representations]
        payload['competing_hypotheses'] = [dict(item) for item in unified.competing_hypotheses]
        payload['candidate_tests'] = [dict(item) for item in unified.candidate_tests]
        payload['evidence_queue'] = [dict(item) for item in unified.evidence_queue]
        payload['workspace_provenance'] = dict(unified.workspace_provenance)
        return payload

    def write(self, context: Dict[str, Any], world_model_summary: Dict[str, Any]) -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        summary = dict(world_model_summary or {})
        if isinstance(summary.get('surfaced_representations'), list):
            unified.surfaced_representations = [
                dict(item) for item in summary.get('surfaced_representations', []) if isinstance(item, dict)
            ]
        if isinstance(summary.get('competing_hypotheses'), list):
            unified.competing_hypotheses = [
                dict(item) for item in summary.get('competing_hypotheses', []) if isinstance(item, dict)
            ]
        if isinstance(summary.get('candidate_tests'), list):
            unified.candidate_tests = [dict(item) for item in summary.get('candidate_tests', []) if isinstance(item, dict)]
        if isinstance(summary.get('evidence_queue'), list):
            unified.evidence_queue = [dict(item) for item in summary.get('evidence_queue', []) if isinstance(item, dict)]
        if isinstance(summary.get('workspace_provenance'), dict):
            unified.workspace_provenance = dict(summary.get('workspace_provenance', {}))
        for key in (
            'surfaced_representations',
            'competing_hypotheses',
            'candidate_tests',
            'evidence_queue',
            'workspace_provenance',
        ):
            summary.pop(key, None)
        unified.active_beliefs_summary = summary
        out = dict(context)
        out['unified_cognitive_context'] = unified.to_dict()
        if self.warnings:
            out.setdefault('contract_warnings', [])
            out['contract_warnings'] = list(out['contract_warnings']) + list(self.warnings)
        return out

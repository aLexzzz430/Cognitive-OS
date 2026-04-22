from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.cognition.unified_context import UnifiedCognitiveContext


@dataclass
class SelfModelContextAdapter:
    """Self-model adapter to centralize reads/writes through canonical unified context."""

    warnings: List[str] = field(default_factory=list)

    def read(self, context: Dict[str, Any]) -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        payload = dict(unified.self_model_summary)
        payload['identity_state'] = dict(unified.identity_state)
        payload['autobiographical_state'] = dict(unified.autobiographical_state)
        payload['uncertainty_vector'] = dict(unified.uncertainty_vector)
        payload['compute_budget'] = dict(unified.compute_budget)
        payload['safety_budget'] = dict(unified.safety_budget)
        return payload

    def write(self, context: Dict[str, Any], self_model_summary: Dict[str, Any], *, resource_pressure: str = 'normal') -> Dict[str, Any]:
        unified, warnings = UnifiedCognitiveContext.from_payload(context.get('unified_cognitive_context'))
        self.warnings.extend(warnings)
        summary = dict(self_model_summary or {})
        if isinstance(summary.get('identity_state'), dict):
            unified.identity_state = dict(summary.get('identity_state', {}))
        if isinstance(summary.get('autobiographical_state'), dict):
            unified.autobiographical_state = dict(summary.get('autobiographical_state', {}))
        if isinstance(summary.get('uncertainty_vector'), dict):
            unified.uncertainty_vector = dict(summary.get('uncertainty_vector', {}))
        if isinstance(summary.get('compute_budget'), dict):
            unified.compute_budget = dict(summary.get('compute_budget', {}))
        if isinstance(summary.get('safety_budget'), dict):
            unified.safety_budget = dict(summary.get('safety_budget', {}))
        for key in (
            'identity_state',
            'autobiographical_state',
            'uncertainty_vector',
            'compute_budget',
            'safety_budget',
        ):
            summary.pop(key, None)
        unified.self_model_summary = summary
        unified.resource_pressure = str(resource_pressure or unified.resource_pressure or 'normal')
        out = dict(context)
        out['unified_cognitive_context'] = unified.to_dict()
        if self.warnings:
            out.setdefault('contract_warnings', [])
            out['contract_warnings'] = list(out['contract_warnings']) + list(self.warnings)
        return out

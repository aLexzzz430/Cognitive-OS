"""
self_model/state.py

High-level self model state representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SelfModelState:
    """High-level self model state used by planner/decision/recovery."""

    capabilities_by_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    capabilities_by_condition: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    known_failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    fragile_regions: List[Dict[str, Any]] = field(default_factory=list)
    recovered_regions: List[Dict[str, Any]] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    identity_markers: Dict[str, Any] = field(default_factory=dict)
    durable_identity: Dict[str, Any] = field(default_factory=dict)
    active_commitments: List[Dict[str, Any]] = field(default_factory=list)
    long_horizon_agenda: List[Dict[str, Any]] = field(default_factory=list)
    known_blind_spots: List[str] = field(default_factory=list)
    capability_envelope: Dict[str, Any] = field(default_factory=dict)
    autobiographical_summary: Dict[str, Any] = field(default_factory=dict)
    self_experiment_queue: List[Dict[str, Any]] = field(default_factory=list)
    teacher_dependence_estimate: float = 0.5
    transfer_readiness: float = 0.0
    continuity_confidence: float = 0.5
    value_commitments_summary: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'capabilities_by_domain': self.capabilities_by_domain,
            'capabilities_by_condition': self.capabilities_by_condition,
            'known_failure_modes': self.known_failure_modes,
            'fragile_regions': self.fragile_regions,
            'recovered_regions': self.recovered_regions,
            'external_dependencies': self.external_dependencies,
            'identity_markers': self.identity_markers,
            'durable_identity': dict(self.durable_identity or {}),
            'active_commitments': list(self.active_commitments),
            'long_horizon_agenda': list(self.long_horizon_agenda),
            'known_blind_spots': list(self.known_blind_spots),
            'capability_envelope': dict(self.capability_envelope or {}),
            'autobiographical_summary': dict(self.autobiographical_summary or {}),
            'self_experiment_queue': list(self.self_experiment_queue),
            'teacher_dependence_estimate': max(0.0, min(1.0, float(self.teacher_dependence_estimate))),
            'transfer_readiness': max(0.0, min(1.0, float(self.transfer_readiness))),
            'continuity_confidence': max(0.0, min(1.0, float(self.continuity_confidence))),
            'value_commitments_summary': self.value_commitments_summary,
            'provenance': dict(self.provenance or {}),
        }


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_CONTEXT_VERSION = 'unified_context.v4'
CANONICAL_CONTEXT_FIELDS: Tuple[str, ...] = (
    'schema_version',
    'current_goal',
    'current_task',
    'active_beliefs_summary',
    'active_hypotheses_summary',
    'plan_state_summary',
    'self_model_summary',
    'recent_failure_profile',
    'recent_progress_markers',
    'retrieval_pressure',
    'retrieval_triggered',
    'probe_pressure',
    'resource_pressure',
    'world_shift_risk',
    'task_frame_summary',
    'object_bindings_summary',
    'goal_hypotheses_summary',
    'solver_state_summary',
    'mechanism_hypotheses_summary',
    'mechanism_control_summary',
    'surfaced_representations',
    'competing_hypotheses',
    'ranked_discriminating_experiments',
    'candidate_tests',
    'active_skills',
    'transfer_candidates',
    'identity_state',
    'autobiographical_state',
    'candidate_programs',
    'candidate_outputs',
    'posterior_summary',
    'deliberation_budget',
    'deliberation_mode',
    'uncertainty_vector',
    'evidence_queue',
    'workspace_provenance',
    'safety_budget',
    'compute_budget',
    'goal_agenda',
    'long_horizon_commitments',
)


def _dict_or_empty(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _dict_list_or_empty(value: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass
class UnifiedCognitiveContext:
    """Single JSON-safe cognitive context snapshot shared by planner/decision paths."""

    schema_version: str = CANONICAL_CONTEXT_VERSION
    current_goal: str = ''
    current_task: str = ''

    active_beliefs_summary: Dict[str, Any] = field(default_factory=dict)
    active_hypotheses_summary: List[Dict[str, Any]] = field(default_factory=list)

    plan_state_summary: Dict[str, Any] = field(default_factory=dict)
    self_model_summary: Dict[str, Any] = field(default_factory=dict)

    recent_failure_profile: List[Dict[str, Any]] = field(default_factory=list)
    recent_progress_markers: List[Dict[str, Any]] = field(default_factory=list)

    retrieval_pressure: float = 0.0
    retrieval_triggered: bool = False
    probe_pressure: float = 0.0
    resource_pressure: str = 'normal'
    world_shift_risk: float = 0.0

    task_frame_summary: Dict[str, Any] = field(default_factory=dict)
    object_bindings_summary: Dict[str, Any] = field(default_factory=dict)
    goal_hypotheses_summary: List[Dict[str, Any]] = field(default_factory=list)
    solver_state_summary: Dict[str, Any] = field(default_factory=dict)
    mechanism_hypotheses_summary: List[Dict[str, Any]] = field(default_factory=list)
    mechanism_control_summary: Dict[str, Any] = field(default_factory=dict)

    surfaced_representations: List[Dict[str, Any]] = field(default_factory=list)
    competing_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    ranked_discriminating_experiments: List[Dict[str, Any]] = field(default_factory=list)
    candidate_tests: List[Dict[str, Any]] = field(default_factory=list)
    active_skills: List[Dict[str, Any]] = field(default_factory=list)
    transfer_candidates: List[Dict[str, Any]] = field(default_factory=list)
    identity_state: Dict[str, Any] = field(default_factory=dict)
    autobiographical_state: Dict[str, Any] = field(default_factory=dict)
    candidate_programs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_outputs: List[Dict[str, Any]] = field(default_factory=list)
    posterior_summary: Dict[str, Any] = field(default_factory=dict)
    deliberation_budget: Dict[str, Any] = field(default_factory=dict)
    deliberation_mode: str = 'reactive'
    uncertainty_vector: Dict[str, Any] = field(default_factory=dict)
    evidence_queue: List[Dict[str, Any]] = field(default_factory=list)
    workspace_provenance: Dict[str, Any] = field(default_factory=dict)
    safety_budget: Dict[str, Any] = field(default_factory=dict)
    compute_budget: Dict[str, Any] = field(default_factory=dict)
    goal_agenda: List[Dict[str, Any]] = field(default_factory=list)
    long_horizon_commitments: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_parts(
        cls,
        *,
        schema_version: Optional[str] = None,
        current_goal: Optional[str] = None,
        current_task: Optional[str] = None,
        active_beliefs_summary: Optional[Dict[str, Any]] = None,
        active_hypotheses_summary: Optional[List[Dict[str, Any]]] = None,
        plan_state_summary: Optional[Dict[str, Any]] = None,
        self_model_summary: Optional[Dict[str, Any]] = None,
        recent_failure_profile: Optional[List[Dict[str, Any]]] = None,
        recent_progress_markers: Optional[List[Dict[str, Any]]] = None,
        retrieval_pressure: Optional[float] = None,
        retrieval_triggered: Optional[bool] = None,
        probe_pressure: Optional[float] = None,
        resource_pressure: Optional[str] = None,
        world_shift_risk: Optional[float] = None,
        task_frame_summary: Optional[Dict[str, Any]] = None,
        object_bindings_summary: Optional[Dict[str, Any]] = None,
        goal_hypotheses_summary: Optional[List[Dict[str, Any]]] = None,
        solver_state_summary: Optional[Dict[str, Any]] = None,
        mechanism_hypotheses_summary: Optional[List[Dict[str, Any]]] = None,
        mechanism_control_summary: Optional[Dict[str, Any]] = None,
        surfaced_representations: Optional[List[Dict[str, Any]]] = None,
        competing_hypotheses: Optional[List[Dict[str, Any]]] = None,
        ranked_discriminating_experiments: Optional[List[Dict[str, Any]]] = None,
        candidate_tests: Optional[List[Dict[str, Any]]] = None,
        active_skills: Optional[List[Dict[str, Any]]] = None,
        transfer_candidates: Optional[List[Dict[str, Any]]] = None,
        identity_state: Optional[Dict[str, Any]] = None,
        autobiographical_state: Optional[Dict[str, Any]] = None,
        candidate_programs: Optional[List[Dict[str, Any]]] = None,
        candidate_outputs: Optional[List[Dict[str, Any]]] = None,
        posterior_summary: Optional[Dict[str, Any]] = None,
        deliberation_budget: Optional[Dict[str, Any]] = None,
        deliberation_mode: Optional[str] = None,
        uncertainty_vector: Optional[Dict[str, Any]] = None,
        evidence_queue: Optional[List[Dict[str, Any]]] = None,
        workspace_provenance: Optional[Dict[str, Any]] = None,
        safety_budget: Optional[Dict[str, Any]] = None,
        compute_budget: Optional[Dict[str, Any]] = None,
        goal_agenda: Optional[List[Dict[str, Any]]] = None,
        long_horizon_commitments: Optional[List[Dict[str, Any]]] = None,
    ) -> 'UnifiedCognitiveContext':
        return cls(
            schema_version=str(schema_version or CANONICAL_CONTEXT_VERSION),
            current_goal=str(current_goal or ''),
            current_task=str(current_task or ''),
            active_beliefs_summary=_dict_or_empty(active_beliefs_summary),
            active_hypotheses_summary=_dict_list_or_empty(active_hypotheses_summary),
            plan_state_summary=_dict_or_empty(plan_state_summary),
            self_model_summary=_dict_or_empty(self_model_summary),
            recent_failure_profile=_dict_list_or_empty(recent_failure_profile),
            recent_progress_markers=_dict_list_or_empty(recent_progress_markers),
            retrieval_pressure=float(retrieval_pressure or 0.0),
            retrieval_triggered=bool(retrieval_triggered),
            probe_pressure=float(probe_pressure or 0.0),
            resource_pressure=str(resource_pressure or 'normal'),
            world_shift_risk=float(world_shift_risk or 0.0),
            task_frame_summary=_dict_or_empty(task_frame_summary),
            object_bindings_summary=_dict_or_empty(object_bindings_summary),
            goal_hypotheses_summary=_dict_list_or_empty(goal_hypotheses_summary),
            solver_state_summary=_dict_or_empty(solver_state_summary),
            mechanism_hypotheses_summary=_dict_list_or_empty(mechanism_hypotheses_summary),
            mechanism_control_summary=_dict_or_empty(mechanism_control_summary),
            surfaced_representations=_dict_list_or_empty(surfaced_representations),
            competing_hypotheses=_dict_list_or_empty(competing_hypotheses),
            ranked_discriminating_experiments=_dict_list_or_empty(ranked_discriminating_experiments),
            candidate_tests=_dict_list_or_empty(candidate_tests),
            active_skills=_dict_list_or_empty(active_skills),
            transfer_candidates=_dict_list_or_empty(transfer_candidates),
            identity_state=_dict_or_empty(identity_state),
            autobiographical_state=_dict_or_empty(autobiographical_state),
            candidate_programs=_dict_list_or_empty(candidate_programs),
            candidate_outputs=_dict_list_or_empty(candidate_outputs),
            posterior_summary=_dict_or_empty(posterior_summary),
            deliberation_budget=_dict_or_empty(deliberation_budget),
            deliberation_mode=str(deliberation_mode or 'reactive'),
            uncertainty_vector=_dict_or_empty(uncertainty_vector),
            evidence_queue=_dict_list_or_empty(evidence_queue),
            workspace_provenance=_dict_or_empty(workspace_provenance),
            safety_budget=_dict_or_empty(safety_budget),
            compute_budget=_dict_or_empty(compute_budget),
            goal_agenda=_dict_list_or_empty(goal_agenda),
            long_horizon_commitments=_dict_list_or_empty(long_horizon_commitments),
        )

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> Tuple['UnifiedCognitiveContext', List[str]]:
        """Build canonical context from arbitrary payload with explicit degradation warnings."""
        raw = payload if isinstance(payload, dict) else {}
        warnings: List[str] = []

        if str(raw.get('schema_version', '')) != CANONICAL_CONTEXT_VERSION:
            warnings.append('context.schema_version_mismatch_or_missing')

        missing = [key for key in CANONICAL_CONTEXT_FIELDS if key not in raw and key != 'schema_version']
        if missing:
            warnings.append(f"context.missing_fields:{','.join(missing)}")

        ctx = cls.from_parts(
            schema_version=str(raw.get('schema_version') or CANONICAL_CONTEXT_VERSION),
            current_goal=raw.get('current_goal'),
            current_task=raw.get('current_task'),
            active_beliefs_summary=raw.get('active_beliefs_summary'),
            active_hypotheses_summary=raw.get('active_hypotheses_summary'),
            plan_state_summary=raw.get('plan_state_summary'),
            self_model_summary=raw.get('self_model_summary'),
            recent_failure_profile=raw.get('recent_failure_profile'),
            recent_progress_markers=raw.get('recent_progress_markers'),
            retrieval_pressure=raw.get('retrieval_pressure', 0.0),
            retrieval_triggered=raw.get('retrieval_triggered', False),
            probe_pressure=raw.get('probe_pressure', 0.0),
            resource_pressure=raw.get('resource_pressure', 'normal'),
            world_shift_risk=raw.get('world_shift_risk', 0.0),
            task_frame_summary=raw.get('task_frame_summary'),
            object_bindings_summary=raw.get('object_bindings_summary'),
            goal_hypotheses_summary=raw.get('goal_hypotheses_summary'),
            solver_state_summary=raw.get('solver_state_summary'),
            mechanism_hypotheses_summary=raw.get('mechanism_hypotheses_summary'),
            mechanism_control_summary=raw.get('mechanism_control_summary'),
            surfaced_representations=raw.get('surfaced_representations'),
            competing_hypotheses=raw.get('competing_hypotheses'),
            ranked_discriminating_experiments=raw.get('ranked_discriminating_experiments'),
            candidate_tests=raw.get('candidate_tests'),
            active_skills=raw.get('active_skills'),
            transfer_candidates=raw.get('transfer_candidates'),
            identity_state=raw.get('identity_state'),
            autobiographical_state=raw.get('autobiographical_state'),
            candidate_programs=raw.get('candidate_programs'),
            candidate_outputs=raw.get('candidate_outputs'),
            posterior_summary=raw.get('posterior_summary'),
            deliberation_budget=raw.get('deliberation_budget'),
            deliberation_mode=raw.get('deliberation_mode', 'reactive'),
            uncertainty_vector=raw.get('uncertainty_vector'),
            evidence_queue=raw.get('evidence_queue'),
            workspace_provenance=raw.get('workspace_provenance'),
            safety_budget=raw.get('safety_budget'),
            compute_budget=raw.get('compute_budget'),
            goal_agenda=raw.get('goal_agenda'),
            long_horizon_commitments=raw.get('long_horizon_commitments'),
        )
        return ctx, warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'current_goal': self.current_goal,
            'current_task': self.current_task,
            'active_beliefs_summary': dict(self.active_beliefs_summary),
            'active_hypotheses_summary': [dict(item) for item in self.active_hypotheses_summary],
            'plan_state_summary': dict(self.plan_state_summary),
            'self_model_summary': dict(self.self_model_summary),
            'recent_failure_profile': [dict(item) for item in self.recent_failure_profile],
            'recent_progress_markers': [dict(item) for item in self.recent_progress_markers],
            'retrieval_pressure': float(self.retrieval_pressure),
            'retrieval_triggered': bool(self.retrieval_triggered),
            'probe_pressure': float(self.probe_pressure),
            'resource_pressure': self.resource_pressure,
            'world_shift_risk': float(self.world_shift_risk),
            'task_frame_summary': dict(self.task_frame_summary),
            'object_bindings_summary': dict(self.object_bindings_summary),
            'goal_hypotheses_summary': [dict(item) for item in self.goal_hypotheses_summary],
            'solver_state_summary': dict(self.solver_state_summary),
            'mechanism_hypotheses_summary': [dict(item) for item in self.mechanism_hypotheses_summary],
            'mechanism_control_summary': dict(self.mechanism_control_summary),
            'surfaced_representations': [dict(item) for item in self.surfaced_representations],
            'competing_hypotheses': [dict(item) for item in self.competing_hypotheses],
            'ranked_discriminating_experiments': [dict(item) for item in self.ranked_discriminating_experiments],
            'candidate_tests': [dict(item) for item in self.candidate_tests],
            'active_skills': [dict(item) for item in self.active_skills],
            'transfer_candidates': [dict(item) for item in self.transfer_candidates],
            'identity_state': dict(self.identity_state),
            'autobiographical_state': dict(self.autobiographical_state),
            'candidate_programs': [dict(item) for item in self.candidate_programs],
            'candidate_outputs': [dict(item) for item in self.candidate_outputs],
            'posterior_summary': dict(self.posterior_summary),
            'deliberation_budget': dict(self.deliberation_budget),
            'deliberation_mode': self.deliberation_mode,
            'uncertainty_vector': dict(self.uncertainty_vector),
            'evidence_queue': [dict(item) for item in self.evidence_queue],
            'workspace_provenance': dict(self.workspace_provenance),
            'safety_budget': dict(self.safety_budget),
            'compute_budget': dict(self.compute_budget),
            'goal_agenda': [dict(item) for item in self.goal_agenda],
            'long_horizon_commitments': [dict(item) for item in self.long_horizon_commitments],
        }

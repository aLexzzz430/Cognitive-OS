from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Stage1RetrievalInput:
    obs_before: Dict[str, Any]
    context: Dict[str, Any]
    continuity_snapshot: Dict[str, Any]


@dataclass
class Stage1RetrievalOutput:
    query: Any
    retrieve_result: Any
    surfaced: List[Any]
    surfacing_protocol: Dict[str, Any]
    llm_retrieval_ctx: Any
    budget: Dict[str, Any]


@dataclass
class Stage2CandidateGenerationInput:
    obs_before: Dict[str, Any]
    surfaced: List[Any]
    continuity_snapshot: Dict[str, Any]
    frame: Any


@dataclass
class Stage2PlanConstraintsInput:
    obs_before: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]


@dataclass
class Stage2SelfModelSuppressionInput:
    candidate_actions: List[Dict[str, Any]]
    continuity_snapshot: Dict[str, Any]
    obs_before: Optional[Dict[str, Any]] = None


@dataclass
class Stage2PredictionBridgeInput:
    bridge: Any


@dataclass
class Stage2GovernanceInput:
    action_to_use: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]
    arm_meta: Dict[str, Any]
    continuity_snapshot: Dict[str, Any]
    obs_before: Dict[str, Any]
    decision_outcome: Any
    frame: Any


@dataclass
class Stage3ExecutionInput:
    action_to_use: Dict[str, Any]
    query: Any
    obs_before: Dict[str, Any]


@dataclass
class Stage3ExecutionOutput:
    result: Dict[str, Any]
    reward: float


@dataclass
class Stage5EvidenceCommitInput:
    action_to_use: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class Stage5EvidenceCommitOutput:
    validated: List[Any]
    committed_ids: List[str]
    formal_evidence_ids: List[str] = field(default_factory=list)
    formal_evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    formal_evidence_summary: Dict[str, Any] = field(default_factory=dict)
    outcome_model_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage6PostCommitInput:
    committed_ids: List[str]
    obs_before: Dict[str, Any]
    result: Dict[str, Any]
    action_to_use: Dict[str, Any]
    reward: float


@dataclass
class Stage6PostCommitOutput:
    integration_summary: Dict[str, Any]


@dataclass
class PostCommitIntegrationInput:
    committed_ids: List[str]
    obs_before: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class PostCommitIntegrationOutput:
    integration_summary: Dict[str, Any]


@dataclass
class ProcessGraduationCandidatesInput:
    episode: int
    tick: int


@dataclass
class ProcessGraduationCandidatesOutput:
    proposals_committed: int


@dataclass
class ApplyLearningUpdatesInput:
    assignments: List[Any]


@dataclass
class ApplyLearningUpdatesOutput:
    updates_logged: int

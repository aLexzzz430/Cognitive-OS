from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalStageInput:
    obs_before: Dict[str, Any]
    context: Dict[str, Any]
    continuity_snapshot: Dict[str, Any]


@dataclass
class RetrievalStageOutput:
    query: Any
    retrieve_result: Any
    surfaced: List[Any]
    surfacing_protocol: Dict[str, Any]
    llm_retrieval_ctx: Any
    budget: Dict[str, Any]


@dataclass
class PlannerStageInput:
    obs_before: Dict[str, Any]
    surfaced: List[Any]
    continuity_snapshot: Dict[str, Any]
    frame: Any


@dataclass
class PlannerStageOutput:
    raw_base_action: Dict[str, Any]
    base_action: Dict[str, Any]
    arm_action: Dict[str, Any]
    arm_meta: Dict[str, Any]
    plan_tick_meta: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]
    visible_functions: List[str]
    discovered_functions: List[str]
    raw_candidates_snapshot: List[Dict[str, Any]]
    decision_context: Dict[str, Any]
    stage_metrics: Dict[str, Any]
    deliberation_result: Dict[str, Any]


@dataclass
class GovernanceStageInput:
    action_to_use: Dict[str, Any]
    planner_output: PlannerStageOutput
    continuity_snapshot: Dict[str, Any]
    obs_before: Dict[str, Any]
    surfaced: List[Any]
    frame: Any


@dataclass
class GovernanceStageOutput:
    candidate_actions: List[Dict[str, Any]]
    decision_outcome: Any
    decision_arbiter_selected: Optional[Dict[str, Any]]
    action_to_use: Dict[str, Any]
    governance_result: Dict[str, Any]


@dataclass
class StateSyncStageInput:
    continuity_snapshot: Dict[str, Any]
    surfaced: List[Any]
    action_to_use: Dict[str, Any]
    result: Dict[str, Any]
    reward: float
    terminal: bool


@dataclass
class StateSyncStageOutput:
    next_obs: Dict[str, Any]

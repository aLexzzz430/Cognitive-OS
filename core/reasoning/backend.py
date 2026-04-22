from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class DeliberationBudget:
    mode: str = "fast"
    depth: int = 1
    branch_budget: int = 2
    verification_budget: int = 0
    hypothesis_limit: int = 3
    test_limit: int = 3
    program_limit: int = 0
    output_limit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "depth": self.depth,
            "branch_budget": self.branch_budget,
            "verification_budget": self.verification_budget,
            "hypothesis_limit": self.hypothesis_limit,
            "test_limit": self.test_limit,
            "program_limit": self.program_limit,
            "output_limit": self.output_limit,
        }


@dataclass(frozen=True)
class ReasoningRequest:
    workspace: Dict[str, Any]
    obs: Dict[str, Any]
    surfaced: Sequence[Any]
    candidate_actions: Sequence[Dict[str, Any]]
    continuity_snapshot: Dict[str, Any]
    task_family: str = ""
    available_functions: Sequence[str] = ()
    llm_client: Any = None
    structured_answer_synthesizer: Any = None


@dataclass
class ReasoningResult:
    ranked_candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidate_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    ranked_discriminating_experiments: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidate_tests: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidate_programs: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidate_outputs: List[Dict[str, Any]] = field(default_factory=list)
    active_test_ids: List[str] = field(default_factory=list)
    deliberation_trace: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rollout_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    posterior_summary: Dict[str, Any] = field(default_factory=dict)
    control_policy: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    backend: str = "symbolic"
    mode: str = "fast"
    probe_before_commit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranked_candidate_actions": [dict(item) for item in self.ranked_candidate_actions if isinstance(item, dict)],
            "ranked_candidate_hypotheses": [dict(item) for item in self.ranked_candidate_hypotheses if isinstance(item, dict)],
            "ranked_discriminating_experiments": [dict(item) for item in self.ranked_discriminating_experiments if isinstance(item, dict)],
            "ranked_candidate_tests": [dict(item) for item in self.ranked_candidate_tests if isinstance(item, dict)],
            "ranked_candidate_programs": [dict(item) for item in self.ranked_candidate_programs if isinstance(item, dict)],
            "ranked_candidate_outputs": [dict(item) for item in self.ranked_candidate_outputs if isinstance(item, dict)],
            "active_test_ids": [str(item) for item in self.active_test_ids if str(item or "").strip()],
            "deliberation_trace": [dict(item) for item in self.deliberation_trace if isinstance(item, dict)],
            "rejected_candidates": [dict(item) for item in self.rejected_candidates if isinstance(item, dict)],
            "rollout_predictions": {
                str(key): dict(value)
                for key, value in self.rollout_predictions.items()
                if isinstance(value, dict)
            },
            "posterior_summary": dict(self.posterior_summary),
            "control_policy": dict(self.control_policy),
            "budget": dict(self.budget),
            "backend": self.backend,
            "mode": self.mode,
            "probe_before_commit": bool(self.probe_before_commit),
        }


class ReasoningBackend(Protocol):
    name: str

    def deliberate(
        self,
        request: ReasoningRequest,
        budget: DeliberationBudget,
    ) -> ReasoningResult:
        ...

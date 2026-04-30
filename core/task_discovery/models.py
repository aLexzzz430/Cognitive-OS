from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence


TASK_DISCOVERY_VERSION = "conos.task_discovery/v1"

TASK_SOURCES = {
    "failure_residue",
    "goal_gap",
    "user_feedback",
    "code_health",
    "hypothesis",
    "opportunity",
    "self_model",
    "skill_learning",
}

READ_ONLY_ACTIONS = ["read_logs", "read_reports", "read_files", "run_readonly_analysis", "write_report"]
ANALYSIS_ACTIONS = [*READ_ONLY_ACTIONS, "run_eval", "run_tests"]
LIMITED_L2_ACTIONS = [*ANALYSIS_ACTIONS, "propose_patch", "edit_in_mirror"]


def clamp01(value: Any, *, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    return max(0.0, min(1.0, number))


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw: Sequence[Any]
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    result: list[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def dict_list(value: Any) -> list[Dict[str, Any]]:
    if isinstance(value, Mapping):
        candidates = value.get("items") or value.get("records") or value.get("hypotheses") or value.get("goals") or []
    else:
        candidates = value
    if not isinstance(candidates, (list, tuple)):
        return []
    return [dict(item) for item in candidates if isinstance(item, Mapping)]


@dataclass
class GoalLedger:
    north_star: str = "Con OS should become a local-first, evidence-governed general intelligence runtime."
    active_goals: list[Dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    evidence: list[Dict[str, Any]] = field(default_factory=list)
    open_gaps: list[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GoalLedger":
        values = dict(payload or {})
        return cls(
            north_star=str(values.get("north_star") or values.get("northStar") or cls.north_star).strip(),
            active_goals=dict_list(values.get("active_goals") or values.get("activeGoals") or []),
            constraints=string_list(values.get("constraints")),
            evidence=dict_list(values.get("evidence")),
            open_gaps=dict_list(values.get("open_gaps") or values.get("openGaps") or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskCandidate:
    task_id: str
    source: str
    observation: str
    gap: str
    proposed_task: str
    expected_value: float
    goal_alignment: float
    evidence_strength: float
    feasibility: float
    risk: float
    cost: float
    reversibility: float
    distraction_penalty: float
    priority: float = 0.0
    success_condition: str = ""
    evidence_needed: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    status: str = "candidate"
    permission_level: str = "L1"
    requires_human_approval: bool = False
    source_refs: list[str] = field(default_factory=list)
    dedup_key: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = self.source if self.source in TASK_SOURCES else "opportunity"
        for name in (
            "expected_value",
            "goal_alignment",
            "evidence_strength",
            "feasibility",
            "risk",
            "cost",
            "reversibility",
            "distraction_penalty",
            "priority",
        ):
            setattr(self, name, clamp01(getattr(self, name)))
        self.evidence_needed = string_list(self.evidence_needed)
        self.allowed_actions = string_list(self.allowed_actions)
        self.forbidden_actions = string_list(self.forbidden_actions)
        self.source_refs = string_list(self.source_refs)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TASK_DISCOVERY_VERSION
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskCandidate":
        values = dict(payload or {})
        return cls(
            task_id=str(values.get("task_id") or ""),
            source=str(values.get("source") or "opportunity"),
            observation=str(values.get("observation") or ""),
            gap=str(values.get("gap") or ""),
            proposed_task=str(values.get("proposed_task") or ""),
            expected_value=clamp01(values.get("expected_value"), fallback=0.5),
            goal_alignment=clamp01(values.get("goal_alignment"), fallback=0.5),
            evidence_strength=clamp01(values.get("evidence_strength"), fallback=0.5),
            feasibility=clamp01(values.get("feasibility"), fallback=0.5),
            risk=clamp01(values.get("risk"), fallback=0.5),
            cost=clamp01(values.get("cost"), fallback=0.5),
            reversibility=clamp01(values.get("reversibility"), fallback=0.5),
            distraction_penalty=clamp01(values.get("distraction_penalty"), fallback=0.5),
            priority=clamp01(values.get("priority"), fallback=0.0),
            success_condition=str(values.get("success_condition") or ""),
            evidence_needed=string_list(values.get("evidence_needed")),
            allowed_actions=string_list(values.get("allowed_actions")),
            forbidden_actions=string_list(values.get("forbidden_actions")),
            status=str(values.get("status") or "candidate"),
            permission_level=str(values.get("permission_level") or "L1"),
            requires_human_approval=bool(values.get("requires_human_approval", False)),
            source_refs=string_list(values.get("source_refs")),
            dedup_key=str(values.get("dedup_key") or ""),
            created_at=str(values.get("created_at") or ""),
            metadata=dict(values.get("metadata") or {}) if isinstance(values.get("metadata"), Mapping) else {},
        )

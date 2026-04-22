"""
planner/plan_schema.py

Sprint 3: 正式规划器官

定义计划相关核心数据结构:
- Plan: 完整计划
- PlanStep: 单个计划步骤
- ExitCriteria: 退出条件
- PlanStatus: 计划状态枚举

Rules:
- 第一版只做结构定义
- 不做复杂搜索
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
from enum import Enum

from core.runtime_budget import (
    merge_llm_capability_specs,
    resolve_llm_capability_policies,
    resolve_llm_capability_policy_entries,
    resolve_llm_route_policies,
)


_SUCCESS_ANY_OF_RE = re.compile(
    r"discovered_functions\s+contains\s+any_of\(([^)]*)\)",
    re.IGNORECASE,
)
_SUCCESS_NON_EMPTY_RE = re.compile(
    r"discovered_functions\s+non_empty",
    re.IGNORECASE,
)
_SUCCESS_RATIO_RE = re.compile(
    r"discovered_ratio_in_universe\s*>=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_SUCCESS_UNIVERSE_RE = re.compile(
    r"universe=([^)]+)",
    re.IGNORECASE,
)


def _normalized_name_set(values: Any) -> set[str]:
    normalized = set()
    for value in list(values or []):
        text = str(value or "").strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _parse_csv_names(raw: str) -> set[str]:
    return {
        str(item or "").strip().lower()
        for item in str(raw or "").split(",")
        if str(item or "").strip()
    }


class PlanStatus(Enum):
    """计划状态"""
    ACTIVE = "active"           # 计划执行中
    COMPLETED = "completed"     # 计划完成
    ABANDONED = "abandoned"     # 计划放弃
    BLOCKED = "blocked"         # 计划阻塞
    REVISED = "revised"         # 计划已修订


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"          # 待执行
    IN_PROGRESS = "in_progress" # 执行中
    COMPLETED = "completed"     # 已完成
    SKIPPED = "skipped"        # 已跳过
    FAILED = "failed"          # 失败


@dataclass
class ExitCriteria:
    """
    计划退出条件.
    
    第一版简单条件:
    - max_steps: 最多步数
    - max_ticks: 最多 tick 数
    - target_reward: 目标奖励
    - success_indicator: 成功指标
    """
    max_steps: int = 10
    max_ticks: int = 50
    target_reward: Optional[float] = None
    success_indicator: Optional[str] = None  # e.g., "discovered_functions contains X"

    def _clause_satisfied(self, clause: str, context: Dict[str, Any]) -> bool:
        discovered = _normalized_name_set(context.get('discovered_functions', []))
        clause_text = str(clause or '').strip()
        if not clause_text:
            return False

        if _SUCCESS_NON_EMPTY_RE.search(clause_text):
            return bool(discovered)

        any_of_match = _SUCCESS_ANY_OF_RE.search(clause_text)
        if any_of_match:
            targets = _parse_csv_names(any_of_match.group(1))
            if targets and discovered.intersection(targets):
                return True

        ratio_match = _SUCCESS_RATIO_RE.search(clause_text)
        if ratio_match:
            try:
                threshold = float(ratio_match.group(1))
            except (TypeError, ValueError):
                threshold = 1.0
            universe = _normalized_name_set(context.get('planning_universe', []))
            if not universe:
                universe_match = _SUCCESS_UNIVERSE_RE.search(clause_text)
                if universe_match:
                    universe = _parse_csv_names(universe_match.group(1))
            if not universe and any_of_match:
                universe = _parse_csv_names(any_of_match.group(1))
            if universe:
                discovered_in_universe = discovered.intersection(universe)
                ratio = len(discovered_in_universe) / float(len(universe))
                if ratio >= threshold:
                    return True

        return False
    
    def is_satisfied(
        self,
        current_step: int,
        current_ticks: int,
        current_reward: float,
        context: Dict[str, Any],
    ) -> bool:
        """检查退出条件是否满足"""
        if current_step >= self.max_steps:
            return True
        if current_ticks >= self.max_ticks:
            return True
        if self.target_reward is not None and current_reward >= self.target_reward:
            return True
        if self.success_indicator:
            clauses = [
                str(part or '').strip()
                for part in re.split(r"\s+or\s+", str(self.success_indicator or ''), flags=re.IGNORECASE)
                if str(part or '').strip()
            ]
            for clause in clauses or [str(self.success_indicator or '').strip()]:
                if self._clause_satisfied(clause, context):
                    return True
        return False
    
    def to_dict(self) -> dict:
        return {
            'max_steps': self.max_steps,
            'max_ticks': self.max_ticks,
            'target_reward': self.target_reward,
            'success_indicator': self.success_indicator,
        }


@dataclass
class PlanStep:
    """
    单个计划步骤.
    
    第一版简单结构:
    - step_id: 步骤 ID
    - description: 步骤描述
    - intent: 步骤意图 (explore/compute/test)
    - target: 目标函数或状态
    - constraints: 约束条件
    - status: 当前状态
    """
    step_id: str
    description: str
    intent: str = "explore"  # explore / exploit / test / compute / wait
    
    # 步骤目标
    target_function: Optional[str] = None  # e.g., "join_tables"
    target_state: Optional[str] = None     # e.g., "state_with_join_capability"
    
    # 约束
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # 状态
    status: StepStatus = StepStatus.PENDING
    execution_attempts: int = 0
    execution_result: Optional[str] = None

    # 执行治理
    verification_gate: Dict[str, Any] = field(default_factory=dict)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    retry_state: Dict[str, Any] = field(default_factory=dict)
    assigned_worker: Dict[str, Any] = field(default_factory=dict)
    approval_requirement: Dict[str, Any] = field(default_factory=dict)
    approval_state: Dict[str, Any] = field(default_factory=dict)
    governance_memory: Dict[str, Any] = field(default_factory=dict)
    branch_targets: List[Dict[str, Any]] = field(default_factory=list)
    rollback_edge: Dict[str, Any] = field(default_factory=dict)
    llm_route_policies: Dict[str, Any] = field(default_factory=dict)
    llm_capability_policies: Dict[str, Any] = field(default_factory=dict)
    
    def mark_in_progress(self):
        self.status = StepStatus.IN_PROGRESS
        self.execution_attempts += 1
    
    def mark_completed(self, result: str = "success"):
        self.status = StepStatus.COMPLETED
        self.execution_result = result
    
    def mark_skipped(self, reason: str = ""):
        self.status = StepStatus.SKIPPED
        self.execution_result = reason
    
    def mark_failed(self, reason: str = ""):
        self.status = StepStatus.FAILED
        self.execution_result = reason
    
    def to_dict(self) -> dict:
        capability_specs = resolve_llm_capability_policies(self.llm_capability_policies)
        return {
            'step_id': self.step_id,
            'description': self.description,
            'intent': self.intent,
            'target_function': self.target_function,
            'target_state': self.target_state,
            'constraints': self.constraints,
            'status': self.status.value,
            'execution_attempts': self.execution_attempts,
            'execution_result': self.execution_result,
            'verification_gate': dict(self.verification_gate),
            'retry_policy': dict(self.retry_policy),
            'retry_state': dict(self.retry_state),
            'assigned_worker': dict(self.assigned_worker),
            'approval_requirement': dict(self.approval_requirement),
            'approval_state': dict(self.approval_state),
            'governance_memory': dict(self.governance_memory),
            'branch_targets': [dict(item) for item in list(self.branch_targets or []) if isinstance(item, dict)],
            'rollback_edge': dict(self.rollback_edge),
            'llm_route_policies': resolve_llm_route_policies(self.llm_route_policies),
            'llm_capability_policies': capability_specs,
            'llm_capability_policy_entries': resolve_llm_capability_policy_entries(capability_specs),
        }


@dataclass
class Plan:
    """
    完整计划.
    
    第一版结构:
    - plan_id: 计划 ID
    - goal: 计划目标描述
    - steps: 步骤列表
    - exit_criteria: 退出条件
    - status: 当前状态
    - current_step_index: 当前步骤索引
    - revision_count: 修订次数
    - created_episode: 创建时的 episode
    - created_tick: 创建时的 tick
    """
    plan_id: str
    goal: str
    
    steps: List[PlanStep] = field(default_factory=list)
    exit_criteria: ExitCriteria = field(default_factory=ExitCriteria)
    
    status: PlanStatus = PlanStatus.ACTIVE
    current_step_index: int = 0
    
    # 追踪
    revision_count: int = 0
    created_episode: int = 0
    created_tick: int = 0
    
    # 元数据
    parent_plan_id: Optional[str] = None  # 如果是修订版，指向父计划
    revision_reasons: List[str] = field(default_factory=list)
    planning_contract: Dict[str, Any] = field(default_factory=dict)
    approval_contract: Dict[str, Any] = field(default_factory=dict)
    verification_contract: Dict[str, Any] = field(default_factory=dict)
    completion_contract: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def current_step(self) -> Optional[PlanStep]:
        """获取当前步骤"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def is_complete(self) -> bool:
        """计划是否完成"""
        return self.status == PlanStatus.COMPLETED
    
    @property
    def remaining_steps(self) -> int:
        """剩余步骤数"""
        return len(self.steps) - self.current_step_index
    
    def advance_step(self):
        """前进到下一步"""
        if self.current_step:
            self.current_step.mark_completed()
        self.current_step_index += 1
        
        if self.current_step_index >= len(self.steps):
            self.status = PlanStatus.COMPLETED
    
    def abandon(self, reason: str = ""):
        """放弃计划"""
        self.status = PlanStatus.ABANDONED
        self.revision_reasons.append(f"abandoned: {reason}")
    
    def mark_blocked(self, reason: str = ""):
        """标记计划阻塞"""
        self.status = PlanStatus.BLOCKED
        self.revision_reasons.append(f"blocked: {reason}")
    
    def revise(self, new_steps: List[PlanStep] = None):
        """修订计划"""
        self.revision_count += 1
        self.status = PlanStatus.REVISED
        if new_steps:
            self.steps = new_steps
            self.current_step_index = 0
            self.status = PlanStatus.ACTIVE
    
    def to_dict(self) -> dict:
        return {
            'plan_id': self.plan_id,
            'goal': self.goal,
            'steps': [s.to_dict() for s in self.steps],
            'exit_criteria': self.exit_criteria.to_dict(),
            'status': self.status.value,
            'current_step_index': self.current_step_index,
            'revision_count': self.revision_count,
            'created_episode': self.created_episode,
            'created_tick': self.created_tick,
            'parent_plan_id': self.parent_plan_id,
            'revision_reasons': self.revision_reasons,
            'remaining_steps': self.remaining_steps,
            'planning_contract': _normalized_contract(self.planning_contract),
            'approval_contract': dict(self.approval_contract),
            'verification_contract': dict(self.verification_contract),
            'completion_contract': dict(self.completion_contract),
        }


def _normalized_contract(value: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(value or {})
    if "llm_route_policies" in payload:
        payload["llm_route_policies"] = resolve_llm_route_policies(payload.get("llm_route_policies", {}))
    if "llm_capability_policies" in payload or "llm_capability_policy_entries" in payload:
        capability_specs = merge_llm_capability_specs(
            payload.get("llm_capability_policies", {}),
            payload.get("llm_capability_policy_entries", []),
        )
        payload["llm_capability_policies"] = capability_specs
        payload["llm_capability_policy_entries"] = resolve_llm_capability_policy_entries(capability_specs)
    return payload

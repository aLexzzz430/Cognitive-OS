"""
planner/__init__.py

Sprint 3: 正式规划器官

提供显式计划管理.

文件:
- plan_schema.py: Plan, PlanStep, ExitCriteria, PlanStatus
- objective_decomposer.py: 将目标分解为可执行计划
- plan_state.py: 维护当前计划状态
- plan_reviser.py: 基于反馈修订计划

Rules:
- 第一版只做简单线性计划
- 不做复杂搜索或多分支
"""

from planner.plan_schema import (
    PlanStatus,
    StepStatus,
    ExitCriteria,
    PlanStep,
    Plan,
)

from planner.objective_decomposer import ObjectiveDecomposer

from planner.plan_state import PlanState

from planner.plan_reviser import (
    RevisionTrigger,
    PlanReviser,
)
from planner.context_adapter import PlannerContextAdapter

__all__ = [
    # Schema
    'PlanStatus',
    'StepStatus',
    'ExitCriteria',
    'PlanStep',
    'Plan',
    # Decomposer
    'ObjectiveDecomposer',
    # State
    'PlanState',
    # Reviser
    'RevisionTrigger',
    'PlanReviser',
    'PlannerContextAdapter',
]

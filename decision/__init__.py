"""
decision/__init__.py

Sprint 2: 正式决策器官

提供统一决策评分和仲裁.

文件:
- utility_schema.py: DecisionCandidate, DecisionScore, DecisionOutcome
- value_model.py: 价值评分模型
- risk_model.py: 风险评分模型
- arbiter.py: DecisionArbiter 决策仲裁器

Rules:
- 第一版只做简单启发式评分
- 不取代 governance gate
- 输出结构化决策理由
"""

from decision.utility_schema import (
    CandidateSource,
    DecisionCandidate,
    ValueScore,
    RiskScore,
    DecisionScore,
    DecisionOutcome,
    ScoreComponent,
    DecisionScoreBreakdown,
)

from decision.value_model import ValueModel
from decision.value_alignment import GoalHealthAssessment, ValueAlignmentPolicy

from decision.risk_model import RiskModel, TypedUncertaintyProfile

from decision.arbiter import DecisionArbiter
from decision.candidate_generator import CandidateGenerator
from decision.mode_arbiter import ModeDecision, infer_mode
from decision.world_model_policy import WorldModelPolicy

__all__ = [
    # Schema
    'CandidateSource',
    'DecisionCandidate',
    'ValueScore',
    'RiskScore',
    'DecisionScore',
    'DecisionOutcome',
    'ScoreComponent',
    'DecisionScoreBreakdown',
    # Models
    'ValueModel',
    'GoalHealthAssessment',
    'ValueAlignmentPolicy',
    'RiskModel',
    'TypedUncertaintyProfile',
    'CandidateGenerator',
    'WorldModelPolicy',
    'ModeDecision',
    'infer_mode',
    # Arbiter
    'DecisionArbiter',
]

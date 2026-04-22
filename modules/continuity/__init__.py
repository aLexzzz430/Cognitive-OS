"""
modules/continuity/__init__.py

Subject Continuity Layer

Four logical ledgers (not necessarily separate files):
1. Identity Ledger - who the agent is, core traits
2. Goal Continuity Ledger - what goals are active, progress
3. Agenda Ledger - what tasks are scheduled/pending
4. Self-Experiment Registry - ongoing experiments, hypotheses about self

Design principle: These are NOT just logs. They persist across sessions
and influence behavior in future rounds.
"""

from .ledger import (
    IdentityLedger,
    GoalContinuityLedger, 
    AgendaLedger,
    SelfExperimentRegistry,
    ContinuityManager,
)
from .continuity_guard import (
    IllegalInheritancePolicy,
    validate_resume,
    detect_identity_drift,
    detect_goal_drift,
    detect_illegal_state_inheritance,
)
from .estimator import estimate_continuity_confidence
from .subject_trace import SubjectTrace, SubjectTraceEvent

__all__ = [
    'IdentityLedger',
    'GoalContinuityLedger',
    'AgendaLedger', 
    'SelfExperimentRegistry',
    'ContinuityManager',
    'IllegalInheritancePolicy',
    'validate_resume',
    'detect_identity_drift',
    'detect_goal_drift',
    'detect_illegal_state_inheritance',
    'estimate_continuity_confidence',
    'SubjectTrace',
    'SubjectTraceEvent',
]

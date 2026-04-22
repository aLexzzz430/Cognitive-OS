"""
modules/teacher/__init__.py

Teacher Protocol Layer - Formal Human Assistance Entry Points

Five formal entry points:
1. teacher_proposal - Human proposes a new skill/hypothesis/action
2. teacher_critique - Human critiques existing proposals
3. teacher_labeling - Human labels data/experience
4. teacher_task_injection - Human injects new tasks/goals
5. teacher_adjudication - Human adjudicate disputes between agents

Each entry point has:
- can_do: what the human is allowed to do
- cannot_do: what the human cannot do
- must_exit: when human must step back
- audit_trail: how decisions are logged
"""

from .protocol import TeacherProtocol, TeacherAction, TeacherEntry

__all__ = [
    'TeacherProtocol',
    'TeacherAction', 
    'TeacherEntry',
]
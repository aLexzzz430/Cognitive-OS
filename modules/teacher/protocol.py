"""
modules/teacher/protocol.py

Teacher Protocol Layer - Formal Human Assistance Entry Points

Design principles:
- Human help is formal, not informal
- All assistance goes through explicit entry points
- Audit trail is mandatory for every action
- Human must exit when system can handle autonomously
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class TeacherEntry(Enum):
    """Five formal entry points for human assistance."""
    PROPOSAL = 'teacher_proposal'       # Human proposes
    CRITIQUE = 'teacher_critique'       # Human critiques
    LABELING = 'teacher_labeling'      # Human labels
    TASK_INJECTION = 'teacher_task_injection'  # Human injects tasks
    ADJUDICATION = 'teacher_adjudication'  # Human adjudicates


@dataclass
class TeacherAction:
    """A single teacher action with full audit trail."""
    entry: TeacherEntry
    timestamp: float
    agent_id: str
    content: Dict[str, Any]
    
    # What was proposed/critiqued/labeled/injected/adjudicated
    target_id: Optional[str] = None
    target_type: Optional[str] = None  # 'skill', 'hypothesis', 'action', 'object'
    
    # Constraints
    can_do: List[str] = field(default_factory=list)
    cannot_do: List[str] = field(default_factory=list)
    
    # Audit
    rationale: str = ''
    audited_by: Optional[str] = None  # Human who performed action


@dataclass
class AuditEntry:
    """Single entry in the audit trail."""
    timestamp: float
    entry_type: TeacherEntry
    actor: str  # Human or 'system'
    action: str
    target_id: Optional[str]
    outcome: str  # 'accepted', 'rejected', 'absorbed', 'pending'
    notes: str = ''


class TeacherProtocol:
    """
    Formal human assistance protocol with mandatory audit trail.
    
    Usage:
        protocol = TeacherProtocol(agent_id='agent_001')
        protocol.record(TeacherAction(entry=TeacherEntry.PROPOSAL, ...))
    """
    
    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._audit_trail: List[AuditEntry] = []
        self._pending_actions: List[TeacherAction] = []
        self._absorbed_assets: Dict[str, Any] = {}  # Assets absorbed from human help
        
        # Entry point definitions
        self._entry_defs = {
            TeacherEntry.PROPOSAL: {
                'can_do': [
                    'propose new skill candidate',
                    'propose new hypothesis',
                    'propose new representation card',
                    'suggest parameter values',
                ],
                'cannot_do': [
                    'directly execute action',
                    'override system decision',
                    'modify governance rules',
                    'delete audit trail',
                ],
                'must_exit_when': [
                    'system has confirmed skill/hypothesis',
                    'similar proposal exists in store',
                    'proposal contradicts established facts',
                ],
            },
            TeacherEntry.CRITIQUE: {
                'can_do': [
                    'critique existing proposal',
                    'suggest modifications to proposal',
                    'mark proposal as incorrect/incomplete',
                    'provide reasoning for critique',
                ],
                'cannot_do': [
                    'directly modify proposal',
                    'force acceptance of critique',
                    'delete proposal without trace',
                    'critique without evidence/reasoning',
                ],
                'must_exit_when': [
                    'proposal has been modified',
                    'critique has been recorded',
                    'system will handle via proposal_validator',
                ],
            },
            TeacherEntry.LABELING: {
                'can_do': [
                    'label training data',
                    'label hypothesis as true/false',
                    'label object quality',
                    'label skill applicability',
                ],
                'cannot_do': [
                    'retroactively label past experiences',
                    'label without explanation',
                    'label system internals',
                    'modify labels after absorption',
                ],
                'must_exit_when': [
                    'label has been recorded',
                    'system will use labels in training',
                ],
            },
            TeacherEntry.TASK_INJECTION: {
                'can_do': [
                    'inject new task into agenda',
                    'set new goal priority',
                    'inject boundary conditions',
                    'define success criteria',
                ],
                'cannot_do': [
                    'execute task directly',
                    'modify running episode',
                    'override system goals',
                    'inject without priority',
                ],
                'must_exit_when': [
                    'task is in agenda with priority',
                    'system acknowledges task',
                ],
            },
            TeacherEntry.ADJUDICATION: {
                'can_do': [
                    'resolve conflict between agents',
                    'adjudicate hypothesis pair',
                    'adjudicate competing skills',
                    'set final precedence',
                ],
                'cannot_do': [
                    'override system governance',
                    'set rules for future cases',
                    'adjudicate without evidence',
                    'modify competing entities directly',
                ],
                'must_exit_when': [
                    'adjudication is recorded',
                    'system will follow governance',
                ],
            },
        }
    
    # ─────────────────────────────────────────────────
    # Entry Point Methods
    # ─────────────────────────────────────────────────
    
    def teacher_proposal(self, target_id: str, target_type: str,
                         content: Dict[str, Any], rationale: str,
                         actor: str = 'human') -> AuditEntry:
        """
        Human proposes a new skill/hypothesis/representation.
        
        Can do: propose new skill/hypothesis/representation/card, suggest params
        Cannot do: execute, override, modify governance, delete audit
        Must exit when: system confirms, similar exists, contradicts facts
        """
        defs = self._entry_defs[TeacherEntry.PROPOSAL]
        
        action = TeacherAction(
            entry=TeacherEntry.PROPOSAL,
            timestamp=time.time(),
            agent_id=self._agent_id,
            content=content,
            target_id=target_id,
            target_type=target_type,
            can_do=defs['can_do'],
            cannot_do=defs['cannot_do'],
            rationale=rationale,
            audited_by=actor,
        )
        
        # Record to audit trail
        entry = self._record(action)
        
        # Absorb into long-term assets if accepted
        if self._should_absorb(action):
            self._absorb(action)
        
        return entry
    
    def teacher_critique(self, target_id: str, target_type: str,
                         content: Dict[str, Any], rationale: str,
                         actor: str = 'human') -> AuditEntry:
        """
        Human critiques existing proposal.
        
        Can do: critique, suggest modifications, mark incorrect
        Cannot do: directly modify, force acceptance, delete, critique without evidence
        Must exit when: proposal modified, critique recorded, system handles
        """
        defs = self._entry_defs[TeacherEntry.CRITIQUE]
        
        action = TeacherAction(
            entry=TeacherEntry.CRITIQUE,
            timestamp=time.time(),
            agent_id=self._agent_id,
            content=content,
            target_id=target_id,
            target_type=target_type,
            can_do=defs['can_do'],
            cannot_do=defs['cannot_do'],
            rationale=rationale,
            audited_by=actor,
        )
        
        entry = self._record(action)
        
        if self._should_absorb(action):
            self._absorb(action)
        
        return entry
    
    def teacher_labeling(self, target_id: str, target_type: str,
                         labels: Dict[str, Any], rationale: str,
                         actor: str = 'human') -> AuditEntry:
        """
        Human labels data/experience.
        
        Can do: label training data, hypothesis truth, object quality
        Cannot do: retroactive labeling, label internals, modify after absorption
        Must exit when: label recorded, system uses in training
        """
        defs = self._entry_defs[TeacherEntry.LABELING]
        
        content = {'labels': labels}
        
        action = TeacherAction(
            entry=TeacherEntry.LABELING,
            timestamp=time.time(),
            agent_id=self._agent_id,
            content=content,
            target_id=target_id,
            target_type=target_type,
            can_do=defs['can_do'],
            cannot_do=defs['cannot_do'],
            rationale=rationale,
            audited_by=actor,
        )
        
        entry = self._record(action)
        
        if self._should_absorb(action):
            self._absorb(action)
        
        return entry
    
    def teacher_task_injection(self, task: Dict[str, Any], 
                               priority: float, rationale: str,
                               actor: str = 'human') -> AuditEntry:
        """
        Human injects new task/goal into agenda.
        
        Can do: inject task, set priority, inject boundary conditions
        Cannot do: execute, modify episode, override goals, inject without priority
        Must exit when: task in agenda, system acknowledges
        """
        defs = self._entry_defs[TeacherEntry.TASK_INJECTION]
        
        content = {'task': task, 'priority': priority}
        
        action = TeacherAction(
            entry=TeacherEntry.TASK_INJECTION,
            timestamp=time.time(),
            agent_id=self._agent_id,
            content=content,
            target_id=task.get('id'),
            target_type='task',
            can_do=defs['can_do'],
            cannot_do=defs['cannot_do'],
            rationale=rationale,
            audited_by=actor,
        )
        
        entry = self._record(action)
        
        if self._should_absorb(action):
            self._absorb(action)
        
        return entry
    
    def teacher_adjudication(self, conflict_id: str, 
                            resolution: Dict[str, Any], rationale: str,
                            actor: str = 'human') -> AuditEntry:
        """
        Human adjudicate dispute between agents/hypotheses/skills.
        
        Can do: resolve conflict, set precedence, adjudicate
        Cannot do: override governance, set future rules, adjudicate without evidence
        Must exit when: adjudication recorded, system follows governance
        """
        defs = self._entry_defs[TeacherEntry.ADJUDICATION]
        
        content = {'resolution': resolution, 'conflict_id': conflict_id}
        
        action = TeacherAction(
            entry=TeacherEntry.ADJUDICATION,
            timestamp=time.time(),
            agent_id=self._agent_id,
            content=content,
            target_id=conflict_id,
            target_type='conflict',
            can_do=defs['can_do'],
            cannot_do=defs['cannot_do'],
            rationale=rationale,
            audited_by=actor,
        )
        
        entry = self._record(action)
        
        if self._should_absorb(action):
            self._absorb(action)
        
        return entry
    
    # ─────────────────────────────────────────────────
    # Audit Trail
    # ─────────────────────────────────────────────────
    
    def _record(self, action: TeacherAction) -> AuditEntry:
        """Record action to audit trail."""
        # Determine outcome
        if self._is_accepted(action):
            outcome = 'accepted'
        elif self._is_rejected(action):
            outcome = 'rejected'
        elif self._should_absorb(action):
            outcome = 'absorbed'
        else:
            outcome = 'pending'
        
        entry = AuditEntry(
            timestamp=action.timestamp,
            entry_type=action.entry,
            actor=action.audited_by or 'system',
            action=f'{action.entry.value}: {action.target_type}/{action.target_id}',
            target_id=action.target_id,
            outcome=outcome,
            notes=action.rationale,
        )
        
        self._audit_trail.append(entry)
        return entry
    
    def _is_accepted(self, action: TeacherAction) -> bool:
        """Check if action was accepted by system."""
        # System must confirm before acceptance
        # For now, auto-accept proposals with valid rationale
        return len(action.rationale) > 0 and action.target_id is not None
    
    def _is_rejected(self, action: TeacherAction) -> bool:
        """Check if action was rejected."""
        # Reject if contradicts existing facts
        if action.target_id in self._absorbed_assets:
            # Check for contradiction
            existing = self._absorbed_assets[action.target_id]
            if existing.get('contradicts'):
                return True
        return False
    
    def _should_absorb(self, action: TeacherAction) -> bool:
        """
        Should this action be absorbed as long-term asset?
        
        Assets are absorbed when:
        1. Action is valid and complete
        2. System can use it in future rounds
        3. Not just temporary feedback
        """
        # Labeling and proposals are absorbed
        if action.entry in [TeacherEntry.LABELING, TeacherEntry.PROPOSAL]:
            return len(action.rationale) > 0
        
        # Critiques and adjudications are absorbed
        if action.entry in [TeacherEntry.CRITIQUE, TeacherEntry.ADJUDICATION]:
            return len(action.rationale) > 0
        
        # Task injection is absorbed as agenda
        if action.entry == TeacherEntry.TASK_INJECTION:
            return True
        
        return False
    
    def _absorb(self, action: TeacherAction):
        """Absorb action into long-term assets."""
        if action.target_id:
            self._absorbed_assets[action.target_id] = {
                'entry': action.entry.value,
                'timestamp': action.timestamp,
                'content': action.content,
                'actor': action.audited_by,
                'rationale': action.rationale,
            }
    
    def get_audit_trail(self) -> List[AuditEntry]:
        """Return full audit trail."""
        return list(self._audit_trail)
    
    def get_absorbed_assets(self) -> Dict[str, Any]:
        """Return assets absorbed from human help."""
        return dict(self._absorbed_assets)
    
    def get_pending_actions(self) -> List[TeacherAction]:
        """Return pending actions."""
        return list(self._pending_actions)
    
    # ─────────────────────────────────────────────────
    # Exit Conditions
    # ─────────────────────────────────────────────────
    
    def must_exit(self, entry: TeacherEntry) -> bool:
        """
        Check if human must exit for given entry type.
        
        Returns True if system can handle autonomously now.
        """
        # Check if pending actions can be processed
        pending_of_type = [
            a for a in self._pending_actions 
            if a.entry == entry
        ]
        
        if pending_of_type:
            # Still has pending actions
            return False
        
        # Check if absorbed assets cover the entry type
        absorbed_of_type = [
            a for a in self._audit_trail 
            if a.entry_type == entry and a.outcome == 'absorbed'
        ]
        
        # If we have absorbed assets, system should handle
        return len(absorbed_of_type) > 0
    
    def __repr__(self):
        return (f'TeacherProtocol(agent_id={self._agent_id}, '
                f'audit_entries={len(self._audit_trail)}, '
                f'absorbed_assets={len(self._absorbed_assets)})')
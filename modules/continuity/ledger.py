"""
modules/continuity/ledger.py

Subject Continuity Layer - Four Logical Ledgers

1. IdentityLedger - who the agent is, core traits
2. GoalContinuityLedger - active goals, progress tracking
3. AgendaLedger - scheduled/pending tasks
4. SelfExperimentRegistry - ongoing experiments about self

These persist across sessions and influence future behavior.
"""

import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from .continuity_guard import GuardVerdict, IllegalInheritancePolicy, validate_resume
from .subject_trace import SubjectTrace


# ─────────────────────────────────────────────────
# Identity Ledger
# ─────────────────────────────────────────────────

@dataclass
class IdentityEntry:
    """Single entry in identity ledger."""
    timestamp: float
    field: str  # 'name', 'traits', 'values', 'capabilities'
    value: Any
    source: str  # 'self_observed', 'human_labeled', 'derived'


class IdentityLedger:
    """
    Identity Ledger - tracks who the agent is.
    
    Persists across sessions.
    Used to maintain consistent self-concept.
    
    Fields:
    - name: agent identifier
    - traits: stable character traits (e.g., 'careful', 'exploratory')
    - values: guiding principles (e.g., 'truthfulness', 'efficiency')
    - capabilities: known capabilities (updated over time)
    - limitations: known limitations (updated over time)
    """
    
    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._entries: List[IdentityEntry] = []
        self._fields: Dict[str, Any] = {
            'name': agent_id,
            'traits': set(),
            'values': set(),
            'capabilities': set(),
            'limitations': set(),
        }
    
    def observe(self, field: str, value: Any, source: str = 'self_observed'):
        """Record observed identity information."""
        entry = IdentityEntry(
            timestamp=time.time(),
            field=field,
            value=value,
            source=source,
        )
        self._entries.append(entry)
        
        # Update current field value
        if field in self._fields:
            if isinstance(self._fields[field], set):
                if isinstance(value, set):
                    self._fields[field].update(value)
                else:
                    self._fields[field].add(value)
            else:
                self._fields[field] = value
    
    def get(self, field: str, default: Any = None) -> Any:
        """Get current value of field."""
        return self._fields.get(field, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all identity fields."""
        result = dict(self._fields)
        for k, v in result.items():
            if isinstance(v, set):
                result[k] = set(v)  # Return copy
        return result
    
    def get_trait_summary(self) -> str:
        """Get human-readable trait summary."""
        traits = ', '.join(sorted(self._fields.get('traits', set())))
        values = ', '.join(sorted(self._fields.get('values', set())))
        caps = ', '.join(sorted(self._fields.get('capabilities', set())))
        return f"Traits: {traits or 'unknown'}. Values: {values or 'unknown'}. Capabilities: {caps or 'unknown'}"
    
    def get_history(self) -> List[IdentityEntry]:
        """Get full history of identity entries."""
        return list(self._entries)
    
    def __repr__(self):
        return (f"IdentityLedger(id={self._agent_id}, "
                f"traits={len(self._fields.get('traits', set()))}, "
                f"capabilities={len(self._fields.get('capabilities', set()))})")


# ─────────────────────────────────────────────────
# Goal Continuity Ledger
# ─────────────────────────────────────────────────

@dataclass 
class GoalEntry:
    """Single goal with progress tracking."""
    goal_id: str
    description: str
    created_at: float
    updated_at: float
    status: str  # 'active', 'completed', 'abandoned', 'suspended'
    progress: float  # 0.0 to 1.0
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # goal_ids
    source: str = 'self'  # 'self', 'human', 'injected'


class GoalContinuityLedger:
    """
    Goal Continuity Ledger - tracks active goals and progress.
    
    Persists across sessions.
    Goals survive across episodes.
    System knows what it's trying to achieve.
    """
    
    def __init__(self):
        self._goals: Dict[str, GoalEntry] = {}
        self._completed_count: int = 0
        self._abandoned_count: int = 0
    
    def add_goal(self, goal_id: str, description: str, 
                 source: str = 'self',
                 dependencies: List[str] = None) -> GoalEntry:
        """Add a new goal."""
        entry = GoalEntry(
            goal_id=goal_id,
            description=description,
            created_at=time.time(),
            updated_at=time.time(),
            status='active',
            progress=0.0,
            dependencies=dependencies or [],
            source=source,
        )
        self._goals[goal_id] = entry
        return entry
    
    def update_progress(self, goal_id: str, progress: float, 
                        milestone: Dict[str, Any] = None):
        """Update goal progress."""
        if goal_id not in self._goals:
            return
        
        goal = self._goals[goal_id]
        goal.progress = min(1.0, max(0.0, progress))
        goal.updated_at = time.time()
        
        if milestone:
            goal.milestones.append({
                'timestamp': time.time(),
                'progress': progress,
                'description': milestone.get('description', ''),
            })
        
        if goal.progress >= 1.0:
            goal.status = 'completed'
            self._completed_count += 1
    
    def mark_completed(self, goal_id: str):
        """Mark goal as completed."""
        if goal_id in self._goals:
            self._goals[goal_id].status = 'completed'
            self._goals[goal_id].updated_at = time.time()
            self._goals[goal_id].progress = 1.0
            self._completed_count += 1
    
    def mark_abandoned(self, goal_id: str, reason: str = ''):
        """Mark goal as abandoned."""
        if goal_id in self._goals:
            self._goals[goal_id].status = 'abandoned'
            self._goals[goal_id].updated_at = time.time()
            self._abandoned_count += 1
    
    def get_active_goals(self) -> List[GoalEntry]:
        """Get all active goals sorted by priority/age."""
        active = [g for g in self._goals.values() if g.status == 'active']
        return sorted(active, key=lambda g: (g.updated_at, g.created_at))
    
    def get_goal(self, goal_id: str) -> Optional[GoalEntry]:
        """Get specific goal."""
        return self._goals.get(goal_id)
    
    def has_active_goal(self, goal_id: str) -> bool:
        """Check if goal is active."""
        return goal_id in self._goals and self._goals[goal_id].status == 'active'
    
    def get_completion_rate(self) -> float:
        """Get overall goal completion rate."""
        total = len(self._goals)
        if total == 0:
            return 0.0
        return self._completed_count / total
    
    def get_summary(self) -> Dict[str, Any]:
        """Get goal summary."""
        active = len([g for g in self._goals.values() if g.status == 'active'])
        completed = self._completed_count
        abandoned = self._abandoned_count
        return {
            'total': len(self._goals),
            'active': active,
            'completed': completed,
            'abandoned': abandoned,
            'completion_rate': self.get_completion_rate(),
        }
    
    def __repr__(self):
        return (f"GoalContinuityLedger(goals={len(self._goals)}, "
                f"active={len([g for g in self._goals.values() if g.status == 'active'])})")


# ─────────────────────────────────────────────────
# Agenda Ledger
# ─────────────────────────────────────────────────

@dataclass
class AgendaEntry:
    """Single task in agenda."""
    task_id: str
    description: str
    priority: float  # Higher = more important
    created_at: float
    scheduled_for: Optional[float] = None  # timestamp or None
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'cancelled'
    assigned_to: Optional[str] = None  # subsystem or None
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgendaLedger:
    """
    Agenda Ledger - tracks scheduled and pending tasks.
    
    Persists across sessions.
    Tasks can have priorities and dependencies.
    """
    
    def __init__(self):
        self._entries: Dict[str, AgendaEntry] = {}
    
    def add_task(self, task_id: str, description: str,
                 priority: float = 0.5,
                 depends_on: List[str] = None,
                 metadata: Dict[str, Any] = None) -> AgendaEntry:
        """Add task to agenda."""
        entry = AgendaEntry(
            task_id=task_id,
            description=description,
            priority=priority,
            created_at=time.time(),
            depends_on=depends_on or [],
            metadata=metadata or {},
        )
        self._entries[task_id] = entry
        return entry
    
    def update_priority(self, task_id: str, priority: float):
        """Update task priority."""
        if task_id in self._entries:
            self._entries[task_id].priority = priority
    
    def mark_completed(self, task_id: str):
        """Mark task as completed."""
        if task_id in self._entries:
            self._entries[task_id].status = 'completed'
    
    def mark_cancelled(self, task_id: str):
        """Mark task as cancelled."""
        if task_id in self._entries:
            self._entries[task_id].status = 'cancelled'
    
    def get_pending(self) -> List[AgendaEntry]:
        """Get pending tasks sorted by priority."""
        pending = [e for e in self._entries.values() if e.status == 'pending']
        return sorted(pending, key=lambda e: e.priority, reverse=True)
    
    def get_next_task(self) -> Optional[AgendaEntry]:
        """Get highest priority pending task."""
        pending = self.get_pending()
        if pending:
            return pending[0]
        return None
    
    def has_task(self, task_id: str) -> bool:
        """Check if task exists."""
        return task_id in self._entries
    
    def get_all(self) -> List[AgendaEntry]:
        """Get all tasks."""
        return list(self._entries.values())
    
    def __repr__(self):
        pending = len([e for e in self._entries.values() if e.status == 'pending'])
        return f"AgendaLedger(tasks={len(self._entries)}, pending={pending})"


# ─────────────────────────────────────────────────
# Self-Experiment Registry
# ─────────────────────────────────────────────────

@dataclass
class ExperimentEntry:
    """Single self-experiment."""
    exp_id: str
    hypothesis: str  # What we're testing about ourselves
    created_at: float
    status: str = 'running'  # 'running', 'completed', 'failed'
    results: List[Dict[str, Any]] = field(default_factory=list)
    conclusion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfExperimentRegistry:
    """
    Self-Experiment Registry - tracks ongoing experiments about self.
    
    The agent tests hypotheses about its own capabilities/limitations.
    Results are used to update Identity Ledger.
    
    Examples:
    - "Can I learn new skills faster when focused?"
    - "Am I better at exploration or exploitation?"
    - "Do I perform better with hints or without?"
    """
    
    def __init__(self):
        self._experiments: Dict[str, ExperimentEntry] = {}
    
    def start_experiment(self, exp_id: str, hypothesis: str,
                         metadata: Dict[str, Any] = None) -> ExperimentEntry:
        """Start a new self-experiment."""
        entry = ExperimentEntry(
            exp_id=exp_id,
            hypothesis=hypothesis,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._experiments[exp_id] = entry
        return entry
    
    def record_result(self, exp_id: str, result: Dict[str, Any]):
        """Record experiment result."""
        if exp_id in self._experiments:
            self._experiments[exp_id].results.append({
                'timestamp': time.time(),
                'data': result,
            })
    
    def conclude(self, exp_id: str, conclusion: str):
        """Conclude experiment with conclusion."""
        if exp_id in self._experiments:
            self._experiments[exp_id].status = 'completed'
            self._experiments[exp_id].conclusion = conclusion
    
    def fail_experiment(self, exp_id: str, reason: str):
        """Mark experiment as failed."""
        if exp_id in self._experiments:
            self._experiments[exp_id].status = 'failed'
            self._experiments[exp_id].conclusion = reason
    
    def get_running(self) -> List[ExperimentEntry]:
        """Get all running experiments."""
        return [e for e in self._experiments.values() if e.status == 'running']
    
    def get_completed(self) -> List[ExperimentEntry]:
        """Get all completed experiments."""
        return [e for e in self._experiments.values() if e.status == 'completed']
    
    def get(self, exp_id: str) -> Optional[ExperimentEntry]:
        """Get specific experiment."""
        return self._experiments.get(exp_id)
    
    def get_all(self) -> List[ExperimentEntry]:
        """Get all experiments."""
        return list(self._experiments.values())
    
    def __repr__(self):
        running = len(self.get_running())
        completed = len(self.get_completed())
        return f"SelfExperimentRegistry(running={running}, completed={completed})"


# ─────────────────────────────────────────────────
# Continuity Manager
# ─────────────────────────────────────────────────

class ContinuityManager:
    """
    Unified interface to all four ledgers.
    
    Single entry point for continuity management.
    """
    
    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._identity = IdentityLedger(agent_id)
        self._goals = GoalContinuityLedger()
        self._agenda = AgendaLedger()
        self._experiments = SelfExperimentRegistry()
        self._autobiographical_summary: Dict[str, Any] = {}
        self._semantic_memory: Dict[str, Any] = {}
        self._procedural_memory: Dict[str, Any] = {}
        self._transfer_memory: Dict[str, Any] = {}
        self._subject_trace = SubjectTrace(agent_id)
        self._guard_log: List[Dict[str, Any]] = []
        self._last_guard_verdict: Optional[GuardVerdict] = None
        self._illegal_inheritance_policy = IllegalInheritancePolicy.DEGRADED_RESTORE
    
    @property
    def identity(self) -> IdentityLedger:
        return self._identity
    
    @property
    def goals(self) -> GoalContinuityLedger:
        return self._goals
    
    @property
    def agenda(self) -> AgendaLedger:
        return self._agenda
    
    @property
    def experiments(self) -> SelfExperimentRegistry:
        return self._experiments
    
    def tick(self) -> Dict[str, Any]:
        """
        Get continuity state snapshot for current tick.
        
        Used by CoreMainLoop to incorporate continuity into decision-making.
        """
        active_goals = self._goals.get_active_goals()
        next_task = self._agenda.get_next_task()
        running_exps = self._experiments.get_running()
        active_commitments = [
            {
                'commitment': g.description,
                'source': g.source,
                'goal_id': g.goal_id,
                'progress': g.progress,
            }
            for g in active_goals[:4]
        ]
        long_horizon_agenda = [
            {
                'task_id': task.task_id,
                'goal': task.description,
                'priority': task.priority,
                'status': task.status,
            }
            for task in self._agenda.get_all()[:4]
        ]
        approved_experiments = [
            {
                'exp_id': exp.exp_id,
                'hypothesis': exp.hypothesis,
                'status': exp.status,
                'metadata': dict(exp.metadata or {}),
            }
            for exp in self._experiments.get_all()
            if bool(exp.metadata.get('approved', False)) or str(exp.status or '').strip().lower() == 'completed'
        ]

        snapshot = {
            'identity_summary': self._identity.get_trait_summary(),
            'active_goal_count': len(active_goals),
            'top_goal': active_goals[0] if active_goals else None,
            'next_task': next_task,
            'running_experiments': len(running_exps),
            'identity_fields': self._identity.get_all(),
            'goal_summary': self._goals.get_summary(),
            'durable_identity': {
                'agent_id': self._agent_id,
                'traits': sorted(self._identity.get('traits', set())),
                'values': sorted(self._identity.get('values', set())),
                'capabilities': sorted(self._identity.get('capabilities', set())),
                'limitations': sorted(self._identity.get('limitations', set())),
            },
            'active_commitments': active_commitments,
            'long_horizon_agenda': long_horizon_agenda,
            'approved_experiments': approved_experiments,
            'autobiographical_summary': dict(self._autobiographical_summary or {}),
            'semantic_memory': dict(self._semantic_memory or {}),
            'procedural_memory': dict(self._procedural_memory or {}),
            'transfer_memory': dict(self._transfer_memory or {}),
        }
        self._subject_trace.record_tick(snapshot)
        self._subject_trace.record_commitments(active_commitments)
        self._subject_trace.record_autobiographical_summary(self._autobiographical_summary)
        snapshot['subject_continuity'] = self._subject_trace.summarize()
        snapshot['continuity_confidence'] = float(snapshot['subject_continuity'].get('continuity_score', 0.5) or 0.5)
        return snapshot

    def record_autobiographical_summary(self, summary: Dict[str, Any]) -> None:
        self._autobiographical_summary = dict(summary or {}) if isinstance(summary, dict) else {}
        self._subject_trace.record_autobiographical_summary(self._autobiographical_summary)

    def record_memory_summary(
        self,
        *,
        semantic_memory: Optional[Dict[str, Any]] = None,
        procedural_memory: Optional[Dict[str, Any]] = None,
        transfer_memory: Optional[Dict[str, Any]] = None,
    ) -> None:
        if semantic_memory is not None:
            self._semantic_memory = dict(semantic_memory or {}) if isinstance(semantic_memory, dict) else {}
        if procedural_memory is not None:
            self._procedural_memory = dict(procedural_memory or {}) if isinstance(procedural_memory, dict) else {}
        if transfer_memory is not None:
            self._transfer_memory = dict(transfer_memory or {}) if isinstance(transfer_memory, dict) else {}

    def cleanup_agenda(self, now_ts: Optional[float] = None) -> int:
        """Cancel stale pending agenda entries based on metadata expiry window."""
        now = float(now_ts if now_ts is not None else time.time())
        cleaned = 0
        for entry in self._agenda.get_all():
            if entry.status != 'pending':
                continue
            ttl_seconds = entry.metadata.get('ttl_seconds') if isinstance(entry.metadata, dict) else None
            if isinstance(ttl_seconds, (int, float)) and ttl_seconds > 0:
                if now - float(entry.created_at) >= float(ttl_seconds):
                    entry.status = 'cancelled'
                    cleaned += 1
        return cleaned

    def to_dict(self) -> Dict[str, Any]:
        """Serialize continuity ledgers for persistence."""
        payload = {
            'agent_id': self._agent_id,
            'identity': {
                'fields': {
                    k: sorted(v) if isinstance(v, set) else v
                    for k, v in self._identity.get_all().items()
                },
            },
            'goals': [
                {
                    'goal_id': g.goal_id,
                    'description': g.description,
                    'created_at': g.created_at,
                    'updated_at': g.updated_at,
                    'status': g.status,
                    'progress': g.progress,
                    'milestones': list(g.milestones),
                    'dependencies': list(g.dependencies),
                    'source': g.source,
                }
                for g in self._goals._goals.values()
            ],
            'agenda': [
                {
                    'task_id': e.task_id,
                    'description': e.description,
                    'priority': e.priority,
                    'created_at': e.created_at,
                    'scheduled_for': e.scheduled_for,
                    'status': e.status,
                    'assigned_to': e.assigned_to,
                    'depends_on': list(e.depends_on),
                    'metadata': dict(e.metadata),
                }
                for e in self._agenda.get_all()
            ],
            'approved_experiments': [
                {
                    'exp_id': exp.exp_id,
                    'hypothesis': exp.hypothesis,
                    'created_at': exp.created_at,
                    'status': exp.status,
                    'results': list(exp.results),
                    'conclusion': exp.conclusion,
                    'metadata': dict(exp.metadata),
                }
                for exp in self._experiments.get_all()
                if bool(exp.metadata.get('approved', False)) or str(exp.status or '').strip().lower() == 'completed'
            ],
            'autobiographical_summary': dict(self._autobiographical_summary or {}),
            'semantic_memory': dict(self._semantic_memory or {}),
            'procedural_memory': dict(self._procedural_memory or {}),
            'transfer_memory': dict(self._transfer_memory or {}),
            'durable_identity': {
                'agent_id': self._agent_id,
                'traits': sorted(self._identity.get('traits', set())),
                'values': sorted(self._identity.get('values', set())),
                'capabilities': sorted(self._identity.get('capabilities', set())),
                'limitations': sorted(self._identity.get('limitations', set())),
            },
            'active_commitments': [
                {
                    'commitment': g.description,
                    'source': g.source,
                    'goal_id': g.goal_id,
                    'progress': g.progress,
                }
                for g in self._goals.get_active_goals()[:4]
            ],
            'long_horizon_agenda': [
                {
                    'task_id': e.task_id,
                    'goal': e.description,
                    'priority': e.priority,
                    'status': e.status,
                }
                for e in self._agenda.get_all()[:4]
            ],
        }
        verdict = validate_resume(payload, payload, illegal_policy=self._illegal_inheritance_policy)
        self._last_guard_verdict = verdict
        self._guard_log.append({
            'timestamp': time.time(),
            'event': 'persist_validate',
            **verdict.to_dict(),
        })
        return payload

    def emit_state_patch(self) -> Dict[str, Any]:
        """Produce continuity-owned state patch."""
        return {'continuity': self.to_dict()}

    def sync_state(self, state_manager: Any, *, reason: str = 'continuity_snapshot_persist') -> None:
        """Write continuity snapshot through StateManager under continuity module ownership."""
        state_manager.update_state(
            self.emit_state_patch(),
            reason=reason,
            module='continuity',
        )

    def load_from_dict(self, snapshot: Dict[str, Any]) -> bool:
        """Restore continuity ledgers from serialized state."""
        if not isinstance(snapshot, dict):
            return False

        current_snapshot = self.to_dict()
        verdict = validate_resume(current_snapshot, snapshot, illegal_policy=self._illegal_inheritance_policy)
        self._last_guard_verdict = verdict
        self._guard_log.append({
            'timestamp': time.time(),
            'event': 'resume_validate',
            **verdict.to_dict(),
        })
        if not verdict.accepted:
            return False

        snapshot_to_load = verdict.sanitized_snapshot if isinstance(verdict.sanitized_snapshot, dict) else snapshot

        identity = snapshot_to_load.get('identity', {})
        fields = identity.get('fields', {}) if isinstance(identity, dict) else {}
        if isinstance(fields, dict):
            normalized = {}
            for k, v in fields.items():
                normalized[k] = set(v) if isinstance(v, list) and k in ('traits', 'values', 'capabilities', 'limitations') else v
            self._identity._fields.update(normalized)

        goals = snapshot_to_load.get('goals', [])
        if isinstance(goals, list):
            self._goals._goals = {}
            self._goals._completed_count = 0
            self._goals._abandoned_count = 0
            for g in goals:
                if not isinstance(g, dict) or not g.get('goal_id'):
                    continue
                entry = GoalEntry(
                    goal_id=str(g.get('goal_id')),
                    description=str(g.get('description', '')),
                    created_at=float(g.get('created_at', time.time())),
                    updated_at=float(g.get('updated_at', time.time())),
                    status=str(g.get('status', 'active')),
                    progress=float(g.get('progress', 0.0)),
                    milestones=list(g.get('milestones', [])),
                    dependencies=list(g.get('dependencies', [])),
                    source=str(g.get('source', 'self')),
                )
                self._goals._goals[entry.goal_id] = entry
                if entry.status == 'completed':
                    self._goals._completed_count += 1
                if entry.status == 'abandoned':
                    self._goals._abandoned_count += 1

        agenda = snapshot_to_load.get('agenda', [])
        if isinstance(agenda, list):
            self._agenda._entries = {}
            for e in agenda:
                if not isinstance(e, dict) or not e.get('task_id'):
                    continue
                entry = AgendaEntry(
                    task_id=str(e.get('task_id')),
                    description=str(e.get('description', '')),
                    priority=float(e.get('priority', 0.5)),
                    created_at=float(e.get('created_at', time.time())),
                    scheduled_for=e.get('scheduled_for'),
                    status=str(e.get('status', 'pending')),
                    assigned_to=e.get('assigned_to'),
                    depends_on=list(e.get('depends_on', [])),
                    metadata=dict(e.get('metadata', {})),
                )
                self._agenda._entries[entry.task_id] = entry

        experiments = snapshot_to_load.get('approved_experiments', snapshot_to_load.get('experiments', []))
        if isinstance(experiments, list):
            self._experiments._experiments = {}
            for exp in experiments:
                if not isinstance(exp, dict) or not exp.get('exp_id'):
                    continue
                entry = ExperimentEntry(
                    exp_id=str(exp.get('exp_id')),
                    hypothesis=str(exp.get('hypothesis', '')),
                    created_at=float(exp.get('created_at', time.time())),
                    status=str(exp.get('status', 'running')),
                    results=list(exp.get('results', [])),
                    conclusion=exp.get('conclusion'),
                    metadata=dict(exp.get('metadata', {})),
                )
                self._experiments._experiments[entry.exp_id] = entry

        self._autobiographical_summary = dict(snapshot_to_load.get('autobiographical_summary', {})) if isinstance(snapshot_to_load.get('autobiographical_summary', {}), dict) else {}
        self._semantic_memory = dict(snapshot_to_load.get('semantic_memory', {})) if isinstance(snapshot_to_load.get('semantic_memory', {}), dict) else {}
        self._procedural_memory = dict(snapshot_to_load.get('procedural_memory', {})) if isinstance(snapshot_to_load.get('procedural_memory', {}), dict) else {}
        self._transfer_memory = dict(snapshot_to_load.get('transfer_memory', {})) if isinstance(snapshot_to_load.get('transfer_memory', {}), dict) else {}

        return True

    def get_guard_log(self) -> List[Dict[str, Any]]:
        """Return guard validation events for resume/persist entrypoints."""
        return list(self._guard_log)

    def get_last_guard_verdict(self) -> Optional[Dict[str, Any]]:
        """Return most recent guard verdict as plain dict for callers."""
        if self._last_guard_verdict is None:
            return None
        return self._last_guard_verdict.to_dict()

    def set_illegal_inheritance_policy(self, policy: str) -> None:
        """Set policy for illegal continuity inheritance handling."""
        self._illegal_inheritance_policy = IllegalInheritancePolicy(policy)
    
    def __repr__(self):
        return (f"ContinuityManager("
                f"id={self._agent_id}, "
                f"goals={len(self._goals._goals)}, "
                f"tasks={len(self._agenda._entries)}, "
                f"experiments={len(self._experiments._experiments)})")

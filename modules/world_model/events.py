"""
modules/world_model/events.py

Canonical event schema for World Model Omega consumption.

Event-driven integration: CoreMainLoop emits events -> World Model Omega derives candidates.
No new control authority. No direct formal writes.

Event types:
- observation_received: per-tick observation from world
- action_selected: action chosen after governance
- action_executed: action result from world
- reward_observed: reward signal received
- object_created: object written to store via committer
- object_consumed: object used in retrieval/surface
- hypothesis_created: new hypothesis formed
- hypothesis_updated: hypothesis confirmed/falsified/competing
- test_generated: discriminating test created
- test_executed: test result received
- commit_written: Step10 formal write completed
- teacher_asset_absorbed: teacher asset absorbed into system
- post_exit_reuse_observed: teacher-absorbed asset reused after teacher exit
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


class EventType(Enum):
    """Canonical event types for world-model consumption."""
    OBSERVATION_RECEIVED = "observation_received"
    ACTION_SELECTED = "action_selected"
    ACTION_EXECUTED = "action_executed"
    REWARD_OBSERVED = "reward_observed"
    OBJECT_CREATED = "object_created"
    OBJECT_CONSUMED = "object_consumed"
    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    TEST_GENERATED = "test_generated"
    TEST_EXECUTED = "test_executed"
    COMMIT_WRITTEN = "commit_written"
    TEACHER_ASSET_ABSORBED = "teacher_asset_absorbed"
    POST_EXIT_REUSE_OBSERVED = "post_exit_reuse_observed"
    ANOMALY_DETECTED = "anomaly_detected"  # T2-P2: For invalid candidate filtering
    RECOVERY_EXECUTED = "recovery_executed"
    RECOVERY_OUTCOME_OBSERVED = "recovery_outcome_observed"
    MECHANISM_EVIDENCE_ADDED = "mechanism_evidence_added"


@dataclass
class WorldModelEvent:
    """
    Canonical event for world-model consumption.
    
    CoreMainLoop emits these at defined points.
    World Model Omega reads these to update belief state.
    
    Rules:
    - Events are immutable records
    - Events do NOT directly mutate formal truth
    - All timestamps are monotonically increasing per episode
    """
    event_type: EventType
    episode: int
    tick: int
    timestamp: float = field(default_factory=time.time)
    
    # Event-specific data (typed dict, not rich objects)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance (for audit and trace)
    source_module: str = "core.main_loop"
    source_stage: Optional[str] = None  # e.g., "retrieval", "action_generation", "testing"
    
    def to_dict(self) -> dict:
        """Serialize to dict for logging and world-model consumption."""
        return {
            'event_type': self.event_type.value,
            'episode': self.episode,
            'tick': self.tick,
            'timestamp': self.timestamp,
            'data': self.data,
            'source_module': self.source_module,
            'source_stage': self.source_stage,
        }


class WorldModelEventBus:
    """
    Event bus for world-model consumption.
    
    CoreMainLoop holds an instance and emits events.
    World Model Omega subscribes to receive events.
    
    This is NOT a message bus with routing logic.
    It is a simple append-only log with subscriber callbacks.
    """
    
    def __init__(self):
        self._events: List[WorldModelEvent] = []
        self._subscribers: List[callable] = []
        self._episode_boundaries: List[int] = []  # indices where episodes change
    
    def emit(self, event: WorldModelEvent) -> None:
        """Emit an event to the bus."""
        self._events.append(event)
        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                # Subscribers should not crash the main loop
                pass
    
    def subscribe(self, callback: callable) -> None:
        """Subscribe to receive events."""
        self._subscribers.append(callback)
    
    def get_events(
        self,
        episode: Optional[int] = None,
        tick_range: Optional[tuple] = None,
        event_types: Optional[List[EventType]] = None,
    ) -> List[WorldModelEvent]:
        """
        Query events by filter.
        
        Args:
            episode: filter by episode
            tick_range: (start, end) tick range
            event_types: list of event types to include
        """
        results = self._events
        
        if episode is not None:
            results = [e for e in results if e.episode == episode]
        
        if tick_range is not None:
            start, end = tick_range
            results = [e for e in results if start <= e.tick <= end]
        
        if event_types is not None:
            results = [e for e in results if e.event_type in event_types]
        
        return results
    
    def get_recent(self, n: int = 10) -> List[WorldModelEvent]:
        """Get the n most recent events."""
        return self._events[-n:]
    
    def episode_count(self) -> int:
        """Number of episodes with events."""
        return len(set(e.episode for e in self._events))
    
    def event_count(self) -> int:
        """Total event count."""
        return len(self._events)
    
    def reset(self) -> None:
        """Reset all events (for testing or episode boundary)."""
        self._events.clear()
        self._subscribers.clear()
        self._episode_boundaries.clear()

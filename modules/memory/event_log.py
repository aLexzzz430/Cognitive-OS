"""
modules/memory/event_log.py

Phase 1: Persistent Raw Event Log

Append-only log of canonical raw events from the main loop.
Used for:
- durable history (replay, audit)
- consolidation input
- reference for formal memory decisions

Rules:
- append-only, never modify
- events are immutable history records
- NOT formal memory until they pass validator + committer

Storage:
- JSONL format
- default path: runtime/logs/event_log.jsonl
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime_paths import default_event_log_path


class EventLog:
    """
    Append-only event log for raw memory.
    
    Stores canonical events from main loop in JSONL format.
    Events are immutable - append only.
    
    Public API:
    - append(event) -> event_id
    - append_many(events) -> list[event_id]
    - get_recent(n) -> list[event]
    - get_episode(episode) -> list[event]
    - find(event_type, episode) -> list[event]
    - replay_episode(episode) -> list[event]
    """
    
    def __init__(self, path: Optional[str] = None):
        self._path = default_event_log_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: List[Dict] = []  # In-memory buffer
        self._buffer_size = 10  # Flush after N events
        self._event_count = 0  # Total events logged
        self._next_event_seq = 0
        self._restore_counters()

    def _restore_counters(self) -> None:
        """Restore persisted counters so event ids stay monotonic across restarts."""
        if not self._path.exists():
            return

        event_count = 0
        next_seq = 0
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    event_count += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_id = str(record.get('event_id', '') or '')
                    if not event_id.startswith('ev_'):
                        continue
                    parts = event_id.split('_', 2)
                    if len(parts) < 3:
                        continue
                    try:
                        next_seq = max(next_seq, int(parts[1]) + 1)
                    except ValueError:
                        continue
        except OSError:
            return

        self._event_count = event_count
        self._next_event_seq = max(next_seq, event_count)
    
    def append(self, event: Dict[str, Any]) -> str:
        """
        Append a single event to the log.
        
        Args:
            event: Event dict with required fields:
                - event_type: str
                - episode: int
                - tick: int
                - data: dict
                - source_module: str (optional, default 'core')
                - source_stage: str (optional)
        
        Returns:
            event_id: stable identifier for this event
        """
        # Assign stable event_id
        event_id = f"ev_{self._next_event_seq:06d}_{uuid.uuid4().hex[:8]}"
        
        # Create canonical record
        record = {
            'event_id': event_id,
            'event_type': event.get('event_type', 'unknown'),
            'episode': event.get('episode', 0),
            'tick': event.get('tick', 0),
            'timestamp': time.time(),
            'data': event.get('data', {}),
            'source_module': event.get('source_module', 'core'),
            'source_stage': event.get('source_stage', ''),
        }
        
        # Buffer for batch write
        self._buffer.append(record)
        self._event_count += 1
        self._next_event_seq += 1
        
        # Flush if buffer full
        if len(self._buffer) >= self._buffer_size:
            self._flush()
        
        return event_id
    
    def append_many(self, events: List[Dict[str, Any]]) -> List[str]:
        """
        Append multiple events to the log.
        
        Args:
            events: List of event dicts
        
        Returns:
            list of event_ids
        """
        event_ids = []
        for event in events:
            event_id = self.append(event)
            event_ids.append(event_id)
        return event_ids
    
    def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        with open(self._path, 'a', encoding='utf-8') as f:
            for record in self._buffer:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        self._buffer.clear()
    
    def flush(self) -> None:
        """Public flush method."""
        self._flush()
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """
        Get the n most recent events.
        
        Uses in-memory buffer + disk.
        """
        self._flush()  # Ensure buffer is written
        
        events = list(self._buffer)  # Remaining in buffer
        
        # Read from disk
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            disk_events = [json.loads(line) for line in lines]
            events = disk_events[-n:] + events
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return events[-n:]
    
    def get_episode(self, episode: int) -> List[Dict]:
        """
        Get all events for a specific episode.
        
        Args:
            episode: Episode number
        
        Returns:
            List of events for that episode
        """
        self._flush()
        
        events = list(self._buffer)
        results = [e for e in events if e.get('episode') == episode]
        
        # Read from disk
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                for line in f:
                    event = json.loads(line)
                    if event.get('episode') == episode:
                        results.append(event)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Sort by tick
        results.sort(key=lambda e: e.get('tick', 0))
        return results
    
    def find(
        self,
        event_type: Optional[str] = None,
        episode: Optional[int] = None,
        tick_range: Optional[tuple] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Find events matching criteria.
        
        Args:
            event_type: Filter by event_type (exact match)
            episode: Filter by episode number
            tick_range: (start, end) tick range
            limit: Maximum events to return
        
        Returns:
            List of matching events
        """
        self._flush()
        
        results = list(self._buffer)
        
        # Read from disk
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                for line in f:
                    event = json.loads(line)
                    results.append(event)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Apply filters
        if event_type is not None:
            results = [e for e in results if e.get('event_type') == event_type]
        
        if episode is not None:
            results = [e for e in results if e.get('episode') == episode]
        
        if tick_range is not None:
            start, end = tick_range
            results = [e for e in results if start <= e.get('tick', 0) <= end]
        
        # Sort by episode, tick and limit
        results.sort(key=lambda e: (e.get('episode', 0), e.get('tick', 0)))
        return results[-limit:]
    
    def replay_episode(self, episode: int) -> List[Dict]:
        """
        Get all events for episode replay.
        
        Returns events in chronological order.
        """
        return self.get_episode(episode)
    
    def event_count(self) -> int:
        """Total number of events in log."""
        self._flush()
        return self._event_count
    
    def clear(self) -> None:
        """
        Clear the event log (for testing only).
        
        WARNING: This is destructive!
        """
        self._buffer.clear()
        if self._path.exists():
            self._path.unlink()
        self._event_count = 0
        self._next_event_seq = 0


class EventLogBuilder:
    """
    Helper to build canonical events for the log.
    
    Use this to create events with correct structure.
    """
    
    @staticmethod
    def observation(episode: int, tick: int, data: Dict, source_stage: str = 'retrieval') -> Dict:
        """Build observation_received event."""
        return {
            'event_type': 'observation_received',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def action_selected(episode: int, tick: int, data: Dict, source_stage: str = 'action_generation') -> Dict:
        """Build action_selected event."""
        return {
            'event_type': 'action_selected',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def action_executed(episode: int, tick: int, data: Dict, source_stage: str = 'execution') -> Dict:
        """Build action_executed event."""
        return {
            'event_type': 'action_executed',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def reward_observed(episode: int, tick: int, data: Dict, source_stage: str = 'execution') -> Dict:
        """Build reward_observed event."""
        return {
            'event_type': 'reward_observed',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def commit_written(episode: int, tick: int, data: Dict, source_stage: str = 'evidence_commit') -> Dict:
        """Build commit_written event."""
        return {
            'event_type': 'commit_written',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def hypothesis_created(episode: int, tick: int, data: Dict, source_stage: str = 'hypothesis_propagation') -> Dict:
        """Build hypothesis_created event."""
        return {
            'event_type': 'hypothesis_created',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def hypothesis_updated(episode: int, tick: int, data: Dict, source_stage: str = 'testing') -> Dict:
        """Build hypothesis_updated event."""
        return {
            'event_type': 'hypothesis_updated',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'core',
            'source_stage': source_stage,
        }
    
    @staticmethod
    def teacher_interaction(episode: int, tick: int, data: Dict, source_stage: str = 'teacher') -> Dict:
        """Build teacher_interaction event."""
        return {
            'event_type': 'teacher_interaction',
            'episode': episode,
            'tick': tick,
            'data': data,
            'source_module': 'teacher',
            'source_stage': source_stage,
        }

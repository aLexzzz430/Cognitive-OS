"""Trace runtime resolver used by ``core.main_loop``."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List


def resolve_trace_runtime():
    """Resolve optional trace runtime with a local no-op fallback."""
    try:
        trace_module = importlib.import_module('trace')
        return (
            trace_module.CausalTraceLogger,
            trace_module.EventTimeline,
            trace_module.DecisionSource,
        )
    except Exception:
        class _FallbackDecisionSource:
            BASE_GENERATION = 'base_generation'
            SKILL_REWRITE = 'skill_rewrite'
            LLM_REWRITE = 'llm_rewrite'
            ARM_EVALUATION = 'arm_evaluation'
            RECOVERY = 'recovery'
            PLANNER = 'planner'
            WAIT_FALLBACK = 'wait_fallback'
            RETRIEVAL = 'retrieval'
            SELF_MODEL = 'self_model'
            HISTORY_REUSE = 'history_reuse'

        class _FallbackTrace:
            def __init__(self, episode: int, tick: int) -> None:
                self.episode = episode
                self.tick = tick
                self.candidates = []
                self.execution = None

        class _FallbackCausalTraceLogger:
            def __init__(self) -> None:
                self._traces: List[Any] = []

            def new_trace(self, episode: int, tick: int):
                trace = _FallbackTrace(episode=episode, tick=tick)
                self._traces.append(trace)
                return trace

            def set_observation_signature(self, trace, signature):
                trace.observation_signature = signature

            def add_candidate(self, *, trace, candidate_id, source, function_name, proposed_action):
                trace.candidates.append({
                    'candidate_id': candidate_id,
                    'source': source,
                    'function_name': function_name,
                    'proposed_action': dict(proposed_action or {}),
                    'selected': False,
                })

            def select_candidate(self, trace, candidate_id):
                trace.selected_candidate_id = candidate_id
                for candidate in trace.candidates:
                    candidate['selected'] = candidate.get('candidate_id') == candidate_id

            def set_governance(self, *, trace, decision, reason='', risk_assessment=None, opportunity_assessment=None, leak_gate_mode=None):
                trace.governance = {
                    'decision': decision,
                    'reason': reason,
                    'risk_assessment': risk_assessment,
                    'opportunity_assessment': opportunity_assessment,
                    'leak_gate_mode': leak_gate_mode,
                }

            def set_execution(self, *, trace, success, terminal, reward, error_type=None):
                trace.execution = type('ExecRecord', (), {
                    'success': bool(success),
                    'terminal': bool(terminal),
                    'reward': float(reward or 0.0),
                    'error_type': error_type,
                })()

            def set_final_action(self, trace, action):
                trace.final_action = dict(action or {})

            def set_env_action(self, trace, action):
                trace.env_action = dict(action or {})

            def set_context(self, *, trace, continuity_snapshot=None, retrieval_bundle_summary=None):
                trace.continuity_snapshot = dict(continuity_snapshot or {})
                trace.retrieval_bundle_summary = dict(retrieval_bundle_summary or {})

            def set_plan_context(self, trace, plan_summary, step_intent):
                trace.plan_summary = dict(plan_summary or {})
                trace.step_intent = step_intent

            def get_recent_traces(self, n: int = 10):
                return self._traces[-max(0, int(n or 0)):]

        class _FallbackEventTimeline:
            def __init__(self) -> None:
                self._events: List[Dict[str, Any]] = []

            def emit_stage_enter(self, episode, tick, stage):
                self._events.append({'type': 'stage_enter', 'episode': episode, 'tick': tick, 'stage': stage})

            def emit_stage_exit(self, episode, tick, stage, payload=None):
                self._events.append({'type': 'stage_exit', 'episode': episode, 'tick': tick, 'stage': stage, 'payload': dict(payload or {})})

            def emit_action_executed(self, *, episode, tick, function_name, success, terminal, reward):
                self._events.append({'type': 'action_executed', 'episode': episode, 'tick': tick, 'function_name': function_name, 'success': bool(success), 'terminal': bool(terminal), 'reward': float(reward or 0.0)})

            def emit_commit(self, episode, tick, committed_ids):
                self._events.append({'type': 'commit', 'episode': episode, 'tick': tick, 'committed_ids': list(committed_ids or [])})

        return _FallbackCausalTraceLogger, _FallbackEventTimeline, _FallbackDecisionSource

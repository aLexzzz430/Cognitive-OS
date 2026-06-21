from typing import List, Dict, Any

# ... (truncated) ...

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

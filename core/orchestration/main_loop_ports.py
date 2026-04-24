from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.orchestration.context_stage import ContextProvider
from core.orchestration.governance_runtime import (
    CounterfactualPort,
    GovernanceLogPort,
    OrganCapabilityPort,
    ReliabilityPort,
)
from core.orchestration.governance_state import GovernanceState, capability_for_organ


class MainLoopContextProvider(ContextProvider):
    """Adapter mapping CoreMainLoop internals to ContextProvider contract."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def beliefs(self) -> Dict[str, Any]:
        return {
            'active': list(self.loop._belief_ledger.get_active_beliefs()),
            'established_count': len(self.loop._belief_ledger.get_established_beliefs()),
        }

    def episode_trace(self) -> List[Dict[str, Any]]:
        return list(self.loop._episode_trace)

    def plan_snapshot(self) -> Dict[str, Any]:
        return {
            'plan_summary': self.loop._plan_state.get_plan_summary(),
            'current_task': str(self.loop._plan_state.get_intent_for_step() or ''),
        }

    def meta_control_snapshot(
        self,
        episode: int,
        tick: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snapshot = self.loop._meta_control.get_snapshot(episode, tick, context=context)
        return {
            'snapshot_id': str(getattr(snapshot, 'snapshot_id', '') or ''),
            'inputs_hash': str(getattr(snapshot, 'inputs_hash', '') or ''),
            'meta_control_snapshot_id': str(getattr(snapshot, 'snapshot_id', '') or ''),
            'meta_control_inputs_hash': str(getattr(snapshot, 'inputs_hash', '') or ''),
            'planner_bias': float(getattr(snapshot, 'planner_bias', 0.0) or 0.0),
            'retrieval_aggressiveness': float(getattr(snapshot, 'retrieval_aggressiveness', 0.0) or 0.0),
            'retrieval_pressure': float(getattr(snapshot, 'retrieval_pressure', 0.0) or 0.0),
            'probe_bias': float(getattr(snapshot, 'probe_bias', 0.0) or 0.0),
            'verification_bias': float(getattr(snapshot, 'verification_bias', 0.0) or 0.0),
            'risk_tolerance': float(getattr(snapshot, 'risk_tolerance', 0.0) or 0.0),
            'recovery_bias': float(getattr(snapshot, 'recovery_bias', 0.0) or 0.0),
            'stability_bias': float(getattr(snapshot, 'stability_bias', 0.0) or 0.0),
            'strategy_mode': str(getattr(snapshot, 'strategy_mode', '') or ''),
            'policy_profile': snapshot.to_policy_profile(),
            'representation_profile': snapshot.to_representation_profile(),
        }

    def hypotheses_snapshot(self, limit: int = 3) -> List[Dict[str, Any]]:
        active_hypotheses: List[Dict[str, Any]] = []
        for hyp in self.loop._hypotheses.get_active()[:limit] if hasattr(self.loop, '_hypotheses') else []:
            if hasattr(hyp, '__dict__'):
                active_hypotheses.append({
                    'hypothesis_id': str(getattr(hyp, 'hypothesis_id', '') or ''),
                    'description': str(getattr(hyp, 'description', '') or ''),
                    'confidence': float(getattr(hyp, 'confidence', 0.0) or 0.0),
                    'status': str(getattr(getattr(hyp, 'status', ''), 'value', getattr(hyp, 'status', '')) or ''),
                })
        return active_hypotheses

    def self_model_summary(self) -> Dict[str, Any]:
        return self.loop._build_self_model_prediction_summary()

    def hidden_state_summary(self) -> Dict[str, Any]:
        if not hasattr(self.loop, '_hidden_state_tracker') or self.loop._hidden_state_tracker is None:
            return {}
        return self.loop._hidden_state_tracker.summary()

    def learning_policy_snapshot(self) -> Dict[str, Any]:
        snapshot = getattr(self.loop, '_learning_policy_snapshot', {})
        return dict(snapshot) if isinstance(snapshot, dict) else {}

    def workspace_state(self) -> Dict[str, Any]:
        state_mgr = getattr(self.loop, '_state_mgr', None)
        if state_mgr is None or not hasattr(state_mgr, 'get_state'):
            return {}
        state = state_mgr.get_state()
        if not isinstance(state, dict):
            return {}
        snapshot: Dict[str, Any] = {}
        for key in ('object_workspace', 'goal_stack', 'self_summary', 'governance_context'):
            value = state.get(key)
            if isinstance(value, dict):
                snapshot[key] = dict(value)
        object_workspace = dict(snapshot.get('object_workspace', {}) or {})
        goal_prior_rows = []
        current_episode = int(getattr(self.loop, '_episode', 0) or 0)
        for row in list(getattr(self.loop, '_llm_initial_goal_hypothesis_candidates', []) or []):
            if not isinstance(row, dict):
                continue
            source_episode = int(row.get('source_episode', current_episode) or current_episode)
            if source_episode != current_episode:
                continue
            goal_prior_rows.append(dict(row))
        transient_analyst_rows = []
        current_tick = int(getattr(self.loop, '_tick', 0) or 0)
        for row in list(getattr(self.loop, '_llm_analyst_hypothesis_candidates', []) or []):
            if not isinstance(row, dict):
                continue
            source_tick = int(row.get('source_tick', current_tick) or current_tick)
            if current_tick - source_tick > 1:
                continue
            transient_analyst_rows.append(dict(row))
        analyst_rows = []
        seen_ids = set()
        for row in goal_prior_rows + transient_analyst_rows:
            object_id = str(row.get('object_id', '') or '')
            dedupe_key = object_id or str(row.get('summary', '') or '')
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            analyst_rows.append(dict(row))
        if analyst_rows:
            object_workspace['analyst_hypothesis_candidates'] = analyst_rows
        elif 'analyst_hypothesis_candidates' in object_workspace:
            object_workspace['analyst_hypothesis_candidates'] = []
        proposal_rows = []
        for row in list(getattr(self.loop, '_llm_world_model_proposal_candidates', []) or []):
            if not isinstance(row, dict):
                continue
            source_episode = int(row.get('source_episode', current_episode) or current_episode)
            if source_episode != current_episode:
                continue
            proposal_rows.append(dict(row))
        validation_rows = []
        for row in list(getattr(self.loop, '_llm_world_model_validation_feedback', []) or []):
            if not isinstance(row, dict):
                continue
            validation_rows.append(dict(row))
        if proposal_rows:
            object_workspace['llm_proposal_candidates'] = proposal_rows[-16:]
        elif 'llm_proposal_candidates' in object_workspace:
            object_workspace['llm_proposal_candidates'] = []
        if validation_rows:
            object_workspace['llm_proposal_validation_feedback'] = validation_rows[-24:]
        elif 'llm_proposal_validation_feedback' in object_workspace:
            object_workspace['llm_proposal_validation_feedback'] = []
        latest_snapshot = getattr(self.loop, '_llm_world_model_snapshot', {})
        if isinstance(latest_snapshot, dict) and latest_snapshot:
            object_workspace['llm_world_model_snapshot'] = dict(latest_snapshot)
        elif 'llm_world_model_snapshot' in object_workspace:
            object_workspace['llm_world_model_snapshot'] = {}
        if object_workspace:
            snapshot['object_workspace'] = object_workspace
        return snapshot

    @staticmethod
    def _representation_context_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, float, str]:
        memory_layer = str(row.get('memory_layer', '') or '').strip().lower()
        memory_type = str(row.get('memory_type', '') or '').strip().lower()
        content = row.get('content', {})
        content = content if isinstance(content, dict) else {}
        content_type = str(content.get('type', '') or '').strip().lower()
        is_mechanism = (
            memory_layer == 'mechanism'
            or memory_type == 'mechanism_summary'
            or content_type == 'mechanism_summary'
        )
        return (
            1 if is_mechanism else 0,
            float(row.get('surface_priority', 0.0) or 0.0),
            float(row.get('confidence', 0.0) or 0.0),
            float(row.get('consumption_count', 0.0) or 0.0),
            str(row.get('updated_at', '') or row.get('created_at', '') or ''),
        )

    def cognitive_object_records(self, object_type: str, limit: int = 8) -> List[Dict[str, Any]]:
        store = getattr(self.loop, '_shared_store', None)
        if store is None:
            return []
        if hasattr(store, 'get_by_object_type'):
            fetched = store.get_by_object_type(object_type)
        elif hasattr(store, 'retrieve'):
            fetched = store.retrieve(object_type=object_type, limit=limit)
        else:
            fetched = []
        rows = [dict(row) for row in list(fetched or []) if isinstance(row, dict)]
        if str(object_type or '').strip() == 'representation':
            rows.sort(key=self._representation_context_sort_key, reverse=True)
        trimmed: List[Dict[str, Any]] = []
        for row in rows[:max(0, int(limit))]:
            if isinstance(row, dict):
                trimmed.append(dict(row))
        return trimmed

    def extraction_function_name(self, action: Any, default: str = '') -> str:
        return self.loop._extract_action_function_name(action, default=default)

    def retrieval_should_query(self) -> bool:
        return bool((self.loop._last_retrieval_decision or {}).get('should_query', False))


class MainLoopGovernancePorts(ReliabilityPort, CounterfactualPort, GovernanceLogPort, OrganCapabilityPort):
    """Temporary adapter bridging governance runtime ports to existing loop objects."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def build_global_failure_strategy(self, *, short_term_pressure: float) -> Any:
        if not hasattr(self.loop, '_reliability_tracker'):
            return None
        return self.loop._reliability_tracker.build_global_failure_strategy(short_term_pressure=short_term_pressure)

    def simulate_action_difference(self, state_slice: Any, action_a: Dict[str, Any], action_b: Dict[str, Any], *, context: Dict[str, Any]) -> Any:
        return self.loop._counterfactual.simulate_action_difference(state_slice, action_a, action_b, context=context)

    def append_governance(self, entry: Dict[str, Any]) -> None:
        self.loop._governance_log.append(dict(entry))

    def append_candidate_viability(self, entry: Dict[str, Any]) -> None:
        self.loop._candidate_viability_log.append(dict(entry))

    def get_capability(self, organ: str, state: GovernanceState) -> str:
        return capability_for_organ(state, organ)

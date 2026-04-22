"""
self_model/facade.py

Unified facade that composes reliability, capability and high-level self state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from self_model.autobiographical_summary import build_autobiographical_summary
from self_model.capability_envelope import build_capability_envelope
from self_model.identity_ledger import DurableIdentityLedger
from self_model.state import SelfModelState


class SelfModelFacade:
    """Facade for assembling high-level self-model signals."""

    def __init__(self, reliability_tracker, capability_profile, state: Optional[SelfModelState] = None):
        self._reliability_tracker = reliability_tracker
        self._capability_profile = capability_profile
        self._state = state or SelfModelState()
        self._identity_ledger = DurableIdentityLedger()

    @property
    def state(self) -> SelfModelState:
        return self._state

    def refresh(
        self,
        *,
        resource_state=None,
        continuity_confidence: Optional[float] = None,
        value_commitments_summary: Optional[str] = None,
        identity_markers: Optional[Dict[str, Any]] = None,
        external_dependencies: Optional[List[str]] = None,
        continuity_snapshot: Optional[Dict[str, Any]] = None,
        teacher_present: Optional[bool] = None,
        autobiographical_summary: Optional[Dict[str, Any]] = None,
    ) -> SelfModelState:
        provenance = dict(self._state.provenance or {})
        continuity = dict(continuity_snapshot or {}) if isinstance(continuity_snapshot, dict) else {}
        self._state.capabilities_by_domain = self._collect_capabilities_by_domain()
        self._state.capabilities_by_condition = self._collect_capabilities_by_condition()
        self._state.known_failure_modes = self._collect_known_failure_modes()
        self._state.fragile_regions = self._collect_fragile_regions()
        self._state.recovered_regions = self._collect_recovered_regions()
        if external_dependencies is not None:
            values = [str(item) for item in list(external_dependencies) if str(item).strip()]
            self._state.external_dependencies = values or ['unknown']
            provenance['external_dependencies'] = 'learned' if values else 'default'
        elif not self._state.external_dependencies:
            self._state.external_dependencies = ['unknown']
            provenance['external_dependencies'] = 'default'
        if identity_markers is not None:
            markers = {
                str(key): value for key, value in dict(identity_markers).items()
                if str(key).strip() and value not in (None, '')
            }
            self._state.identity_markers = markers or {'agent_id': 'unknown', 'arm_mode': 'unknown'}
            provenance['identity_markers'] = 'learned' if markers else 'default'
        elif not self._state.identity_markers:
            self._state.identity_markers = {'agent_id': 'unknown', 'arm_mode': 'unknown'}
            provenance['identity_markers'] = 'default'
        if value_commitments_summary is not None:
            summary = str(value_commitments_summary).strip()
            self._state.value_commitments_summary = summary or 'unknown'
            provenance['value_commitments_summary'] = 'learned' if summary else 'default'
        elif not self._state.value_commitments_summary:
            self._state.value_commitments_summary = 'unknown'
            provenance['value_commitments_summary'] = 'default'
        self._state.continuity_confidence = self._resolve_continuity_confidence(
            continuity_confidence=continuity_confidence,
            resource_state=resource_state,
        )
        self._state.active_commitments = self._collect_active_commitments(continuity)
        self._state.long_horizon_agenda = self._collect_long_horizon_agenda(continuity)
        self._state.self_experiment_queue = self._collect_self_experiment_queue(continuity)
        self._state.autobiographical_summary = build_autobiographical_summary(
            continuity_snapshot=continuity,
            explicit_summary=autobiographical_summary,
            recent_failure_modes=self._state.known_failure_modes,
        )
        durable_identity = self._identity_ledger.build(
            identity_markers=self._state.identity_markers,
            continuity_snapshot=continuity,
            active_commitments=self._state.active_commitments,
            autobiographical_summary=self._state.autobiographical_summary,
            continuity_confidence=self._state.continuity_confidence,
        )
        self._state.durable_identity = durable_identity.to_dict()
        envelope = build_capability_envelope(
            capability_profile=self._capability_profile,
            reliability_tracker=self._reliability_tracker,
            continuity_snapshot=continuity,
            continuity_confidence=self._state.continuity_confidence,
            teacher_present=teacher_present,
        )
        self._state.capability_envelope = envelope.to_dict()
        self._state.known_blind_spots = list(self._state.capability_envelope.get('known_blind_spots', []))
        self._state.teacher_dependence_estimate = float(self._state.capability_envelope.get('teacher_dependence_estimate', 0.5) or 0.5)
        self._state.transfer_readiness = float(self._state.capability_envelope.get('transfer_readiness', 0.0) or 0.0)
        if resource_state is not None and hasattr(resource_state, 'update_exploration_ratio'):
            resource_state.update_exploration_ratio(
                float(self._state.capability_envelope.get('exploration_ratio_target', 0.5) or 0.5)
            )
        provenance['continuity_confidence'] = 'learned' if continuity_confidence is not None else 'inferred'
        provenance['durable_identity'] = 'continuity_ledger' if continuity else 'derived'
        provenance['capability_envelope'] = 'self_model_modulation'
        provenance['autobiographical_summary'] = 'continuity_ledger' if self._state.autobiographical_summary else 'default'
        self._state.provenance = provenance
        return self._state

    def build_prediction_summary(self, *, resource_state=None, include_high_level_state: bool = True) -> Dict[str, Any]:
        state = self._state
        if not (
            state.capability_envelope
            or state.identity_markers
            or state.capabilities_by_domain
            or state.provenance
        ):
            state = self.refresh(resource_state=resource_state)
        elif resource_state is not None and hasattr(resource_state, 'update_exploration_ratio'):
            resource_state.update_exploration_ratio(
                float(state.capability_envelope.get('exploration_ratio_target', 0.5) or 0.5)
            )
        reliability_by_function = {}
        if hasattr(self._reliability_tracker, 'get_reliability_by_action_type'):
            reliability_by_function = dict(self._reliability_tracker.get_reliability_by_action_type())
        recovery_availability = 0.5
        if hasattr(self._reliability_tracker, 'get_overall_recovery_success_rate'):
            recovery_availability = float(self._reliability_tracker.get_overall_recovery_success_rate())
        resource_tightness = 'normal'
        budget_tight = False
        if resource_state is not None and hasattr(resource_state, 'budget_band'):
            resource_tightness = str(resource_state.budget_band() or 'normal')
            budget_tight = bool(getattr(resource_state, 'is_tight_budget', lambda: False)())
        global_reliability = 0.5
        if reliability_by_function:
            global_reliability = sum(float(v) for v in reliability_by_function.values()) / len(reliability_by_function)
        if include_high_level_state:
            self_model_state = state.to_dict()
        else:
            self_model_state = {
                'capabilities_by_domain': dict(state.capabilities_by_domain or {}),
                'capabilities_by_condition': dict(state.capabilities_by_condition or {}),
            }
        capability_envelope = dict(state.capability_envelope or {})
        planner_control_profile = {
            'strategy_mode': str(capability_envelope.get('strategy_mode_hint', 'balanced') or 'balanced'),
            'branch_budget_delta': int(capability_envelope.get('branch_budget_delta', 0) or 0),
            'verification_budget_delta': int(capability_envelope.get('verification_budget_delta', 0) or 0),
            'search_depth_bias': int(capability_envelope.get('search_depth_bias', 0) or 0),
            'fallback_bias': str(capability_envelope.get('fallback_bias', 'balanced') or 'balanced'),
            'teacher_off_escalation': bool(capability_envelope.get('teacher_off_escalation', False)),
        }
        return {
            'self_model_state': self_model_state,
            'reliability_subscores': {
                'reliability_by_function': reliability_by_function,
                'global_reliability': max(0.0, min(1.0, float(global_reliability))),
                'recovery_availability': max(0.0, min(1.0, float(recovery_availability))),
            },
            # Backward-compatible keys consumed by existing decision stack.
            'reliability_by_function': reliability_by_function,
            'recent_failure_modes': list(state.known_failure_modes),
            'resource_tightness': resource_tightness,
            'budget_tight': budget_tight,
            'high_level_state_included': bool(include_high_level_state),
            'global_reliability': max(0.0, min(1.0, float(global_reliability))),
            'recovery_availability': max(0.0, min(1.0, float(recovery_availability))),
            'capability_confidence': max(0.0, min(1.0, 0.4 + 0.6 * len(state.capabilities_by_domain) / 5.0)),
            'continuity_confidence': state.continuity_confidence,
            'teacher_dependence_estimate': max(0.0, min(1.0, float(state.teacher_dependence_estimate))),
            'transfer_readiness': max(0.0, min(1.0, float(state.transfer_readiness))),
            'budget_multiplier': max(0.0, min(1.0, float(capability_envelope.get('budget_multiplier', 1.0) or 1.0))),
            'exploration_ratio_target': max(0.0, min(1.0, float(capability_envelope.get('exploration_ratio_target', 0.5) or 0.5))),
            'capability_envelope': capability_envelope,
            'planner_control_profile': planner_control_profile,
            'autobiographical_summary': dict(state.autobiographical_summary or {}),
            'durable_identity': dict(state.durable_identity or {}),
        }

    @staticmethod
    def _collect_active_commitments(continuity_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(continuity_snapshot, dict):
            return []
        commitments = continuity_snapshot.get('active_commitments', [])
        if isinstance(commitments, list) and commitments:
            return [dict(item) for item in commitments if isinstance(item, dict)][:6]
        top_goal = continuity_snapshot.get('top_goal')
        if hasattr(top_goal, 'description'):
            return [{
                'commitment': str(getattr(top_goal, 'description', '') or ''),
                'source': str(getattr(top_goal, 'source', 'continuity') or 'continuity'),
                'goal_id': str(getattr(top_goal, 'goal_id', '') or ''),
            }]
        return []

    @staticmethod
    def _collect_long_horizon_agenda(continuity_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(continuity_snapshot, dict):
            return []
        agenda = continuity_snapshot.get('long_horizon_agenda', [])
        if isinstance(agenda, list) and agenda:
            return [dict(item) for item in agenda if isinstance(item, dict)][:6]
        next_task = continuity_snapshot.get('next_task')
        if hasattr(next_task, 'description'):
            return [{
                'task_id': str(getattr(next_task, 'task_id', '') or ''),
                'goal': str(getattr(next_task, 'description', '') or ''),
                'priority': float(getattr(next_task, 'priority', 0.5) or 0.5),
            }]
        return []

    @staticmethod
    def _collect_self_experiment_queue(continuity_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(continuity_snapshot, dict):
            return []
        queue = continuity_snapshot.get('approved_experiments', [])
        if isinstance(queue, list):
            return [dict(item) for item in queue if isinstance(item, dict)][:6]
        return []

    def _collect_known_failure_modes(self) -> List[Dict[str, Any]]:
        if hasattr(self._reliability_tracker, 'get_recent_failure_profile'):
            return list(self._reliability_tracker.get_recent_failure_profile(limit=8))
        return []

    def _collect_capabilities_by_domain(self) -> Dict[str, Dict[str, float]]:
        contextual = getattr(self._capability_profile, 'contextual_capabilities', {}) or {}
        by_domain: Dict[str, Dict[str, float]] = {}
        for fn_name, contexts in contextual.items():
            if not isinstance(contexts, dict):
                continue
            for stat in contexts.values():
                if not isinstance(stat, dict):
                    continue
                domain = str(stat.get('task_family', 'unknown') or 'unknown')
                total = float(stat.get('total_calls', 0) or 0)
                success = float(stat.get('success_count', 0) or 0)
                if total <= 0:
                    continue
                fn_bucket = by_domain.setdefault(domain, {})
                fn_bucket[fn_name] = max(fn_bucket.get(fn_name, 0.0), success / total)
        return by_domain

    def _collect_capabilities_by_condition(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        contextual = getattr(self._capability_profile, 'contextual_capabilities', {}) or {}
        by_condition: Dict[str, Dict[str, Dict[str, float]]] = {}
        for fn_name, contexts in contextual.items():
            if not isinstance(contexts, dict):
                continue
            fn_bucket: Dict[str, Dict[str, float]] = {}
            for ctx_key, stat in contexts.items():
                if not isinstance(stat, dict):
                    continue
                total = float(stat.get('total_calls', 0) or 0)
                success = float(stat.get('success_count', 0) or 0)
                fn_bucket[str(ctx_key)] = {
                    'success_rate': (success / total) if total > 0 else 0.0,
                    'confidence': float(stat.get('confidence', 0.0) or 0.0),
                    'total_calls': total,
                }
            if fn_bucket:
                by_condition[fn_name] = fn_bucket
        return by_condition

    def _collect_fragile_regions(self) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for fn_name, contexts in (getattr(self._capability_profile, 'contextual_capabilities', {}) or {}).items():
            if not isinstance(contexts, dict):
                continue
            for ctx_key, stat in contexts.items():
                if not isinstance(stat, dict):
                    continue
                total = float(stat.get('total_calls', 0) or 0)
                if total < 2:
                    continue
                success_rate = float(stat.get('success_count', 0) or 0) / total
                if success_rate < 0.4:
                    regions.append({'function_name': fn_name, 'context': str(ctx_key), 'success_rate': success_rate})
        return regions[:8]

    def _collect_recovered_regions(self) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for fn_name, contexts in (getattr(self._capability_profile, 'contextual_capabilities', {}) or {}).items():
            if not isinstance(contexts, dict):
                continue
            for ctx_key, stat in contexts.items():
                if not isinstance(stat, dict):
                    continue
                total = float(stat.get('total_calls', 0) or 0)
                if total < 3:
                    continue
                success_rate = float(stat.get('success_count', 0) or 0) / total
                if success_rate >= 0.67 and float(stat.get('failure_count', 0) or 0) > 0:
                    regions.append({'function_name': fn_name, 'context': str(ctx_key), 'success_rate': success_rate})
        return regions[:8]

    def _resolve_continuity_confidence(self, *, continuity_confidence: Optional[float], resource_state=None) -> float:
        if continuity_confidence is not None:
            return max(0.0, min(1.0, float(continuity_confidence)))
        recovery = 0.5
        if hasattr(self._reliability_tracker, 'get_overall_recovery_success_rate'):
            recovery = float(self._reliability_tracker.get_overall_recovery_success_rate())
        budget_factor = 0.0
        if resource_state is not None and hasattr(resource_state, 'is_tight_budget'):
            budget_factor = -0.1 if bool(resource_state.is_tight_budget()) else 0.05
        return max(0.0, min(1.0, recovery * 0.7 + 0.3 + budget_factor))

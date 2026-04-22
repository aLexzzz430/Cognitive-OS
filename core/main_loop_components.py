from __future__ import annotations

"""Components extracted from core.main_loop to keep CoreMainLoop thin."""

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.governance.object_store import ObjectStore, GovernanceDecision, ACCEPT_NEW, MERGE_UPDATE_EXISTING
from modules.representations.store import get_warehouse, get_runtime_store


# =============================================================================
# Arm Mode Constants
# =============================================================================

ARM_MODE_FULL = 'full'
ARM_MODE_WRONG_BINDING = 'wrong_binding'
ARM_MODE_LOCAL_ONLY = 'local_only'
ARM_MODE_SHUFFLED = 'shuffled'
ARM_MODE_FRESH = 'fresh'
ARM_MODE_NO_TRANSFER = 'no_transfer'
ARM_MODE_NO_PREDICTION = 'no_prediction'
ARM_MODE_NO_PROCEDURE_LEARNING = 'no_procedure_learning'
CAPABILITY_ADVISORY = 'advisory'
CAPABILITY_CONSTRAINED_CONTROL = 'constrained_control'
CAPABILITY_PRIMARY_CONTROL = 'primary_control'
LOW_RISK_CONTROL_FUNCTIONS = {'wait', 'probe', 'rank_candidates', 'candidate_rerank', 'inspect'}
ORGAN_CAPABILITY_KEYS = ('world_model', 'planner', 'self_model', 'prediction')


# =============================================================================
# P0-1: Episodic Retrieval System
# =============================================================================

class RetrievalQuery:
    """Explicit query object for episodic retrieval."""
    def __init__(self, query_text: str, context: dict, tick: int, episode: int):
        self.query_text = query_text
        self.context = context
        self.tick = tick
        self.episode = episode
        self.constraints: List[str] = []
        self.exclude_ids: List[str] = []

    def add_constraint(self, c: str):
        self.constraints.append(c)

    def add_exclusion(self, obj_id: str):
        self.exclude_ids.append(obj_id)


class RetrievedCandidate:
    """A retrieved candidate from the object store."""
    def __init__(self, object_id: str, object: dict, relevance_score: float, rank: int):
        self.object_id = object_id
        self.object = object
        self.relevance_score = relevance_score
        self.rank = rank
        self.action_influence: str = 'none'  # 'direct', 'constrain', 'none'


@dataclass
class RetrieveResult:
    candidates: List[RetrievedCandidate]
    selected_ids: List[str]
    action_influence: str = 'none'
    contract: Optional[Dict[str, Any]] = None


def _representation_attr(entity: Any, *names: str, default: str = '') -> str:
    for name in names:
        value = getattr(entity, name, None)
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else str(value).strip()
        if text:
            return text
    return default


def _representation_runtime_value(runtime_snap: Any, *names: str, default: Any = 0) -> Any:
    for source in (runtime_snap, getattr(runtime_snap, 'lifecycle', None)):
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return default


def _build_representation_candidate(
    card: Any,
    *,
    runtime_store: Any,
    rank: int,
    helpful_only: bool = False,
) -> Optional[RetrievedCandidate]:
    card_id = _representation_attr(card, 'rep_id', 'card_id')
    if not card_id:
        return None

    runtime_snap = runtime_store.get_card_runtime_state(card_id)
    helpful = int(_representation_runtime_value(runtime_snap, 'times_helpful', default=0) or 0)
    harmful = int(_representation_runtime_value(runtime_snap, 'times_harmful', default=0) or 0)
    activated = int(_representation_runtime_value(runtime_snap, 'times_activated', default=0) or 0)
    status = _representation_attr(
        runtime_snap,
        'lifecycle_status',
        'current_status',
        default='candidate',
    )

    if helpful_only:
        if helpful <= 0:
            return None
    else:
        if status == 'garbage' or (helpful == 0 and harmful > 0):
            return None
        if helpful <= 0 and activated <= 0:
            return None

    pattern = _representation_attr(
        card,
        'proposed_pattern',
        'summary',
        'name',
        default=f'representation {card_id}',
    )
    rep_obj = {
        'object_id': card_id,
        'object_type': 'representation',
        'summary': pattern,
        'content': {
            'type': 'representation',
            'card_id': card_id,
            'pattern': pattern,
            'abstraction_level': _representation_attr(
                card,
                'abstraction_level',
                'semantic_class',
                default='unknown',
            ),
            'origin_family': _representation_attr(
                card,
                'origin_family',
                'family',
                default='representation',
            ),
        },
    }
    score = 0.5 + (helpful / (helpful + harmful + 1)) * 0.3
    candidate = RetrievedCandidate(
        object_id=card_id,
        object=rep_obj,
        relevance_score=score,
        rank=rank,
    )
    candidate.action_influence = 'representation_card'
    return candidate


@dataclass
class TickContextFrame:
    """Tick-local immutable-ish context bundle to avoid repeated recomputation."""
    episode: int
    tick: int
    perception_summary: Dict[str, Any]
    meta_control_snapshot: Dict[str, Any]
    world_model_summary: Dict[str, Any]
    world_model_transition_priors: Dict[str, Any]
    self_model_summary: Dict[str, Any]
    unified_context: UnifiedCognitiveContext

    def to_context_frame(
        self,
        *,
        task_contract: Optional[Dict[str, Any]] = None,
        goal_ref: str = "",
        task_ref: str = "",
        graph_ref: str = "",
    ) -> Dict[str, Any]:
        from core.conos_kernel import build_context_frame

        return build_context_frame(
            self,
            task_contract=task_contract or {},
            goal_ref=goal_ref,
            task_ref=task_ref,
            graph_ref=graph_ref,
        ).to_dict()


@dataclass
class PlannerStageOutput:
    raw_base_action: Dict[str, Any]
    base_action: Dict[str, Any]
    arm_action: Dict[str, Any]
    arm_meta: Dict[str, Any]
    plan_tick_meta: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]
    visible_functions: List[str]
    discovered_functions: List[str]
    raw_candidates_snapshot: List[Dict[str, Any]]
    deliberation_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionBridgeInput:
    obs_before: Dict[str, Any]
    surfaced: List[Any]
    continuity_snapshot: Dict[str, Any]
    plan_tick_meta: Dict[str, Any]
    candidate_actions: List[Dict[str, Any]]
    frame: TickContextFrame
    deliberation_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceStageOutput:
    candidate_actions: List[Dict[str, Any]]
    decision_outcome: Any
    decision_arbiter_selected: Optional[Dict[str, Any]]
    action_to_use: Dict[str, Any]
    governance_result: Dict[str, Any]


@dataclass
class _NoopMetaSnapshot:
    snapshot_id: str = 'noop_snapshot'
    inputs_hash: str = 'noop_hash'
    retrieval_aggressiveness: float = 0.5
    probe_bias: float = 0.5
    planner_bias: float = 0.5
    retrieval_pressure: float = 0.5
    verification_bias: float = 0.5
    risk_tolerance: float = 0.5
    recovery_bias: float = 0.5
    stability_bias: float = 0.5
    strategy_mode: str = 'balanced'
    last_episode_reward: float = 0.0
    retention_tuning: Dict[str, float] = field(default_factory=dict)

    def to_policy_profile(self) -> Dict[str, float]:
        return {
            'retrieval_aggressiveness': self.retrieval_aggressiveness,
            'probe_bias': self.probe_bias,
            'planner_bias': self.planner_bias,
            'verification_bias': self.verification_bias,
            'risk_tolerance': self.risk_tolerance,
            'recovery_bias': self.recovery_bias,
            'stability_bias': self.stability_bias,
            'strategy_mode': self.strategy_mode,
            'retention_tuning': dict(self.retention_tuning),
        }

    def to_representation_profile(self) -> Dict[str, float]:
        return {
            'retrieval_aggressiveness': self.retrieval_aggressiveness,
            'probe_bias': self.probe_bias,
            'retrieval_pressure': self.retrieval_pressure,
            'verification_bias': self.verification_bias,
            'recovery_bias': self.recovery_bias,
            'stability_bias': self.stability_bias,
            'strategy_mode': self.strategy_mode,
        }


@dataclass
class _AdaptiveMetaSnapshot(_NoopMetaSnapshot):
    pass


class _NoopUpdateEngine:
    def __init__(self) -> None:
        self._stats = {'accepted': 0, 'rejected': 0}

    def get_update_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def get_predictor_trust(self) -> Dict[str, float]:
        return {}

    def summarize(self) -> Dict[str, Any]:
        return {'type': 'noop', 'accepted': 0, 'rejected': 0}

    def get_recent_errors(self) -> List[Any]:
        return []


class _FormalUpdateEngine:
    """Applies governed field updates to existing store objects."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self._stats = {'accepted': 0, 'rejected': 0}
        self._recent_errors: List[Dict[str, Any]] = []

    def apply_object_update(
        self,
        *,
        object_id: str,
        patch: Dict[str, Any],
        reason: str,
        evidence_ids: List[str],
    ) -> Dict[str, Any]:
        target_id = str(object_id or '')
        if not target_id:
            self._stats['rejected'] += 1
            self._recent_errors.append({'object_id': '', 'reason': 'missing_object_id'})
            del self._recent_errors[:-20]
            return {'updated': False, 'result': 'rejected_missing_object_id'}

        if not isinstance(patch, dict) or not patch:
            self._stats['rejected'] += 1
            self._recent_errors.append({'object_id': target_id, 'reason': 'empty_patch'})
            del self._recent_errors[:-20]
            return {'updated': False, 'result': 'rejected_empty_patch'}

        updated_id = self._store.update_fields(
            target_id,
            patch=patch,
            reason=str(reason or 'learning_adjustment'),
            evidence_ids=list(evidence_ids or []),
        )
        if not updated_id:
            self._stats['rejected'] += 1
            self._recent_errors.append({
                'object_id': target_id,
                'reason': 'store_rejected_update',
                'patch_keys': sorted(patch.keys()),
            })
            del self._recent_errors[:-20]
            return {'updated': False, 'result': 'rejected_by_store'}

        self._stats['accepted'] += 1
        return {'updated': True, 'result': 'committed', 'object_id': updated_id, 'applied_patch': dict(patch)}

    def get_update_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def get_predictor_trust(self) -> Dict[str, float]:
        return {}

    def summarize(self) -> Dict[str, Any]:
        return {
            'type': 'formal_update_engine',
            'accepted': int(self._stats.get('accepted', 0)),
            'rejected': int(self._stats.get('rejected', 0)),
        }

    def get_recent_errors(self) -> List[Any]:
        return list(self._recent_errors)


class _NoopPromotionEngine:
    """Safe default so episode-end learning can run before promotions are implemented."""

    def check_promotions(self, episode: int) -> List[Any]:
        return []

    def to_update_payload(self, candidate: Any, episode: int, tick: int) -> Dict[str, Any]:
        return {'asset_status': '', 'evidence': []}

    def record_promotion_result(self, candidate: Any, result: str) -> None:
        return None


class _NoopMetaControl:
    def __init__(self) -> None:
        self.policy_profile_object_id = ''
        self.representation_profile_object_id = ''
        self.policy_read_fallback_events: List[str] = []

    def get_snapshot(self, episode: int, tick: int, context: Optional[Dict[str, Any]] = None) -> _NoopMetaSnapshot:
        return _NoopMetaSnapshot()

    def for_retrieval_gate(self, **kwargs) -> Dict[str, Any]:
        return {
            'retrieval_aggressiveness': 0.5,
            'retrieval_pressure': 0.5,
            'probe_bias': 0.5,
            'planner_bias': 0.5,
            'verification_bias': 0.5,
            'risk_tolerance': 0.5,
            'recovery_bias': 0.5,
            'stability_bias': 0.5,
            'strategy_mode': 'balanced',
            'meta_control_snapshot_id': 'noop_snapshot',
            'meta_control_inputs_hash': 'noop_hash',
        }

    def for_rerank_query_gate(self, **kwargs) -> Dict[str, Any]:
        return {
            'retrieval_aggressiveness': 0.5,
            'retrieval_pressure': 0.5,
            'probe_bias': 0.5,
            'planner_bias': 0.5,
            'verification_bias': 0.5,
            'risk_tolerance': 0.5,
            'recovery_bias': 0.5,
            'stability_bias': 0.5,
            'strategy_mode': 'balanced',
        }

    def for_probe_gate(self, **kwargs) -> Dict[str, Any]:
        return {
            'retrieval_aggressiveness': 0.5,
            'retrieval_pressure': 0.5,
            'probe_bias': 0.5,
            'planner_bias': 0.5,
            'verification_bias': 0.5,
            'risk_tolerance': 0.5,
            'recovery_bias': 0.5,
            'stability_bias': 0.5,
            'strategy_mode': 'balanced',
            'meta_control_snapshot_id': 'noop_snapshot',
            'meta_control_inputs_hash': 'noop_hash',
        }

    def for_planner_replan(self, **kwargs) -> Dict[str, Any]:
        return {
            'retrieval_aggressiveness': 0.5,
            'retrieval_pressure': 0.5,
            'probe_bias': 0.5,
            'planner_bias': 0.5,
            'verification_bias': 0.5,
            'risk_tolerance': 0.5,
            'recovery_bias': 0.5,
            'stability_bias': 0.5,
            'strategy_mode': 'balanced',
            'meta_control_snapshot_id': 'noop_snapshot',
            'meta_control_inputs_hash': 'noop_hash',
            'policy_profile': {
                'retrieval_aggressiveness': 0.5,
                'probe_bias': 0.5,
                'planner_bias': 0.5,
                'verification_bias': 0.5,
                'risk_tolerance': 0.5,
                'recovery_bias': 0.5,
                'stability_bias': 0.5,
                'strategy_mode': 'balanced',
            },
            'representation_profile': {
                'retrieval_aggressiveness': 0.5,
                'probe_bias': 0.5,
                'retrieval_pressure': 0.5,
                'verification_bias': 0.5,
                'recovery_bias': 0.5,
                'stability_bias': 0.5,
                'strategy_mode': 'balanced',
            },
        }

    def bootstrap_policy_profile_object(self, **kwargs) -> None:
        return None

    def apply_learning_policy_updates(self, **kwargs) -> Dict[str, Any]:
        return {'updated': False}

    def apply_runtime_hints(self, **kwargs) -> None:
        return None

    def describe_state(self) -> Dict[str, Any]:
        snapshot = self.get_snapshot(episode=0, tick=0, context={'phase': 'describe_state'})
        return {
            'policy_state': snapshot.to_policy_profile(),
            'representation_state': snapshot.to_representation_profile(),
            'strategy_state': {
                'strategy_mode': str(getattr(snapshot, 'strategy_mode', 'balanced') or 'balanced'),
                'verification_bias': float(getattr(snapshot, 'verification_bias', 0.5) or 0.5),
                'risk_tolerance': float(getattr(snapshot, 'risk_tolerance', 0.5) or 0.5),
                'recovery_bias': float(getattr(snapshot, 'recovery_bias', 0.5) or 0.5),
                'stability_bias': float(getattr(snapshot, 'stability_bias', 0.5) or 0.5),
            },
            'signal_state': {},
        }


class _AdaptiveMetaControl:
    """Lightweight online meta-controller for retrieval / probe / planner biases."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self.policy_profile_object_id = ''
        self.representation_profile_object_id = ''
        self.policy_read_fallback_events: List[str] = []
        self._state_version = 0
        self._last_episode_reward = 0.0
        self._policy_state: Dict[str, float] = {
            'retrieval_aggressiveness': 0.5,
            'probe_bias': 0.5,
            'planner_bias': 0.5,
        }
        self._representation_state: Dict[str, float] = {
            'retrieval_aggressiveness': 0.5,
            'probe_bias': 0.5,
            'retrieval_pressure': 0.5,
        }
        self._strategy_state: Dict[str, Any] = {
            'strategy_mode': 'balanced',
            'verification_bias': 0.5,
            'risk_tolerance': 0.5,
            'recovery_bias': 0.5,
            'stability_bias': 0.5,
            'dominant_signal': 'bootstrap',
            'last_transition_reason': 'bootstrap',
        }
        self._signal_state: Dict[str, float] = {
            'reward_delta_ema': 0.0,
            'retrieval_miss_ratio_ema': 0.0,
            'prediction_error_ema': 0.0,
            'recovery_instability_ema': 0.0,
            'mechanism_support_ema': 0.5,
            'runtime_hint_pressure_ema': 0.0,
            'retention_failure_severity_ema': 0.0,
            'branch_persistence_collapse_ema': 0.0,
        }
        self._runtime_hints: Dict[str, float] = {
            'retrieval_delta': 0.0,
            'probe_delta': 0.0,
            'planner_delta': 0.0,
        }
        self._retention_tuning_state: Dict[str, float] = {
            'world_shift_replan_threshold_delta': 0.0,
            'hidden_drift_replan_threshold_delta': 0.0,
            'high_risk_replan_threshold_delta': 0.0,
            'value_drop_threshold_delta': 0.0,
            'uncertainty_threshold_delta': 0.0,
            'belief_branch_margin_threshold_delta': 0.0,
            'branch_persistence_margin_threshold_delta': 0.0,
            'low_branch_persistence_threshold_delta': 0.0,
            'branch_budget_bonus': 0.0,
            'verification_budget_bonus': 0.0,
        }

    def _append_fallback_event(self, event: str) -> None:
        if not event:
            return
        self.policy_read_fallback_events.append(str(event))
        del self.policy_read_fallback_events[:-20]

    @staticmethod
    def _clamp(value: Any, minimum: float = 0.05, maximum: float = 0.95, default: float = 0.5) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(default)
        return max(minimum, min(maximum, numeric))

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _ema(previous: Any, current: Any, *, momentum: float = 0.65, default: float = 0.0) -> float:
        prev_value = _AdaptiveMetaControl._clamp(previous, minimum=0.0, maximum=1.0, default=default)
        curr_value = _AdaptiveMetaControl._clamp(current, minimum=0.0, maximum=1.0, default=default)
        momentum = max(0.0, min(0.95, float(momentum)))
        return ((momentum * prev_value) + ((1.0 - momentum) * curr_value))

    @staticmethod
    def _restore_float_mapping(target: Dict[str, float], source: Dict[str, Any], keys: List[str]) -> None:
        for key in keys:
            if key in source:
                target[key] = _AdaptiveMetaControl._clamp(
                    source.get(key),
                    minimum=0.0,
                    maximum=1.0,
                    default=target.get(key, 0.5),
                )

    @staticmethod
    def _restore_delta_mapping(target: Dict[str, float], source: Dict[str, Any], keys: List[str], *, minimum: float, maximum: float) -> None:
        for key in keys:
            if key in source:
                target[key] = _AdaptiveMetaControl._clamp(
                    source.get(key),
                    minimum=minimum,
                    maximum=maximum,
                    default=target.get(key, 0.0),
                )

    @staticmethod
    def _retention_tuning_keys() -> List[str]:
        return [
            'world_shift_replan_threshold_delta',
            'hidden_drift_replan_threshold_delta',
            'high_risk_replan_threshold_delta',
            'value_drop_threshold_delta',
            'uncertainty_threshold_delta',
            'belief_branch_margin_threshold_delta',
            'branch_persistence_margin_threshold_delta',
            'low_branch_persistence_threshold_delta',
            'branch_budget_bonus',
            'verification_budget_bonus',
        ]

    def _find_profile_object(self, profile_type: str) -> Tuple[str, Dict[str, Any]]:
        best_id = ''
        best_obj: Dict[str, Any] = {}
        best_created_at = ''
        for obj in self._store.iter_objects():
            if not isinstance(obj, dict):
                continue
            content = self._as_dict(obj.get('content'))
            metadata = self._as_dict(obj.get('memory_metadata'))
            if str(content.get('profile_type') or metadata.get('profile_type') or '') != profile_type:
                continue
            tags = set(str(tag) for tag in list(obj.get('retrieval_tags', []) or []))
            if 'meta_control' not in tags:
                continue
            created_at = str(obj.get('created_at') or '')
            if created_at >= best_created_at:
                best_created_at = created_at
                best_id = str(obj.get('object_id') or '')
                best_obj = obj
        return best_id, best_obj

    def _restore_profile_state(self, profile_type: str, obj: Dict[str, Any]) -> None:
        content = self._as_dict(obj.get('content'))
        metadata = self._as_dict(obj.get('memory_metadata'))
        if profile_type == 'policy_profile':
            self._restore_float_mapping(
                self._policy_state,
                content,
                ['retrieval_aggressiveness', 'probe_bias', 'planner_bias'],
            )
            self._restore_float_mapping(
                self._strategy_state,
                content,
                ['verification_bias', 'risk_tolerance', 'recovery_bias', 'stability_bias'],
            )
        elif profile_type == 'representation_profile':
            self._restore_float_mapping(
                self._representation_state,
                content,
                ['retrieval_aggressiveness', 'probe_bias', 'retrieval_pressure'],
            )

        mode = str(content.get('strategy_mode') or metadata.get('strategy_mode') or self._strategy_state.get('strategy_mode', 'balanced') or 'balanced')
        if mode:
            self._strategy_state['strategy_mode'] = mode
        dominant_signal = str(metadata.get('dominant_signal') or content.get('dominant_signal') or self._strategy_state.get('dominant_signal', '') or '')
        if dominant_signal:
            self._strategy_state['dominant_signal'] = dominant_signal
        transition_reason = str(metadata.get('last_transition_reason') or content.get('last_transition_reason') or self._strategy_state.get('last_transition_reason', '') or '')
        if transition_reason:
            self._strategy_state['last_transition_reason'] = transition_reason

        state_version = metadata.get('state_version', content.get('state_version', self._state_version))
        try:
            self._state_version = max(int(self._state_version), int(state_version or 0))
        except (TypeError, ValueError):
            pass
        last_episode_reward = content.get('last_episode_reward', metadata.get('last_episode_reward', self._last_episode_reward))
        try:
            self._last_episode_reward = float(last_episode_reward or 0.0)
        except (TypeError, ValueError):
            self._last_episode_reward = 0.0

        signal_state = self._as_dict(content.get('signal_state'))
        self._restore_float_mapping(
            self._signal_state,
            signal_state,
            [
                'reward_delta_ema',
                'retrieval_miss_ratio_ema',
                'prediction_error_ema',
                'recovery_instability_ema',
                'mechanism_support_ema',
                'runtime_hint_pressure_ema',
                'retention_failure_severity_ema',
                'branch_persistence_collapse_ema',
            ],
        )
        retention_tuning = self._as_dict(content.get('retention_tuning'))
        self._restore_delta_mapping(
            self._retention_tuning_state,
            retention_tuning,
            [
                'world_shift_replan_threshold_delta',
                'hidden_drift_replan_threshold_delta',
                'high_risk_replan_threshold_delta',
                'value_drop_threshold_delta',
                'uncertainty_threshold_delta',
                'belief_branch_margin_threshold_delta',
                'branch_persistence_margin_threshold_delta',
                'low_branch_persistence_threshold_delta',
            ],
            minimum=-0.25,
            maximum=0.25,
        )
        self._restore_delta_mapping(
            self._retention_tuning_state,
            retention_tuning,
            ['branch_budget_bonus', 'verification_budget_bonus'],
            minimum=0.0,
            maximum=2.0,
        )

    def _current_mode_targets(self, mode: str) -> Dict[str, float]:
        table = {
            'balanced': {'retrieval_aggressiveness': 0.50, 'probe_bias': 0.50, 'planner_bias': 0.50, 'retrieval_pressure': 0.50},
            'explore': {'retrieval_aggressiveness': 0.72, 'probe_bias': 0.66, 'planner_bias': 0.46, 'retrieval_pressure': 0.70},
            'verify': {'retrieval_aggressiveness': 0.62, 'probe_bias': 0.78, 'planner_bias': 0.58, 'retrieval_pressure': 0.67},
            'recover': {'retrieval_aggressiveness': 0.74, 'probe_bias': 0.82, 'planner_bias': 0.72, 'retrieval_pressure': 0.76},
            'exploit': {'retrieval_aggressiveness': 0.38, 'probe_bias': 0.34, 'planner_bias': 0.64, 'retrieval_pressure': 0.36},
        }
        return dict(table.get(str(mode or 'balanced'), table['balanced']))

    def _derive_strategy_state(
        self,
        *,
        reward_delta: float,
        retrieval_miss_ratio: float,
        error_level: float,
        recovery_instability: float,
        mechanism_support: float,
        runtime_hint_pressure: float,
        retention_failure_type: str = '',
        retention_failure_severity: float = 0.0,
    ) -> Dict[str, Any]:
        self._signal_state['reward_delta_ema'] = self._ema(
            self._signal_state.get('reward_delta_ema', 0.0),
            max(0.0, min(1.0, abs(float(reward_delta)))),
            momentum=0.60,
            default=0.0,
        )
        self._signal_state['retrieval_miss_ratio_ema'] = self._ema(
            self._signal_state.get('retrieval_miss_ratio_ema', 0.0),
            retrieval_miss_ratio,
            default=0.0,
        )
        self._signal_state['prediction_error_ema'] = self._ema(
            self._signal_state.get('prediction_error_ema', 0.0),
            error_level,
            default=0.0,
        )
        self._signal_state['recovery_instability_ema'] = self._ema(
            self._signal_state.get('recovery_instability_ema', 0.0),
            recovery_instability,
            default=0.0,
        )
        self._signal_state['mechanism_support_ema'] = self._ema(
            self._signal_state.get('mechanism_support_ema', 0.5),
            mechanism_support,
            default=0.5,
        )
        self._signal_state['runtime_hint_pressure_ema'] = self._ema(
            self._signal_state.get('runtime_hint_pressure_ema', 0.0),
            runtime_hint_pressure,
            momentum=0.55,
            default=0.0,
        )
        branch_persistence_collapse = retention_failure_severity if str(retention_failure_type or '') == 'branch_persistence_collapse' else 0.0
        self._signal_state['retention_failure_severity_ema'] = self._ema(
            self._signal_state.get('retention_failure_severity_ema', 0.0),
            retention_failure_severity,
            momentum=0.55,
            default=0.0,
        )
        self._signal_state['branch_persistence_collapse_ema'] = self._ema(
            self._signal_state.get('branch_persistence_collapse_ema', 0.0),
            branch_persistence_collapse,
            momentum=0.55,
            default=0.0,
        )

        negative_reward = max(0.0, min(1.0, -float(reward_delta)))
        positive_reward = max(0.0, min(1.0, float(reward_delta)))
        retrieval_pressure = float(self._signal_state['retrieval_miss_ratio_ema'])
        prediction_pressure = float(self._signal_state['prediction_error_ema'])
        recovery_pressure = float(self._signal_state['recovery_instability_ema'])
        support_pressure = float(self._signal_state['mechanism_support_ema'])
        hint_pressure = float(self._signal_state['runtime_hint_pressure_ema'])
        retention_pressure = float(self._signal_state['retention_failure_severity_ema'])
        branch_pressure = float(self._signal_state['branch_persistence_collapse_ema'])

        mode_scores = {
            'recover': min(
                1.0,
                recovery_instability * 0.42
                + recovery_pressure * 0.18
                + error_level * 0.20
                + prediction_pressure * 0.10
                + retention_pressure * 0.18
                + branch_pressure * 0.16
                + negative_reward * 0.18
                + runtime_hint_pressure * 0.12
            ),
            'verify': min(
                1.0,
                error_level * 0.28
                + prediction_pressure * 0.22
                + retention_pressure * 0.16
                + runtime_hint_pressure * 0.12
                + retrieval_miss_ratio * 0.10
                + retrieval_pressure * 0.12
                + max(0.0, 1.0 - mechanism_support) * 0.08
                + (0.12 if str(retention_failure_type or '') in {'prediction_drift', 'governance_overrule_misfire'} else 0.0)
            ),
            'explore': min(
                1.0,
                retrieval_miss_ratio * 0.26
                + retrieval_pressure * 0.20
                + runtime_hint_pressure * 0.08
                + max(0.0, 0.6 - mechanism_support) * 0.20
                + max(0.0, 0.4 - recovery_instability) * 0.10
            ),
            'exploit': min(
                1.0,
                positive_reward * 0.36
                + support_pressure * 0.26
                + mechanism_support * 0.20
                + max(0.0, 0.7 - error_level) * 0.08
                + max(0.0, 0.6 - recovery_instability) * 0.08
                - retention_pressure * 0.08
            ),
            'balanced': 0.34,
        }

        previous_mode = str(self._strategy_state.get('strategy_mode', 'balanced') or 'balanced')
        mode_scores[previous_mode] = mode_scores.get(previous_mode, 0.0) + 0.04
        next_mode = max(mode_scores.items(), key=lambda item: item[1])[0]
        if previous_mode != next_mode and mode_scores[next_mode] < mode_scores.get(previous_mode, 0.0) + 0.03:
            next_mode = previous_mode

        dominant_signal = max(
            [
                ('recovery_instability', recovery_pressure),
                ('prediction_error', prediction_pressure),
                ('retrieval_miss', retrieval_pressure),
                ('mechanism_support', support_pressure),
                ('runtime_hint', hint_pressure),
                ('retention_failure', retention_pressure),
                ('branch_persistence', branch_pressure),
                ('negative_reward', negative_reward),
                ('positive_reward', positive_reward),
            ],
            key=lambda item: item[1],
        )[0]
        transition_reason = dominant_signal if next_mode != previous_mode else f'stay:{dominant_signal}'

        verification_bias = self._clamp(
            0.48
            + prediction_pressure * 0.22
            + retrieval_pressure * 0.08
            + retention_pressure * 0.12
            + (0.10 if next_mode == 'verify' else 0.0)
            + (0.06 if next_mode == 'recover' else 0.0)
            - support_pressure * 0.10,
            default=float(self._strategy_state.get('verification_bias', 0.5) or 0.5),
        )
        recovery_bias = self._clamp(
            0.44
            + recovery_pressure * 0.28
            + branch_pressure * 0.12
            + hint_pressure * 0.12
            + (0.14 if next_mode == 'recover' else 0.0)
            - positive_reward * 0.08,
            default=float(self._strategy_state.get('recovery_bias', 0.5) or 0.5),
        )
        risk_tolerance = self._clamp(
            0.50
            + (0.12 if next_mode == 'exploit' else 0.0)
            + (0.06 if next_mode == 'explore' else 0.0)
            - prediction_pressure * 0.15
            - recovery_pressure * 0.18
            - retention_pressure * 0.12
            + support_pressure * 0.10,
            default=float(self._strategy_state.get('risk_tolerance', 0.5) or 0.5),
        )
        stability_bias = self._clamp(
            0.50
            + support_pressure * 0.12
            + positive_reward * 0.08
            - prediction_pressure * 0.14
            - recovery_pressure * 0.10
            - branch_pressure * 0.08
            + (0.08 if next_mode == 'exploit' else 0.0)
            + (0.05 if next_mode == 'verify' else 0.0),
            default=float(self._strategy_state.get('stability_bias', 0.5) or 0.5),
        )

        return {
            'strategy_mode': next_mode,
            'verification_bias': verification_bias,
            'risk_tolerance': risk_tolerance,
            'recovery_bias': recovery_bias,
            'stability_bias': stability_bias,
            'dominant_signal': dominant_signal,
            'last_transition_reason': transition_reason,
            'mode_scores': mode_scores,
            'previous_mode': previous_mode,
        }

    def _policy_object_content(self) -> Dict[str, Any]:
        return {
            'profile_type': 'policy_profile',
            'state_version': int(self._state_version),
            'retrieval_aggressiveness': float(self._policy_state['retrieval_aggressiveness']),
            'probe_bias': float(self._policy_state['probe_bias']),
            'planner_bias': float(self._policy_state['planner_bias']),
            'verification_bias': float(self._strategy_state['verification_bias']),
            'risk_tolerance': float(self._strategy_state['risk_tolerance']),
            'recovery_bias': float(self._strategy_state['recovery_bias']),
            'stability_bias': float(self._strategy_state['stability_bias']),
            'strategy_mode': str(self._strategy_state['strategy_mode']),
            'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
            'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
            'signal_state': dict(self._signal_state),
            'retention_tuning': dict(self._retention_tuning_state),
            'last_episode_reward': float(self._last_episode_reward),
        }

    def _representation_object_content(self) -> Dict[str, Any]:
        return {
            'profile_type': 'representation_profile',
            'state_version': int(self._state_version),
            'retrieval_aggressiveness': float(self._representation_state['retrieval_aggressiveness']),
            'probe_bias': float(self._representation_state['probe_bias']),
            'retrieval_pressure': float(self._representation_state['retrieval_pressure']),
            'verification_bias': float(self._strategy_state['verification_bias']),
            'recovery_bias': float(self._strategy_state['recovery_bias']),
            'stability_bias': float(self._strategy_state['stability_bias']),
            'strategy_mode': str(self._strategy_state['strategy_mode']),
            'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
            'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
            'signal_state': dict(self._signal_state),
            'last_episode_reward': float(self._last_episode_reward),
        }

    def _build_profile_proposal(self, *, profile_type: str, episode: int, tick: int) -> Dict[str, Any]:
        is_policy = profile_type == 'policy_profile'
        content = self._policy_object_content() if is_policy else self._representation_object_content()
        tags = ['meta_control', profile_type, 'adaptive']
        return {
            'content': content,
            'confidence': 0.72,
            'memory_type': 'control_profile',
            'memory_layer': 'semantic',
            'retrieval_tags': tags,
            'memory_metadata': {
                'profile_type': profile_type,
                'owner': 'meta_control',
                'state_version': int(self._state_version),
                'last_episode_reward': float(self._last_episode_reward),
                'strategy_mode': str(self._strategy_state.get('strategy_mode', 'balanced') or 'balanced'),
                'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
                'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
            },
            'source_module': 'meta_control',
            'source_stage': 'bootstrap',
            'trigger_source': 'meta_control',
            'trigger_episode': int(episode),
            'episode': int(episode),
            'provenance': {'tick': int(tick), 'profile_type': profile_type},
        }

    def _ensure_profile_objects(self, episode: int, tick: int) -> None:
        if not self.policy_profile_object_id or self._store.get(self.policy_profile_object_id) is None:
            existing_id, existing_obj = self._find_profile_object('policy_profile')
            if existing_id and existing_obj:
                self.policy_profile_object_id = existing_id
                self._restore_profile_state('policy_profile', existing_obj)
                self._append_fallback_event('policy_profile_restored')
            else:
                proposal = self._build_profile_proposal(profile_type='policy_profile', episode=episode, tick=tick)
                self.policy_profile_object_id = self._store.add(proposal, ACCEPT_NEW, [])
                self._append_fallback_event('policy_profile_bootstrap')
        if not self.representation_profile_object_id or self._store.get(self.representation_profile_object_id) is None:
            existing_id, existing_obj = self._find_profile_object('representation_profile')
            if existing_id and existing_obj:
                self.representation_profile_object_id = existing_id
                self._restore_profile_state('representation_profile', existing_obj)
                self._append_fallback_event('representation_profile_restored')
            else:
                proposal = self._build_profile_proposal(profile_type='representation_profile', episode=episode, tick=tick)
                self.representation_profile_object_id = self._store.add(proposal, ACCEPT_NEW, [])
                self._append_fallback_event('representation_profile_bootstrap')

    def _build_snapshot(self, episode: int, tick: int, context: Optional[Dict[str, Any]] = None) -> _AdaptiveMetaSnapshot:
        self._ensure_profile_objects(episode, tick)
        context_payload = self._as_dict(context)
        retrieval_aggressiveness = self._clamp(
            self._policy_state['retrieval_aggressiveness'] + self._runtime_hints['retrieval_delta'],
            default=self._policy_state['retrieval_aggressiveness'],
        )
        probe_bias = self._clamp(
            self._policy_state['probe_bias'] + self._runtime_hints['probe_delta'],
            default=self._policy_state['probe_bias'],
        )
        planner_bias = self._clamp(
            self._policy_state['planner_bias'] + self._runtime_hints['planner_delta'],
            default=self._policy_state['planner_bias'],
        )
        retrieval_pressure = self._clamp(
            self._representation_state['retrieval_pressure'] + self._runtime_hints['retrieval_delta'] * 0.65,
            default=self._representation_state['retrieval_pressure'],
        )
        verification_bias = self._clamp(
            self._strategy_state['verification_bias'] + max(0.0, self._runtime_hints['probe_delta']) * 0.25,
            default=self._strategy_state['verification_bias'],
        )
        risk_tolerance = self._clamp(
            self._strategy_state['risk_tolerance'] + self._runtime_hints['planner_delta'] * 0.20,
            default=self._strategy_state['risk_tolerance'],
        )
        recovery_bias = self._clamp(
            self._strategy_state['recovery_bias'] + max(0.0, self._runtime_hints['retrieval_delta']) * 0.22,
            default=self._strategy_state['recovery_bias'],
        )
        stability_bias = self._clamp(
            self._strategy_state['stability_bias'] - max(0.0, self._runtime_hints['retrieval_delta']) * 0.08,
            default=self._strategy_state['stability_bias'],
        )
        strategy_mode = str(self._strategy_state.get('strategy_mode', 'balanced') or 'balanced')
        payload = {
            'episode': int(episode),
            'tick': int(tick),
            'state_version': int(self._state_version),
            'context': context_payload,
            'policy_state': {
                'retrieval_aggressiveness': retrieval_aggressiveness,
                'probe_bias': probe_bias,
                'planner_bias': planner_bias,
            },
            'representation_state': {
                'retrieval_aggressiveness': self._representation_state['retrieval_aggressiveness'],
                'probe_bias': self._representation_state['probe_bias'],
                'retrieval_pressure': retrieval_pressure,
            },
            'strategy_state': {
                'strategy_mode': strategy_mode,
                'verification_bias': verification_bias,
                'risk_tolerance': risk_tolerance,
                'recovery_bias': recovery_bias,
                'stability_bias': stability_bias,
                'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
                'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
            },
            'retention_tuning': dict(self._retention_tuning_state),
            'last_episode_reward': float(self._last_episode_reward),
        }
        inputs_hash = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:12]
        snapshot_id = f"meta_{episode}_{tick}_{self._state_version}_{inputs_hash[:6]}"
        return _AdaptiveMetaSnapshot(
            snapshot_id=snapshot_id,
            inputs_hash=inputs_hash,
            retrieval_aggressiveness=retrieval_aggressiveness,
            probe_bias=probe_bias,
            planner_bias=planner_bias,
            retrieval_pressure=retrieval_pressure,
            verification_bias=verification_bias,
            risk_tolerance=risk_tolerance,
            recovery_bias=recovery_bias,
            stability_bias=stability_bias,
            strategy_mode=strategy_mode,
            last_episode_reward=float(self._last_episode_reward),
            retention_tuning=dict(self._retention_tuning_state),
        )

    def get_snapshot(self, episode: int, tick: int, context: Optional[Dict[str, Any]] = None) -> _AdaptiveMetaSnapshot:
        return self._build_snapshot(episode, tick, context=context)

    def _snapshot_to_gate(self, snapshot: _AdaptiveMetaSnapshot) -> Dict[str, Any]:
        return {
            'retrieval_aggressiveness': float(snapshot.retrieval_aggressiveness),
            'retrieval_pressure': float(snapshot.retrieval_pressure),
            'probe_bias': float(snapshot.probe_bias),
            'planner_bias': float(snapshot.planner_bias),
            'verification_bias': float(snapshot.verification_bias),
            'risk_tolerance': float(snapshot.risk_tolerance),
            'recovery_bias': float(snapshot.recovery_bias),
            'stability_bias': float(snapshot.stability_bias),
            'strategy_mode': str(snapshot.strategy_mode),
            'meta_control_snapshot_id': str(snapshot.snapshot_id),
            'meta_control_inputs_hash': str(snapshot.inputs_hash),
            'policy_profile': snapshot.to_policy_profile(),
            'representation_profile': snapshot.to_representation_profile(),
            'retention_tuning': dict(snapshot.retention_tuning),
        }

    def for_retrieval_gate(self, **kwargs) -> Dict[str, Any]:
        snapshot = self._build_snapshot(
            int(kwargs.get('episode', 0) or 0),
            int(kwargs.get('tick', 0) or 0),
            context={'gate': 'retrieval', **self._as_dict(kwargs.get('context'))},
        )
        return self._snapshot_to_gate(snapshot)

    def for_rerank_query_gate(self, **kwargs) -> Dict[str, Any]:
        snapshot = self._build_snapshot(
            int(kwargs.get('episode', 0) or 0),
            int(kwargs.get('tick', 0) or 0),
            context={'gate': 'rerank_query', **self._as_dict(kwargs.get('context'))},
        )
        return self._snapshot_to_gate(snapshot)

    def for_probe_gate(self, **kwargs) -> Dict[str, Any]:
        snapshot = self._build_snapshot(
            int(kwargs.get('episode', 0) or 0),
            int(kwargs.get('tick', 0) or 0),
            context={'gate': 'probe', **self._as_dict(kwargs.get('context'))},
        )
        return self._snapshot_to_gate(snapshot)

    def for_planner_replan(self, **kwargs) -> Dict[str, Any]:
        snapshot = self._build_snapshot(
            int(kwargs.get('episode', 0) or 0),
            int(kwargs.get('tick', 0) or 0),
            context={'gate': 'planner', **self._as_dict(kwargs.get('context'))},
        )
        return self._snapshot_to_gate(snapshot)

    def bootstrap_policy_profile_object(self, **kwargs) -> None:
        self._ensure_profile_objects(
            int(kwargs.get('episode', 0) or 0),
            int(kwargs.get('tick', 0) or 0),
        )

    def apply_learning_policy_updates(self, **kwargs) -> Dict[str, Any]:
        episode = int(kwargs.get('episode', 0) or 0)
        tick = int(kwargs.get('tick', 0) or 0)
        episode_reward = float(kwargs.get('episode_reward', 0.0) or 0.0)
        adaptation_inputs = self._as_dict(kwargs.get('adaptation_inputs'))
        enable_representation_adaptation = bool(kwargs.get('enable_representation_adaptation', True))
        freeze_retrieval_pressure = bool(kwargs.get('freeze_retrieval_pressure', False))

        self._ensure_profile_objects(episode, tick)

        reward_delta = episode_reward - float(self._last_episode_reward)
        episode_trace = self._as_dict(adaptation_inputs.get('episode_trace'))
        prediction_error = self._as_dict(adaptation_inputs.get('prediction_error'))
        recovery_trace = self._as_dict(adaptation_inputs.get('recovery_trace'))
        mechanism_evidence = self._as_dict(adaptation_inputs.get('mechanism_evidence'))
        failure_summary = self._as_dict(adaptation_inputs.get('self_model_failure_summary'))
        retention_failure = self._as_dict(adaptation_inputs.get('retention_failure'))

        retrieval_miss_ratio = float(episode_trace.get('retrieval_miss_ratio', 0.0) or 0.0)
        error_level = float(prediction_error.get('error_level', 0.0) or 0.0)
        recovery_quality = float(recovery_trace.get('quality', 1.0) or 1.0)
        recovery_steps = float(recovery_trace.get('steps', 0.0) or 0.0)
        mechanism_support = float(mechanism_evidence.get('support', 0.0) or 0.0)
        retention_failure_type = str(retention_failure.get('failure_type', '') or '')
        retention_failure_severity = self._clamp(
            retention_failure.get('severity', 0.0),
            minimum=0.0,
            maximum=1.0,
            default=0.0,
        )
        recovery_instability = max(
            0.0,
            min(1.0, ((1.0 - max(0.0, min(1.0, recovery_quality))) * 0.7) + (min(1.0, recovery_steps / 4.0) * 0.3)),
        )
        runtime_hint_pressure = max(
            0.0,
            min(1.0, abs(float(self._runtime_hints.get('retrieval_delta', 0.0) or 0.0)) + abs(float(self._runtime_hints.get('probe_delta', 0.0) or 0.0)) + abs(float(self._runtime_hints.get('planner_delta', 0.0) or 0.0))),
        )
        strategy_state = self._derive_strategy_state(
            reward_delta=reward_delta,
            retrieval_miss_ratio=retrieval_miss_ratio,
            error_level=error_level,
            recovery_instability=recovery_instability,
            mechanism_support=mechanism_support,
            runtime_hint_pressure=runtime_hint_pressure,
            retention_failure_type=retention_failure_type,
            retention_failure_severity=retention_failure_severity,
        )
        previous_strategy_snapshot = dict(self._strategy_state)
        previous_retention_tuning = dict(self._retention_tuning_state)
        previous_mode = str(strategy_state.get('previous_mode', self._strategy_state.get('strategy_mode', 'balanced')) or 'balanced')
        strategy_mode = str(strategy_state.get('strategy_mode', previous_mode) or previous_mode)
        self._strategy_state.update({
            'strategy_mode': strategy_mode,
            'verification_bias': float(strategy_state.get('verification_bias', self._strategy_state.get('verification_bias', 0.5)) or 0.5),
            'risk_tolerance': float(strategy_state.get('risk_tolerance', self._strategy_state.get('risk_tolerance', 0.5)) or 0.5),
            'recovery_bias': float(strategy_state.get('recovery_bias', self._strategy_state.get('recovery_bias', 0.5)) or 0.5),
            'stability_bias': float(strategy_state.get('stability_bias', self._strategy_state.get('stability_bias', 0.5)) or 0.5),
            'dominant_signal': str(strategy_state.get('dominant_signal', self._strategy_state.get('dominant_signal', '')) or ''),
            'last_transition_reason': str(strategy_state.get('last_transition_reason', self._strategy_state.get('last_transition_reason', '')) or ''),
        })
        mode_targets = self._current_mode_targets(strategy_mode)

        trigger_evidence: List[str] = []
        control_deltas = {
            'retrieval_aggressiveness': 0.0,
            'probe_bias': 0.0,
            'planner_bias': 0.0,
        }
        representation_deltas = {
            'retrieval_aggressiveness': 0.0,
            'probe_bias': 0.0,
            'retrieval_pressure': 0.0,
        }
        trigger_evidence.append(f'strategy_mode:{strategy_mode}')
        if previous_mode != strategy_mode:
            trigger_evidence.append(f'strategy_transition:{previous_mode}->{strategy_mode}')

        if reward_delta <= -0.05:
            trigger_evidence.append('negative_reward_delta')
            control_deltas['retrieval_aggressiveness'] += 0.06
            control_deltas['probe_bias'] += 0.05
            control_deltas['planner_bias'] += 0.04
            representation_deltas['retrieval_pressure'] += 0.05
        elif reward_delta >= 0.05:
            trigger_evidence.append('positive_reward_delta')
            control_deltas['retrieval_aggressiveness'] -= 0.03
            control_deltas['probe_bias'] -= 0.02
            control_deltas['planner_bias'] -= 0.03
            representation_deltas['retrieval_pressure'] -= 0.02

        if retrieval_miss_ratio >= 0.25:
            trigger_evidence.append('retrieval_miss_ratio_high')
            magnitude = min(0.18, 0.04 + retrieval_miss_ratio * 0.18)
            control_deltas['retrieval_aggressiveness'] += magnitude
            representation_deltas['retrieval_aggressiveness'] += magnitude * 0.65
            representation_deltas['retrieval_pressure'] += magnitude * 0.9

        if error_level >= 0.35:
            trigger_evidence.append('prediction_error_high')
            magnitude = min(0.2, 0.04 + error_level * 0.16)
            control_deltas['probe_bias'] += magnitude
            control_deltas['planner_bias'] += magnitude * 0.6
            representation_deltas['probe_bias'] += magnitude * 0.55
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += magnitude * 0.45

        if recovery_quality <= 0.65 or recovery_steps >= 2.0:
            trigger_evidence.append('recovery_instability')
            control_deltas['planner_bias'] += 0.07
            control_deltas['probe_bias'] += 0.04
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += 0.05

        failure_event = str(failure_summary.get('event', '') or '')
        if failure_event:
            trigger_evidence.append(f'self_model_failure:{failure_event}')
            if 'shift' in failure_event or 'failure' in failure_event:
                control_deltas['planner_bias'] += 0.04

        if retention_failure_type:
            trigger_evidence.append(f'retention_failure:{retention_failure_type}')
            magnitude = min(0.24, 0.04 + retention_failure_severity * 0.18)
            control_deltas['planner_bias'] += max(
                0.0,
                float(retention_failure.get('planner_replan_bias_delta', 0.0) or 0.0),
            ) + (
                magnitude * 0.85 if retention_failure_type in {'branch_persistence_collapse', 'planner_target_switch'}
                else magnitude * 0.45
            )
            control_deltas['probe_bias'] += max(
                0.0,
                float(retention_failure.get('probe_bias_delta', 0.0) or 0.0),
            ) + (
                magnitude * 0.75 if retention_failure_type in {'prediction_drift', 'governance_overrule_misfire'}
                else magnitude * 0.35
            )
            if retention_failure_type in {'branch_persistence_collapse', 'planner_target_switch'}:
                control_deltas['retrieval_aggressiveness'] += magnitude * 0.30
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += max(
                    0.0,
                    float(retention_failure.get('retrieval_pressure_delta', 0.0) or 0.0),
                ) + (
                    magnitude * 0.40 if retention_failure_type in {'prediction_drift', 'branch_persistence_collapse'}
                    else magnitude * 0.22
                )
            verification_budget_hint = int(retention_failure.get('verification_budget_hint', 0) or 0)
            branch_budget_hint = int(retention_failure.get('branch_budget_hint', 0) or 0)
            if verification_budget_hint >= 2:
                trigger_evidence.append('retention_failure_verification_budget')
                control_deltas['probe_bias'] += 0.04
            if branch_budget_hint >= 2:
                trigger_evidence.append('retention_failure_branch_budget')
                control_deltas['planner_bias'] += 0.05
            strategy_mode_hint = str(retention_failure.get('strategy_mode_hint', '') or '')
            if strategy_mode_hint:
                trigger_evidence.append(f'retention_strategy_hint:{strategy_mode_hint}')
                if strategy_mode_hint == 'verify':
                    control_deltas['probe_bias'] += 0.05
                elif strategy_mode_hint == 'recover':
                    control_deltas['planner_bias'] += 0.04

        if mechanism_support >= 0.7 and reward_delta >= 0.0:
            trigger_evidence.append('mechanism_support_strong')
            control_deltas['probe_bias'] -= 0.03
            control_deltas['planner_bias'] -= 0.02

        if any(abs(value) > 1e-6 for value in self._runtime_hints.values()):
            trigger_evidence.append('runtime_hint_feedback')
            control_deltas['retrieval_aggressiveness'] += self._runtime_hints['retrieval_delta'] * 0.5
            control_deltas['probe_bias'] += self._runtime_hints['probe_delta'] * 0.5
            control_deltas['planner_bias'] += self._runtime_hints['planner_delta'] * 0.5
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += self._runtime_hints['retrieval_delta'] * 0.4

        if strategy_mode == 'recover':
            control_deltas['retrieval_aggressiveness'] += 0.05
            control_deltas['probe_bias'] += 0.08
            control_deltas['planner_bias'] += 0.07
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += 0.08
        elif strategy_mode == 'verify':
            control_deltas['probe_bias'] += 0.08
            control_deltas['planner_bias'] += 0.04
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += 0.05
        elif strategy_mode == 'explore':
            control_deltas['retrieval_aggressiveness'] += 0.06
            control_deltas['probe_bias'] += 0.04
            control_deltas['planner_bias'] -= 0.02
            representation_deltas['retrieval_aggressiveness'] += 0.05
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] += 0.06
        elif strategy_mode == 'exploit':
            control_deltas['retrieval_aggressiveness'] -= 0.07
            control_deltas['probe_bias'] -= 0.08
            control_deltas['planner_bias'] += 0.06
            representation_deltas['retrieval_aggressiveness'] -= 0.05
            if not freeze_retrieval_pressure:
                representation_deltas['retrieval_pressure'] -= 0.08

        retention_targets = {key: 0.0 for key in self._retention_tuning_keys()}
        if retention_failure_type == 'prediction_drift':
            retention_targets.update({
                'world_shift_replan_threshold_delta': -(0.03 + retention_failure_severity * 0.08),
                'hidden_drift_replan_threshold_delta': -(0.04 + retention_failure_severity * 0.08),
                'high_risk_replan_threshold_delta': -(0.02 + retention_failure_severity * 0.05),
                'value_drop_threshold_delta': -(0.03 + retention_failure_severity * 0.06),
                'uncertainty_threshold_delta': -(0.05 + retention_failure_severity * 0.10),
                'verification_budget_bonus': 1.0 + (1.0 if retention_failure_severity >= 0.72 else 0.0),
            })
        elif retention_failure_type == 'branch_persistence_collapse':
            retention_targets.update({
                'value_drop_threshold_delta': -(0.02 + retention_failure_severity * 0.04),
                'belief_branch_margin_threshold_delta': -(0.02 + retention_failure_severity * 0.04),
                'branch_persistence_margin_threshold_delta': -(0.05 + retention_failure_severity * 0.08),
                'low_branch_persistence_threshold_delta': 0.05 + retention_failure_severity * 0.12,
                'branch_budget_bonus': 1.0 + (1.0 if retention_failure_severity >= 0.72 else 0.0),
                'verification_budget_bonus': 1.0 if retention_failure_severity >= 0.55 else 0.0,
            })
        elif retention_failure_type == 'planner_target_switch':
            retention_targets.update({
                'value_drop_threshold_delta': -(0.04 + retention_failure_severity * 0.06),
                'belief_branch_margin_threshold_delta': -(0.04 + retention_failure_severity * 0.07),
                'branch_persistence_margin_threshold_delta': -(0.02 + retention_failure_severity * 0.05),
                'branch_budget_bonus': 1.0,
                'verification_budget_bonus': 1.0 if retention_failure_severity >= 0.65 else 0.0,
            })
        elif retention_failure_type == 'governance_overrule_misfire':
            retention_targets.update({
                'world_shift_replan_threshold_delta': -(0.02 + retention_failure_severity * 0.05),
                'high_risk_replan_threshold_delta': -(0.04 + retention_failure_severity * 0.07),
                'uncertainty_threshold_delta': -(0.04 + retention_failure_severity * 0.08),
                'verification_budget_bonus': 1.0 + (1.0 if retention_failure_severity >= 0.68 else 0.0),
            })

        new_retention_tuning: Dict[str, float] = {}
        for key in self._retention_tuning_keys():
            previous_value = float(self._retention_tuning_state.get(key, 0.0) or 0.0)
            target_value = float(retention_targets.get(key, 0.0) or 0.0)
            if key in {'branch_budget_bonus', 'verification_budget_bonus'}:
                new_retention_tuning[key] = self._clamp(
                    previous_value * 0.78 + target_value * 0.22,
                    minimum=0.0,
                    maximum=2.0,
                    default=previous_value,
                )
            else:
                new_retention_tuning[key] = self._clamp(
                    previous_value * 0.82 + target_value * 0.18,
                    minimum=-0.25,
                    maximum=0.25,
                    default=previous_value,
                )

        new_policy_state = {
            'retrieval_aggressiveness': self._clamp(
                (self._policy_state['retrieval_aggressiveness'] * 0.45)
                + (mode_targets['retrieval_aggressiveness'] * 0.55)
                + control_deltas['retrieval_aggressiveness'],
                default=self._policy_state['retrieval_aggressiveness'],
            ),
            'probe_bias': self._clamp(
                (self._policy_state['probe_bias'] * 0.45)
                + (mode_targets['probe_bias'] * 0.55)
                + control_deltas['probe_bias'],
                default=self._policy_state['probe_bias'],
            ),
            'planner_bias': self._clamp(
                (self._policy_state['planner_bias'] * 0.45)
                + (mode_targets['planner_bias'] * 0.55)
                + control_deltas['planner_bias'],
                default=self._policy_state['planner_bias'],
            ),
        }
        new_representation_state = {
            'retrieval_aggressiveness': self._clamp(
                (self._representation_state['retrieval_aggressiveness'] * 0.45)
                + (mode_targets['retrieval_aggressiveness'] * 0.55)
                + representation_deltas['retrieval_aggressiveness'],
                default=self._representation_state['retrieval_aggressiveness'],
            ),
            'probe_bias': self._clamp(
                (self._representation_state['probe_bias'] * 0.45)
                + (mode_targets['probe_bias'] * 0.55)
                + representation_deltas['probe_bias'],
                default=self._representation_state['probe_bias'],
            ),
            'retrieval_pressure': self._representation_state['retrieval_pressure'],
        }
        if enable_representation_adaptation and not freeze_retrieval_pressure:
            new_representation_state['retrieval_pressure'] = self._clamp(
                (self._representation_state['retrieval_pressure'] * 0.45)
                + (mode_targets['retrieval_pressure'] * 0.55)
                + representation_deltas['retrieval_pressure'],
                default=self._representation_state['retrieval_pressure'],
            )

        updated = (
            any(abs(new_policy_state[key] - self._policy_state[key]) >= 0.01 for key in new_policy_state)
            or any(abs(new_representation_state[key] - self._representation_state[key]) >= 0.01 for key in new_representation_state)
            or any(abs(new_retention_tuning[key] - previous_retention_tuning.get(key, 0.0)) >= 0.01 for key in new_retention_tuning)
            or abs(reward_delta) >= 0.01
            or previous_mode != strategy_mode
            or any(
                abs(float(previous_strategy_snapshot.get(key, 0.5) or 0.5) - float(strategy_state.get(key, previous_strategy_snapshot.get(key, 0.5)) or 0.5)) >= 0.01
                for key in ('verification_bias', 'risk_tolerance', 'recovery_bias', 'stability_bias')
            )
        )

        self._policy_state = new_policy_state
        self._representation_state = new_representation_state
        self._retention_tuning_state = new_retention_tuning
        self._last_episode_reward = episode_reward
        self._state_version += 1
        self._runtime_hints = {'retrieval_delta': 0.0, 'probe_delta': 0.0, 'planner_delta': 0.0}

        policy_patch = {
            'content': self._policy_object_content(),
            'confidence': self._clamp(0.72 + abs(reward_delta) * 0.08, minimum=0.35, maximum=0.95, default=0.72),
            'retrieval_tags': ['meta_control', 'policy_profile', 'adaptive'],
            'memory_metadata': {
                'profile_type': 'policy_profile',
                'state_version': int(self._state_version),
                'last_episode_reward': float(self._last_episode_reward),
                'strategy_mode': str(strategy_mode),
                'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
                'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
                'trigger_evidence': list(trigger_evidence),
            },
        }
        representation_patch = {
            'content': self._representation_object_content(),
            'confidence': self._clamp(0.7 + abs(reward_delta) * 0.06, minimum=0.35, maximum=0.95, default=0.7),
            'retrieval_tags': ['meta_control', 'representation_profile', 'adaptive'],
            'memory_metadata': {
                'profile_type': 'representation_profile',
                'state_version': int(self._state_version),
                'last_episode_reward': float(self._last_episode_reward),
                'strategy_mode': str(strategy_mode),
                'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
                'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
                'trigger_evidence': list(trigger_evidence),
            },
        }
        policy_result = self._store.update_fields(
            self.policy_profile_object_id,
            patch=policy_patch,
            reason='meta_control_policy_update',
            evidence_ids=[],
        )
        representation_result = self._store.update_fields(
            self.representation_profile_object_id,
            patch=representation_patch,
            reason='meta_control_representation_update',
            evidence_ids=[],
        )
        if not policy_result:
            self._append_fallback_event('policy_profile_update_rejected')
        if not representation_result:
            self._append_fallback_event('representation_profile_update_rejected')

        adaptation_report = {
            'control_layer': {
                'updated': bool(updated),
                'deltas': {key: float(value) for key, value in control_deltas.items()},
            },
            'representation_layer': {
                'updated': bool(enable_representation_adaptation and not freeze_retrieval_pressure),
                'deltas': {key: float(value) for key, value in representation_deltas.items()},
                'freeze_retrieval_pressure': bool(freeze_retrieval_pressure),
            },
            'strategy_layer': {
                'previous_mode': previous_mode,
                'strategy_mode': strategy_mode,
                'dominant_signal': str(self._strategy_state.get('dominant_signal', '') or ''),
                'last_transition_reason': str(self._strategy_state.get('last_transition_reason', '') or ''),
                'mode_scores': dict(strategy_state.get('mode_scores', {})),
                'signal_state': dict(self._signal_state),
            },
            'retention_layer': {
                'updated': any(abs(new_retention_tuning[key] - previous_retention_tuning.get(key, 0.0)) >= 0.01 for key in new_retention_tuning),
                'tuning_state': dict(self._retention_tuning_state),
                'failure_type': retention_failure_type,
                'failure_severity': float(retention_failure_severity),
            },
        }
        return {
            'updated': bool(updated),
            'result': 'committed' if policy_result and representation_result else 'partial_commit',
            'reward_delta': float(reward_delta),
            'policy_state': dict(self._policy_state),
            'representation_state': dict(self._representation_state),
            'strategy_state': dict(self._strategy_state),
            'retention_tuning': dict(self._retention_tuning_state),
            'policy_update_trace': {
                'trigger_evidence': list(trigger_evidence),
                'effect_evaluation': {
                    'updated': bool(updated),
                    'policy_state': dict(self._policy_state),
                    'representation_state': dict(self._representation_state),
                    'strategy_state': dict(self._strategy_state),
                    'retention_tuning': dict(self._retention_tuning_state),
                },
            },
            'policy_profile_object_id': str(self.policy_profile_object_id),
            'representation_profile_object_id': str(self.representation_profile_object_id),
            'adaptation_report': adaptation_report,
        }

    def apply_runtime_hints(self, **kwargs) -> None:
        self._runtime_hints['retrieval_delta'] = max(
            -0.2,
            min(0.2, self._runtime_hints['retrieval_delta'] + float(kwargs.get('retrieval_delta', 0.0) or 0.0)),
        )
        self._runtime_hints['probe_delta'] = max(
            -0.2,
            min(0.2, self._runtime_hints['probe_delta'] + float(kwargs.get('probe_delta', 0.0) or 0.0)),
        )
        self._runtime_hints['planner_delta'] = max(
            -0.2,
            min(0.2, self._runtime_hints['planner_delta'] + float(kwargs.get('planner_delta', 0.0) or 0.0)),
        )

    def describe_state(self) -> Dict[str, Any]:
        return {
            'policy_state': dict(self._policy_state),
            'representation_state': dict(self._representation_state),
            'strategy_state': dict(self._strategy_state),
            'retention_tuning': dict(self._retention_tuning_state),
            'signal_state': dict(self._signal_state),
            'runtime_hints': dict(self._runtime_hints),
            'policy_profile_object_id': str(self.policy_profile_object_id),
            'representation_profile_object_id': str(self.representation_profile_object_id),
        }


class TransferTraceEvent:
    """Single event in the transfer applicability trace."""
    def __init__(self, tick: int, episode: int, event_type: str,
                 object_id: Optional[str] = None, fn: Optional[str] = None,
                 influence: str = 'none', arm_mode: str = 'full'):
        self.tick = tick
        self.episode = episode
        self.event_type = event_type
        self.object_id = object_id
        self.fn = fn
        self.influence = influence
        self.arm_mode = arm_mode
        self.timestamp = time.time()


def _extract_candidate_function_name(obj: dict) -> Optional[str]:
    """
    T2-P2 FIX: Extract function_name from candidate object with correct field priority.
    
    Priority:
    1. obj['payload']['tool_args']['function_name']
    2. obj['content']['tool_args']['function_name']
    3. obj['content']['function_name']
    4. obj['function_name']
    4. None (representation candidates and invalid candidates)
    
    Returns None for invalid/non-dict inputs instead of falling back to 'wait'.
    """
    if not isinstance(obj, dict):
        return None
    
    # Representation candidates have type='representation' — return None (will be filtered)
    content = obj.get('content', {})
    if isinstance(content, dict) and content.get('type') == 'representation':
        return None

    payload = obj.get('payload', {})
    payload_tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
    fn = payload_tool_args.get('function_name', '')
    if fn:
        return fn
    
    # Try nested tool_args path first
    tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
    fn = tool_args.get('function_name', '')
    if fn:
        return fn
    
    # Try direct content function_name
    fn = content.get('function_name', '')
    if fn:
        return fn
    
    # Try top-level function_name
    fn = obj.get('function_name', '')
    if fn:
        return fn
    
    return None


class TransferTraceLogger:
    """P0-1: Logs transfer applicability trace per decision cycle."""
    def __init__(self):
        self._cycles: List[Dict] = []
        self._current_cycle: Dict = {}
        self._cycle_count = 0

    def next_decision_cycle(self) -> Dict:
        self._cycle_count += 1
        self._current_cycle = {
            'cycle_id': self._cycle_count,
            'events': [],
            'arm_mode': 'full',
        }
        self._cycles.append(self._current_cycle)
        return self._current_cycle

    def emit(self, event: TransferTraceEvent):
        self._current_cycle.setdefault('events', []).append(vars(event))

    def get_cycles(self) -> List[Dict]:
        return list(self._cycles)


class TestOpportunityEntry:
    """One test opportunity with pre/post metrics."""
    def __init__(self, tick: int, episode: int, test_id: str,
                 pre_reward: float, post_reward: float,
                 entropy_before: float, entropy_after: float,
                 information_gain: float):
        self.tick = tick
        self.episode = episode
        self.test_id = test_id
        self.pre_reward = pre_reward
        self.post_reward = post_reward
        self.entropy_before = entropy_before
        self.entropy_after = entropy_after
        self.information_gain = information_gain


class TestOpportunityAudit:
    """P0-2: Tracks pre-saturation test opportunity cost and information gain."""
    def __init__(self):
        self.entries: List[TestOpportunityEntry] = []

    def add(self, entry: TestOpportunityEntry):
        self.entries.append(entry)

    def get_net_info_gain(self) -> float:
        return sum(e.information_gain for e in self.entries)

    def get_entries(self) -> List[TestOpportunityEntry]:
        return list(self.entries)


# =============================================================================
# Episodic Retriever - P0-1 Core
# =============================================================================

class EpisodicRetriever:
    """
    P0-1: Build query → retrieve → surface → consume trace.

    This is the base retriever. Arm distortions are applied via subclasses
    that override surface() or arm_evaluate().
    """
    def __init__(self, object_store, seed: int = 0):
        self._store = object_store
        self._rng = random.Random(seed)
        self._consumed_fns: Set[str] = set()
        self._transfer_log: List[dict] = []

    def build_query(self, obs: dict, ctx: dict) -> RetrievalQuery:
        """Build an explicit retrieval query from observation."""
        query_text = ctx.get('phase', 'active')
        fn_list = []
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else []
        for fn in discovered:
            query_text += f" {fn}"
        plan_target = ctx.get('plan_target_function')
        if plan_target:
            query_text += f" plan_target:{plan_target}"
        failure_modes = ctx.get('failure_modes', []) or []
        for mode in failure_modes[:3]:
            query_text += f" failure:{mode}"
        world_focus = ctx.get('world_focus_variables', []) or []
        for variable in world_focus[:3]:
            query_text += f" belief:{variable}"
        required_probes = ctx.get('world_model_required_probes', []) or []
        for probe in required_probes[:3]:
            query_text += f" probe:{probe}"
        hidden_phase = str(ctx.get('world_model_hidden_phase', '') or '')
        if hidden_phase:
            query_text += f" hidden_phase:{hidden_phase}"
        dominant_branch_id = str(ctx.get('world_model_dominant_branch_id', '') or '')
        if dominant_branch_id:
            query_text += f" latent_branch:{dominant_branch_id}"
        query = RetrievalQuery(query_text, ctx, ctx.get('tick', 0), ctx.get('episode', 1))
        return query

    def retrieve(self, query: RetrievalQuery) -> RetrieveResult:
        """Search object store and return ranked candidates."""
        all_objs = self._store.retrieve(sort_by='confidence', limit=50)
        candidates = []
        for i, obj in enumerate(all_objs):
            obj_id = obj.get('object_id', obj.get('id', ''))
            if obj_id in query.exclude_ids:
                continue
            relevance = self._compute_relevance(query, obj)
            candidates.append(RetrievedCandidate(
                object_id=obj_id,
                object=obj,
                relevance_score=relevance,
                rank=i,
            ))
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return RetrieveResult(
            candidates=candidates,
            selected_ids=[c.object_id for c in candidates[:5]],
            action_influence='direct' if candidates else 'none',
            contract={
                'selected_ids': [c.object_id for c in candidates[:5]],
                'action_influence': 'direct' if candidates else 'none',
                'candidate_count': len(candidates),
            },
        )

    def _compute_relevance(self, query: RetrievalQuery, obj: dict) -> float:
        """State-conditioned retrieval relevance scoring."""
        content = obj.get('content', {})
        tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
        fn = tool_args.get('function_name', '')
        retrieval_tags = obj.get('retrieval_tags', []) if isinstance(obj.get('retrieval_tags', []), list) else []
        score = 0.0

        for kw in query.query_text.split():
            if kw in fn:
                score += 1.0
            if kw.startswith('failure:'):
                mode = kw.split(':', 1)[1]
                if any(mode in str(tag) for tag in retrieval_tags):
                    score += 1.2
            if kw.startswith('belief:'):
                variable = kw.split(':', 1)[1]
                if variable and variable in str(content):
                    score += 0.8
            if kw.startswith('probe:'):
                probe = kw.split(':', 1)[1]
                if probe and (probe in str(content) or any(probe in str(tag) for tag in retrieval_tags)):
                    score += 1.0
            if kw.startswith('hidden_phase:'):
                hidden_phase = kw.split(':', 1)[1]
                if hidden_phase and hidden_phase in str(content):
                    score += 0.6
            if kw.startswith('latent_branch:'):
                branch_id = kw.split(':', 1)[1]
                if branch_id and branch_id in str(content):
                    score += 0.7
            if kw.startswith('plan_target:'):
                target = kw.split(':', 1)[1]
                if target and target == fn:
                    score += 1.5

        ctx = query.context if isinstance(query.context, dict) else {}
        if ctx.get('phase') and str(ctx.get('phase')) in str(content):
            score += 0.5

        rel_obj = obj.get('related_objects', [])
        focus_objects = ctx.get('focus_object_ids', []) if isinstance(ctx.get('focus_object_ids', []), list) else []
        if isinstance(rel_obj, list) and focus_objects:
            overlap = len(set(str(x) for x in rel_obj) & set(str(x) for x in focus_objects))
            score += min(1.0, overlap * 0.5)

        recency_bonus = 0.0
        try:
            tick = int(obj.get('tick', 0) or 0)
            recency_bonus = max(0.0, 0.3 - abs((query.tick or 0) - tick) * 0.01)
        except Exception:
            recency_bonus = 0.0
        return score + obj.get('confidence', 0.5) + recency_bonus

    def surface(self, result: RetrieveResult, top_k: int = 5, consumed_fns: Set[str] = None) -> List[RetrievedCandidate]:
        """Present top-k candidates, excluding already-consumed functions.

        Task 1.3: Representations deep integration.
        Adds representation cards as candidates based on runtime activation.
        Cards with positive helpful_ratio affect future decisions.
        """
        consumed = consumed_fns or set()
        surfaced = []

        # Add retrieval candidates first
        for tc in result.candidates:
            content = tc.object.get('content', {})
            tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
            fn = tool_args.get('function_name', '')
            if fn and fn in consumed:
                continue
            tc.action_influence = result.action_influence
            surfaced.append(tc)
            if len(surfaced) >= top_k:
                break

        # Task 1.3: Add representation cards as candidates based on activation
        # Cards that were helpful in runtime should influence future decisions
        if len(surfaced) < top_k:
            warehouse = get_warehouse()
            runtime_store = get_runtime_store()

            for card in warehouse.iter_cards():
                if len(surfaced) >= top_k:
                    break

                rep_candidate = _build_representation_candidate(
                    card,
                    runtime_store=runtime_store,
                    rank=len(surfaced),
                )
                if rep_candidate is not None:
                    surfaced.append(rep_candidate)

        return surfaced

    def consume(self, action: dict, result: dict, query: RetrievalQuery):
        """Record retrieval influence on action."""
        fn = action.get('payload', {}).get('tool_args', {}).get('function_name', '')
        if fn:
            self._consumed_fns.add(fn)
        self._transfer_log.append({
            'tick': query.tick,
            'episode': query.episode,
            'fn': fn,
            'action': str(action)[:80],
        })

    def arm_evaluate(self, surfaced: List[RetrievedCandidate], base_action: dict, obs: dict) -> Tuple[dict, dict]:
        """
        Arm evaluation entry point. Called at action evaluation time.

        Base implementation: no arm distortion (arm='base').
        Subclasses override to apply distortions.
        """
        return base_action, {'arm': 'base'}

    def get_transfer_log(self) -> List[dict]:
        return list(self._transfer_log)


class _BaseArmEpisodicRetriever(EpisodicRetriever):
    """
    Base for arm variants. Overrides arm_evaluate() to apply distortions.
    surface() is NOT modified - retrieval and surfacing are identical across all arms.
    """
    pass


class _WrongBindingEpisodicRetriever(_BaseArmEpisodicRetriever):
    """Arm: Corrupt skill_ids to wrong function names at surface time."""
    _FN_TO_FAMILY = {
        'compute_stats': 'stats',
        'filter_by_predicate': 'filter',
        'array_transform': 'transform',
        'join_tables': 'join',
        'aggregate_group': 'aggregate',
    }
    _BINDING_TABLE = {
        'compute_stats': ('filter_by_predicate', 'CROSS_FAMILY_filter'),
        'filter_by_predicate': ('join_tables', 'CROSS_FAMILY_join'),
        'array_transform': ('compute_stats', 'CROSS_FAMILY_stats'),
        'join_tables': ('aggregate_group', 'CROSS_FAMILY_aggregate'),
        'aggregate_group': ('array_transform', 'CROSS_FAMILY_transform'),
    }
    _SECONDARY_BINDINGS = {
        'compute_stats': ('array_transform', 'CROSS_FAMILY_transform'),
        'filter_by_predicate': ('compute_stats', 'CROSS_FAMILY_stats'),
        'array_transform': ('filter_by_predicate', 'CROSS_FAMILY_filter'),
        'join_tables': ('array_transform', 'CROSS_FAMILY_transform'),
        'aggregate_group': ('join_tables', 'CROSS_FAMILY_join'),
    }

    def arm_evaluate(self, surfaced: List[RetrievedCandidate], base_action: dict, obs: dict) -> Tuple[dict, dict]:
        if not surfaced:
            return base_action, {'arm': 'base'}

        tc = surfaced[0]
        content = tc.object.get('content', {})
        tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
        tfn = tool_args.get('function_name', '')

        # Check if already discovered
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = set(api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else [])
        if tfn in discovered:
            return base_action, {'arm': 'base'}

        # Get wrong binding target
        primary = self._BINDING_TABLE.get(tfn)
        if not primary:
            return base_action, {'arm': 'base'}

        wrong_fn, binding_target = primary
        if wrong_fn in discovered:
            secondary = self._SECONDARY_BINDINGS.get(tfn)
            if secondary:
                wrong_fn, binding_target = secondary
            if wrong_fn in discovered:
                return base_action, {'arm': 'base'}

        # Build corrupted action
        original_kwargs = tool_args.get('kwargs', {})
        corrupted_action = {
            'kind': 'call_tool',
            'payload': {
                'tool_name': 'call_hidden_function',
                'tool_args': {
                    'function_name': wrong_fn,
                    'kwargs': original_kwargs,
                },
            },
        }
        return corrupted_action, {
            'arm': 'wrong_binding',
            'binding_target': binding_target,
            'original_function': tfn,
            'wrong_function': wrong_fn,
        }


class _LocalOnlyEpisodicRetriever(_BaseArmEpisodicRetriever):
    """Arm: Restrict to same function family only."""
    _FN_TO_FAMILY = {
        'compute_stats': 'stats',
        'filter_by_predicate': 'filter',
        'array_transform': 'transform',
        'join_tables': 'join',
        'aggregate_group': 'aggregate',
    }

    def arm_evaluate(self, surfaced: List[RetrievedCandidate], base_action: dict, obs: dict) -> Tuple[dict, dict]:
        if not surfaced:
            return base_action, {'arm': 'base'}

        base_fn = base_action.get('payload', {}).get('tool_args', {}).get('function_name', '')
        base_family = self._FN_TO_FAMILY.get(base_fn, '')

        if not base_family:
            fallback_content = surfaced[0].object.get('content', {})
            fallback_tool_args = fallback_content.get('tool_args', {}) if isinstance(fallback_content, dict) else {}
            fallback_fn = fallback_tool_args.get('function_name', '')
            fallback_kwargs = fallback_tool_args.get('kwargs', {})
            if not isinstance(fallback_kwargs, dict):
                fallback_kwargs = {}
            fallback_action = {
                'kind': 'call_tool',
                'payload': {
                    'tool_name': 'call_hidden_function',
                    'tool_args': {
                        'function_name': fallback_fn,
                        'kwargs': fallback_kwargs,
                    },
                },
            }
            return fallback_action, {'arm': 'base', 'fallback': 'unknown_family'}

        # Filter surfaced to same family
        same_family = []
        for tc in surfaced:
            content = tc.object.get('content', {})
            tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
            tfn = tool_args.get('function_name', '')
            if self._FN_TO_FAMILY.get(tfn, '') == base_family:
                same_family.append(tc)

        if same_family:
            same_content = same_family[0].object.get('content', {})
            same_tool_args = same_content.get('tool_args', {}) if isinstance(same_content, dict) else {}
            same_fn = same_tool_args.get('function_name', '')
            same_kwargs = same_tool_args.get('kwargs', {})
            if not isinstance(same_kwargs, dict):
                same_kwargs = {}

            action = {
                'kind': 'call_tool',
                'payload': {
                    'tool_name': 'call_hidden_function',
                    'tool_args': {
                        'function_name': same_fn,
                        'kwargs': same_kwargs,
                    },
                },
            }
            return action, {
                'arm': 'local_only',
                'blocked_cross_family': len(surfaced) - len(same_family),
                'base_family': base_family,
            }

        return base_action, {
            'arm': 'local_only',
            'fallback': 'no_same_family',
            'base_family': base_family,
        }


class _ShuffledEpisodicRetriever(_BaseArmEpisodicRetriever):
    """Arm: Shuffle the mapping between surfaced candidates and their function bindings."""
    def arm_evaluate(self, surfaced: List[RetrievedCandidate], base_action: dict, obs: dict) -> Tuple[dict, dict]:
        if len(surfaced) < 2:
            return base_action, {'arm': 'base'}

        shuffled = surfaced.copy()
        self._rng.shuffle(shuffled)

        # Take first shuffled candidate but use original base_action's function
        tc = shuffled[0]
        return base_action, {
            'arm': 'shuffled',
            'shuffled_to_index': surfaced.index(tc),
        }


class _FreshEpisodicRetriever(EpisodicRetriever):
    """Arm: Disable surface-based retrieval, force pure base_generation."""
    def surface(self, result: RetrieveResult, top_k: int = 5, consumed_fns: Set[str] = None) -> List[RetrievedCandidate]:
        # Fresh arm: no retrieval candidates, representation cards still available
        # Task 1.3: representation cards can still influence even in fresh mode
        if len(result.candidates) == 0:
            # When no retrieval candidates, representation cards might provide signal
            warehouse = get_warehouse()
            runtime_store = get_runtime_store()

            candidates = []
            for card in list(warehouse.iter_cards())[:2]:
                rep_candidate = _build_representation_candidate(
                    card,
                    runtime_store=runtime_store,
                    rank=len(candidates),
                    helpful_only=True,
                )
                if rep_candidate is not None:
                    candidates.append(rep_candidate)
            return candidates
        return []


class _NoTransferEpisodicRetriever(EpisodicRetriever):
    """Arm: Disable both retrieve and surface."""
    def retrieve(self, query: RetrievalQuery) -> RetrieveResult:
        return RetrieveResult(
            candidates=[],
            selected_ids=[],
            action_influence='none',
            contract={'selected_ids': [], 'action_influence': 'none', 'candidate_count': 0},
        )

    def surface(self, result: RetrieveResult, top_k: int = 5, consumed_fns: Set[str] = None) -> List[RetrievedCandidate]:
        return []


# =============================================================================
# NovelAPI Committer (Step 10)
# =============================================================================

class NovelAPICommitter:
    """Step 10 committer - commits validated evidence packets to ObjectStore."""
    def __init__(self, object_store: Optional[ObjectStore] = None):
        self._store = object_store or ObjectStore()
        self._log: List[dict] = []

    @property
    def object_store(self):
        return self._store

    def commit(self, evidence_packets: List[Dict], top_k: int = 5) -> List[str]:
        """
        Commit evidence packets as formal objects in the store.
        
        Args:
            evidence_packets: List of packets OR tuples of (packet, decision)
                - If tuples: (packet, decision) where decision is GovernanceDecision
                - If plain packets: will use ACCEPT_NEW as default
        Returns:
            List of committed object IDs.
        """
        committed = []
        passthrough_keys = {
            'memory_type', 'memory_layer', 'retrieval_tags', 'memory_metadata',
            'trigger', 'source', 'episode', 'trigger_source', 'trigger_episode',
            'provenance', 'source_module', 'source_stage',
            'object_type', 'family', 'summary', 'structured_payload',
            'applicability', 'failure_conditions', 'commit_epoch',
            'version', 'supersedes', 'reopened_from', 'surface_priority',
            'supporting_evidence', 'contradicting_evidence',
            'source_family', 'target_family', 'reuse_evidence',
            'identity_profile', 'episode_refs', 'continuity_markers',
        }
        for item in evidence_packets[:top_k]:
            # Handle both tuple (packet, decision) and plain packet
            if isinstance(item, tuple) and len(item) == 2:
                pkt, decision = item
            else:
                pkt = item
                decision = GovernanceDecision(ACCEPT_NEW, "Step 10 commit")
            
            if hasattr(pkt, 'content'):
                pkt = {
                    'content': pkt.content,
                    'confidence': getattr(pkt, 'confidence', 0.5),
                    'content_hash': getattr(pkt, 'content_hash', ''),
                    'evidence_ids': [getattr(pkt, 'evidence_id', '')],
                    'tick': getattr(pkt, 'tick', 0),
                    'episode': getattr(pkt, 'episode', 1),
                }
            proposal = {
                'content': pkt.get('content', pkt),
                'confidence': pkt.get('confidence', 0.5),
                'content_hash': pkt.get('content_hash', ''),
            }
            for key in passthrough_keys:
                if key in pkt:
                    proposal[key] = pkt[key]
            
            # Use actual governance decision, not hardcoded ACCEPT_NEW
            governance_decision = decision.decision if hasattr(decision, 'decision') else ACCEPT_NEW
            
            # T0-P1 FIX: When MERGE_UPDATE_EXISTING, propagate decision.object_id as existing_object_id
            proposal_to_add = dict(proposal)
            if governance_decision == MERGE_UPDATE_EXISTING:
                existing_id = decision.object_id if hasattr(decision, 'object_id') and decision.object_id else proposal.get('existing_object_id', '')
                proposal_to_add['existing_object_id'] = existing_id
                additional_content = {}
                if isinstance(proposal.get('content', {}), dict):
                    additional_content.update({
                        k: v for k, v in proposal.get('content', {}).items()
                        if k not in ('object_id', 'status', 'evidence_ids', 'created_at')
                    })
                for key in ('retrieval_tags', 'memory_metadata'):
                    if key in proposal:
                        additional_content[key] = proposal[key]
                if additional_content:
                    proposal_to_add['additional_content'] = additional_content
            
            obj_id = self._store.add(proposal_to_add, governance_decision, pkt.get('evidence_ids', []))
            if obj_id:
                committed.append(obj_id)
                self._log.append({
                    'object_id': obj_id,
                    'tick': pkt.get('tick'),
                    'episode': pkt.get('episode'),
                })
        return committed

    def get_committed_objects(self, top_k: int = 50) -> List[dict]:
        return self._store.retrieve(sort_by='confidence', limit=top_k)

    def get_log(self) -> List[dict]:
        return list(self._log)

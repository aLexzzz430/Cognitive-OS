from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class OutcomeSignal:
    episode: int
    tick: int
    function_name: str
    task_family: str
    success: bool
    reward: float
    recovery_triggered: bool
    recovery_cost: float
    prediction_mismatch: float
    recovery_type: str = ''
    phase: str = 'unknown'
    observation_mode: str = 'unknown'
    resource_band: str = 'normal'
    failure_strategy_profile: Dict[str, Any] | None = None
    global_failure_strategy: Dict[str, Any] | None = None
    world_model_required_probes: List[str] | None = None
    world_model_probe_pressure: float = 0.0
    world_model_latent_instability: float = 0.0
    world_model_dominant_branch_id: str = ''
    world_model_anchor_functions: List[str] | None = None
    world_model_risky_functions: List[str] | None = None
    world_model_required_probe_match: bool = False
    world_model_anchor_match: bool = False
    world_model_risky_match: bool = False
    selected_source: str = ''
    governance_reason: str = ''
    retention_failure_type: str = ''
    retention_failure_severity: float = 0.0
    retention_failure_context: Dict[str, Any] | None = None


@dataclass
class LearningUpdate:
    update_type: str
    key: str
    delta: float
    confidence: float
    evidence: Dict[str, Any]

    def to_proposal_content(self) -> Dict[str, Any]:
        return {
            'type': 'learning_update',
            'update_type': self.update_type,
            'key': self.key,
            'delta': float(self.delta),
            'confidence': float(self.confidence),
            'evidence': dict(self.evidence),
            'schema_version': 'learning_update_v1',
        }


@dataclass
class TraceCreditAssignment:
    object_id: str
    credit_amount: float
    evidence_trace_ids: List[str]
    based_on: str


class CreditAssignment:
    """Maps post-commit outcome signals into persistent learning updates."""

    _OBJECT_ID_KEYS = {'object_id'}
    _OBJECT_ID_LIST_KEYS = {'selected_ids', 'surfaced_from'}

    def assign_from_outcome(self, signal: OutcomeSignal) -> List[LearningUpdate]:
        fn_name = str(signal.function_name or 'wait')
        family = str(signal.task_family or 'unknown')
        mismatch = max(0.0, min(1.0, float(signal.prediction_mismatch or 0.0)))
        recovery_cost = max(0.0, min(1.0, float(signal.recovery_cost or 0.0)))
        required_probes = self._string_list(signal.world_model_required_probes or [])
        dominant_anchor_functions = self._string_list(signal.world_model_anchor_functions or [])
        dominant_risky_functions = self._string_list(signal.world_model_risky_functions or [])
        probe_pressure = max(0.0, min(1.0, float(signal.world_model_probe_pressure or 0.0)))
        latent_instability = max(0.0, min(1.0, float(signal.world_model_latent_instability or 0.0)))
        dominant_branch_id = str(signal.world_model_dominant_branch_id or '').strip()
        required_probe_match = bool(signal.world_model_required_probe_match or fn_name in set(required_probes))
        anchor_match = bool(signal.world_model_anchor_match or fn_name in set(dominant_anchor_functions))
        risky_match = bool(signal.world_model_risky_match or fn_name in set(dominant_risky_functions))
        competition_active = probe_pressure >= 0.34 or latent_instability >= 0.55

        success_delta = 0.18 if signal.success else -0.16
        if signal.success and competition_active and (required_probe_match or anchor_match):
            success_delta += 0.04 + probe_pressure * 0.03 + latent_instability * 0.03
        elif (not signal.success) and competition_active and risky_match:
            success_delta -= 0.05 + latent_instability * 0.05
        selector_delta = success_delta - (0.08 * mismatch) - (0.06 * recovery_cost)
        selector_conf = max(0.25, min(0.95, 0.55 + abs(selector_delta) * 0.8))

        agenda_delta = (0.10 if signal.success else -0.08) - (0.06 * mismatch)
        agenda_conf = max(0.2, min(0.9, 0.5 + abs(agenda_delta) * 0.7))

        recovery_gain = 0.0
        if signal.recovery_triggered:
            recovery_gain = (0.18 if signal.success else 0.06) - (0.12 * recovery_cost)
        recovery_conf = max(0.25, min(0.9, 0.45 + abs(recovery_gain) * 0.9))

        updates = [
            LearningUpdate(
                update_type='selector_bias',
                key=fn_name,
                delta=selector_delta,
                confidence=selector_conf,
                evidence={
                    'episode': signal.episode,
                    'tick': signal.tick,
                    'reward': signal.reward,
                    'success': signal.success,
                    'prediction_mismatch': mismatch,
                    'recovery_cost': recovery_cost,
                },
            ),
            LearningUpdate(
                update_type='agenda_prior',
                key=family,
                delta=agenda_delta,
                confidence=agenda_conf,
                evidence={
                    'episode': signal.episode,
                    'tick': signal.tick,
                    'task_family': family,
                    'success': signal.success,
                },
            ),
        ]

        if signal.recovery_triggered and signal.recovery_type:
            updates.append(
                LearningUpdate(
                    update_type='recovery_shortcut',
                    key=str(signal.recovery_type),
                    delta=recovery_gain,
                    confidence=recovery_conf,
                    evidence={
                        'episode': signal.episode,
                        'tick': signal.tick,
                        'recovery_type': signal.recovery_type,
                        'recovery_cost': recovery_cost,
                        'success': signal.success,
                    },
                )
            )

        context_key = self._failure_preference_context_key(signal)
        strategy_profile = dict(signal.failure_strategy_profile or {}) if isinstance(signal.failure_strategy_profile, dict) else {}
        global_profile = dict(signal.global_failure_strategy or {}) if isinstance(signal.global_failure_strategy, dict) else {}
        preferred_verification_functions = self._string_list(
            required_probes if competition_active else [],
            dominant_anchor_functions if latent_instability >= 0.55 else [],
            strategy_profile.get('preferred_verification_functions', []),
            global_profile.get('preferred_verification_functions', []),
        )
        preferred_fallback_functions = self._string_list(
            dominant_anchor_functions if latent_instability >= 0.55 else [],
            strategy_profile.get('preferred_fallback_functions', []),
            global_profile.get('preferred_fallback_functions', []),
        )
        blocked_action_classes = self._string_list(
            strategy_profile.get('blocked_action_classes', []),
            global_profile.get('blocked_action_classes', []),
            [fn_name] if (competition_active and risky_match and not signal.success) else [],
        )
        strategy_mode_hint = str(
            strategy_profile.get(
                'strategy_mode_hint',
                global_profile.get('strategy_mode_hint', 'recover' if not signal.success else 'balanced'),
            ) or ('recover' if not signal.success else 'balanced')
        )
        branch_budget_hint = self._bounded_int(
            strategy_profile.get('branch_budget_hint', global_profile.get('branch_budget_hint', 0)),
            minimum=0,
            maximum=4,
            default=0,
        )
        verification_budget_hint = self._bounded_int(
            strategy_profile.get('verification_budget_hint', global_profile.get('verification_budget_hint', 0)),
            minimum=0,
            maximum=3,
            default=0,
        )
        if not signal.success:
            failure_pref_delta = 0.22 + (0.10 * recovery_cost) + (0.08 * mismatch)
        elif signal.recovery_triggered or strategy_profile or global_profile:
            failure_pref_delta = 0.16 - (0.05 * mismatch) - (0.04 * recovery_cost)
        else:
            failure_pref_delta = -0.22
        failure_pref_delta = max(-0.8, min(0.8, failure_pref_delta))
        failure_pref_conf = max(0.2, min(0.95, 0.45 + abs(failure_pref_delta) * 0.85))
        updates.append(
            LearningUpdate(
                update_type='failure_preference_policy',
                key=context_key,
                delta=failure_pref_delta,
                confidence=failure_pref_conf,
                evidence={
                    'episode': signal.episode,
                    'tick': signal.tick,
                    'function_name': fn_name,
                    'task_family': family,
                    'phase': str(signal.phase or 'unknown'),
                    'observation_mode': str(signal.observation_mode or 'unknown'),
                    'resource_band': str(signal.resource_band or 'normal'),
                    'context_key': context_key,
                    'strategy_mode_hint': strategy_mode_hint,
                    'branch_budget_hint': branch_budget_hint,
                    'verification_budget_hint': verification_budget_hint,
                    'safe_fallback_class': str(
                        strategy_profile.get(
                            'safe_fallback_class',
                            global_profile.get('safe_fallback_class', 'wait'),
                        ) or 'wait'
                    ),
                    'preferred_verification_functions': preferred_verification_functions,
                    'preferred_fallback_functions': preferred_fallback_functions,
                    'blocked_action_classes': blocked_action_classes,
                    'source_action': str(
                        global_profile.get(
                            'persistence_source_action',
                            strategy_profile.get('action_type', fn_name),
                        ) or fn_name
                    ),
                    'success': bool(signal.success),
                    'reward': float(signal.reward),
                    'recovery_triggered': bool(signal.recovery_triggered),
                    'required_probes': required_probes,
                    'probe_pressure': probe_pressure,
                    'latent_instability': latent_instability,
                    'dominant_branch_id': dominant_branch_id,
                    'dominant_anchor_functions': dominant_anchor_functions,
                    'dominant_risky_functions': dominant_risky_functions,
                    'required_probe_match': required_probe_match,
                    'anchor_match': anchor_match,
                    'risky_match': risky_match,
                    'selected_source': str(signal.selected_source or ''),
                    'governance_reason': str(signal.governance_reason or ''),
                },
            )
        )

        retention_type = str(signal.retention_failure_type or '').strip().lower()
        retention_severity = max(0.0, min(1.0, float(signal.retention_failure_severity or 0.0)))
        retention_context = dict(signal.retention_failure_context or {}) if isinstance(signal.retention_failure_context, dict) else {}
        if retention_type and retention_severity > 0.0:
            retention_context_key = f"{context_key}|failure={retention_type}"
            default_strategy_mode = 'recover'
            if retention_type in {'prediction_drift', 'governance_overrule_misfire'}:
                default_strategy_mode = 'verify'
            elif retention_type == 'planner_target_switch':
                default_strategy_mode = 'recover'
            branch_budget_hint = max(
                self._bounded_int(retention_context.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
                2 if retention_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0,
            )
            verification_budget_hint = max(
                self._bounded_int(retention_context.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
                2 if retention_type in {'prediction_drift', 'governance_overrule_misfire'} else (1 if retention_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0),
            )
            planner_replan_bias_delta = 0.05 + retention_severity * (
                0.18 if retention_type == 'branch_persistence_collapse'
                else 0.15 if retention_type == 'planner_target_switch'
                else 0.10
            )
            probe_bias_delta = 0.04 + retention_severity * (
                0.16 if retention_type in {'prediction_drift', 'governance_overrule_misfire'}
                else 0.10
            )
            retrieval_pressure_delta = 0.03 + retention_severity * (
                0.12 if retention_type in {'prediction_drift', 'branch_persistence_collapse'}
                else 0.08
            )
            retention_delta = 0.10 + retention_severity * (0.32 if not signal.success else 0.18)
            retention_delta = max(-0.8, min(0.8, retention_delta))
            retention_conf = max(0.30, min(0.95, 0.42 + retention_severity * 0.48 + (0.06 if not signal.success else 0.0)))
            updates.append(
                LearningUpdate(
                    update_type='retention_failure_policy',
                    key=retention_context_key,
                    delta=retention_delta,
                    confidence=retention_conf,
                    evidence={
                        'episode': signal.episode,
                        'tick': signal.tick,
                        'function_name': fn_name,
                        'task_family': family,
                        'phase': str(signal.phase or 'unknown'),
                        'observation_mode': str(signal.observation_mode or 'unknown'),
                        'resource_band': str(signal.resource_band or 'normal'),
                        'context_key': retention_context_key,
                        'base_context_key': context_key,
                        'failure_type': retention_type,
                        'severity': retention_severity,
                        'strategy_mode_hint': str(retention_context.get('strategy_mode_hint', default_strategy_mode) or default_strategy_mode),
                        'branch_budget_hint': branch_budget_hint,
                        'verification_budget_hint': verification_budget_hint,
                        'planner_replan_bias_delta': max(0.0, min(0.4, planner_replan_bias_delta)),
                        'probe_bias_delta': max(0.0, min(0.4, probe_bias_delta)),
                        'retrieval_pressure_delta': max(0.0, min(0.4, retrieval_pressure_delta)),
                        'success': bool(signal.success),
                        'reward': float(signal.reward),
                        'required_probes': required_probes,
                        'probe_pressure': probe_pressure,
                        'latent_instability': latent_instability,
                        'dominant_branch_id': dominant_branch_id,
                        'dominant_anchor_functions': dominant_anchor_functions,
                        'dominant_risky_functions': dominant_risky_functions,
                        'required_probe_match': required_probe_match,
                        'anchor_match': anchor_match,
                        'risky_match': risky_match,
                        'selected_source': str(signal.selected_source or ''),
                        'governance_reason': str(signal.governance_reason or ''),
                        **retention_context,
                    },
                )
            )

        return updates

    def _failure_preference_context_key(self, signal: OutcomeSignal) -> str:
        family = str(signal.task_family or 'unknown')
        phase = str(signal.phase or 'unknown')
        observation_mode = str(signal.observation_mode or 'unknown')
        resource_band = str(signal.resource_band or 'normal')
        return f"task_family={family}|phase={phase}|observation_mode={observation_mode}|resource_band={resource_band}"

    @staticmethod
    def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _string_list(*values: Any) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in values:
            pool = value if isinstance(value, list) else []
            for item in pool:
                text = str(item or '').strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                merged.append(text)
        return merged

    def assign_credit_from_trace(
        self,
        *,
        trace: Any,
        outcome_success: bool,
        reward: float,
    ) -> List[TraceCreditAssignment]:
        primary_ids, support_ids = self._extract_trace_credit_targets(trace)
        if not primary_ids and not support_ids:
            return []

        trace_id = self._trace_id(trace)
        reward_value = float(reward or 0.0)
        success = bool(outcome_success)

        if success:
            primary_credit = min(1.0, 0.82 + max(0.0, reward_value) * 0.08)
            support_credit = min(0.92, 0.58 + max(0.0, reward_value) * 0.05)
        else:
            primary_credit = max(-1.0, -0.55 - abs(reward_value) * 0.08)
            support_credit = max(-0.85, -0.35 - abs(reward_value) * 0.05)

        assignments: List[TraceCreditAssignment] = []
        for object_id in primary_ids:
            assignments.append(
                TraceCreditAssignment(
                    object_id=object_id,
                    credit_amount=primary_credit,
                    evidence_trace_ids=[trace_id],
                    based_on='causal_trace_selected_action',
                )
            )
        for object_id in support_ids:
            assignments.append(
                TraceCreditAssignment(
                    object_id=object_id,
                    credit_amount=support_credit,
                    evidence_trace_ids=[trace_id],
                    based_on='causal_trace_retrieval_context',
                )
            )
        return assignments

    def _extract_trace_credit_targets(self, trace: Any) -> Tuple[List[str], List[str]]:
        primary_ids = self._extract_object_ids(getattr(trace, 'final_action', {}))

        selected_candidate = self._selected_trace_candidate(trace)
        if selected_candidate is not None:
            primary_ids.extend(self._extract_object_ids(getattr(selected_candidate, 'proposed_action', {})))

        support_ids = []
        support_ids.extend(self._extract_object_ids(getattr(trace, 'retrieval_bundle_summary', {})))

        for candidate in list(getattr(trace, 'candidates', []) or []):
            if bool(getattr(candidate, 'selected', False)):
                continue
            support_ids.extend(self._extract_object_ids(getattr(candidate, 'proposed_action', {})))

        primary = self._dedupe_object_ids(primary_ids)
        support = [object_id for object_id in self._dedupe_object_ids(support_ids) if object_id not in primary]
        return primary[:4], support[:4]

    def _selected_trace_candidate(self, trace: Any) -> Any:
        selected_candidate_id = str(getattr(trace, 'selected_candidate_id', '') or '')
        for candidate in list(getattr(trace, 'candidates', []) or []):
            candidate_id = str(getattr(candidate, 'candidate_id', '') or '')
            if bool(getattr(candidate, 'selected', False)) or (selected_candidate_id and candidate_id == selected_candidate_id):
                return candidate
        return None

    def _extract_object_ids(self, payload: Any) -> List[str]:
        found: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in self._OBJECT_ID_KEYS:
                        self._append_object_id(found, child)
                        continue
                    if key in self._OBJECT_ID_LIST_KEYS and isinstance(child, (list, tuple, set)):
                        for item in child:
                            self._append_object_id(found, item)
                        continue
                    if isinstance(child, (dict, list, tuple, set)):
                        walk(child)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item)

        walk(payload)
        return found

    def _append_object_id(self, bucket: List[str], value: Any) -> None:
        object_id = str(value or '').strip()
        if object_id:
            bucket.append(object_id)

    def _dedupe_object_ids(self, object_ids: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for object_id in object_ids:
            normalized = str(object_id or '').strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _trace_id(self, trace: Any) -> str:
        episode = int(getattr(trace, 'episode', 0) or 0)
        tick = int(getattr(trace, 'tick', 0) or 0)
        return f'trace_ep{episode}_tick{tick}'


def aggregate_learning_updates(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    selector_bias: Dict[str, float] = {}
    agenda_prior: Dict[str, float] = {}
    recovery_shortcut: Dict[str, Dict[str, float]] = {}
    failure_preference_policy: Dict[str, Dict[str, Any]] = {}
    retention_failure_policy: Dict[str, Dict[str, Any]] = {}

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        content = obj.get('content', {}) if isinstance(obj.get('content', {}), dict) else {}
        if content.get('type') != 'learning_update':
            continue

        update_type = str(content.get('update_type', '') or '')
        key = str(content.get('key', '') or '')
        if not key:
            continue
        delta = float(content.get('delta', 0.0) or 0.0)
        conf = max(0.0, min(1.0, float(content.get('confidence', obj.get('confidence', 0.5)) or 0.5)))
        weighted = delta * (0.6 + 0.4 * conf)

        if update_type == 'selector_bias':
            selector_bias[key] = max(-0.8, min(0.8, selector_bias.get(key, 0.0) + weighted))
        elif update_type == 'agenda_prior':
            agenda_prior[key] = max(-0.8, min(0.8, agenda_prior.get(key, 0.0) + weighted))
        elif update_type == 'recovery_shortcut':
            bucket = recovery_shortcut.setdefault(key, {'delta': -1.0, 'confidence': 0.0})
            if weighted >= bucket.get('delta', -1.0):
                bucket['delta'] = weighted
                bucket['confidence'] = conf
        elif update_type == 'failure_preference_policy':
            evidence = content.get('evidence', {}) if isinstance(content.get('evidence', {}), dict) else {}
            bucket = failure_preference_policy.setdefault(
                key,
                {
                    'delta': 0.0,
                    'confidence': 0.0,
                    'context_key': key,
                    'strategy_mode_hint': 'balanced',
                    'branch_budget_hint': 0,
                    'verification_budget_hint': 0,
                    'safe_fallback_class': 'wait',
                    'preferred_verification_functions': [],
                    'preferred_fallback_functions': [],
                    'blocked_action_classes': [],
                    'source_action': '',
                    'required_probes': [],
                    'probe_pressure': 0.0,
                    'latent_instability': 0.0,
                    'dominant_branch_id': '',
                    'dominant_anchor_functions': [],
                    'dominant_risky_functions': [],
                },
            )
            bucket['delta'] = max(-0.9, min(0.9, float(bucket.get('delta', 0.0) or 0.0) + weighted))
            bucket['confidence'] = max(float(bucket.get('confidence', 0.0) or 0.0), conf)
            strategy_mode_hint = str(evidence.get('strategy_mode_hint', bucket.get('strategy_mode_hint', 'balanced')) or bucket.get('strategy_mode_hint', 'balanced'))
            if conf >= float(bucket.get('confidence', 0.0) or 0.0) or abs(weighted) >= abs(float(bucket.get('delta', 0.0) or 0.0)):
                bucket['strategy_mode_hint'] = strategy_mode_hint
                bucket['branch_budget_hint'] = max(
                    int(bucket.get('branch_budget_hint', 0) or 0),
                    CreditAssignment._bounded_int(evidence.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
                )
                bucket['verification_budget_hint'] = max(
                    int(bucket.get('verification_budget_hint', 0) or 0),
                    CreditAssignment._bounded_int(evidence.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
                )
                bucket['safe_fallback_class'] = str(evidence.get('safe_fallback_class', bucket.get('safe_fallback_class', 'wait')) or bucket.get('safe_fallback_class', 'wait'))
                bucket['source_action'] = str(evidence.get('source_action', bucket.get('source_action', '')) or bucket.get('source_action', ''))
            bucket['preferred_verification_functions'] = CreditAssignment._string_list(
                bucket.get('preferred_verification_functions', []),
                evidence.get('preferred_verification_functions', []),
            )
            bucket['preferred_fallback_functions'] = CreditAssignment._string_list(
                bucket.get('preferred_fallback_functions', []),
                evidence.get('preferred_fallback_functions', []),
            )
            bucket['blocked_action_classes'] = CreditAssignment._string_list(
                bucket.get('blocked_action_classes', []),
                evidence.get('blocked_action_classes', []),
            )
            bucket['required_probes'] = CreditAssignment._string_list(
                bucket.get('required_probes', []),
                evidence.get('required_probes', []),
            )
            bucket['dominant_anchor_functions'] = CreditAssignment._string_list(
                bucket.get('dominant_anchor_functions', []),
                evidence.get('dominant_anchor_functions', []),
            )
            bucket['dominant_risky_functions'] = CreditAssignment._string_list(
                bucket.get('dominant_risky_functions', []),
                evidence.get('dominant_risky_functions', []),
            )
            bucket['probe_pressure'] = max(
                float(bucket.get('probe_pressure', 0.0) or 0.0),
                max(0.0, min(1.0, float(evidence.get('probe_pressure', 0.0) or 0.0))),
            )
            bucket['latent_instability'] = max(
                float(bucket.get('latent_instability', 0.0) or 0.0),
                max(0.0, min(1.0, float(evidence.get('latent_instability', 0.0) or 0.0))),
            )
            dominant_branch_id = str(evidence.get('dominant_branch_id', '') or '')
            if dominant_branch_id and (
                not str(bucket.get('dominant_branch_id', '') or '')
                or float(bucket.get('probe_pressure', 0.0) or 0.0) <= float(evidence.get('probe_pressure', 0.0) or 0.0)
            ):
                bucket['dominant_branch_id'] = dominant_branch_id
        elif update_type == 'retention_failure_policy':
            evidence = content.get('evidence', {}) if isinstance(content.get('evidence', {}), dict) else {}
            failure_type = str(evidence.get('failure_type', '') or '')
            bucket = retention_failure_policy.setdefault(
                key,
                {
                    'delta': 0.0,
                    'confidence': 0.0,
                    'context_key': key,
                    'base_context_key': str(evidence.get('base_context_key', '') or ''),
                    'dominant_failure_type': failure_type,
                    'severity': 0.0,
                    'strategy_mode_hint': 'balanced',
                    'branch_budget_hint': 0,
                    'verification_budget_hint': 0,
                    'planner_replan_bias_delta': 0.0,
                    'probe_bias_delta': 0.0,
                    'retrieval_pressure_delta': 0.0,
                    'occurrences': 0,
                    'required_probes': [],
                    'probe_pressure': 0.0,
                    'latent_instability': 0.0,
                    'dominant_branch_id': '',
                    'dominant_anchor_functions': [],
                    'dominant_risky_functions': [],
                },
            )
            bucket['delta'] = max(-0.9, min(0.9, float(bucket.get('delta', 0.0) or 0.0) + weighted))
            bucket['confidence'] = max(float(bucket.get('confidence', 0.0) or 0.0), conf)
            bucket['occurrences'] = int(bucket.get('occurrences', 0) or 0) + 1
            severity = max(0.0, min(1.0, float(evidence.get('severity', 0.0) or 0.0)))
            if severity >= float(bucket.get('severity', 0.0) or 0.0) or conf >= float(bucket.get('confidence', 0.0) or 0.0):
                bucket['severity'] = severity
                bucket['dominant_failure_type'] = failure_type or str(bucket.get('dominant_failure_type', '') or '')
                bucket['base_context_key'] = str(evidence.get('base_context_key', bucket.get('base_context_key', '')) or bucket.get('base_context_key', ''))
                bucket['strategy_mode_hint'] = str(evidence.get('strategy_mode_hint', bucket.get('strategy_mode_hint', 'balanced')) or bucket.get('strategy_mode_hint', 'balanced'))
            bucket['branch_budget_hint'] = max(
                int(bucket.get('branch_budget_hint', 0) or 0),
                CreditAssignment._bounded_int(evidence.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
            )
            bucket['verification_budget_hint'] = max(
                int(bucket.get('verification_budget_hint', 0) or 0),
                CreditAssignment._bounded_int(evidence.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
            )
            bucket['planner_replan_bias_delta'] = max(
                float(bucket.get('planner_replan_bias_delta', 0.0) or 0.0),
                max(0.0, min(0.4, float(evidence.get('planner_replan_bias_delta', 0.0) or 0.0))),
            )
            bucket['probe_bias_delta'] = max(
                float(bucket.get('probe_bias_delta', 0.0) or 0.0),
                max(0.0, min(0.4, float(evidence.get('probe_bias_delta', 0.0) or 0.0))),
            )
            bucket['retrieval_pressure_delta'] = max(
                float(bucket.get('retrieval_pressure_delta', 0.0) or 0.0),
                max(0.0, min(0.4, float(evidence.get('retrieval_pressure_delta', 0.0) or 0.0))),
            )
            bucket['required_probes'] = CreditAssignment._string_list(
                bucket.get('required_probes', []),
                evidence.get('required_probes', []),
            )
            bucket['dominant_anchor_functions'] = CreditAssignment._string_list(
                bucket.get('dominant_anchor_functions', []),
                evidence.get('dominant_anchor_functions', []),
            )
            bucket['dominant_risky_functions'] = CreditAssignment._string_list(
                bucket.get('dominant_risky_functions', []),
                evidence.get('dominant_risky_functions', []),
            )
            bucket['probe_pressure'] = max(
                float(bucket.get('probe_pressure', 0.0) or 0.0),
                max(0.0, min(1.0, float(evidence.get('probe_pressure', 0.0) or 0.0))),
            )
            bucket['latent_instability'] = max(
                float(bucket.get('latent_instability', 0.0) or 0.0),
                max(0.0, min(1.0, float(evidence.get('latent_instability', 0.0) or 0.0))),
            )
            dominant_branch_id = str(evidence.get('dominant_branch_id', '') or '')
            if dominant_branch_id and (
                not str(bucket.get('dominant_branch_id', '') or '')
                or float(bucket.get('probe_pressure', 0.0) or 0.0) <= float(evidence.get('probe_pressure', 0.0) or 0.0)
            ):
                bucket['dominant_branch_id'] = dominant_branch_id
            for passthrough_key in (
                'forced_replan_events',
                'rollout_branch_persistence_ratio',
                'rollout_branch_target_phase',
                'rollout_final_phase',
                'governance_reason',
                'selected_name',
            ):
                if passthrough_key in evidence and evidence.get(passthrough_key) not in (None, '', []):
                    bucket[passthrough_key] = evidence.get(passthrough_key)

    return {
        'selector_bias': selector_bias,
        'agenda_prior': agenda_prior,
        'recovery_shortcut': recovery_shortcut,
        'failure_preference_policy': failure_preference_policy,
        'retention_failure_policy': retention_failure_policy,
    }

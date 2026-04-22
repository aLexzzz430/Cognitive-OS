"""
self_model/reliability_tracker.py

Sprint 5: self_model/ 自我认知

追踪模块和动作类型的可靠度.

Rules:
- 第一版只做简单可靠度追踪
- 不做复杂置信度模型
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict


@dataclass
class ModuleReliability:
    """模块可靠度"""
    module_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    
    # Recovery 相关
    recovery_attempts: int = 0
    recovery_successes: int = 0
    
    @property
    def reliability_score(self) -> float:
        """可靠度分数"""
        if self.total_attempts == 0:
            return 0.5  # 未知 = 中等
        return self.successful_attempts / self.total_attempts
    
    @property
    def is_reliable(self) -> bool:
        """是否可靠"""
        return self.reliability_score >= 0.7 and self.total_attempts >= 3
    
    def to_dict(self) -> dict:
        return {
            'module_name': self.module_name,
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts,
            'failed_attempts': self.failed_attempts,
            'recovery_attempts': self.recovery_attempts,
            'recovery_successes': self.recovery_successes,
            'reliability_score': self.reliability_score,
            'is_reliable': self.is_reliable,
        }


@dataclass
class FailureStrategyProfile:
    """Structured failure strategy consumable by planner/governance."""
    action_type: str
    dominant_failure_mode: str = ''
    short_term_failure_pressure: float = 0.0
    long_term_unreliability: float = 0.0
    safe_fallback_class: str = 'wait'
    blocked_action_classes: List[str] = field(default_factory=list)
    recovery_priority: str = 'monitor'
    strategy_mode_hint: str = 'balanced'
    branch_budget_hint: int = 0
    verification_budget_hint: int = 0
    preferred_verification_functions: List[str] = field(default_factory=list)
    preferred_fallback_functions: List[str] = field(default_factory=list)
    persistence_strength: float = 0.0
    persistence_ttl: int = 0
    persistence_source_action: str = ''

    def to_dict(self) -> dict:
        return {
            'action_type': self.action_type,
            'dominant_failure_mode': self.dominant_failure_mode,
            'short_term_failure_pressure': self.short_term_failure_pressure,
            'long_term_unreliability': self.long_term_unreliability,
            'safe_fallback_class': self.safe_fallback_class,
            'blocked_action_classes': list(self.blocked_action_classes),
            'recovery_priority': self.recovery_priority,
            'strategy_mode_hint': self.strategy_mode_hint,
            'branch_budget_hint': int(self.branch_budget_hint),
            'verification_budget_hint': int(self.verification_budget_hint),
            'preferred_verification_functions': list(self.preferred_verification_functions),
            'preferred_fallback_functions': list(self.preferred_fallback_functions),
            'persistence_strength': float(self.persistence_strength),
            'persistence_ttl': int(self.persistence_ttl),
            'persistence_source_action': self.persistence_source_action,
        }


class ReliabilityTracker:
    """
    可靠度追踪器.
    
    第一版职责:
    1. 追踪模块级别的可靠度
    2. 追踪 recovery 类型的成功率
    3. 提供可靠度查询
    
    不做:
    - 复杂置信度模型
    - 预测模型
    """
    
    def __init__(self):
        self._module_reliability: Dict[str, ModuleReliability] = {}
        self._action_type_reliability: Dict[str, ModuleReliability] = {}
        self._failure_mode_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'count': 0, 'recent': 0})
        self._failure_mode_by_action: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._action_outcome_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._action_context_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {'attempts': 0, 'failures': 0}))
        self._failure_preference_memory: Dict[str, Dict[str, Any]] = {}
        self._learned_failure_preference_policy: Dict[str, Dict[str, Any]] = {}
        self._failure_preference_audit_log: List[Dict[str, Any]] = []
        self._failure_preference_audit_seq: int = 0
        self._teacher_guided_events: int = 0
        self._teacher_guided_successes: int = 0
        self._autonomous_events: int = 0
        self._autonomous_successes: int = 0
        self._transfer_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'teacher_guided_attempts': 0,
            'teacher_guided_successes': 0,
        })
        
        # Recovery 类型统计
        self._recovery_type_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'attempts': 0, 'successes': 0
        })
    
    def record_module_outcome(
        self,
        module_name: str,
        success: bool,
        is_recovery: bool = False,
    ) -> None:
        """记录模块执行结果"""
        if module_name not in self._module_reliability:
            self._module_reliability[module_name] = ModuleReliability(module_name=module_name)
        
        mr = self._module_reliability[module_name]
        mr.total_attempts += 1
        
        if success:
            mr.successful_attempts += 1
        else:
            mr.failed_attempts += 1
        
        if is_recovery:
            mr.recovery_attempts += 1
            if success:
                mr.recovery_successes += 1
    
    def record_action_type_outcome(
        self,
        action_type: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录动作类型结果"""
        if action_type not in self._action_type_reliability:
            self._action_type_reliability[action_type] = ModuleReliability(module_name=action_type)
        
        atr = self._action_type_reliability[action_type]
        atr.total_attempts += 1
        if success:
            atr.successful_attempts += 1
        else:
            atr.failed_attempts += 1

        action = str(action_type or '')
        ctx_key = self._context_key(context)
        self._action_outcome_history[action].append({
            'success': bool(success),
            'context_key': ctx_key,
        })
        self._action_outcome_history[action] = self._action_outcome_history[action][-30:]
        if ctx_key:
            stat = self._action_context_stats[action][ctx_key]
            stat['attempts'] += 1
            if not success:
                stat['failures'] += 1

    def record_failure_mode(
        self,
        action_type: str,
        failure_mode: str,
    ) -> None:
        """Track failure modes by action/module to support self-model suppression."""
        if not failure_mode:
            return
        mode = str(failure_mode)
        self._failure_mode_stats[mode]['count'] += 1
        self._failure_mode_stats[mode]['recent'] = min(20, self._failure_mode_stats[mode]['recent'] + 1)
        self._failure_mode_by_action[str(action_type)][mode] += 1

    def decay_failure_recency(self) -> None:
        """Decay recent failure pressure once per tick/episode update."""
        for stats in self._failure_mode_stats.values():
            stats['recent'] = max(0, int(stats.get('recent', 0)) - 1)
        self._decay_failure_preference_memory()

    def record_failure_preference(
        self,
        action_type: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        action_meta: Optional[Dict[str, Any]] = None,
        reward: float = 0.0,
    ) -> None:
        action = str(action_type or '').strip()
        if not action or action == 'wait':
            return
        meta = action_meta if isinstance(action_meta, dict) else {}
        failure_profile = meta.get('failure_strategy_profile', {}) if isinstance(meta.get('failure_strategy_profile', {}), dict) else {}
        global_profile = meta.get('global_failure_strategy', {}) if isinstance(meta.get('global_failure_strategy', {}), dict) else {}
        preference_guidance = meta.get('failure_preference_guidance', {}) if isinstance(meta.get('failure_preference_guidance', {}), dict) else {}
        # Preserve the original anti-noise guard for bare failures, but honor explicit
        # planner/governance guidance immediately when the runtime already emitted a
        # structured failure strategy profile for this action.
        action_stats = self._action_type_reliability.get(action)
        total_attempts = action_stats.total_attempts if action_stats else 0
        has_structured_guidance = bool(failure_profile or global_profile or preference_guidance)
        if total_attempts < 3 and not has_structured_guidance:
            return
        control_profile = {
            'strategy_mode': preference_guidance.get('strategy_mode', failure_profile.get('strategy_mode_hint', global_profile.get('strategy_mode_hint', 'recover'))),
            'branch_budget': preference_guidance.get('branch_budget_hint', failure_profile.get('branch_budget_hint', global_profile.get('branch_budget_hint', 0))),
            'verification_budget': preference_guidance.get('verification_budget_hint', failure_profile.get('verification_budget_hint', global_profile.get('verification_budget_hint', 0))),
        }
        fallback_class = str(
            preference_guidance.get('safe_fallback_class', '')
            or failure_profile.get('safe_fallback_class', '')
            or global_profile.get('safe_fallback_class', '')
            or 'wait'
        )
        blocked = self._merge_ordered_lists(
            [action],
            failure_profile.get('blocked_action_classes', []),
            global_profile.get('blocked_action_classes', []),
        )
        preferred_verification = self._merge_ordered_lists(
            failure_profile.get('preferred_verification_functions', []),
            global_profile.get('preferred_verification_functions', []),
        )
        preferred_fallback = self._merge_ordered_lists(
            failure_profile.get('preferred_fallback_functions', []),
            global_profile.get('preferred_fallback_functions', []),
        )
        snapshot = self.build_failure_strategy(
            action,
            short_term_pressure=1.0 if reward < 0.0 else 0.7,
            context=context,
            planner_control_profile=control_profile,
        ).to_dict()
        snapshot['safe_fallback_class'] = fallback_class
        snapshot['blocked_action_classes'] = self._merge_ordered_lists(blocked, snapshot.get('blocked_action_classes', []))
        snapshot['preferred_verification_functions'] = self._merge_ordered_lists(
            preferred_verification,
            snapshot.get('preferred_verification_functions', []),
        )
        snapshot['preferred_fallback_functions'] = self._merge_ordered_lists(
            preferred_fallback,
            snapshot.get('preferred_fallback_functions', []),
        )
        snapshot['persistence_strength'] = 1.0
        snapshot['persistence_ttl'] = max(3, int(snapshot.get('branch_budget_hint', 1) or 1))
        snapshot['persistence_source_action'] = action
        snapshot['context_key'] = self._context_key(context)
        for key in ('__global__', snapshot['context_key']):
            self._store_failure_preference_snapshot(key, snapshot)

    def get_failure_preference_snapshot(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        key = self._context_key(context)
        learned = {}
        transient = {}
        if key and key in self._learned_failure_preference_policy:
            learned = dict(self._learned_failure_preference_policy[key])
        elif '__global__' in self._learned_failure_preference_policy:
            learned = dict(self._learned_failure_preference_policy['__global__'])
        if key and key in self._failure_preference_memory:
            transient = dict(self._failure_preference_memory[key])
        elif '__global__' in self._failure_preference_memory:
            transient = dict(self._failure_preference_memory['__global__'])
        return self._merge_failure_preference_entries(learned, transient)

    def synchronize_failure_preference_learning(self, policies: Dict[str, Any]) -> None:
        learned: Dict[str, Dict[str, Any]] = {}
        for key, raw in (policies.items() if isinstance(policies, dict) else []):
            payload = dict(raw) if isinstance(raw, dict) else {}
            delta = float(payload.get('delta', 0.0) or 0.0)
            confidence = max(0.0, min(1.0, float(payload.get('confidence', 0.0) or 0.0)))
            if delta <= -0.12 or confidence < 0.2:
                continue
            strength = max(0.0, min(1.0, delta * (0.65 + 0.35 * confidence)))
            if strength < 0.10:
                continue
            normalized_key = str(key or payload.get('context_key', '') or '').strip() or '__global__'
            learned[normalized_key] = {
                'context_key': normalized_key,
                'strategy_mode_hint': str(payload.get('strategy_mode_hint', 'recover') or 'recover'),
                'branch_budget_hint': self._coerce_int(payload.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
                'verification_budget_hint': self._coerce_int(payload.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
                'safe_fallback_class': str(payload.get('safe_fallback_class', 'wait') or 'wait'),
                'preferred_verification_functions': self._merge_ordered_lists(payload.get('preferred_verification_functions', [])),
                'preferred_fallback_functions': self._merge_ordered_lists(payload.get('preferred_fallback_functions', [])),
                'blocked_action_classes': self._merge_ordered_lists(payload.get('blocked_action_classes', [])),
                'persistence_strength': strength,
                'persistence_ttl': 0,
                'persistence_source_action': str(payload.get('source_action', '') or ''),
                'confidence': confidence,
                'delta': delta,
                'source': 'learning_update',
            }
            self._append_failure_preference_audit_event(
                layer='learned',
                event='learning_sync',
                context_key=normalized_key,
                payload=learned[normalized_key],
            )
        self._learned_failure_preference_policy = learned

    def build_failure_preference_audit_report(self) -> Dict[str, Any]:
        context_keys = sorted({
            key
            for key in list(self._learned_failure_preference_policy.keys()) + list(self._failure_preference_memory.keys())
            if str(key or '').strip()
        })
        contexts: List[Dict[str, Any]] = []
        for key in context_keys:
            transient = dict(self._failure_preference_memory.get(key, {})) if isinstance(self._failure_preference_memory.get(key, {}), dict) else {}
            learned = dict(self._learned_failure_preference_policy.get(key, {})) if isinstance(self._learned_failure_preference_policy.get(key, {}), dict) else {}
            merged = self._merge_failure_preference_entries(learned, transient)
            contexts.append({
                'context_key': key,
                'transient': transient,
                'learned': learned,
                'merged': merged,
            })
        return {
            'current_transient': {key: dict(value) for key, value in self._failure_preference_memory.items()},
            'current_learned': {key: dict(value) for key, value in self._learned_failure_preference_policy.items()},
            'contexts': contexts,
            'retention_curve': self.get_failure_preference_retention_curve(),
            'audit_log': self.get_failure_preference_audit_log(),
        }

    def get_failure_preference_audit_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        window = max(1, int(limit or 1))
        return [dict(row) for row in self._failure_preference_audit_log[-window:]]

    def get_failure_preference_retention_curve(self) -> List[Dict[str, Any]]:
        baselines: Dict[tuple, float] = {}
        curve: List[Dict[str, Any]] = []
        for row in self._failure_preference_audit_log:
            if not isinstance(row, dict):
                continue
            context_key = str(row.get('context_key', '') or '')
            layer = str(row.get('layer', '') or '')
            strength = max(0.0, min(1.0, float(row.get('persistence_strength', 0.0) or 0.0)))
            baseline_key = (context_key, layer)
            if baseline_key not in baselines and strength > 0.0:
                baselines[baseline_key] = strength
            baseline = baselines.get(baseline_key, 0.0)
            retention_vs_initial = (strength / baseline) if baseline > 0.0 else 0.0
            curve.append({
                'event_index': int(row.get('event_index', 0) or 0),
                'layer': layer,
                'event': str(row.get('event', '') or ''),
                'context_key': context_key,
                'persistence_strength': strength,
                'persistence_ttl': max(0, int(row.get('persistence_ttl', 0) or 0)),
                'retention_vs_initial': max(0.0, min(1.25, float(retention_vs_initial))),
                'strategy_mode_hint': str(row.get('strategy_mode_hint', '') or ''),
                'source': str(row.get('source', '') or ''),
                'source_action': str(row.get('persistence_source_action', '') or ''),
            })
        return curve

    def get_action_failure_risk(self, action_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Estimate failure risk from long-term + recent-window + context-window evidence."""
        modes = self._failure_mode_by_action.get(str(action_type), {})
        total = sum(modes.values())
        if total > 0:
            dominant = max(modes.values())
            long_term_risk = min(1.0, (dominant / total) * min(1.0, total / 5.0))
        else:
            long_term_risk = 0.0

        action = str(action_type or '')
        history = self._action_outcome_history.get(action, [])
        recent = history[-8:]
        if recent:
            failures = sum(1 for row in recent if not bool(row.get('success', False)))
            recent_window_risk = failures / len(recent)
        else:
            recent_window_risk = 0.0

        ctx_key = self._context_key(context)
        if ctx_key:
            cstats = self._action_context_stats.get(action, {}).get(ctx_key, {})
            attempts = int(cstats.get('attempts', 0) or 0)
            failures = int(cstats.get('failures', 0) or 0)
            context_window_risk = (failures / attempts) if attempts > 0 else 0.0
        else:
            attempts = 0
            context_window_risk = 0.0

        if not recent and attempts <= 0:
            return max(0.0, min(1.0, long_term_risk))

        return max(0.0, min(1.0, long_term_risk * 0.4 + recent_window_risk * 0.35 + context_window_risk * 0.25))

    def get_top_failure_modes(self, limit: int = 5) -> List[Dict[str, Any]]:
        ranked = sorted(
            ((mode, stats['count'], stats['recent']) for mode, stats in self._failure_mode_stats.items()),
            key=lambda item: (item[2], item[1]),
            reverse=True,
        )
        return [
            {'failure_mode': mode, 'count': count, 'recent': recent}
            for mode, count, recent in ranked[:limit]
        ]

    def get_reliability_by_action_type(self) -> Dict[str, float]:
        """Return stable action/function reliability mapping for external consumers."""
        return {
            action_type: stats.reliability_score
            for action_type, stats in self._action_type_reliability.items()
        }

    def get_recent_failure_profile(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return recent failure profile, ordered by recency then frequency."""
        return self.get_top_failure_modes(limit=limit)

    def get_recent_contextual_risk(self, action_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Expose contextual/recent risk for downstream consumers."""
        return self.get_action_failure_risk(action_type, context=context)

    def get_overall_recovery_success_rate(self) -> float:
        """Aggregate recovery success rate across recovery types."""
        attempts = 0
        successes = 0
        for stats in self._recovery_type_stats.values():
            attempts += int(stats.get('attempts', 0) or 0)
            successes += int(stats.get('successes', 0) or 0)
        if attempts <= 0:
            return 0.5
        return max(0.0, min(1.0, successes / attempts))
    

    def build_failure_strategy(
        self,
        action_type: str,
        short_term_pressure: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> FailureStrategyProfile:
        action = str(action_type or '')
        modes = self._failure_mode_by_action.get(action, {})
        dominant_mode = ''
        dominant_count = 0
        total = sum(modes.values())
        if modes:
            dominant_mode, dominant_count = max(modes.items(), key=lambda item: item[1])

        rel = self.get_action_type_reliability(action)
        long_term_unreliable = 1.0 - (rel.reliability_score if rel else 0.5)

        blocked: List[str] = []
        is_bare_click_family = str(action or '').strip().upper() == 'ACTION6'
        if long_term_unreliable >= 0.65 and total >= 3 and not is_bare_click_family:
            blocked.append(action)

        recovery_priority = 'high' if (short_term_pressure >= 0.6 or long_term_unreliable >= 0.7) else 'medium'
        safe_fallback = 'wait' if long_term_unreliable >= 0.6 else 'compute_stats'
        control_profile = planner_control_profile if isinstance(planner_control_profile, dict) else {}
        preference_memory = self.get_failure_preference_snapshot(context=context)
        has_explicit_mode = bool(str(control_profile.get('strategy_mode', '') or '').strip())
        strategy_mode_hint = str(control_profile.get('strategy_mode', '') or '')
        if not strategy_mode_hint:
            if preference_memory:
                strategy_mode_hint = str(preference_memory.get('strategy_mode_hint', '') or '')
        if not strategy_mode_hint:
            if short_term_pressure >= 0.7 or long_term_unreliable >= 0.72:
                strategy_mode_hint = 'recover'
            elif dominant_mode and any(token in dominant_mode.lower() for token in ('low_trust', 'uncertain', 'mismatch', 'verify', 'phase')):
                strategy_mode_hint = 'verify'
            else:
                strategy_mode_hint = 'balanced'
        branch_budget_hint = self._coerce_int(control_profile.get('branch_budget', 0), minimum=0, maximum=4, default=0)
        verification_budget_hint = self._coerce_int(control_profile.get('verification_budget', 0), minimum=0, maximum=3, default=0)
        if preference_memory:
            if branch_budget_hint <= 0 or not has_explicit_mode:
                branch_budget_hint = max(
                    branch_budget_hint,
                    self._coerce_int(preference_memory.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
                )
            if verification_budget_hint <= 0 or not has_explicit_mode:
                verification_budget_hint = max(
                    verification_budget_hint,
                    self._coerce_int(preference_memory.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
                )
        if branch_budget_hint <= 0:
            if strategy_mode_hint == 'recover':
                branch_budget_hint = 3
            elif strategy_mode_hint == 'verify':
                branch_budget_hint = 2
            elif strategy_mode_hint in {'explore', 'exploit'}:
                branch_budget_hint = 2
            else:
                branch_budget_hint = 1
        if verification_budget_hint <= 0:
            if strategy_mode_hint == 'recover':
                verification_budget_hint = 2 if short_term_pressure >= 0.75 or long_term_unreliable >= 0.72 else 1
            elif strategy_mode_hint == 'verify':
                verification_budget_hint = 1 if dominant_mode else 2
            else:
                verification_budget_hint = 0
        if preference_memory and preference_memory.get('safe_fallback_class'):
            safe_fallback = str(preference_memory.get('safe_fallback_class') or safe_fallback)
        blocked = self._merge_ordered_lists(blocked, preference_memory.get('blocked_action_classes', []) if preference_memory else [])
        preferred_verification = self._rank_preferred_functions(
            context=context,
            action_type=action,
            safe_fallback_class=safe_fallback,
            strategy_mode_hint=strategy_mode_hint,
            verification_only=True,
            blocked_action_classes=blocked,
        )
        preferred_fallback = self._rank_preferred_functions(
            context=context,
            action_type=action,
            safe_fallback_class=safe_fallback,
            strategy_mode_hint=strategy_mode_hint,
            verification_only=False,
            blocked_action_classes=blocked,
        )
        if preference_memory:
            preferred_verification = self._merge_ordered_lists(
                preference_memory.get('preferred_verification_functions', []),
                preferred_verification,
            )
            preferred_fallback = self._merge_ordered_lists(
                preference_memory.get('preferred_fallback_functions', []),
                preferred_fallback,
            )

        return FailureStrategyProfile(
            action_type=action,
            dominant_failure_mode=dominant_mode,
            short_term_failure_pressure=max(0.0, min(1.0, float(short_term_pressure))),
            long_term_unreliability=max(0.0, min(1.0, float(long_term_unreliable))),
            safe_fallback_class=safe_fallback,
            blocked_action_classes=blocked,
            recovery_priority=recovery_priority,
            strategy_mode_hint=strategy_mode_hint,
            branch_budget_hint=branch_budget_hint,
            verification_budget_hint=verification_budget_hint,
            preferred_verification_functions=preferred_verification,
            preferred_fallback_functions=preferred_fallback,
            persistence_strength=max(0.0, min(1.0, float(preference_memory.get('persistence_strength', 0.0) or 0.0))) if preference_memory else 0.0,
            persistence_ttl=max(0, int(preference_memory.get('persistence_ttl', 0) or 0)) if preference_memory else 0,
            persistence_source_action=str(preference_memory.get('persistence_source_action', '') or '') if preference_memory else '',
        )

    def build_global_failure_strategy(
        self,
        short_term_pressure: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        planner_control_profile: Optional[Dict[str, Any]] = None,
    ) -> FailureStrategyProfile:
        top = self.get_top_failure_modes(limit=1)
        dominant = top[0]['failure_mode'] if top else ''
        action_candidates = list(self._action_type_reliability.keys())
        if action_candidates:
            # action with lowest reliability
            action = min(action_candidates, key=lambda a: self._action_type_reliability[a].reliability_score)
        else:
            action = ''
        return self.build_failure_strategy(
            action,
            short_term_pressure=short_term_pressure,
            context=context,
            planner_control_profile=planner_control_profile,
        )

    def record_recovery_type(
        self,
        recovery_type: str,
        success: bool,
    ) -> None:
        """记录 recovery 类型结果"""
        stats = self._recovery_type_stats[recovery_type]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
    
    def get_module_reliability(self, module_name: str) -> Optional[ModuleReliability]:
        """获取模块可靠度"""
        return self._module_reliability.get(module_name)
    
    def get_action_type_reliability(self, action_type: str) -> Optional[ModuleReliability]:
        """获取动作类型可靠度"""
        return self._action_type_reliability.get(action_type)
    
    def get_recovery_type_stats(self, recovery_type: str) -> Dict[str, int]:
        """获取 recovery 类型统计"""
        return dict(self._recovery_type_stats.get(recovery_type, {'attempts': 0, 'successes': 0}))

    def record_teacher_dependence_event(self, *, teacher_present: bool, success: bool) -> None:
        """Track whether successful behavior still depends on teacher presence."""
        if teacher_present:
            self._teacher_guided_events += 1
            if success:
                self._teacher_guided_successes += 1
            return
        self._autonomous_events += 1
        if success:
            self._autonomous_successes += 1

    def estimate_teacher_dependence(self) -> float:
        guided_rate = (
            self._teacher_guided_successes / self._teacher_guided_events
            if self._teacher_guided_events > 0 else 0.5
        )
        autonomous_rate = (
            self._autonomous_successes / self._autonomous_events
            if self._autonomous_events > 0 else (0.0 if self._teacher_guided_events > 0 else 0.5)
        )
        evidence_ratio = (
            self._teacher_guided_events / max(1, (self._teacher_guided_events + self._autonomous_events))
        )
        dependence = 0.5 + ((guided_rate - autonomous_rate) * 0.45) + ((evidence_ratio - 0.5) * 0.20)
        return max(0.0, min(1.0, dependence))

    def record_transfer_attempt(
        self,
        *,
        source_family: str,
        target_family: str,
        success: bool,
        teacher_present: bool = False,
    ) -> None:
        source = str(source_family or '').strip()
        target = str(target_family or '').strip()
        if not source or not target:
            return
        key = f'{source}->{target}'
        stats = self._transfer_stats[key]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
        if teacher_present:
            stats['teacher_guided_attempts'] += 1
            if success:
                stats['teacher_guided_successes'] += 1

    def estimate_transfer_readiness(self) -> float:
        if not self._transfer_stats:
            return 0.0
        scores = []
        for stats in self._transfer_stats.values():
            attempts = int(stats.get('attempts', 0) or 0)
            if attempts <= 0:
                continue
            success_rate = float(stats.get('successes', 0) or 0) / attempts
            teacher_ratio = float(stats.get('teacher_guided_attempts', 0) or 0) / attempts
            scores.append(max(0.0, min(1.0, success_rate * (1.0 - (teacher_ratio * 0.35)))))
        if not scores:
            return 0.0
        return max(0.0, min(1.0, sum(scores) / len(scores)))
    
    def get_most_reliable_action_type(self, candidates: List[str]) -> Optional[str]:
        """从候选中选择最可靠的动作类型"""
        best = None
        best_score = -1
        
        for action_type in candidates:
            rel = self._action_type_reliability.get(action_type)
            if rel:
                score = rel.reliability_score
                if score > best_score:
                    best_score = score
                    best = action_type
        
        return best
    
    def get_all_module_reliabilities(self) -> List[Dict[str, Any]]:
        """获取所有模块可靠度"""
        return [mr.to_dict() for mr in self._module_reliability.values()]
    
    def get_unreliable_modules(self) -> List[str]:
        """获取不可靠模块列表"""
        return [
            name for name, mr in self._module_reliability.items()
            if not mr.is_reliable and mr.total_attempts >= 3
        ]
    
    def to_dict(self) -> dict:
        return {
            'modules': self.get_all_module_reliabilities(),
            'action_types': [
                atr.to_dict() for atr in self._action_type_reliability.values()
            ],
            'failure_modes': self.get_top_failure_modes(limit=10),
            'failure_strategy_global': self.build_global_failure_strategy().to_dict(),
            'failure_preference_memory': {
                'transient': {key: dict(value) for key, value in self._failure_preference_memory.items()},
                'learned': {key: dict(value) for key, value in self._learned_failure_preference_policy.items()},
            },
            'failure_preference_audit': self.build_failure_preference_audit_report(),
            'teacher_dependence_estimate': self.estimate_teacher_dependence(),
            'transfer_readiness': self.estimate_transfer_readiness(),
            'recovery_types': {
                rt: dict(stats) for rt, stats in self._recovery_type_stats.items()
            },
        }

    def _context_key(self, context: Optional[Dict[str, Any]]) -> str:
        if not isinstance(context, dict):
            return ''
        task_family = str(context.get('task_family', 'unknown') or 'unknown')
        phase = str(context.get('phase', 'unknown') or 'unknown')
        observation_mode = str(context.get('observation_mode', 'unknown') or 'unknown')
        resource_band = str(context.get('resource_band', 'normal') or 'normal')
        return f"task_family={task_family}|phase={phase}|observation_mode={observation_mode}|resource_band={resource_band}"

    def _store_failure_preference_snapshot(self, key: str, snapshot: Dict[str, Any]) -> None:
        normalized_key = str(key or '').strip() or '__global__'
        existing = self._failure_preference_memory.get(normalized_key, {})
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update({
            'strategy_mode_hint': str(snapshot.get('strategy_mode_hint', merged.get('strategy_mode_hint', 'recover')) or 'recover'),
            'branch_budget_hint': max(
                self._coerce_int(merged.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
                self._coerce_int(snapshot.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
            ),
            'verification_budget_hint': max(
                self._coerce_int(merged.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
                self._coerce_int(snapshot.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
            ),
            'safe_fallback_class': str(snapshot.get('safe_fallback_class', merged.get('safe_fallback_class', 'wait')) or 'wait'),
            'blocked_action_classes': self._merge_ordered_lists(
                snapshot.get('blocked_action_classes', []),
                merged.get('blocked_action_classes', []),
            ),
            'preferred_verification_functions': self._merge_ordered_lists(
                snapshot.get('preferred_verification_functions', []),
                merged.get('preferred_verification_functions', []),
            ),
            'preferred_fallback_functions': self._merge_ordered_lists(
                snapshot.get('preferred_fallback_functions', []),
                merged.get('preferred_fallback_functions', []),
            ),
            'persistence_strength': max(
                max(0.0, min(1.0, float(merged.get('persistence_strength', 0.0) or 0.0))),
                max(0.0, min(1.0, float(snapshot.get('persistence_strength', 1.0) or 1.0))),
            ),
            'persistence_ttl': max(
                max(0, int(merged.get('persistence_ttl', 0) or 0)),
                max(0, int(snapshot.get('persistence_ttl', 0) or 0)),
            ),
            'persistence_source_action': str(snapshot.get('persistence_source_action', merged.get('persistence_source_action', '')) or ''),
            'context_key': str(snapshot.get('context_key', normalized_key) or normalized_key),
        })
        self._failure_preference_memory[normalized_key] = merged
        self._append_failure_preference_audit_event(
            layer='transient',
            event='recorded',
            context_key=normalized_key,
            payload=merged,
        )

    def _decay_failure_preference_memory(self) -> None:
        expired: List[str] = []
        for key, entry in list(self._failure_preference_memory.items()):
            if not isinstance(entry, dict):
                expired.append(key)
                continue
            ttl = max(0, int(entry.get('persistence_ttl', 0) or 0) - 1)
            strength = max(0.0, min(1.0, float(entry.get('persistence_strength', 0.0) or 0.0) * 0.72))
            entry['persistence_ttl'] = ttl
            entry['persistence_strength'] = strength
            self._append_failure_preference_audit_event(
                layer='transient',
                event='decayed',
                context_key=key,
                payload=entry,
            )
            if ttl <= 0 or strength < 0.18:
                expired.append(key)
        for key in expired:
            payload = dict(self._failure_preference_memory.get(key, {})) if isinstance(self._failure_preference_memory.get(key, {}), dict) else {}
            if payload:
                payload['persistence_ttl'] = 0
                payload['persistence_strength'] = 0.0
                self._append_failure_preference_audit_event(
                    layer='transient',
                    event='expired',
                    context_key=key,
                    payload=payload,
                )
            self._failure_preference_memory.pop(key, None)

    def _merge_failure_preference_entries(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(base, dict) and not isinstance(overlay, dict):
            return {}
        merged = dict(base) if isinstance(base, dict) else {}
        extra = dict(overlay) if isinstance(overlay, dict) else {}
        if not merged:
            return extra
        if not extra:
            return merged
        merged['context_key'] = str(extra.get('context_key', merged.get('context_key', '')) or merged.get('context_key', ''))
        merged['strategy_mode_hint'] = str(extra.get('strategy_mode_hint', merged.get('strategy_mode_hint', 'balanced')) or merged.get('strategy_mode_hint', 'balanced'))
        merged['branch_budget_hint'] = max(
            self._coerce_int(merged.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
            self._coerce_int(extra.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
        )
        merged['verification_budget_hint'] = max(
            self._coerce_int(merged.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
            self._coerce_int(extra.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
        )
        merged['safe_fallback_class'] = str(extra.get('safe_fallback_class', merged.get('safe_fallback_class', 'wait')) or merged.get('safe_fallback_class', 'wait'))
        merged['preferred_verification_functions'] = self._merge_ordered_lists(
            extra.get('preferred_verification_functions', []),
            merged.get('preferred_verification_functions', []),
        )
        merged['preferred_fallback_functions'] = self._merge_ordered_lists(
            extra.get('preferred_fallback_functions', []),
            merged.get('preferred_fallback_functions', []),
        )
        merged['blocked_action_classes'] = self._merge_ordered_lists(
            extra.get('blocked_action_classes', []),
            merged.get('blocked_action_classes', []),
        )
        merged['persistence_strength'] = max(
            max(0.0, min(1.0, float(merged.get('persistence_strength', 0.0) or 0.0))),
            max(0.0, min(1.0, float(extra.get('persistence_strength', 0.0) or 0.0))),
        )
        merged['persistence_ttl'] = max(
            max(0, int(merged.get('persistence_ttl', 0) or 0)),
            max(0, int(extra.get('persistence_ttl', 0) or 0)),
        )
        merged['persistence_source_action'] = str(extra.get('persistence_source_action', merged.get('persistence_source_action', '')) or merged.get('persistence_source_action', ''))
        merged['source'] = str(extra.get('source', merged.get('source', '')) or merged.get('source', ''))
        return merged

    def _append_failure_preference_audit_event(
        self,
        *,
        layer: str,
        event: str,
        context_key: str,
        payload: Dict[str, Any],
    ) -> None:
        self._failure_preference_audit_seq += 1
        row = {
            'event_index': int(self._failure_preference_audit_seq),
            'layer': str(layer or ''),
            'event': str(event or ''),
            'context_key': str(context_key or ''),
            'strategy_mode_hint': str(payload.get('strategy_mode_hint', '') or ''),
            'branch_budget_hint': self._coerce_int(payload.get('branch_budget_hint', 0), minimum=0, maximum=4, default=0),
            'verification_budget_hint': self._coerce_int(payload.get('verification_budget_hint', 0), minimum=0, maximum=3, default=0),
            'safe_fallback_class': str(payload.get('safe_fallback_class', '') or ''),
            'preferred_verification_functions': self._merge_ordered_lists(payload.get('preferred_verification_functions', [])),
            'preferred_fallback_functions': self._merge_ordered_lists(payload.get('preferred_fallback_functions', [])),
            'blocked_action_classes': self._merge_ordered_lists(payload.get('blocked_action_classes', [])),
            'persistence_strength': max(0.0, min(1.0, float(payload.get('persistence_strength', 0.0) or 0.0))),
            'persistence_ttl': max(0, int(payload.get('persistence_ttl', 0) or 0)),
            'persistence_source_action': str(payload.get('persistence_source_action', payload.get('source_action', '')) or ''),
            'source': str(payload.get('source', '') or ''),
        }
        self._failure_preference_audit_log.append(row)
        del self._failure_preference_audit_log[:-400]

    @staticmethod
    def _merge_ordered_lists(*values: Any) -> List[str]:
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

    def _rank_preferred_functions(
        self,
        *,
        context: Optional[Dict[str, Any]],
        action_type: str,
        safe_fallback_class: str,
        strategy_mode_hint: str,
        verification_only: bool,
        blocked_action_classes: List[str],
    ) -> List[str]:
        candidates = self._candidate_functions_from_context(context)
        blocked = {str(value or '').strip() for value in blocked_action_classes if str(value or '').strip()}
        action = str(action_type or '').strip()
        filtered: List[str] = []
        seen = set()
        for fn_name in candidates:
            if not fn_name or fn_name in seen or fn_name == action or fn_name in blocked:
                continue
            is_verification = self._is_verification_function(fn_name)
            if verification_only and not is_verification:
                continue
            if not verification_only and is_verification:
                continue
            seen.add(fn_name)
            filtered.append(fn_name)
        ranked = sorted(
            filtered,
            key=lambda fn_name: self._function_priority(
                fn_name,
                context=context,
                safe_fallback_class=safe_fallback_class,
                strategy_mode_hint=strategy_mode_hint,
                verification_only=verification_only,
            ),
            reverse=True,
        )
        return ranked

    def _function_priority(
        self,
        fn_name: str,
        *,
        context: Optional[Dict[str, Any]],
        safe_fallback_class: str,
        strategy_mode_hint: str,
        verification_only: bool,
    ) -> float:
        reliability = 0.5
        rel = self._action_type_reliability.get(str(fn_name or ''))
        if rel is not None:
            reliability = float(rel.reliability_score)
        risk = self.get_action_failure_risk(fn_name, context=context)
        hidden_focus = self._hidden_focus_functions(context)
        fn_class = self._classify_function(fn_name)
        score = (reliability * 0.65) - (risk * 0.45)
        if fn_name in hidden_focus:
            score += 0.28
        if verification_only and self._is_verification_function(fn_name):
            score += 0.32
        if not verification_only and safe_fallback_class and fn_class == safe_fallback_class:
            score += 0.20
        if strategy_mode_hint == 'recover' and fn_class in {'wait', 'probe', 'compute_stats'}:
            score += 0.16
        elif strategy_mode_hint == 'verify' and self._is_verification_function(fn_name):
            score += 0.18
        elif strategy_mode_hint == 'exploit' and fn_class == 'compute_stats':
            score += 0.10
        elif strategy_mode_hint == 'explore' and fn_class == 'action':
            score += 0.10
        return float(score)

    def _candidate_functions_from_context(self, context: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(context, dict):
            return []
        hidden_focus = self._hidden_focus_functions(context)
        pools = (
            hidden_focus,
            context.get('visible_functions', []) or [],
            context.get('discovered_functions', []) or [],
            context.get('available_functions', []) or [],
        )
        candidates: List[str] = []
        seen = set()
        for pool in pools:
            if not isinstance(pool, list):
                continue
            for fn in pool:
                fn_name = str(fn or '').strip()
                if not fn_name or fn_name in seen:
                    continue
                seen.add(fn_name)
                candidates.append(fn_name)
        return candidates

    def _hidden_focus_functions(self, context: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(context, dict):
            return []
        hidden = context.get('world_model_hidden_state', {})
        if isinstance(hidden, dict):
            focus = hidden.get('focus_functions', [])
            if isinstance(focus, list):
                return [str(value).strip() for value in focus if str(value or '').strip()]
        world_model = context.get('world_model_summary', {})
        if isinstance(world_model, dict):
            hidden_state = world_model.get('hidden_state', {})
            if isinstance(hidden_state, dict):
                focus = hidden_state.get('focus_functions', [])
                if isinstance(focus, list):
                    return [str(value).strip() for value in focus if str(value or '').strip()]
        return []

    @staticmethod
    def _classify_function(fn_name: str) -> str:
        name = str(fn_name or '').strip().lower()
        if not name or name == 'wait':
            return 'wait'
        if any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test')):
            return 'probe'
        if any(token in name for token in ('stat', 'count', 'measure', 'summary', 'score', 'analy', 'calc', 'compute')):
            return 'compute_stats'
        return 'action'

    @staticmethod
    def _is_verification_function(fn_name: str) -> bool:
        return ReliabilityTracker._classify_function(fn_name) == 'probe'

    @staticmethod
    def _coerce_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return default

"""
self_model/capability_profile.py

Sprint 5: self_model/ 自我认知

追踪系统能力画像.

Rules:
- 第一版只记录已知能力
- 不做 blindspot registry 全量版
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import re


class CapabilityLevel(Enum):
    """能力等级"""
    UNKNOWN = "unknown"
    TESTED_SUCCESS = "tested_success"
    TESTED_FAILURE = "tested_failure"
    CONFIRMED = "confirmed"
    RELIABLE = "reliable"


@dataclass
class FunctionCapability:
    """单个函数的能力画像"""
    function_name: str
    level: CapabilityLevel = CapabilityLevel.UNKNOWN
    
    # 能力详情
    success_count: int = 0
    failure_count: int = 0
    total_calls: int = 0
    
    # 置信度
    confidence: float = 0.0  # 0.0-1.0
    
    # 最后使用
    last_used_episode: Optional[int] = None
    last_successful_episode: Optional[int] = None
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls
    
    @property
    def is_reliable(self) -> bool:
        """是否可靠"""
        return self.level in (CapabilityLevel.RELIABLE, CapabilityLevel.CONFIRMED)
    
    def to_dict(self) -> dict:
        return {
            'function_name': self.function_name,
            'level': self.level.value,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_calls': self.total_calls,
            'success_rate': self.success_rate,
            'confidence': self.confidence,
            'last_used_episode': self.last_used_episode,
            'last_successful_episode': self.last_successful_episode,
        }


@dataclass
class CapabilityProfile:
    """
    整体能力画像.
    
    第一版字段:
    - discovered_functions: 已发现函数
    - function_capabilities: 函数能力详情
    - composite_capabilities: 组合能力
    """
    agent_id: str = ""
    
    # 已发现函数
    discovered_functions: Set[str] = field(default_factory=set)
    
    # 函数能力
    function_capabilities: Dict[str, FunctionCapability] = field(default_factory=dict)
    
    # 能力等级统计
    reliable_count: int = 0
    confirmed_count: int = 0
    tested_success_count: int = 0
    tested_failure_count: int = 0
    unknown_count: int = 0
    
    # 非函数调用的独立统计通道（例如 hypothesis/object_id 等）
    auxiliary_channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 拒绝写入能力画像的输入计数
    rejected_inputs: int = 0
    # 条件化能力统计：function -> context_key -> stats
    contextual_capabilities: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    _FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
    
    def add_function(self, function_name: str) -> None:
        """添加发现的函数"""
        if function_name not in self.discovered_functions:
            self.discovered_functions.add(function_name)
            self.function_capabilities[function_name] = FunctionCapability(
                function_name=function_name,
            )
    
    def record_call(
        self,
        function_name: str,
        success: bool,
        episode: int,
        allowed_functions: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录函数调用结果"""
        if not self._is_valid_function_call_input(
            function_name=function_name,
            success=success,
            episode=episode,
            allowed_functions=allowed_functions,
        ):
            self.rejected_inputs += 1
            return

        if function_name not in self.function_capabilities:
            self.add_function(function_name)
        
        cap = self.function_capabilities[function_name]
        cap.total_calls += 1
        cap.last_used_episode = episode
        
        if success:
            cap.success_count += 1
            cap.last_successful_episode = episode
            # 更新等级
            if cap.success_count >= 5 and cap.success_rate > 0.8:
                cap.level = CapabilityLevel.RELIABLE
            elif cap.success_count >= 3 and cap.success_rate > 0.6:
                cap.level = CapabilityLevel.CONFIRMED
            elif cap.success_count >= 1:
                cap.level = CapabilityLevel.TESTED_SUCCESS
        else:
            cap.failure_count += 1
            if cap.failure_count >= 3:
                cap.level = CapabilityLevel.TESTED_FAILURE
        
        # 重新计算置信度
        cap.confidence = min(1.0, cap.success_rate * (1.0 + 0.1 * cap.total_calls))
        self._record_contextual_call(function_name, success, episode, context)
        
        # 更新统计
        self._recompute_stats()

    def record_auxiliary_signal(
        self,
        channel: str,
        identifier: str,
        success: bool,
        episode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录非函数能力画像的辅助统计。"""
        if not isinstance(channel, str) or not channel.strip():
            channel = "unknown_aux"
        if not isinstance(identifier, str) or not identifier.strip():
            identifier = "unknown_identifier"
        if not isinstance(episode, int):
            episode = -1
        success = bool(success)
        metadata = metadata if isinstance(metadata, dict) else {}

        bucket = self.auxiliary_channels.setdefault(channel, {})
        stat = bucket.setdefault(identifier, {
            'total': 0,
            'success': 0,
            'failure': 0,
            'last_episode': None,
            'last_success_episode': None,
            'metadata': {},
        })
        stat['total'] += 1
        stat['last_episode'] = episode
        if success:
            stat['success'] += 1
            stat['last_success_episode'] = episode
        else:
            stat['failure'] += 1
        if metadata:
            stat['metadata'].update(metadata)

    def _is_valid_function_call_input(
        self,
        function_name: Any,
        success: Any,
        episode: Any,
        allowed_functions: Optional[Set[str]] = None,
    ) -> bool:
        """输入校验，避免污染函数能力画像。"""
        if not isinstance(function_name, str):
            return False
        if not isinstance(success, bool):
            return False
        if not isinstance(episode, int):
            return False
        if not function_name or not self._FUNCTION_NAME_PATTERN.match(function_name):
            return False
        lowered = function_name.lower()
        if lowered.startswith('hyp_'):
            return False
        if lowered.startswith('obj_'):
            return False
        if not (allowed_functions is None or function_name in allowed_functions):
            return False
        return True
    
    def _recompute_stats(self) -> None:
        """重新计算统计"""
        self.reliable_count = 0
        self.confirmed_count = 0
        self.tested_success_count = 0
        self.tested_failure_count = 0
        self.unknown_count = 0
        
        for cap in self.function_capabilities.values():
            if cap.level == CapabilityLevel.RELIABLE:
                self.reliable_count += 1
            elif cap.level == CapabilityLevel.CONFIRMED:
                self.confirmed_count += 1
            elif cap.level == CapabilityLevel.TESTED_SUCCESS:
                self.tested_success_count += 1
            elif cap.level == CapabilityLevel.TESTED_FAILURE:
                self.tested_failure_count += 1
            else:
                self.unknown_count += 1
    
    def get_capability(self, function_name: str) -> Optional[FunctionCapability]:
        """获取函数能力"""
        return self.function_capabilities.get(function_name)

    def get_capability_for_context(
        self,
        function_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[FunctionCapability]:
        """
        获取上下文化能力画像。
        如果上下文命中，则返回按上下文聚合的能力；否则回退到全局能力。
        """
        global_cap = self.get_capability(function_name)
        ctx_key = self._context_key(context)
        if not ctx_key:
            return global_cap
        fn_stats = self.contextual_capabilities.get(function_name, {})
        stat = fn_stats.get(ctx_key)
        if not isinstance(stat, dict):
            return global_cap
        total_calls = int(stat.get('total_calls', 0) or 0)
        success_count = int(stat.get('success_count', 0) or 0)
        failure_count = int(stat.get('failure_count', 0) or 0)
        context_confidence = float(stat.get('confidence', 0.0) or 0.0)
        context_level = CapabilityLevel.UNKNOWN
        if total_calls >= 5 and total_calls > 0 and (success_count / total_calls) > 0.8:
            context_level = CapabilityLevel.RELIABLE
        elif total_calls >= 3 and total_calls > 0 and (success_count / total_calls) > 0.6:
            context_level = CapabilityLevel.CONFIRMED
        elif success_count > 0:
            context_level = CapabilityLevel.TESTED_SUCCESS
        elif failure_count >= 3:
            context_level = CapabilityLevel.TESTED_FAILURE
        return FunctionCapability(
            function_name=function_name,
            level=context_level,
            success_count=success_count,
            failure_count=failure_count,
            total_calls=total_calls,
            confidence=context_confidence,
            last_used_episode=stat.get('last_used_episode'),
            last_successful_episode=stat.get('last_successful_episode'),
        )
    
    def get_reliable_functions(self) -> List[str]:
        """获取可靠函数列表"""
        return [
            fn for fn, cap in self.function_capabilities.items()
            if cap.is_reliable
        ]
    
    def get_best_function(self, preferred_functions: List[str]) -> Optional[str]:
        """从候选中选择最佳函数"""
        best = None
        best_score = -1
        
        for fn in preferred_functions:
            cap = self.function_capabilities.get(fn)
            if cap:
                score = cap.success_rate * cap.confidence
                if score > best_score:
                    best_score = score
                    best = fn
        
        return best
    
    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'discovered_functions': list(self.discovered_functions),
            'function_capabilities': {
                fn: cap.to_dict() for fn, cap in self.function_capabilities.items()
            },
            'stats': {
                'reliable': self.reliable_count,
                'confirmed': self.confirmed_count,
                'tested_success': self.tested_success_count,
                'tested_failure': self.tested_failure_count,
                'unknown': self.unknown_count,
            },
            'auxiliary_channels': self.auxiliary_channels,
            'rejected_inputs': self.rejected_inputs,
            'contextual_capabilities': self.contextual_capabilities,
        }

    def _context_key(self, context: Optional[Dict[str, Any]]) -> str:
        if not isinstance(context, dict):
            return ""
        task_family = str(context.get('task_family', 'unknown') or 'unknown')
        phase = str(context.get('phase', 'unknown') or 'unknown')
        observation_mode = str(context.get('observation_mode', 'unknown') or 'unknown')
        resource_band = str(context.get('resource_band', 'normal') or 'normal')
        return f"task_family={task_family}|phase={phase}|observation_mode={observation_mode}|resource_band={resource_band}"

    def _record_contextual_call(
        self,
        function_name: str,
        success: bool,
        episode: int,
        context: Optional[Dict[str, Any]],
    ) -> None:
        ctx_key = self._context_key(context)
        if not ctx_key:
            return
        fn_stats = self.contextual_capabilities.setdefault(function_name, {})
        stat = fn_stats.setdefault(ctx_key, {
            'task_family': str((context or {}).get('task_family', 'unknown') or 'unknown'),
            'phase': str((context or {}).get('phase', 'unknown') or 'unknown'),
            'observation_mode': str((context or {}).get('observation_mode', 'unknown') or 'unknown'),
            'resource_band': str((context or {}).get('resource_band', 'normal') or 'normal'),
            'success_count': 0,
            'failure_count': 0,
            'total_calls': 0,
            'confidence': 0.0,
            'last_used_episode': None,
            'last_successful_episode': None,
        })
        stat['total_calls'] += 1
        stat['last_used_episode'] = episode
        if success:
            stat['success_count'] += 1
            stat['last_successful_episode'] = episode
        else:
            stat['failure_count'] += 1
        success_rate = (stat['success_count'] / stat['total_calls']) if stat['total_calls'] else 0.0
        stat['confidence'] = min(1.0, success_rate * (1.0 + 0.1 * stat['total_calls']))

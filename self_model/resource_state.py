"""
self_model/resource_state.py

Sprint 5: self_model/ 自我认知

追踪资源状态.

Rules:
- 第一版只做简单资源追踪
- 不做复杂预算管理
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResourceBudget:
    """资源预算"""
    max_ticks: int = 50
    max_tests: int = 10
    max_hypotheses: int = 20
    max_commits_per_episode: int = 5
    
    # 当前消耗
    ticks_used: int = 0
    tests_run: int = 0
    hypotheses_created: int = 0
    commits_this_episode: int = 0
    
    @property
    def ticks_remaining(self) -> int:
        return max(0, self.max_ticks - self.ticks_used)
    
    @property
    def tests_remaining(self) -> int:
        return max(0, self.max_tests - self.tests_run)
    
    @property
    def hypotheses_remaining(self) -> int:
        return max(0, self.max_hypotheses - self.hypotheses_created)
    
    @property
    def commits_remaining(self) -> int:
        return max(0, self.max_commits_per_episode - self.commits_this_episode)
    
    @property
    def is_exhausted(self) -> bool:
        """是否资源耗尽"""
        return (
            self.ticks_remaining == 0 or
            self.tests_remaining == 0 or
            self.hypotheses_remaining == 0
        )
    
    def to_dict(self) -> dict:
        return {
            'max_ticks': self.max_ticks,
            'max_tests': self.max_tests,
            'max_hypotheses': self.max_hypotheses,
            'max_commits_per_episode': self.max_commits_per_episode,
            'ticks_used': self.ticks_used,
            'ticks_remaining': self.ticks_remaining,
            'tests_run': self.tests_run,
            'tests_remaining': self.tests_remaining,
            'hypotheses_created': self.hypotheses_created,
            'hypotheses_remaining': self.hypotheses_remaining,
            'commits_this_episode': self.commits_this_episode,
            'commits_remaining': self.commits_remaining,
            'is_exhausted': self.is_exhausted,
        }


@dataclass 
class StateIndicators:
    """状态指标"""
    # 探索 vs 利用
    exploration_ratio: float = 0.5  # 0.0 = pure exploitation, 1.0 = pure exploration
    
    # 成功率趋势
    recent_success_rate: float = 0.5  # 最近 5 次的成功率
    
    # 活跃度
    active_hypotheses_count: int = 0
    active_tests_count: int = 0
    
    # Reward 趋势
    reward_trend: str = "neutral"  # "positive", "neutral", "negative"
    cumulative_reward: float = 0.0
    
    # Memory 压力
    memory_objects_count: int = 0
    memory_utilization: float = 0.0  # 0.0-1.0
    
    def to_dict(self) -> dict:
        return {
            'exploration_ratio': self.exploration_ratio,
            'recent_success_rate': self.recent_success_rate,
            'active_hypotheses_count': self.active_hypotheses_count,
            'active_tests_count': self.active_tests_count,
            'reward_trend': self.reward_trend,
            'cumulative_reward': self.cumulative_reward,
            'memory_objects_count': self.memory_objects_count,
            'memory_utilization': self.memory_utilization,
        }


class ResourceState:
    """
    资源状态管理器.
    
    第一版职责:
    1. 追踪资源预算消耗
    2. 追踪状态指标
    3. 提供资源查询
    
    不做:
    - 复杂预算分配
    - 动态资源调整
    """
    
    def __init__(self):
        self._budget = ResourceBudget()
        self._indicators = StateIndicators()
        self._episode_budgets: List[ResourceBudget] = []
    
    @property
    def budget(self) -> ResourceBudget:
        return self._budget
    
    @property
    def indicators(self) -> StateIndicators:
        return self._indicators
    
    def new_episode(self) -> None:
        """开始新 episode，重置预算"""
        # 保存当前预算历史
        self._episode_budgets.append(self._budget)
        
        # 重置
        self._budget = ResourceBudget()
    
    def consume_tick(self) -> bool:
        """消耗一个 tick，返回是否成功"""
        if self._budget.ticks_remaining <= 0:
            return False
        self._budget.ticks_used += 1
        return True
    
    def consume_test(self) -> bool:
        """消耗一个测试，返回是否成功"""
        if self._budget.tests_remaining <= 0:
            return False
        self._budget.tests_run += 1
        return True
    
    def consume_hypothesis(self) -> bool:
        """消耗一个 hypothesis slot，返回是否成功"""
        if self._budget.hypotheses_remaining <= 0:
            return False
        self._budget.hypotheses_created += 1
        return True
    
    def consume_commit(self) -> bool:
        """消耗一个 commit slot，返回是否成功"""
        if self._budget.commits_remaining <= 0:
            return False
        self._budget.commits_this_episode += 1
        return True
    
    def update_exploration_ratio(self, new_ratio: float) -> None:
        """更新探索比例"""
        self._indicators.exploration_ratio = max(0.0, min(1.0, new_ratio))
    
    def update_success_rate(self, recent_successes: int, recent_total: int) -> None:
        """更新成功率"""
        if recent_total > 0:
            self._indicators.recent_success_rate = recent_successes / recent_total
    
    def update_reward_trend(self, trend: str) -> None:
        """更新 reward 趋势"""
        self._indicators.reward_trend = trend
    
    def update_memory_state(self, objects_count: int, utilization: float) -> None:
        """更新 memory 状态"""
        self._indicators.memory_objects_count = objects_count
        self._indicators.memory_utilization = utilization
    
    def update_active_counts(
        self,
        hypotheses: int,
        tests: int,
    ) -> None:
        """更新活跃计数"""
        self._indicators.active_hypotheses_count = hypotheses
        self._indicators.active_tests_count = tests
    
    def should_exploit(self) -> bool:
        """是否应该利用（而非探索）"""
        return self._indicators.exploration_ratio < 0.3 or self._indicators.reward_trend == "negative"
    
    def should_explore(self) -> bool:
        """是否应该探索"""
        return self._indicators.exploration_ratio > 0.7 or self._indicators.reward_trend == "positive"

    def is_tight_budget(self) -> bool:
        """Whether any major budget axis is close to exhaustion."""
        ratios = [
            (self._budget.ticks_remaining / self._budget.max_ticks) if self._budget.max_ticks > 0 else 0.0,
            (self._budget.tests_remaining / self._budget.max_tests) if self._budget.max_tests > 0 else 0.0,
            (self._budget.hypotheses_remaining / self._budget.max_hypotheses) if self._budget.max_hypotheses > 0 else 0.0,
            (self._budget.commits_remaining / self._budget.max_commits_per_episode) if self._budget.max_commits_per_episode > 0 else 0.0,
        ]
        return min(ratios) <= 0.2

    def budget_band(self) -> str:
        """Return coarse budget band: tight / normal / ample."""
        ratios = [
            (self._budget.ticks_remaining / self._budget.max_ticks) if self._budget.max_ticks > 0 else 0.0,
            (self._budget.tests_remaining / self._budget.max_tests) if self._budget.max_tests > 0 else 0.0,
            (self._budget.hypotheses_remaining / self._budget.max_hypotheses) if self._budget.max_hypotheses > 0 else 0.0,
            (self._budget.commits_remaining / self._budget.max_commits_per_episode) if self._budget.max_commits_per_episode > 0 else 0.0,
        ]
        floor = min(ratios)
        if floor <= 0.2:
            return 'tight'
        if floor >= 0.6:
            return 'ample'
        return 'normal'
    
    def get_advisory_signals(self) -> Dict[str, Any]:
        """获取咨询信号，供 decision/planner 使用"""
        return {
            'should_exploit': self.should_exploit(),
            'should_explore': self.should_explore(),
            'budget_exhausted': self._budget.is_exhausted,
            'budget_tight': self.is_tight_budget(),
            'budget_band': self.budget_band(),
            'ticks_remaining': self._budget.ticks_remaining,
            'tests_remaining': self._budget.tests_remaining,
            'reward_trend': self._indicators.reward_trend,
            'recent_success_rate': self._indicators.recent_success_rate,
            'memory_pressure': self._indicators.memory_utilization > 0.8,
        }
    
    def to_dict(self) -> dict:
        return {
            'budget': self._budget.to_dict(),
            'indicators': self._indicators.to_dict(),
            'episode_history_count': len(self._episode_budgets),
        }

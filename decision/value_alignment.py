"""
decision/value_alignment.py

Sprint 1: value uncertainty trace

为价值模型提供目标健康评估与对齐策略，避免在目标状态不健康时盲目提高 alignment。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GoalHealthAssessment:
    """目标健康度评估结果。"""

    goal_id: str
    health_score: float
    confidence_score: float
    progress_score: float
    evidence_score: float

    @property
    def is_healthy(self) -> bool:
        return self.health_score >= 0.55


class ValueAlignmentPolicy:
    """价值对齐策略：在目标健康度低时抑制盲目 alignment。"""

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _read_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def assess_goal_health(self, top_goal: Any, context: Dict[str, Any]) -> GoalHealthAssessment:
        """结合 top_goal 结构化属性和 context 信号评估目标健康度。"""
        goal_data: Dict[str, Any]
        if isinstance(top_goal, dict):
            goal_data = top_goal
            goal_id = str(goal_data.get('goal_id') or '')
        else:
            goal_data = {
                'confidence': getattr(top_goal, 'confidence', None),
                'progress': getattr(top_goal, 'progress', None),
                'evidence_count': getattr(top_goal, 'evidence_count', None),
                'stability': getattr(top_goal, 'stability', None),
            }
            goal_id = str(getattr(top_goal, 'goal_id', '') or '')

        confidence = self._read_float(goal_data.get('confidence', 0.5), 0.5)
        progress = self._read_float(goal_data.get('progress', 0.5), 0.5)
        evidence_count = self._read_float(goal_data.get('evidence_count', 0.0), 0.0)
        stability = self._read_float(goal_data.get('stability', 0.5), 0.5)

        goal_health_ctx = context.get('goal_health', {})
        if isinstance(goal_health_ctx, dict):
            confidence = self._read_float(goal_health_ctx.get('confidence', confidence), confidence)
            progress = self._read_float(goal_health_ctx.get('progress', progress), progress)
            stability = self._read_float(goal_health_ctx.get('stability', stability), stability)

        evidence_score = self._clamp(min(1.0, evidence_count / 3.0))
        confidence_score = self._clamp((confidence + stability) / 2.0)
        progress_score = self._clamp(progress)

        health_score = self._clamp(
            confidence_score * 0.45 +
            progress_score * 0.35 +
            evidence_score * 0.20
        )

        return GoalHealthAssessment(
            goal_id=goal_id,
            health_score=health_score,
            confidence_score=confidence_score,
            progress_score=progress_score,
            evidence_score=evidence_score,
        )

    def adjust_alignment(
        self,
        alignment: float,
        assessment: GoalHealthAssessment,
        candidate_meta: Dict[str, Any],
    ) -> float:
        """在目标健康度差时降低盲目 alignment。"""
        adjusted = alignment

        if assessment.health_score < 0.4 and alignment > 0.55:
            adjusted = alignment - 0.25
        elif assessment.health_score < 0.55 and alignment > 0.7:
            adjusted = alignment - 0.12

        if candidate_meta.get('planner_matches_step') and not assessment.is_healthy:
            adjusted -= 0.08

        return self._clamp(adjusted)

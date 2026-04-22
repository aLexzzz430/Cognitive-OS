"""
representations/symptom_graph.py

症状节点（SymptomNode）和假设卡（HypothesisCard）架构
====================================================

核心区分：
  RepresentationCard（症状卡）    → 描述"发生了什么现象"
  HypothesisCard（假设卡）         → 解释"为什么会发生"
  DiscriminatingTestCard（区分卡） → 主动探测"哪个解释更可能"

SymptomNode
-----------
一个症状节点代表一个经过验证的现象（从 B1 来）。
它本身不解释原因，只是说"这个结构在这里出现了"。

属性：
  anchor_card_id: 指向已验证的 RepresentationCard
  activation_triggers: 什么条件下这个症状被观察到
  linked_hypotheses: 围绕这个症状的候选假设列表

HypothesisCard
--------------
一个假设卡代表一个因果解释。
它挂在某个症状节点上，提供：
  mechanism: 什么机制导致了这个症状
  discriminating_conditions: 什么条件能区分这个假设和其他假设
  probe: 如果探测，探测什么（最小成本）
  retirement_conditions: 什么观测会让这个假设被否定

DiscriminatingTestCard
----------------------
执行区分测试的卡。
  hypothesis_a vs hypothesis_b
  probe_action: 最小探测动作
  probe_cost: 能量/时间成本
  outcomes: 各种观测结果 → 假设权重更新
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ============================================================
# 症状节点
# ============================================================

@dataclass
class SymptomNode:
    """
    症状节点：经过验证的现象描述。

    例如 progress-stall 的症状是：
      "progress_delta ≤ 1 AND hazard_delta ≥ 2"
    这个结构本身是经过验证的，但它不解释"为什么"。
    """
    symptom_id: str
    anchor_card_id: str           # 指向 RepresentationCard
    phenomenon_description: str   # 人类可读：什么现象被观察到了
    structure_signature: dict      # 具体的 threshold_conditions
    linked_hypotheses: list[str] = field(default_factory=list)  # 假设卡ID列表

    def __repr__(self):
        return f"SymptomNode({self.symptom_id}, anchor={self.anchor_card_id}, hypotheses={len(self.linked_hypotheses)})"


# ============================================================
# 假设卡
# ============================================================

@dataclass
class DiscriminatingCondition:
    """一个区分条件：观测 key + 操作符 + 值 + 描述"""
    observation_key: str
    operator: str    # eq, lt, gt, lte, gte, delta_gte, delta_lte, peaked, stable
    value: any
    description: str
    # 这个条件对这个假设的支持程度（0-1）
    support_weight: float = 1.0


@dataclass
class ProbeSpec:
    """探测规格：最小探测动作"""
    action: str           # 执行什么动作
    observation_key: str  # 观测什么变量
    expected_outcome_if_true: str
    expected_outcome_if_false: str
    cost_energy: float   # 能量成本
    cost_steps: int      # 时间成本


@dataclass
class HypothesisCard:
    """
    假设卡：对一个症状节点的因果解释。

    核心属性：
      hypothesis_id: 唯一标识
      symptom_id: 挂靠在哪个症状节点
      mechanism: 什么机制导致了这个症状
      confidence: 当前置信度 0-1
      discriminating_conditions: 区分条件列表
      probe: 如果需要探测，最小探测规格
      retirement_conditions: 什么情况下这个假设被否定
      prior_count: 被先验激活次数
      successful_count: 成功预测次数
    """
    hypothesis_id: str
    symptom_id: str
    mechanism: str                     # 人类可读：什么因果机制
    confidence: float = 0.5            # 先验置信度

    # 区分条件：机器可执行
    discriminating_conditions: list[DiscriminatingCondition] = field(default_factory=list)

    # 探测规格（如果需要主动探测）
    probe: ProbeSpec | None = None

    # 退休条件：触发则置信度→0
    retirement_conditions: list[str] = field(default_factory=list)

    # 统计
    prior_count: int = 0
    successful_count: int = 0

    def support_probability(self) -> float:
        """基于历史计算后验置信度"""
        if self.prior_count == 0:
            return self.confidence
        return self.successful_count / self.prior_count

    def update_confidence(self, success: bool):
        self.prior_count += 1
        if success:
            self.successful_count += 1

    def should_retire(self) -> bool:
        """检查是否应该退休"""
        if self.prior_count >= 3:
            success_rate = self.successful_count / self.prior_count
            if success_rate < 0.2:
                return True
        return False

    def __repr__(self):
        return f"HypothesisCard({self.hypothesis_id}, conf={self.confidence:.2f}, n={self.prior_count})"


# ============================================================
# 症状图谱：管理所有症状节点和假设卡
# ============================================================

class SymptomGraph:
    """
    症状图谱：管理症状节点和假设卡的关系。
    """

    def __init__(self):
        self.symptoms: dict[str, SymptomNode] = {}
        self.hypotheses: dict[str, HypothesisCard] = {}

    def add_symptom(self, symptom: SymptomNode):
        self.symptoms[symptom.symptom_id] = symptom

    def add_hypothesis(self, hypothesis: HypothesisCard):
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        # Link to symptom
        if hypothesis.symptom_id in self.symptoms:
            s = self.symptoms[hypothesis.symptom_id]
            if hypothesis.hypothesis_id not in s.linked_hypotheses:
                s.linked_hypotheses.append(hypothesis.hypothesis_id)

    def get_symptom(self, symptom_id: str) -> SymptomNode | None:
        return self.symptoms.get(symptom_id)

    def get_hypothesis(self, hyp_id: str) -> HypothesisCard | None:
        return self.hypotheses.get(hyp_id)

    def get_hypotheses_for_symptom(self, symptom_id: str) -> list[HypothesisCard]:
        symptom = self.symptoms.get(symptom_id)
        if not symptom:
            return []
        return [self.hypotheses[hid] for hid in symptom.linked_hypotheses
                if hid in self.hypotheses]

    def get_competing_hypotheses(self, symptom_id: str) -> list[HypothesisCard]:
        """返回同一个症状节点下的所有假设（互相竞争）"""
        return self.get_hypotheses_for_symptom(symptom_id)

    def discriminate(self, symptom_id: str, observation: dict) -> dict[str, float]:
        """
        给定症状节点和当前观测，计算每个假设的支持概率。
        返回 {hypothesis_id: support_score}
        """
        hyps = self.get_hypotheses_for_symptom(symptom_id)
        if not hyps:
            return {}

        scores = {}
        for hyp in hyps:
            score = self._compute_support(hyp, observation)
            scores[hyp.hypothesis_id] = score
        return scores

    def _compute_support(self, hyp: HypothesisCard, obs: dict) -> float:
        """
        计算一个假设在当前观测下的支持度。
        每个 discriminating_condition 检查：
          - 如果满足 → 增加支持
          - 如果不满足 → 降低支持
        """
        if not hyp.discriminating_conditions:
            return hyp.support_probability()

        satisfied = 0
        total_weight = 0.0
        for dc in hyp.discriminating_conditions:
            total_weight += dc.support_weight
            if self._check_condition(dc, obs):
                satisfied += dc.support_weight
            # 不满足 → 减少支持
            else:
                satisfied -= dc.support_weight * 0.5

        if total_weight == 0:
            return hyp.support_probability()

        raw_score = satisfied / total_weight
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, raw_score))
        # Blend with prior
        prior = hyp.support_probability()
        return 0.3 * prior + 0.7 * score

    def _check_condition(self, dc: DiscriminatingCondition, obs: dict) -> bool:
        """检查单个区分条件是否满足"""
        key = dc.observation_key
        op = dc.operator
        val = dc.value
        obs_val = obs.get(key)

        if obs_val is None:
            return False

        if op == "eq":
            return obs_val == val
        elif op == "lt":
            return obs_val < val
        elif op == "lte":
            return obs_val <= val
        elif op == "gt":
            return obs_val > val
        elif op == "gte":
            return obs_val >= val
        elif op == "delta_gte":
            history = obs.get(f"{key}_history", [])
            if len(history) < 2:
                return False
            delta = history[-1] - history[0]
            return delta >= val
        elif op == "delta_lte":
            history = obs.get(f"{key}_history", [])
            if len(history) < 2:
                return False
            delta = history[-1] - history[0]
            return delta <= val
        elif op == "peaked":
            history = obs.get(f"{key}_history", [])
            if len(history) < 3:
                return False
            # peak must be in middle of history
            peak_val = max(history)
            peak_idx = history.index(peak_val)
            if peak_idx == 0 or peak_idx == len(history) - 1:
                return False
            return peak_val >= val
        elif op == "stable":
            history = obs.get(f"{key}_history", [])
            if len(history) < 2:
                return False
            max_v = max(history)
            min_v = min(history)
            return (max_v - min_v) <= val
        return False

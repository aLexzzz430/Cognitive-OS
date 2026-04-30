"""
modules/representations/proposer_llm.py

A6: B1 自主候选提案 — 直接改 LLM 版

B1 的核心任务本来是表征候选提案，而不是正式裁决。
LLM 可以成为主力 proposer，再由对象系统筛、验、记账。

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

from modules.llm.capabilities import (
    REPRESENTATION_CARD_PROPOSAL,
    REPRESENTATION_FALSE_ABSTRACTION_CHECK,
)
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.json_adaptor import normalize_llm_output

@dataclass
class TrajectoryEntry:
    """单条轨迹条目：observation → action → outcome"""
    tick: int
    observation: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float = 0.0


@dataclass
class RepresentationCard:
    """
    候选 RepresentationCard — 由 LLM 自主生成。
    字段对齐原始 B1 实现。
    """
    card_id: str
    origin_family: str
    origin_trace: str
    proposed_pattern: str
    abstraction_level: str  # "raw_rename" | "threshold_heuristic" | "structural_pattern" | "invariant"
    confidence: float
    retrieval_count: int = 0
    activation_count: int = 0
    activation_reasons: List[str] = field(default_factory=list)
    whitelist: bool = False
    weight_adjustment: float = 0.0
    false_abstraction_flags: List[str] = field(default_factory=list)
    step5_consumed: bool = False
    step5_effect: Optional[float] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class ValidationResult:
    valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    false_abstraction_flags: List[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0


class LLMRepresentationProposer:
    """
    B1 直接改 LLM 版：从轨迹流自主生成 RepresentationCard 候选。

    LLM 能力：
    - 从轨迹中发现模式（proposed_pattern）
    - 判断 abstraction_level
    - 评估 confidence

    验证仍走对象系统（validate_in_core）：
    - 检查候选格式
    - false_abstraction_flags 检查
    - Step10 记账
    """

    LLM_ROUTE_NAME = "representation"
    LLM_CAPABILITY_NAMESPACE = "representation"

    def __init__(self, llm_client=None, card_store=None):
        """
        Args:
            llm_client: LLM API client. If None, uses rule-based fallback.
            card_store: CardWarehouse instance for validation and storage.
        """
        self._llm_gateway = ensure_llm_gateway(
            llm_client,
            route_name=self.LLM_ROUTE_NAME,
            capability_prefix=self.LLM_CAPABILITY_NAMESPACE,
        )
        self._llm = self._llm_gateway
        self._card_store = card_store

    def _llm_available(self) -> bool:
        return self._llm_gateway is not None and bool(self._llm_gateway.is_available())

    def _request_text(self, capability: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_gateway is None:
            return ""
        return self._llm_gateway.request_text(capability, prompt, **kwargs)

    # ─────────────────────────────────────────────────
    # Core: Propose from trajectories
    # ─────────────────────────────────────────────────

    def propose_from_trajectories(
        self,
        trajectories: List[List[TrajectoryEntry]],
        families: List[str],
    ) -> List[RepresentationCard]:
        """
        从多条轨迹生成 RepresentationCard 候选。

        Args:
            trajectories: 每条轨迹是一个 TrajectoryEntry 列表
            families: 对应的 family 名称列表

        Returns:
            List of proposed RepresentationCards
        """
        if not trajectories:
            return []

        # Build trajectory summary for LLM
        trace_summaries = []
        for i, (trace, family) in enumerate(zip(trajectories, families)):
            lines = [f"Family {family} trajectory {i+1}:"]
            for entry in trace[-5:]:  # Last 5 ticks
                action_str = str(entry.action.get('payload', {}).get('tool_args', {}).get('function_name', '?'))
                outcome_str = str(entry.outcome)[:80]
                lines.append(f"  tick{entry.tick}: action={action_str} → {outcome_str[:60]}")
            trace_summaries.append('\n'.join(lines))

        if not self._llm_available():
            # Rule-based fallback: simple pattern detection
            return self._rule_based_propose(trajectories, families)

        prompt = f"""You are proposing RepresentationCards from agent trajectories.

Trajectories:
{chr(10).join(trace_summaries)}

Task: For each trajectory, identify 1-2 recurring patterns that could become
stable representations (RepresentationCards).

For each pattern, determine:
1. proposed_pattern: what the pattern describes (2-3 sentences)
2. abstraction_level: one of:
   - "raw_rename" (just renaming, no real abstraction) — reject these
   - "threshold_heuristic" (threshold-based, fragile)
   - "structural_pattern" (genuine structural pattern)
   - "invariant" (true invariant that holds across episodes)
3. confidence: 0.0-1.0 based on how stable the pattern appears

Format your response as a JSON list of cards:
[
  {{"origin_family": "family_name", "proposed_pattern": "...", "abstraction_level": "...", "confidence": 0.X}},
  ...
]

Return ONLY the JSON list, nothing else."""

        response = self._request_text(REPRESENTATION_CARD_PROPOSAL, prompt)
        try:
            normalized = normalize_llm_output(
                response,
                output_kind="representation_card_proposal",
                expected_type="list",
            )
            raw_cards = normalized.parsed_list()
            cards = []
            for idx, raw in enumerate(raw_cards):
                if not isinstance(raw, dict):
                    continue
                family = raw.get('origin_family') or (
                    families[idx] if idx < len(families) else 'unknown'
                )
                card = RepresentationCard(
                    card_id=f"card_{family}_{int(time.time()*1000)%100000}",
                    origin_family=family,
                    origin_trace=raw.get('origin_trace', ''),
                    proposed_pattern=raw.get('proposed_pattern', ''),
                    abstraction_level=raw.get('abstraction_level', 'structural_pattern'),
                    confidence=raw.get('confidence', 0.5),
                )
                cards.append(card)
            return cards
        except Exception:
            return self._rule_based_propose(trajectories, families)

    # ─────────────────────────────────────────────────
    # Validation: validate_in_core()
    # ─────────────────────────────────────────────────

    def validate_candidate(self, card: RepresentationCard) -> ValidationResult:
        """
        对象系统验证候选。

        检查：
        1. 格式完整性
        2. false_abstraction_flags（淘汰 rename/threshold tricks）
        3. abstraction_level 合理性
        """
        reasons = []
        flags = []

        # Check abstraction_level
        if card.abstraction_level == 'raw_rename':
            flags.append('raw_rename')
            reasons.append('Reject: raw_rename is not a genuine abstraction')

        if card.abstraction_level == 'threshold_heuristic':
            flags.append('threshold_heuristic')
            reasons.append('Warning: threshold_heuristic is fragile')

        # Check confidence
        if card.confidence < 0.2:
            reasons.append(f'Low confidence: {card.confidence}')

        # Check pattern length
        if len(card.proposed_pattern) < 20:
            reasons.append('Pattern description too short')

        valid = len(flags) == 0 and card.confidence >= 0.2

        return ValidationResult(
            valid=valid,
            rejection_reasons=reasons if not valid else [],
            false_abstraction_flags=flags,
            confidence_adjustment=-0.1 if 'threshold_heuristic' in flags else 0.0,
        )

    # ─────────────────────────────────────────────────
    # False Abstraction Check
    # ─────────────────────────────────────────────────

    def false_abstraction_check(self, card: RepresentationCard) -> List[str]:
        """
        检查候选是否属于 false abstraction。

        淘汰类型：
        - rename: 只是换个名字
        - threshold tricks: 硬编码阈值
        - surface label: 表面标签没有结构性
        """
        flags = []

        if not self._llm_available():
            # Rule-based fallback
            pattern_lower = card.proposed_pattern.lower()
            if any(kw in pattern_lower for kw in ['rename', 'renamed', 'name', 'label']):
                flags.append('raw_rename')
            if any(kw in pattern_lower for kw in ['threshold', 'greater than', 'less than', '> ', '< ']):
                flags.append('threshold_heuristic')
            return flags

        prompt = f"""Check if this RepresentationCard is a false abstraction:

Pattern: {card.proposed_pattern}
Abstraction level: {card.abstraction_level}
Origin family: {card.origin_family}

Answer YES if it is a false abstraction (just renaming, hardcoded threshold, or surface label).
Answer NO if it is a genuine structural pattern.

Return ONLY YES or NO."""

        response = self._request_text(REPRESENTATION_FALSE_ABSTRACTION_CHECK, prompt).strip().upper()
        if response == 'YES':
            flags.append('false_abstraction')
        return flags

    # ─────────────────────────────────────────────────
    # Step10: commit_via_step10()
    # ─────────────────────────────────────────────────

    def commit_via_step10(self, card: RepresentationCard, object_store) -> str:
        """
        将验证通过的候选写入 ObjectStore（走 Step 10 路径）。

        Returns object_id if committed, empty string if rejected.
        """
        if not self.validate_candidate(card).valid:
            return ""

        proposal = {
            'content': {
                'card_id': card.card_id,
                'proposed_pattern': card.proposed_pattern,
                'abstraction_level': card.abstraction_level,
                'origin_family': card.origin_family,
                'skill_type': 'representation',
            },
            'confidence': card.confidence,
            'content_hash': f"{card.proposed_pattern[:50]}_{card.origin_family}",
        }

        from modules.governance.object_store import ACCEPT_NEW
        obj_id = object_store.add(proposal, ACCEPT_NEW, [])
        return obj_id

    # ─────────────────────────────────────────────────
    # Rule-based fallback (no LLM)
    # ─────────────────────────────────────────────────

    def _rule_based_propose(
        self,
        trajectories: List[List[TrajectoryEntry]],
        families: List[str],
    ) -> List[RepresentationCard]:
        """当没有 LLM client 时的基于规则的 fallback"""
        cards = []
        for trace, family in zip(trajectories, families):
            if len(trace) < 3:
                continue
            # Simple heuristic: if same function called repeatedly
            fn_counts: Dict[str, int] = {}
            for entry in trace:
                fn = entry.action.get('payload', {}).get('tool_args', {}).get('function_name', '')
                if fn:
                    fn_counts[fn] = fn_counts.get(fn, 0) + 1

            for fn, count in fn_counts.items():
                if count >= 3:
                    card = RepresentationCard(
                        card_id=f"card_{family}_{fn}_{int(time.time()*1000)%100000}",
                        origin_family=family,
                        origin_trace=f"fn={fn} count={count}",
                        proposed_pattern=f"Function '{fn}' called {count} times in {family}",
                        abstraction_level='structural_pattern',
                        confidence=min(0.9, count * 0.15),
                    )
                    cards.append(card)
        return cards

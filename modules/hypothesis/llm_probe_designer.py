"""
modules/hypothesis/llm_probe_designer.py

A8: B3 主动探针 — 半 LLM 化（设计助手不做 adjudicator）

B3 通过条件不是"会提问题"，而是 test 必须有：
- target hypotheses
- expected outcomes
- information gain

LLM 能力（设计助手）：
- 提出 probe 候选
- 预测 expected outcomes
- 解释 information gain

不交给 LLM：
- probe adjudicator（DiscriminatingTestEngine 本体）
- target hypotheses 绑定
- information gain 测量

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from modules.llm.capabilities import (
    REASONING_INFORMATION_GAIN_EXPLANATION,
    REASONING_PROBE_DESIGN,
    REASONING_PROBE_OUTCOME_PREDICTION,
    REASONING_PROBE_URGENCY_ADVICE,
)
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.json_adaptor import normalize_llm_output

if TYPE_CHECKING:
    from modules.hypothesis.hypothesis_tracker import Hypothesis, DiscriminatingTest
    from modules.hypothesis.hypothesis_tracker import DiscriminatingTestEngine

from dataclasses import dataclass


@dataclass
class ProbeSpec:
    """Probe 设计规范 — 由 LLM 生成，由 DiscriminatingTestEngine 验证"""
    probe_id: str
    target_function: str  # 要测试的函数
    test_params: dict      # 测试参数
    expected_outcome_if_a_correct: str  # 如果 hypothesis A 正确，预期结果
    expected_outcome_if_b_correct: str  # 如果 hypothesis B 正确，预期结果
    information_gain_hypothesis: str  # 信息增益的来源说明
    confidence: float = 0.5
    llm_rationale: str = ""  # LLM 的设计理由（供审计）


@dataclass
class ProbeDesignContext:
    """给 LLM 的探针设计上下文"""
    episode: int
    tick: int
    hypothesis_a_claim: str
    hypothesis_b_claim: str
    hypothesis_a_confidence: float
    hypothesis_b_confidence: float
    competing_functions: List[str]  # 两个假设涉及的函数
    known_functions: List[str]  # 当前已知的函数


class LLMProbeDesigner:
    """
    B3 半 LLM 化：

    LLM 做设计助手（propose_with_llm）：
    - 生成 probe 候选
    - 预测 expected outcomes
    - 解释 information gain

    不交给 LLM：
    - DiscriminatingTestEngine 本体（adjudicator）
    - information gain 实际测量
    - commit_via_step10（通过 DiscriminatingTestEngine 走对象路径）
    """

    LLM_ROUTE_NAME = "probe"
    LLM_CAPABILITY_NAMESPACE = "reasoning"

    def __init__(self, test_engine, llm_client=None):
        """
        Args:
            test_engine: DiscriminatingTestEngine instance
            llm_client: LLM API client. If None, uses rule-based fallback.
        """
        self._test_engine = test_engine
        self._llm_gateway = ensure_llm_gateway(
            llm_client,
            route_name=self.LLM_ROUTE_NAME,
            capability_prefix=self.LLM_CAPABILITY_NAMESPACE,
        )
        self._llm = self._llm_gateway

    def _llm_available(self) -> bool:
        return self._llm_gateway is not None and bool(self._llm_gateway.is_available())

    def _request_text(self, capability: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_gateway is None:
            return ""
        return self._llm_gateway.request_text(capability, prompt, **kwargs)

    # ─────────────────────────────────────────────────
    # 1. Generate probe candidates
    # ─────────────────────────────────────────────────

    def generate_probe_candidates(
        self,
        competing_pair: Tuple[Any, Any],
        known_functions: List[str],
        top_k: int = 3,
    ) -> List[ProbeSpec]:
        """
        LLM 提出 probe 候选。

        输入：DiscriminatingTestEngine.discriminating_pair() 返回的竞争假设对
        输出：ProbeSpec 列表（供 DiscriminatingTestEngine.generate() 参考）

        原则：LLM 只产出候选设计，实际 test 生成仍走 DiscriminatingTestEngine。
        """
        if not competing_pair or not known_functions:
            return []

        ha, hb = competing_pair
        ctx = ProbeDesignContext(
            episode=getattr(ha, 'created_at_episode', 0),
            tick=getattr(ha, 'created_at_tick', 0),
            hypothesis_a_claim=ha.claim,
            hypothesis_b_claim=hb.claim,
            hypothesis_a_confidence=ha.confidence,
            hypothesis_b_confidence=hb.confidence,
            competing_functions=self._extract_functions([ha, hb]),
            known_functions=known_functions,
        )

        if not self._llm_available():
            return self._rule_based_probes(ctx, known_functions, top_k)

        fn_list = ', '.join(known_functions) if known_functions else 'none'
        ab_diff = abs(ha.confidence - hb.confidence)

        # P3-C: Extract mechanism fields from hypotheses
        ha_trigger = getattr(ha, 'trigger_condition', None) or 'unknown'
        hb_trigger = getattr(hb, 'trigger_condition', None) or 'unknown'
        ha_transition = getattr(ha, 'expected_transition', None) or 'unknown'
        hb_transition = getattr(hb, 'expected_transition', None) or 'unknown'
        ha_falsifiers = getattr(ha, 'falsifiers', []) or []
        hb_falsifiers = getattr(hb, 'falsifiers', []) or []
        all_falsifiers = ', '.join(ha_falsifiers + hb_falsifiers) or 'none specified'

        prompt = f"""You are designing discriminating probes for an agent's hypothesis testing system.

Two hypotheses are competing:
A: {ha.claim} (confidence: {ha.confidence:.2f})
   trigger_condition: {ha_trigger}
   expected_transition: {ha_transition}
   falsifiers: {', '.join(ha_falsifiers) if ha_falsifiers else 'none'}
B: {hb.claim} (confidence: {hb.confidence:.2f})
   trigger_condition: {hb_trigger}
   expected_transition: {hb_transition}
   falsifiers: {', '.join(hb_falsifiers) if hb_falsifiers else 'none'}
Confidence gap: {ab_diff:.2f} (larger gap = less urgent to test)

Known functions available for testing: {fn_list}

Generate {top_k} probe designs. For each probe specify:
1. target_function: which function to call as the test
2. test_params: parameters to pass (simple, focused on distinguishing A vs B)
3. expected_outcome_if_a_correct: what the agent observes if A is right
4. expected_outcome_if_b_correct: what the agent observes if B is right
5. information_gain_hypothesis: WHY this test would give information (what it would eliminate)
6. confidence: 0.0-1.0 how confident are you this probe can discriminate

Return a JSON list of probes:
[
  {{"target_function": "...", "test_params": {{...}}, "expected_outcome_if_a_correct": "...", "expected_outcome_if_b_correct": "...", "information_gain_hypothesis": "...", "confidence": 0.X}},
  ...
]

Return ONLY the JSON list, nothing else."""

        response = self._request_text(REASONING_PROBE_DESIGN, prompt)
        try:
            import time
            normalized = normalize_llm_output(
                response,
                output_kind="probe_design",
                expected_type="list",
            )
            raw_probes = normalized.parsed_list()
            probes = []
            for raw in raw_probes[:top_k]:
                if not isinstance(raw, dict):
                    continue
                probe = ProbeSpec(
                    probe_id=f"probe_{int(time.time()*1000)%100000}",
                    target_function=raw.get('target_function', known_functions[0] if known_functions else 'compute_stats'),
                    test_params=raw.get('test_params', {'data': [1, 2, 3]}),
                    expected_outcome_if_a_correct=raw.get('expected_outcome_if_a_correct', ''),
                    expected_outcome_if_b_correct=raw.get('expected_outcome_if_b_correct', ''),
                    information_gain_hypothesis=raw.get('information_gain_hypothesis', ''),
                    confidence=raw.get('confidence', 0.5),
                    llm_rationale=f"A: {ha.claim[:40]}... vs B: {hb.claim[:40]}...",
                )
                probes.append(probe)
            return probes
        except Exception:
            return self._rule_based_probes(ctx, known_functions, top_k)

    # ─────────────────────────────────────────────────
    # 2. Predict outcomes
    # ─────────────────────────────────────────────────

    def predict_outcomes(self, probe: ProbeSpec, hypothesis: str = 'A') -> str:
        """
        LLM 预测 expected outcomes（供审计用）。

        输入：ProbeSpec + hypothesis 标识（'A' 或 'B'）
        输出：预期观察结果的文字描述
        """
        if not self._llm_available():
            return probe.expected_outcome_if_a_correct if hypothesis == 'A' else probe.expected_outcome_if_b_correct

        field = f'expected_outcome_if_{hypothesis.lower()}_correct'
        expected = getattr(probe, field, '')
        if expected:
            return expected  # Use pre-filled from probe spec

        prompt = f"""You are predicting what the agent would observe if hypothesis {hypothesis} is correct.

Probe: {probe.target_function}({probe.test_params})
Information gain from this probe: {probe.information_gain_hypothesis}

Predict what the agent would observe. Be specific about:
- What return value or effect would be seen?
- Would a discovery event be triggered?
- Would there be an error or error recovery?

Return ONLY the prediction, nothing else."""

        return self._request_text(REASONING_PROBE_OUTCOME_PREDICTION, prompt).strip()

    # ─────────────────────────────────────────────────
    # 3. Explain information gain
    # ─────────────────────────────────────────────────

    def explain_information_gain(self, test_result: Dict[str, Any], probe: ProbeSpec) -> str:
        """
        LLM 解释 information gain（供审计用）。

        在 test 执行后调用，解释本次 test 的信息价值。
        不影响 CoreMainLoop 的 decision，仅供记录和审计。
        """
        if not self._llm_available():
            return f"Probe {probe.target_function}: {'passed' if test_result.get('discovery_event') else 'no discovery'}"

        passed = test_result.get('discovery_event') or test_result.get('correct_function')
        outcome = 'PASSED (hypothesis confirmed)' if passed else 'FAILED (hypothesis refuted)'

        prompt = f"""You are explaining the information gain from a discriminating test.

Probe design rationale: {probe.llm_rationale}
Target function: {probe.target_function}
Test params: {probe.test_params}
Information gain hypothesis: {probe.information_gain_hypothesis}

Test result: {outcome}
Actual discovery: {test_result.get('discovery_event', False)}
Actual error: {test_result.get('error', 'none')}

Explain in 1-2 sentences:
1. Did the test produce the expected information gain?
2. What did we learn about the competing hypotheses?

Return ONLY the explanation, nothing else."""

        return self._request_text(REASONING_INFORMATION_GAIN_EXPLANATION, prompt).strip()

    # ─────────────────────────────────────────────────
    # 4. Should we run a test now? (decision advisory)
    # ─────────────────────────────────────────────────

    def should_run_test(self, competing_pair: Tuple[Any, Any]) -> Dict[str, Any]:
        """
        LLM 判断当前是否应该运行测试（decision advisory）。

        原则：决定由 CoreMainLoop 做出，LLM 只提供建议。
        返回 dict 包含：
        - should_test: bool
        - urgency: 'high' | 'medium' | 'low'
        - reason: str
        """
        if not competing_pair or not self._llm_available():
            ha, hb = competing_pair if competing_pair else (None, None)
            gap = abs(ha.confidence - hb.confidence) if ha and hb else 1.0
            return {
                'should_test': gap < 0.3,  # Rule-based fallback
                'urgency': 'high' if gap < 0.2 else 'medium' if gap < 0.4 else 'low',
                'reason': f'confidence gap={gap:.2f}',
            }

        ha, hb = competing_pair
        gap = abs(ha.confidence - hb.confidence)

        prompt = f"""Two hypotheses are competing:
A: {ha.claim} (confidence: {ha.confidence:.2f})
B: {hb.claim} (confidence: {hb.confidence:.2f})
Confidence gap: {gap:.2f}

Should we run a discriminating test NOW? Consider:
- High urgency: gap is small (< 0.2), both hypotheses are plausible
- Medium urgency: gap is moderate (0.2-0.4)
- Low urgency: gap is large (> 0.4), one hypothesis is much more plausible

Answer with JSON:
{{"should_test": true/false, "urgency": "high/medium/low", "reason": "..."}}

Return ONLY the JSON, nothing else."""

        response = self._request_text(REASONING_PROBE_URGENCY_ADVICE, prompt)
        try:
            normalized = normalize_llm_output(
                response,
                output_kind="probe_urgency",
                expected_type="dict",
            )
            parsed = normalized.parsed_dict()
            return parsed if parsed else {'should_test': gap < 0.3, 'urgency': 'medium', 'reason': 'parse failed'}
        except Exception:
            return {'should_test': gap < 0.3, 'urgency': 'medium', 'reason': 'parse failed'}

    # ─────────────────────────────────────────────────
    # 5. Rule-based fallback
    # ─────────────────────────────────────────────────

    def _rule_based_probes(
        self,
        ctx: ProbeDesignContext,
        known_functions: List[str],
        top_k: int,
    ) -> List[ProbeSpec]:
        """当没有 LLM client 时的基于规则的 fallback"""
        import time
        probes = []
        fn = known_functions[0] if known_functions else 'compute_stats'

        for i in range(min(top_k, 2)):
            probe = ProbeSpec(
                probe_id=f"probe_rb_{int(time.time()*1000)%100000}_{i}",
                target_function=fn,
                test_params={'data': [1, 2, 3, 4, 5]},
                expected_outcome_if_a_correct=f'Discovery event on {fn}',
                expected_outcome_if_b_correct=f'No discovery or error on {fn}',
                information_gain_hypothesis=f'Testing whether {fn} works with default params',
                confidence=0.5,
                llm_rationale='rule_based_fallback',
            )
            probes.append(probe)
        return probes

    def _extract_functions(self, hypotheses: List[Any]) -> List[str]:
        """从 hypothesis claims 中提取函数名"""
        import re
        funcs = []
        for h in hypotheses:
            m = re.search(r"'([^']+)'", h.claim)
            if m:
                fn = m.group(1)
                if fn not in funcs:
                    funcs.append(fn)
        return funcs

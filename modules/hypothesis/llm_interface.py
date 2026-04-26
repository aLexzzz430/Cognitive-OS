"""
modules/hypothesis/llm_interface.py

A7: B2 竞争假设 — LLM 接口（HypothesisTracker 本体不动）

B2 不是单纯"会不会想出假设"，而是要把 competing hypotheses
放进正式对象路径里。LLM 很适合生成和扩展候选假设，
但不适合直接担任 HypothesisTracker 本体。

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()

HypothesisTracker 本体保持 symbolic，不因 LLM 能力强就替换。
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
from modules.llm.capabilities import (
    REASONING_HYPOTHESIS_COMPETITOR_EXPANSION,
    REASONING_HYPOTHESIS_GENERATION,
)
from modules.llm.gateway import ensure_llm_gateway
from modules.llm.thinking_policy import apply_thinking_policy
from modules.hypothesis.hypothesis_tracker import Hypothesis, HypothesisStatus


class LLMHypothesisInterface:
    """
    LLM 生成假设候选，HypothesisTracker 本体不动。

    LLM 能力：
    - 生成候选假设（claim, type, competing_with）
    - 扩展 competing_with 列表

    不交给 LLM：
    - HypothesisTracker.validate_in_core()
    - 状态转换（ACTIVE → CONFIRMED/REFUTED）
    - commit_via_step10()
    """

    LLM_ROUTE_NAME = "hypothesis"
    LLM_CAPABILITY_NAMESPACE = "reasoning"
    _GENERATION_KWARGS = apply_thinking_policy("hypothesis", {
        "max_tokens": 256,
        "temperature": 0.0,
    })
    _COMPETITOR_KWARGS = apply_thinking_policy("hypothesis", {
        "max_tokens": 64,
        "temperature": 0.0,
    })

    def __init__(self, hypothesis_tracker, llm_client=None):
        self._tracker = hypothesis_tracker
        self._llm_gateway = ensure_llm_gateway(
            llm_client,
            route_name=self.LLM_ROUTE_NAME,
            capability_prefix=self.LLM_CAPABILITY_NAMESPACE,
        )
        self._llm = self._llm_gateway
        self._last_llm_error = ""

    def _llm_available(self) -> bool:
        return self._llm_gateway is not None and bool(self._llm_gateway.is_available())

    def _request_text(self, capability: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_gateway is None:
            return ""
        try:
            response = self._llm_gateway.request_text(capability, prompt, **kwargs)
        except Exception as exc:
            self._last_llm_error = f"{type(exc).__name__}: {exc}"
            return ""
        gateway_error = str(getattr(self._llm_gateway, "last_error", "") or "")
        if gateway_error:
            self._last_llm_error = gateway_error
            return ""
        self._last_llm_error = ""
        return str(response or "")

    @property
    def last_llm_error(self) -> str:
        return self._last_llm_error

    def generate_hypothesis_candidates(
        self,
        obs: dict,
        context: str,
        known_functions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        LLM 生成假设候选。

        输出字段对齐 Hypothesis.create_from_object() 格式：
        - hyp_id (placeholder — actual ID assigned by tracker)
        - claim
        - hyp_type: 'function_existence' | 'parameter_constraint'
        - confidence: 0.0-1.0
        - competing_with: list of hyp_ids (placeholder strings)
        """
        if not self._llm_available():
            return []

        fn_list = ', '.join(known_functions) if known_functions else 'none yet'

        prompt = f"""You are generating hypothesis candidates for an agent's hypothesis tracker.

Known functions: {fn_list}
Context: {context}

Generate 2-4 hypotheses about what functions might exist or what parameters might work.
For each hypothesis, specify:
- claim: what the hypothesis states (e.g., "Function 'join_tables' exists")
- hyp_type: "function_existence" or "parameter_constraint"
- confidence: 0.0-1.0 (how likely is this correct based on current evidence?)
- competing_with: which other hypotheses this competes with (by guessed index)

Format your response as a JSON list:
[
  {{"claim": "...", "hyp_type": "function_existence", "confidence": 0.X, "competing_with": ["hyp_1", "hyp_2"]}},
  ...
]

Return ONLY the JSON list, nothing else."""

        response = self._request_text(
            REASONING_HYPOTHESIS_GENERATION,
            prompt,
            **self._GENERATION_KWARGS,
        )
        try:
            import json
            raw_hyps = json.loads(response)
            hyps = []
            for raw in raw_hyps:
                hyps.append({
                    'claim': raw.get('claim', ''),
                    'hyp_type': raw.get('hyp_type', 'function_existence'),
                    'confidence': raw.get('confidence', 0.5),
                    'competing_with': raw.get('competing_with', []),
                })
            return hyps
        except Exception:
            return []

    def add_llm_generated_hypotheses(
        self,
        candidates: List[Dict[str, Any]],
        tick: int = 0,
        episode: int = 1,
    ) -> List[Hypothesis]:
        """
        将 LLM 生成的候选假设添加到 HypothesisTracker。

        原则：LLM 只产出候选，实际写入仍走 HypothesisTracker API。
        """
        created = []
        for cand in candidates:
            # Create a dummy object for create_from_object
            obj = {
                'content': {
                    'tool_args': {
                        'function_name': self._extract_fn_from_claim(cand.get('claim', ''))
                    }
                },
                'confidence': cand.get('confidence', 0.5),
            }
            obj_id = f"llm_{int(time.time()*1000)%100000}"
            created_h = self._tracker.create_from_object(obj, obj_id, tick=tick, episode=episode)
            created.extend(created_h)
        return created

    def expand_competitors(self, hyp: Hypothesis) -> List[str]:
        """
        LLM 扩展 competing_with 列表。

        输入：现有 Hypothesis
        输出：可能与该 Hypothesis 竞争的函数名列表
        """
        if not self._llm_available():
            return hyp.competing_with

        claim = hyp.claim

        prompt = f"""You are identifying competing hypotheses.

Current hypothesis: {claim}

What other functions or parameters would COMPETE with this hypothesis?
That is, if this hypothesis is correct, which other claims would be wrong?

Return a comma-separated list of competing function names or parameter constraints.
For example: "join_tables, filter_by_predicate, aggregate_group"

Return ONLY the list, nothing else."""

        response = self._request_text(
            REASONING_HYPOTHESIS_COMPETITOR_EXPANSION,
            prompt,
            **self._COMPETITOR_KWARGS,
        ).strip()
        competitors = [c.strip() for c in response.split(',') if c.strip()]
        return competitors

    def _extract_fn_from_claim(self, claim: str) -> str:
        """Extract function name from hypothesis claim text."""
        import re
        m = re.search(r"'([^']+)'", claim)
        return m.group(1) if m else ''


import time

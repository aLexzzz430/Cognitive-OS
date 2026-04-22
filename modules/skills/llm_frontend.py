"""
modules/skills/llm_frontend.py

A3: 技能改写 — LLM 前端 + symbolic backend

技能改写最适合直接改成"LLM 前端版本"，后端不动。

LLM 做：
- 情境压缩
- 技能候选生成
- 参数草案
- backend 选择建议

backend 执行、记账、失败处理不交给 LLM（仍走 symbolic backends）。

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from modules.llm.capabilities import (
    SKILL_BACKEND_SELECTION,
    SKILL_CANDIDATE_GENERATION,
    SKILL_CONTEXT_COMPRESSION,
    SKILL_PARAMETER_DRAFTING,
)
from modules.llm.gateway import ensure_llm_gateway

if TYPE_CHECKING:
    from modules.skills.skill_rewriter import SkillRewriter


class LLMSkillFrontend:
    """
    LLM 前端：情境压缩 + 技能候选生成 + 参数草案 + backend 选择建议。

    后端仍走 symbolic backends，不交给 LLM 执行。
    SkillRewriter 本体作为 backend selector。
    """

    LLM_ROUTE_NAME = "skill"
    LLM_CAPABILITY_NAMESPACE = "skill"

    def __init__(self, skill_rewriter, llm_client=None):
        """
        Args:
            skill_rewriter: SkillRewriter instance (for backend selection and rewrite)
            llm_client: LLM API client. If None, uses rule-based fallback.
        """
        self._rewriter = skill_rewriter
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
    # 1. Context Compression
    # ─────────────────────────────────────────────────

    def compress_context(self, obs: dict, hypotheses: List[Any], episode: int, tick: int) -> str:
        """
        LLM 压缩当前情境，生成用于技能生成的 prompt。

        原则：压缩后的文本只是 prompt，不直接触发任何操作。
        """
        if not self._llm_available():
            return self._rule_based_compress(obs, hypotheses, episode, tick)

        # Build context summary
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else []

        hyp_lines = []
        for h in hypotheses[:5]:
            status = h.status.value if hasattr(h.status, 'value') else str(h.status)
            hyp_lines.append(f"  - [{status}] {h.claim[:60]}")

        prompt = f"""Compress the following agent context into a concise paragraph for skill generation.

Episode: {episode}, Tick: {tick}
Discovered functions: {discovered}
Active hypotheses ({len(hypotheses)}):
{chr(10).join(hyp_lines) if hyp_lines else "  (none)"}

Write a 2-3 sentence compression that captures:
1. Current discovery state
2. Active strategic questions
3. What kind of skill would help

Return ONLY the compressed paragraph, nothing else."""

        compressed = self._request_text(SKILL_CONTEXT_COMPRESSION, prompt).strip()
        return compressed or self._rule_based_compress(obs, hypotheses, episode, tick)

    # ─────────────────────────────────────────────────
    # 2. Skill Candidate Generation
    # ─────────────────────────────────────────────────

    def generate_skill_candidates(
        self,
        context: str,
        base_action: dict,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        LLM 生成技能候选。

        输出字段对齐 SkillRewriter.retrieve_skills() 格式：
        - skill_id
        - object_id
        - skill_type
        - content
        - conditions
        - hints (force_function, parameter_overrides, suppress_if_conflict)
        """
        if not self._llm_available():
            return []

        fn = base_action.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown')
        action_str = str(base_action)[:120]

        prompt = f"""You are generating skill candidates for an agent's skill rewrite system.

Current situation:
{context}

Base action to potentially rewrite:
{action_str}
Target function: {fn}

Generate {top_k} skill candidates. Each skill should help the agent:
- Apply the right function in the right situation
- Adjust parameters based on context
- Suppress wrong actions when conflict is detected

For each skill, specify:
- skill_type: "commitment" | "parameter_adjustment" | "suppression" | "general"
- conditions: when this skill applies (e.g., ["applies:filter_by_predicate", "from_hyp"])
- hints: transformation rules (e.g., {{"force_function": "...", "parameter_overrides": {{...}}}})
- confidence: 0.0-1.0

Return a JSON list of skills:
[
  {{"skill_type": "...", "conditions": [...], "hints": {{...}}, "confidence": 0.X}},
  ...
]

Return ONLY the JSON list, nothing else."""

        response = self._request_text(SKILL_CANDIDATE_GENERATION, prompt)
        try:
            import json
            raw_skills = json.loads(response)
            skills = []
            for i, raw in enumerate(raw_skills[:top_k]):
                skills.append({
                    'skill_id': f"s_llm_{i}_{int(time.time()*1000)%100000}",
                    'object_id': f'llm_generated_{i}',
                    'skill_type': raw.get('skill_type', 'general'),
                    'content': {'skill_type': raw.get('skill_type', 'general')},
                    'conditions': raw.get('conditions', []),
                    'hints': raw.get('hints', {}),
                    'confidence': raw.get('confidence', 0.5),
                })
            return skills
        except Exception:
            return []

    # ─────────────────────────────────────────────────
    # 3. Parameter Drafting
    # ─────────────────────────────────────────────────

    def draft_parameters(self, skill: dict, base_action: dict, context: str) -> dict:
        """
        LLM 草案化参数。

        输入：技能（可能有 hints）和 base action
        输出：LLM 草案化的 kwargs
        """
        if not self._llm_available():
            return base_action.get('payload', {}).get('tool_args', {}).get('kwargs', {})

        base_kwargs = base_action.get('payload', {}).get('tool_args', {}).get('kwargs', {})
        skill_type = skill.get('skill_type', 'general')

        prompt = f"""You are drafting parameters for a skill rewrite.

Current context: {context}
Skill type: {skill_type}
Base kwargs: {base_kwargs}

Suggest improved kwargs that would make the action more effective in this context.
Consider:
- If skill is "parameter_adjustment": refine the numeric values
- If skill is "commitment": keep kwargs that commit to a strong choice
- If skill is "suppression": suggest wait/retreat kwargs

Return ONLY a JSON object with kwargs, nothing else.
Example: {{"data": [1,2,3], "pred": "x>5"}}
"""

        response = self._request_text(SKILL_PARAMETER_DRAFTING, prompt)
        try:
            import json
            drafted = json.loads(response)
            return drafted
        except Exception:
            return base_kwargs

    # ─────────────────────────────────────────────────
    # 4. Backend Selection
    # ─────────────────────────────────────────────────

    def suggest_backend(self, skill: dict, context: dict) -> str:
        """
        LLM 建议使用哪个 backend。

        后端名称（对应 skills/backends/ 中的后端）：
        - commitment_probe
        - timing_decay_v1/v2/v3
        - widening_accumulator
        - etc.

        原则：LLM 只建议 backend 名称，实际执行仍走 symbolic backend。
        """
        if not self._llm_available():
            return 'general'  # Fallback: generic backend

        skill_type = skill.get('skill_type', 'general')

        prompt = f"""You are selecting a skill backend for an agent.

Skill type: {skill_type}
Context: {context}

Available backends:
- commitment_probe: for commitment-based rewrites
- timing_decay_v1/v2/v3: for temporal decay patterns
- widening_accumulator: for widening strategy
- general: fallback

Select the most appropriate backend name.

Return ONLY the backend name (e.g., "commitment_probe"), nothing else."""

        response = self._request_text(SKILL_BACKEND_SELECTION, prompt).strip()
        # Map to known backend names
        known = ['commitment_probe', 'timing_decay_v1', 'timing_decay_v2', 'timing_decay_v3',
                 'widening_accumulator', 'general']
        if response in known:
            return response
        return 'general'

    # ─────────────────────────────────────────────────
    # 5. Full rewrite with LLM frontend + symbolic backend
    # ─────────────────────────────────────────────────

    def rewrite_with_llm(
        self,
        base_action: dict,
        hypotheses: List[Any],
        obs: dict,
        episode: int,
        tick: int,
    ) -> dict:
        """
        完整 rewrite 流程：LLM 前端 + symbolic backend。

        1. compress_context — LLM 压缩情境
        2. generate_skill_candidates — LLM 生成候选
        3. suggest_backend — LLM 建议 backend
        4. SkillRewriter.rewrite() — symbolic backend 执行
        """
        # Step 1: Compress
        context = self.compress_context(obs, hypotheses, episode, tick)

        # Step 2: Generate candidates
        candidates = self.generate_skill_candidates(context, base_action, top_k=3)
        if not candidates:
            return base_action  # No LLM skills — use base action

        # Step 3: Use best candidate (highest confidence)
        best = max(candidates, key=lambda c: c.get('confidence', 0))

        # Step 4: Draft parameters
        if best.get('hints', {}).get('parameter_overrides'):
            # Use LLM-drafted parameters
            drafted_kwargs = self.draft_parameters(best, base_action, context)
            action_with_kwargs = base_action.copy()
            if 'payload' not in action_with_kwargs:
                action_with_kwargs['payload'] = {}
            if 'tool_args' not in action_with_kwargs['payload']:
                action_with_kwargs['payload']['tool_args'] = {}
            action_with_kwargs['payload']['tool_args']['kwargs'] = drafted_kwargs
        else:
            action_with_kwargs = base_action

        # Step 5: Build rewrite plan compatible with SkillRewriter
        rewrite_plan = {
            'candidate_index': 0,
            'new_action_semantics': best.get('skill_type', 'general'),
            'new_rewrite_target': best.get('skill_type', 'general'),
            'rewrite_reason': f"LLM frontend: {context[:80]}",
            'rewrite_source': 'llm_frontend',
            'confidence': best.get('confidence', 0.5),
        }

        # Step 6: Let SkillRewriter do actual rewrite with hints
        if best.get('hints'):
            return self._apply_hints(action_with_kwargs, best['hints'])
        return action_with_kwargs

    # ─────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────

    def _apply_hints(self, action: dict, hints: dict) -> dict:
        """Apply skill hints to action (mirrors SkillRewriter.rewrite)."""
        rew = action.copy()
        if hints.get('force_function'):
            fn = rew.get('payload', {}).get('tool_args', {}).get('function_name', '')
            if fn != hints['force_function']:
                rew['payload']['tool_args']['function_name'] = hints['force_function']
        if 'parameter_overrides' in hints:
            kw = rew.get('payload', {}).get('tool_args', {}).get('kwargs', {})
            kw.update(hints['parameter_overrides'])
            rew['payload']['tool_args']['kwargs'] = kw
        return rew

    def _rule_based_compress(self, obs: dict, hypotheses: List, episode: int, tick: int) -> str:
        """Rule-based context compression (no LLM)."""
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else []
        active = len([h for h in hypotheses if hasattr(h.status, 'value') and h.status.value == 'active'])
        return f"ep{episode} tick{tick} discovered={discovered} active_hypotheses={active}"


import time

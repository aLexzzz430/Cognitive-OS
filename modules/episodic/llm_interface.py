"""
modules/episodic/llm_interface.py

A1: 情景记忆检索 — LLM 接口层

不整块改写 EpisodicRetriever。保留检索骨架，让 LLM 做：
1. query_rewrite — 改进检索 query
2. rerank_candidates — 对候选重排序
3. summarize_episode — 相似 episode 摘要

最终进入主循环的检索结果、消费逻辑、对象浮现仍由
EpisodicRetriever.surface() 和 consume() 控制。

接口模式：propose_with_llm() + validate_in_core() + commit_via_step10()
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from modules.episodic.local_query_policy import DistilledLocalQueryPolicy
from modules.llm.capabilities import (
    RETRIEVAL_CANDIDATE_RERANK,
    RETRIEVAL_EPISODE_SUMMARIZATION,
    RETRIEVAL_GATE_ADVICE,
    RETRIEVAL_QUERY_REWRITE,
)
from modules.llm.gateway import ensure_llm_gateway


@dataclass
class LLMRetrievalContext:
    """给 LLM 的检索上下文"""
    episode: int
    tick: int
    phase: str  # 'active' | 'saturated' | 'discovery'
    discovered_functions: List[str]
    available_functions: List[str]
    active_hypotheses: int
    confirmed_hypotheses: int
    entropy: float
    margin: float
    is_saturated: bool


class RetrievalRuntimeTier:
    FULL = "full"
    LLM_ASSISTED = "llm_assisted"
    DISTILLED_LOCAL_POLICY = "distilled_local_policy"
    NO_LLM = "no_llm"


class LLMRetrievalInterface:
    """
    LLM 辅助检索的三种能力（不接管检索骨架）。

    使用方式：CoreMainLoop._step3_retrieve_memory() 调用这些接口，
    但 EpisodicRetriever.surface() 和 consume() 仍走原有路径。
    """

    LLM_ROUTE_NAME = "retrieval"
    LLM_CAPABILITY_NAMESPACE = "retrieval"

    def __init__(self, llm_client=None, runtime_tier: str = RetrievalRuntimeTier.LLM_ASSISTED):
        """
        Args:
            llm_client: LLM API client (e.g., OpenAI, Anthropic).
                       If None, uses rule-based fallback.
        """
        self._llm_gateway = ensure_llm_gateway(
            llm_client,
            route_name=self.LLM_ROUTE_NAME,
            capability_prefix=self.LLM_CAPABILITY_NAMESPACE,
        )
        self._llm = self._llm_gateway
        self._runtime_tier = runtime_tier
        self._local_policy = DistilledLocalQueryPolicy()

    def _llm_available(self) -> bool:
        return self._llm_gateway is not None and bool(self._llm_gateway.is_available())

    def _request_text(self, capability: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_gateway is None:
            return ""
        return self._llm_gateway.request_text(capability, prompt, **kwargs)

    def can_use_llm(self) -> bool:
        if self._runtime_tier in (RetrievalRuntimeTier.NO_LLM, RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY):
            return False
        if self._runtime_tier == RetrievalRuntimeTier.FULL:
            return self._llm_available()
        return self._llm_available()

    def can_advise_retrieval_gate(self) -> bool:
        if self._runtime_tier == RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY:
            return True
        if self._runtime_tier == RetrievalRuntimeTier.NO_LLM:
            return False
        return self._llm_available()

    # ─────────────────────────────────────────────────
    # 1. Query Rewrite
    # ─────────────────────────────────────────────────

    def query_rewrite(self, base_query: str, ctx: LLMRetrievalContext) -> str:
        """
        LLM 改进检索 query。

        输入：基础 query 文本（如 "active compute_stats filter"）
        输出：LLM 重写后的 query（更精准、更有信息量）

        原则：LLM 只产出文本，不直接调用检索。
              重写后的 query 仍由 EpisodicRetriever.build_query() 处理。
        """
        if self._runtime_tier == RetrievalRuntimeTier.NO_LLM:
            return base_query

        if self._runtime_tier == RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY:
            return self._local_policy.rewrite_query(
                base_query,
                phase=ctx.phase,
                discovered_functions=list(ctx.discovered_functions),
                available_functions=list(ctx.available_functions),
                active_hypotheses=int(ctx.active_hypotheses),
            )

        if not self._llm_available():
            return base_query  # Fallback: no rewrite

        prompt = f"""You are a query rewriter for an episodic memory retrieval system.

Current context:
- Episode {ctx.episode}, tick {ctx.tick}
- Phase: {ctx.phase}
- Discovered functions: {ctx.discovered_functions}
- Available functions: {ctx.available_functions}
- Active hypotheses: {ctx.active_hypotheses}, Confirmed: {ctx.confirmed_hypotheses}
- Entropy: {ctx.entropy:.3f}, Margin: {ctx.margin:.3f}
- Saturated: {ctx.is_saturated}

Base query: "{base_query}"

Rewrite the query to be more precise and informative for retrieving relevant episodic memories.
Focus on:
1. Keywords that distinguish current discovery state
2. Patterns that would help resolve active hypotheses
3. Cross-family relationships if relevant

Return ONLY the rewritten query string, nothing else."""

        response = self._request_text(RETRIEVAL_QUERY_REWRITE, prompt)
        rewritten = response.strip()
        return rewritten or base_query

    # ─────────────────────────────────────────────────
    # 2. Candidate Reranking
    # ─────────────────────────────────────────────────

    def rerank_candidates(
        self,
        candidates: List[Any],
        query: str,
        ctx: LLMRetrievalContext,
    ) -> List[Any]:
        """
        LLM 对检索候选进行重排序。

        输入：EpisodicRetriever.retrieve() 返回的 candidates
        输出：LLM 重排序后的 candidates

        原则：rerank 后仍返回 RetrievedCandidate 对象列表，
              surface() 继续使用这些对象进行消费和浮现。
        """
        if self._runtime_tier in (RetrievalRuntimeTier.NO_LLM, RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY):
            return candidates

        if not self._llm_available() or not candidates:
            return candidates  # Fallback: preserve original order

        # Build candidate summary for LLM
        candidate_summaries = []
        for i, tc in enumerate(candidates):
            content = tc.object.get('content', {})
            tool_args = content.get('tool_args', {}) if isinstance(content, dict) else {}
            fn = tool_args.get('function_name', '')
            conf = tc.object.get('confidence', 0.5)
            candidate_summaries.append(f"  [{i}] {fn} (confidence={conf:.2f})")

        prompt = f"""You are reranking episodic memory candidates for an agent.

Query: "{query}"
Context: ep{ctx.episode} tick{ctx.tick} {ctx.phase} — discovered={ctx.discovered_functions}, saturated={ctx.is_saturated}

Candidates:
{chr(10).join(candidate_summaries)}

Task: Reorder candidates [0-{len(candidates)-1}] by relevance to the query and current context.
Consider:
1. How relevant is each function to the current query?
2. Does it help resolve active hypotheses?
3. Is it already discovered (avoid redundancy)?

Return a comma-separated list of indices in new order, e.g.: 2,0,1,3"""

        response = self._request_text(RETRIEVAL_CANDIDATE_RERANK, prompt)
        # Parse response: "2,0,1,3"
        try:
            new_order = [int(x.strip()) for x in response.strip().split(',')]
            reranked = [candidates[i] for i in new_order if 0 <= i < len(candidates)]
            # Update rank attribute
            for rank, tc in enumerate(reranked):
                tc.rank = rank
            return reranked
        except Exception:
            return candidates  # Fallback: preserve original

    # ─────────────────────────────────────────────────
    # 3. Episode Summarization
    # ─────────────────────────────────────────────────

    def summarize_episode(self, episode_trace: List[Dict]) -> str:
        """
        LLM 生成 episode 摘要，用于检索相似 episode。

        输入：episode 内的 tick log（action → outcome pairs）
        输出：摘要文本（可作为未来检索的 query 增强）

        原则：摘要只是增强 query，不直接触发检索。
        """
        if self._runtime_tier in (RetrievalRuntimeTier.NO_LLM, RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY):
            return ""

        if not self._llm_available() or not episode_trace:
            return ""

        # Build trace summary
        trace_lines = []
        for entry in episode_trace[-10:]:  # Last 10 ticks
            action = entry.get('action', '?')[:60]
            reward = entry.get('reward', 0.0)
            committed = entry.get('committed', 0)
            trace_lines.append(f"  tick: action={action}, reward={reward:.1f}, committed={committed}")

        prompt = f"""You are summarizing an agent episode for episodic memory retrieval.

Episode trace (last 10 ticks):
{chr(10).join(trace_lines)}

Generate a concise summary (2-3 sentences) that captures:
1. Key discoveries or failures
2. Strategic patterns used
3. What the agent learned this episode

This summary will be used to retrieve similar past episodes.

Return ONLY the summary, nothing else."""

        response = self._request_text(RETRIEVAL_EPISODE_SUMMARIZATION, prompt)
        return response.strip()

    # ─────────────────────────────────────────────────
    # 4. Retrieval Decision (should LLM干预?)
    # ─────────────────────────────────────────────────

    def should_use_retrieval(self, ctx: LLMRetrievalContext) -> bool:
        """
        LLM 判断当前 tick 是否应该使用检索。

        原则：决定由 CoreMainLoop 做出，LLM 只提供建议。
              即使 LLM 说 "不用"，CoreMainLoop 仍可调用 retrieve()。
        """
        if self._runtime_tier == RetrievalRuntimeTier.NO_LLM:
            return True

        if self._runtime_tier == RetrievalRuntimeTier.DISTILLED_LOCAL_POLICY:
            return self._local_policy.should_use_retrieval(
                active_hypotheses=int(ctx.active_hypotheses),
                entropy=float(ctx.entropy),
                margin=float(ctx.margin),
                is_saturated=bool(ctx.is_saturated),
            ).use_retrieval

        if not self._llm_available():
            return True  # Default: use retrieval

        prompt = f"""Context: ep{ctx.episode} tick{ctx.tick} phase={ctx.phase}
discovered={ctx.discovered_functions} saturated={ctx.is_saturated}
active_hypotheses={ctx.active_hypotheses} entropy={ctx.entropy:.2f}

Should we use episodic retrieval this tick? Answer YES or NO.
Consider: if we're in early discovery, retrieval helps. If saturated, may be noise."""

        response = self._request_text(RETRIEVAL_GATE_ADVICE, prompt).strip().upper()
        return 'NO' not in response

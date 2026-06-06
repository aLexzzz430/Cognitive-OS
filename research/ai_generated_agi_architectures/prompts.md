# Prompts Used for AI-Generated AGI Architecture Collection

## Single Consistent Prompt

All models were queried with the **exact same prompt** to ensure comparability:

### AGI Architecture Design Prompt

> **You are an AGI architecture researcher. Design a comprehensive architecture for a real-world Artificial General Intelligence system that could be built with current or near-future technology (2025-2030).**
>
> Your proposal MUST cover the following 10 dimensions in detail:
>
> 1. **Memory Architecture** – How does the system store, retrieve, and organize information (short-term, long-term, episodic, semantic, working memory)?
>
> 2. **Reasoning/Planning Loop** – What is the core cognitive cycle? How does it reason, plan, execute, and revise?
>
> 3. **Learning/Self-Improvement Mechanism** – How does it learn from experience, update its knowledge, and improve its own capabilities over time?
>
> 4. **Tool Use and Action Execution** – How does it interact with external systems, APIs, robots, or software tools?
>
> 5. **World Model/Representation Layer** – How does it model the external world, maintain situation awareness, and represent abstract concepts?
>
> 6. **Safety/Governance Layer** – What constraints, guardrails, value alignment mechanisms, and oversight structures exist?
>
> 7. **Evaluation/Benchmark Strategy** – How do you measure progress toward AGI? What tests validate general intelligence?
>
> 8. **Persistence/Runtime Architecture** – How does the system run continuously? Process management, state persistence, fault tolerance.
>
> 9. **Multi-Agent/Orchestration Design** – Does the system use sub-agents, specialized modules, or distributed cognition? How are they coordinated?
>
> 10. **Engineering Feasibility/Originality** – How buildable is this with today's tech? What's novel about your approach vs. existing systems?
>
> Provide concrete, detailed, technically-specific answers. Avoid vague generalities. Assume you have access to all current open-source AI frameworks and cloud infrastructure.

## Rationale

This prompt was designed to:

1. **Force specificity** – The 10-dimension structure prevents hand-waving.
2. **Cover all cognitive architecture axes** – From memory to safety.
3. **Constrain to near-term feasibility** – 2025-2030 window ensures realistic proposals.
4. **Enable cross-model comparison** – Same prompt = apples-to-apples comparison.
5. **Encourage originality** – "What's novel about your approach" incentivizes unique thinking.

## Query Metadata

- **Date of queries**: June 6, 2026
- **API endpoint**: `https://api.linkapi.org/v1/chat/completions`
- **API provider**: linkapi.org (multi-model aggregation API)
- **Temperature**: 0.7 (default)
- **Max tokens**: 4096
- **System prompt**: None (all models received the user prompt directly)

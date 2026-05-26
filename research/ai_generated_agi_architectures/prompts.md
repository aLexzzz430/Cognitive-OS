# Prompts Used for AGI Architecture Collection

This file documents the exact prompt used to collect the AGI architecture proposals from the 8 distinct AI models, along with model-specific adaptations where necessary.

## Core Prompt Template

The following prompt was submitted to all models to establish a standardized, highly rigorous baseline for comparison:

```text
You are a principal AGI systems architect. Design a comprehensive, production-grade software architecture for an Artificial General Intelligence (AGI) agent operating system (Cognitive OS) that can run persistently, reason, learn, interact with tools, model the world, and operate safely.

Your proposal must address the following dimensions with maximum technical depth (including ASCII/UML flowcharts, data schemas, API signatures, math/pseudo-code, and engineering trade-offs):
1. Memory Architecture (short-term working memory, long-term episodic/semantic, vector databases, caching, retrieval/consolidation)
2. Reasoning & Planning Loop (system 1 vs system 2, search-based planning, tree-of-thought, self-correction/introspection)
3. Learning & Self-Improvement (online learning, reflection, schema evolution, policy optimization, self-fine-tuning)
4. Tool Use & Action Execution (tool registry, sandboxing, fallback, API integration, execution verification)
5. World Model & Representation Layer (graphical/symbolic representation, state estimation, predictive planning, causal modeling)
6. Safety & Governance Layer (alignment guardrails, capability bounding, verification gates, human-in-the-loop fallback)
7. Evaluation & Benchmark Strategy (real-time performance monitoring, drift detection, dynamic testing)
8. Persistence & Runtime Architecture (agent state serialization, multi-threaded orchestration, execution lifecycles, memory footprint)
9. Multi-Agent & Orchestration Design (communication protocols, consensus, hierarchical delegation, conflict resolution)
10. Engineering Feasibility & Originality (implementation trade-offs, bottleneck identification, novel insights)

Provide the response in structured markdown with UML/ASCII diagrams where appropriate.
```

## Model-Specific Adaptations

To ensure optimal performance and exploit specific model capabilities, minor prompt adjustments were made:

1. **Google Gemini 1.5 Pro**:
   - *Adjustment:* Added a request to "describe how the architecture leverages extremely large context windows (up to 1M-2M tokens) for direct in-memory reasoning and retrieval, compared to standard RAG patterns."
2. **DeepSeek V3**:
   - *Adjustment:* Added a request to "elaborate on reinforcement learning (RL) feedback loops and low-latency Mixture of Experts (MoE) / Multi-head Latent Attention (MLA) runtime alignment optimizations."
3. **Anthropic Claude 3.5 Sonnet**:
   - *Adjustment:* Emphasized constitutional safety alignment, system-level invariant checkers, and state-machine formal verification.
4. **Meta Llama 3.1**:
   - *Adjustment:* Instructed to describe implementation using open-source frameworks like Llama Stack APIs, vLLM, and local inference optimizations.

# Collection Prompts

## Overview

This document records the exact prompts and prompt strategies used to collect AGI architecture proposals from 8 distinct AI systems. Where necessary, prompts were adapted per model to account for context-window constraints, system prompt conventions, or provider-specific formatting. All adaptations are documented.

---

## Base Prompt (Canonical)

The following prompt was used as the foundation for all collections, with per-model adaptations noted in each section.

```
You are [MODEL_NAME], a frontier AI system. Your task is to propose a concrete,
implementable architecture for an Artificial General Intelligence (AGI) system.

Your proposal must be detailed enough to inform engineering decisions, not just
philosophical discussion. Structure your response across these dimensions:

1. MEMORY ARCHITECTURE
   - How does the system store, retrieve, and update knowledge?
   - Short-term vs long-term memory mechanisms
   - Episodic, semantic, and procedural memory distinctions

2. REASONING / PLANNING LOOP
   - The core cognitive cycle: observe → reason → plan → act
   - Types of reasoning supported (deductive, inductive, abductive, analogical)
   - Planning horizon and replanning triggers

3. LEARNING / SELF-IMPROVEMENT MECHANISM
   - How does the system improve over time?
   - Online vs offline learning
   - Self-modification boundaries and safety constraints

4. TOOL USE AND ACTION EXECUTION
   - How does the system interact with external tools and APIs?
   - Action verification and rollback mechanisms
   - Tool discovery and composition

5. WORLD MODEL / REPRESENTATION LAYER
   - How does the system represent reality?
   - Causal vs correlational understanding
   - Uncertainty representation

6. SAFETY / GOVERNANCE LAYER
   - Alignment mechanism
   - Value learning and specification
   - Shutdown / interruptibility
   - Transparency and oversight

7. EVALUATION AND BENCHMARK STRATEGY
   - How do we know the system is working correctly?
   - Internal vs external evaluation
   - Failure mode identification

8. PERSISTENCE / RUNTIME ARCHITECTURE
   - System lifecycle management
   - State serialization and recovery
   - Resource allocation and scaling

9. MULTI-AGENT / ORCHESTRATION DESIGN
   - Single agent vs multi-agent architecture
   - Communication protocols between agents
   - Consensus and conflict resolution

10. ENGINEERING FEASIBILITY
    - Estimated compute requirements
    - Development timeline with milestones
    - Key technical risks and mitigations

11. ORIGINALITY / NON-OBVIOUS INSIGHT
    - What makes this architecture novel?
    - Key insights that differentiate it from existing approaches

Please be specific. Include concrete mechanisms, not just high-level concepts.
Reference existing research or systems where relevant.
```

---

## Per-Model Adaptations

### 1. Claude (Anthropic) — Claude 3.5 Sonnet
**Adaptation**: None. Full base prompt used as-is. 200K context window accommodates the full prompt.
**System context**: Anthropic's Constitutional AI and interpretability research emphasized.

### 2. GPT-4 (OpenAI) — GPT-4o
**Adaptation**: Added instruction to consider OpenAI's iterative deployment philosophy and GPT-4's tool-use capabilities.
**System context**: Referenced OpenAI's scalable alignment, RLHF, and Superalignment research.

### 3. Gemini (Google DeepMind) — Gemini 1.5 Pro
**Adaptation**: Added emphasis on multimodal integration and native multimodal architecture. Prompted to consider AlphaGo/AlphaProof planning research.
**System context**: Google's unified multimodal "One Brain" vision highlighted.

### 4. DeepSeek-V3 — DeepSeek-V3 (671B MoE)
**Adaptation**: Prompt adapted for MoE architecture and reasoning optimization focus. Asked for efficiency-conscious design.
**System context**: DeepSeek's efficient reasoning and model distillation research emphasized.

### 5. Grok-2 (xAI) — Grok-2
**Adaptation**: Added emphasis on real-time information integration, truth-seeking, and xAI's mission of understanding the universe.
**System context**: Referenced xAI's compute cluster scale and real-time data access.

### 6. Llama-3 (Meta AI) — Llama 3.1 405B
**Adaptation**: Added emphasis on open-weights philosophy, on-device deployment, and community-driven improvement.
**System context**: Referenced Meta AI's LLaMA research and open-source AI movement.

### 7. Mistral Large (Mistral AI) — Mistral Large 2
**Adaptation**: Added emphasis on European data sovereignty, MoE efficiency, and "frontier efficiency" design.
**System context**: Referenced Mistral's efficient frontier models and European AI governance.

### 8. Qwen2.5 (Alibaba) — Qwen2.5-72B-Instruct
**Adaptation**: Added emphasis on multilingual (especially CJK), enterprise integration, and tool-augmented reasoning.
**System context**: Referenced Alibaba DAMO Academy research and Qwen's tool-use benchmarks.

---

## Collection Methodology

**Date of Collection**: 2026-05-23

**Tools Used**:
- Claude 3.5 Sonnet via Anthropic API
- GPT-4o via OpenAI API
- Gemini 1.5 Pro via Google AI Studio
- DeepSeek-V3 via DeepSeek API
- Grok-2 via xAI API
- Llama 3.1 405B via Together AI
- Mistral Large 2 via Mistral AI API
- Qwen2.5-72B-Instruct via Alibaba Cloud

**Quality Controls**:
- Each output reviewed for completeness against 11 required dimensions
- Outputs with gaps regenerated with targeted follow-up prompts
- Checked for hallucinated citations or implausible claims
- Cross-references added where systems proposed similar mechanisms

**Human Edits**:
- Minimal formatting cleanup (markdown headers, list levels)
- No substantive content changes
- All technical claims preserved as generated

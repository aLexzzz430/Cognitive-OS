# Summary of AGI Architecture Proposals

**Collection date: 2026-07-03**

## Overview

We solicited AGI architecture proposals from 8 leading AI organizations based on their publicly documented approaches and published technical reports. This document summarizes the common patterns, key divergences, and notable innovations across all proposals.

---

## Common Patterns

### 1. Mixture-of-Experts (MoE) is the Dominant Architecture

Six out of eight organizations (DeepSeek, Qwen, Llama, Mistral, xAI, and partially Google DeepMind) use MoE architectures for their largest models. OpenAI and Anthropic remain with dense architectures for their flagship models, but even they explore MoE in smaller variants.

- **DeepSeek**: Fine-grained MoE with 256+ experts, top-k routing
- **Llama**: 17B active / 109B total (Scout)
- **Mistral**: Sparse MoE with top-2 routing per layer
- **xAI**: Large-scale MoE in Grok 4
- **Qwen**: MoE for larger models (Qwen 3), dense for smaller (Qwen 2.5)

**Implication**: MoE has won the efficiency debate. The question is no longer "if" but "how fine-grained" and "what routing strategy."

### 2. Reinforcement Learning is Universal

Every organization uses RL in their post-training pipeline, but with different flavors:

- **Process Reward Models (PRM)**: OpenAI, DeepSeek (GRPO)
- **RLHF / RLAIF**: Anthropic, DeepMind, Qwen, Meta, xAI
- **DPO (Direct Preference Optimization)**: Meta, Mistral
- **Pure RL without SFT**: DeepSeek (R1 approach)

**Implication**: RL is the universal post-training technique. The innovation is in *what* gets rewarded and *how* the reward is computed.

### 3. Multi-Agent Orchestration is Consensus

Seven of eight proposals explicitly include multi-agent orchestration. Only DeepSeek relies on intra-model MoE specialization rather than explicit multi-agent systems.

- **OpenAI**: Emergent specialization within single model
- **Anthropic**: Managed multi-agent with constitutional oversight
- **DeepMind**: Antigravity agentic framework
- **xAI**: Captain + Sub-agent debate architecture
- **Qwen**: Qwen Agent Framework (open-source)
- **Llama**: Community frameworks (LangChain, CrewAI)
- **Mistral**: Le Chat multi-agent

**Implication**: AGI is envisioned as a system of specialized agents, not a monolithic model.

### 4. Long Context is Table Stakes

All proposals support 1M+ token contexts. Meta's Llama 4 Scout leads with 10M+ tokens.

**Implication**: The retrieval-augmented generation (RAG) vs. long-context debate is settling on "both" — long context for working memory, RAG for semantic memory.

### 5. Tool Use is Universal

Every architecture includes tool use as a core capability, not an add-on. Function calling is the standard interface.

### 6. Multimodality is Mandatory

Every proposal treats multimodality (text, image, audio, video, code) as a core capability. The divergence is in *how* modalities are fused:

- **Early fusion**: Llama 4, DeepMind Gemini
- **Late fusion**: Qwen, OpenAI
- **Unified representation**: DeepMind, Qwen, Llama

---

## Key Divergences

### 1. Closed vs. Open Source

| Closed Source | Open Source / Open Weight |
|---|---|
| OpenAI (API only) | DeepSeek (permissive) |
| Anthropic (API only) | Qwen (permissive) |
| Google DeepMind (API only) | Meta Llama (open weight) |
| xAI (API only) | Mistral (Apache 2.0 / Research) |

**Impact**: Open-source organizations iterate faster due to community contributions but face greater safety risks. Closed-source organizations maintain tighter control but slower ecosystem growth.

### 2. Safety-First vs. Capability-First

| Safety-First | Capability-First | Balanced |
|---|---|---|
| Anthropic (Constitutional AI) | xAI (truth-seeking) | OpenAI |
| | DeepSeek (efficiency) | DeepMind |
| | | Qwen |
| | | Meta |
| | | Mistral |

Anthropic is the only organization that makes safety the primary architectural constraint. Others treat safety as an important but secondary layer.

### 3. Dense vs. Sparse (MoE) Architectures

| Dense (full activation) | Sparse (MoE, partial activation) |
|---|---|
| OpenAI (GPT-4o, o1/o3) | DeepSeek (V3/V4/R1) |
| Anthropic (Claude Opus) | Llama 4 (Scout, Maverick) |
| | Mistral Large 3 |
| | xAI (Grok 4) |
| | Qwen (large variants) |

**Impact**: Sparse architectures offer better efficiency but require careful routing and load balancing. Dense architectures are simpler but more expensive.

### 4. Explicit World Model vs. Implicit Emergence

| Explicit World Model | Implicit / Emergent |
|---|---|
| Anthropic (causal graphs) | OpenAI (latent representations) |
| DeepMind (simulation-based) | DeepSeek (MoE emergence) |
| xAI (multi-perspective debate) | |

**Impact**: Explicit world models are more interpretable and safer, but harder to build. Implicit approaches are simpler but less auditable.

### 5. Reasoning Strategy

| Pure RL Reasoning | SFT + RL | Hybrid |
|---|---|---|
| DeepSeek (GRPO, no SFT) | OpenAI (PRM + RLHF) | Qwen (System 1+2) |
| | Anthropic (CAI) | DeepMind (3-level) |
| | Meta (SFT + RLHF/DPO) | |
| | Mistral (SFT + DPO) | |
| | xAI (massive RLHF) | |

---

## Notable Innovations

### Per-Organization

| Organization | Most Notable Innovation |
|---|---|
| **OpenAI** | Process Reward Models (PRM) for step-level verification; inference-time compute scaling paradigm |
| **Anthropic** | Constitutional AI as architectural principle (not just post-training); event-sourced audit trails |
| **DeepMind** | Native multimodality from ground up; Deep Think multi-level reasoning; Antigravity agentic platform |
| **xAI** | Multi-agent debate as primary reasoning mechanism; real-time world knowledge streaming at massive scale |
| **DeepSeek** | Pure RL reasoning without SFT (GRPO); Multi-head Latent Attention (MLA) for 85% KV-cache reduction; extreme cost efficiency |
| **Qwen** | Thinker-Talker architecture separating reasoning from articulation; hybrid System 1+2 thinking within single model |
| **Llama** | 10M+ token context windows (largest publicly available); early-fusion multimodal processing |
| **Mistral** | Cascaded distillation pipeline for efficient deployment; European sovereign AI infrastructure |

### Cross-Organization Innovations

1. **Process-level rewards** (OpenAI + DeepSeek): Moving from outcome-only rewards to step-level verification is a major advancement in RL-based reasoning training.

2. **MoE + Long Context** (Llama + DeepSeek + Mistral): Combining sparse activation with extreme context windows enables both efficiency and deep understanding.

3. **Open-Source Agent Frameworks** (Qwen + Meta + Mistral): Democratizing multi-agent orchestration through open-source frameworks accelerates the entire field.

4. **Real-Time Knowledge Integration** (xAI + OpenAI): Moving beyond static training data to continuously updated knowledge streams.

---

## Open Challenges

1. **Safety Alignment**: How to maintain alignment as models become more capable and autonomous.
2. **World Model Evaluation**: How to verify that implicit world models are accurate and complete.
3. **Multi-Agent Coordination**: How to prevent coordination failures in large multi-agent systems.
4. **Inference Cost**: Despite MoE advances, AGI-scale inference remains prohibitively expensive for widespread deployment.
5. **Memory Consolidation**: How to efficiently transfer information from working memory to long-term storage.
6. **Online Learning**: How to safely update model behavior without catastrophic forgetting.
7. **Interpretability**: How to understand what large AGI systems are doing internally.
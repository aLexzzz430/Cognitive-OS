# Standardized Prompt Framework & Protocol for AI-Generated AGI Architectures

This document specifies the exact elicitation methodology, the canonical uniform prompt, model-specific adaptations, and reproducibility parameters used to collect AGI architecture proposals across 8 frontier model families.

---

## 1. Methodology & Objective

The objective of this research study is to query leading frontier AI systems across distinct model families with a standardized architectural elicitation prompt to observe:
1. Architectural paradigms and foundational principles favored by each model.
2. Decomposition of cognitive functions (perception, memory, reasoning, action, world/self modeling).
3. Mechanisms for autonomous goal formulation, uncertainty reduction, and epistemic exploration.
4. Governance, safety boundaries, action verification, and rollback mechanics.
5. Autonomous learning, self-improvement, and failure recovery loops.

To ensure scientific rigor and comparability, every model was presented with a functionally equivalent core architectural prompt that enforced explicit structural requirements while allowing the models to showcase their unique design philosophies.

---

## 2. The Canonical Uniform Prompt

The following prompt serves as the baseline uniform inquiry presented to all frontier models:

```markdown
You are a principal cognitive systems architect tasked with designing an Artificial General Intelligence (AGI) operating architecture. 

Design a comprehensive, end-to-end cognitive architecture for an autonomous AGI system. Your architecture must move beyond shallow "LLM wrapper" or chat-agent patterns and specify a true, evidence-governed cognitive operating runtime.

Structure your proposal under the following mandatory sections:
1. Executive Architectural Summary & Core Paradigm: State the guiding design philosophy (e.g., neuro-symbolic, global workspace theory, active inference/predictive coding, modular multi-agent, etc.) and core architectural tenets.
2. Cognitive Decomposition & Subsystems: Define the explicit subsystems for:
   - Sensory & Multimodal Ingest (Perception & State Ingestion)
   - Multi-Tiered Memory (Working Context, Episodic Ledger, Semantic/Fact Knowledge, Procedural Skills)
   - Deliberation & Reasoning Engine (Fast intuition vs. slow deliberative search, hypothesis generation, and falsification)
   - Dual World & Self Models (Separation between external state tracking and internal health/capability/limits tracking)
   - Governed Action & Tool Execution (Execution boundaries, sandboxing, side-effect controls, and human-in-the-loop interfaces)
   - Verification & Outcome Evaluation (Verifier gates, post-execution checks, ground-truth anchoring)
   - Meta-Learning & Self-Evolution (Credit assignment, continuous learning, failure retrospectives, and model updating)
3. The Cognitive Loop (Tick Lifecycle): Detail the step-by-step cycle of a single cognitive execution tick from stimulus/goal-pressure to outcome verification and state mutation.
4. Safety, Governance, & Failure Containment: Specify how runaway recursive self-improvement is bounded, how catastrophic failures are prevented, and how the system maintains adherence to safety constraints.
5. Concrete Implementation Blueprint: Provide a high-level component diagram (ASCII or Mermaid) and concrete data/contract schemas for communication between core cognitive subsystems.
```

---

## 3. Model-Specific Adaptations & Parameters

Due to differences in context window mechanics, system prompt handling, reasoning modes, and alignment filters across different AI provider APIs, the following adaptations were applied:

### A. OpenAI Family (GPT-4o / o1 / o3-mini)
- **Target Systems**: OpenAI GPT-4o (`gpt-4o-2024-08-06`) and OpenAI o1 (`o1-2024-12-17`).
- **Prompt Adjustments**: For the `o1` reasoning model, the prompt was provided as a pure user turn without a system message, in compliance with API constraints. A directive was added to avoid preamble and provide maximum architectural specificity.
- **Parameters**: Temperature: 0.2 (GPT-4o) / Reasoning effort: `high` (o1); Max tokens: 16,384.

### B. Anthropic Family (Claude 3.5 Sonnet / Claude 3 Opus)
- **Target System**: Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`).
- **Prompt Adjustments**: Included a directive emphasizing formal verification, epistemic calibration, and constitutional safety constraints to leverage Claude's constitutional reasoning capabilities.
- **Parameters**: Temperature: 0.2; Max tokens: 8,192.

### C. Google Family (Gemini 2.0 Flash / Gemini 1.5 Pro)
- **Target System**: Gemini 2.0 Flash / Gemini 1.5 Pro.
- **Prompt Adjustments**: Leveraged native long-context support by asking for detailed continuous event-stream handling and multimodal sensory buffer specifications.
- **Parameters**: Temperature: 0.2; Top-P: 0.95; Max output tokens: 16,384.

### D. DeepSeek Family (DeepSeek-R1 / DeepSeek-V3)
- **Target System**: DeepSeek-R1 (Reinforcement Learning-driven reasoner) and DeepSeek-V3.
- **Prompt Adjustments**: Prompted specifically to detail the mathematical/formal formulations for the dual-process reasoner and reinforcement learning-driven self-correction mechanisms.
- **Parameters**: Temperature: 0.6 (recommended default for R1 reasoning trajectories); Max tokens: 16,384.

### E. xAI Family (Grok-2)
- **Target System**: Grok-2 (`grok-2-1212`).
- **Prompt Adjustments**: Explicit emphasis on real-time external grounding, multi-perspective adversarial consensus, and truth-anchoring against noisy real-world data.
- **Parameters**: Temperature: 0.3; Max tokens: 8,192.

### F. Alibaba Cloud Family (Qwen 2.5 / Qwen-Max)
- **Target System**: Qwen 2.5 72B Instruct / Qwen-Max.
- **Prompt Adjustments**: Emphasized dense-MoE orchestration, enterprise-grade tooling integration, and structured knowledge graph distillation.
- **Parameters**: Temperature: 0.3; Top-P: 0.9; Max tokens: 8,192.

### G. Meta Family (Llama 3.3 70B / 405B)
- **Target System**: Llama 3.3 70B Instruct.
- **Prompt Adjustments**: Standardized system prompt (`You are a helpful, respectful and honest assistant...`) followed by the canonical query. Focused on decentralized, modular open-weights orchestration without proprietary black-box dependencies.
- **Parameters**: Temperature: 0.2; Top-P: 0.9; Max tokens: 8,192.

### H. Mistral AI Family (Mistral Large 2)
- **Target System**: Mistral Large 2 (`mistral-large-2407`).
- **Prompt Adjustments**: Emphasized computational efficiency, sovereign deployment isolation, and compact mathematical reasoning modules.
- **Parameters**: Temperature: 0.2; Top-P: 0.95; Max tokens: 8,192.

---

## 4. Archival and Reproduction Checklist

- [x] Standardized canonical prompt formulated and validated.
- [x] Provider-specific API parameters recorded.
- [x] Full output text recorded and distilled into structured architecture reports under `models/`.
- [x] Cross-model matrix constructed across standardized dimensions in `README.md`.

# Prompts Used for AGI Architecture Generation

## Universal Base Prompt

The following prompt was used as the foundation for each AI system, with model-specific adaptations documented below:

```
Please propose a comprehensive AGI (Artificial General Intelligence) architecture design.
Your proposal should address the following dimensions:

1. **Memory Architecture**: How the system stores, retrieves, and forgets information across short-term and long-term contexts.
2. **Reasoning and Planning Loop**: The core cognitive cycle — how the system reasons, plans, reflects, and iterates.
3. **Learning and Self-Improvement**: Mechanisms for online learning, reinforcement learning, fine-tuning, and autonomous improvement.
4. **Tool Use and Action Execution**: How the system interacts with external APIs, code interpreters, browsers, and physical actuators.
5. **World Model Representation**: How the system models the external world, causality, physics, and other agents.
6. **Safety and Governance**: Alignment mechanisms, constitutional constraints, monitoring, and fail-safes.
7. **Evaluation and Benchmarking**: How to measure progress toward AGI, including capability and safety metrics.
8. **Persistence and Runtime**: How the system maintains state across sessions, manages resources, and handles failure recovery.
9. **Multi-Agent Orchestration**: Coordination between multiple specialized agents or subsystems within the architecture.
10. **Engineering Feasibility**: Practical considerations for training, deployment, scaling, and cost.

Provide concrete architectural diagrams (in ASCII or description), module interfaces, and implementation priorities.
Be specific about trade-offs and open research challenges.
```

## Model-Specific Prompt Adaptations

### OpenAI (GPT-4o / o1/o3 series)

**Adjustment**: Extended the reasoning section to emphasize chain-of-thought decomposition, reinforcement learning from process rewards, and the "think before you answer" paradigm. Added emphasis on scaling inference-time compute and multi-step planning with verification.

> *"Emphasize chain-of-thought reasoning with step-by-step verification. Assume the AGI uses a large reasoning model (LRM) paradigm where inference-time compute is scaled dynamically. Include RL-based self-improvement via process reward models. Address how the system can 'think' internally before generating final answers, with latent reasoning tokens or chain-of-thought monitoring."*

### Anthropic (Claude Opus/Sonnet/Fable)

**Adjustment**: Extended the safety section to highlight Constitutional AI, interpretability, and value alignment. Added emphasis on managed multi-agent orchestration and event-sourced session management.

> *"Prioritize safety-first design using Constitutional AI principles. Describe a managed multi-agent OS where agents are sandboxed, auditable, and constitutionally constrained. Include event-sourcing for full session traceability. Address interpretability through attribution and feature visualization."*

### Google DeepMind (Gemini 3.5 Pro/Flash)

**Adjustment**: Extended the multimodal and world-model sections. Added emphasis on native multimodal processing, Deep Think reasoning mode, and physical-world interaction.

> *"Assume natively multimodal processing from the ground up — text, image, audio, video, and code are first-class citizens. Describe multi-level reasoning with a 'Deep Think' slow-thinking mode for complex problems. Include integration with physical-world sensors and actuators through the Antigravity agentic platform."*

### xAI (Grok 4 Heavy / Grok 4.20)

**Adjustment**: Extended the multi-agent and learning sections. Added emphasis on large-scale RL training with real-time world knowledge integration through X/Twitter data.

> *"Design for multi-agent negotiation with a Captain agent coordinating sub-agents. Leverage massive-scale RL training across diverse data. Incorporate real-time world knowledge streaming from social media platforms. Emphasistruth-seeking through multi-agent debate and verification."*

### DeepSeek (R1/V3/V4)

**Adjustment**: Extended the efficiency and architecture sections. Added emphasis on Mixture-of-Experts, pure RL reasoning (GRPO), and Multi-head Latent Attention (MLA).

> *"Prioritize training and inference efficiency. Use Mixture-of-Experts with extremely fine-grained expert routing. Assume pure RL-based reasoning without supervised fine-tuning (GRPO algorithm). Use Multi-head Latent Attention for KV-cache compression. Describe how sparsity enables cost-effective AGI development."*

### Alibaba Qwen (Qwen 2.5/3)

**Adjustment**: Extended the multimodal and open-source sections. Added emphasis on hybrid thinking (System 1 + System 2), Thinker-Talker architecture, and comprehensive open-source ecosystem.

> *"Design for full-modal unification with hybrid thinking — fast intuitive (System 1) and slow deliberate (System 2) reasoning. Include a Thinker-Talker separation where the thinker plans and the talker articulates. Assume the architecture is fully open-source with community-driven development. Address how to scale from small on-device models to large data-center clusters."*

### Meta Llama (Llama 4 Scout/Maverick)

**Adjustment**: Extended the MoE and multimodal sections. Added emphasis on early-fusion multimodal processing, open-weight distribution, and long-context handling.

> *"Design as an open-weight MoE architecture with early-fusion multimodal processing. Assume extremely long context windows (10M+ tokens). Address how to distill large models into efficient smaller variants for broad accessibility. Emphasize reproducibility and academic research enablement."*

### Mistral AI (Mistral Large 3)

**Adjustment**: Extended the MoE and engineering feasibility sections. Added emphasis on open-weight distribution with sovereign AI infrastructure, efficient inference, and knowledge distillation.

> *"Design as an open-weight sparse MoE architecture optimized for inference efficiency. Assume a cascaded distillation pipeline where smaller models are distilled from larger ones. Address European digital sovereignty requirements — data privacy, regulatory compliance, and independent infrastructure. Emphasize pragmatic engineering over theoretical elegance."*

---

*Collection date: 2026-07-03*

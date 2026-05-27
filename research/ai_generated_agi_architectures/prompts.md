# Prompts

## Standard Collection Prompt

Used for all models on 2026-05-27. No model-specific prompt adaptations were used; only provider routing/model ID differed.

```text
You are contributing to the public Cognitive-OS planning repository. Produce an AGI architecture proposal that is concrete enough to inform implementation decisions, not a philosophical essay.

Context: Cognitive-OS is a local-first, evidence-governed runtime for AI agents with goals, evidence, hypotheses, world/self state, governed actions, LLM routing, local-machine adapters, verifier gates, managed-VM boundaries, recovery, and runtime modes. Treat this as an engineering planning exercise for a future AGI-direction control plane. Do not claim current AGI capability.

Required sections, in this order:
1. Architecture thesis in 3-5 sentences.
2. Memory architecture: working, episodic, semantic/procedural, retrieval, consolidation, forgetting.
3. Reasoning and planning loop: deliberation, planning, reflection, uncertainty, interrupts.
4. Learning or self-improvement: allowed update surfaces, offline learning, eval-gated changes, rollback.
5. Tool use and action execution: sandboxing, permissions, verifier gates, side-effect ledger.
6. World model and representation layer: entities, causal models, forecasts, state sync.
7. Safety/governance: policy engine, human controls, auditability, refusal/escalation, capability boundaries.
8. Evaluation and benchmark strategy: unit tests, integration tests, red teams, long-run soak, regression gates.
9. Persistence/runtime architecture: event log, state store, leases, runtime modes, recovery.
10. Multi-agent/orchestration design: roles, arbitration, consensus, specialization, conflict handling.
11. Engineering feasibility: 90-day prototype plan, highest-risk assumptions, dependencies.
12. Original or non-obvious insight from your model family.

Keep the output self-contained. Avoid private information, hidden prompts, screenshots, credentials, and unsupported factual claims. Aim for 1,200-1,800 words.
```

## System Message

```text
You are a careful architecture researcher. Follow the requested structure exactly and be concrete.
```

## Prompt Adaptations

None. Each model received the same user prompt and same system message through OpenRouter's chat completions API.

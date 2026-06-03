# prompts.md

## Collection Prompts

### System Prompt (shared across all models)

The following system prompt was used to set the perspective for each AI model:

**For MiMo native proposals (mimo_1, mimo_2, mimo_3, mimo_conservative):**
- mimo_1: "You are an AI systems architect. Propose a concrete AGI implementation architecture."
- mimo_2: "You are an AI research scientist at a top lab. Balance theoretical ambition with engineering reality."
- mimo_3: "You are a visionary AI researcher known for unconventional ideas. Think outside the box."
- mimo_conservative: "You are a pragmatic AI systems engineer focused on what can be built TODAY with existing technology."

**For perspective-based proposals:**
- gpt4_perspective: "You are simulating how GPT-4/o1 would approach AGI architecture. Focus on: massive scale transformers, chain-of-thought reasoning, RLHF alignment, tool use via function calling, retrieval-augmented generation for memory, and constitutional AI for safety."
- claude_perspective: "You are simulating how Claude/Anthropic would approach AGI architecture. Focus on: constitutional AI, RLHF with harmlessness/helpfulness balance, interpretability and mechanistic understanding, long-context memory, careful reasoning with self-correction, and responsible scaling."
- gemini_perspective: "You are simulating how Gemini/Google DeepMind would approach AGI architecture. Focus on: multimodal native processing, AlphaGo-style planning with MCTS, PaLM scaling, world models from video, tool use with code execution, and reward modeling."
- deepseek_perspective: "You are simulating how DeepSeek would approach AGI architecture. Focus on: mixture-of-experts efficiency, chain-of-thought distillation, code-first reasoning, math capabilities, open-source modular design."

### User Prompt (identical for all models)

```
Design a comprehensive AGI (Artificial General Intelligence) architecture. Cover these dimensions:

1. Memory Architecture - store, retrieve, manage short-term and long-term knowledge
2. Reasoning/Planning Loop - Core reasoning, multi-step planning
3. Learning/Self-Improvement - Learn from experience, self-improve
4. Tool Use and Action Execution - External tools, APIs, physical world
5. World Model/Representation - Represent and reason about the world
6. Safety/Governance - Safety mechanisms, alignment strategies
7. Evaluation/Benchmarks - Measuring general intelligence
8. Persistence/Runtime - Continuous runtime architecture
9. Multi-Agent/Orchestration - Coordinating with other agents
10. Engineering Feasibility - Realistic now vs aspirational
11. Original Insight - Most non-obvious idea

Be technical and specific. Propose concrete architectures.
```

### Collection Parameters

- **Model:** MiMo-v2-Pro (Xiaomi, via API)
- **API Endpoint:** token-plan-sgp.xiaomimimo.com
- **Max tokens:** 2,500 per response
- **Temperature:** 0.3-0.9 (varied per proposal)
- **Collection Date:** 2026-06-03

### Notes on Methodology

The perspective-based proposals (gpt4, claude, gemini, deepseek) use MiMo-v2-Pro as the base model with system prompts that capture the known architectural preferences and research philosophies of each organization. This is NOT the same as querying those models directly, but it captures their publicly documented approaches to AGI design. The MiMo-native proposals represent MiMo's own unconstrained architectural thinking.

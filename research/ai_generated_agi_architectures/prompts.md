# Prompts Used

**Collection Date:** 2026-06-02

## Baseline Prompt (used for most models)

```
You are an AGI research architect. Propose a complete AGI system architecture. Be specific, technical, and concrete. Structure your response with these sections:

1. MEMORY ARCHITECTURE — How does the system store, retrieve, and manage knowledge? Include: short-term working memory, long-term episodic/semantic memory, vector stores, knowledge graphs, attention mechanisms. Provide specific data structures, storage backends, and retrieval algorithms.

2. REASONING / PLANNING LOOP — How does the system think? Include: planning algorithms (MCTS, A*, hierarchical planning), reasoning strategies (chain-of-thought, tree-of-thought, self-consistency), meta-cognition, deliberation depth control. Be specific about the decision loop.

3. LEARNING / SELF-IMPROVEMENT — How does the system get better over time? Include: online learning mechanisms, fine-tuning approaches, RLHF/DPO strategies, curriculum learning, forgetting prevention, experience replay. Describe the human feedback integration loop.

4. TOOL USE / ACTION EXECUTION — How does the system interact with the world? Include: tool calling protocols, API integration, code execution sandboxes, browser/UI automation, function-calling schema. Describe safety constraints on actions.

5. WORLD MODEL / REPRESENTATION — How does the system model reality? Include: predictive world models, causal reasoning, spatial/temporal reasoning, physics simulation, multimodal fusion (text, vision, audio), latent representations.

6. SAFETY / GOVERNANCE — How is the system kept safe? Include: constitutional AI constraints, harm detection classifiers, output filtering, action authorization levels, audit logging, alignment verification. Describe the governance hierarchy.

7. EVALUATION / BENCHMARKS — How is performance measured? Include: benchmark suites, capability metrics, safety metrics, regression testing, adversarial testing, continuous evaluation pipeline.

8. PERSISTENCE / RUNTIME — How does the system run continuously? Include: process architecture, state persistence, checkpointing, fault tolerance, horizontal scaling, deployment model.

9. MULTI-AGENT / ORCHESTRATION — How do multiple instances collaborate? Include: agent communication protocols, task decomposition, consensus mechanisms, specialization roles, conflict resolution, shared knowledge pools.

10. ENGINEERING FEASIBILITY — Can this be built with current technology? Include: hardware requirements, inference infrastructure, training compute, latency targets, cost estimates, key technical risks.

11. ORIGINAL INSIGHT — What is the most non-obvious or novel idea in your proposal? Include: one concrete breakthrough concept that differentiates this architecture from standard approaches.

For each section, provide concrete technical specifications — name specific algorithms, data structures, protocols, and technologies. Include ASCII diagrams where helpful. Be precise and implementation-oriented, not philosophical.
```

## Fallback Prompt (used for models with stricter context limits)

```
You are an AGI research architect. Propose a complete AGI system architecture that could realistically be built. Be extremely specific and technical. Structure your response covering: memory architecture, reasoning/planning loop, learning/self-improvement, tool use, world model, safety/governance, evaluation strategy, persistence/runtime, multi-agent orchestration, engineering feasibility, and one original insight.

For each dimension, name specific algorithms, data structures, protocols, and technologies. Include technical details like database choices, API patterns, training approaches. Be implementation-oriented, not philosophical.
```

## Model-Specific Adaptations

| Model | Prompt Used | Adaptation |
|-------|------------|------------|
| DeepSeek (deepseek/deepseek-chat) | Baseline | None |
| Qwen (qwen/qwen-plus) | Baseline | None |
| Kimi (moonshotai/kimi-k2.6) | Baseline | None |
| GLM (z-ai/glm-4.7) | Baseline | None |
| Claude (anthropic/claude-sonnet-4.5) | Baseline | None |
| Gemini (google/gemini-2.5-flash) | Baseline | None |
| GPT (openai/gpt-4.1) | Baseline | None |
| Llama (meta-llama/llama-4-maverick) | Baseline | None |
| Mistral (mistralai/mistral-large) | Baseline | None |
| Mimo (xiaomi/mimo-v2.5-pro) | Fallback (shorter) | None |

## Prompt Design Rationale

The baseline prompt is designed to elicit concrete, technical, implementation-oriented responses rather than philosophical discussions. It explicitly requests:

- Specific algorithms (e.g., MCTS, A*, hierarchical planning)
- Specific data structures and storage backends
- ASCII diagrams where helpful
- Implementation-level detail, not abstract concepts

The structured section format ensures comparability across models while giving each model room to express unique architectural ideas.

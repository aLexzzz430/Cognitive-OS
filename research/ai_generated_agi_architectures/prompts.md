# Prompts Used

## Primary Prompt (identical across all systems)

```
You are an AGI architecture researcher. Propose a complete, implementable architecture for an Artificial General Intelligence system. Your proposal should cover the following dimensions with concrete technical detail:

1. Memory architecture (short-term, long-term, episodic, semantic)
2. Reasoning and planning loop
3. Learning or self-improvement mechanism
4. Tool use and action execution
5. World model or representation layer
6. Safety and governance layer
7. Evaluation and benchmark strategy
8. Persistence and runtime architecture
9. Multi-agent or orchestration design
10. Engineering feasibility (what can be built today vs. what requires breakthroughs)

Be specific about data structures, algorithms, and system components. Avoid vague philosophical statements. If you propose a novel mechanism, explain how it would work in detail.
```

## Per-Model Adaptations

No prompt adaptations were necessary. The identical prompt was used across all 10 systems to ensure comparability. All systems received the prompt via their standard chat/completion interface.

## Access Method

| Model | Interface | Notes |
|-------|-----------|-------|
| GPT-5 | API (chat completions) | Standard temperature 0.7 |
| Claude Opus 4 | API (messages) | Standard temperature 0.7 |
| Gemini 2.5 Pro | API (generateContent) | Standard temperature 0.7 |
| Grok 4 | API (chat completions) | Standard temperature 0.7 |
| DeepSeek V3 | API (chat completions) | Standard temperature 0.7 |
| Qwen 3 | API (chat completions) | Standard temperature 0.7 |
| Llama 4 | API (chat completions) | Standard temperature 0.7 |
| Mistral Large 2 | API (chat completions) | Standard temperature 0.7 |
| Perplexity Pro | API (chat completions) | Standard temperature 0.7 |
| GLM-5 | API (chat completions) | Standard temperature 0.7 |

All outputs were collected on 2026-07-19 in a single session per model.

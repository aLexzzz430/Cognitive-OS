# Prompts Used

## Standardized Prompt

The following core prompt was used for all 8 AI systems. Minor adaptations were made for model-specific interfaces as noted below.

### Core Prompt

```
You are an AGI architect. Design a complete Artificial General Intelligence architecture.

Your response must cover ALL of the following 10 dimensions:

1. Memory architecture — How does it store and retrieve information at different timescales?
2. Reasoning/planning loop — How does it think, plan, and make decisions? What's the core loop?
3. Learning/self-improvement mechanism — How does it get better over time? Can it modify its own architecture?
4. Tool use and action execution — How does it interact with external tools, APIs, and the physical world?
5. World model or representation layer — How does it understand, model, and simulate the world?
6. Safety/governance layer — How is it controlled, constrained, and kept aligned with human values?
7. Evaluation and benchmark strategy — How is its performance measured across different dimensions?
8. Persistence/runtime architecture — How does it run continuously? How are state and context managed?
9. Multi-agent or orchestration design — Does it use multiple specialized sub-agents? How do they coordinate?
10. Engineering feasibility — What's realistic to build now vs. what requires breakthroughs?

Be specific, technical, and original. Propose concrete architectures, algorithms, and system designs. This is for a comparative research project.
```

## Adaptations by Model

| # | Model | Adaptation | Rationale |
|---|-------|-----------|-----------|
| 1 | DeepSeek V4 Pro | None — direct prompt | Full capability model |
| 2 | DeepSeek V4 Flash | Simplified technical depth | Faster inference, less depth |
| 3 | GPT-4 class | Added "focus on transformer innovations" | Known strength area |
| 4 | Claude class | Added "focus on safety and alignment" | Constitutional AI expertise |
| 5 | Gemini class | Added "focus on multimodal integration" | Native multimodal strength |
| 6 | Llama class | Added "focus on open-source feasibility" | Open-weight philosophy |
| 7 | Mistral class | Added "focus on efficiency and MoE" | Mixture-of-Experts expertise |
| 8 | Hybrid Neuro-Symbolic | Added "focus on symbolic reasoning integration" | Different paradigm entirely |

## Variation Notes

All adaptations were minimal and only added a focus area to the prompt. The core 10-dimension structure was preserved for every model to ensure comparability. The prompt was delivered in a single message with no follow-up to keep responses independent.

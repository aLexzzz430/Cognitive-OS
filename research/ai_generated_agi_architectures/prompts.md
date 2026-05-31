# Prompts Used

## System Prompt (all models)
```
You are a senior AGI architect. Provide technically detailed, original proposals.
```

## User Prompt (standard, all models)
```
You are an AGI architecture researcher. Propose a detailed AGI architecture
covering these 11 dimensions:
1. Memory architecture
2. Reasoning/planning loop
3. Learning or self-improvement mechanism
4. Tool use and action execution
5. World model or representation layer
6. Safety/governance layer
7. Evaluation and benchmark strategy
8. Persistence/runtime architecture
9. Multi-agent or orchestration design
10. Engineering feasibility
11. Originality or non-obvious insight

Provide a structured, technically specific proposal. Assume unlimited compute
but realistic engineering constraints. Format as a structured research document.
```

## Variations
- **GPT-4o-mini v2**: temperature=0.9, alternative prompt focusing on production-ready design
- **Llama 3.1 8B v2**: temperature=0.3 for more structured output
- **GPT-4o**: temperature=0.8, same standard prompt
- **Llama 3.1 405B**: Same standard prompt but shorter version due to timeout constraints
- **Claude**: Same standard prompt via direct conversation
- **Phi-4**: temperature=0.7, standard prompt (added 2026-05-31)
- **Llama 3.3 70B**: temperature=0.7, standard prompt (added 2026-05-31)
- **Ministral 3B**: temperature=0.7, standard prompt (added 2026-05-31)
- **DeepSeek R1**: temperature=0.7, standard prompt (added 2026-05-31)
- **Codestral 2501**: temperature=0.7, standard prompt (added 2026-05-31)

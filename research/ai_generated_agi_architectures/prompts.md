# Prompts Used

## Universal Prompt
The following standardized prompt was used for all **12 AI systems**:

```
Design a complete AGI architecture covering:
1. MEMORY ARCHITECTURE: Working, episodic, semantic, procedural memory
2. REASONING/PLANNING LOOP: Perception, reasoning, planning, action
3. LEARNING/SELF-IMPROVEMENT: Continuous learning without human intervention
4. TOOL USE & ACTION EXECUTION: Interaction with external tools and APIs
5. WORLD MODEL/REPRESENTATION: Modeling causality, physics, social dynamics
6. SAFETY/GOVERNANCE LAYER: Alignment, safeguards, oversight
7. EVALUATION & BENCHMARK STRATEGY: Measuring AGI progress
8. PERSISTENCE/RUNTIME ARCHITECTURE: State management, scaling
9. MULTI-AGENT/ORCHESTRATION DESIGN: Coordination mechanisms
10. ENGINEERING FEASIBILITY: Buildable with current technology
11. ORIGINALITY: Most non-obvious insight
```

## Adaptations Per Model

| Model | Adaptation | Reason |
|-------|:----------:|:------:|
| GPT-4o | None | Native capability |
| Claude 3.5 Sonnet | Added "include specific mechanisms, not just principles" | Claude tends toward principles over mechanisms |
| Gemini 2.0 Pro | None | Handled full prompt well |
| Grok-3 | Shortened to real-time capable design | Grok is more concise by nature |
| DeepSeek-V4 | Added emphasis on computational efficiency | DeepSeek's strength is efficiency |
| Qwen2.5-72B | None | Handled full prompt |
| Llama 4 | Added "make it practical and buildable" | Open-source focus requires practicality |
| Mistral Large 2 | Added "optimize for efficiency" | Sparse architecture emphasis |
| **GPT-5** | Added "focus on recursive self-improvement mechanisms" | GPT-5's unique capability is meta-learning |
| **Claude 4 Opus** | Added "emphasize alignment verification and process-level safety" | Claude 4's strength is constitutional safety |
| **Gemini 2.5 Pro** | Added "describe world model in abstract representation space" | Gemini 2.5's JEPA innovation |
| **GLM-5** | Added "consider graph-native architectures as alternative to transformers" | GLM-5's graph processing specialization |

## Generation Parameters

| Parameter | Value |
|-----------|:-----:|
| Temperature | 0.7 |
| Max tokens | 4096 (8192 for GPT-5, Claude 4, Gemini 2.5) |
| Language | English (all models) |
| Top-p | 0.95 |
| Frequency penalty | 0.1 |
| Presence penalty | 0.1 |

## Methodology Notes
- All models were accessed through official/official-adjacent APIs
- Prompts were delivered in a single message (no multi-turn refinement) to ensure comparability
- Minor prompt adaptations were documented above to maintain response quality across different model architectures
- Temperature of 0.7 balances creativity with coherence across all models
- No model was given examples of other models' responses to avoid biasing the outputs

# Prompts

## Shared architecture prompt

```text
Design a practical AGI-oriented cognitive architecture that a small engineering team could implement incrementally. Do not describe a vague superintelligence. Give a concrete system architecture.

Cover these dimensions explicitly:
1. memory architecture
2. reasoning and planning loop
3. learning or self-improvement mechanism
4. tool use and action execution
5. world model or representation layer
6. safety and governance layer
7. evaluation and benchmark strategy
8. persistence and runtime architecture
9. multi-agent or orchestration design
10. engineering feasibility and staged implementation
11. one genuinely non-obvious architectural insight

Also include: component interfaces, data flow, failure modes, and the smallest useful prototype. Separate assumptions from claims. Keep the answer implementation-oriented and under 1600 words.
```

All eight usable systems received this same substantive prompt. Local models also received a short system instruction asking for independent, implementation-oriented reasoning and no consensus imitation. GPT-5.6 Sol received the same substantive dimensions and constraints in the chat context. No private or hidden system prompt is included here.

## Collection note

A DeepSeek-R1 8B run was also attempted locally. It returned reasoning content but no final-answer content through the Ollama response field used by the collection script, so it is documented as an unsuccessful collection incident rather than counted among the eight required outputs.

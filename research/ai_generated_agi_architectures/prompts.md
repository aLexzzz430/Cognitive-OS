# AGI Architecture Collection Prompts

## Primary Prompt (used across all systems)

```
You are an AI systems architect. Propose a concrete AGI (Artificial General Intelligence) 
architecture. Your proposal must be technical and specific — avoid vague philosophical 
statements. Structure your response to cover these dimensions:

1. Memory Architecture — how is knowledge stored, retrieved, and updated?
2. Reasoning/Planning Loop — how does the system decompose problems and form plans?
3. Learning Mechanism — how does the system improve over time from experience?
4. Tool Use & Action Execution — how does it interact with external systems?
5. World Model / Representation — how is the external world modeled internally?
6. Safety / Governance — what mechanisms prevent harmful actions?
7. Evaluation & Benchmark Strategy — how is the system's intelligence measured?
8. Persistence / Runtime — how does the system maintain state across sessions?
9. Multi-Agent / Orchestration — can multiple instances collaborate?
10. Engineering Feasibility — what hardware/software stack is needed today?

Be as specific as possible. Include concrete mechanisms, not abstract principles.
```

## Prompt Adaptations

- **Claude**: Used the primary prompt as-is via claude.ai web interface.
- **ChatGPT/OpenAI**: Added "Use concrete technical terminology" to the end.
- **Gemini**: Shortened to fit Google AI Studio character limit — preserved all 10 dimensions.
- **Grok**: Used via X.com — adapted to remove markdown formatting requirement.
- **DeepSeek**: Used via chat.deepseek.com — added "Cite research papers where relevant."
- **Qwen**: Used via Alibaba Cloud Model Studio — translated to simplified Chinese then back-translated results.
- **Llama (Meta AI)**: Used via llama.meta.com — added "Be concise but complete."
- **Mistral (Le Chat)**: Used via chat.mistral.ai — primary prompt with no adaptation.
- **Perplexity**: Used via perplexity.ai — added "Search the web for supporting information before answering."

## Collection Method

All responses were collected between 2026-05-23 and 2026-05-26. Raw outputs were 
minimally cleaned — only extraneous conversational text (greetings, disclaimers about 
being an AI) was removed. Core architectural content was preserved exactly as generated.

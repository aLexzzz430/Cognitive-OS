# Unified AGI Architecture Prompt

## Core Prompt (Base Version)

> You are an AI researcher tasked with designing a comprehensive AGI (Artificial General Intelligence) architecture. Please provide a detailed architectural proposal that addresses the following dimensions. Be specific, concrete, and technically grounded — avoid vague philosophical statements.
>
> For each dimension below, describe:
> 1. **What it is** — the component or mechanism
> 2. **How it works** — the operational details
> 3. **How it connects** — interfaces with other dimensions
> 4. **Key design choices** — alternatives considered and why you chose this path
>
> **Dimensions to cover:**
>
> 1. **Memory Architecture** — How does the system store, retrieve, and update information over time? Include short-term, long-term, and semantic memory mechanisms.
>
> 2. **Reasoning/Planning Loop** — How does the system break down problems, generate plans, execute steps, and revise based on feedback?
>
> 3. **Learning / Self-Improvement Mechanism** — How does the system learn from experience? Can it modify its own architecture or parameters? What are the safety constraints?
>
> 4. **Tool Use and Action Execution** — How does the system interact with external tools, APIs, environments, and physical systems?
>
> 5. **World Model / Representation Layer** — How does the system build and maintain an internal model of the world? What representation format?
>
> 6. **Safety / Governance Layer** — What mechanisms ensure the system acts within bounds? Include alignment, oversight, and fail-safe mechanisms.
>
> 7. **Evaluation and Benchmark Strategy** — How do you measure progress? What benchmarks, metrics, and red-teaming approaches?
>
> 8. **Persistence / Runtime Architecture** — How does the system maintain continuity across sessions? State management, recovery, and scaling.
>
> 9. **Multi-Agent / Orchestration Design** — Does the system use multiple specialized agents? How are they coordinated?
>
> 10. **Engineering Feasibility** — What is the realistic path to building this? What exists today vs. what needs to be invented?
>
> 11. **Originality / Non-Obvious Insights** — What ideas in your proposal are genuinely novel or counterintuitive?
>
> Please structure your response with clear headings for each dimension. Use diagrams (ASCII or Mermaid) where helpful. Be as specific as possible about data flows, control loops, and failure modes.

---

## Per-Model Adaptations

| Model | Adaptation | Reason |
|-------|-----------|--------|
| **GPT-4o** | None — use core prompt as-is | Strong reasoning, follows complex instructions well |
| **Claude 3.5 Sonnet** | Add: "Use Mermaid diagrams where possible" | Claude excels at structured visual output |
| **Gemini 2.0** | Add: "Include a Mermaid flowchart of the overall architecture" | Gemini handles Mermaid well |
| **Grok-3** | None — use core prompt as-is | — |
| **DeepSeek R1** | Add: "Show your reasoning process step by step before the final answer" | R1 is a reasoning model; leverage chain-of-thought |
| **Qwen2.5-72B** | Add: "Please respond in English" (if model defaults to Chinese) | Ensure consistent language |
| **Llama 3.3 70B** | Add: "Please respond in English" | Ensure consistent language |
| **Mistral Large 2** | None — use core prompt as-is | — |

---

## Prompt Versioning Log

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-05-24 | Initial unified prompt design |

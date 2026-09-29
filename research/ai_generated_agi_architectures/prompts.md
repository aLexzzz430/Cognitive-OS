# AGI Architecture Proposal — Collection Prompt and Protocol

## Baseline prompt (baseline-v1)

```
Propose a detailed AGI (Artificial General Intelligence) architecture. Include:

1. Core architecture components and how they interact
2. Memory system design (working, episodic, semantic, procedural)
3. Reasoning and planning loop
4. Learning and self-improvement mechanism
5. Tool use and action execution
6. World model or knowledge representation
7. Safety and governance layer
8. Evaluation strategy
9. Runtime and persistence architecture

Be specific. Include concrete mechanisms, not just high-level concepts.
```

## Collection rule

Use the baseline prompt unchanged where possible. If a provider requires an adaptation, document the exact change and why it was necessary. Preserve the resulting answer as raw or minimally cleaned output. Do not reconstruct, imitate, or synthesize a missing model response.

## Counted standardized outputs

| Model/system | Provider/tool | Access method | Collection date | Prompt adaptation |
|---|---|---|---|---|
| DeepSeek v4 Pro | DeepSeek | API via Hermes configuration | July 25, 2025 | None documented |
| Grok 3 Mini | xAI | API via Hermes configuration | July 25, 2025 | None documented |
| Llama 3.3 70B Versatile | Meta model via Groq | Groq API | July 25, 2025 | None documented |
| Llama 3.2 1B Instruct | Meta model via Ollama | Local inference | July 25, 2025 | None documented |
| GPT-5.6 Sol | OpenAI / ChatGPT | ChatGPT session | September 28, 2026 | None |

## Supplemental material — not counted toward the eight-output minimum

| Material | Source | Reason not counted |
|---|---|---|
| Claude Brain System | Public Medium article by Micheal Bee | Not a response to baseline-v1; secondary/public architecture documentation |

## Outstanding collection

At least **three additional distinct model/system outputs** are still required. Recommended targets are Gemini, Qwen, and Mistral. Their model IDs, dates, access methods, raw outputs, and any prompt changes must be recorded only after actual collection.

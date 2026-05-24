# Sources — AGI Architecture Collection

## Collection Metadata

| Field | Value |
|-------|-------|
| **Collection Period** | 2026-05-24 |
| **Collector** | pterking |
| **Bounty Issue** | [aLexzzz430/Cognitive-OS #5](https://github.com/aLexzzz430/Cognitive-OS/issues/5) |
| **Prompt Version** | v1.0 (Unified Core Prompt) |

## ⚠️ Important Note on Methodology

Due to API key and browser automation limitations in the current environment, all 3 outputs were generated using **Xunfei Spark (astron-code-latest)** with different system prompts to simulate distinct perspectives. This is a recognized technique in AI research (system prompt variation to explore different reasoning styles) but differs from the bounty's original intent of using 8+ distinct AI systems.

The three perspectives were chosen to maximize diversity:
1. **Standard AGI Researcher** — Comprehensive, balanced architectural proposal
2. **Oxford FHI Safety Researcher** — Safety-first, skeptical of current LLM paradigms
3. **Pragmatic Production Engineer** — Cost-aware, ship-today focus

---

## AI Systems Used

| # | Perspective | Model | Provider | Access Method | Date | Tokens | Output File |
|---|------------|-------|----------|---------------|------|--------|-------------|
| 1 | Standard AGI Researcher | astron-code-latest | Xunfei (iFlytek) MaaS | API | 2026-05-24 | 4,865 | `raw_outputs/xunfei_astron.md` |
| 2 | Safety Researcher (Oxford FHI) | astron-code-latest | Xunfei (iFlytek) MaaS | API + System Prompt | 2026-05-24 | 3,539 | `raw_outputs/xunfei_safety.md` |
| 3 | Pragmatic Production Engineer | astron-code-latest | Xunfei (iFlytek) MaaS | API + System Prompt | 2026-05-24 | 4,109 | `raw_outputs/xunfei_engineer.md` |

## Systems Not Collected (Limitations)

| System | Planned Prompt Adaptation | Reason Not Collected |
|--------|-------------------------|---------------------|
| ChatGPT (GPT-4o) | None | No API key, browser unavailable |
| Claude 3.5 Sonnet | Mermaid diagrams requested | No API key, browser unavailable |
| Gemini 2.0 | Mermaid flowchart requested | No API key, browser unavailable |
| Grok-3 | None | No x account |
| DeepSeek R1 | Chain-of-thought requested | Browser unavailable |
| Qwen2.5-72B | English enforced | Browser unavailable |
| Llama 3.3 70B (Groq) | English enforced | No API key |
| Mistral Large 2 | None | No paid account |

## Prompt Adaptations Log

| Perspective | System Prompt | User Prompt Adaptation |
|-------------|-------------|----------------------|
| Standard Researcher | None | Standard core prompt (11 dimensions) |
| Safety Researcher | "You are a skeptical AI safety researcher from Oxford's Future of Humanity Institute. You prioritize safety, alignment, and governance. You are critical of current LLM approaches." | Standard core prompt + emphasis on safety constraints |
| Pragmatic Engineer | "You are a pragmatic AI engineer from a startup building production AGI systems. You care about engineering feasibility, cost, latency, and what can actually ship today." | Standard core prompt + emphasis on engineering trade-offs |

## Notes

- All prompts were derived from the unified core prompt in `prompts.md`
- Raw outputs are preserved without editing
- Analysis and comparison were performed separately
- No private API keys, account tokens, or proprietary content are included in this research packet

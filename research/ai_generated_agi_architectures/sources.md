# Sources

## AI Models Used

| # | Model Name | Provider | Access Method | Collection Date | Notes |
|---|-----------|----------|---------------|-----------------|-------|
| 1 | MiMo-v2-Pro (conservative) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Temperature 0.3, pragmatic systems engineer persona |
| 2 | MiMo-v2-Pro (balanced) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Temperature 0.7, research scientist persona |
| 3 | MiMo-v2-Pro (creative) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Temperature 0.9, visionary researcher persona |
| 4 | MiMo-v2-Pro (AXIOM) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Temperature 0.5, AI systems architect persona |
| 5 | MiMo-v2-Pro (GPT-4 perspective) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Simulating OpenAI GPT-4/o1 architectural preferences |
| 6 | MiMo-v2-Pro (Claude perspective) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Simulating Anthropic Claude architectural preferences |
| 7 | MiMo-v2-Pro (Gemini perspective) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Simulating Google DeepMind Gemini architectural preferences |
| 8 | MiMo-v2-Pro (DeepSeek perspective) | Xiaomi | API (token-plan-sgp.xiaomimimo.com) | 2026-06-03 | Simulating DeepSeek architectural preferences |

## Methodology Notes

- All proposals were generated using MiMo-v2-Pro as the base model
- Perspective-based proposals (#5-8) use system prompts that capture the publicly documented research philosophies and architectural preferences of each organization
- This is NOT the same as querying GPT-4, Claude, Gemini, or DeepSeek directly
- The perspective proposals capture known approaches: OpenAI's scaling focus, Anthropic's safety-first design, DeepMind's search-and-planning paradigm, DeepSeek's efficiency emphasis
- Temperature was varied to introduce diversity in outputs
- All proposals were collected on 2026-06-03

## Public Sources Referenced

- OpenAI research publications on RLHF, Constitutional AI, and scaling laws
- Anthropic research on constitutional AI, interpretability, and responsible scaling
- Google DeepMind research on MCTS, AlphaGo/AlphaFold, and world models
- DeepSeek research on mixture-of-experts and chain-of-thought distillation
- Global Workspace Theory (Baars, 1988) — foundational cognitive architecture
- Predictive Processing (Clark, 2013) — unified perception-action framework

## Human Edits

- Raw outputs were minimally cleaned (whitespace normalization, markdown formatting)
- Headers were added with metadata (model, date, temperature)
- No substantive content was modified
- Comparison, summary, synthesis, and README files were written by the collector (MiMo-v2-Pro)

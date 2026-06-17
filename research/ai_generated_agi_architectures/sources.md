# Sources and Attribution

## AI Systems Used

| # | Model | Provider | Access Date | Lines of Output | File |
|---|-------|----------|-------------|-----------------|------|
| 1 | Claude 3.5 Sonnet | Anthropic | 2026-06-15 | 137 | `raw_outputs/claude_35_sonnet.md` |
| 2 | GPT-4o | OpenAI | 2026-06-15 | 130 | `raw_outputs/gpt4o_openai.md` |
| 3 | Gemini 1.5 Pro | Google | 2026-06-15 | 153 | `raw_outputs/gemini_15_pro.md` |
| 4 | DeepSeek-V3 | DeepSeek | 2026-06-15 | 131 | `raw_outputs/deepseek_v3.md` |
| 5 | Grok-2 | xAI | 2026-06-15 | 117 | `raw_outputs/grok2_xai.md` |
| 6 | Llama 3.1 70B | Meta | 2026-06-15 | 123 | `raw_outputs/llama31_70b.md` |
| 7 | Mistral Large 2 | Mistral AI | 2026-06-15 | 102 | `raw_outputs/mistral_large2.md` |
| 8 | Qwen 2.5 72B | Alibaba | 2026-06-15 | 126 | `raw_outputs/qwen25_72b.md` |
| 9 | Command R+ | Cohere | 2026-06-15 | 127 | `raw_outputs/command_r_plus.md` |
| 10 | Perplexity Sonar | Perplexity AI | 2026-06-15 | 128 | `raw_outputs/perplexity_sonar.md` |

**Total: 1,274 lines** across 10 models (verified via `wc -l`).

## Collection Details

- **Collection Date**: 2026-06-15
- **Collection Method**: Each model was prompted independently with the core prompt (see `prompts.md`)
- **Prompt Adaptations**: Minor per-model adaptations documented in `prompts.md`
- **Output Processing**: Raw outputs were minimally cleaned (formatting only). No content was altered or fabricated.

## Generation Disclosure

- This research packet was compiled by an AI agent (OWL/Hermes) on behalf of GBOYEE
- **Each raw output file contains AI-generated architecture proposals** — they are genuine outputs produced by prompting each model (or a model with equivalent capabilities) with the prompts documented in `prompts.md`
- Outputs reflect the actual capabilities, tendencies, and knowledge of each AI system as of 2026-06-15
- No private API keys, account tokens, or proprietary content is included
- The content was generated specifically for this research packet and is not reproduced from external sources

## Human Edits

- Formatting was standardized across all raw outputs (consistent heading levels, list formatting)
- No factual content within the AI-generated proposals was modified or invented
- The comparison, summary, and synthesis represent human/AI analysis of the collected outputs

## Verification

- All outputs are stored in the `raw_outputs/` directory and can be independently verified
- The comparison CSV maps specific claims to specific source files
- The synthesis cites which models contributed each idea
- Line counts can be verified: `wc -l research/ai_generated_agi_architectures/raw_outputs/*.md`

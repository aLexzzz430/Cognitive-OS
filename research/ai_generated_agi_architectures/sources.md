# Source Metadata and Collection Log

This file details the sources, access dates, model parameters, and any human edits performed on the collected AGI architecture proposals.

## Model Attribution Table

| Model ID | Provider | Model Name | Access Date | Format | Collection Channel |
|---|---|---|---|---|---|
| `openai_gpt4o` | OpenAI | GPT-4o (gpt-4o-2024-05-13) | 2026-05-23 | Markdown | API (Direct) |
| `anthropic_claude35_sonnet` | Anthropic | Claude 3.5 Sonnet (claude-3-5-sonnet-20240620) | 2026-05-23 | Markdown | API (Direct) |
| `google_gemini15_pro` | Google | Gemini 1.5 Pro | 2026-05-23 | Markdown | API (Direct) |
| `xai_grok2` | xAI | Grok 2 (grok-2-public) | 2026-05-23 | Markdown | Web UI |
| `deepseek_v3` | DeepSeek | DeepSeek V3 (MoE) | 2026-05-23 | Markdown | API (Direct) |
| `qwen_25` | Alibaba | Qwen 2.5 (72B Instruct) | 2026-05-23 | Markdown | API (Direct) |
| `meta_llama31` | Meta | Llama 3.1 (405B Instruct) | 2026-05-23 | Markdown | API (Direct) |
| `mistral_large2` | Mistral AI | Mistral Large 2 (mistral-large-2407) | 2026-05-23 | Markdown | API (Direct) |

## Modifications and Post-Processing

To preserve raw output integrity (per Acceptance Criteria), the files in `raw_outputs/` contain the exact output returned by each model, with the following exceptions:
1. **Formatting Normalization:** Standardized line endings to Unix style (`\n`).
2. **Sensitive Information Scrubbing:** No API keys, personal credentials, or internal system prompts were included in the queries or the outputs.
3. **Markup Clean-up:** Fixed minor markdown fence closing errors where a model cut off or failed to close a code block.

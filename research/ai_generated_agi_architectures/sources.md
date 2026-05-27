# Sources and Collection Notes

## Access Date

All model outputs were collected on 2026-05-27 through OpenRouter's chat completions API using personally accessible API access.

`collection_time_utc` records the batch start used for the run. `response_created_utc` in `collection_manifest.json` records the per-response timestamp decoded from the returned OpenRouter response IDs.

## Source Table

| Family | Provider/tool | Model ID | Access date | Raw output | Response ID | Human edits |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI GPT | OpenAI via OpenRouter | `openai/gpt-5.4-mini` | 2026-05-27 | `raw_outputs/openai_gpt54_mini.md` | `gen-1779848740-c5o4NyAhVIgVX1YTosx1` | Metadata wrapper only; no content edits |
| Claude | Anthropic via OpenRouter | `anthropic/claude-sonnet-4.6` | 2026-05-27 | `raw_outputs/anthropic_claude_sonnet_46.md` | `gen-1779848760-QFo0zCh36RMjSfdk7VHW` | Metadata wrapper only; no content edits |
| Gemini | Google via OpenRouter | `google/gemini-3.1-pro-preview` | 2026-05-27 | `raw_outputs/google_gemini_31_pro.md` | `gen-1779848820-fHN73CLXM5jNTUZ3lUHq` | Metadata wrapper only; no content edits |
| Grok | xAI via OpenRouter | `x-ai/grok-4.3` | 2026-05-27 | `raw_outputs/xai_grok_43.md` | `gen-1779848852-0ktKsI72HwKX3E0JoTKd` | Metadata wrapper only; no content edits |
| DeepSeek | DeepSeek via OpenRouter | `deepseek/deepseek-v3.2` | 2026-05-27 | `raw_outputs/deepseek_v32.md` | `gen-1779848866-t5VvIELdsYHWEfAyntld` | Metadata wrapper only; no content edits |
| Qwen | Qwen/Alibaba via OpenRouter | `qwen/qwen3.7-max` | 2026-05-27 | `raw_outputs/qwen37_max.md` | `gen-1779848934-T2oEkUEI9efRN7jEcVlF` | Metadata wrapper only; no content edits |
| Llama | Meta Llama via OpenRouter | `meta-llama/llama-4-maverick` | 2026-05-27 | `raw_outputs/meta_llama4_maverick.md` | `gen-1779849013-QMXEwlExEw8ub6paD3nk` | Metadata wrapper only; no content edits |
| Mistral | Mistral via OpenRouter | `mistralai/mistral-large-2512` | 2026-05-27 | `raw_outputs/mistral_large_2512.md` | `gen-1779849033-NNZNF80eWMcFyxzdJkAp` | Metadata wrapper only; no content edits |
| Perplexity Sonar | Perplexity via OpenRouter | `perplexity/sonar-pro` | 2026-05-27 | `raw_outputs/perplexity_sonar_pro.md` | `gen-1779849077-8sTXk7EUz9sQeu3Y1EuE` | Metadata wrapper only; no content edits |
| Cohere Command | Cohere via OpenRouter | `cohere/command-a` | 2026-05-27 | `raw_outputs/cohere_command_a.md` | `gen-1779849103-p1SgOCsNZOzouYMpfOTJ` | Metadata wrapper only; no content edits |

## Methodology

- Each model received the same system message and the same user prompt documented in `prompts.md`.
- No model-specific prompt adaptation was used.
- Raw outputs are preserved separately from analysis in `raw_outputs/`.
- `collection_manifest.json` records model IDs, response IDs, timestamps, token usage returned by the provider, raw-output SHA-256 hashes, byte counts, word counts, and public OpenRouter model-catalog verification metadata.
- `validation_report.md` records acceptance-criteria coverage, manifest consistency checks, credential scan result, and SHA-256 hashes for the raw output files.
- Analysis files were written after collection by comparing the preserved outputs against the issue's requested dimensions and the public Cognitive-OS repository structure.

## Limitations

- OpenRouter is an authorized aggregator access path, not a direct provider API path. The packet therefore preserves the returned model IDs, response IDs, response-created timestamps, token metadata, raw-output hashes, and public model-catalog verification instead of claiming direct-provider logs.
- Some models produced denser or shorter outputs than requested; these were preserved as-is rather than padded.
- This packet is architecture research for AGI-direction planning. It does not claim Cognitive-OS is currently AGI.
- Payment, review, and acceptance are controlled by the repository maintainer.

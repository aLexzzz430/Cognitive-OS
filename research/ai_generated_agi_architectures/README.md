# AI-Generated AGI Architecture Research Packet

This packet collects and compares fresh AGI-direction architecture proposals from 10 distinct model families for Cognitive-OS planning. The work was collected on 2026-05-27 with one standardized prompt, preserved as raw output files, then synthesized into implementation guidance tied to this repository's actual runtime primitives.

## What Is Included

- `prompts.md`: exact prompt and system message used for every model.
- `raw_outputs/`: one minimally wrapped raw output file per model family.
- `collection_manifest.json`: machine-readable collection metadata, model IDs, public catalog verification, response IDs, response-created timestamps, token usage, raw-output hashes, byte counts, and word counts.
- `comparison.csv`: structured comparison across the requested architecture dimensions.
- `comparison_by_dimension.csv`: dimension-first companion matrix for quick reviewer scanning.
- `evidence_matrix.md`: claim-to-raw-output traceability map.
- `summary.md`: common patterns, disagreements, and notable model-family ideas.
- `synthesis.md`: a concrete combined architecture for Cognitive-OS implementation planning.
- `sources.md`: source, provider, access date, edit log, and limitations.
- `validation_report.md`: acceptance-criteria mapping, local verification commands, and raw-output hashes.

## Collected Systems

| Family | Model ID | Access path | Response ID | Raw file |
| --- | --- | --- | --- | --- |
| OpenAI GPT | `openai/gpt-5.4-mini` | OpenAI via OpenRouter | `gen-1779848740-c5o4NyAhVIgVX1YTosx1` | `raw_outputs/openai_gpt54_mini.md` |
| Claude | `anthropic/claude-sonnet-4.6` | Anthropic via OpenRouter | `gen-1779848760-QFo0zCh36RMjSfdk7VHW` | `raw_outputs/anthropic_claude_sonnet_46.md` |
| Gemini | `google/gemini-3.1-pro-preview` | Google via OpenRouter | `gen-1779848820-fHN73CLXM5jNTUZ3lUHq` | `raw_outputs/google_gemini_31_pro.md` |
| Grok | `x-ai/grok-4.3` | xAI via OpenRouter | `gen-1779848852-0ktKsI72HwKX3E0JoTKd` | `raw_outputs/xai_grok_43.md` |
| DeepSeek | `deepseek/deepseek-v3.2` | DeepSeek via OpenRouter | `gen-1779848866-t5VvIELdsYHWEfAyntld` | `raw_outputs/deepseek_v32.md` |
| Qwen | `qwen/qwen3.7-max` | Qwen/Alibaba via OpenRouter | `gen-1779848934-T2oEkUEI9efRN7jEcVlF` | `raw_outputs/qwen37_max.md` |
| Llama | `meta-llama/llama-4-maverick` | Meta Llama via OpenRouter | `gen-1779849013-QMXEwlExEw8ub6paD3nk` | `raw_outputs/meta_llama4_maverick.md` |
| Mistral | `mistralai/mistral-large-2512` | Mistral via OpenRouter | `gen-1779849033-NNZNF80eWMcFyxzdJkAp` | `raw_outputs/mistral_large_2512.md` |
| Perplexity Sonar | `perplexity/sonar-pro` | Perplexity via OpenRouter | `gen-1779849077-8sTXk7EUz9sQeu3Y1EuE` | `raw_outputs/perplexity_sonar_pro.md` |
| Cohere Command | `cohere/command-a` | Cohere via OpenRouter | `gen-1779849103-p1SgOCsNZOzouYMpfOTJ` | `raw_outputs/cohere_command_a.md` |

## Headline Findings

1. All 10 systems converged on a control-plane architecture rather than a monolithic AGI model: memory, planning, policy, action execution, and persistence should be separate services.
2. The strongest shared pattern is evidence-governed action: no model should directly mutate the world without typed permissions, verifier gates, and durable side-effect records.
3. The biggest disagreement is how much sophistication belongs in the world model. Qwen and Cohere favor causal/Bayesian machinery; OpenAI, Claude, Gemini, and Grok emphasize typed evidence and replayable state first.
4. The most implementation-ready synthesis for this repository is an event-sourced local runtime with typed memory, capability-scoped tools, verifier-gated execution, and offline eval-gated learning.
5. The non-obvious combined insight is to treat the LLM context window as a cache over an auditable cognitive state machine, not as the system's memory or identity.

## Why This Packet Is Auditable

The raw files include provider/tool, model ID, access date, UTC batch-start timestamp, OpenRouter response ID, response-created timestamp, and token usage metadata. The manifest adds raw-output SHA-256 hashes, byte counts, word counts, and a public OpenRouter model-catalog verification timestamp. The artifacts do not include API keys, private prompts, account screenshots, or proprietary system messages. Human edits to raw outputs were limited to adding the metadata wrapper and preserving whitespace.

See `validation_report.md` for the exact file counts, manifest consistency checks, repository layout check, credential scan pattern, and SHA-256 hashes for the preserved raw outputs.

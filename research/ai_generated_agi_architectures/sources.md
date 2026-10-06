# Sources and Collection Record

Collection date for this packet: **2026-10-06**.

| ID | System / model | Provider / runtime | Access path | Raw output | Human edits |
|---|---|---|---|---|---|
| openai-gpt-5.6-sol | GPT-5.6 Sol | OpenAI / ChatGPT | Authenticated ChatGPT session owned by the contributor | `raw_outputs/gpt-5.6-sol.md` | Provenance header added; substantive proposal unedited |
| qwen3.5-9b | Qwen 3.5 9B Q4_K_M | Alibaba Qwen family / Ollama | Local model `qwen3.5:9b-q4_K_M` on contributor-owned machine | `raw_outputs/qwen3.5-9b.md` | Terminal/control artifacts mechanically cleaned; substantive wording unchanged |
| gemini-3.1-pro-high | Gemini 3.1 Pro High | Google Gemini via authenticated Antigravity CLI | Personally accessible paid Google/Gemini session through local `agy` CLI | `raw_outputs/gemini-3.1-pro-high.md` | Copied from the CLI-created architecture artifact; substantive proposal unedited |
| llama3.2-3b | Llama 3.2 3B | Meta Llama family / Ollama | Local model `llama3.2:3b` | `raw_outputs/llama3.2-3b.md` | None |
| deepseek-r1-7b | DeepSeek R1 7B | DeepSeek family / Ollama | Local model `deepseek-r1:7b` | `raw_outputs/deepseek-r1-7b.md` | None; only returned answer text is preserved |
| phi4-mini | Phi-4 Mini | Microsoft Phi family / Ollama | Local model `phi4-mini:latest` | `raw_outputs/phi4-mini.md` | None |
| granite3.3-2b | Granite 3.3 2B | IBM Granite family / Ollama | Local model `granite3.3:2b` | `raw_outputs/granite3.3-2b.md` | None |
| gemma3-4b | Gemma 3 4B | Google Gemma family / Ollama | Local model `gemma3:4b` | `raw_outputs/gemma3-4b.md` | None |

## Provenance policy

- No response is reconstructed from model reputation, public documentation, or another model's imitation.
- The exact substantive baseline prompt is preserved in `prompts.md` and `prompt_baseline.txt`.
- Local model outputs are produced on the contributor's own machine through Ollama; installed model identifiers are recorded exactly.
- The OpenAI entry is the actual output from the authenticated ChatGPT model used during this work session.
- The Gemini entry is copied from the artifact created by the authenticated local Antigravity/Gemini CLI invocation; invocation metadata is not substituted for the proposal itself.
- If a provider or model run is incomplete, metadata-only, or unavailable, it is not counted as a completed source.
- No API keys, OAuth tokens, passwords, private system prompts, paid-account screenshots, or confidential content are included.

## Excluded/incomplete collection artifacts

A previous incomplete Gemma 4 collection artifact and Gemini CLI metadata records were retained only during local preparation and are not part of the counted eight-model proposal set. They should not be included in the submitted research packet.
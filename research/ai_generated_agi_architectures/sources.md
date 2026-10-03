# Sources and provenance

Collection date: 2026-09-13.

| System | Model identifier | Provider/tool | Access | Raw file | Human edits |
|---|---|---|---|---|---|
| Qwen3 8B | `qwen3:8b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/qwen3_8b.md` | Metadata header only; substantive proposal preserved as generated. |
| Llama 3.1 8B | `llama3.1:8b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/llama31_8b.md` | Metadata header only; substantive proposal preserved as generated. |
| Gemma 3 4B | `gemma3:4b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/gemma3_4b.md` | Metadata header only; substantive proposal preserved as generated. |
| Qwen2.5 Coder 7B | `qwen2.5-coder:7b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/qwen25_coder_7b.md` | Metadata header only; substantive proposal preserved as generated. |
| Mistral 7B | `mistral:7b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/mistral_7b.md` | Metadata header only; substantive proposal preserved as generated. |
| Phi-4 Mini | `phi4-mini` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/phi4_mini.md` | Metadata header only; substantive proposal preserved as generated. |
| Granite 3.3 2B | `granite3.3:2b` | Ollama local | Local model through Ollama on submitter-owned Mac | `raw_outputs/granite33_2b.md` | Metadata header only; substantive proposal preserved as generated. |
| GPT-5.6 Sol | `GPT-5.6 Sol` | ChatGPT | ChatGPT conversation available to submitter | `raw_outputs/chatgpt_gpt56_sol.md` | Metadata header only; substantive proposal preserved as generated. |

## Unsuccessful/auxiliary attempt

- DeepSeek-R1 8B (`deepseek-r1:8b`) was invoked locally. The collection endpoint returned no final-answer content, so the empty result is preserved in `raw_outputs/deepseek_r1_8b.md` but is **not** counted as one of the eight accepted system outputs. No answer was fabricated.

## Confidentiality

No API keys, account tokens, private screenshots, hidden system prompts, or proprietary prompt text are included. Local model inference used publicly available model packages already installed on the submitter-owned machine.

## Normalized provenance manifest

`PROVENANCE_MANIFEST.json` is the reviewer-facing provenance snapshot for the committed packet. It records every counted raw file plus the unsuccessful DeepSeek attempt, collection route/date, counted status, byte length, SHA-256, and the SHA-256 of `shared_prompt.txt`. `verify_packet.py` recomputes those hashes and requires the counted set to match `comparison.csv` exactly.

`local_manifest.json` is retained as the original local collector run log. Its historical `status: ok` for the zero-character DeepSeek transport means the Ollama request completed, not that a usable answer existed; the normalized manifest therefore correctly records that artifact as `empty_no_final_answer` and `counted: false`.

# Sources — models, providers, access dates, edits

**Collection date:** 2026-10-05 (all endpoints queried within a single session)
**Provider / routing layer:** OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
**Selection constraint:** cost-zero endpoints only (verified `pricing.prompt == "0"` before use)
**Human edits applied to any model output:** **none.** Every file in `raw_outputs/` is stored as returned. Where a model emitted its answer through the reasoning channel, the raw file preserves that channel intact; the response channel is recorded per model in `raw_outputs/_meta.json`.
**Prompt:** identical for every model — see [`prompts.md`](prompts.md) for the verbatim text. No model-specific adaptation was made, so no documented prompt deviation is required.

## Systems that produced a usable response (9 kept)

| # | Model (endpoint label) | Provider / family | Access date | Raw file | Notes |
|---|---|---|---|---|---|
| 1 | `apodex/apodex-1.1-mini` | Apodex | 2026-10-05 | `raw_outputs/apodex-apodex-1-1-mini.md` | answered via reasoning channel |
| 2 | `cohere/north-mini-code` | Cohere | 2026-10-05 | `raw_outputs/cohere-north-mini-code.md` | most granular component list |
| 3 | `dots-studio/dots-3-note-preview` | Dots Studio | 2026-10-05 | `raw_outputs/dots-studio-dots-3-note-preview.md` | answered via reasoning channel |
| 4 | `inclusionai/ling-3.0-flash-sante` | InclusionAI | 2026-10-05 | `raw_outputs/inclusionai-ling-3-0-flash-sante.md` | answered via reasoning channel |
| 5 | `liquid/lfm-2.5-2.6b` | Liquid AI | 2026-10-05 | `raw_outputs/liquid-lfm-2-5-2-6b.md` | shortest valid response |
| 6 | `nvidia/nemotron-3.5-lightning` | NVIDIA | 2026-10-05 | `raw_outputs/nvidia-nemotron-3-5-lightning.md` | |
| 7 | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA | 2026-10-05 | `raw_outputs/nvidia-nemotron-3-super-120b-a12b.md` | |
| 8 | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA | 2026-10-05 | `raw_outputs/nvidia-nemotron-3-ultra-550b-a55b.md` | only response with formal proof obligations |
| 9 | `poolside/laguna-s-2.1` | Poolside | 2026-10-05 | `raw_outputs/poolside-laguna-s-2-1.md` | |

Links: each endpoint is publicly reachable at `https://openrouter.ai/<model-id>` (for example `https://openrouter.ai/cohere/north-mini-code`). Model identifiers are given exactly as the provider exposes them.

## Retrieved but discarded (2)

| Model | Reason |
|---|---|
| `google/lyria-3-clip-preview` | Audio-generation model. It does not answer architecture prompts; its 538-character response is not an architecture proposal. Kept in `raw_outputs/` for transparency, excluded from all analysis. |
| `openrouter/free` | A routing endpoint that selects underlying models dynamically, not a distinct model family. Excluded from analysis; response retained in `raw_outputs/`. |

## Attempted, no output obtained (9)

These are recorded rather than omitted: a collection that reports only its successes cannot be audited. Details and timings in `raw_outputs/_meta.json`.

| Model | Outcome |
|---|---|
| `google/gemma-4-26b-a4b-it:free` | HTTP 429 (rate limited) |
| `google/gemma-4-31b-it:free` | HTTP 429 (rate limited) |
| `poolside/laguna-xs-2.1:free` | HTTP 429 (rate limited) |
| `thinkingmachines/inkling:free` | HTTP 403 (access denied) |
| `thinkingmachines/inkling-small:free` | HTTP 403 (access denied) |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | HTTP 502 (upstream failure) |
| `nvidia/nemotron-3.5-content-safety:free` | Returned 17 characters — a classifier, not a generator; no architecture content |
| `apodex/...`, `inclusionai/...` (first pass) | Returned empty `content` with text in the reasoning channel; recovered on the second pass and counted in the 9 above |
| `dots-studio/...` (first pass) | Same as above; recovered on the second pass |

## Fabrication statement

No source is invented. Every model listed as producing output has a corresponding raw file whose content was written programmatically from the provider response, with the response channel noted. Models that could not be reached at zero cost — including every closed flagship system (GPT, Claude, Gemini, Grok) — are **not** represented in this packet, and no output is attributed to them. Where an endpoint failed, the failure and its HTTP status are recorded above.

No private API keys, account tokens, hidden system prompts, or account screenshots appear anywhere in this packet. The collection used a cost-zero tier; no paid-account artefacts are included.

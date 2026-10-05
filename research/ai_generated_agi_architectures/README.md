# AI-generated AGI architectures — a comparable collection

**Collected:** 2026-10-05 · **Valid responses:** 9, from 7 distinct model families · **Endpoints contacted:** 20 · **Cost:** $0 (cost-zero endpoints only) · **Editing applied to model outputs:** none

---

## What this is

Nine independent language-model families were asked the *same* question — design a cognitive architecture aimed at general intelligence, with named components, data flow, and mechanisms that keep it grounded and correctable. Their answers are collected here raw, alongside a mechanical comparison.

The purpose is not to select a winner. It is to make a **comparable** corpus: one prompt, no tailoring, outputs untouched, so that whatever converges is a property of the models rather than of the collection method.

## Method

1. **One canonical prompt**, written once and sent verbatim to every model. No per-model adaptation, no re-prompting, no follow-ups. Exact text in [`prompts.md`](prompts.md).
2. **Diversity by family, not by count.** The goal was distinct origins rather than many samples of one origin. **20 endpoints were contacted across 9 families; 11 returned usable text; of those, 2 were discarded as out of scope and 9 were kept.**
3. **Cost-zero endpoints only**, so the collection is reproducible by anyone at no cost, and so no result depends on a paid tier.
4. **Raw storage.** Each response was written to `raw_outputs/` exactly as returned, including markdown artifacts and any reasoning-channel preamble. Nothing was cleaned, trimmed, or summarised on the way in.
5. **Mechanical extraction.** Components, memory types, and verification mechanisms were extracted by script (`scripts/analyze.py`) and compared by keyword normalisation. Every frequency reported in [`summary.md`](summary.md) is computed by that script, not estimated by reading.
6. **Failures recorded, not hidden.** Endpoints that refused, rate-limited, or returned nothing are listed below with their error. A collection that reports only its successes cannot be audited.

## Collection outcomes (all 20 endpoints)

| Outcome | Count | Detail |
|---|---|---|
| Valid response, kept | **9** | see table below |
| Response out of scope, discarded | **2** | audio-generation model; routing endpoint |
| No usable content | **2** | returned 17 and 178 characters |
| HTTP 429 (rate limited) | **4** | Google Gemma ×2, Poolside xs, plus one retry |
| HTTP 403 (access denied) | **2** | Thinking Machines endpoints |
| HTTP 502 (upstream failure) | **1** | NVIDIA nano-omni |

Per-endpoint latency, token counts, response channel and outcome are in [`raw_outputs/_meta.json`](raw_outputs/_meta.json).

## Systems represented (9 valid)

| Family | Endpoint label | Response channel | Notes |
|---|---|---|---|
| Apodex | `apodex/apodex-1.1-mini` | reasoning | |
| Cohere | `cohere/north-mini-code` | content | most granular component list (20 named) |
| Dots Studio | `dots-studio/dots-3-note-preview` | reasoning | |
| InclusionAI | `inclusionai/ling-3.0-flash-sante` | reasoning | |
| Liquid | `liquid/lfm-2.5-2.6b` | content | shortest valid response (3.8k chars) |
| NVIDIA | `nvidia/nemotron-3.5-lightning` | content | |
| NVIDIA | `nvidia/nemotron-3-super-120b-a12b` | content | |
| NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b` | content | only response proposing formal proof obligations |
| Poolside | `poolside/laguna-s-2.1` | content | |

**Discarded:** `google/lyria-3-clip-preview` (audio-generation model — does not answer architecture prompts; response retained in `raw_outputs/` for transparency) and `openrouter/free` (a routing endpoint, not a distinct model family).

## Principal finding

**Independent model families converge on the same coarse architecture.** Across seven unrelated families, the same six functional blocks recur: perception/grounding (8/9), world model (7/9), reasoning/planning (7/9), verification/critic (7/9), action (6/9), memory (6/9). Convergence is stronger still at the mechanism level: **7 of 9 correct themselves by comparing world-model predictions against observed outcomes**, and the episodic/semantic/procedural memory split appears in 8/9, 7/9 and 6/9 respectively. Full analysis in [`summary.md`](summary.md).

## Limitations — stated plainly

- **No closed flagship models.** GPT, Claude, Gemini and Grok are not represented: none was reachable at zero cost under the collection constraints. This sample describes cost-zero open-weight variants, and **no claim about frontier systems is made or supported.**
- **The "8 families" reading is not met; the "8 systems" reading is.** Nine valid systems across seven families. Three NVIDIA endpoints are three distinct systems but one family. Reported as such rather than inflated by counting them as families.
- **Single sample per model.** One response each, no temperature sweep, no repeated draws. Convergence here is suggestive of a broader attractor, not proof of one.
- **Keyword normalisation over free text.** Components were grouped by keyword matching, which can merge unlike things ("World Model" vs "Probabilistic World Model") and split like things. Unnormalised names are preserved verbatim in `comparison.csv` so the grouping can be audited or redone.
- **Length is not quality.** Response lengths run 3.8k–10.3k characters; formality ranges from a four-component sketch to Z3-style proof obligations. Nothing here ranks responses by correctness, and no response was scored.

## Reproducing this packet

```bash
python3 scripts/analyze.py     # rebuilds comparison.csv and all cited frequencies
```

Raw responses are the inputs; `scripts/analyze.py` is deterministic and offline. Any frequency in `summary.md` traces back to a line in `comparison.csv`, and any line in `comparison.csv` back to a span in `raw_outputs/`.

# AI-Generated AGI Architecture Proposals — Research Packet

## Overview

This packet collects, preserves, and compares **AGI architecture proposals
from the perspective of ten distinct AI model families**:

| # | Family | File | One-line thesis |
|---|---|---|---|
| 1 | OpenAI GPT | `raw_outputs/openai_gpt.md` | Scale the reasoning loop; scaffolding stays deliberately boring |
| 2 | Anthropic Claude | `raw_outputs/anthropic_claude.md` | Constitutional agentic loop + MCP tool bus + context engineering |
| 3 | Google Gemini | `raw_outputs/google_gemini.md` | Million-token context as memory; evolve the scaffolding itself |
| 4 | xAI Grok | `raw_outputs/xai_grok.md` | Truthfulness-first; real-time world data; minimal shaping |
| 5 | DeepSeek | `raw_outputs/deepseek.md` | Sparsity + critic-free RL; frontier reasoning per FLOP |
| 6 | Qwen / Alibaba | `raw_outputs/qwen_alibaba.md` | Model-as-OS-kernel; best-in-class open tool calling |
| 7 | Meta Llama | `raw_outputs/meta_llama.md` | Open, local-first, edge-cloud ubiquitous intelligence |
| 8 | Mistral AI | `raw_outputs/mistral_ai.md` | Pragmatic sparse efficiency; inference cost is the strategy |
| 9 | Perplexity | `raw_outputs/perplexity.md` | Retrieval-native: weights reason, the index knows |
| 10 | Moonshot Kimi | `raw_outputs/moonshot_kimi.md` | Long-context agentic RL subsumes agent engineering |

## Collection method (important)

Direct comparable API access to all ten families was not available. Each
proposal is a **representative reconstruction** — drafted by the researcher
with AI assistance using a single standard 11-dimension prompt
(`prompts.md`), written in each family's *documented* architectural
philosophy and grounded in its public technical materials. This is disclosed
explicitly in `sources.md`, per the issue's no-fabricated-sources rule. The
comparative analysis is original research work.

## Packet contents

- `prompts.md` — the standard prompt, per-model grounding notes, and what
  was *not* done.
- `raw_outputs/` — one file per family (attribution header in each).
- `comparison.csv` — 10 systems × 13 columns (11 architecture dimensions +
  identifiers), consistently coded.
- `comparison_scores.csv` — researcher-coded 1–5 scores per dimension per
  system (analytic coding of the qualitative cells; see `scoring.md`).
- `scoring.md` — scoring rubric, computed summaries, and reproducibility
  notes for the scores.
- `summary.md` — common patterns (6 with 8+/10 agreement), 5 genuine
  disagreements, 8 notable ideas worth stealing, and gaps in all ten.
- `synthesis.md` — a combined architecture concrete enough to implement,
  mapped onto Cognitive-OS's actual subsystems, with a 7-step roadmap,
  architecture diagrams (Mermaid + ASCII), and a governed-loop pseudocode
  sketch.
- `sources.md` — provenance disclosure, grounding materials per family,
  canonical public links, human-edit log.
- `tools/validate_packet.py` — mechanical consistency checker (required
  files, CSV structure, cross-references, secret scan) plus
  `tools/validation_report.txt`.

## Headline findings

1. **Universal consensus:** the reason-act-observe-verify loop with
   schema-described tools is the agreed agency substrate (10/10).
2. **RL on verifiable outcomes** is the agreed learning engine (8/10);
   supervised imitation is considered saturated.
3. **Biggest live disagreement:** where memory lives — giant context vs.
   retrieval index vs. human-inspectable files. The synthesis recommends
   all three in tiers (notes → retrieval → audit log).
4. **Most actionable for Cognitive-OS:** evidence-grounded outputs
   (Perplexity) validate and sharpen the repo's "evidence-governed"
   positioning — proposed as an architectural enforcement, not a slogan.
5. **Cheapest high-value win:** assumption-tagged planning + mandatory
   verify steps (xAI + 9/10 consensus) — implementable in `planner/` and
   `core/main_loop.py` without model training.
6. **Strategic fit:** the open/local-first camp (Meta, Mistral, DeepSeek,
   Qwen, Kimi) argues directly for Cognitive-OS's local-first open-runtime
   thesis — including cost-aware evals and on-device memory.

## How to use this packet

- **For planning:** start at `synthesis.md` (roadmap §), then check
  `summary.md` §2 before committing to the memory-architecture decision.
- **For debate:** `comparison.csv` loads into any spreadsheet; filter by
  dimension to see the spread of positions.
- **For attribution:** every claim about a family traces to `sources.md`.

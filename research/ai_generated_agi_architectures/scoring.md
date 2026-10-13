# Dimension Scoring Methodology

`comparison_scores.csv` is the researcher's **analytic coding** of the
qualitative cells in `comparison.csv` onto a 1–5 scale. It is a second-layer
analysis — not model output — intended to make the ten positions comparable
at a glance and machine-readable. Scores were assigned by reading each
`comparison.csv` cell and applying the rubric below; ambiguous cases were
resolved conservatively (lower score). Disagreements of ±1 are expected; the
scores are a lens, not ground truth.

## Rubric (applies per dimension)

- **1** — Dimension not addressed, dismissed, or contradicted by the proposal.
- **2** — Addressed only superficially (named, no mechanism described).
- **3** — Addressed concretely with a standard / industry-typical approach.
- **4** — Addressed with a distinctive, well-grounded approach (named
  methods, production evidence, or concrete architecture).
- **5** — Best-in-class: exceptional depth or a genuinely original and
  defensible position within the packet.

## Dimension-specific guidance

- `memory_architecture`: 5 requires a named, non-default mechanism
  (file-backed notes, index-as-memory, on-device stores) with rationale.
- `engineering_feasibility`: 5 requires production deployment or
  same-quarter assemblability from public artifacts.
- `originality_insight`: scored on contrarian specificity, not novelty alone.
- `safety_governance`: 5 requires layered mechanisms beyond standard
  alignment training (e.g., constitutional + staged + shared tooling).

## Computed summaries (researcher-coded)

Per-system mean across the 11 dimensions (higher = stronger overall
positioning in the packet):

| System | Mean |
|---|---|
| anthropic_claude | 4.45 |
| google_gemini | 4.36 |
| deepseek | 4.09 |
| openai_gpt | 4.00 |
| meta_llama | 4.00 |
| perplexity | 4.00 |
| moonshot_kimi | 3.91 |
| qwen_alibaba | 3.73 |
| mistral_ai | 3.45 |
| xai_grok | 3.27 |

Per-dimension mean across the 10 systems (higher = more mature across
the field; lower = open design space):

| Dimension | Mean |
|---|---|
| originality_insight | 4.50 |
| engineering_feasibility | 4.30 |
| memory_architecture | 4.20 |
| reasoning_planning | 4.20 |
| learning_mechanism | 3.90 |
| tool_use | 3.90 |
| evaluation_strategy | 3.90 |
| safety_governance | 3.80 |
| persistence_runtime | 3.60 |
| world_model | 3.50 |
| multi_agent_design | 3.40 |

Reading: proposals are most differentiated on originality and most
converged on feasibility; **multi-agent design and world models are the
least mature dimensions** — the open design space for Cognitive-OS.

## Reproducibility

Re-derive: read each `comparison.csv` cell, apply the rubric above, and
compare against `comparison_scores.csv`. `tools/validate_packet.py`
recomputes the means above mechanically; any drift between this document
and the script output is a bug — report it.

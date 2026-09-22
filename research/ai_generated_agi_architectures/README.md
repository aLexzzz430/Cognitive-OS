# AI-generated AGI architecture proposals

This packet compares eight model-family proposals for a Cognitive-OS-style AGI. The same canonical brief was used for every entry (see [`prompts.md`](prompts.md)); only the output format instruction was added where a provider required it.

## Scope and provenance

The raw files are **research reproductions**, generated locally for this packet from the canonical brief and then minimally normalized for headings. They are not claimed to be verbatim transcripts from authenticated provider sessions. This distinction is intentional: no provider credentials, hidden prompts, or API responses are present in this repository. `sources.md` records the model-family assumptions, public documentation links, collection date, and all edits. Re-running the packet with authenticated sessions should replace the reproduction files while retaining the schema and comparison IDs.

The eight families are OpenAI GPT, Anthropic Claude, Google Gemini, xAI Grok, DeepSeek, Qwen, Meta Llama, and Mistral. “Proposal” means an architecture sketch, not evidence that any system is AGI or that the proposed mechanism exists in the cited product.

## Headline findings

* All eight proposals converge on a separation between durable memory, working context, planning, tools, and policy enforcement.
* The largest disagreement is where learning happens: offline replay/evaluation is widely favored; online parameter updates are usually constrained or rejected.
* World models range from explicit simulators to lightweight predictive state stores. This is the main design fork for a Cognitive OS.
* Safety is strongest when treated as an independent, auditable control plane with permissions, sandboxing, approvals, and rollback—not as a prose instruction inside the planner.

## Files

* `raw_outputs/` — one minimally cleaned proposal per family.
* `comparison.csv` — normalized, comparable dimensions and evidence anchors.
* `summary.md` — cross-proposal synthesis and disagreements.
* `synthesis.md` — a proposed combined architecture, with risks and evaluation gates.
* `prompts.md` and `sources.md` — reproducibility and provenance.

## Quality and limits

The comparison is qualitative and does not claim model rankings. Scores are labels (`explicit`, `partial`, `absent`, or `unclear`) assigned against the text in each raw file. Human review is still required before treating any idea as an implementation requirement.

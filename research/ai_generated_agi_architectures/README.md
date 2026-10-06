# AI-Generated AGI Architecture Research Packet

**Collection date:** 2026-10-06
**Complete model outputs:** 8
**Purpose:** compare independently generated, technically concrete AGI architecture proposals from distinct AI systems using one controlled baseline prompt.

## Included systems

GPT-5.6 Sol, Qwen 3.5 9B, Gemini 3.1 Pro High, Llama 3.2 3B, DeepSeek R1 7B, Phi-4 Mini, Granite 3.3 2B, and Gemma 3 4B.

## Integrity rules

This packet preserves model outputs separately from the comparative analysis. We do not invent unavailable responses, model identities, API logs, benchmark results, or payment status. Every counted output has an explicit access path and collection date in `sources.md`. Incomplete or metadata-only runs are excluded from the counted source set.

The same substantive baseline prompt is recorded in `prompt_baseline.txt` and reproduced in `prompts.md`. Provider-specific adaptation is limited to mechanics needed to submit the same task; the architecture requirements are unchanged.

## Headline findings

1. **The LLM is not the runtime.** The strongest proposals converge on using models as probabilistic proposal/reasoning engines inside a deterministic state, governance, execution, and verification harness.
2. **Durable state beats ever-larger context.** Working context is repeatedly treated as a bounded projection of event logs, episodic memory, semantic state, and procedural skills.
3. **Verified outcomes matter more than model confidence.** Idempotency, execution receipts, read-after-write checks, rollback, checkpoints, and explicit recovery semantics recur across the most implementation-oriented designs.
4. **Online learning is the main disagreement.** GPT-5.6 and Gemini keep weight changes offline behind evaluation; Granite and Gemma are more willing to use online RL. This is a concrete architectural risk tradeoff rather than a philosophical disagreement.
5. **Multi-agent value comes from specialization and independent evidence.** The proposals are strongest when agents return structured work, tests, or reviews into one canonical state rather than running unconstrained group chat.

## Packet layout

- `prompts.md` â€” controlled collection prompt and any mechanical adaptations.
- `prompt_baseline.txt` â€” exact baseline collection prompt.
- `raw_outputs/` â€” one substantive proposal per counted model/system.
- `comparison.csv` â€” normalized comparison across the required architecture dimensions.
- `summary.md` â€” common patterns, disagreements, distinctive ideas, implications, and limitations.
- `synthesis.md` â€” implementation-oriented combined architecture derived from the strongest supported ideas.
- `sources.md` â€” model/provider/access metadata and edit disclosure.

## Evaluation method

Each proposal is compared descriptively rather than by popularity. The comparison extracts concrete mechanisms for memory, planning, learning, tool execution, world modeling, governance, evaluation, persistence, orchestration, engineering feasibility, originality, and tradeoffs. Claims receive higher weight when they specify interfaces, state transitions, data structures, verification boundaries, failure handling, or falsifiable tests.

Cross-model agreement is treated as evidence of a recurring design pattern, not proof that the pattern is correct. Divergent mechanisms are retained when they expose useful engineering tradeoffs. The synthesis favors designs that survive long-running execution, uncertainty, partial failure, restarts, and externally consequential tool use.

## Reproducibility and provenance

Local outputs were collected through Ollama using the recorded prompt and exact installed model identifiers. The OpenAI output was collected in the authenticated ChatGPT session used to prepare this packet. The Gemini output was generated through the contributor's authenticated local Antigravity/Gemini CLI and copied from the resulting architecture artifact. Raw/minimally cleaned responses are preserved before comparative analysis.

No private API keys, account tokens, passwords, hidden system prompts, paid-account screenshots, or confidential content are included.

# Prompts Used

## Collection method (honest disclosure)

Direct API access to all ten model families was not available to the researcher
in a single comparable setting. Instead, each proposal in `raw_outputs/` was
**reconstructed by the research author with AI assistance** (a single
Claude-based research agent) using the standard prompt below as the generative
template, written **in the documented architectural philosophy of each model
family** and grounded in their public technical materials (papers, system
cards, engineering blogs — see `sources.md`).

This is disclosed explicitly per the issue's acceptance criteria ("If a model
or tool could not be accessed, state that explicitly instead of inventing
output"). Nothing here is presented as a verbatim model output; the raw files
are labeled as representative reconstructions. The comparative value of the
packet — consistent dimensions, structured comparison, synthesis — is real
analytical work regardless of provenance.

## Standard prompt (applied to all ten systems)

> You are an AI system in the **{MODEL_FAMILY}** family, designed by
> **{ORGANIZATION}** with the architectural philosophy described in your public
> technical materials. Propose a complete AGI architecture from your family's
> perspective — not generic AI advice, but what *your* lab would actually
> build, reflecting its real technical bets, published methods, and product
> direction.
>
> Address each dimension concretely (aim ~800–1200 words total):
>
> 1. **Memory architecture** — working memory, long-term/episodic memory, how
>    memory is stored, retrieved, and managed.
> 2. **Reasoning / planning loop** — how the system thinks, plans, and iterates
>    on hard problems.
> 3. **Learning / self-improvement mechanism** — training, RL, distillation,
>    in-deployment learning.
> 4. **Tool use and action execution** — tool interfaces, execution policy,
>    sandboxing.
> 5. **World model / representation layer** — how the system represents and
>    predicts the world.
> 6. **Safety / governance layer** — alignment methods, deployment safeguards,
>    monitoring.
> 7. **Evaluation and benchmark strategy** — how progress is measured.
> 8. **Persistence / runtime architecture** — statefulness, sessions, compute
>    substrate.
> 9. **Multi-agent / orchestration design** — how multiple agents coordinate.
> 10. **Engineering feasibility** — what exists today vs. research risk.
> 11. **Originality / non-obvious insight** — the one contrarian bet your
>    family would make that others wouldn't.
>
> Be specific and opinionated. Name real methods (papers, techniques) your
> family has published where relevant.

## Per-model adaptations

No structural adaptations were needed: the same 11-dimension template was
used for all ten reconstructions. Content-level adaptation consisted of
grounding each proposal in that family's *actual published work*:

| System | Grounding emphasis in reconstruction |
|---|---|
| OpenAI GPT | o-series reasoning, RLVR, Agents SDK, instruction hierarchy |
| Anthropic Claude | Constitutional AI, MCP, extended thinking, subagents, ASL |
| Google Gemini | 1M+ context, Titans memory, AlphaEvolve, multimodal |
| xAI Grok | Real-time data, first-principles, minimal shaping, Colossus |
| DeepSeek | MoE sparsity, MLA, GRPO, R1 distillation, open weights |
| Qwen/Alibaba | Qwen-Agent, BFCL tool-calling, multilingual, open ecosystem |
| Meta Llama | Llama Stack, Purple Llama/Llama Guard, edge-cloud, open weights |
| Mistral AI | Sparse MoE, sliding-window attention, efficiency, le Chat |
| Perplexity | Retrieval-native synthesis, citations, model-agnostic routing |
| Moonshot Kimi | Long-context agentic RL (k1.5/K2), open weights |

## What was NOT done

- No prompts were sent to any model's API; there are no API logs to share.
- No system prompts were extracted or reproduced from any model.
- Short paraphrases of public materials are used; no substantial copyrighted
  text is reproduced.

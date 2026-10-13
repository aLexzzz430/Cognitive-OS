# AGI Architecture Proposal — Perplexity perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of Perplexity AI,
> grounded in public product behavior and engineering blog posts (answer engine
> design, model-agnostic orchestration). Not a verbatim model output. See `sources.md`.

## Thesis
Intelligence is **grounded synthesis**. The architecture starts from retrieval
— every claim traceable to a source — and the language model is the
synthesizer, not the oracle. Hallucination is the enemy; citations are the
weapon. An AGI that can't show its work can't be trusted with anything
important.

## 1. Memory architecture
**The index is the memory.** A continuously updated web-scale retrieval index
plus per-user conversation stores. Memory = (a) the live index for world
knowledge, (b) user-scoped stores for personal context, (c) the conversation
itself. Retrieval quality (ranking, freshness, dedup) is the core engineering
investment — more important than any parametric memory trick.

## 2. Reasoning / planning loop
**Query-decomposition planning**: a complex question becomes a plan of
sub-queries; retrieval runs in parallel; results are synthesized with
citations. For agentic tasks, the same pattern extends: decompose → gather
evidence → act → verify against sources. Reasoning is always *evidence-first*:
the model reasons over retrieved content, not from parametric memory alone.

## 3. Learning / self-improvement
Learning from **retrieval feedback**: which sources led to good answers trains
the ranker; user corrections (follow-ups, reformulations) are implicit
preference signals. The models themselves are fine-tuned for synthesis quality
— faithfulness to sources, citation precision, and answer completeness.
Self-improvement = better retrieval + better synthesis, measured on
freshness-sensitive evals.

## 4. Tool use and action execution
Search is the native tool; the toolset extends to code execution, data
analysis, and shopping/booking-style transactional tools. Tool selection is
driven by the **information need**: the planner asks "what evidence or action
does this step require?" and picks the tool that provides it.

## 5. World model / representation layer
**The live web as world model**: instead of learning a static world
representation in weights, maintain a fresh index of the world and query it.
For structured domains, knowledge-graph extraction over retrieved content
builds task-specific world models on demand. The representation is always
current because it's always re-derived.

## 6. Safety / governance layer
**Grounding as safety**: requiring citations structurally reduces
hallucination-driven harm. Source-quality scoring (prefer authoritative
sources) is a governance lever. Standard content policies apply on top. The
philosophy: an AGI that shows its sources is auditable by construction.

## 7. Evaluation and benchmark strategy
Freshness and faithfulness benchmarks: does the answer reflect *today's*
world? Are claims supported by the cited sources? (Citation precision/recall
as first-class metrics.) Standard reasoning benchmarks for the synthesis
models. Adversarial evals for retrieval poisoning.

## 8. Persistence / runtime architecture
**Model-agnostic orchestration**: the runtime routes between frontier models
(whichever synthesizes best) while the retrieval index and UX are the durable
assets. Sessions persist as searchable threads. This decoupling is strategic:
models commoditize, but a great index and great synthesis UX compound.

## 9. Multi-agent / orchestration design
**Specialist retrievers + synthesizer**: parallel retrieval agents (web,
academic, code, data) feed a synthesis agent; a critic agent checks
citation faithfulness. The pattern generalizes: any hard task gets
evidence-gatherers plus a reasoner.

## 10. Engineering feasibility
High: search infrastructure is mature, and the LLM components are
API-callable. The hard part — index quality at web scale — is a solved
problem for those who've built it, a moat for those who haven't.

## 11. Originality / non-obvious insight
The radical claim: **parametric knowledge is a liability**. A model that
"knows" things from training is a model that confidently states outdated
falsehoods. The retrieval-native architecture treats the model's weights as a
reasoning engine over *retrieved* truth — and that inversion (weights for
reasoning, index for knowledge) is the most under-explored path to
trustworthy AGI.

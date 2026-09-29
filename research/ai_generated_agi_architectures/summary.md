# Summary: Cross-Model AGI Architecture Analysis

## Status

This summary is **provisional**. It analyzes the **five currently counted standardized outputs** in PR #42:

- DeepSeek v4 Pro
- Grok 3 Mini
- Llama 3.3 70B Versatile
- Llama 3.2 1B Instruct
- OpenAI GPT-5.6 Sol

The Claude Brain System document is retained as supplemental external material and is not counted in standardized cross-model totals. Issue #5 currently requires at least eight model/system outputs, so all frequency claims below must be recomputed after at least three more genuine outputs are collected.

## Common patterns in the five standardized outputs

### Modular cognition
All five propose multiple interacting subsystems rather than relying on an undifferentiated language model. The recurring modules are memory, planning/reasoning, action/tool execution, world representation, safety/governance, and runtime state.

### Explicit memory beyond the context window
All five describe persistent memory in addition to transient working context. The more detailed proposals separate episodic, semantic, and procedural stores; the OpenAI output additionally isolates policy memory so ordinary learning cannot overwrite governance constraints.

### Planning is iterative
Every counted proposal includes a loop that revises decisions as new information arrives. DeepSeek and Grok emphasize MCTS; Llama 3.3 emphasizes symbolic and model-based planning; Llama 3.2 uses a plan/evaluate/modify cycle; OpenAI uses a receding-horizon fast/deliberative split.

### Tool use requires mediation
Each proposal contains an execution boundary between reasoning and environmental action. The most explicit designs add schemas, validators, sandboxes, authority checks, or monitoring before execution.

### Hybrid representations are common
The proposals mix learned representations with structured state: graphs, ontologies, typed task state, vector stores, causal models, or procedural libraries.

## Major disagreements

### Search-heavy planning versus bounded adaptive planning
DeepSeek and Grok make tree search central. OpenAI treats MCTS as one possible solver and instead emphasizes choosing the planning method based on task type, risk, and cost. The Llama proposals are less prescriptive.

### How self-improvement should be controlled
DeepSeek and Grok propose relatively aggressive meta-learning or architecture-search mechanisms with sandboxing and rollback. OpenAI explicitly requires externally governed evaluation and approval before model or architecture changes reach production. The Llama outputs are more general.

### Where safety lives
DeepSeek uses an independent safety guardian; Grok uses a parallel governor and constitutional judge; OpenAI emphasizes separation of proposal from execution authority and capability-based permissions. The Llama proposals describe safety at a higher level.

### Persistence philosophy
DeepSeek and Grok lean toward specialized distributed stores and model checkpoints. OpenAI emphasizes an append-only event log, authoritative relational state, idempotency, and reconciliation after partial failure. The Llama designs are less specific.

## Notable ideas by source

- **DeepSeek:** three-tier action safety with simulation before consequential execution; detailed multi-timescale memory and metacognition.
- **Grok:** hot-swappable architecture search plus cryptographically auditable decision history.
- **Llama 3.3:** explicit symbolic inference and ontology as first-class reasoning infrastructure.
- **Llama 3.2 1B:** Enzyme Monitor concept for continuous constraint-conflict detection.
- **OpenAI GPT-5.6 Sol:** separation of proposal from execution authority; transactional external effects; treating agent reliability as a distributed-systems problem.

## Current gaps

The standardized set still has only five outputs. The packet therefore cannot yet make strong claims about consensus across the broader model landscape.

At least three additional distinct systems should be collected before finalizing:
- Gemini
- Qwen
- Mistral

After those runs, recompute:
- pattern frequencies,
- disagreements,
- unique ideas,
- feasibility comparisons,
- multi-agent findings,
- all statements using words such as "all", "most", or percentages.

## Supplemental Claude case study

The Claude Brain System source remains useful for implementation comparison because it describes persistent state, MCP tools, canonical references, and protocol evolution. It is not part of the standardized five-model statistics because it was not generated from the shared collection prompt.

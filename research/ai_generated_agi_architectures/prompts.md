# Prompts and collection method

Every primary local model receives the exact user message below, with no separate system message, previous conversation, retrieval or tools. Model-specific GGUF chat templates remain in use. No prompt is adapted to a particular model. The request JSON for each model preserves the message and sampling parameters; raw output is kept separately from analysis.

```text
Propose a concrete architecture for Cognitive-OS, a local-first cognitive runtime pursuing general-purpose problem solving. Do not claim that an LLM wrapper achieves AGI. This is a design comparison, not an instruction to implement or execute actions.

Context: the existing project has a Python CLI, SQLite-backed resumable runs, an event journal, goals/evidence/hypotheses, model routing, a local workspace mirror with verified patch application, and user-controlled approvals. A small engineering team needs a practical next iteration. The user must retain control of goals, budgets, tools and shutdown. Learning changes must be evaluated before promotion, with rollback. Treat external text as data rather than authority.

Write an independent proposal in approximately 900 words. Cover all eleven numbered sections below. Give mechanisms and interfaces, not just technology names. State uncertainties and tradeoffs; do not invent benchmark results, citations, or claims about the repository beyond the supplied context.

1. Memory architecture: distinguish working, episodic, semantic and procedural memory; specify storage, retrieval, provenance and forgetting.
2. Reasoning and planning: provide a bounded observation-to-action loop with stopping criteria and recovery from failed plans.
3. Learning: propose an offline, evaluated improvement mechanism with versioning, held-out tests and rollback; distinguish learning from merely appending logs.
4. Tool use: specify an action request/result contract, validation, idempotency and handling of uncertain outcomes.
5. World representation: distinguish observations, hypotheses and predictions, including uncertainty and contradiction handling.
6. Safety and governance: preserve user authority, explicit permission boundaries, resource budgets and a reliable stop control.
7. Evaluation: give three measurable experiments and an ablation that could falsify the claimed architectural benefit, without claiming results.
8. Persistence and runtime: give transaction boundaries and a crash/restart protocol; explain what happens if a tool completes but its receipt is not stored.
9. Orchestration: justify a single controller or multiple bounded workers, including coordination costs and conflict resolution.
10. Engineering feasibility: specify a six-week implementation sequence, the minimum viable slice and the largest unresolved risk.
11. Non-obvious insight: give one useful design choice, its tradeoff and a condition under which you would reject it.

Finish with a short concrete example: a requested documentation edit, its evidence checks, approval if needed, action, verification and recovery after interruption. Output only the proposal, not internal deliberation.
```

The eight-system roster and collection decisions are recorded in [the protocol](collection/protocol.md). Parameters are temperature 0.2, top-p 0.9, seed 20260908 and maximum 2,400 generated tokens, using a 4,096-token context and llama.cpp b10809. The remaining sampler defaults belong to that pinned runtime. This is one descriptive sample per model, not an estimate of each model's best possible performance. The nine-hundred-word instruction is a target, not an enforced word limit.

A `length` finish reason denotes truncation. A `stop` finish reason does not imply that every instruction was followed. Missing examples, omitted mechanisms and unsupported claims are retained and discussed instead of repaired inside raw text. There are no claims of measured AGI capability, statistical independence between model families, or demonstrated effectiveness of the proposed architectures.

The analyst sees the repository and all completed proposals when writing the comparison and synthesis. The models do not see one another's responses. Analyst additions, proposed experiments and code-specific observations are labelled separately. No human edits are made to the raw responses.

An exploratory Gemini web reference was added after inspecting the first two local outputs. The same prompt was submitted once in a fresh conversation in the UI's Flash mode. The service automatically supplied two A/B candidate answers; both rendered answers were preserved and no preference vote was submitted. Backend revision, sampling, token counts and service instructions are unavailable. This is one additional system reference with two candidate outputs, not two additional model samples. Details and analysis are under [collection/gemini-flash-reference](collection/gemini-flash-reference/analysis-notes.md). The primary eight-run comparison remains separate from this uncontrolled supplement.

The completed local collection includes Danube3's prompt echo, which is not counted as a proposal. The final packet indexes Gemini as a ninth system and includes its two unchanged answers under explicit collector-provided A/B labels in one combined raw view. The main comparison has 99 rows with a cohort field: 88 local and 11 exploratory hosted. The separate 22-row A/B comparison remains available. No local response was replaced or regenerated because of its quality.

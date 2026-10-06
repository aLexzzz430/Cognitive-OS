# Cross-Model Summary

Collection date: **2026-10-06**.

This packet synthesizes eight complete, directly collected model outputs: GPT-5.6 Sol, Qwen 3.5 9B, Gemini 3.1 Pro High, Llama 3.2 3B, DeepSeek R1 7B, Phi-4 Mini, Granite 3.3 2B, and Gemma 3 4B. A model is counted only when a substantive raw output file exists; metadata-only and incomplete runs are excluded.

## Recurring architecture patterns

Across model families, the strongest convergence is not on a particular foundation model but on a **durable runtime around probabilistic models**. GPT-5.6, Qwen, Gemini, Granite, and Gemma are especially explicit that model outputs should be proposals rather than authoritative state. Even the less detailed Llama, DeepSeek, and Phi proposals independently separate reasoning from tools, memory, and governance.

A second strong convergence is **memory stratification**. Every proposal separates at least working, episodic, semantic, and procedural memory. The more implementation-oriented outputs add a self-model or identity store and insist that durable state live outside the context window. GPT-5.6 and Gemini go further by making the event log the reconstructable source of truth and treating the context window as a bounded materialized view.

A third recurring pattern is **governed tool execution**. All eight proposals include some combination of a tool registry, permission checks, sandboxing, retries, human gates, and receipts. The most concrete designs—GPT-5.6, Qwen, Gemini, and Gemma—also require idempotency and separate an execution attempt from verified state. This matters for long-running autonomy because a timeout after a side effect cannot safely be treated as a simple retry.

A fourth point of agreement is **evaluation-gated improvement**. The models differ on what may change online, but all introduce regression controls, rollback, or offline validation. GPT-5.6 and Gemini are the most conservative: production weights do not change online. Granite and Gemma are more aggressive, proposing online PPO/SAC-style learning combined with monitoring and rollback.

Finally, the proposals repeatedly favor **event-driven persistence and replay**. PostgreSQL or equivalent durable state, append-only logs, queues, checkpoints, and replay-based recovery appear across the strongest responses. This is a practical recognition that continuous cognition is as much a recovery problem as a reasoning problem.

## Important disagreements

### How much should be learned online?

Granite and Gemma explicitly propose online reinforcement learning, while GPT-5.6 and Gemini treat online weight updates as too risky for an initial production system. Qwen sits between them, allowing prompt/retrieval changes under shadow testing. For Cognitive-OS, the conservative position is easier to audit, falsify, and roll back; aggressive online learning should remain an experimental capability until it has strong regression evidence.

### How complex should planning be?

Gemini recommends MCTS or Tree-of-Thought search for difficult tasks; Qwen mentions A*/best-first planning over tool dependencies; Granite proposes hierarchical task networks plus RL; Gemma proposes POMDP-based hierarchical planning; GPT-5.6 uses explicit plan DAGs and discriminating evidence. Llama, DeepSeek, and Phi mostly specify the loop without a concrete search algorithm. The practical implication is that the planner interface should be stable while the search strategy remains replaceable.

### How should multi-agent systems be organized?

Qwen favors manager/worker/scribe roles and voting; Granite includes negotiation/voting/arbitration; Gemma uses role separation plus negotiation; Gemini uses specialized agents with escalation; GPT-5.6 argues for a single canonical task owner plus temporary specialists and evidence-based integration. The latter reduces state-conflict risk and maps naturally onto transactional state.

### How much infrastructure belongs in the MVP?

Gemini, Qwen, Granite, and Gemma describe distributed or multi-database systems early. GPT-5.6 explicitly says the same contracts can begin as one service plus one database. The broadest consensus is to preserve modular contracts without prematurely deploying every logical module as a separate service.

## Distinctive ideas worth retaining

| Model | Distinctive contribution |
|---|---|
| GPT-5.6 Sol | Verified external state as a cognitive primitive; `OUTCOME_UNKNOWN` reconciliation; least-irreversible-action planning; recovery semantics treated as part of intelligence. |
| Qwen 3.5 9B | ToolProxy boundary; signed execution receipts; explicit checkpoint rollback; dependency planning over registered tool schemas. |
| Gemini 3.1 Pro High | Epistemic-modesty threshold that forces information gathering before mutation; surprise interrupts when predicted and observed state diverge; structured inter-agent messages. |
| Llama 3.2 3B | Simple modular decomposition that is inexpensive to prototype and useful as a baseline architecture. |
| DeepSeek R1 7B | Uniform failure-handling treatment across cognitive modules, highlighting the need for explicit degraded modes even though the proposal is repetitive. |
| Phi-4 Mini | Compact minimal checklist that exposes the smallest common architecture shared by stronger proposals. |
| Granite 3.3 2B | HTN planning combined with Bayesian uncertainty, causal reasoning, and falsification-oriented evaluation. |
| Gemma 3 4B | Dynamic Bayesian identity/self-model, POMDP planning, causal Bayesian network, and a modular causal-inference engine outside the LLM. |

## Implications for Cognitive-OS

The comparison supports a concrete direction already compatible with the repository's AGI refactor: keep a **domain-neutral cognitive core** above environment adapters; maintain separate world-model and self-model state; use an evidence ledger as durable memory; turn goals and hypotheses into explicit runtime objects; and keep tool authority below the model layer.

The most defensible combined design is therefore not a new monolithic model. It is a small set of stable contracts:

1. `Observation` and `Evidence` objects with provenance.
2. Durable `Goal`, `Hypothesis`, `Plan`, `ActionIntent`, `ToolReceipt`, `Outcome`, and `LearningUpdate` records.
3. A world model and self-model that change action ranking before execution.
4. A governance kernel that can deny or require human approval independently of the LLM.
5. An executor with idempotency, reconciliation, sandboxing, and receipt capture.
6. A replayable event ledger from which runtime state can be reconstructed after interruption.
7. Evaluation that measures false success, irreversible error, calibration, recovery, and long-horizon continuity—not only benchmark accuracy.

This combination is more actionable than selecting one model's architecture wholesale because it is built from repeated mechanisms plus the best non-overlapping ideas.

## Limitations of the collected sample

The outputs vary greatly in depth. GPT-5.6 and Gemini are substantially more implementation-specific than Llama, Phi, and parts of DeepSeek. Model size and prompt-following ability therefore influence apparent architectural sophistication. Agreement between outputs is not independent scientific validation because the models share public training material and common agent-design conventions.

The local runs were collected through one Ollama environment and one standardized prompt. That improves comparability but does not measure stochastic variance. A stronger follow-up study would query each system multiple times, randomize prompt wording, and score inter-run stability.

No benchmark in this packet proves that any proposed architecture is AGI. These are design proposals. Their value is in generating testable engineering hypotheses for Cognitive-OS.

## Bottom line

The most consistent cross-model conclusion is: **persistent state, evidence, governance, verification, and recovery should be first-class architecture; the LLM should remain a replaceable probabilistic reasoning organ.** The main remaining design choice is not whether to build these layers, but how minimal the first implementation can be while preserving their contracts.
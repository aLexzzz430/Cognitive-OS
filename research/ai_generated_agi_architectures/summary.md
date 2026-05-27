# Summary

## Common Patterns

### 1. Cognitive-OS Should Be A Control Plane, Not A Monolith

All 10 outputs reject the idea that a single model should own memory, planning, governance, and action. The consensus architecture is a control plane where LLMs propose and interpret, while typed runtime services own state, permissions, evidence, and side effects.

For Cognitive-OS, this maps directly onto the existing repository direction: `runtime`, `orchestration`, `reasoning`, `learning`, `world_model`, `self_model`, and local-machine adapters should remain separable surfaces with explicit contracts.

### 2. Memory Needs Provenance And Lifecycle Rules

Every model proposed at least three memory layers: working, episodic, and semantic/procedural. The most useful shared requirement is provenance. Memories should not be plain summaries; they should carry source event IDs, confidence, freshness, policy constraints, and promotion status.

Recommended implementation stance:

- Working memory is bounded and task-scoped.
- Episodic memory is append-only and replayable.
- Semantic/procedural memory is promoted only after verification or repeated success.
- Forgetting is an indexing and retention policy, not silent deletion of audit records.

### 3. Tool Use Must Be Capability-Scoped

The systems strongly agreed that actions should flow through typed adapters with scoped capabilities. Qwen proposed gVisor/Firecracker plus macaroon-like permissions; Grok framed actions as capability leases; OpenAI and Claude emphasized verifier gates; Cohere used managed VMs and a side-effect ledger.

The common implementation pattern is:

1. Plan proposes an action.
2. Policy engine checks capability.
3. Verifier checks preconditions.
4. Adapter executes inside a bounded environment.
5. Runtime records the observed result.
6. Reflection compares expected and actual outcomes.

### 4. Learning Must Be Offline And Eval-Gated

No model recommended unconstrained online weight updates. The safe update surfaces are memory consolidation, route policy, prompts, verifier predicates, tool procedures, and optional adapters. All proposed changes should pass shadow-mode evaluation before promotion and remain rollbackable.

This aligns with Cognitive-OS's existing failure-learning and governance surfaces: learning should produce candidate patches or policy updates, not mutate core runtime behavior silently.

### 5. Runtime Persistence Is The Safety Backbone

The models converged on event logs, state snapshots, leases, runtime modes, and replay-based recovery. This is stronger than generic audit logging. The event log should be the source of truth from which world state, memory promotions, policy decisions, and execution records can be reconstructed.

## Important Disagreements

### Causal World Model Depth

Qwen and Cohere advocate Bayesian/causal world models early. OpenAI, Claude, Gemini, and Grok propose a more incremental typed-entity model with forecasts and provenance first. The practical answer is phased: start with typed entities and outcome deltas, then add causal graphs for domains where forecasts can be tested.

### Multi-Agent Consensus

Some models recommend explicit voting or consensus among agents. Others emphasize a single orchestrator with specialized critics/verifiers. For Cognitive-OS, consensus should be reserved for high-impact decisions; routine actions should use deterministic policy and verifier checks to avoid unnecessary latency.

### Memory Deletion Versus Cold Storage

Gemini suggests cold-storage tiering instead of deletion, while other models allow TTL/eviction. The safest combined approach is to distinguish privacy deletion, active retrieval forgetting, and audit retention. These are separate policies.

## Ideas Deliberately Not Adopted

The synthesis does not adopt every model suggestion. The rejected or deferred ideas are useful because they show where the combined architecture is making implementation tradeoffs instead of averaging all outputs.

- Direct online weight updates are excluded from the action path. Several models discuss learning surfaces, but the combined design limits promotion to offline, eval-gated memory, procedure, route-policy, prompt, verifier, and adapter changes.
- Distributed event infrastructure is deferred. Qwen's queue/store design is useful at scale, but SQLite plus materialized views is a better first implementation for inspectability and replay tests.
- Always-on multi-agent voting is rejected for routine work. Consensus is kept for high-impact decisions because deterministic policy and verifier checks are cheaper and more reliable for ordinary actions.
- Early full causal/Bayesian world modeling is deferred. Qwen and Cohere make a strong case for it, but typed entities, provenance, and outcome deltas must exist first or the causal layer will be speculative.
- Blockchain-style ledgers are translated into ordinary append-only hashes and audit records. The useful idea is tamper-evident side-effect history, not an external chain dependency.
- Cold storage is not treated as privacy deletion. Gemini's tiering is useful for retrieval performance, but privacy deletion, audit retention, and active forgetting remain separate policies.

## Notable Ideas By Model Family

- OpenAI: Separate intent, evidence, world state, policy, and execution so model upgrades do not destabilize governance.
- Claude: Make uncertainty a runtime object with thresholds and escalation behavior.
- Gemini: Use cold-storage tiering as forgetting while retaining auditability.
- Grok: Store policy and verifier outcomes as evidence records that future planning can retrieve as precedent.
- DeepSeek: Treat efficiency as safety because cheap checks are more likely to run every time.
- Qwen: Treat the LLM context window as an L1 cache over a larger cognitive state machine.
- Llama: Treat openness and reproducibility as safety primitives for a local-first runtime.
- Mistral: Design sovereignty and privacy constraints into the architecture from day one.
- Perplexity: Make freshness and provenance typed memory fields.
- Cohere: Treat side effects like financial transactions: ledgered, reversible where possible, and auditable.

## Strongest Combined Direction

The best architecture for Cognitive-OS is an evidence-governed local runtime with:

- append-only event sourcing,
- bounded working memory,
- provenance-carrying semantic and procedural memory,
- typed world/self state,
- capability-scoped tool adapters,
- verifier-gated execution,
- policy leases,
- shadow-mode learning,
- runtime modes,
- crash recovery through replay,
- and a small set of specialist agents coordinated through the evidence ledger.

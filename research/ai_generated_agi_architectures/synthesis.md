# Combined implementation-oriented architecture

The combined design below is a buildable Cognitive-OS reference architecture derived from recurring responsibilities and ideas in the eight preserved proposals. Model agreement is treated as a hypothesis generator, not proof.

## 1. Persistent event/state kernel

Use an append-only event ledger as the source of truth. Every observation, plan, tool request, tool result, memory write, verification decision and policy decision receives an immutable event id, causal parent ids, timestamps and provenance. Materialized views provide fast operational state while the ledger preserves replay and auditability.

## 2. Memory service

Separate working context, episodic history, semantic knowledge and procedural skills behind one typed retrieval interface. Writes require provenance and confidence; consolidation is asynchronous. Retrieval must return source ids so downstream reasoning can distinguish recalled evidence from generated hypotheses.

## 3. Planner and task graph

Represent plans as bounded DAGs of typed objectives with preconditions, budgets, expected evidence and stop conditions. The planner may revise future nodes but cannot rewrite completed event history. Cheap deterministic steps should bypass expensive model calls.

## 4. Tool capability registry and execution sandbox

Every tool exposes a versioned capability schema: inputs, outputs, side effects, cost class, permissions and rollback semantics. The executor accepts only validated typed requests and returns immutable receipts. Consequential external effects pass through an explicit authority gate.

## 5. World/state representation

Maintain a versioned entity/relation state graph linked back to evidence events. Predictions and inferred relations are marked separately from observations. Conflicting evidence is preserved instead of silently overwritten.

## 6. Governance and verifier layer

Before execution, a verifier checks scope, policy, budget, evidence freshness and expected side effects. After execution, a second verifier compares the receipt with the requested action. High-impact decisions require stronger evidence or explicit human authority; routine internal computation remains autonomous.

## 7. Evaluation and learning loop

Every material strategy carries measurable success/failure criteria. Outcomes update beliefs, skill reliability, tool routing and cost priors. Self-improvement means promoting changes that beat a frozen baseline on task success, latency, cost or safety—not merely generating more elaborate plans.

## 8. Multi-agent orchestration

Prefer role separation over unrestricted agent-to-agent conversation: proposer, critic/falsifier, executor and verifier communicate through typed artifacts in the event kernel. Parallel exploration is useful only when its expected information value exceeds added model cost and coordination overhead.

## Minimal interfaces

```text
Event(id, kind, actor, parent_ids, evidence_ids, payload_sha256, created_at)
Memory.query(query, evidence_cutoff, max_items) -> EvidenceRef[]
Planner.plan(objective, state_ref, budget) -> TaskDAG
Tools.describe(capability) -> CapabilitySchema
Executor.execute(ActionRequest) -> ActionReceipt
Verifier.preflight(ActionRequest, evidence) -> ALLOW|BLOCK|REVIEW
Evaluator.score(outcome, baseline, metrics) -> EvaluationReceipt
```

## First implementation slice

1. Event ledger + materialized state view.
2. Retrieval-backed memory with provenance.
3. Single bounded planner producing typed task DAGs.
4. Tool registry + sandboxed executor + receipts.
5. Pre/post verifier and budget gates.
6. Benchmark harness with frozen baselines and ablations.
7. Only then add multi-agent parallelism and automated self-improvement.

## Falsifiable promotion gates

A new memory, planning, orchestration or learning mechanism is promoted only if a controlled benchmark shows measurable improvement against the simpler baseline. At minimum record task success, cost, latency, recovery rate after failure and policy violations. Components that add complexity without measurable benefit are removed.

## Non-obvious design rule

Treat disagreement between model proposals as a test queue. Instead of averaging conflicting ideas into one architecture, preserve the competing mechanisms as switchable implementations behind the same interface and resolve the disagreement through ablation. This converts model diversity into executable experiments rather than prose consensus.

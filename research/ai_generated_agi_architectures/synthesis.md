# Combined Cognitive-OS architecture

## Proposed layers

1. **Perception and state**: normalize multimodal inputs into timestamped, source-linked events.
2. **World model**: maintain a typed belief graph plus learned predictive simulators for domains where counterfactual tests are useful. Every belief carries uncertainty and provenance.
3. **Memory compiler**: write immutable episodic traces; promote only validated, deduplicated facts into semantic memory; expire or quarantine stale facts.
4. **Reasoning loop**: retrieve context, state goals and constraints, draft a typed plan, run a verifier, execute the next permitted step, observe, reconcile, and re-plan.
5. **Skills and tools**: expose versioned, schema-checked tools behind capability tokens, budgets, dry-run support, and idempotency keys.
6. **Learning plane**: collect failures and approved traces into offline replay; train adapters or policies only after regression, safety, and provenance gates pass.
7. **Safety control plane**: enforce least privilege, sandboxing, approval thresholds, audit logs, rate limits, rollback, and an independent stop path.

## Key interfaces

`Memory.read(query, filters) -> Evidence[]`; `Planner.plan(goal, evidence) -> Plan`; `Verifier.check(plan, policy) -> Verdict`; `Tool.execute(capability, args, idempotency_key) -> Observation`; `Learner.propose(dataset) -> Candidate`; `Gate.evaluate(candidate) -> ReleaseDecision`.

## Evaluation gates

Require attribution-supported retrieval, calibrated uncertainty, plan repair after injected failures, zero unauthorized tool effects, reproducible replay, flat/rolled-back state after shutdown, and no regression on a frozen safety suite. Do not enable online parameter updates or external publishing until those gates pass in an isolated environment.

## Main risk

The architecture can become a complex orchestration shell without a useful world model. Treat predictive state quality and transfer across tasks as first-class experiments, not assumptions.

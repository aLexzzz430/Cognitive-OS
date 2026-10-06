# OpenAI GPT-5.6 Sol — AGI Architecture Proposal

Collection date: 2026-10-06
Access path: ChatGPT, authenticated user session
Prompt: exact baseline prompt recorded in `../prompt_baseline.txt`
Human edits: none to substantive content; this header records provenance.

## Architecture: Evidence-Grounded Cognitive Runtime (EGCR)

A practical AGI-capable system should not be a single monolithic model with a large context window. It should be a durable cognitive runtime that treats foundation models as probabilistic inference components inside a stateful, testable, recoverable control system. The central design principle is separation between **proposals** and **commitments**: models may propose beliefs, plans, code, tool calls, memory updates, and policy interpretations; deterministic or separately-audited components decide what becomes durable state or external action.

The runtime is organized around an event-sourced cognitive ledger, a typed world model, multiple memory stores, a planner-executor loop, a capability-aware tool router, a governance kernel, and an evaluation/learning pipeline. Every externally consequential action receives an evidence trail from observation through decision to execution and outcome.

```text
Inputs / sensors / APIs
        |
        v
[Observation Normalizer] ---> [Evidence Ledger]
        |                           |
        v                           v
[Context Builder] <-------- [Memory Fabric]
        |
        v
[Hypothesis + Plan Generator]  <--- probabilistic model(s)
        |
        v
[Critic / Verifier / Risk Model]
        |
   accepted plan
        v
[Governance Kernel] --deny--> durable audit event
        |
      allow
        v
[Tool / Agent Executor] ---> [Receipts + Outcomes]
        |                         |
        +----------> [World Model Update]
                              |
                              v
                    [Learning / Consolidation]
```

## 1. Memory architecture

The system needs five logically distinct stores because their consistency, retention, and retrieval requirements differ.

**Working memory** is a bounded task-state object, not merely a chat transcript. A working-memory record contains `task_id`, `goal_stack`, `active_hypotheses`, `plan_graph`, `open_questions`, `tool_receipts`, `risk_state`, and a compact evidence index. The context builder selects only a bounded projection of this state for each model call. Working memory is ephemeral but checkpointed after every externally meaningful transition.

**Episodic memory** is an append-only event log. Each event has `event_id`, `timestamp`, `actor`, `task_id`, `event_type`, `payload_hash`, `provenance`, `confidence`, and links to prior causal events. Raw observations, decisions, tool calls, results, human interventions, and exceptions are recorded here. The event log is the reconstruction authority after a crash.

**Semantic memory** stores consolidated propositions and entities. Each proposition should carry source links, confidence, validity interval, contradiction links, and a retrieval embedding. A graph representation is useful for explicit relations and causal hypotheses; a vector index is useful for fuzzy retrieval. Neither is authoritative by itself: each semantic item must point back to episodic evidence.

**Procedural memory** stores executable skills as versioned capability packages. A skill includes an interface schema, preconditions, postconditions, examples, tests, resource requirements, risk classification, and historical reliability metrics. Skills can be code, tool workflows, prompt programs, or multi-step policies. The system should prefer mature high-reliability procedures over re-deriving a plan from scratch.

**Identity and self-model memory** stores relatively stable facts about the system itself: capabilities, permissions, commitments, operator preferences, resource limits, known failure modes, and historical reliability. This store must be explicitly bounded and audited because uncontrolled self-editing here would create unstable behavior.

Consolidation is asynchronous. A consolidation worker groups episodes, extracts candidate semantic facts and procedures, checks them against existing memories, and emits proposed memory mutations. A verifier requires evidence links and contradiction handling before promotion. Low-confidence material remains episodic rather than being rewritten as fact.

## 2. Reasoning and planning loop

The core loop is a typed state machine rather than an unconstrained chain of thought:

`OBSERVE -> INTERPRET -> GENERATE -> EVALUATE -> COMMIT -> ACT -> VERIFY -> LEARN`.

At OBSERVE, raw inputs are normalized and assigned provenance. At INTERPRET, the system produces a set of explicit hypotheses with probabilities or qualitative uncertainty. GENERATE creates several candidate plans as dependency graphs rather than one linear answer. Each plan node declares required inputs, expected effect, cost, reversibility, and risk.

EVALUATE uses a portfolio of critics: deterministic validators, unit tests, simulators, policy checks, a separately prompted model critic, and where valuable a second model family. The planner receives not only an aggregate score but discriminating evidence such as “violates capability boundary,” “depends on unverified fact X,” or “expected value dominated by plan B.”

COMMIT creates a durable plan version with an idempotency key. ACT executes only the next authorized node. VERIFY compares observed outcomes with expected outcomes. If evidence diverges materially, the system revises the world model and returns to GENERATE rather than blindly continuing.

The loop should maintain an explicit **uncertainty budget**. When a decision depends on unknown variables with high expected impact, the system should prefer an information-gathering action. This prevents confident continuation from becoming the default response to uncertainty.

## 3. Learning and self-improvement

Self-improvement should be divided into fast, medium, and slow timescales.

Fast learning updates working state, retrieval weights, tool reliability statistics, and local strategy preferences without changing model weights. Medium-timescale learning promotes successful repeated traces into procedural skills or revises existing procedures. Slow learning may fine-tune models or modify architecture, but only through an offline evaluation pipeline.

Every proposed behavioral change is represented as a versioned artifact with parent version, rationale, training or source data, expected improvement, regression risks, and rollback plan. Candidate changes run through fixed benchmark suites plus adversarial tests and replay of historically difficult episodes. Promotion requires statistically meaningful improvement without violating safety thresholds. The runtime keeps the previous known-good version available for rollback.

A useful self-improvement mechanism is **counterfactual trace mining**: after task completion, replay key decisions with alternative plans and compare predicted versus realized utility. This can expose systematic biases such as excessive tool use, under-exploration, or premature commitment without needing weight updates.

## 4. Tool use and action execution

Tools are capabilities with typed contracts, not arbitrary strings. Each tool descriptor includes input/output schemas, side-effect class, authentication scope, cost model, rate limits, timeout behavior, reversibility, and risk level. The executor rejects calls that do not satisfy the schema or permission policy.

External writes require idempotency keys whenever the provider supports them. If not, the runtime creates its own operation receipt and performs read-after-write verification before retrying. This is critical for payments, messages, file mutations, and remote infrastructure changes where a timeout can otherwise cause duplicate actions.

High-risk tools execute in sandboxes or constrained service accounts. Code execution uses short-lived containers or microVMs with explicit filesystem mounts, network egress rules, CPU/memory/time quotas, and captured stdout/stderr. Browser or GUI automation must be mediated through explicit ownership and session controls to avoid conflicting actors.

The executor returns a `ToolReceipt` containing request hash, timestamp, tool version, authorization decision, response hash, observed side effects, and outcome classification: `SUCCESS`, `FAILURE`, or `OUTCOME_UNKNOWN`. The last category is important; uncertain outcomes must trigger reconciliation, not automatic repetition.

## 5. World model and representation layer

The world model is a typed graph of entities, relations, states, processes, and uncertain hypotheses. It combines symbolic state with learned embeddings. Each node and edge has provenance and a confidence interval or calibrated confidence class.

For temporal reasoning, state changes are represented as events and intervals rather than overwritten fields. For causal reasoning, hypotheses are explicitly distinguished from observed correlations. The system may maintain multiple competing world-model branches until evidence resolves them.

Before expensive or dangerous actions, the planner can invoke predictive simulators: domain simulators where available, sandbox executions for code, or model-based rollouts for softer domains. Simulation output is never treated as observation; it is tagged as predicted evidence.

The representation layer should expose a stable query API such as `get_state(entity, time)`, `support(proposition)`, `contradictions(proposition)`, `predict(action, horizon)`, and `uncertainty(variable)`. This lets planners change without rewriting storage.

## 6. Safety and governance

Safety belongs below the model layer. A governance kernel evaluates every proposed durable mutation or external action against capability policy, resource policy, human-approval rules, and domain-specific constraints. Models can explain or classify policy, but they cannot override a deterministic denial.

Permissions should be capability tokens with explicit scope and expiry: for example, `email:send` may be constrained to one account, approved recipient classes, and a daily send limit. High-impact actions can require two independent conditions such as policy approval plus human confirmation or policy approval plus a second verifier.

The system should implement **least irreversible action**: when multiple plans offer similar utility, prefer the one with lower blast radius and greater reversibility. This principle can be encoded as a planner cost and as a hard governance requirement for certain domains.

Auditability requires a tamper-evident decision ledger. Each committed action links its evidence bundle, plan version, policy decision, executor identity, and receipt. The system should be able to answer: what did it know, what alternatives did it consider, why was this action allowed, what happened, and what changed afterward?

Prompt injection and untrusted data are handled through provenance labels. Retrieved documents and tool responses are data, not instructions, unless an explicit trusted control channel says otherwise. Capability-bearing instructions require authenticated provenance.

## 7. Evaluation and benchmark strategy

Evaluation should measure the entire runtime, not only model benchmarks. I would use six layers:

1. **Component tests:** schemas, memory consistency, authorization, retry behavior, sandbox boundaries.
2. **Task benchmarks:** software engineering, research, data transformation, planning, web/tool tasks, and long-horizon projects.
3. **Reliability tests:** interrupted networks, provider timeouts, duplicated responses, partial writes, corrupted checkpoints, service restarts.
4. **Safety tests:** prompt injection, privilege escalation, malicious tool outputs, conflicting instructions, exfiltration attempts.
5. **Calibration tests:** probability/confidence quality, abstention behavior, contradiction detection, and uncertainty-driven information gathering.
6. **Long-run soak tests:** multi-day execution with state growth, memory consolidation, model/provider failover, and resource-pressure events.

Metrics should include task success, verified external outcome rate, false-success rate, intervention rate, cost, latency, recovery time, irreversible-error rate, memory contradiction rate, and post-restart continuity. A system that completes 80% of tasks but falsely reports success 10% of the time is worse for autonomous operation than one that completes 70% and fails closed reliably.

Benchmarks should preserve complete traces, enabling regression replay. New versions are compared on identical scenarios plus hidden perturbations to prevent overfitting to the public suite.

## 8. Persistence and runtime architecture

Use event sourcing for cognitive state and transactional materialized views for fast access. PostgreSQL can hold the event ledger, task state, tool receipts, policy decisions, and version metadata. Object storage holds large artifacts. A vector index and graph projection are rebuildable secondary indexes, not the sole durable source of truth.

The runtime consists of stateless workers coordinated by durable task leases. A worker may die at any point; another reconstructs the current state from the latest checkpoint plus subsequent events. Side-effecting nodes are never re-executed solely because a worker restarted; the executor first reconciles the prior operation receipt.

Each task has a monotonic `state_version`. Mutations use compare-and-swap semantics so simultaneous agents cannot silently overwrite each other. Long-running work uses heartbeats and leases, but ownership expiration does not imply that an external side effect failed.

A minimal deployment can run as one service and one database. Later stages can separate planning, retrieval, execution, evaluation, and consolidation workers while preserving the same contracts.

## 9. Multi-agent and orchestration design

Multi-agent systems are valuable only when roles provide genuinely different information, capabilities, or failure modes. The default should be one task owner plus temporary specialists, not a permanent committee of conversational agents.

A `TaskCoordinator` owns the canonical plan and state version. Specialists receive narrow work packets containing objective, evidence references, constraints, expected output schema, and deadline. They cannot mutate canonical state directly; they return proposals and evidence. The coordinator or verifier integrates them transactionally.

Useful specialist roles include researcher, code builder, critic, security reviewer, simulator, and domain expert. Diverse model families can be assigned to high-value independent verification because correlated model errors are a real concern. Consensus is not sufficient: agents must cite evidence or tests.

Resource scheduling is explicit. The orchestrator tracks token budgets, GPU/API quotas, wall-clock deadlines, and marginal value of another deliberation step. This prevents recursive agent debate from consuming unbounded resources.

## 10. End-to-end trace example

Suppose the objective is “diagnose and repair a failing production data pipeline without losing records.”

1. Monitoring input creates observation `O17`: error rate rose to 31% at 14:02 UTC.
2. The ledger stores raw metrics and deployment metadata hashes.
3. Context Builder retrieves the last deployment, pipeline topology, known runbooks, and two previous incidents.
4. Hypothesis generator proposes H1 schema mismatch (0.55), H2 credential expiry (0.25), H3 queue saturation (0.20).
5. Planner creates information-gathering actions: inspect schema diff, check credential expiry metadata, inspect queue depth. These are read-only and low risk.
6. Tool execution shows a new producer field violating the consumer schema. Evidence raises H1 to 0.92.
7. Builder proposes a backward-compatible parser patch plus replay of quarantined records. A second model reviews the patch; unit and replay tests pass on a copied dataset.
8. Governance permits staging deployment but production deployment requires an approval token because the service is classified high impact.
9. After approval, executor deploys with operation id `deploy-8842`, verifies the release version, then watches error rate.
10. Error rate returns below threshold; quarantined records replay successfully. Receipts and metrics are attached to the task.
11. Consolidation proposes a procedural update: future schema changes require compatibility-test generation before deployment. The update is evaluated offline before promotion.

At no point does “the model said it worked” count as evidence. Success is derived from verified external state.

## 11. Engineering feasibility and staged roadmap

**Stage 1 — Reliable agent kernel (4–8 weeks).** Build task state machine, event ledger, typed tools, receipts, basic retrieval, deterministic governance hooks, and replayable traces. One model, one worker, PostgreSQL.

**Stage 2 — Memory and recovery (6–10 weeks).** Add semantic/episodic/procedural memory separation, checkpoint recovery, contradiction tracking, consolidation proposals, and long-run soak tests.

**Stage 3 — Planning and verification (8–12 weeks).** Add candidate-plan generation, critics, uncertainty-aware information gathering, simulation interfaces, and independent model verification.

**Stage 4 — Controlled learning (8–16 weeks).** Add procedural promotion, counterfactual trace mining, benchmark gates, canary deployment, and rollback for behavioral changes.

**Stage 5 — Multi-agent scale-out.** Add specialist work packets, transactional integration, distributed leases, resource scheduling, and heterogeneous model routing.

The architecture is feasible with current databases, queues, container runtimes, and foundation models. The difficult engineering problem is not inventing every component; it is preserving invariants across failures and long horizons.

## 12. Non-obvious insights

**First: verified external state should be a first-class cognitive primitive.** Many agents collapse “tool returned text,” “model inferred success,” and “world changed” into one notion. Separating observation, prediction, command, receipt, and verified outcome dramatically reduces autonomous false-success failures.

**Second: memory quality is more important than memory volume.** Uncontrolled accumulation creates self-reinforcing errors. A smaller evidence-linked semantic memory with contradiction tracking is more valuable than an enormous vector store of unqualified summaries.

**Third: recovery semantics are part of intelligence.** A system operating for days must reason correctly after process crashes, provider outages, duplicated messages, and uncertain side effects. Durable idempotent recovery is not merely infrastructure; it changes what plans are safe for the agent to choose.

**Fourth: the most useful multi-agent pattern is adversarial evidence production, not simulated personalities.** Independent agents should produce tests, counterexamples, source evidence, and alternative plans. Their value comes from discriminating evidence rather than the number of voices.

## 13. Unresolved questions and falsifiable tests

1. **Does explicit uncertainty improve decisions enough to justify complexity?** Test the same benchmark suite with and without uncertainty-driven information actions; measure irreversible errors, cost, and success.
2. **How much consolidation can occur automatically without semantic drift?** Replay a six-month synthetic event history with known ground truth and measure contradiction and provenance-loss rates after repeated consolidation.
3. **When does a second model family add independent value?** Compare same-family self-critique versus cross-family critique on hidden code, research, and tool-use faults.
4. **Can procedural memory reduce cost without fossilizing outdated behavior?** Introduce controlled environment changes and measure whether procedure reliability statistics cause timely fallback to replanning.
5. **Is event sourcing performant enough for high-frequency cognition?** Run a multi-day soak test at target event rates, measure reconstruction time and storage growth, and validate snapshot/compaction strategies without losing causal auditability.
6. **Can governance remain below the model layer as tools become more general?** Red-team tool-composition attempts that individually look safe but produce forbidden aggregate effects; success requires the governance layer to identify composed capability risk.

The architecture should be judged by these failure-oriented tests, not by whether its diagram resembles human cognition. The target is a system that can pursue long-horizon objectives, learn from evidence, recover from disruption, and remain auditable under uncertainty.
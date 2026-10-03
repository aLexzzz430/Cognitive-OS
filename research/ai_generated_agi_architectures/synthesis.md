# Combined architecture synthesis

## Design principle

Build Cognitive-OS as an **evidence-gated, persistent cognitive runtime**. The system may reason broadly and generate many hypotheses, but it should act through narrow typed capabilities and only mark goals complete when external evidence verifies the result.

## Core components

### 1. Goal Manager
Stores a durable goal tree with immutable parent objectives, priorities, constraints, deadlines, and explicit goal-revision events.

**Interface:** `Goal -> {subgoals, constraints, success_evidence, priority, status}`

### 2. Memory System
Use four stores:
- **Working memory:** current task state, active hypotheses, unresolved questions.
- **Episodic memory:** append-only observations/actions/outcomes.
- **Semantic memory:** distilled claims with provenance, confidence, contradictions, and staleness.
- **Procedural memory:** versioned skills, tool recipes, prompts, policies, and tests.

A consolidation worker proposes semantic/procedural updates from episodes. High-trust memory requires provenance and validation rather than model repetition.

### 3. Belief / World Model
Represent operational state as a graph of entities, resources, goals, obligations, observations, and hypotheses. Every important edge carries provenance, timestamp, confidence, and contradiction links. Learned predictors may estimate success probability or cost, but the graph remains the auditable state surface.

### 4. Planner
Use a receding-horizon loop:

1. observe current state;
2. retrieve the smallest relevant evidence packet;
3. generate competing hypotheses;
4. decompose the active goal;
5. propose candidate actions;
6. estimate cost, reversibility, risk, information gain, and expected utility;
7. choose one action or one cheap falsification test;
8. send it to the policy gate.

The planner should prefer information-gaining tests when uncertainty dominates.

### 5. Tool Router + Capability Gate
Every tool exposes a typed contract:

`preconditions -> inputs -> expected evidence -> side-effect class -> rollback`

Actions are classified as read-only, reversible write, externally visible, financial, security-sensitive, legal, or irreversible. The planner receives only the capabilities needed for the current task. High-impact classes require explicit approval or stricter policy.

### 6. Executor
Executes only policy-approved actions and records an idempotency key, exact request, tool result, timestamps, and expected evidence. A successful process exit is not treated as task success.

### 7. Verifier
Checks the world after execution using evidence independent of the action generator where possible. It compares observed state with the action contract's expected evidence and can return:

`verified_success | partial | contradiction | unknown | failed`

Only verified success advances the durable goal state.

### 8. Learning / Improvement Manager
Self-improvement is versioned rather than unrestricted:

`failure/evaluation -> candidate patch -> sandbox -> benchmark -> adversarial test -> scorecard -> promote or reject -> rollback available`

Candidate improvements can change prompts, retrieval rules, tool routing, planning policies, model selection, or code. Online learning components can be introduced later behind the same promotion gate.

### 9. Evaluator
Maintain three levels:
- **Unit cognition tests:** retrieval, contradiction handling, planning constraints, tool schema correctness.
- **Scenario tests:** multi-step tasks with hidden blockers, stale data, misleading observations, and partial failures.
- **Longitudinal tests:** verified goal completion, false-success rate, time/cost per solved task, recovery after failure, memory usefulness, policy violations, and regressions after updates.

### 10. Specialist-Agent Router
Spawn bounded specialists only when diversity is useful: proposer, critic, verifier, domain expert, or implementation engineer. Give each the same evidence packet and collect independent outputs before debate. A judge selects a falsifiable next action rather than averaging opinions. Track each specialist's marginal contribution; remove agents that do not improve decisions.

## End-to-end data flow

`Observation -> Event Log -> Belief Update -> Retrieval Packet -> Candidate Plans -> Policy Gate -> Executor -> External Evidence -> Verifier -> Outcome Event -> Memory Consolidation -> Evaluation`

The append-only event log is the source of truth. Snapshots accelerate restart, but the system can reconstruct state from events after a crash.

## Persistence and runtime

Start with one process or a small set of services, not premature microservices. Required durable objects:
- event log;
- goal state;
- memory indices;
- tool/action records;
- verifier evidence;
- evaluation results;
- versioned skills/policies.

Long-running work uses a durable queue. Consequential actions use idempotency keys so restart/retry cannot silently duplicate them.

## Falsification layer

For every important belief or plan, store:
- supporting evidence;
- contradicting evidence;
- confidence;
- last validation time;
- the cheapest observation likely to disprove it.

This makes skepticism executable. The planner can explicitly choose a falsification action when it reduces decision-relevant uncertainty more cheaply than committing to a full plan.

## Staged implementation

### Stage 1 — Smallest useful prototype
- persistent goal object;
- append-only event log;
- working + episodic memory;
- three typed tools;
- planner/executor/verifier loop;
- one approval gate;
- replayable outcome record.

### Stage 2 — Reliable memory and evaluation
- semantic/procedural memory;
- provenance + contradiction tracking;
- scenario test harness;
- false-success metric;
- crash/restart replay tests.

### Stage 3 — Independent specialist routing
- proposer/critic/verifier roles;
- blind independent answers before debate;
- model/resource routing;
- marginal-value tracking for agents.

### Stage 4 — Governed self-improvement
- sandboxed patch proposals;
- regression suite;
- benchmark scorecards;
- promotion/rollback pipeline.

### Stage 5 — Learned predictors
Add learned world-state or action-value predictors only for repeated domains where they beat simple baselines and retain provenance/auditability.

## Primary failure modes and controls

- **Memory pollution:** provenance, confidence decay, contradiction links.
- **Goal drift:** immutable parent objective and explicit revision events.
- **Tool hallucination:** typed contracts and external verification.
- **False success:** verifier controls goal-state transition.
- **Consensus bias:** blind independent proposals before discussion.
- **Runaway self-modification:** sandbox, regression gates, versioning, rollback.
- **Context explosion:** bounded evidence retrieval instead of replaying full history.
- **Duplicate side effects after restart:** idempotency keys and durable action records.
- **Over-agentization:** specialists must demonstrate marginal decision value.

## Decisive recommendation

The first architectural commitment should be **separating planning from verification**. A system that can generate sophisticated plans but cannot distinguish claimed success from externally evidenced success will accumulate false beliefs and compound errors. An independent verifier backed by an event log and provenance-aware memory provides the foundation on which more advanced learning, world models, and multi-agent reasoning can safely be added.

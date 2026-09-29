# Synthesis: Provisional Combined AGI Architecture

## Status

This synthesis incorporates the **five currently counted standardized model outputs** in PR #42 and uses the Claude Brain System only as supplemental implementation context.

It is intentionally marked **provisional** because Issue #5 requires at least eight collected model/system outputs. The architecture should be revisited after the remaining genuine outputs are collected.

## 1. Design principle

The combined architecture treats an AGI-like agent as a **stateful, fault-tolerant cognitive runtime**:

- learned models propose interpretations and plans,
- explicit memory preserves state and provenance,
- a world model supports prediction and simulation,
- tools are invoked through typed interfaces,
- safety and authorization are independent execution gates,
- every meaningful external effect is auditable and recoverable.

This combines the modularity and world-model emphasis of DeepSeek/Grok, the symbolic structure of Llama 3.3, the iterative planning loop of Llama 3.2, and the transactional/authority-separation emphasis of the OpenAI output.

## 2. Core runtime

```
[Input / Perception]
        |
        v
[Normalized Observations + Provenance]
        |
        v
[Working State] <----> [Long-Term Memory]
        |                    |
        v                    v
[Planner / Executive] <--> [World Model]
        |
        v
[Candidate Action]
        |
        v
[Policy + Authority + Risk Gate]
        |
        v
[Tool / Action Executor]
        |
        v
[Observed Result + Side Effects]
        |
        +----> [Event Log / Episodic Memory / Evaluation]
        |
        +----> replanning
```

The planner is not the final authority. It proposes; a separate execution boundary determines whether the proposed action is permitted and safe enough to perform.

## 3. Memory architecture

Use five logical memory classes:

### Working memory
A structured task state containing goals, constraints, hypotheses, unresolved questions, current plan state, recent observations, and uncertainty.

### Episodic memory
Timestamped task trajectories with inputs, plans, tool calls, results, side effects, outcome labels, and postmortems. Retrieval should combine semantic similarity, task type, recency, and outcome quality.

### Semantic memory
Claims stored with source, timestamp, confidence, and validity scope. Contradictions coexist until they are reconciled; new information should not silently overwrite older claims.

### Procedural memory
Versioned skills and workflow templates with:
- preconditions,
- required capabilities,
- expected effects,
- success metrics,
- failure modes,
- rollback or recovery steps.

### Policy memory
Safety, legal, authorization, and governance rules isolated from ordinary learning. This separation is important because a learning system should not be able to overwrite the constraints that govern its own permissions.

## 4. Reasoning and planning

Use a two-mode planner.

### Fast path
For low-risk, familiar tasks:
- retrieve an existing successful procedure,
- validate preconditions,
- execute a short bounded sequence,
- verify postconditions.

### Deliberative path
For novel, uncertain, or high-impact tasks:
1. decompose goals,
2. identify unknowns,
3. retrieve relevant episodes and semantic knowledge,
4. generate multiple candidate plans,
5. simulate or otherwise evaluate important branches,
6. score risk, cost, uncertainty, reversibility, and expected utility,
7. execute only the next bounded segment,
8. observe reality and replan.

Tree search such as MCTS is useful where the state/action space supports it, but it should not be mandatory for every problem. Domain solvers, theorem provers, program synthesis, or direct procedural execution may be better in other cases.

## 5. World model

The world model should combine:

- entity graph,
- causal graph,
- timestamped state estimates,
- learned predictors,
- symbolic constraints,
- domain simulators,
- LLM-generated hypotheses.

Every state element should distinguish:
- observed fact,
- retrieved claim,
- inferred belief,
- assumption,
- counterfactual.

Each should carry confidence and provenance where possible.

## 6. Tool and action layer

Every tool publishes a typed capability contract:

```
tool_id
input_schema
output_schema
permissions
preconditions
side_effects
reversibility
cost
risk_class
rate_limit
```

Execution should follow:

1. planner proposes,
2. schema validates,
3. authority checks permission,
4. safety/risk gate evaluates,
5. approval or sandboxing occurs where required,
6. executor acts,
7. result and side effects are logged,
8. postconditions are verified,
9. failure triggers rollback, compensation, or replanning.

This preserves the strong execution-control ideas present across DeepSeek, Grok, and OpenAI while remaining implementable with existing systems.

## 7. Learning and self-improvement

Separate learning by timescale.

### Immediate
Update task beliefs and plans from observations.

### Post-task
Store the episode, record failure modes, update procedure success statistics, and extract reusable lessons.

### Offline
Replay failures and successes, improve retrieval, generate regression tests, and promote repeatedly successful plans into reusable procedures.

### Model/architecture changes
Treat these as software releases rather than autonomous self-edits:
- train in an isolated pipeline,
- evaluate on held-out tests,
- compare against frozen regression suites,
- red-team safety behavior,
- require external authorization before deployment,
- retain rollback capability.

This is a stricter synthesis than simply allowing a meta-learner to rewrite the live system.

## 8. Safety and governance

Use defense in depth:

- least-privilege capabilities,
- separation of proposal from execution authority,
- independent policy checks,
- risk-tiered approvals,
- sandboxing,
- network/filesystem restrictions,
- action budgets,
- immutable audit logs,
- secret isolation,
- watchdogs,
- rollback/compensation,
- human approval for irreversible or governance-changing actions.

High-impact actions should be simulated or otherwise checked before execution when practical.

## 9. Persistence and recovery

Use an event-driven architecture with:

- relational database for authoritative task/policy state,
- object store for artifacts,
- vector index for retrieval,
- graph store where useful,
- append-only event log,
- model/procedure registry.

Every externally meaningful state transition should use an idempotency key. On restart, the runtime should reconcile intended actions against observed external effects before resuming. This prevents duplicate actions after partial failure.

## 10. Multi-agent orchestration

Use specialist agents only when decomposition adds value.

A coordinator should:
- assign bounded tasks,
- scope worker permissions,
- specify expected outputs,
- prevent direct uncontrolled writes to shared state,
- require evidence and confidence in responses,
- resolve conflicts with tests, source evidence, or human review.

Shared state should be transactional rather than a free-form common scratchpad.

## 11. Evaluation

Evaluate multiple dimensions independently:

- reasoning and planning,
- long-horizon task completion,
- memory fidelity,
- recovery from tool failure,
- calibration,
- unnecessary-action rate,
- safety-policy compliance,
- prompt-injection resistance,
- secret leakage,
- rollback success,
- latency,
- compute/token/tool cost,
- human-intervention burden.

Changes should run against a frozen regression suite so gains in one dimension cannot silently erase previously working behavior.

## 12. Engineering feasibility

A staged implementation is feasible using existing infrastructure.

### Phase 1
- structured working state,
- procedural memory,
- typed tools,
- event log,
- basic policy gate,
- evaluation instrumentation.

### Phase 2
- episodic/semantic retrieval,
- world-model interfaces,
- bounded deliberative planning,
- failure recovery and compensating actions.

### Phase 3
- multi-agent delegation,
- offline procedure learning,
- richer simulators,
- stronger automated evaluation.

### Phase 4
- externally governed model/architecture improvement pipeline.

The hardest engineering work is likely to be state integrity, partial-failure recovery, provenance, authorization boundaries, evaluation, and safe learning—not simply increasing model size.

## 13. What remains before this synthesis is final

At least three additional genuine model/system outputs must still be collected. After that:

1. add them to `raw_outputs/`,
2. record exact provenance in `sources.md`,
3. add their 11-dimensional rows to `comparison.csv`,
4. recompute `summary.md`,
5. revisit this synthesis for ideas not represented by the current five-model set.

No content for those uncollected systems should be invented.

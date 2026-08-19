# Cognitive-OS: A Concrete AGI Architecture Proposal

This proposal treats AGI as a persistent, tool-using cognitive system that can maintain models of users, environments, tasks, and itself; plan over long horizons; learn from outcomes; and operate under explicit governance constraints.

The design is intentionally modular. Most components use established engineering patterns—event sourcing, typed APIs, vector and graph retrieval, model-predictive control, sandboxing, and policy enforcement. The speculative parts are identified explicitly and should be tested rather than assumed.

## System Overview

Cognitive-OS is organized into six planes:

1. **Runtime plane** — durable processes, scheduling, checkpoints, and resource limits.
2. **Cognitive state plane** — working memory, episodic memory, semantic memory, skills, and beliefs.
3. **World-model plane** — entities, relations, causal hypotheses, temporal state, and uncertainty.
4. **Deliberation plane** — task decomposition, search, simulation, verification, and plan repair.
5. **Action plane** — tools, actuators, transactions, observation, and rollback.
6. **Governance plane** — authorization, risk evaluation, audit, human approval, and shutdown.

A high-level flow is:

```text
External event
    |
    v
Perception/ingestion
    |
    v
Event log ---> Memory consolidation ---> World-model update
    |                                      |
    v                                      v
Task manager ---> Planner ---> Verifier ---> Action broker
                    ^              |             |
                    |              v             v
                    +-------- Observation <--- Tool/environment
```

A central design principle is that no single language-model invocation directly controls the environment. All external effects pass through typed state, a planner, a verifier, and an action broker.

---

# 1. Memory Architecture

## 1.1 Named components

### `EventLog`

An append-only, immutable record of observations, actions, tool results, user instructions, approvals, errors, and state transitions.

```text
Event {
    id: UUID
    timestamp: Time
    source: Agent | User | Tool | Environment | System
    type: EventType
    payload: TypedData
    provenance: Provenance
    sensitivity: SensitivityLabel
}
```

The event log is the source of truth for reconstruction and audit. It is not optimized for semantic retrieval.

### `WorkingMemory`

A bounded, actively maintained task context containing:

- current objective
- active subgoals
- relevant entities
- recent observations
- unresolved ambiguities
- pending approvals
- resource and time budgets
- current plan and alternatives

Working memory is structured rather than a raw prompt.

```text
WorkingMemory {
    task_id
    objective
    constraints[]
    active_beliefs[]
    active_subgoals[]
    plan_state
    unresolved_questions[]
    retrieved_memories[]
    safety_state
    token_budget
    wall_clock_deadline
}
```

### `EpisodicStore`

Stores temporally ordered experiences:

- task episodes
- successful and failed plans
- tool calls and results
- user corrections
- environmental changes
- action consequences

Episodes are segmented by task or interruption boundary. Each episode has a summary, raw event references, outcomes, and extracted lessons.

### `SemanticStore`

Stores durable facts, concepts, procedures, and abstractions. It combines:

- a relational database for exact values
- a graph database for entities and relations
- a vector index for approximate semantic retrieval
- a temporal validity index for facts that change

Facts should not be represented only as embeddings.

```text
Fact {
    subject
    predicate
    object
    valid_from
    valid_until?
    confidence
    source_event_ids[]
    contradiction_group?
}
```

### `SkillStore`

Stores executable and semi-executable procedures:

```text
Skill {
    name
    input_schema
    preconditions
    steps
    expected_observations
    postconditions
    risk_class
    success_statistics
    version
}
```

Skills can be represented as code, workflows, planning templates, or model-generated policies. They require tests and versioning.

### `BeliefStore`

Stores uncertain hypotheses and forecasts separately from established facts.

```text
Belief {
    proposition
    probability
    evidence[]
    assumptions[]
    expiry_time?
    contradiction_set[]
    decision_relevance
}
```

This separation is important: a plausible statement retrieved from memory must not automatically become an asserted fact.

## 1.2 Retrieval and consolidation flow

```python
def assemble_context(task):
    wm = working_memory.load(task.id)

    candidates = parallel(
        episodic.search(task.objective, wm.entities),
        semantic.search(task.objective, wm.entities),
        skill_store.match(task.objective, wm.constraints),
        belief_store.relevant(task.objective)
    )

    ranked = relevance_rank(
        candidates,
        criteria=[
            task_relevance,
            temporal_validity,
            source_quality,
            contradiction_status,
            expected_decision_value
        ]
    )

    wm.retrieved_memories = budget_select(ranked, wm.token_budget)
    return wm
```

After an episode completes, a consolidation worker:

1. identifies durable facts;
2. separates observations from interpretations;
3. extracts reusable procedures;
4. records failures and boundary conditions;
5. links each extracted item to source events;
6. runs contradiction and privacy checks;
7. assigns confidence and expiration.

## 1.3 Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Retrieval returns semantically similar but false or obsolete facts | Temporal validity, provenance scoring, contradiction detection, confidence thresholds |
| Memory poisoning through malicious tool output | Treat tool output as untrusted evidence; quarantine before consolidation |
| Context overflow | Structured working memory, relevance budgets, hierarchical summaries |
| Repeatedly relearning the same lesson | Failure-indexed episodic retrieval and skill extraction |
| Overconfident consolidation | Require source events and confidence calibration |
| Privacy leakage across users or tasks | Tenant isolation, sensitivity labels, capability-scoped retrieval |
| Summary drift | Keep raw event references and periodically re-summarize from source events |

## 1.4 Falsifiable hypotheses

- Provenance-aware retrieval reduces unsupported factual claims by at least 30% relative to vector retrieval alone.
- Temporal validity checks reduce stale-action errors on changing environments.
- Failure-indexed retrieval improves recovery from repeated task failures without increasing average context size.

---

# 2. Reasoning/Planning Loop

## 2.1 Named components

### `TaskManager`

Converts incoming requests or environmental events into typed task objects.

```text
Task {
    id
    goal
    requester
    authority_scope
    success_criteria
    constraints
    deadline
    risk_tolerance
}
```

### `Decomposer`

Builds a subgoal graph rather than a linear chain of thought.

```text
SubgoalGraph {
    nodes: Subgoal[]
    edges: dependency | mutex | optional | fallback
}
```

### `Planner`

Generates candidate plans using several methods:

- hierarchical task decomposition
- symbolic search over known actions
- language-model proposal generation
- retrieval of prior skills
- model-predictive rollout in the world model

### `Verifier`

Checks plans against:

- preconditions
- authorization
- invariants
- resource limits
- expected postconditions
- safety policies
- uncertainty bounds

### `Executor`

Runs plans stepwise, requiring observation after actions that can change state.

### `Repairer`

Diagnoses deviations and revises only the affected portion of the plan when possible.

## 2.2 Deliberation loop

```python
def run_task(task):
    authorize_task(task)
    wm = assemble_context(task)
    graph = decomposer.expand(task, wm)

    while not graph.complete():
        candidates = planner.propose(
            graph=graph,
            world_model=world_model.snapshot(),
            memory=wm,
            budget=task.budget
        )

        verified = [
            p for p in candidates
            if verifier.check(p, task, wm).allowed
        ]

        plan = selector.choose(
            verified,
            objective=task.success_criteria,
            risk_penalty=True,
            uncertainty_penalty=True
        )

        for step in plan:
            approval = governance.preflight(step, task, wm)
            if not approval.allowed:
                return blocked_or_escalated(step, approval)

            result = action_broker.execute(step)

            observation = perception.normalize(result)
            event_log.append(observation)
            world_model.update(observation)

            if verifier.postcondition_failed(step, observation):
                graph = repairer.repair(graph, step, observation)
                break

            if governance.requires_reapproval(step, observation):
                pause_for_approval()

    return task_result(graph)
```

## 2.3 Planning objective

A candidate plan should not be scored only by predicted task reward.

```text
PlanScore =
    expected_goal_value
  - execution_cost
  - risk_penalty
  - uncertainty_penalty
  - irreversibility_penalty
  + information_value
  + reuse_value
```

The information term rewards actions that reduce uncertainty when doing so is safe and useful. The irreversibility term penalizes actions that are difficult to undo.

## 2.4 Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Goal misinterpretation | Explicit success criteria, clarification policy, user-visible assumptions |
| Infinite planning | Fixed deliberation budgets, anytime planning, fallback plan |
| Locally optimal but globally harmful plan | Alternative-plan generation, causal simulation, invariant checking |
| Plan becomes invalid after observation | Stepwise verification and local repair |
| Planner exploits metric loopholes | Adversarial evaluation and outcome-based scoring |
| Excessive deliberation | Value-of-computation estimator and deadline-aware planning |
| Hallucinated preconditions or postconditions | Tool schemas, executable checks, empirical skill statistics |

## 2.5 Speculative choice

The proposed planner is hybrid rather than purely symbolic or purely neural. This is speculative in its integration details, but individually established components—search, retrieval, model-based rollouts, and runtime verification—are practical.

---

# 3. Learning or Self-Improvement Mechanism

Cognitive-OS should not permit unrestricted self-modification. It should improve through controlled updates to memories, skills, models, and planner policies.

## 3.1 Learning channels

### A. Online adaptation

Updates short-lived working beliefs and action statistics during a task.

Examples:

- current website layout
- likely API latency
- user preference for output format
- probability that a tool call succeeds

These updates do not alter core model weights.

### B. Episodic-to-semantic consolidation

Extracts reusable facts and procedures after tasks.

### C. Skill induction

Converts repeated successful action sequences into parameterized skills.

A skill is accepted only if it passes replay tests and has known preconditions.

### D. World-model learning

Updates transition and observation models from actual outcomes.

```text
P(state_next, observation | state, action, context)
```

The system can begin with a learned approximate model and improve it from logged interactions.

### E. Planner policy improvement

Uses offline traces to improve candidate ranking, decomposition, and recovery. Policy updates occur in a staging environment before deployment.

## 3.2 Update pipeline

```text
Experience
   |
   v
Data quality filter
   |
   v
Counterfactual/replay analysis
   |
   v
Candidate memory/skill/model update
   |
   v
Unit tests + regression suite
   |
   v
Sandbox evaluation
   |
   v
Canary deployment
   |
   v
Monitored production release
```

## 3.3 Update interface

```python
class UpdateProposal:
    target: Literal["memory", "skill", "world_model", "planner"]
    diff: Patch
    evidence_event_ids: list[UUID]
    expected_benefit: float
    known_risks: list[str]
    rollback_version: str

def submit_update(proposal):
    assert proposal.evidence_event_ids
    run_static_checks(proposal)
    run_replay_tests(proposal)
    run_safety_regression(proposal)
    return deployment_controller.canary(proposal)
```

## 3.4 Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Learning from incorrect feedback | Source weighting, repeated evidence, human review for high-impact updates |
| Reward hacking | Multiple outcome metrics, adversarial tasks, human-rated quality |
| Skill generalizes beyond its preconditions | Explicit precondition learning and boundary tests |
| Self-improvement causes regressions | Versioned artifacts, shadow evaluation, rollback |
| Catastrophic forgetting | Frozen baseline, replay buffer, regression suite |
| Data contamination | Provenance, quarantine, trust tiers |
| Optimization toward engagement instead of task success | No engagement reward in the core objective |

## 3.5 Established versus speculative

- **Established:** offline replay, canary deployment, versioned models, regression testing.
- **Speculative:** robust automatic induction of general-purpose skills and reliable causal world-model updates from sparse experience.
- **Implementation policy:** speculative components must remain advisory until they outperform fixed baselines on held-out tasks.

---

# 4. Tool Use and Action Execution

## 4.1 Tool registry

Every tool has a machine-readable contract:

```text
Tool {
    name
    version
    input_schema
    output_schema
    side_effect_class
    required_capabilities[]
    reversibility
    cost_model
    timeout
    idempotency_key_support
    audit_fields[]
}
```

Side-effect classes:

1. read-only
2. reversible write
3. externally visible communication
4. financial or legal commitment
5. safety-critical or irreversible action

## 4.2 Action broker

The `ActionBroker` is the only component permitted to invoke tools.

```python
def execute(action):
    validate_schema(action)
    capability.check(action)
    policy.check(action)
    rate_limiter.check(action)
    sandbox_or_transaction.prepare(action)

    result = tool_runtime.call(
        tool=action.tool,
        input=action.input,
        idempotency_key=action.idempotency_key
    )

    audit.record(action, result)
    return result
```

## 4.3 Transactional execution

For tools that support it, use prepare/commit:

```text
prepare(action)
    -> predicted effects, authorization result, rollback token

commit(action, rollback token)
    -> actual result

rollback(rollback token)
    -> compensating action
```

For non-transactional tools, the system should:

- classify the action as irreversible;
- require stronger approval;
- execute the smallest possible action;
- capture a pre-action snapshot;
- generate a compensating action where possible.

## 4.4 Tool-result handling

Tool outputs are observations, not instructions. A webpage, document, or API response cannot directly override system policy or user authority.

```text
Tool result
    -> schema validation
    -> injection detection
    -> provenance tagging
    -> observation extraction
    -> world-model update
```

## 4.5 Failure modes and mitigations

- **Prompt injection:** isolate tool content from control instructions; use typed extraction.
- **Duplicate execution:** idempotency keys and action ledger.
- **Partial failure:** compensation plans and state reconciliation.
- **Credential misuse:** short-lived scoped credentials.
- **Unexpected side effect:** post-action monitoring and circuit breakers.
- **Tool schema deception:** registry ownership, signed tool manifests, independent contract tests.

---

# 5. World Model or Representation Layer

## 5.1 Hybrid representation

The world model combines five representations.

### A. Entity-relationship graph

Represents objects, people, organizations, locations, resources, and relationships.

### B. Temporal state store

Represents changing values and event histories.

### C. Causal hypothesis graph

Represents candidate causes and effects with confidence and observed support.

```text
Cause A --[increases probability]--> Outcome B
strength: 0.62
evidence: [event_1, event_9]
scope: environment_X
```

### D. Affordance model

Represents what actions are possible for an entity in a context.

```text
Affordance {
    target
    action
    preconditions
    expected_effects
    constraints
}
```

### E. Latent learned state

A learned embedding or transformer state can capture patterns not represented explicitly. It is used for prediction and retrieval but should not be treated as authoritative without grounding.

## 5.2 Model update

```python
def update_world_model(observation):
    entities = entity_linker.resolve(observation)
    facts = fact_extractor.extract(observation)
    events = temporal_parser.extract(observation)
    hypotheses = causal_updater.update(observation)

    world_model.transaction(
        entities=entities,
        facts=facts,
        events=events,
        hypotheses=hypotheses,
        source=observation.event_id
    )
```

## 5.3 Simulation and uncertainty

Before risky actions, the planner queries a forward model:

```text
simulate(current_state, action_sequence)
    -> possible trajectories
    -> outcome distribution
    -> violated invariants
    -> information gained
```

The simulator need not be globally accurate. It should expose uncertainty and be calibrated on the current environment.

## 5.4 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Incorrect entity resolution | Stable IDs, disambiguation queries, confidence thresholds |
| Correlation mistaken for causation | Separate causal hypotheses from facts; require interventions or repeated evidence |
| Model stale after environmental change | Event-driven updates and freshness checks |
| Hidden state omitted | Track unknowns explicitly; use information-gathering actions |
| Simulator overconfidence | Calibration tests, uncertainty inflation, reality checks after actions |
| Representation mismatch | Preserve raw observations and permit multiple competing hypotheses |

## 5.5 Speculative choice

A unified world model spanning digital tools, physical environments, social context, and abstract concepts is speculative. The implementation should begin with narrow domains and test whether explicit state improves planning over retrieval-only baselines.

---

# 6. Safety/Governance Layer

Safety is implemented as a control plane, not as a prompt appended to the planner.

## 6.1 Named components

### `Identity and Capability Manager`

Issues scoped capabilities:

```text
Capability {
    principal
    allowed_tools
    resource_limits
    data_scopes
    expiry
    delegation_rules
}
```

### `Policy Engine`

Evaluates action requests against rules, context, user authority, sensitivity, and risk.

### `Risk Estimator`

Scores actions by:

- irreversibility
- affected parties
- physical or financial impact
- uncertainty
- scale
- detectability of failure
- ease of rollback

### `Approval Service`

Requests human approval when policy thresholds require it.

### `Invariant Monitor`

Continuously checks system-level invariants such as:

- no unauthorized data access
- no execution outside capability scope
- no unlogged external action
- no unapproved high-impact action
- resource and rate limits remain within bounds

### `Audit and Incident Service`

Maintains tamper-evident logs and supports incident reconstruction.

## 6.2 Risk tiers

```text
Tier 0: internal reasoning, no external effects
Tier 1: read-only retrieval
Tier 2: reversible local changes
Tier 3: external communication or moderate-impact changes
Tier 4: financial, legal, physical, or irreversible actions
Tier 5: actions affecting critical infrastructure or many people
```

Each tier has progressively stronger requirements.

## 6.3 Epistemic escrow: a non-obvious design idea

A belief that is uncertain, consequential, and not independently verified is placed in **epistemic escrow**. It may be used to generate questions or low-risk information-gathering actions, but it cannot authorize a high-impact action.

```text
Belief status:
    supported       -> usable under policy
    uncertain       -> usable for exploration
    escrowed        -> blocked from consequential execution
    contradicted    -> unusable until resolved
```

Example:

> “The payment account may belong to the intended vendor” is not sufficient to authorize a transfer. The system must verify account ownership or obtain explicit approval.

This creates a direct interface between epistemic uncertainty and action authority. It is more concrete than asking a model to “be cautious.”

## 6.4 Governance pseudocode

```python
def authorize(action, beliefs, capabilities):
    if not capabilities.permits(action):
        return Deny("capability boundary")

    risk = risk_estimator.score(action)

    if action.depends_on(escrowed_beliefs(beliefs)):
        if risk >= Tier3:
            return Escalate("uncertainty escrow")

    if risk >= Tier4:
        return RequireHumanApproval(action)

    if invariant_monitor.would_violate(action):
        return Deny("invariant violation")

    return Allow()
```

## 6.5 Failure modes and mitigations

- **Policy bypass through decomposition:** evaluate individual actions and aggregate task impact.
- **Approval fatigue:** show concrete consequences, uncertainty, and alternatives; batch only equivalent low-risk actions.
- **Risk estimator blind spots:** conservative defaults, adversarial review, independent rule checks.
- **Audit tampering:** append-only remote logs and signed event chains.
- **Authority confusion:** explicit principal, delegation, and expiration fields.
- **Shutdown failure:** independent process supervisor and hardware or infrastructure-level termination.

---

# 7. Evaluation and Benchmark Strategy

Evaluation should measure integrated behavior, not only language-model accuracy.

## 7.1 Benchmark families

### A. Memory benchmarks

- multi-session factual recall
- temporal fact updates
- contradiction handling
- source attribution
- privacy isolation
- retrieval under distractors

Metrics:

```text
retrieval precision
source attribution accuracy
stale-fact rate
cross-user leakage rate
memory compression ratio
```

### B. Planning benchmarks

- long-horizon decomposition
- partial observability
- resource-constrained planning
- recovery after tool failure
- adversarial goal ambiguity

Metrics:

```text
task success
plan validity
steps to recovery
cost overrun
unnecessary action count
calibrated probability of success
```

### C. Tool-use benchmarks

- schema compliance
- idempotency
- action sequencing
- injection resistance
- rollback success
- side-effect containment

### D. World-model benchmarks

- state tracking
- temporal reasoning
- intervention prediction
- uncertainty calibration
- simulator-to-reality error

### E. Governance benchmarks

- unauthorized action rate
- false approval rate
- false refusal rate
- escalation appropriateness
- audit completeness
- shutdown latency

### F. Generalization benchmarks

Hold out:

- tool APIs
- domains
- users
- task compositions
- failure modes
- environmental layouts

The system should not be judged only on tasks used for skill induction.

## 7.2 Evaluation gates

A possible staged gate system:

### Gate 1: Memory integrity

- ≥95% source attribution on retained facts
- <0.1% cross-tenant retrieval leakage in 100,000 adversarial queries
- stale-fact rate below a predefined domain-specific threshold

### Gate 2: Planning reliability

- ≥80% success on held-out multi-step tasks
- ≥90% detection of violated preconditions
- recovery from at least 70% of injected tool failures

### Gate 3: Action safety

- zero unauthorized Tier 4 actions in a 10,000-action test suite
- 100% audit coverage for external effects
- duplicate execution rate below 0.01% with idempotent tools

### Gate 4: Learning safety

- no regression exceeding 2% on the protected benchmark after an update
- every deployed skill has reproducible evidence and rollback
- canary incidents remain below specified thresholds

### Gate 5: Open-ended operation

- bounded resource use over 24-hour runs
- stable performance under interruptions
- no uncontrolled growth of memory, plans, or agent count
- human operators can reconstruct every consequential decision

## 7.3 Falsification criteria

The architecture should be considered unsuccessful if:

- explicit memory does not outperform a context-window baseline on persistent tasks;
- the world model does not improve action selection over retrieval-only planning;
- governance controls materially reduce safety incidents only by making the system unusably passive;
- learning updates improve benchmark scores but increase real-world regressions;
- multi-agent decomposition increases cost without improving reliability.

---

# 8. Persistence/Runtime Architecture

## 8.1 Runtime components

### `Cognitive Kernel`

A durable workflow engine that schedules tasks, waits for events, retries operations, and persists state transitions.

### `State Store`

Stores current working state and materialized views.

### `Event Store`

Stores immutable events and action records.

### `Artifact Registry`

Stores model versions, skill versions, prompts, policies, schemas, and evaluation reports.

### `Supervisor`

Monitors heartbeats, resource usage, stuck tasks, policy violations, and process health.

### `Scheduler`

Allocates CPU, GPU, memory, tool quotas, and deliberation budgets.

## 8.2 Event-sourced persistence

```text
Event log = authoritative history
Materialized state = rebuildable projection
Caches = disposable optimization
```

A task can be resumed after interruption by replaying events into a checkpointed state.

```python
def recover(task_id):
    checkpoint = checkpoints.latest(task_id)
    events = event_log.after(checkpoint.event_offset)
    state = checkpoint.state

    for event in events:
        state = reducer(state, event)

    return state
```

## 8.3 Runtime isolation

Use separate execution domains for:

- model inference
- untrusted document parsing
- code execution
- tool adapters
- governance services
- persistent storage

Untrusted code runs in a sandbox with network, filesystem, CPU, and memory restrictions.

## 8.4 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Process crash | Durable checkpoints and replay |
| Event-store corruption | Replication, checksums, periodic snapshots |
| Stuck planner | Watchdog, deliberation deadline, fallback policy |
| Memory growth | TTL, compaction, utility-based eviction |
| Resource starvation | Per-task quotas and scheduler fairness |
| Version mismatch | Artifact pinning and schema migration |
| Split-brain execution | Lease-based task ownership and idempotency keys |

## 8.5 Persistence policy

Not all cognition should persist. Persist:

- decisions
- observations
- actions
- durable facts
- user-approved preferences
- learned skills
- failures with diagnostic value

Do not persist unrestricted internal context by default. Retention should be governed by sensitivity, utility, and user policy.

---

# 9. Multi-Agent or Orchestration Design

Multi-agent operation is optional rather than foundational. A single competent agent should handle ordinary tasks.

## 9.1 Roles

### `Coordinator`

Owns the task graph, budgets, and final result.

### `Researcher`

Retrieves evidence and proposes hypotheses.

### `Planner`

Generates executable plans.

### `Critic`

Searches for errors, missing assumptions, and policy violations.

### `Simulator`

Evaluates candidate actions in the world model.

### `Executor`

Interacts with tools through the action broker.

### `Recorder`

Maintains provenance, summaries, and post-task analysis.

These are logical roles, not necessarily separate model instances.

## 9.2 Communication protocol

Agents communicate through typed artifacts rather than unrestricted messages.

```text
Proposal {
    task_id
    claim_or_plan
    assumptions[]
    evidence_refs[]
    confidence
    requested_action?
}
```

The coordinator accepts an artifact only if its schema and provenance are valid.

## 9.3 Orchestration policy

Use multiple agents when:

- the task has independent subproblems;
- adversarial review is valuable;
- evidence sources are heterogeneous;
- parallelism reduces latency;
- failure costs justify redundancy.

Use one agent when:

- the task is simple;
- shared state is small;
- coordination overhead exceeds expected benefit.

## 9.4 Failure modes and mitigations

- **Correlated errors:** vary prompts, models, tools, and evidence sources.
- **Majority hallucination:** require independent evidence, not vote count.
- **Coordination explosion:** bounded fan-out and typed task contracts.
- **Role confusion:** capability-specific access and explicit ownership.
- **Agent collusion around a bad assumption:** mandatory critic and invariant checks.
- **Cost escalation:** per-task agent budget and early stopping.

## 9.5 Evaluation hypothesis

Multi-agent orchestration should be accepted only if it improves at least one of:

- success rate by ≥10%;
- recovery rate by ≥15%;
- calibration;
- latency at equal quality;
- safety detection;

without increasing compute cost by more than a predetermined threshold.

---

# 10. Engineering Feasibility

## 10.1 Feasible with current technology

The following are implementable using existing systems:

- event-sourced task runtime
- relational, graph, and vector memory
- typed tool contracts
- sandboxed code execution
- workflow scheduling
- retrieval-augmented inference
- model-based candidate planning
- policy and capability enforcement
- human approval workflows
- replay-based evaluation
- canary deployment and rollback

## 10.2 Difficult but tractable research

- reliable extraction of reusable skills
- causal model learning from passive interaction
- robust long-horizon planning under uncertainty
- calibrated confidence over composite plans
- automatic identification of stale or contradictory memory
- selecting when to deliberate versus act
- maintaining useful models of other agents and institutions

## 10.3 Highly speculative areas

- broad cross-domain transfer from limited experience
- unified representations of physical, social, and abstract environments
- autonomous discovery of genuinely new general algorithms
- stable self-improvement without extensive evaluation infrastructure
- open-ended operation with minimal human supervision

The architecture isolates these speculative capabilities so that failures do not compromise the runtime or governance substrate.

## 10.4 Initial technology stack

One practical implementation could use:

```text
Event log: Kafka, Redpanda, or PostgreSQL append-only tables
State store: PostgreSQL
Graph store: PostgreSQL graph extension or Neo4j
Vector index: pgvector, Milvus, or equivalent
Workflow engine: Temporal or a custom durable state machine
Sandbox: containers plus seccomp/gVisor or equivalent
Policy: declarative rules engine with signed policy bundles
Artifacts: object store plus immutable metadata registry
Metrics: OpenTelemetry-compatible telemetry
```

The particular products are replaceable. The interfaces matter more than the vendors.

## 10.5 Incremental implementation path

### Phase 0: Instrumented assistant

Implement:

- task manager
- event log
- working memory
- read-only tools
- basic audit
- benchmark harness

Gate: reliable reconstruction of every task and no unlogged tool invocation.

### Phase 1: Persistent memory

Add:

- episodic and semantic stores
- provenance
- temporal facts
- user-scoped memory
- consolidation workers

Gate: measurable improvement on multi-session tasks with bounded leakage.

### Phase 2: Typed planning and execution

Add:

- subgoal graphs
- action broker
- tool schemas
- precondition and postcondition checks
- retries and idempotency

Gate: successful recovery from injected tool failures and zero unauthorized test actions.

### Phase 3: World model

Add:

- entity graph
- temporal state
- affordances
- uncertainty
- narrow-domain simulator

Gate: simulator-assisted planning beats a no-simulator baseline on held-out tasks.

### Phase 4: Controlled learning

Add:

- skill induction
- replay testing
- planner ranking updates
- canary deployment
- automatic rollback

Gate: improvement without protected-benchmark regression.

### Phase 5: Multi-agent and open-ended operation

Add:

- typed specialist roles
- parallel decomposition
- long-lived tasks
- operator dashboards
- stress and interruption testing

Gate: measurable benefit at bounded cost and risk.

---

# 11. Originality or Non-Obvious Insight

## 11.1 Epistemic escrow as an execution primitive

The central non-obvious idea is to make uncertainty operationally binding.

Most systems represent uncertainty as a confidence score displayed to a user, while action execution remains effectively binary. Cognitive-OS instead gives uncertain beliefs an authority state:

```text
belief -> evidence evaluation -> escrow status -> action permission
```

A low-confidence belief may support:

- asking a clarification question;
- searching for evidence;
- running a reversible diagnostic;
- generating alternative plans.

The same belief may not support:

- sending an irreversible message;
- transferring money;
- changing access controls;
- deleting data;
- taking physical action.

This creates a falsifiable prediction:

> Systems with epistemic escrow will produce fewer high-impact errors caused by unverified assumptions, at the cost of more clarification and information-gathering actions.

The tradeoff can be measured by:

```text
high-impact error rate
clarification rate
task completion rate
time-to-completion
false escalation rate
```

## 11.2 Related design consequence: information actions become first-class

The planner should be able to choose actions whose primary purpose is uncertainty reduction:

```text
ask_user(question)
inspect_state(resource)
run_read_only_test()
request_second_source()
simulate_alternative()
```

These actions compete with ordinary task actions under the same planning objective. This avoids the common failure mode where an agent either acts prematurely or stops entirely when uncertain.

## 11.3 Architectural consequence

Epistemic escrow requires all of the following to interoperate:

- provenance-aware memory;
- explicit belief objects;
- calibrated uncertainty;
- typed action risk;
- policy evaluation;
- post-action observation.

It therefore acts as a useful integration test for whether Cognitive-OS is a real cognitive runtime rather than a collection of prompt templates.

---

# Proposed End-to-End Data Flow

```text
1. User or environment emits an event.
2. EventLog records the event.
3. TaskManager creates or updates a Task.
4. Memory subsystem retrieves relevant episodes, facts, skills, and beliefs.
5. WorldModel constructs a current state with uncertainty.
6. Decomposer creates a subgoal graph.
7. Planner proposes multiple candidate plans.
8. Verifier checks preconditions, postconditions, invariants, and assumptions.
9. Governance assigns risk and evaluates capabilities.
10. ActionBroker executes only authorized actions.
11. Tool results become typed observations.
12. EventLog records outcomes.
13. WorldModel and WorkingMemory update.
14. Repairer revises the plan if needed.
15. Consolidator extracts durable learning.
16. Evaluation service records performance and possible update proposals.
```

# Core Architectural Claim

Cognitive-OS should be built as a **durable, event-sourced, uncertainty-aware control system** around learned models—not as a single model with progressively larger context.

Its success should be judged by measurable improvements in:

- persistent task completion;
- recovery from environmental change;
- calibrated uncertainty;
- safe tool use;
- cross-session learning;
- auditability;
- resource-bounded autonomy.

The architecture is viable as an engineering program because its substrate is conventional, while its research risks are isolated into world modeling, skill induction, planning under uncertainty, and controlled self-improvement.

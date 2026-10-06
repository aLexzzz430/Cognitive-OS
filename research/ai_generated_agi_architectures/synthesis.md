# Synthesis: Cognitive Transactions for Cognitive OS

## Executive thesis

The strongest recurring design across the collected proposals is not a particular database, model, or agent hierarchy. It is a boundary: **probabilistic cognition proposes; durable state, governance, execution, and verification commit**. That boundary already matches Cognitive OS's documented direction away from an "LLM amplifier" and toward an evidence-governed runtime. The most useful synthesis is therefore not a new parallel architecture. It is a concrete transaction protocol that ties the repository's existing goal, world-model, self-model, planning, governance, evidence, action, and learning modules into one auditable state transition.

I call that protocol a **Cognitive Transaction**.

```text
Durable Goal Pressure
       |
       v
Situation Snapshot -----> Evidence Retrieval
       |                         |
       +----------+--------------+
                  v
        Hypotheses / Beliefs
                  |
                  v
          Candidate Plans
                  |
         +--------+--------+
         |                 |
   Self-model score   World-model lookahead
         |                 |
         +--------+--------+
                  v
         Governance Decision
           DENY / GATE / ALLOW
                  |
                  v
            Action Intent
                  |
                  v
        Environment Adapter
                  |
                  v
            Tool Receipt
                  |
                  v
        Verified Observation
                  |
        +---------+---------+
        |                   |
   World-model delta   Self-model delta
        |                   |
        +---------+---------+
                  v
          Learning Update
                  |
                  v
         Evidence Commit
```

The design extends the repository without moving local-machine repair, VM execution, model routing, or provider credentials back into the cognitive center. Those remain body/substrate capabilities beneath a domain-neutral contract.

## 1. The Cognitive Transaction schema

The repository already has many of the required mechanisms, but they become more general if each tick can emit and consume one domain-neutral record. A minimal schema is:

```text
CognitiveTransaction
  transaction_id
  tick_id
  goal_refs[]
  situation_version
  evidence_refs[]
  beliefs[]
  hypotheses[]
  plan_candidates[]
  selected_plan_ref
  selected_action
  self_model_assessment
  world_model_prediction
  governance_decision
  approval_state
  execution_receipt
  verified_outcome
  world_model_delta
  self_model_delta
  learning_delta
  status
  parent_transaction_id?
```

The important property is not the exact serialization. It is that every consequential transition has explicit causal links: **goal → evidence → belief/hypothesis → plan → governance → action → receipt → verified outcome → learning**. This prevents a model answer or tool response from silently becoming ground truth.

This schema maps naturally onto the project's current modules: `core/main_loop.py` remains the one canonical tick driver; goal pressure remains distinct from tasks; `core/world_model` and `self_model/` become pre-action inputs and post-outcome update targets; `planner/` produces advisory candidates; `modules/control_plane/action_governance.py` remains the permission authority; environment adapters execute; and the evidence ledger preserves the trace.

## 2. Preserve the one-loop invariant

Cognitive OS already specifies one formal 10-step tick. The synthesis should harden that invariant rather than introduce a second autonomous loop. The proposed mapping is:

1. **Perceive** → normalize observations and attach provenance.
2. **Refresh state** → create an immutable situation snapshot/version.
3. **Retrieve** → select evidence, episodes, semantic facts, and procedures relevant to the active goal pressure.
4. **Predict / hypothesize** → create competing beliefs and hypotheses, not one unqualified answer.
5. **Generate candidates** → plans/actions with stated preconditions, expected outcomes, costs, and reversibility.
6. **Model influence** → world-model lookahead plus self-model reliability modifies viability and score.
7. **Govern** → capability/risk policy emits `ALLOW`, `DENY`, or `HUMAN_GATE` with reason.
8. **Act** → adapter returns a structured receipt; timeout may yield `OUTCOME_UNKNOWN`, never assumed failure.
9. **Verify / appraise** → compare observed external state to expected postconditions.
10. **Learn / commit** → update evidence, world model, self model, procedure candidates, and goal pressure.

The loop remains domain-neutral. A repository patch, a research experiment, a browser task, or a simulated-world action all produce the same transaction types.

## 3. Memory: evidence-linked, typed, and reconstructible

Across the proposals, a useful consensus is that context windows are not durable memory. Cognitive OS should keep four operational memory classes plus the existing self-model:

- **Working state**: bounded task/tick material needed for the next decision. This is a projection, not the source of truth.
- **Episodic evidence**: append-only observations, decisions, receipts, outcomes, interventions, and failures.
- **Semantic memory**: propositions/entities with provenance, confidence, validity range, contradiction links, and retrieval features.
- **Procedural memory / skills**: versioned reusable procedures with preconditions, postconditions, tests, risk class, and historical reliability.
- **Self model**: capability/reliability/resource estimates that influence action selection but do not directly override governance.

A semantic item should never be promoted solely because an LLM summarized it confidently. Promotion requires evidence references. Consolidation should generate candidate mutations, run contradiction checks, then commit or retain the material only as an episode. This fits the repository's explicit preference for formal evidence over chat memory.

Forgetting should be asymmetric. Raw high-value evidence and action receipts have retention policy; derived embeddings and summaries are rebuildable. Low-value working context can expire aggressively. Procedural skills should decay in confidence when environment versions change or verification failures accumulate.

## 4. Planning: from single active plan to bounded plan graphs

The current planner already has decomposition, bounded beam search, constraints, revision, salvage, and world-model lookahead. The next increment should not be unrestricted recursive planning. It should be **bounded alternative plan graphs**.

Each candidate plan node declares:

```text
step_id
objective
preconditions[]
required_capabilities[]
expected_observation
expected_world_delta
estimated_cost
risk_class
reversibility
verification_method
fallback_ref?
```

The planner remains advisory. It cannot force execution. A plan becomes active only after feasibility scoring, self-model calibration, world-model prediction, and governance filtering.

Uncertainty should be actionable. If the expected value difference between two plans depends on an uncertain variable, the planner may propose an information-gathering action whose value is the expected reduction in decision uncertainty. That provides a concrete mechanism for curiosity without converting uncertainty into uncontrolled exploration.

## 5. Hypotheses and discriminating experiments

The repository's North Star explicitly requires competing hypotheses and posterior updates. The synthesis makes hypotheses first-class records:

```text
Hypothesis
  id
  claim
  prior_confidence
  evidence_for[]
  evidence_against[]
  discriminating_tests[]
  current_confidence
  status
```

An experiment is useful when its possible outcomes discriminate between live hypotheses. This avoids "research" that merely collects more supporting text. After an experiment, the observation is committed separately from the model's interpretation, and confidence changes cite that observation.

A key transfer test should require the same hypothesis lifecycle in at least two non-code environments. That directly targets the repository's stated architectural acceptance gate.

## 6. Governance below the model layer

All models should be treated as untrusted proposers of state transitions. Governance is a deterministic or independently-audited layer below them.

A capability decision should evaluate:

- requested action class;
- caller/agent identity;
- target resource and scope;
- credential requirement;
- side-effect and reversibility class;
- estimated monetary/resource cost;
- user/human approval requirement;
- evidence completeness;
- previous failure constraints;
- current self/world risk signals.

The result is an auditable record, not merely a boolean. `DENY` is terminal for that action intent. `HUMAN_GATE` preserves the exact pending action and required approval so that the system can resume deterministically later. `ALLOW` grants only the narrow capability needed for that step.

Prompt injection resistance follows from provenance: retrieved web pages, documents, tool outputs, and peer-agent messages are data unless they arrive through an authenticated control channel with authority to issue instructions.

## 7. Execution: receipts and outcome reconciliation

The most important operational distinction is among **command accepted**, **tool returned**, and **world changed as intended**.

Every side-effecting action receives an idempotency or operation key. The adapter returns an `ExecutionReceipt`:

```text
operation_id
adapter
capability
request_hash
started_at
completed_at?
provider_receipt?
raw_result_ref?
side_effect_class
status: SUCCESS | FAILURE | OUTCOME_UNKNOWN
```

`OUTCOME_UNKNOWN` is a first-class state. A network timeout after an external write must not trigger blind replay. The next action is reconciliation: inspect external state or provider receipts. This is particularly important for messages, purchases, deployments, file writes, task submissions, and payments.

VM/mirror execution remains a valuable body boundary for code and local-machine actions, but it is not the cognition. Other adapters—web, APIs, simulated worlds, future robotics—must fit the same receipt contract.

## 8. World model: typed state plus uncertainty

The world model should represent entities, relations, state intervals, and hypotheses with provenance and confidence. It should not be only a vector index. The minimum useful query interface is:

```text
state(entity, time?)
support(proposition)
contradictions(proposition)
uncertainty(variable)
predict(action, horizon)
```

Observed facts and simulated predictions remain distinct. When a planner invokes a simulator or model rollout, the result is tagged `PREDICTED`; after action, verifier evidence is `OBSERVED`. This prevents a forecast from being recursively remembered as a fact.

The current world-model influence hook can evolve from candidate scoring into explicit before/after state deltas. A failed prediction is then useful training data for model calibration rather than merely a task failure.

## 9. Self model: reliability as a decision input

The existing self model already tracks capability and reliability. The next useful additions are:

- a **blindspot registry** for recurring unknown/underestimated failure modes;
- calibration curves per action/tool/task family;
- environment/version qualifiers on reliability;
- evidence links behind every capability score;
- a distinction between insufficient evidence and measured low reliability.

The self model should influence candidate ranking and when to seek a stronger verifier, but governance remains the authority. Repeated verified success can create pressure to promote a workflow into a skill candidate; repeated failure creates repair/investigation pressure and new constraints. This closes the loop already envisioned by Cognitive OS without allowing the system to declare itself competent merely from self-reflection.

## 10. Multi-agent design: evidence producers, not personality theater

The proposals support a restrained multi-agent pattern: one canonical task coordinator plus temporary specialists. Specialists do not own durable global state. They receive bounded work packets and return typed proposals/evidence.

Useful specialists include:

- researcher/source verifier;
- builder;
- test/verification agent;
- security/risk critic;
- alternative-plan generator;
- domain specialist.

A second model family is most valuable where correlated model error is costly. Consensus alone is not evidence. The coordinator should ask specialists for counterexamples, tests, source evidence, or alternative predictions, then integrate results through a transaction.

Budget is explicit: each deliberation step consumes time, token/API quota, compute, or money. The coordinator stops additional debate when expected information value falls below marginal cost.

## 11. Learning and controlled self-improvement

Learning should operate at three timescales.

**Online:** update working state, evidence, reliability statistics, retrieval priorities, and hypothesis confidence.

**Consolidation:** convert repeated verified traces into skill candidates; convert failures into regression tests, constraints, and retrieval objects.

**Architectural/model evolution:** offline only. Any code, policy, prompt, model, or procedure update is versioned and evaluated against fixed regression suites plus adversarial tests. Promotion requires evidence and rollback metadata.

A particularly useful mechanism is counterfactual trace replay: after a task, replay selected decision points with alternative candidates, using known outcomes to detect systematic biases such as premature commitment, unnecessary tool calls, under-exploration, or overconfidence. These analyses propose changes; they do not directly rewrite governance or identity state.

## 12. Evaluation plan tied to the repository's actual AGI gate

The research synthesis should produce tests that distinguish real architectural progress from an LLM wrapper. The next benchmark families should include:

1. **Hidden-rule environment:** infer rules from experiments, update hypotheses, succeed without fixture-specific code.
2. **Novel-tool transfer:** expose a new tool schema and measure safe acquisition/use of the capability.
3. **Long-horizon memory:** interrupt and restart a multi-stage task; verify causal continuity from durable state.
4. **Ambiguous-objective refusal:** reward the system for recognizing missing authority or unsafe ambiguity rather than guessing.
5. **Cross-domain hypothesis transfer:** run the same hypothesis/experiment state machine in code and one non-code environment.
6. **Outcome-unknown recovery:** inject a timeout after an external write and verify reconciliation rather than duplicate execution.
7. **Failure-learning regression:** cause a repeatable failure, commit the learning object, then demonstrate measurably changed later behavior.
8. **Self-model calibration:** compare predicted action success probabilities with observed frequencies across task families.

Metrics should include verified task success, false-success rate, intervention rate, irreversible-error rate, calibration error, recovery time, cost, evidence completeness, and contradiction rate. False success deserves special weight because autonomous systems become dangerous when they report completion without world-state evidence.

## 13. Staged implementation roadmap

### Stage A — Transaction envelope

Add the domain-neutral Cognitive Transaction type and adapters from current tick state. Do not move existing modules. Make transaction IDs/evidence references visible in audit output.

**Exit criterion:** existing workflows run through the envelope without behavior regression.

### Stage B — Outcome semantics

Standardize execution receipts and `OUTCOME_UNKNOWN`; add reconciliation hooks and idempotency metadata to side-effecting adapters.

**Exit criterion:** injected post-write timeouts do not cause duplicate side effects.

### Stage C — Hypothesis and plan graphs

Unify hypothesis records outside the local-machine adapter and extend planner state from one active plan toward bounded alternative plan graphs.

**Exit criterion:** a non-code benchmark forms competing hypotheses, runs a discriminating experiment, and updates confidence.

### Stage D — Evidence-linked consolidation

Promote episodic/semantic/procedural memory contracts; require provenance on semantic promotion and verified traces for skill candidates.

**Exit criterion:** a failure changes future candidate ranking and a repeated success creates a testable skill candidate.

### Stage E — Transfer evaluation

Add hidden rules, novel tools, ambiguous objectives, restart/recovery, and long-horizon memory benchmarks.

**Exit criterion:** G1/G2 claims are backed by repeatable multi-domain evidence rather than repository-repair performance alone.

## 14. Non-obvious conclusions

**Recovery semantics are part of cognition.** A system that cannot distinguish uncertain external outcomes after a crash cannot choose safe long-horizon plans. Idempotency and reconciliation change the agent's feasible action space; they are not merely infrastructure details.

**Memory quality dominates memory volume.** Evidence-linked, contradiction-aware state is more valuable than an enormous vector store of summaries. Unqualified memory can amplify model hallucination across days.

**The strongest multi-agent contribution is discriminating evidence.** Additional agents are useful when they produce tests, contrary evidence, independent predictions, or domain expertise. Simulated roles that merely restate opinions add coordination cost without reducing uncertainty.

**A useful AGI boundary is transactional, not anthropomorphic.** The system does not need to mimic human cognitive labels to become more capable. It needs stable goals, explicit uncertainty, causal evidence, governed action, verified outcomes, and learning that measurably changes future behavior.

## 15. Falsifiable claims

This synthesis should be rejected or revised if tests show any of the following:

- the transaction envelope adds complexity without reducing false-success or recovery errors;
- world/self-model inputs do not measurably change better action selection;
- evidence-linked consolidation performs no better than uncontrolled summarization on contradiction/retention tests;
- cross-family critics add no independent fault detection relative to same-model critique at comparable cost;
- procedural memory fails to adapt when environment/tool versions change;
- the domain-neutral hypothesis lifecycle cannot transfer from code to a non-code environment without task-specific special cases.

Those are engineering questions, not philosophical ones. The architecture earns its place only where the evidence shows that it improves broad, persistent, safe autonomy.
# Combined architecture: evidence-backed local task execution

This is an analyst design informed by eight recorded local responses, the exploratory Gemini reference and separately attributed repository inspection. Danube3 echoed its prompt and supplies no architecture proposal; its failure is preserved. The design is not an implemented Cognitive-OS change or an AGI claim.

## Choice and attribution

Keep one authoritative controller, a bounded working view over existing state, and a durable action record connecting intent, observable effect and task verification. Start with one documentation-edit workflow. Expand memory or worker machinery only when a fixed evaluation shows that it improves verified completion enough to justify its cost.

SmolLM2 and Qwen2.5 name the requested memory, learning and recovery categories but do not supply sufficient mechanisms for implementation. The hosted candidates add useful request/receipt fields, observation/hypothesis separation, offline promotion and bounded-controller designs. Candidate A's transaction envelope is rejected at one measured boundary; candidate B's recovery requires an explicit pre-dispatch intent commit that its prose does not show. These are qualitative readings of these particular outputs, not a ranking of model families. References: [SmolLM2 notes](collection/smollm2/analysis-notes.md), [Qwen2.5 notes](collection/qwen25/analysis-notes.md), [hosted comparison](collection/gemini-flash-reference/analysis-notes.md).

The current repository already has a single tick loop, SQLite state, an event journal, failure-learning records and relevance matching. The proposal extends those surfaces instead of assuming an empty implementation. The inspected code is pinned to `e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca`; [repository-context.md](collection/repository-context.md) distinguishes source observations from proposed additions. The upstream minimal layout check and ten public smoke tests passed; no end-to-end runtime performance or recovery experiment has been run.

[Granite](collection/granite33/analysis-notes.md) supplies a different tradeoff: Neo4j-style semantic storage, two-phase commit, Raft-style replication and Paxos-style conflict resolution alongside a central controller. Those additions are not adopted for the minimum local slice because its output supplies neither a requiring workload nor the missing transaction-participant and recovery contracts. Its explicit episodic-memory ablation and modularity-versus-overhead tradeoff do inform the proposed evaluation. More named components are not themselves evidence of a better architecture.

[TinyLlama](collection/tinyllama/analysis-notes.md) repeats broad requirements and several whole sections while omitting requested mechanisms. It contributes no sufficiently specified new component. Keeping that result visible prevents the synthesis from attributing analyst-invented mechanisms to a model that did not provide them.

[Phi-3](collection/phi3/analysis-notes.md) makes baseline time/resource comparisons more explicit, which supports measuring cost on the same task set. Its claim that flushing a transaction log will store a missing receipt is not adopted: no durable receipt source or reconstruction probe is supplied. The synthesis instead requires an observation of the effect and retains uncertainty when that observation cannot decide.

[Qwen3](collection/qwen3/analysis-notes.md) explicitly leaves a missing-receipt completion unacknowledged, a useful limit on completion claims. Its task/context-based memory hierarchy motivates a retrieval index over evidence-linked records. That index is an analyst interpretation, not a replacement of the existing stores or a demonstrated speedup. Qwen3's retention numbers and unclear training/held-out partition are not adopted as defaults.

[Falcon3](collection/falcon3/analysis-notes.md) emphasizes coordination overhead and probabilistic uncertainty but gives no effect-reconciliation mechanism or ablation. It reinforces the need to measure added worker costs; it does not justify replacing observable verification with a confidence score. [Danube3](collection/danube3/analysis-notes.md) contributes no design mechanism because it returns the prompt.

## State and interfaces

```text
Goal + task-specific completion predicate
                  |
          Observe current evidence
                  |
      Model proposes bounded next step
                  |
    Controller validates current authority,
    expected state and remaining budget
                  |
       Commit durable action intent
                  |
        Dispatch one bounded action
                  |
      Observe / reconcile actual result
                  |
   Commit receipt and task-state transition
                  |
     Verify completion predicate or replan
```

These are proposed schema fields, not claims about existing classes:

| Record | Minimum content | Purpose |
| --- | --- | --- |
| Task | Goal, completion predicate/version, evidence IDs, status, step/replan limits | Make completion independently checkable and bound repeated attempts. |
| Action intent | Stable action ID, task ID, request hash, expected state, proposed result, authority revision, budget reservation, attempt state | Preserve the identity and assumptions of an invocation across interruption. |
| Receipt | Action ID, observed status, result/evidence IDs, verification outcome, actual resource use | Distinguish tool response, observable effect and verified task success. |
| Claim | Content, source observations, timestamp, confidence/uncertainty, contradiction status | Keep an interpretation linked to evidence that may later change. |
| Candidate lesson | Applicability conditions, source failure, proposed routine/version, training-case IDs, evaluation references | Make an improvement candidate distinct from a log entry. |

Use action states `PROPOSED → READY → DISPATCHED → OBSERVED → VERIFIED`, with `UNCERTAIN`, `FAILED` and `CANCELLED` alternatives. Persist `READY` before dispatch. A restart does not assign a fresh logical ID to an already recorded invocation. Only the controller commits task transitions; optional bounded workers return observations or proposals. Serialization determines update order, but a separate evidence-resolution step must handle contradictory claims.

Budget reservation occurs before dispatch, with actual cost reconciled afterward. Replanning consumes the same run budget and has its own limit; a depth cap alone does not bound repeated plan revisions. Stop requests prevent further dispatch and retain the observable state of in-flight work. This describes required behavior; it does not claim that a signal handler can instantly cancel every external operation or undo completed effects.

## Recovery for a reconcilable local edit

Gemini A proposes placing intent, filesystem effect and receipt inside one SQL transaction. The [executed example](collection/transaction-example.md) showed why that cannot supply its claimed pending-action scan: process exit left one file effect and zero pending rows. Committing intent before the effect left one pending row, permitting this specific effect to be reconciled without repeating it. The result is limited to the supplied temporary-file example, not power loss or the actual Cognitive-OS implementation.

The following is conceptual controller pseudocode, not a drop-in adapter:

```python
def propose_edit(task, observed_file, proposed_bytes):
    action = validated_intent(
        task=task,
        before_hash=hash_bytes(observed_file),
        after_hash=hash_bytes(proposed_bytes),
        proposed_content=proposed_bytes,
        stable_action_id=new_action_id(),
    )
    commit_intent_and_reservation(action)  # completes before dispatch
    return action

def reconcile_or_dispatch(action):
    observed = read_current_target(action)
    if observed.unavailable:
        return retain_uncertain(action)
    if observed.hash == action.after_hash:
        return verify_then_record_receipt(action, observed)
    if observed.hash != action.before_hash:
        return record_conflict_and_reobserve(action, observed)
    if not current_authority_and_budget_allow(action):
        return pause_without_dispatch(action)
    # Adapter must check the recorded precondition when applying the edit.
    result = apply_conditional_edit(action)
    # A timeout is not evidence that no edit occurred.
    return observe_then_verify_or_retain_uncertain(action, result)
```

The conditional-edit adapter needs a specified concurrency contract. A separate read followed by an unconditional write is insufficient when another process can edit the same file. This proposal does not assume that a whole directory can be atomically swapped across arbitrary filesystems. Start with the existing mirror/patch path and document its supported preconditions before adding a replacement adapter.

| Observation after interruption | Controller decision |
| --- | --- |
| Target matches proposed final hash | Recover the receipt; run semantic/task checks before declaring completion. |
| Target matches original hash | Recheck authority, budget and edit preconditions; retry the same logical action only if still valid. |
| Target matches neither | Preserve the conflicting observation and construct a new proposal from fresh evidence. |
| Target cannot be observed | Keep the outcome uncertain and bound further observation attempts. |

For a remote effect with neither a supported idempotency key nor an adequate read probe, retain uncertainty. This architecture does not offer a generic exactly-once effect guarantee. Within SQLite, receipt and corresponding task transition can share a transaction; a derived JSONL mirror must not silently become a second competing authority.

## Memory and learning

Working memory is a bounded task view, episodic memory records action/outcome sequences, semantic memory stores evidence-linked claims, and procedural memory stores evaluated routine versions. These categories are common in the collected text; the following operational contract is an analyst addition.

Retrieval returns the record ID, age, applicability reason and unresolved contradictions with the content. Begin with the repository's existing failure objects and tag-based relevance matching. Add a vector index only if an ablation demonstrates benefit against that real baseline. Evict working-cache entries separately from retaining evidence required to explain an active claim or accepted action.

Use task ID and task family as retrieval dimensions across memory types, following the useful direction in Qwen3's context-based proposal. Same-task observations are immediately relevant, while cross-task lessons require an explicit applicability match. Do not restrict all retrieval to the current task ID, which would defeat transfer to new tasks. Treat an index as a view over retained source records, not a second copy whose provenance can diverge.

A failed assumption creates a candidate lesson linked to a reproducible case. Freeze training cases separately from evaluation cases. Promote a routine only after a versioned evaluation reports verified completion, repeated mistakes and resource cost against the current routine. Historical transcripts alone do not establish a held-out test: case-family and content overlap must be tracked. Retain the prior version and its evaluation record for rollback. Appending logs or generating a plausible new prompt is not measured learning.

## One documentation-edit path

For a request to document a new `timeout_ms` parameter, first inspect the source signature, units, default and relevant behavior. Define the success predicate: the requested documentation accurately describes those facts, the intended file changes, and unrelated content remains intact. A compilation pass or a clean textual diff alone cannot establish that predicate.

The planner proposes a minimal edit bound to the observed source/document versions. The controller applies the current user's authority and budget rules, commits the intent, and dispatches through the supported local edit adapter. It verifies both the resulting content and the source facts before completion. If interrupted after the edit, it uses the recorded hashes to reconcile rather than blindly applying the patch again. If the source changed meanwhile, it rechecks the documentation's meaning even if the document's final hash matches the old proposal.

## Evaluation before promotion

The following three experiments are **proposed and not yet run**. They are broader than the one executed SQLite example. Freeze model, prompts, verifier, budgets and case selection within each comparison. Report failure counts and unresolved cases as well as averages.

1. **Interrupted edits:** compare the existing task path with the proposed intent/receipt path at predeclared interruption boundaries. Measure false completion declarations, duplicate mutations, unresolved outcomes and recovery cost. The target invariants are no false completion and no duplicate effect for the specified reconcilable file adapter. Observing those invariants in a finite suite would not prove them for arbitrary tools.
2. **Changing evidence:** change source or document content between proposal and application, including ordinary concurrent edits. Measure stale edits, correctly identified conflicts, completion after fresh observation and overhead. Ablate expected-state binding while holding the rest fixed. Reject an improvement claim if fewer stale edits result only from indiscriminate stopping and reduced useful completion.
3. **Failure-memory transfer:** compare current tag matching, no lesson retrieval and the proposed applicability filter on disjoint task families and new input variants. Measure verified completion, repeated known mistakes, irrelevant-lesson use and added token/tool cost. Reject the learning claim if gains disappear outside memorized cases or unrelated tasks regress beyond a threshold declared before the test.

## Six-week implementation estimate

| Week | Deliverable and decision |
| --- | --- |
| 1 | Freeze one documentation task corpus and completion predicates; measure the current baseline before replacing components. |
| 2 | Specify durable action records and one supported local edit/reconciliation path. |
| 3 | Exercise process-interruption and conflicting-edit cases; document receipt and evidence semantics. |
| 4 | Connect one candidate-lesson type to existing failure-memory storage and retrieval. |
| 5 | Run the fixed-budget, disjoint-case evaluation and ablations. |
| 6 | Review cost and failure tradeoffs; promote only supported changes and preserve a rollback path. |

The minimum useful slice is one task family, one controller and one observable edit effect. The principal engineering risk is that record/retrieval complexity increases cost without improving verified completion. The schedule is a planning estimate, and the proposed architecture provides no evidence that general intelligence has been achieved.

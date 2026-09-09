# Proposed integration: bounded planning over attributed state

This is an analyst synthesis, not a claim that the collected models designed or implemented this exact system. It combines modest useful ideas with corrections to their failure modes. The goal is a practical next iteration of a governed local runtime, not an assertion of AGI.

## Existing repository anchors

Inspected baseline: `e20d2ff4d5c84d4c11c87218c4ae9a04ab0046ca`.

- [`core/runtime/state_store.py`](../../core/runtime/state_store.py) defines SQLite-backed runs, tasks, events and approvals. It requests WAL and `synchronous=NORMAL`. This is existing storage; the proposals' general SQLite descriptions do not establish a new contribution or a stronger power-loss guarantee.
- [`core/runtime/evidence_ledger.py`](../../core/runtime/evidence_ledger.py) defines structured evidence references/hashes. Use those references for verifier observations instead of accepting a planner's success statement.
- [`core/orchestration/execution_control.py`](../../core/orchestration/execution_control.py) already contains capability and execution-ticket machinery. The proposed schema below is a research interface sketch to map onto these contracts, not a replacement authority system.
- [`modules/local_mirror/mirror.py`](../../modules/local_mirror/mirror.py), `apply_sync_plan`, checks plan identity, approver conditions, source/mirror hashes and patch-result hashes. Integration should reuse its authorized workflow; this packet neither reports an exploit nor demonstrates a defect in it.

## Proposed data and control boundary

Qwen's simple planner is the baseline. Spark/Phi's proposed gated execution and Granite's authority point motivate a candidate interface with an independent validator. The following fields are analyst additions resolving omissions in their outputs:

```python
ActionCandidate = {
    "action_id": "stable-id", "task_id": "task-id", "kind": "file_patch",
    "relative_path": "config/app.ini", "before_sha256": "...",
    "after_sha256": "...", "patch_sha256": "...",
    "policy_version": 4, "approval_ref": "...", "verifier_refs": ["..."],
    "preconditions": ["source hash unchanged", "path within declared scope"],
    "budget": {"max_attempts": 2, "max_seconds": 60}
}
```

The planner may propose candidates and reasons; it cannot issue its own authority. A trusted executor resolves the relative path against the approved workspace, rejects escapes/symlink races using the actual platform adapter, checks current policy and verifies the exact patch bytes. An approval refers to an immutable action digest and scope; editing an action or changing the policy invalidates stale authorization. A verifier records observed file hashes and test outputs as evidence. A model saying “done” changes no completion state.

State transitions: `proposed → validated → authorized → intent_committed → effect_observed → verified`. Rejection, conflict and expired authorization are explicit terminal or attention states. A queue initially permits one effectful worker; planning can be speculative, but two writers to the same target require serialization or an explicit conflict protocol. A task cannot silently increase its attempt/time limits or privileges after failure.

## Memory, world state and learning

Durable task/event/evidence records are authoritative for what the runtime observed and authorized. They are not an omniscient world model. World/self views are projections containing observation time, source evidence, capability version and uncertainty. Phi's graph and caching suggestions may become materialized projections if query measurements justify them. Cache entries are reconstructible, never the sole durable record.

Budget the GPU independently: model weights, KV cache, temporary activations and framework overhead all consume the 12 GB capacity. This collection demonstrates that its individual inference requests ran within available hardware; it does not benchmark a concurrent runtime. Start with one loaded model and one active generation. Do not implement Spark's fixed “persistent GPU state” allocation or Falcon's unavailable cloud/quantum proposal.

For self-improvement, persist candidate heuristic/ranking changes as data and evaluate them offline against frozen task fixtures. Compare verified success, invalid-action rate, wall time and peak memory against the current heuristic baseline. Accept only candidates meeting predeclared thresholds, with a separately authorized version change and rollback. The authority evaluator and approval scope are not learnable outputs. No RL training or self-improvement was executed in this study.

## A file edit across process death

Example: append `verified=true` to `mode=local`. Bind approval to the initial content, intended result and policy version. Persist this intent before replacing the file. Stage the entire resulting content in an owned temporary file, flush it, then atomically replace the target within the same filesystem. Persist observed completion afterwards.

If the process dies after replacement but before the completion record, restart inspects the target:

1. If its hash equals the intended result, record the observed effect without appending again.
2. If its hash equals the approved input, recheck current authority and apply the pending action if still allowed.
3. Otherwise stop with a conflict; do not overwrite an unrelated user edit.
4. If authority was revoked before a pending write, stop with denied status. A prior successful effect can be recorded without performing a new write; reversal would be a separate action requiring its own authority.

This is reconciliation for a bounded, single-file effect. It does not make SQLite and the filesystem one atomic transaction. Production integration additionally needs platform-specific directory durability, symlink-safe path handling, coordination against concurrent writers, disk-full behavior and multi-file semantics. Network actions need provider-specific idempotency/reconciliation; file hashes do not solve arbitrary payments or messages.

## Executed experiment and next gates

`recovery_experiment.py` compares two isolated implementations using subprocess termination via `os._exit(87)` at five named boundaries: before intent, after intent, after staging, after replace and after completion. Each is restarted twice. Two additional cases per implementation change the file or revoke permission after intent. The baseline interprets pending as permission to append again; the proposed version reconciles hashes and current policy. Both use temporary files and SQLite. This experiment does not import Cognitive-OS code.

The recorded Windows run has 14 cases. The reconciliation implementation satisfies all seven exact final-content/status invariants and leaves content unchanged on the second restart. The baseline satisfies four of seven, duplicates the append after replacement, and fails the two intervention cases. These are deterministic counterexamples to this explicit baseline, not a general performance comparison or an independent proof of production safety. The baseline is intentionally simple; SQLite journaling alone does not make it equivalent to a mature editor.

Process-death tests leave the operating system running and do not emulate a power cut or storage-controller failure. Atomic replace assumptions apply only within the owned local temporary directory. The example does not cover hostile concurrent actors, multiple files, arbitrary tool effects or full permission policy. Its zero-exit completion means its declared assertions ran; it does not mean the repository's entire runtime passed.

Before integrating this design, add adapter-level tests for conflict, revocation, disk-full, repeated recovery and each supported filesystem; use existing execution tickets and mirror validation. Measure a fixed set of successful and deliberately rejected file tasks against the heuristic baseline. Require no unapproved writes, no duplicate effects and no false completion evidence in the declared fault set. Then measure benefit/cost of caching or extra planners one change at a time. No benchmark numbers or AGI claims should be inferred from this proposal.

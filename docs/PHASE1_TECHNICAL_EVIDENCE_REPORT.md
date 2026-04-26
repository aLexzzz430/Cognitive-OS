# Phase 1 Technical Evidence Report

Status: Phase 1H evidence freeze plus Phase 1J true external Codex CLI baseline completed.

Artifact freeze: `artifacts/phase1h_evidence/`

True external baseline freeze: `artifacts/phase1j_external_baseline/`

Latest suite summary: `experiments/phase1c_suite/suite_summary.json`

## Claims Proven

1. The Phase 1 closed-loop suite is repeatable over the current controlled fixture set.
2. Full Cognitive OS outperforms posterior-free, discriminating-free, tool-only, and internal deterministic coding-agent baselines on task success.
3. Full Cognitive OS now creates explicit competing hypotheses as real run objects, not empty report fields.
4. Full Cognitive OS binds discriminating experiments to hypothesis ids.
5. Full Cognitive OS mutates real hypothesis posterior state before patch selection.
6. Full Cognitive OS links patch proposal/application to the leading hypothesis.
7. Verifier/completion gating prevents verified completion before final test evidence.
8. The ablation matrix is not contaminated by disabled target binding or disabled patch proposal capability.
9. Against the true Codex CLI baseline on this suite, Full Cognitive OS preserves mechanism traceability and auditability advantages but does not show task-success superiority.

## Claims Not Proven

1. This does not prove open-world software engineering competence.
2. This does not prove robustness on large repositories.
3. This does not prove superiority over a live frontier coding agent with unconstrained context.
4. This does not prove the patch proposal layer is general beyond the current bounded proposal families.
5. This does not prove long-horizon memory consolidation or cross-task transfer.
6. This does not prove task-success superiority over Codex CLI. Codex CLI matched Full Cognitive OS on the current 9-fixture suite.
7. This does not prove a fair model-capability comparison, because Codex CLI used `gpt-5.5` while the current Con OS full reports record `llm_provider = none`.

## Fixture Suite

| Fixture | Bug type | Expected behavior |
| --- | --- | --- |
| `amount_parse_bug` | amount parsing | Patch `ledger_core/amounts.py` and pass tests. |
| `misleading_currency_bug` | misleading currency normalization | Avoid surface traceback file and patch `ledger_core/currency.py`. |
| `semantic_discount_boundary_bug` | threshold boundary | Avoid surface invoice file and patch `ledger_core/discounts.py`. |
| `aggregation_key_bug` | grouping key | Patch true grouping implementation. |
| `state_mutation_bug` | mutable state leak | Patch true state owner with bounded diff. |
| `ambiguous_no_safe_patch` | ambiguous spec | Refuse unsafe patch and request human review. |
| `dual_cause_total_bug` | tax vs discount ambiguity | Use discriminating evidence and patch true downstream cause. |
| `decoy_patch_bug` | targeted-pass decoy | Avoid decoy surface patch; verify full suite. |
| `ambiguous_two_patches_bug` | two plausible patches | Use disambiguating evidence; avoid arbitrary patch. |

## Variant Definitions

| Variant | Meaning |
| --- | --- |
| `full` | Full Phase 1 loop: explicit hypotheses, posterior writeback, discriminating experiment selection, posterior-to-action bridge, target binding, bounded patch proposal, verifier gate. |
| `no_posterior` | Keeps investigation tools but disables posterior writeback. |
| `no_discriminating_experiment` | Keeps posterior and tools but disables discriminating experiment selection/bonus. |
| `tool_only_greedy` | Allows tools but disables posterior writeback, hypothesis posterior ranking, discriminating selection, and posterior-to-action bridge. |
| `external_coding_agent_baseline` | Internal deterministic coding-agent baseline: run tests, read failure, bind target, read file, generate bounded patch proposal, run targeted/full tests, rollback failed patch. It does not use explicit hypotheses, posterior writeback, discriminating selection, posterior bridge, or leading-hypothesis requirement. |
| `codex_cli` | True external Codex CLI baseline. It receives only the generated task package (`TASK.md`, `repo/`, `verifier.json`), runs independently, and is judged by the same pytest verifier. |

## Metrics Definitions

| Metric | Definition |
| --- | --- |
| `task_success` | The task outcome is correct: verified patch for solvable fixtures or justified refusal for ambiguous fixture. |
| `cognitive_success` | Requires task success plus evidence of the intended cognitive mechanism path. External baseline is not eligible for cognitive success. |
| `hypothesis_lifecycle_complete` | At least two competing hypotheses exist, a discriminating experiment references hypothesis ids, posterior evidence updates a hypothesis, a leading hypothesis exists before patch, patch references that hypothesis, and verifier/refusal outcome is valid. |
| `ablation_contaminated` | Ablation failed because core tools were disabled rather than because the cognitive mechanism was removed. |
| `wrong_patch_attempt_count` | Patch attempted on decoy or non-true target file. |
| `rollback_count` | Failed verifier-gated patch attempts rolled back. |
| `verification_waste_ticks` | Failed verifier actions after patch attempt plus rollback cost. |

## 180-Run Phase 1H Summary

Frozen artifact: `artifacts/phase1h_evidence/suite_summary.json`

| Variant | Runs | Task success | Solvable cognitive success | Hypotheses created | Lifecycle complete | Contamination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 45 | 45/45 | 40/40 | 45/45 | 45/45 | 0 |
| `no_posterior` | 45 | 0/45 | 0/40 | 45/45 | 0/45 | 0 |
| `no_discriminating_experiment` | 45 | 5/45 | 0/40 | 45/45 | 0/45 | 0 |
| `tool_only_greedy` | 45 | 0/45 | 0/40 | 45/45 | 0/45 | 0 |

## 225-Run Internal Baseline Summary

Latest command:

```bash
.venv/bin/python experiments/phase1c_suite/run_suite.py --repeats 5 --max-ticks 25 --include-external-baseline
.venv/bin/python experiments/phase1c_suite/analyze_suite.py
```

| Variant | Runs | Task success | Solvable task success | Solvable cognitive success | Wrong patches | Rollbacks | Verification waste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 45 | 45/45 | 40/40 | 40/40 | 0 | 0 | 0 |
| `no_posterior` | 45 | 0/45 | 0/40 | 0/40 | 0 | 0 | 0 |
| `no_discriminating_experiment` | 45 | 5/45 | 0/40 | 0/40 | 0 | 0 | 0 |
| `tool_only_greedy` | 45 | 0/45 | 0/40 | 0/40 | 0 | 0 | 0 |
| `external_coding_agent_baseline` | 45 | 25/45 | 20/40 | 0/40 | 5 | 5 | 10 |

Internal deterministic baseline task deltas where Full beats baseline:

| Fixture | Full - external task delta |
| --- | ---: |
| `amount_parse_bug` | 1.0 |
| `misleading_currency_bug` | 1.0 |
| `aggregation_key_bug` | 1.0 |
| `ambiguous_two_patches_bug` | 1.0 |
| `semantic_discount_boundary_bug` | 0.0 |
| `state_mutation_bug` | 0.0 |
| `dual_cause_total_bug` | 0.0 |
| `decoy_patch_bug` | 0.0 |
| `ambiguous_no_safe_patch` | 0.0 |

The internal deterministic baseline produced 5 wrong decoy patch attempts, all on `ambiguous_two_patches_bug`, followed by 5 rollbacks and 10 wasted verifier ticks.

## True External Codex CLI Baseline

Frozen artifact: `artifacts/phase1j_external_baseline/`

Runner command:

```bash
.venv/bin/python experiments/external_baseline/run_external_baseline.py --mode command-adapter --agent-name codex_cli --timeout-seconds 900 --agent-command "codex exec --skip-git-repo-check --ephemeral --full-auto --sandbox workspace-write 'Read TASK.md and complete the external baseline task. Work only inside repo/. Do not modify tests. Run python -m pytest -q from repo/ before claiming success.'"
.venv/bin/python experiments/external_baseline/analyze_external_vs_conos.py --input artifacts/phase1j_external_baseline
```

Codex CLI transcript identity:

```text
provider=openai
model=gpt-5.5
```

The true external Codex CLI baseline matched Full Cognitive OS on task success:

| Agent | Task success | Cognitive success | Repo cleanliness | Minimal patch | Tests modified |
| --- | ---: | ---: | ---: | ---: | ---: |
| `conos_full` | 1.0 | 0.888889 | 1.0 | 1.0 | 0 |
| `codex_cli` | 1.0 | 0.0 | 0.866667 | 0.966667 | 0 |

External Codex CLI per-fixture task outcome:

| Fixture | Codex task success | Changed paths | Cleanliness note |
| --- | ---: | --- | --- |
| `amount_parse_bug` | 1.0 | `ledger_core/amounts.py` | source-only |
| `misleading_currency_bug` | 1.0 | `ledger_core/currency.py`, `uv.lock` | extra lockfile artifact |
| `semantic_discount_boundary_bug` | 1.0 | `ledger_core/discounts.py` | source-only |
| `aggregation_key_bug` | 1.0 | `sales_core/grouping.py` | source-only |
| `state_mutation_bug` | 1.0 | `state_core/state.py`, `uv.lock` | extra lockfile artifact |
| `ambiguous_no_safe_patch` | 1.0 | none | correct refusal, `ambiguous_spec` |
| `dual_cause_total_bug` | 1.0 | `checkout_core/discounts.py` | source-only |
| `decoy_patch_bug` | 1.0 | `billing_core/discounts.py` | source-only |
| `ambiguous_two_patches_bug` | 1.0 | `promo_core/policy.py` | source-only |

Comparison summary:

| Metric | Value |
| --- | ---: |
| `full_vs_codex_task_delta` | 0.0 |
| `full_vs_codex_cleanliness_delta` | 0.133333 |
| `full_vs_codex_traceability_delta` | 1.0 |
| `codex_cli` `non_source_artifact_count` | 2 |
| `codex_cli` `lockfile_created_without_need` | 2 |
| `codex_cli` test modification violations | 0 |

## Supported Claims After External Baseline

1. Full Cognitive OS remains task-success competitive with Codex CLI on the current controlled 9-fixture suite.
2. Full Cognitive OS retains stronger mechanism traceability: explicit hypotheses, discriminating experiment binding, posterior update evidence, patch-hypothesis linkage, and verifier-gated completion.
3. Full Cognitive OS retains stronger artifact cleanliness on this run because Codex CLI introduced two unnecessary `uv.lock` files.
4. Full Cognitive OS retains ablation evidence: posterior-free, discriminating-free, and tool-only variants remain below Full on the same fixture suite.
5. Both Full Cognitive OS and Codex CLI correctly handled the ambiguous no-safe-patch fixture without modifying tests.

## Unsupported Claims After External Baseline

1. The current suite does not support a claim that Full Cognitive OS has higher task success than Codex CLI.
2. The current suite does not support a claim that Con OS uses a stronger model than Codex CLI. Codex CLI used `gpt-5.5`; Con OS full reports currently record no LLM provider/model.
3. The current suite does not support open-world superiority over frontier coding agents.
4. The current suite is too small and synthetic to separate strong coding-agent competence from Con OS mechanism advantage by task success alone.
5. Future differentiation needs harder held-out tasks, larger repositories, or same-model weak/strong amplifier comparisons.

## Ablation Contamination Audit

All variants have `ablation_contaminated = 0`.

The baseline and ablations still had access to core non-cognitive capabilities where intended:

- run tests
- read files
- target binding
- bounded patch proposal
- patch verification

Failures therefore reflect removed posterior/discriminating/hypothesis mechanisms rather than disabled basic tools.

## Full Mechanism Trace Example

Example report: `experiments/phase1c_suite/reports/amount_parse_bug_full_ticks25_run1.json`

Action sequence:

```text
repo_tree -> run_test -> file_read -> apply_patch -> run_test -> run_test -> no_op_complete ...
```

Mechanism evidence:

- Tick 2 created two hypotheses:
  - `hyp_surface_wrapper_e6f4446b29fa`
  - `hyp_unresolved_spec_3b5206f2904b`
- Tick 2 proposed discriminating test `dtest_f126d1e66219c089`.
- The discriminating test references both hypothesis ids.
- `posterior_events_bound_to_hypotheses_count = 3`.
- `leading_hypothesis_before_patch = hyp_surface_wrapper_e6f4446b29fa`.
- `patch_referenced_hypothesis = true`.
- Patch target: `ledger_core/amounts.py`.
- Terminal state: `completed_verified` at tick 5.

## Residual Risks

1. The fixture suite is still small and synthetic.
2. The internal external baseline is deterministic and bounded; the true Codex CLI baseline is live but only one run per fixture.
3. Full relies on existing bounded patch families; broader patch generation remains unproven.
4. Current checkout has no `.git` directory, so repository-level version tags cannot be created here.
5. Evidence is frozen by file hashes and artifacts, not by a Git commit hash in this local directory.

## Next External Baseline Plan

The next baseline should compare same-model agent modes under the same fixture suite:

1. Run bare weak models (`qwen3:8b`, `batiai/gemma4-e4b`) outside Con OS.
2. Run the same weak models through Con OS.
3. Compare task success, verifier waste, wrong patch rate, refusal quality, and repo cleanliness.
4. Keep the same task packages and verifier.
5. Freeze reports next to `artifacts/phase1j_external_baseline/` for side-by-side audit.

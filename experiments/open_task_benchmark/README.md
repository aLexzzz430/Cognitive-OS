# Open Task Benchmark

This harness is the first reusable layer for real open-ended project tests. It does not claim Con OS wins on arbitrary GitHub projects. It creates leak-free task packages and normalizes result reports so Con OS, Codex, Claude Code, local models, and other baselines can be compared on the same terms.

## Package Tasks

```bash
python experiments/open_task_benchmark/run_benchmark.py --mode package-only --limit 2
```

The package-only mode is offline. It writes task prompts, verifier contracts,
metadata, an execution contract, a runbook, and an agent result template under
`runtime/reports/open_task_benchmark/tasks/`.

Each `execution_contract.json` records the public source reference, verifier
policy, budget, cost policy, context policy, failure recovery policy, forbidden
actions, and required evidence. The contract is deliberately answer-free: it
does not contain hidden target files, expected patches, or solution metadata.

## Analyze Results

```bash
python experiments/open_task_benchmark/analyze_results.py --input runtime/reports/open_task_benchmark/results
```

The analyzer reports verified success, task success, cost, traceability, route
traceability, budget observability, patch minimality, repo cleanliness,
test-modification violations, hidden fallback-patch violations, failure recovery
events, and non-source artifact pollution. Amplification Efficiency is computed
when both Con OS and the selected baseline have nonzero success and cost.

## Leak Prevention

Task packages reject project metadata fields such as `true_bug_file`, `expected_patch`, `hidden_solution`, and `target_file`. The task prompt only contains the public repository reference, natural-language objective, verifier command, and benchmark rules.

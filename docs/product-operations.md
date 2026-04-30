# Product Operations Manual

This manual is the operator-facing recovery path for the distilled Cognitive OS
runtime. It is intentionally CLI-first and does not require a desktop app.

## First Run

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
conos setup --one-click
conos validate-install
conos doctor
```

`conos setup --one-click` prepares local runtime directories, writes the setup
manifest, installs the user-level launchd service file, runs the VM boundary
gate, and returns one JSON report with `operator_panel` and
`recovery_diagnosis_tree`. It does not start the service unless
`--start-service` is present, and it does not execute VM setup unless
`--execute-vm-setup` is present. `conos doctor` checks the core import path,
repository layout, runtime state directory, VM provider, and LLM policy
contracts.

`conos validate-install` is the final verifier for installation. It reports
`setup_actions` when files or service manifests are still missing, and
`validation_remaining` when setup is complete but the current machine still
needs live validation such as starting launchd or proving the VM guest-agent
boundary.

For product readiness, run:

```bash
conos validate-install --product
```

This is stricter than the normal developer verifier. It requires the managed VM
default side-effect boundary, patch-gated source sync, approval-backed
sync-back, side-effect audit events, credential isolation, and policy-controlled
network access. If the VM boundary is not ready, it returns
`product_deployment_gate.status=BLOCKED` and the install must not be described
as deployable AGI.

## Daily Status

```bash
conos status
conos logs --tail 120
conos approvals
conos vm setup-plan
conos vm setup-default
conos vm status
```

The setup/status commands emit JSON with `operator_summary`, `operator_panel`,
`recovery_guidance`, and `recovery_diagnosis_tree`. `operator_panel` is the
human-facing control surface used by `conos setup`, `conos status`,
`conos doctor`, `conos logs`, `conos approvals`, and `conos validate-install`.
It contains:

- `health`: `healthy`, `warning`, or `needs_action`.
- `message`: one short human-readable explanation.
- `top_issues`: stable issue codes.
- `next_actions`: the next safe commands or checks.

Each `recovery_guidance` item has:

- `issue`: stable machine-readable problem code.
- `message`: short human-readable explanation.
- `next_actions`: safe next commands or checks.
- `severity`: `action_needed` or `warning`.

`recovery_diagnosis_tree` groups those issues into stable recovery categories:

- `install_runtime`: Python, dependencies, setup manifest, launchd, repo layout.
- `vm_boundary`: runner, base image, runtime process, guest agent, default VM boundary.
- `model_runtime`: Ollama/OpenAI/Codex endpoint, auth, timeout, format, budget.
- `approval_permission`: WAITING_APPROVAL, permission denied, credential/network/sync-back approvals.
- `verifier_tests`: verifier or pytest failure, degraded run, missing test dependency.
- `logs_runtime`: stderr, error-like log signals, empty logs, watchdog degraded.

Each category has `matched_issues`, `likely_cause`, `recovery_path`,
`blocked_until`, and `escalation_condition` so operators can recover without
guessing which low-level module produced the warning.

## Recovery Map

| Symptom | Guidance issue | Next action |
| --- | --- | --- |
| VM runner missing | `missing_vm_runner` | `conos vm build-runner` |
| VM image missing | `missing_vm_image` | `conos vm recipe-report`, then bootstrap/import an image |
| VM start blocked | `vm_start_blocked` | `conos vm start-instance`, inspect runner stderr |
| Guest agent not ready | `guest_agent_not_ready` | `conos vm agent-status`, then `conos vm recover-instance` |
| Model endpoint unavailable | `model_unavailable` | `conos llm check`, verify Ollama/OpenAI/Codex auth |
| Approval is pending | `waiting_for_approval` | `conos approvals`, then `conos approve <approval_id>` |
| Permission denied | `permission_denied` | Check policy and only approve the needed capability layer |
| Tests fail | `run_failed_or_degraded` | Read logs/evidence before retrying or resuming |

Timeouts are terminal timeout events by default. Patch fallback is disabled by
default and must never start silently after a model timeout. Configured model
fallback or cloud escalation is allowed only when the route policy records an
auditable `llm_failure_policy_decision` event with the failure type, recommended
action, retry/escalation flags, and fallback permissions.

## VM Setup Gate

`conos vm setup-plan` is the read-only product gate for the built-in VM. It
collapses the lower-level VM commands into a single readiness plan:

- runner: bundled Apple Virtualization launcher is built.
- base image: image manifest and disk artifact are present.
- instance: default instance manifest is prepared.
- runtime process: a live VM pid is recorded.
- guest agent: the vsock guest-agent gate is ready.
- execution boundary: Con OS can run tasks without falling back to host
  execution.

Only when `safe_to_run_tasks=true` should open-ended local-machine tasks use the
managed VM as the default execution boundary. If it is false, follow the first
`next_actions` group before retrying.

`conos vm setup-default` runs the same flow as an auditable setup workflow. It
stays in dry-run mode unless `--execute` is present, and image downloads remain
off unless `--allow-artifact-download` is also present. Executed stages append
events to `~/.conos/vm/logs/setup-audit.jsonl`.

## Open Task Benchmark

Use the open-task harness to prepare real-project benchmark packages and compare
agents without leaking hidden answers:

```bash
python experiments/open_task_benchmark/run_benchmark.py --mode package-only --limit 2
python experiments/open_task_benchmark/analyze_results.py --dry-run
```

Generated task packages live under `runtime/reports/open_task_benchmark/` by
default and are not part of the source package. Each package includes
`execution_contract.json` and `RUNBOOK.md` so agents are evaluated against the
same public source reference, verifier, budget, cost policy, context policy, and
failure recovery rules. Result reports should include `llm_route_usage`,
`budget_summary`, `failure_recovery_events`, and cost or an explicit
`unknown_cost_reason`; the analyzer scores route traceability and budget
observability separately from task success.

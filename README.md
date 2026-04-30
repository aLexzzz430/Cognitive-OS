# Cognitive OS

Cognitive OS is a local-first, evidence-governed runtime for building toward a
general intelligence operating system. It is not documented as an achieved AGI;
the product boundary is a verifiable cognitive runtime that maintains goals,
evidence, hypotheses, world/self state, governed actions, and long-running
recovery. This distilled repository keeps the core runtime, model routing,
local-machine adapter, empty-first mirror, verifier gates, failure learning,
and managed VM provider. Desktop UI shells, WebArena, ARC-AGI-3 adapters,
benchmark fixtures, and frozen report artifacts have been removed from the core
distribution.

## Quick Start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
conos preflight --strict-dev
conos layout
pytest -q
```

Runtime-only smoke checks:

```bash
conos version
conos preflight
conos setup --one-click
conos validate-install
conos setup --dry-run
conos doctor
conos status
conos logs --tail 120
conos approvals
conos vm status
conos run local-machine --help
conos mirror --help
conos llm --help
conos vm --help
```

## Product Entry Point

`conos` is the only public CLI entry point.

```bash
conos run local-machine --instruction "inspect README" --candidate README.md --max-ticks 2
conos mirror init --mirror-root runtime/mirrors/session-1
conos llm control-plane --route structured_answer --required-capability structured_output --permission generate_text
conos auth codex status
conos vm report
conos supervisor --help
```

The distilled run target is `local-machine`. Removed targets such as
`arc-agi3`, `webarena`, `app`, `ui`, `eval`, and `dashboard` are intentionally
not part of this package.

## Core Capabilities

- **Runtime supervision**: SQLite-backed resumable runs, task state, leases,
  approvals, event journal, resource watchdog, soak mode, pause/resume, and
  crash recovery. Product status is normalized into runtime modes such as
  `SLEEP`, `IDLE`, `ROUTINE_RUN`, `DEEP_THINK`, `CREATING`, `ACTING`, `DREAM`,
  `WAITING_HUMAN`, and `DEGRADED_RECOVERY`; see
  `docs/runtime-modes.md`.
- **AGI-direction control plane**: a North Star ledger, autonomous task
  discovery, explicit evidence/hypothesis lifecycle, self/world-state
  separation, model-state-driven action influence, outcome-driven self/world
  model updates, runtime goal pressure from self-model learning,
  skill-candidate discovery, planner-visible active goal pressure, normal
  no-user-instruction autonomous ticks for safe L0/L1 goals, homeostasis
  diagnostics from self/world/runtime pressure, formal evidence commit for
  idle-time diagnostics, pressure-resolution escalation to `WAITING_HUMAN`,
  `DEEP_THINK`, or approved limited L2 mirror investigation, refusal gates, and
  external baseline comparisons. See `docs/agi-north-star.md`.
- **LLM routing**: provider inventory, model profiles, route policies,
  thinking-policy control, budget-aware routing, Ollama/OpenAI/Codex CLI
  adapters, and explicit runtime contracts.
- **Governance**: action capability checks, permission gates, verifier policy,
  source-sync approval, failure learning ledger, and structured audit events.
- **Local-machine adapter**: atomic repo/file/test/edit actions, action
  grounding, target binding, bounded patch proposal, verifier-gated patching,
  and local mirror integration.
- **Empty-first mirror**: materializes only requested files, executes inside a
  controlled workspace, builds patch-gated sync plans, verifies source hashes,
  writes rollback checkpoints, and audits every apply.
- **Managed VM provider**: Con OS-owned VM state root, Apple Virtualization
  runner launcher, guest-agent initrd bundle, base-image bundle export/import,
  and default execution boundary for local-machine commands. There is no
  silent host-exec fallback when the VM path is unavailable.

## Local Mirror

```bash
conos mirror init --mirror-root runtime/mirrors/session-1
conos mirror fetch --mirror-root runtime/mirrors/session-1 --path README.md
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend local \
  --allow-command python3 \
  -- python3 -c "from pathlib import Path; print(Path('README.md').exists())"
conos mirror plan --mirror-root runtime/mirrors/session-1
```

`mirror exec` and `conos run local-machine` default to the Con OS managed-VM
boundary. The `--backend local` form above is a development-only host-exec
opt-in for quick smoke checks; product execution should keep the default
managed VM boundary and bootstrap the VM first.

`mirror apply` is a patch gate: it verifies planned source hashes and mirror
hashes before changing the source tree. Rollback checkpoints are recorded for
applied plans.

Security governance is layered into `read`, `propose_patch`, `execute`,
`network`, `credential`, and `sync-back` capabilities. Side-effecting actions
emit structured audit events, network access is policy controlled, credential
use is explicit and redacted in audit, and source sync is never copy-back: only
approved patch-gate plans can cross from mirror/VM back into the source tree.
Local-machine runs can also constrain those layers per task through action
governance policy, for example allowing only `read` and `propose_patch` while
blocking `execute`, `network`, or `sync-back` before the action reaches the
adapter. Capability-layer approvals are first-class runtime inbox items:
`WAITING_APPROVAL` records can be listed with `conos approvals`, approved with
`conos approve <approval_id>`, and the approved layer is injected back into the
same run's governance state on resume. Side-effecting local-machine actions
also pass through a final audit guard: if a normal governance decision is
present, that decision becomes the side-effect audit record; if governance is
disabled in a development path, the adapter still emits and persists a
`side_effect_audit_event` before returning the action result.

Execution credentials are isolated by default. Local host execution no longer
inherits the full host process environment; it receives only a small sanitized
base environment plus explicit `extra_env` keys. Env values are redacted in
mirror/audit records, and sensitive env key names such as tokens, passwords, or
API keys are treated as credential access by action governance before the
command reaches the execution adapter. If a credential must be injected, it has
to be supplied as a credential lease with a `lease_id`; leased credentials enter
the `credential` capability path, can require approval, and are redacted from
command output and audit payloads.

Network ingress is also policy-gated. Internet tools are hidden unless a run
explicitly enables them, private/local network targets require governance
approval, URL credentials are rejected, and git-based project fetches run with a
sanitized process environment instead of inheriting host Git/API credentials.
Runs can declare host allow/block lists through the local-machine adapter or
CLI (`--internet-allowed-host`, `--internet-blocked-host`,
`--internet-blocked-host-suffix`). Successful network artifacts carry a
`network_policy_audit` record in both the artifact metadata and the side-effect
audit path.

## Managed VM

```bash
conos vm report
conos vm build-runner
conos vm build-guest-initrd --state-root ~/.conos/vm
conos vm bundle-base-image --state-root ~/.conos/vm --image-id conos-base
conos vm install-base-image-bundle --bundle-dir ~/.conos/vm/image-bundles/conos-base
```

The managed VM path is the default execution boundary. `agent-exec` is blocked
until `runtime.json` proves a live VM process plus guest-agent readiness. There
is no silent fallback to host execution; local host execution requires an
explicit `--execution-backend local` or `--backend local` opt-in.

The long-running runtime can also watch and recover the managed VM through
launchd:

```bash
conos install-service --vm-watchdog --vm-auto-recover
conos start
conos status --vm-watchdog
```

When enabled, the daemon writes VM health into the service status snapshot. A
bad VM boundary marks active runs `DEGRADED`; with `--vm-auto-recover`, the
daemon calls the same `recover-instance` path used by the CLI.

For a bounded failure-injection check:

```bash
conos vm recovery-drill --instance-id default
conos vm recovery-soak --instance-id default --rounds 3
```

The drill kills the recorded VM runner process, confirms the boundary becomes
unhealthy, recovers it, and verifies guest execution with a small agent command.
The soak repeats that drill, records recovery-time distribution, and adds a
small guest disk probe after each recovered round.
The managed guest agent is configured as an early `sysinit.target` service and
the Con OS cloud-init seed overrides `systemd-networkd-wait-online.service`, so
VM recovery waits for the Con OS vsock boundary instead of a full
network-online userspace chain. Repeated EFI-disk starts also reuse an existing
Con OS observable GRUB configuration when the fallback loaders are already
present, avoiding a full root-disk boot-artifact scan during normal recovery.

## Product Operations

For ordinary operation, start with:

```bash
conos setup --one-click
conos validate-install
conos doctor
conos status
conos logs --tail 120
conos approvals
conos vm setup-plan
conos vm setup-default
conos vm status
```

`conos doctor`, `conos status`, `conos vm setup-plan`, and `conos vm status` include
`operator_summary` plus machine-readable `recovery_guidance`. Common failures
are mapped to stable issue codes:

- `missing_vm_runner`: build the bundled Apple Virtualization runner with
  `conos vm build-runner`.
- `missing_vm_image`: create, import, or bootstrap a Con OS base image before
  expecting VM execution.
- `vm_start_blocked` or `guest_agent_not_ready`: inspect `conos vm status`,
  then use `conos vm recover-instance` or `conos vm recovery-drill`.
- `model_unavailable`: check the model endpoint or login with `conos llm check`
  / `conos auth codex status`; model timeout does not secretly start fallback
  patching.
- `permission_denied` or `waiting_for_approval`: use `conos approvals` and
  approve only the exact capability layer you want to allow.
- test/verifier failures: inspect `conos logs --tail 200` and the run evidence
  before resuming or retrying.

`conos vm setup-plan` is the product-level readiness gate for the default VM
execution boundary. It is read-only: it does not download images or start a VM.
It reports runner, base image, instance manifest, live runtime process, guest
agent, and execution-boundary readiness, then returns the first safe next
actions needed before open-ended tasks can run inside the built-in VM.

`conos vm setup-default` uses the same gate as an audited workflow. By default
it is also a dry run. Add `--execute` to run the next setup stages, and add
`--allow-artifact-download` only when you want Con OS to fetch a digest-pinned
base image recipe.

`conos setup --one-click` is the user-level install path. It prepares runtime
storage, installs the per-user launchd service file, runs the VM boundary gate,
and returns a single `one_click_report`. It does not start the service unless
`--start-service` is present, and it does not execute VM setup unless
`--execute-vm-setup` is present.

`conos validate-install` is the post-install verifier. It separates missing
setup work from live validation work: `setup_actions` means the installer still
needs to prepare files, while `validation_remaining` means the remaining work is
to start or prove the runtime/VM boundary on the current machine.

For deployment claims, use the stricter product gate:

```bash
conos validate-install --product
```

Product mode treats the managed VM default side-effect boundary as required.
If the VM gate is not ready, the command fails and returns a
`product_deployment_gate` report. Development-only local host execution remains
available only through explicit backend opt-in and does not count as deployable
AGI execution.

## LLM Providers

```bash
conos llm --provider ollama profile --discover-visible --catalog-only
conos llm --provider codex profile --discover-visible --catalog-only
conos llm route --route patch_proposal --required-capability coding
```

When a provider is connected, Con OS can inventory visible models, build model
profiles, and emit route policies. Cheap routes default to no-thinking; planning
and patch design can use larger thinking budgets and longer timeouts. Model
outputs pass through reliability adapters before they become tool kwargs or
patch proposals: malformed JSON, missing required kwargs, repeated actions, and
timeouts become structured trace records. Patch fallback is disabled by default
when a real model times out; any escalation or fallback path must be explicit and
auditable.

## Open Task Benchmark

The real-project benchmark protocol starts in
`experiments/open_task_benchmark/`. It is intentionally offline by default:

```bash
python experiments/open_task_benchmark/run_benchmark.py --mode package-only --limit 2
python experiments/open_task_benchmark/analyze_results.py --dry-run
```

The package generator creates leak-free task packages for public GitHub
projects without cloning them. The analyzer normalizes Con OS, Codex, Claude
Code, local-model, and baseline reports into comparable metrics: task success,
verified success, cost, traceability, minimal patch score, repo cleanliness,
test-modification violations, non-source artifacts, and Amplification
Efficiency when a valid baseline cost/success pair exists.

## Repository Layout

- `conos_cli.py`: unified product CLI.
- `core/`: runtime, orchestration, reasoning, auth, object, and world-model
  contracts.
- `integrations/local_machine/`: the retained environment adapter.
- `modules/llm/`: provider clients, model profiling, route policies, and budget
  controls.
- `modules/local_mirror/`: empty-first mirror and managed VM provider.
- `modules/control_plane/`: action governance and agent control-plane checks.
- `planner/`, `decision/`, `self_model/`, `modules/memory/`: retained cognitive
  runtime support.
- `scripts/check_runtime_preflight.py`: quickstart readiness checks.
- `scripts/check_conos_repo_layout.py`: public/private boundary checks.
- `experiments/open_task_benchmark/`: leak-free real-project task package and
  result comparison protocol.
- `tests/`: tests for retained runtime capabilities.

Generated outputs belong under local runtime paths such as `runtime/`,
`reports/`, `audit/`, or `dist/`; they are ignored and are not part of the
distilled source package.

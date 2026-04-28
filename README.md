# Cognitive OS

Cognitive OS is a local-first cognitive runtime for governed AI agents. This
distilled repository keeps the core runtime, model routing, local-machine
adapter, empty-first mirror, verifier gates, failure learning, and managed VM
provider. Desktop UI shells, WebArena, ARC-AGI-3 adapters, benchmark fixtures,
and frozen report artifacts have been removed from the core distribution.

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
  crash recovery.
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
  and no host-exec fallback when the VM path is selected.

## Local Mirror

```bash
conos mirror init --mirror-root runtime/mirrors/session-1
conos mirror fetch --mirror-root runtime/mirrors/session-1 --path README.md
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --allow-command python3 \
  -- python3 -c "from pathlib import Path; print(Path('README.md').exists())"
conos mirror plan --mirror-root runtime/mirrors/session-1
```

`mirror apply` is a patch gate: it verifies planned source hashes and mirror
hashes before changing the source tree. Rollback checkpoints are recorded for
applied plans.

## Managed VM

```bash
conos vm report
conos vm build-runner
conos vm build-guest-initrd --state-root ~/.conos/vm
conos vm bundle-base-image --state-root ~/.conos/vm --image-id conos-base
conos vm install-base-image-bundle --bundle-dir ~/.conos/vm/image-bundles/conos-base
```

The managed VM path is product-owned but still explicit: `agent-exec` is blocked
until `runtime.json` proves a live VM process plus guest-agent readiness. There
is no silent fallback to host execution.

## LLM Providers

```bash
conos llm --provider ollama profile --discover-visible --catalog-only
conos llm --provider codex profile --discover-visible --catalog-only
conos llm route --route patch_proposal --required-capability coding
```

When a provider is connected, Con OS can inventory visible models, build model
profiles, and emit route policies. Cheap routes default to no-thinking; planning
and patch design can use larger thinking budgets and longer timeouts.

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
- `tests/`: tests for retained runtime capabilities.

Generated outputs belong under local runtime paths such as `runtime/`,
`reports/`, `audit/`, or `dist/`; they are ignored and are not part of the
distilled source package.
